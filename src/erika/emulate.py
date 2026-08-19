"""A virtual Sigma SM 8200i, for testing the pipeline without paper.

Two layers, matching the two real ones:

``expand()`` turns .etp opcodes into the raw bytes the firmware pushes at the
typewriter. It is a deliberate re-implementation of
``ErikaImagePrinter::fetchNext()`` -- if the two disagree, one of them is
wrong, and the test suite says which.

``Typewriter`` then consumes those raw bytes the way the machine does: it
moves a carriage and a platen and records where every strike lands. Feeding
that back through ``preview.render`` reconstructs the picture, so the whole
chain -- optimizer choices, layer flattening, motion planning, opcode
encoding, firmware expansion, machine semantics -- is checked end to end
against the mockup the optimizer produced.

Keep expand() in sync with erika_ai/src/erika_image.cpp.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from erika import erika_codes as ec
from erika import etp


class EmulationError(RuntimeError):
    pass


def expand(body: bytes) -> list[int]:
    """Opcodes -> the raw byte stream the firmware sends to the typewriter."""
    out: list[int] = []
    for _, op, operand in etp.iter_ops(body):
        if op == etp.OP_END:
            break
        elif op in (etp.OP_RIGHT, etp.OP_LEFT):
            right = op == etp.OP_RIGHT
            # Full steps first, then the odd half -- same order as the firmware.
            out += [ec.SPACE if right else ec.BACKSPACE] * (operand // 2)
            if operand & 1:
                out.append(ec.HALF_STEP_FORWARD if right else ec.HALF_STEP_BACK)
        elif op in (etp.OP_DOWN, etp.OP_UP):
            code = ec.HALF_LINE_FORWARD if op == etp.OP_DOWN else ec.HALF_LINE_BACK
            out += [code] * operand
        elif op in (etp.OP_MICRO_DOWN, etp.OP_MICRO_UP):
            code = ec.MICRO_LINE_FORWARD if op == etp.OP_MICRO_DOWN else ec.MICRO_LINE_BACK
            out += [code] * operand
        elif op == etp.OP_CR:
            out.append(ec.CARRIAGE_RETURN)
        elif op == etp.OP_NEWLINE:
            out += [ec.NEWLINE] * operand
        elif op in (etp.OP_STRIKE, etp.OP_STRIKE_NA):
            out.append(operand)
        elif op == etp.OP_SET_FORCE:
            # Two bytes: the command, then the force. The firmware sends them
            # as a pending byte plus a tail byte, in this order.
            out += [ec.SET_STRIKE_FORCE, operand]
        elif op == etp.OP_DELAY:
            pass  # timing only; nothing reaches the paper
        else:
            raise EmulationError(f"expand() has no case for opcode 0x{op:02X}")
    return out


@dataclass
class Impression:
    """One mark on the paper."""

    y: int  #: half-lines below where the paper started
    x: int  #: half-steps right of the left margin
    code: int  #: the key that struck
    force: int | None = None  #: the strike force in effect, if one was set


@dataclass
class Typewriter:
    """A carriage, a platen, and a sheet of paper.

    Positions are in half-steps / half-lines, so they line up directly with
    the planner's coordinates.
    """

    x: int = 0
    y: int = 0
    impressions: list[Impression] = field(default_factory=list)
    #: Set when the carriage is driven past the machine's physical limit.
    overruns: int = 0
    max_columns: int = 65
    #: The strike force in effect. None until a job sets one.
    force: int | None = None
    #: True between ERIKA_SET_STRIKE_FORCE and the byte that names the force.
    _awaiting_force: bool = False

    def feed(self, code: int) -> None:
        # The force command swallows the byte after it, whatever that byte is.
        # Modelled here rather than in expand() because it is the *machine* that
        # behaves this way, and it is the reason a force must never be a motion
        # code: on a machine that lacks the command, this branch does not exist
        # and the byte lands as a character.
        if self._awaiting_force:
            self.force = code
            self._awaiting_force = False
            return
        if code == ec.SET_STRIKE_FORCE:
            self._awaiting_force = True
            return

        if code == ec.SPACE:
            self._move(2)
        elif code == ec.BACKSPACE:
            self._move(-2)
        elif code == ec.HALF_STEP_FORWARD:
            self._move(1)
        elif code == ec.HALF_STEP_BACK:
            self._move(-1)
        elif code == ec.HALF_LINE_FORWARD:
            self.y += 1
        elif code == ec.HALF_LINE_BACK:
            self.y -= 1
        elif code == ec.MICRO_LINE_FORWARD or code == ec.MICRO_LINE_BACK:
            pass  # sub-half-line; the planner never relies on it for placement
        elif code == ec.CARRIAGE_RETURN:
            self.x = 0
        elif code == ec.NEWLINE:
            self.x = 0
            self.y += 2
        elif code == ec.TAB:
            raise EmulationError("TAB has no defined position model")
        else:
            glyph = ec.glyph_for_code(code)
            if glyph is None:
                raise EmulationError(f"struck unknown key 0x{code:02X}")
            self.impressions.append(Impression(self.y, self.x, code, self.force))
            if glyph.advances:
                self._move(2)

    def _move(self, half_steps: int) -> None:
        self.x += half_steps
        if self.x < 0 or self.x > 2 * self.max_columns:
            self.overruns += 1
            self.x = max(0, min(self.x, 2 * self.max_columns))

    def run(self, codes) -> "Typewriter":
        for code in codes:
            self.feed(code)
        return self


def type_job(job: etp.Job, max_columns: int = 65) -> Typewriter:
    """Run a whole print job through the virtual machine."""
    return Typewriter(max_columns=max_columns).run(expand(job.body))


def impressions_to_strikes(machine: Typewriter, charset) -> list:
    """Map struck key codes back to charset indices, for re-rendering.

    Fails loudly on a code the charset does not contain -- that would mean
    the plan is telling the machine to type something the optimizer never
    chose.

    The key is (code, force), not the code alone. With strike force in play the
    same key appears at several indices carrying different amounts of ink, and
    keying on the code would recover the first of them for every strike -- which
    would make the render agree with the plan by throwing away the very thing
    the force was for.
    """
    from erika.planner import Strike

    forces = charset.forces or [None] * len(charset.codes)
    by_key: dict[tuple[int, int | None], int] = {}
    for index, code in enumerate(charset.codes):
        by_key.setdefault((code, forces[index]), index)

    strikes = []
    for imp in machine.impressions:
        # A charset with no force of its own is indifferent to what the machine
        # is set to, so fall back to the code alone rather than refusing.
        key = (imp.code, imp.force if charset.has_forces else None)
        index = by_key.get(key)
        if index is None:
            raise EmulationError(
                f"key 0x{imp.code:02X} ({ec.describe_code(imp.code)})"
                + (f" at force 0x{imp.force:02X}" if key[1] is not None else "")
                + f" is not in charset '{charset.name}'"
            )
        strikes.append(Strike(imp.y, imp.x, index))
    return strikes
