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


def _step_chunks(code: int, steps: int) -> list[int]:
    """One direct-step run as the bytes the firmware would send for it.

    A deliberate re-implementation of ``startStepChunk()``, in the same spirit
    as ``expand()`` itself: one command carries at most 127 steps, and the
    platen refuses four counts in the middle of its range, so a run comes out as
    a sequence of pairs. If this and the firmware disagree, one of them is wrong
    and the test suite says which.
    """
    out: list[int] = []
    while steps:
        take = max(-ec.MAX_STEPS_PER_COMMAND, min(ec.MAX_STEPS_PER_COMMAND, steps))
        if code == ec.PLATEN_STEPS and abs(take) in ec.FORBIDDEN_PLATEN_STEPS:
            take = 2 if take > 0 else -2
        out += [code, ec.encode_step_operand(take)]
        steps -= take
    return out


#: The three settings behind the firmware's ``IMG STEPS``.
STEPS_OFF = "off"
STEPS_AUTO = "auto"
STEPS_ALL = "all"


def _use_steps(mode: str, code: int, steps: int, keystrokes: int) -> bool:
    """Would this move go out as step commands? Mirrors ``stepsAreWorthIt()``.

    Two bytes per command against one per keystroke, and strictly fewer rather
    than no worse -- a tie stays on the mechanism that has been on paper.
    """
    if mode == STEPS_ALL:
        return True
    if mode == STEPS_OFF:
        return False
    if mode != STEPS_AUTO:
        raise EmulationError(f"unknown direct-step mode {mode!r}")
    return len(_step_chunks(code, steps)) < keystrokes


def expand(body: bytes, direct_steps: str = STEPS_OFF, pitch: int = 10) -> list[int]:
    """Opcodes -> the raw byte stream the firmware sends to the typewriter.

    ``direct_steps`` mirrors the firmware's ``IMG STEPS``: "auto" sends a move as
    a count of absolute motor steps wherever that is fewer bytes than the
    keystrokes would be, "all" does it everywhere. Every setting has to put the
    head in the same place, which is what
    ``test_direct_steps_land_where_the_keystrokes_would`` checks -- offline,
    because the alternative is finding out on an hour of paper.
    """
    out: list[int] = []
    per_half_step = ec.carriage_steps_per_half_step(pitch)
    for _, op, operand in etp.iter_ops(body):
        if op == etp.OP_END:
            break
        elif op in (etp.OP_RIGHT, etp.OP_LEFT):
            right = op == etp.OP_RIGHT
            keystrokes = operand // 2 + (operand & 1)
            steps = operand * per_half_step
            if _use_steps(direct_steps, ec.CARRIAGE_STEPS, steps, keystrokes):
                out += _step_chunks(ec.CARRIAGE_STEPS, steps if right else -steps)
                continue
            # Full steps first, then the odd half -- same order as the firmware.
            out += [ec.SPACE if right else ec.BACKSPACE] * (operand // 2)
            if operand & 1:
                out.append(ec.HALF_STEP_FORWARD if right else ec.HALF_STEP_BACK)
        elif op in (etp.OP_DOWN, etp.OP_UP):
            down = op == etp.OP_DOWN
            steps = operand * ec.PLATEN_STEPS_PER_HALF_LINE
            if _use_steps(direct_steps, ec.PLATEN_STEPS, steps, operand):
                out += _step_chunks(ec.PLATEN_STEPS, steps if down else -steps)
                continue
            out += [ec.HALF_LINE_FORWARD if down else ec.HALF_LINE_BACK] * operand
        elif op in (etp.OP_MICRO_DOWN, etp.OP_MICRO_UP):
            down = op == etp.OP_MICRO_DOWN
            # A micro line is a twentieth of a line, which is two steps.
            steps = 2 * operand
            if _use_steps(direct_steps, ec.PLATEN_STEPS, steps, operand):
                out += _step_chunks(ec.PLATEN_STEPS, steps if down else -steps)
                continue
            out += [ec.MICRO_LINE_FORWARD if down else ec.MICRO_LINE_BACK] * operand
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
        elif op == etp.OP_RAW:
            out.append(operand)
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
    the planner's coordinates. The direct-step codes move by a fraction of
    those, so there is a residue underneath each -- see `_x_fine`.

    What this models and what it merely *records* is the distinction to keep in
    view. The keystroke motions, the strike force and the direct step commands
    are modelled, because a plan's placement depends on them and offline
    verification is the point of the whole arrangement. The mode switches are
    recorded in `probes` and otherwise ignored: what 0x8E or 0x8C do to this
    machine is what `erika.pipeline codes` exists to find out, and a guess here
    would be a guess that the test suite then certifies.
    """

    x: int = 0
    y: int = 0
    impressions: list[Impression] = field(default_factory=list)
    #: Set when the carriage is driven past the machine's physical limit.
    overruns: int = 0
    max_columns: int = 65
    #: Pitch, which is what a 1/120" carriage step is a fraction *of*.
    pitch: int = 10
    #: The strike force in effect. None until a job sets one.
    force: int | None = None
    #: Every control code the machine was sent that this model does not act on,
    #: as (code, operand or None). Empty for an ordinary print job.
    probes: list[tuple[int, int | None]] = field(default_factory=list)
    #: True between ERIKA_SET_STRIKE_FORCE and the byte that names the force.
    _awaiting_force: bool = False
    #: The control code whose operand byte is expected next, if any.
    _awaiting_operand: int | None = None
    #: 0xA9 seen: the next strike prints where it stands.
    _no_advance: bool = False
    #: Sub-half-step and sub-half-line residue, in the machine's own motor steps.
    _x_fine: int = 0
    _y_fine: int = 0

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

        # A command that takes a step count: apply it, in the mechanism's own
        # units, and carry whole half-steps up into the position the planner
        # thinks in. divmod floors, which is what makes a backwards residue come
        # out right.
        if self._awaiting_operand is not None:
            command, self._awaiting_operand = self._awaiting_operand, None
            steps = ec.decode_step_operand(code)
            if command == ec.CARRIAGE_STEPS:
                self._x_fine += steps
                carry, self._x_fine = divmod(
                    self._x_fine, ec.carriage_steps_per_half_step(self.pitch)
                )
                self._move(carry)
            elif command == ec.PLATEN_STEPS:
                self._y_fine += steps
                carry, self._y_fine = divmod(
                    self._y_fine, ec.PLATEN_STEPS_PER_HALF_LINE
                )
                self.y += carry
            else:
                self.probes.append((command, code))
            return
        if code in ec.OPERAND_CODES:
            self._awaiting_operand = code
            return
        if code == 0xA9:  # Doppeldruck: print the next character where we stand
            self._no_advance = True
            return
        if code in ec.CONTROL_CODE_NAMES and code not in ec.CONTROL_CODES:
            # A mode switch, or something else this model has no opinion about.
            self.probes.append((code, None))
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
            if glyph.advances and not self._no_advance:
                self._move(2)
            self._no_advance = False

    def _move(self, half_steps: int) -> None:
        self.x += half_steps
        if self.x < 0 or self.x > 2 * self.max_columns:
            self.overruns += 1
            self.x = max(0, min(self.x, 2 * self.max_columns))

    def run(self, codes) -> "Typewriter":
        for code in codes:
            self.feed(code)
        return self


def type_job(job: etp.Job, max_columns: int = 65,
             direct_steps: str = STEPS_OFF) -> Typewriter:
    """Run a whole print job through the virtual machine."""
    return Typewriter(max_columns=max_columns, pitch=job.pitch).run(
        expand(job.body, direct_steps=direct_steps, pitch=job.pitch)
    )


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
