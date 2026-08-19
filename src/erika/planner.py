"""Turn ``results/choices.json`` into a physical motion plan for the Sigma.

typewriter-art's optimizer thinks in terms of a grid of character cells with
several overlapping layers, each layer offset by a fraction of a cell. On a
real typewriter that fraction has to be a motion the machine can actually
make, and something has to keep track of where the print head is.

That is this module. It flattens all layers into one absolute list of
strikes, orders them so the paper only ever feeds forward, and emits the
carriage/paper moves between them.

Units throughout:
    x -- half-steps right of the left margin  (1 half-step = half a cell width)
    y -- half-lines below the top of the image (1 half-line = half a line feed)

so a layer offset of 0.5 is exactly one half-step or one half-line.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from erika import SRC_DIR, erika_codes as ec
from erika import etp

#: Layer offsets the machine can hit. The head moves in half-steps and the
#: platen in half-lines, so quarter-cell schemes ("16x1", "daisy_full") have
#: no physical realisation on this typewriter.
SUPPORTED_OFFSETS = (0.0, 0.5)


class PlanError(ValueError):
    pass


@dataclass(frozen=True)
class Strike:
    """One key press at an absolute position on the page."""

    y: int  #: half-lines below the top of the image
    x: int  #: half-steps right of the left margin
    index: int  #: charset index, as chosen by optimize.py


@dataclass
class Charset:
    """The index -> key mapping written by make_charset.py."""

    name: str
    pitch: int
    cell_w: int
    cell_h: int
    max_columns: int
    codes: list[int]
    advances: list[bool]
    chars: list[str]
    #: Strike force per index, or None for an index that needs no force command.
    #: A charset built before strike force existed has None throughout, and then
    #: nothing in the plan mentions force at all -- which is what keeps every
    #: charset already on disk printing exactly as it did.
    forces: list[int | None] = field(default_factory=list)
    #: The distinct forces, in the order the charset lays them out (hardest
    #: first, as make_charset writes them). This is the order a row is typed in,
    #: so it decides which pass goes on the paper first.
    force_order: list[int] = field(default_factory=list)

    @property
    def has_forces(self) -> bool:
        return any(f is not None for f in self.forces)

    def force_rank(self, index: int) -> int:
        """Where index's force comes in the typing order."""
        force = self.forces[index] if index < len(self.forces) else None
        if force is None:
            return 0
        return self.force_order.index(force)

    @classmethod
    def load(cls, charset: str, base_path: str | None = None) -> "Charset":
        base_path = base_path or SRC_DIR
        path = charset
        if not os.path.isfile(path):
            path = os.path.join(base_path, "charsets", charset, "glyphs.json")
        if not os.path.isfile(path):
            raise PlanError(
                f"no glyphs.json for charset '{charset}'. The bundled charsets are "
                "scans of other machines and carry no key mapping -- generate a "
                "Sigma charset first:\n"
                "    python -m erika.make_charset --pitch 10"
            )
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        glyphs = data["glyphs"]
        for i, g in enumerate(glyphs):
            if g["index"] != i:
                raise PlanError(f"{path}: glyph list is not densely indexed at {i}")
        forces = [g.get("force") for g in glyphs]
        order = data.get("forces") or []
        missing = {f for f in forces if f is not None} - set(order)
        if missing:
            raise PlanError(
                f"{path}: glyphs use force(s) {sorted(missing)} that the file's own "
                "'forces' list does not name. That list is the typing order, so a "
                "force missing from it has no place in the plan -- rebuild the "
                "charset."
            )
        return cls(
            name=data["charset_name"],
            pitch=data["pitch"],
            cell_w=data["cell_width_px"],
            cell_h=data["cell_height_px"],
            max_columns=data["max_columns"],
            codes=[g["code"] for g in glyphs],
            advances=[g["advances"] for g in glyphs],
            chars=[g["char"] for g in glyphs],
            forces=forces,
            force_order=list(order),
        )

    def __len__(self) -> int:
        return len(self.codes)


@dataclass
class Plan:
    """An ordered, absolute list of strikes plus the grid it came from."""

    strikes: list[Strike]
    cols: int
    rows: int
    charset: Charset
    layer_offsets: list[tuple[float, float]] = field(default_factory=list)
    home_each_row: bool = True

    @property
    def width_cells(self) -> float:
        return max((s.x for s in self.strikes), default=0) / 2 + 1

    @property
    def height_cells(self) -> float:
        return max((s.y for s in self.strikes), default=0) / 2 + 1


def _offset_to_units(value: float, what: str) -> int:
    """Convert a fractional layer offset to half-cell units."""
    for i, allowed in enumerate(SUPPORTED_OFFSETS):
        if abs(value - allowed) < 1e-9:
            return i
    raise PlanError(
        f"layer {what} offset {value} is not typeable on the Sigma. The machine "
        f"moves in half-steps and half-lines, so offsets must be one of "
        f"{list(SUPPORTED_OFFSETS)}. Re-run optimize.py with a layer scheme "
        f"built from halves (1x1, 1x2, 1x4, 2Hx1, 2Hx2, 2Vx1, 2Vx2, 4x1, 4x2)."
    )


def load_choices(path: str) -> tuple[list[Strike], int, int, list[tuple[float, float]]]:
    """Flatten choices.json into absolute strikes.

    Keys look like ``layer3_0.5_0`` -- layer index, vertical offset, horizontal
    offset, the latter two as fractions of a cell (this is how optimize.py
    writes them).
    """
    with open(path, encoding="utf-8") as f:
        choices = json.load(f)
    if not choices:
        raise PlanError(f"{path} contains no layers")

    strikes: list[Strike] = []
    offsets: list[tuple[float, float]] = []
    cols = rows = 0
    for key, grid in choices.items():
        try:
            off_v, off_h = (float(p) for p in key.split("_")[1:3])
        except ValueError as exc:
            raise PlanError(f"{path}: cannot parse layer key {key!r}") from exc
        offsets.append((off_v, off_h))
        dy = _offset_to_units(off_v, "vertical")
        dx = _offset_to_units(off_h, "horizontal")
        rows = max(rows, len(grid))
        for i, row in enumerate(grid):
            cols = max(cols, len(row))
            for j, index in enumerate(row):
                if index:  # index 0 is the blank cell -- nothing to type
                    strikes.append(Strike(2 * i + dy, 2 * j + dx, index))
    return strikes, cols, rows, offsets


def build_plan(
    choices_path: str,
    charset: Charset,
    home_each_row: bool = True,
    boustrophedon: bool = True,
    group_by_force: bool = True,
) -> Plan:
    strikes, cols, rows, offsets = load_choices(choices_path)

    bad = [s for s in strikes if not 0 <= s.index < len(charset)]
    if bad:
        raise PlanError(
            f"{choices_path} references character index {bad[0].index}, but charset "
            f"'{charset.name}' only has {len(charset)}. The choices were optimized "
            "against a different charset -- re-run optimize.py with -c "
            f"<the charset you intend to print with>."
        )

    used_cols = max((s.x for s in strikes), default=0) // 2 + 1
    if used_cols > charset.max_columns:
        # The -r to suggest is one *less* than the carriage's column count, not
        # equal to it: resizeTarget pads the target by half a cell on every side,
        # so a print of n characters per row occupies n + 1 columns of cells. The
        # message used to suggest max_columns, which is precisely the value that
        # has just failed -- following the advice failed again.
        raise PlanError(
            f"image is {used_cols} columns wide but the carriage only reaches "
            f"{charset.max_columns} at pitch {charset.pitch}. Re-run optimize.py "
            f"with -r {charset.max_columns - 1} or lower (a print takes one more "
            f"column than its characters per row -- half a cell of margin at each "
            f"side)"
            + (", or switch to pitch 12." if charset.pitch == 10 else ".")
        )

    # Sort by paper position first so the platen only ever feeds forward:
    # reversing the feed introduces backlash that shows up as banding.
    #
    # With strike force in play there is a second decision inside each line: type
    # it force by force, or in one sweep switching force as often as the picture
    # asks. Grouping is the default, for the reason the paper gives in 5.7.2 --
    # "arrange the typing instructions so that the typist will first type all
    # 'hard' characters, then fill in the 'soft' characters" -- which for a
    # machine is not about fatigue but about how often the mechanism has to
    # change state: a switch costs a full character delay, and interleaved they
    # can outnumber the strikes. What it costs is one carriage sweep per force
    # per line, so any error the carriage accumulates across a line lands
    # differently in each group; group_by_force=False buys that back.
    if group_by_force and charset.has_forces:
        strikes.sort(key=lambda s: (s.y, charset.force_rank(s.index), s.x))
    else:
        strikes.sort(key=lambda s: (s.y, s.x))

    if boustrophedon and not home_each_row:
        strikes = _serpentine(strikes)
    return Plan(strikes, cols, rows, charset, offsets, home_each_row)


def _serpentine(strikes: list[Strike]) -> list[Strike]:
    """Reverse every other pass so the carriage sweeps back and forth."""
    out: list[Strike] = []
    start = 0
    flip = False
    for i in range(1, len(strikes) + 1):
        if i == len(strikes) or strikes[i].y != strikes[start].y:
            row = strikes[start:i]
            out.extend(reversed(row) if flip else row)
            flip = not flip
            start = i
    return out


def encode(
    plan: Plan,
    settle_ms: int = 0,
    cr_delay_ms: int = 0,
    no_advance: bool = False,
) -> etp.Job:
    """Emit the opcode stream for a plan, tracking the head as we go.

    `no_advance` types a stack of glyphs in one cell with Doppeldruck (0xA9)
    instead of with a backspace between each pair. Same number of bytes; the
    difference is that the escapement never moves, so the marks land on top of
    each other rather than as close as the escapement's repeatability allows.
    A picture is mostly stacked characters, so that is not a small difference --
    but the code has not been on paper on this machine, which is what
    `erika.pipeline codes` part 6 is for, so it is off by default.
    """
    enc = etp.Encoder()
    cs = plan.charset
    x = y = 0
    prev_y: int | None = None
    # None until the first force is asserted. A charset without forces never
    # asserts one, so its job is byte-identical to what it was before strike
    # force existed -- which is what makes this safe to leave switched on.
    force: int | None = None

    strikes = plan.strikes
    for i, s in enumerate(strikes):
        # A stack is glyphs at the same place, one after another. They are only
        # adjacent in the list when nothing separates them -- with strike force
        # in play a stack can be split across force groups, and then these are
        # two ordinary strikes with a carriage sweep between them, which is what
        # the backspace path is still there for.
        stacked = (
            no_advance
            and i + 1 < len(strikes)
            and strikes[i + 1].y == s.y
            and strikes[i + 1].x == s.x
            and cs.advances[s.index]
        )
        if s.y != prev_y:
            dy = s.y - y
            if dy < 0:  # build_plan sorts by y, so this cannot happen
                raise PlanError(f"plan feeds the paper backwards to row {s.y}")
            if plan.home_each_row:
                # A full NEWLINE drives the standard line-feed mechanism, which
                # is more repeatable than stacking half-line steps -- use it
                # whenever the gap happens to be a whole number of lines.
                if dy and dy % 2 == 0:
                    enc.newline(dy // 2)
                else:
                    enc.carriage_return()
                    enc.vertical(dy)
                x = 0
                if cr_delay_ms:
                    enc.delay_ms(cr_delay_ms)
            else:
                enc.vertical(dy)
            if settle_ms:
                enc.delay_ms(settle_ms)
            y = s.y
            prev_y = s.y

        want = cs.forces[s.index] if s.index < len(cs.forces) else None
        if want is not None and want != force:
            enc.set_force(want)
            force = want

        enc.horizontal(s.x - x)
        x = s.x
        advances = cs.advances[s.index] and not stacked
        if stacked:
            enc.no_advance()
        enc.strike(cs.codes[s.index], cs.advances[s.index])
        if advances:
            x += 2

    # Hand the machine back the way it was found. Strike force is state that
    # outlives the job -- leave it soft and the next thing typed on this
    # typewriter, by this firmware's chatbot or by hand, comes out faint for no
    # visible reason. The hardest force the charset uses is the closest thing to
    # "normal" we can name; see erika_codes.FULL_STRIKE_FORCE for why there is
    # no better answer yet.
    if force is not None and force != cs.force_order[0]:
        enc.set_force(cs.force_order[0])

    # Roll the paper clear of the platen so the sheet can be read/removed.
    # NEWLINE already returns the carriage, so no separate CR is needed.
    enc.newline(2)
    enc.end()

    return etp.Job(
        body=enc.body(),
        cols=plan.cols,
        rows=plan.rows,
        strikes=enc.strikes,
        pitch=cs.pitch,
        home_each_row=plan.home_each_row,
    )


def summarize(plan: Plan, job: etp.Job, ops_per_second: float = 10.0) -> str:
    cs = plan.charset
    mech_ops = 0
    force_changes = 0
    for _, op, operand in etp.iter_ops(job.body):
        if op in (etp.OP_RIGHT, etp.OP_LEFT, etp.OP_DOWN, etp.OP_UP,
                  etp.OP_MICRO_DOWN, etp.OP_MICRO_UP):
            mech_ops += operand
        elif op in (etp.OP_STRIKE, etp.OP_STRIKE_NA, etp.OP_CR):
            mech_ops += 1
        elif op == etp.OP_NEWLINE:
            mech_ops += operand
        elif op == etp.OP_SET_FORCE:
            # Two bytes, and the machine has to settle the hammer setting
            # between them -- so it costs about what a strike does.
            mech_ops += 2
            force_changes += 1

    w_mm = plan.width_cells * ec.PITCH_WIDTH_MM[cs.pitch]
    h_mm = plan.height_cells * ec.LINE_HEIGHT_MM
    seconds = mech_ops / ops_per_second
    cells = plan.cols * plan.rows
    layers = len(plan.layer_offsets)
    stacks = sum(1 for _, op, _ in etp.iter_ops(job.body)
                 if op == etp.OP_NO_ADVANCE)
    force_line = []
    if cs.has_forces:
        force_line = [
            f"  strike force {len(cs.force_order)} levels "
            f"({', '.join(f'0x{f:02X}' for f in cs.force_order)}), "
            f"{force_changes} changes"
        ]
    return "\n".join(
        [
            f"  charset      {cs.name}",
            f"  grid         {plan.cols} x {plan.rows} cells, {layers} layers "
            f"({cells * layers} slots, {job.strikes} inked)",
            f"  on paper     {plan.width_cells:.1f} x {plan.height_cells:.1f} cells "
            f"= {w_mm:.0f} x {h_mm:.0f} mm",
            f"  job size     {etp.HEADER_SIZE + len(job.body)} bytes",
            *force_line,
            f"  mechanics    {mech_ops} head operations, "
            f"~{seconds / 60:.0f} min at {ops_per_second:g}/s",
            f"  paper feed   {'carriage return every pass' if plan.home_each_row else 'serpentine, no return'}",
            *([f"  overstrike   {stacks} glyphs typed without advancing (0xA9), "
               "so no backspace between a stack"] if stacks else []),
        ]
    )
