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
        return cls(
            name=data["charset_name"],
            pitch=data["pitch"],
            cell_w=data["cell_width_px"],
            cell_h=data["cell_height_px"],
            max_columns=data["max_columns"],
            codes=[g["code"] for g in glyphs],
            advances=[g["advances"] for g in glyphs],
            chars=[g["char"] for g in glyphs],
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
        raise PlanError(
            f"image is {used_cols} columns wide but the carriage only reaches "
            f"{charset.max_columns} at pitch {charset.pitch}. Re-run optimize.py "
            f"with -r {charset.max_columns} or lower"
            + (", or switch to pitch 12." if charset.pitch == 10 else ".")
        )

    # Sort by paper position first so the platen only ever feeds forward:
    # reversing the feed introduces backlash that shows up as banding.
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
) -> etp.Job:
    """Emit the opcode stream for a plan, tracking the head as we go."""
    enc = etp.Encoder()
    cs = plan.charset
    x = y = 0
    prev_y: int | None = None

    for s in plan.strikes:
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

        enc.horizontal(s.x - x)
        x = s.x
        advances = cs.advances[s.index]
        enc.strike(cs.codes[s.index], advances)
        if advances:
            x += 2

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
    for _, op, operand in etp.iter_ops(job.body):
        if op in (etp.OP_RIGHT, etp.OP_LEFT, etp.OP_DOWN, etp.OP_UP,
                  etp.OP_MICRO_DOWN, etp.OP_MICRO_UP):
            mech_ops += operand
        elif op in (etp.OP_STRIKE, etp.OP_STRIKE_NA, etp.OP_CR):
            mech_ops += 1
        elif op == etp.OP_NEWLINE:
            mech_ops += operand

    w_mm = plan.width_cells * ec.PITCH_WIDTH_MM[cs.pitch]
    h_mm = plan.height_cells * ec.LINE_HEIGHT_MM
    seconds = mech_ops / ops_per_second
    cells = plan.cols * plan.rows
    layers = len(plan.layer_offsets)
    return "\n".join(
        [
            f"  charset      {cs.name}",
            f"  grid         {plan.cols} x {plan.rows} cells, {layers} layers "
            f"({cells * layers} slots, {job.strikes} inked)",
            f"  on paper     {plan.width_cells:.1f} x {plan.height_cells:.1f} cells "
            f"= {w_mm:.0f} x {h_mm:.0f} mm",
            f"  job size     {etp.HEADER_SIZE + len(job.body)} bytes",
            f"  mechanics    {mech_ops} head operations, "
            f"~{seconds / 60:.0f} min at {ops_per_second:g}/s",
            f"  paper feed   {'carriage return every pass' if plan.home_each_row else 'serpentine, no return'}",
        ]
    )
