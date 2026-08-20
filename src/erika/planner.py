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

#: Layer offsets the *keyboard* can hit. The head moves in half-steps and the
#: platen in half-lines, and those are the finest keystrokes the machine has.
SUPPORTED_OFFSETS = (0.0, 0.5)


class PlanError(ValueError):
    pass


@dataclass(frozen=True)
class Strike:
    """One key press at an absolute position on the page.

    The position is a half-cell grid point plus, for a layer offset the keyboard
    cannot reach, a residue in the machine's own motor steps. Both residues are
    zero for every scheme built from halves, which is every scheme that was
    typeable before ``--fine``, so nothing about an existing plan changes.
    """

    y: int  #: half-lines below the top of the image
    x: int  #: half-steps right of the left margin
    index: int  #: charset index, as chosen by optimize.py
    fy: int = 0  #: extra 1/240" platen steps below y
    fx: int = 0  #: extra 1/120" carriage steps right of x


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

    def force_for(self, index: int) -> int | None:
        """The force this index is typed at, or None if the charset has none."""
        return self.forces[index] if index < len(self.forces) else None

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


def offset_to_units(value: float, what: str, pitch: int, fine: bool = False
                    ) -> tuple[int, int]:
    """Split a fractional layer offset into half-cells and leftover motor steps.

    Returns ``(half_units, steps)``. ``steps`` is zero for the offsets the
    keyboard can reach -- 0 and 0.5 -- and that is every scheme built from
    halves, which is every scheme this pipeline could type before ``fine``
    existed.

    With ``fine`` -- the default, since the machine has been shown to honour
    0xA5 and 0xA6 -- an offset is typeable if it lands on a whole motor step: the
    carriage moves in 1/120" and the platen in 1/240", which is far finer than
    any keystroke. That is what makes the quarter-cell schemes physically
    realisable, and it is more general than quarters -- ``daisy_full``'s
    eighth-cell offsets come out exact at pitch 15 (one carriage step each) and
    its fifth-line offsets at eight platen steps.

    What it cannot do is round. A quarter of a cell is three carriage steps at
    pitch 10 and two at pitch 15, but two and a half at pitch 12, and half a
    motor step does not exist -- so the same scheme is typeable at one pitch and
    not at another, and the error has to say which.
    """
    for i, allowed in enumerate(SUPPORTED_OFFSETS):
        if abs(value - allowed) < 1e-9:
            return i, 0

    schemes = ("1x1, 1x2, 1x4, 2Hx1, 2Hx2, 2Vx1, 2Vx2, 4x1, 4x2")
    if not fine:
        raise PlanError(
            f"layer {what} offset {value} is not typeable by keystroke. The "
            f"machine's finest keys move half a cell across and half a line "
            f"down, so offsets have to be one of {list(SUPPORTED_OFFSETS)}. "
            f"Either re-run optimize.py with a layer scheme built from halves "
            f"({schemes}), or pass --fine, which places strikes with the "
            f"machine's own motor steps instead -- 1/120 inch across, 1/240 "
            f"down. Those are 0xA5 and 0xA6, which this machine honours -- "
            f"`erika.pipeline codes` sections 2 to 4 -- so the flag is normally "
            f"on and something has turned it off."
        )

    per_half = (
        ec.carriage_steps_per_half_step(pitch)
        if what == "horizontal"
        else ec.PLATEN_STEPS_PER_HALF_LINE
    )
    exact = value * 2 * per_half  # motor steps into the cell / line
    if abs(exact - round(exact)) > 1e-9:
        unit = "1/120 inch carriage" if what == "horizontal" else "1/240 inch platen"
        raise PlanError(
            f"layer {what} offset {value} is not a whole number of {unit} steps "
            f"at pitch {pitch} -- it comes to {exact:.4g} of them, and half a "
            f"motor step does not exist. "
            + (
                "A quarter of a cell is exact at pitch 10 (three steps) and at "
                "pitch 15 (two), but not at pitch 12. Rebuild the charset at "
                "another pitch, or use a layer scheme built from halves."
                if what == "horizontal"
                else "Use a layer scheme whose vertical offsets are whole "
                "1/240 inch steps of a line -- halves, quarters, fifths and "
                "eighths all are."
            )
        )
    steps = int(round(exact))
    return divmod(steps, per_half)


def load_choices(
    path: str, pitch: int = 10, fine: bool = False
) -> tuple[list[Strike], int, int, list[tuple[float, float]]]:
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
        dy, fy = offset_to_units(off_v, "vertical", pitch, fine)
        dx, fx = offset_to_units(off_h, "horizontal", pitch, fine)
        rows = max(rows, len(grid))
        for i, row in enumerate(grid):
            cols = max(cols, len(row))
            for j, index in enumerate(row):
                if index:  # index 0 is the blank cell -- nothing to type
                    strikes.append(
                        Strike(2 * i + dy, 2 * j + dx, index, fy, fx)
                    )
    return strikes, cols, rows, offsets


def build_plan(
    choices_path: str,
    charset: Charset,
    home_each_row: bool = True,
    boustrophedon: bool = True,
    group_by_force: bool = True,
    fine: bool = True,
) -> Plan:
    strikes, cols, rows, offsets = load_choices(choices_path, charset.pitch, fine)

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
    # The residues are part of the position, so they are part of the sort -- the
    # paper must still only ever feed forward, and within a line the carriage
    # must still only sweep one way.
    if group_by_force and charset.has_forces:
        strikes.sort(key=lambda s: (s.y, s.fy, charset.force_rank(s.index),
                                    s.x, s.fx))
    else:
        strikes.sort(key=lambda s: (s.y, s.fy, s.x, s.fx))

    if boustrophedon and not home_each_row:
        strikes = _serpentine(strikes)
    return Plan(strikes, cols, rows, charset, offsets, home_each_row)


def _pass_key(s: Strike) -> tuple[int, int]:
    """What makes two strikes part of the same carriage sweep.

    The paper position, and that is (y, fy) rather than y alone: a layer offset
    by a quarter of a line is a *different* sweep from the one above it, and
    grouping the two together would let the serpentine reverse them into each
    other and feed the platen backwards to reach the second.
    """
    return (s.y, s.fy)


def _serpentine(strikes: list[Strike]) -> list[Strike]:
    """Reverse every other pass so the carriage sweeps back and forth."""
    out: list[Strike] = []
    start = 0
    flip = False
    for i in range(1, len(strikes) + 1):
        if i == len(strikes) or _pass_key(strikes[i]) != _pass_key(strikes[start]):
            row = strikes[start:i]
            out.extend(reversed(row) if flip else row)
            flip = not flip
            start = i
    return out


#: Shortest run worth typing backwards. Three, and the arithmetic is worth
#: writing out because it is the whole case for the feature.
#:
#: Walking k adjacent cells leftwards in forward mode costs one byte for the
#: first glyph and three for each of the rest -- two backspaces to undo the
#: escapement's advance and take another cell off, then the glyph -- so 3k-2.
#: Backwards it is one byte to enter the mode, one per glyph and one to leave:
#: k+2. The approach differs too, by exactly one byte in backward's favour,
#: since the run is entered one cell further right than it is begun.
#:
#: That makes two cells a saving of one byte where the head has to travel to the
#: run at all, and a tie where it is already standing on the first cell. Three
#: is the shortest run that is strictly cheaper either way, and a tie stays on
#: the mechanism that has been on paper longest -- the same rule
#: ``stepsAreWorthIt`` follows in the firmware.
MIN_BACKWARD_RUN = 3


def _continues_backwards(a: Strike, b: Strike, cs: Charset) -> bool:
    """Can `b` be typed as the backward strike that follows `a`?

    Backward printing is one motion welded to one strike: the head steps a whole
    cell left and marks there. So `b` has to be exactly that -- one cell left of
    `a`, on the same pass, at the same sub-cell offset, with nothing needed
    between them that the mode has not been asked about.

    The three refusals are all the same refusal. A different paper position or a
    different residue needs a *motion* in between, and whether the motion keys
    invert with the printing direction is unasked. A stacked glyph is at the
    same x, which needs 0xA9, and whether that still means "print where the head
    stands" is unasked. A dead key does not feed at all, and what "does not
    feed" means in a mode whose content is the direction of the feed is unasked
    hardest of all. Section 7 of the sheet answered plain advancing strikes and
    this is the run of them.

    A force change is refused for a plainer reason: it is two bytes on the wire,
    which is the whole saving for two cells, and it would sit inside a mode that
    the machine has only been seen to hold across characters.
    """
    return (
        (b.y, b.fy) == (a.y, a.fy)
        and b.fx == a.fx
        and b.x == a.x - 2
        and cs.advances[a.index]
        and cs.advances[b.index]
        and cs.force_for(a.index) == cs.force_for(b.index)
    )


def _backward_runs(strikes: list[Strike], cs: Charset) -> dict[int, int]:
    """Which stretches of the plan to type right to left.

    Returns ``{first strike index: one past the last}``. Runs are maximal, which
    is what makes the greedy scan the right answer here: the relation is between
    neighbours, so a run that cannot be extended contains every shorter run
    inside it, and splitting one could only add mode switches.

    Empty unless the plan is a serpentine -- a pass that sweeps left to right
    never has two strikes in descending order, let alone three.
    """
    runs: dict[int, int] = {}
    i, n = 0, len(strikes)
    while i < n:
        j = i + 1
        while j < n and _continues_backwards(strikes[j - 1], strikes[j], cs):
            j += 1
        if j - i >= MIN_BACKWARD_RUN:
            runs[i] = j
        i = j
    return runs


def _one_mechanism(delta: int, per_unit: int) -> tuple[int, int]:
    """Split a move into whole keystroke units and motor steps -- or all steps.

    A move with no residue is left to the keystrokes: the detented line-feed and
    the escapement are what the machine is built around, and the calibration
    sheet says they are repeatable.

    A move that needs motor steps is done *entirely* in them, rather than as
    "whole units, then the remainder". That is a measurement, not a preference.
    Part 5 of `erika.pipeline codes` feeds the platen five steps at a time, eight
    times over; every gap came out equal except the first, which follows the
    detented line feed that ends the section heading and came out short. Part 4
    feeds forty steps at a time from the same starting condition and looked even
    -- which is the same fault seen from further away, since one or two steps
    lost is a fortieth of that gap and a fifth of part 5's.

    So a motor-step feed immediately after the detented mechanism appears to
    lose a step or two taking up the detent, and the way not to pay it is not to
    change mechanism part-way through a move.

    It is cheaper too, which is the sort of agreement worth noticing: the
    remainder already costs an opcode and an operand, so folding the whole move
    into it removes one and shortens the byte stream rather than lengthening it.
    """
    whole, residue = _split(delta, per_unit)
    if residue:
        return 0, delta
    return whole, residue


def _split(delta: int, per_unit: int) -> tuple[int, int]:
    """Split a signed motion into whole units and a residue of the same sign.

    Same sign, and that is the point. A move computed as "whole units to the new
    grid point, then the residue" mixes directions whenever the residue shrinks:
    going from a quarter-line layer to the next row's unoffset one would feed the
    platen a whole half-line forward and then ten steps *back*, which is a
    reversal the planner exists to avoid -- it is where banding comes from. Doing
    the arithmetic on the absolute step count first leaves one direction.
    """
    sign = -1 if delta < 0 else 1
    whole, residue = divmod(abs(delta), per_unit)
    return sign * whole, sign * residue


def encode(
    plan: Plan,
    settle_ms: int = 0,
    cr_delay_ms: int = 0,
    no_advance: bool = True,
    backward: bool = True,
) -> etp.Job:
    """Emit the opcode stream for a plan, tracking the head as we go.

    `no_advance` types a stack of glyphs in one cell with Doppeldruck (0xA9)
    instead of with a backspace between each pair. Same number of bytes on the
    wire and two fewer in the file; the difference that matters is that the
    escapement never moves, so the marks land on top of each other rather than
    as close as the escapement's repeatability allows. A picture is mostly
    stacked characters, so that is not a small difference.

    On by default since section 6 of `erika.pipeline codes` came back with eight
    O struck through and no backspace anywhere in the plan. Pass False to get
    the backspace behaviour, which is what to do if a sheet comes out with the
    stacks smeared and you want to know which mechanism is at fault.

    `backward` types a serpentine's reverse passes with Rückwärtsdruck (0x8E)
    instead of backspacing between every cell: the machine moves left and
    strikes on one byte instead of three. It costs a byte at each end of a run,
    so `_backward_runs` picks the stretches where it pays, and a plan that
    returns the carriage every row has no such stretches -- this does nothing to
    one. On by default since section 7 of the sheet came back reading EDCBA;
    pass False for the backspaces, for the same fault-isolating reason as above.
    """
    enc = etp.Encoder()
    cs = plan.charset
    x = y = 0
    # Where the head is *within* the half-cell, in the machine's motor steps.
    # Always zero unless a layer offset asked for a position the keyboard cannot
    # reach, which keeps a half-cell plan byte-identical to what it was.
    x_fine = y_fine = 0
    prev_y: int | None = None
    # None until the first force is asserted. A charset without forces never
    # asserts one, so its job is byte-identical to what it was before strike
    # force existed -- which is what makes this safe to leave switched on.
    force: int | None = None

    strikes = plan.strikes
    # The stretches to type right to left, as {first index: one past the last}.
    # Worked out up front because whether a strike starts a run decides where
    # the carriage has to be *before* it, which is a move this loop has already
    # emitted by the time it reaches the strike.
    runs = _backward_runs(strikes, cs) if backward else {}
    run_stop = 0  # one past the last strike of the run being typed, if any

    for i, s in enumerate(strikes):
        if i < run_stop:
            # Inside a run: the strike carries its own motion, so there is
            # nothing to emit but the glyph. Everything the branch below does --
            # feed the paper, change force, move the carriage -- is a thing
            # _continues_backwards refused, which is what makes this safe.
            enc.strike(cs.codes[s.index], cs.advances[s.index])
            x = s.x  # it moved first, so the head stands on the mark
            if i + 1 == run_stop:
                enc.backward_off()
            continue

        # A stack is glyphs at the same place, one after another. They are only
        # adjacent in the list when nothing separates them -- with strike force
        # in play a stack can be split across force groups, and then these are
        # two ordinary strikes with a carriage sweep between them, which is what
        # the backspace path is still there for.
        stacked = (
            no_advance
            and i + 1 < len(strikes)
            and (strikes[i + 1].y, strikes[i + 1].fy) == (s.y, s.fy)
            and (strikes[i + 1].x, strikes[i + 1].fx) == (s.x, s.fx)
            and cs.advances[s.index]
        )
        if (s.y, s.fy) != prev_y:
            # In the platen's own steps, so that a residue which shrinks between
            # rows still comes out as one forward feed rather than a whole
            # half-line forward and a fraction back.
            delta = ((s.y * ec.PLATEN_STEPS_PER_HALF_LINE + s.fy)
                     - (y * ec.PLATEN_STEPS_PER_HALF_LINE + y_fine))
            if delta < 0:  # build_plan sorts by (y, fy), so this cannot happen
                raise PlanError(f"plan feeds the paper backwards to row {s.y}")
            dy, dy_fine = _one_mechanism(delta, ec.PLATEN_STEPS_PER_HALF_LINE)
            if plan.home_each_row:
                # A full NEWLINE drives the standard line-feed mechanism, which
                # is more repeatable than stacking half-line steps -- use it
                # whenever the gap happens to be a whole number of lines.
                if dy and dy % 2 == 0 and dy_fine == 0:
                    enc.newline(dy // 2)
                else:
                    enc.carriage_return()
                    enc.vertical(dy)
                    enc.vertical_fine(dy_fine)
                x = 0
                x_fine = 0
                if cr_delay_ms:
                    enc.delay_ms(cr_delay_ms)
            else:
                enc.vertical(dy)
                enc.vertical_fine(dy_fine)
            if settle_ms:
                enc.delay_ms(settle_ms)
            y, y_fine = s.y, s.fy
            prev_y = (s.y, s.fy)

        want = cs.force_for(s.index)
        if want is not None and want != force:
            enc.set_force(want)
            force = want

        # Where the carriage has to be for this strike. One cell right of the
        # mark if a backward run starts here, because the machine takes that
        # cell off itself before the hammer comes down -- and that is exactly
        # where the escapement would stand after typing at s.x, so it is not a
        # position the carriage has to reach and could not before.
        start = runs.get(i)
        target = s.x + 2 if start is not None else s.x

        # The same arithmetic across, for the same reason.
        per_half = ec.carriage_steps_per_half_step(cs.pitch)
        dx, dx_fine = _one_mechanism(
            (target * per_half + s.fx) - (x * per_half + x_fine), per_half
        )
        enc.horizontal(dx)
        enc.horizontal_fine(dx_fine)
        x, x_fine = target, s.fx

        if start is not None:
            enc.backward_on()
            run_stop = start
            enc.strike(cs.codes[s.index], cs.advances[s.index])
            x = s.x
            continue

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
        elif op in (etp.OP_RIGHT_FINE, etp.OP_LEFT_FINE,
                    etp.OP_DOWN_FINE, etp.OP_UP_FINE):
            # One command whatever the distance, and it moves less than any
            # keystroke can, so it costs about what a carriage step does.
            mech_ops += 1
        elif op in (etp.OP_NO_ADVANCE, etp.OP_BACKWARD_ON, etp.OP_BACKWARD_OFF):
            mech_ops += 1
        elif op in (etp.OP_STRIKE, etp.OP_STRIKE_NA, etp.OP_CR):
            # A backward strike is still one operation: the escapement takes its
            # cell and the hammer comes down, which is what a forward strike
            # does in the other order.
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
    backward_runs = 0
    backward_strikes = 0
    typing_backwards = False
    for _, op, _ in etp.iter_ops(job.body):
        if op == etp.OP_BACKWARD_ON:
            typing_backwards = True
            backward_runs += 1
        elif op == etp.OP_BACKWARD_OFF:
            typing_backwards = False
        elif op == etp.OP_STRIKE and typing_backwards:
            backward_strikes += 1
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
            *([f"  backward     {backward_strikes} glyphs typed right to left "
               f"(0x8E) over {backward_runs} runs, so one byte a cell on the "
               "reverse sweeps instead of three"] if backward_runs else []),
        ]
    )
