"""Photograph -> typewriter art -> Sigma SM 8200i print job.

    python -m erika.pipeline print -t images/mwdog_crop.png -r 40

runs the optimizer, plans the head motion, writes ``results/photo.etp``, and
saves a preview of what the machine will actually put on paper. Send the
result to the typewriter with::

    python -m erika.send results/photo.etp --port /dev/cu.usbmodem1101

Subcommands:
    charset    build the Sigma charset (index -> key mapping)
    print      full run: image -> optimize -> plan -> .etp
    plan       re-plan an existing results/choices.json without re-optimizing
    verify     render a plan and diff it against optimize.py's own mockup
    calibrate  a small .etp test pattern for checking the motion codes
    area       mark the four corners of the printable area on a sheet
    sheet      an .etp that types the charset, for scanning back in
    forces     sweep the strike-force command, to find what this machine takes
    codes      put the control codes the pipeline does not use yet on paper
    rewind     measure what a reverse paper feed costs in registration
    feed       how far down the sheet the paper still feeds true
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")  # optimize.py builds a figure

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from erika import SRC_DIR, erika_codes as ec
from erika import etp, planner, preview
from erika.planner import Charset, PlanError

#: Layer schemes in layers.json whose offsets are all multiples of 0.5 -- the
#: ones the *keyboard* can place, and so the ones typeable without --fine at any
#: pitch. Written out rather than derived so that importing this module does not
#: read layers.json; ``typeable_layer_schemes`` derives the same answer, and a
#: test compares the two.
TYPEABLE_LAYER_SCHEMES = ("1x1", "1x2", "1x4", "2Hx1", "2Hx2", "2Vx1", "2Vx2", "4x1", "4x2")


def typeable_layer_schemes(pitch: int = 10, fine: bool = False) -> tuple[str, ...]:
    """Which schemes in layers.json this machine can place, at this pitch.

    Derived from the planner's own rule rather than listed, because with --fine
    the answer depends on the pitch: an eighth of a cell is one carriage step at
    pitch 15, one and a half at pitch 10, and there is no such thing as half a
    motor step. ``daisy_full`` is typeable on one of those and not the other.
    """
    with open(os.path.join(SRC_DIR, "layers.json"), encoding="utf-8") as f:
        schemes = json.load(f)
    out = []
    for name, offsets in schemes.items():
        try:
            for off_v, off_h in offsets:
                planner.offset_to_units(off_v, "vertical", pitch, fine)
                planner.offset_to_units(off_h, "horizontal", pitch, fine)
        except PlanError:
            continue
        out.append(name)
    return tuple(out)

#: Glyph used for the calibration rulers. It wants to be a narrow vertical
#: mark, so a half-step offset is obvious, and a character every type wheel
#: certainly carries. '|' is neither -- plenty of wheels place it oddly or
#: omit it, which makes it a poor reference for the one test that matters.
RULER_CHAR = "!"

#: Characters per line on the glyph-check rows.
GLYPH_CHECK_WIDTH = 33

#: The corner brackets typed by `area`: how many cells each arm runs, and what
#: draws it. '_' sits on the baseline, so it reads as an edge; '!' is the narrow
#: vertical every wheel carries (see RULER_CHAR). Together they make an L that
#: is legible at arm's length, which is the whole point of the sheet.
CORNER_ARM = 2
CORNER_EDGE_CHAR = "_"
CORNER_SIDE_CHAR = "!"

#: Lines of paper the three paper sheets -- `area`, `rewind`, `feed` -- assume by
#: default. Derived rather than chosen: every line that lands on an A4 sheet
#: loaded the way erika_codes' "Where the paper sits" describes.
#:
#: It used to be 60, with a comment claiming roughly 20 mm spare at each end. The
#: top margin turned out to be 32 mm measured, which left 11 mm under the last
#: line rather than 20 -- so the number was a guess and the guess was wrong at
#: both ends. Now it is arithmetic over two measurements.
#:
#: Deliberately past both limits the paper has: `deepest_printable_line` is three
#: lines shorter and `deepest_rewindable_line` twelve. All three sheets exist to
#: show where things stop coming out right, and a default that stopped short of
#: the trouble would hide it.
DEFAULT_PAPER_ROWS = ec.lines_on_paper()


#: The `rewind` vernier. One block of the ladder is two cells of staircase and
#: two of rule, side by side with no gap: the eye judges whether a broken line
#: is straight far better than it judges the distance between two separated
#: marks, and that judgement is the entire measurement. '_' sits on the
#: baseline, so a pair of them reads as a rule rather than as two characters --
#: the same reason `area` draws its edges with it.
REWIND_BLOCK_COLS = 4
REWIND_BAR_CHAR = "_"
REWIND_BAR_CELLS = 2

#: One spare column at the left of every band, and it is there for the labels.
#:
#: A block's number belongs under that block's staircase bar -- the bar is what
#: the number counts steps of, and a number that sits anywhere else has to be
#: matched to its bar by eye along a row of fifteen near-identical ones. So the
#: labels are right-aligned to end under the staircase bar's last cell, which
#: puts every digit in the same column whatever its sign. The minus then hangs a
#: cell to the left of the bar, and under the leftmost block -- the most negative
#: one -- that cell is off the paper unless the whole band is moved over. This is
#: that cell. It costs one column of the carriage's reach and nothing else.
REWIND_LEFT_INSET = 1

#: Half the ladder, in 1/240" platen steps. Seven either side spans +-0.74 mm
#: and 60 of the 65 columns the carriage reaches at pitch 10. It has to stay
#: under half a line: every band walks its staircase down and then has to reach
#: the next whole row without ever feeding backwards.
REWIND_HALF_RANGE = 7

#: Rows one band occupies: a header, the ladder, a clear line, the labels, and
#: a blank before the next band.
REWIND_BAND_ROWS = 5
REWIND_LADDER_ROW = 1
REWIND_LABEL_ROW = 3

#: Rewind distances the sheet tries, in whole lines, and how many times each.
#: Typed longest first so that the deepest excursion happens while the band is
#: still near the top of the paper -- the sheet is as tall as its first trial
#: reaches, not as tall as the sum of them.
#:
#: Three distances rather than one because the two failure modes look different:
#: backlash is a fixed slop and reads the same at every distance, while slip
#: accumulates and grows with it. Only the first can be corrected by winding
#: back further than asked.
REWIND_DISTANCES = (40, 20, 10)
REWIND_REPEATS = 2


#: The `feed` sheet. One bar per line, its column stepping across the page, so
#: the marks of a run of lines lie on a straight line -- and the eye judges a
#: straight line far better than it judges a distance, which is the same reason
#: the rewind sheet is a vernier.
#:
#: The stepping is what buys the sensitivity. A mark misplaced downward by `e`
#: lies off the line by `e * dx / hypot(dx, dy)`, so the shallower the run the
#: closer that gets to `e` itself: at one column per line only half the error
#: shows, and at six columns per line nearly all of it. That is what `--sweeps`
#: sets -- more sweeps, fewer lines in each, further across for every line down.
FEED_BAR_CHAR = "_"
FEED_BAR_CELLS = 2

#: Columns kept clear at the left for the line numbers, which is also the column
#: the first mark of every sweep stands in: those marks are the sheet's other
#: reading, a row of them one sweep apart for a steel rule to be laid against.
FEED_MARGIN_COLS = 4

#: A line number this often, so a kink can be named rather than counted to.
FEED_LABEL_EVERY = 5

#: How many times the pattern crosses the page. Six over sixty lines puts ten
#: marks in each sweep -- long enough to sight along, shallow enough that a
#: misplaced line shows almost its whole error.
FEED_SWEEPS = 6


@contextlib.contextmanager
def in_src_dir():
    """optimize.py and utils.py resolve everything relative to os.getcwd()."""
    old = os.getcwd()
    os.chdir(SRC_DIR)
    try:
        yield SRC_DIR
    finally:
        os.chdir(old)


# ---------------------------------------------------------------------------
# text helpers, used by the calibration jobs
# ---------------------------------------------------------------------------


def type_text(enc: etp.Encoder, text: str) -> None:
    """Emit a string as strikes. Unsupported characters become '?'."""
    for char in text:
        if char == " ":
            enc.right(2)
            continue
        glyph = ec.glyph_for_char(char) or ec.glyph_for_char("?")
        enc.strike(glyph.code, glyph.advances)


def type_line(enc: etp.Encoder, text: str, lines_after: int = 1) -> None:
    type_text(enc, text)
    enc.newline(lines_after)


def type_row(enc: etp.Encoder, marks: list[tuple[int, str]]) -> None:
    """Type pieces of text at absolute columns on one line, left to right.

    Starts from the left margin so a column means the same thing on every row.
    Everything in GLYPHS advances the head one full step -- no dead keys here --
    so the column after a piece of text is simply its column plus its length.
    Marks must be given in order and must not overlap; the encoder refuses a
    negative move, so a mistake fails here rather than on paper.
    """
    enc.carriage_return()
    col = 0
    for at, text in marks:
        enc.right(2 * (at - col))
        type_text(enc, text)
        col = at + len(text)


def glyph_check_rows(width: int = GLYPH_CHECK_WIDTH) -> list[str]:
    """Every typeable glyph, in charset order, split into printable rows.

    Charset order is also the order the charset sheet is built in, so a row
    that comes out wrong points straight at the offending index.
    """
    chars = [g.char for g in ec.GLYPHS]
    return ["".join(chars[i : i + width]) for i in range(0, len(chars), width)]


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_charset(args) -> int:
    from erika.make_charset import make_charset, parse_densities, parse_forces

    try:
        return _build_charset(args, make_charset, parse_forces, parse_densities)
    except ValueError as exc:
        # Same reasoning as run_optimizer's: the builder is a library and is
        # right to raise ValueError; a mistyped --forces should not print a
        # traceback at someone.
        raise PlanError(str(exc)) from exc


def _build_charset(args, make_charset, parse_forces, parse_densities) -> int:
    make_charset(
        name=args.name,
        pitch=args.pitch,
        cell_height=args.cell_height,
        font=args.font,
        bleed=args.bleed,
        dead_keys=args.dead_keys,
        sheet_cols=args.sheet_cols,
        scan=args.from_scan,
        deskew_scan=args.deskew,
        base_path=SRC_DIR,
        ink=args.ink,
        spread=args.spread,
        forces=parse_forces(args.forces),
        force_densities=parse_densities(args.force_density),
    )
    return 0


#: Where a target converted by `--grey` is written, relative to src/. Derived
#: and disposable, so results/ -- and a fixed name, so a run does not leave a
#: trail of them behind.
GREY_TARGET = "results/target-grey.png"


def preprocess_target(args) -> str:
    """Apply `--grey` to the target and return the path to hand the optimizer.

    ``kword`` opens its target with ``cv2.IMREAD_GRAYSCALE``, which is Rec. 601
    luma with no way to ask for anything else. So another conversion has to be a
    file on disk, written here and pointed at.

    Worth doing at all because of what the paper's section 5.3 says and leaves
    alone: with the picture reduced to some fifty printable greys, whatever the
    colour conversion merges is merged for good. ``erika.stress`` has the
    argument in full.
    """
    if args.grey == "luma":
        return args.target

    from erika import stress

    if args.grey not in stress.METHODS:
        raise PlanError(
            f"unknown --grey '{args.grey}'. Pick one of: {', '.join(stress.METHODS)}"
        )

    source = args.target if os.path.isabs(args.target) else os.path.join(SRC_DIR, args.target)
    if os.path.isdir(source):
        raise PlanError(
            f"--grey {args.grey} needs a single image; {args.target} is a directory. "
            "Convert the frames first, or run with --grey luma."
        )

    destination = os.path.join(SRC_DIR, GREY_TARGET)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    print(f"converting {args.target} to grey by {args.grey} ...")
    stress.convert_file(
        source,
        destination,
        args.grey,
        radius=args.stress_radius,
        samples=args.stress_samples,
        iterations=args.stress_iterations,
        seed=args.stress_seed,
        enhance_shadows=args.stress_shadows,
    )
    return GREY_TARGET


def align_target(args, target: str) -> str:
    """Run the crop search, if asked, and return the target to optimize.

    After ``--grey`` rather than before, for the reason erika-studio gives for
    doing its conversion once: STRESS is spatial and stochastic, so converting
    each of 64 candidates would compare 64 different pictures. Aligning the
    converted picture compares one.
    """
    if not args.align:
        return target

    # Imported here so its defaults live in one place -- the module that has the
    # measurements behind them -- without the parser having to import it.
    from erika import align

    steps = args.align_steps or align.DEFAULT_STEPS
    loops = args.align_loops or align.DEFAULT_LOOPS
    if loops < 1:
        raise PlanError(f"--align-loops must be at least 1, got {loops}")

    if os.path.isdir(os.path.join(SRC_DIR, target)):
        raise PlanError(
            f"--align needs a single image; {target} is a directory. The crop is "
            "chosen for one picture and would be wrong for the rest."
        )

    total = steps ** 3
    print(f"aligning to the character grid: {total} "
          f"crop{'' if total == 1 else 's'} x {loops} greedy "
          f"cycle{'' if loops == 1 else 's'} ...")

    # One line, rewritten in place: a 64-line wall of scores says nothing the
    # winner does not, but a search this slow should not look like a hang. Only
    # on a terminal, though -- a carriage return is not cursor movement in a
    # captured log, it is 64 lines of half-overwritten text.
    live = sys.stdout.isatty()

    def progress(done, count, crop, score):
        if live:
            print(f"\r  {done}/{count}  {crop}  {score:.3f}   ", end="", flush=True)

    try:
        result, path = align.apply_to_file(
            target, args.charset, args.row_length, args.layers,
            steps=steps, loops=loops,
            asymmetry=args.asymmetry, base_path=SRC_DIR, progress=progress,
        )
    except ValueError as exc:
        raise PlanError(str(exc)) from exc
    if live:
        print("\r" + " " * 60 + "\r", end="")

    if result.crop.is_identity:
        print(f"  the picture is already best aligned as it stands "
              f"({result.seconds:.0f}s, {result.candidates} "
              f"crop{'' if result.candidates == 1 else 's'})")
        return target
    print(f"  {result.crop}, scoring {result.score:.3f} against "
          f"{result.identity_score:.3f} unaligned (+{result.gain:.3f}) "
          f"in {result.seconds:.0f}s")
    print(f"  cropped target -> {path}")
    return path


def run_optimizer(args) -> str:
    """Run optimize.py and return the path to the choices it wrote."""
    import optimize

    pitch = Charset.load(args.charset, SRC_DIR).pitch
    allowed = typeable_layer_schemes(pitch, args.fine)
    if args.layers not in allowed:
        # A scheme that --fine would accept is a different problem from one
        # nothing can type, and the difference is the whole of what to do next.
        with_fine = typeable_layer_schemes(pitch, fine=True)
        hint = ""
        if not args.fine and args.layers in with_fine:
            hint = (
                " It is typeable with the machine's own motor steps, which is "
                "the default -- drop --keystrokes-only."
            )
        elif args.layers in typeable_layer_schemes(15, fine=True):
            hint = (
                " Its offsets are not a whole number of motor steps at pitch "
                f"{pitch}, but they are at pitch 15 -- an eighth of a cell is "
                "exactly one carriage step there. Nothing can act on that yet: "
                "make_charset builds 10 and 12, and whether this machine even "
                "accepts 15 characters per inch is what part 9 of "
                "`erika.pipeline codes` asks."
            )
        raise PlanError(
            f"layer scheme '{args.layers}' uses offsets this machine cannot hit "
            f"at pitch {pitch}"
            + (" by keystroke." if not args.fine else ".")
            + hint
            + f" Pick one of: {', '.join(allowed)}"
        )

    target = preprocess_target(args)
    target = align_target(args, target)

    choices_path = os.path.join(SRC_DIR, "results", "choices.json")
    if os.path.exists(choices_path):
        os.remove(choices_path)

    if args.match_blur > 0:
        # Imported here, not in the parser: softmatch is a numba module, and
        # `--match-block`'s default lives in it precisely so that `calibrate
        # --help` does not pay two seconds to find out what it is.
        from erika import softmatch

        block = args.match_block or softmatch.DEFAULT_BLOCK
        charset = Charset.load(args.charset, SRC_DIR)
        try:
            softmatch.validate(args.match_blur, block, (charset.cell_h, charset.cell_w))
        except ValueError as exc:
            # A library raising ValueError is right; a CLI printing a traceback
            # over a mistyped flag is not.
            raise PlanError(str(exc)) from exc
        softmatch.install(args.match_blur, block)
        print(f"scoring on {block}x{block} block tone at weight "
              f"{args.match_blur} as well as per pixel")

    print(f"optimizing {target} at {args.row_length} columns, "
          f"{args.layers} layers, {args.num_loops} loops ...")
    try:
        optimize.kword(
            charset=args.charset,
            target=target,
            layers=args.layers,
            row_length=args.row_length,
            num_loops=args.num_loops,
            init_mode=args.init_mode,
            asymmetry=args.asymmetry,
            search=args.search,
            init_temp=args.init_temp,
            display=0,
            shuffle=args.shuffle,
            out_file="results/final.png",
            nowait=True,
        )
    except Exception as exc:  # noqa: BLE001
        # kword writes choices.json before its final plotting block, so a
        # failure in the cosmetic tail still leaves us a usable result.
        if not os.path.exists(choices_path):
            raise
        print(f"  note: optimize.py raised after writing choices.json ({exc!r})")

    if not os.path.exists(choices_path):
        raise PlanError("optimize.py did not write results/choices.json")
    return choices_path


def plan_and_encode(args, choices_path: str) -> tuple[planner.Plan, etp.Job]:
    charset = Charset.load(args.charset, SRC_DIR)
    plan = planner.build_plan(
        choices_path,
        charset,
        home_each_row=not args.no_home,
        boustrophedon=not args.no_serpentine,
        fine=args.fine,
        indent=args.indent,
        offset=args.offset,
    )
    job = planner.encode(plan, settle_ms=args.settle_ms, cr_delay_ms=args.cr_delay_ms,
                         no_advance=args.no_advance, backward=args.backward)
    return plan, job


def write_outputs(args, plan: planner.Plan, job: etp.Job) -> None:
    out = args.out if os.path.isabs(args.out) else os.path.join(SRC_DIR, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    size = etp.save(out, job)

    print(f"\nprint job -> {out} ({size} bytes)")
    print(planner.summarize(plan, job, ops_per_second=args.ops_per_second))

    if not args.no_preview:
        from utils import prep_charset

        with in_src_dir() as base:
            tiles, _, _ = prep_charset(args.charset, base)
            paths = preview.save_previews(
                plan, tiles, os.path.join(base, "results"), jitter=args.jitter
            )
            ref = os.path.join(base, "results", "final.png")
            print("\npreview:")
            for label, path in paths.items():
                print(f"  {label:<8} {os.path.relpath(path, base)}")
            if os.path.exists(ref):
                stats = preview.compare(preview.render(plan, tiles), ref)
                _report_comparison(stats)


def _report_comparison(stats: dict) -> bool:
    if not stats["ok"]:
        print(f"  MISMATCH vs optimize.py mockup: {stats['reason']}")
        return False
    if stats["max_abs"] <= 1:
        print("  plan reproduces optimize.py's mockup exactly")
        return True
    pct = 100 * stats["differing_px"] / stats["total_px"]
    print(f"  differs from optimize.py's mockup: max {stats['max_abs']}/255, "
          f"{pct:.3f}% of pixels")
    return stats["max_abs"] <= 4


def cmd_print(args) -> int:
    with in_src_dir():
        choices_path = run_optimizer(args)
    plan, job = plan_and_encode(args, choices_path)
    write_outputs(args, plan, job)
    return 0


def cmd_plan(args) -> int:
    choices_path = args.choices
    if not os.path.isabs(choices_path):
        choices_path = os.path.join(SRC_DIR, choices_path)
    plan, job = plan_and_encode(args, choices_path)
    write_outputs(args, plan, job)
    return 0


def cmd_verify(args) -> int:
    from utils import prep_charset

    choices_path = args.choices
    if not os.path.isabs(choices_path):
        choices_path = os.path.join(SRC_DIR, choices_path)
    charset = Charset.load(args.charset, SRC_DIR)
    plan = planner.build_plan(choices_path, charset)
    with in_src_dir() as base:
        tiles, _, _ = prep_charset(args.charset, base)
        rendered = preview.render(plan, tiles)
        ref = args.reference
        if not os.path.isabs(ref):
            ref = os.path.join(base, ref)
        stats = preview.compare(rendered, ref)
    print(f"comparing plan render against {os.path.basename(ref)}")
    return 0 if _report_comparison(stats) else 1


def cmd_calibrate(args) -> int:
    """A short test pattern that exercises every motion code.

    Type this first. It is the only way to confirm the half-step code, which
    the firmware's own key-code comments do not document.
    """
    enc = etp.Encoder()
    n = 20

    type_line(enc, "SIGMA CALIBRATION")
    enc.newline(1)

    # 1. Full-step ruler: a reference comb at one-cell pitch.
    type_line(enc, "1 FULL STEP RULER")
    for _ in range(n):
        enc.strike(ec.glyph_for_char(RULER_CHAR).code)
        enc.right(2)
    enc.newline(2)

    # 2. The same ruler struck twice over, the second pass shifted half a cell.
    #    Both passes are on one line, so a working half step shows as pairs of
    #    bars and a dead one shows as the ruler unchanged.
    type_line(enc, "2 HALF STEP (EACH BAR SHOULD GAIN A TWIN BESIDE IT)")
    for _ in range(n):
        enc.strike(ec.glyph_for_char(RULER_CHAR).code)
        enc.right(2)
    enc.carriage_return()
    enc.right(1)  # <- the half-step under test
    for _ in range(n):
        enc.strike(ec.glyph_for_char(RULER_CHAR).code)
        enc.right(2)
    enc.newline(2)

    # 3. Line-feed pitch. Four marks driven by the detented full-line
    #    mechanism, then eight by the half-line key. The planner mixes the two
    #    -- whole-line gaps go through NEWLINE, odd ones through half steps --
    #    so what matters is that two half lines come to exactly one full line.
    #    Everything sits in one column, so a feed that disturbs the carriage
    #    shows up as a sideways shift.
    type_line(enc, "3 LINE FEED PITCH (LOWER GAPS = HALF THE UPPER ONES)")
    ladder_column = 18
    for _ in range(4):
        enc.carriage_return()
        enc.right(2 * ladder_column)
        enc.strike(ec.glyph_for_char("_").code)
        enc.newline(1)  # one whole line, the reference
    for _ in range(8):
        enc.carriage_return()
        enc.right(2 * ladder_column)
        enc.strike(ec.glyph_for_char("_").code)
        enc.down(1)  # half a line, the mechanism under test
    enc.newline(2)

    # 4. Overstrike registration: O, back up, then a hyphen through it.
    type_line(enc, "4 OVERSTRIKE (O SHOULD BE STRUCK THROUGH)")
    for _ in range(8):
        enc.strike(ec.glyph_for_char("O").code)
        enc.left(2)
        enc.strike(ec.glyph_for_char("-").code)
        enc.right(2)
    enc.newline(2)

    # 5. Round trip: out and back must land on the same column.
    type_line(enc, "5 ROUND TRIP (THE TWO X SHOULD COINCIDE)")
    enc.right(40)
    enc.strike(ec.glyph_for_char("X").code)
    enc.left(42)  # 40 out, plus the 2 the strike advanced
    enc.right(40)
    enc.strike(ec.glyph_for_char("X").code)
    enc.newline(2)

    # 6. Every typeable glyph, in charset order. This is the only check that
    #    the byte -> glyph table in erika_char_map.cpp matches the type wheel
    #    actually fitted. It matters more than it looks: the optimizer chooses
    #    characters by how much ink they lay down, so a wheel that disagrees
    #    with the table makes every tonal decision in the picture wrong.
    type_line(enc, "6 GLYPH CHECK (COMPARE WITH THE LIST THE TOOL PRINTS)")
    for row in glyph_check_rows():
        type_line(enc, row)
    enc.newline(2)

    enc.carriage_return()
    enc.end()

    job = etp.Job(body=enc.body(), cols=n, rows=14, strikes=enc.strikes,
                  pitch=args.pitch, home_each_row=True)
    out = args.out if os.path.isabs(args.out) else os.path.join(SRC_DIR, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    size = etp.save(out, job)
    print(f"calibration job -> {out} ({size} bytes, {job.strikes} strikes)")
    print("\nWhat to check on the printed sheet:")
    print(f"  1  A row of evenly spaced '{RULER_CHAR}'. The reference for part 2.")
    print("  2  The same ruler, struck a second time half a step across, so every")
    print("     bar should have gained a twin close beside it. If part 2 looks")
    print("     like part 1 -- single bars, same spacing -- the half step did")
    print("     nothing and this machine uses a code other than 0x73. Find the")
    print("     right one, then change HALF_STEP_FORWARD in erika/erika_codes.py")
    print("     and ERIKA_HALF_STEP_FWD in erika_ai/src/erika_image.h. The test")
    print("     suite fails if you change only one of them.")
    print("  3  Twelve marks in ONE column: four gaps of a whole line, then seven")
    print("     of half a line. The lower gaps must be exactly half the upper")
    print("     ones -- that is what lets the planner mix the two. Uneven gaps")
    print("     mean the platen slips, which prints as banding. Any sideways")
    print("     shift means a line feed is nudging the carriage.")
    print("  4  A clean strike-through confirms backspace registration.")
    print("  5  Two X marks side by side mean the carriage loses steps.")
    print("  6  Part 6 must read exactly as below. A character that comes out")
    print("     wrong means the type wheel disagrees with erika_char_map.cpp --")
    print("     the optimizer picks glyphs by how much ink they lay down, so a")
    print("     wheel that disagrees makes every tone in the picture wrong.")
    print("     Fix the offending entry in erika_ai/src/erika_char_map.cpp and")
    print("     erika/erika_codes.py, then rebuild the charset.")
    print()
    for row in glyph_check_rows():
        print(f"       {row}")
    return 0


def cmd_area(args) -> int:
    """Bracket the four corners of the area a print can occupy.

    Two of the four edges are real machine limits and two are not, which is the
    thing this sheet is for. The width is hard: the carriage reaches 65 columns
    at pitch 10, 78 at pitch 12, and the planner refuses anything wider. The
    height is not a limit at all -- the platen keeps feeding as long as it grips
    the sheet -- so the vertical extent is whatever --rows asks for, and what
    the sheet shows is whether that many lines actually fit on the paper.

    The top-left bracket lands wherever the head is when the job starts, so the
    sheet marks the area relative to how the paper is loaded, exactly as a photo
    print would be.
    """
    limit = ec.MAX_COLUMNS[args.pitch]
    columns = args.columns or limit
    if columns > limit:
        raise PlanError(
            f"{columns} columns is past the carriage limit of {limit} at pitch "
            f"{args.pitch}"
            + (" -- pitch 12 reaches 78." if args.pitch == 10 else ".")
        )
    if columns < 2 * CORNER_ARM or args.rows < 2 * CORNER_ARM:
        raise PlanError(
            f"an area of {columns} x {args.rows} cells leaves no room for corner "
            f"brackets {CORNER_ARM} cells on a side"
        )

    w_mm = columns * ec.PITCH_WIDTH_MM[args.pitch]
    h_mm = args.rows * ec.LINE_HEIGHT_MM

    enc = etp.Encoder()
    edge = CORNER_EDGE_CHAR * CORNER_ARM
    row = 0

    def go_to(target: int) -> None:
        """Feed to an absolute row. Every gap here is a whole number of lines,
        so this always drives the detented line-feed mechanism."""
        nonlocal row
        enc.newline(target - row)
        row = target

    # Top edge, then the sides hanging below it.
    type_row(enc, [(0, edge), (columns - CORNER_ARM, edge)])
    for _ in range(CORNER_ARM - 1):
        go_to(row + 1)
        type_row(enc, [(0, CORNER_SIDE_CHAR), (columns - 1, CORNER_SIDE_CHAR)])

    # What the sheet is a picture of, typed on the sheet -- otherwise two of
    # them side by side are hard to tell apart. Skipped if the area is too
    # small to hold it without touching the brackets.
    caption = (f"{columns} X {args.rows} CELLS = {w_mm:.0f} X {h_mm:.0f} MM "
               f"AT PITCH {args.pitch}")
    if args.rows > 2 * CORNER_ARM and len(caption) <= columns:
        go_to(CORNER_ARM)
        type_row(enc, [(0, caption)])

    # Bottom sides, then the bottom edge.
    for arm in range(CORNER_ARM - 1, 0, -1):
        go_to(args.rows - 1 - arm)
        type_row(enc, [(0, CORNER_SIDE_CHAR), (columns - 1, CORNER_SIDE_CHAR)])
    go_to(args.rows - 1)
    type_row(enc, [(0, edge), (columns - CORNER_ARM, edge)])

    enc.newline(2)  # roll the sheet clear of the platen, as planner.encode does
    enc.end()

    job = etp.Job(body=enc.body(), cols=columns, rows=args.rows,
                  strikes=enc.strikes, pitch=args.pitch, home_each_row=True)
    out = args.out if os.path.isabs(args.out) else os.path.join(SRC_DIR, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    size = etp.save(out, job)
    print(f"print-area job -> {out} ({size} bytes, {job.strikes} strikes)")
    print(f"  area   {columns} x {args.rows} cells "
          f"= {w_mm:.0f} x {h_mm:.0f} mm at pitch {args.pitch}")
    print(f"  print  -r {columns} fills it edge to edge")
    print("\nLoad a sheet the way you would for a photo, with the paper where you")
    print("want the top-left of the print, then type this:")
    print(f"  python -m erika.send {os.path.relpath(out, SRC_DIR)} --print")
    print("\nWhat to check on the printed sheet:")
    print("  - Four brackets, each pointing into the area a print can occupy.")
    print(f"    The width is the machine's own limit -- {limit} columns at pitch "
          f"{args.pitch},")
    print("    which is the widest -r the planner will accept.")
    print("  - A right-hand bracket that is short, smeared or missing means the")
    print("    carriage hit its stop early: type it again with --columns lowered")
    print("    until it comes out clean, and use that as your -r ceiling.")
    print("  - The height is not a machine limit, only paper. If the bottom")
    print("    brackets ran off the sheet or the platen lost its grip, lower")
    print("    --rows; that number caps how tall a print can be on this paper.")
    return 0


def _feed_down(enc: etp.Encoder, steps: int) -> None:
    """Feed the paper down by a count of 1/240" platen steps.

    Forward only, and it says so rather than obliging: on this sheet a backwards
    feed *is* the measurement, so one that slipped in by accident somewhere else
    would be indistinguishable from the answer.

    A whole number of lines goes out as NEWLINE for the reason planner.encode
    gives -- the detented line-feed mechanism is more repeatable than a stack of
    half-line keystrokes -- which matters more here than anywhere else, because
    repeatability is the quantity being put on the paper.
    """
    if steps < 0:
        raise PlanError(
            f"the rewind sheet tried to feed the paper back {-steps} platen "
            "steps outside a trial. Every motion on this sheet is downward "
            "except the one reversal each band exists to measure"
        )
    per_line = 2 * ec.PLATEN_STEPS_PER_HALF_LINE
    if steps and steps % per_line == 0:
        enc.newline(steps // per_line)
        return
    enc.down(steps // ec.PLATEN_STEPS_PER_HALF_LINE)
    enc.down_fine(steps % ec.PLATEN_STEPS_PER_HALF_LINE)


def _rewind_staircase(enc: etp.Encoder, half_range: int, step: int) -> int:
    """Type the reference: one bar per block, each `step` platen steps lower.

    Returns how far the paper moved, in platen steps. Blocks run left to right
    and the steps run downward, so neither mechanism reverses inside the pass --
    which is the point. A staircase that had to reverse to draw itself would be
    measuring the same backlash it is there to detect.
    """
    enc.carriage_return()
    col = 0
    for j in range(2 * half_range + 1):
        if j:
            enc.down_fine(step)
        target = REWIND_LEFT_INSET + j * REWIND_BLOCK_COLS
        enc.right(2 * (target - col))
        type_text(enc, REWIND_BAR_CHAR * REWIND_BAR_CELLS)
        col = target + REWIND_BAR_CELLS
    return 2 * half_range * step


def _rewind_rule(enc: etp.Encoder, half_range: int) -> None:
    """Type the probe: one bar per block, all of them at the same height."""
    type_row(enc, [(REWIND_LEFT_INSET + j * REWIND_BLOCK_COLS + REWIND_BAR_CELLS,
                    REWIND_BAR_CHAR * REWIND_BAR_CELLS)
                   for j in range(2 * half_range + 1)])


def _rewind_labels(enc: etp.Encoder, half_range: int, step: int) -> None:
    """Under each block, the error in platen steps if the null lands there.

    Centred on that block's staircase bar, and centred *exactly* -- which the
    cell grid cannot do. A bar is two cells wide and a one-character label is one
    cell, so on whole columns a lone digit sits under the left half of its bar or
    the right half, never the middle, while a label with a minus sign in front of
    it covers the bar exactly. Every label from zero up is a lone digit, so the
    row read as though the positive half had been nudged sideways against the
    negative half -- on the one row of this sheet whose whole job is to say
    which block you are looking at.

    The machine has a half-step, and it is the same half-step the entire layer
    scheme is built on, so this row is placed in half-steps rather than columns.
    A two-character label lands on the cell grid exactly where it always did; a
    lone digit lands half a cell left of that, centred; a three-character one
    half a cell further left again, into REWIND_LEFT_INSET.
    """
    enc.carriage_return()
    x = 0  # half-steps right of the left margin, which is what the head moves in
    for j in range(2 * half_range + 1):
        text = str((j - half_range) * step)
        bar = REWIND_LEFT_INSET + j * REWIND_BLOCK_COLS
        # The bar's ink runs from 2*bar to 2*bar + 2*REWIND_BAR_CELLS half-steps,
        # so its middle is REWIND_BAR_CELLS along. The label is len(text) cells
        # wide, so it starts half its own width before that.
        start = 2 * bar + REWIND_BAR_CELLS - len(text)
        enc.right(start - x)
        type_text(enc, text)
        x = start + 2 * len(text)


def _rewind_band(enc: etp.Encoder, *, y: int, top: int, lines: int,
                 half_range: int, step: int, mechanism: str, header: str) -> int:
    """Type one vernier band; return where the paper stands afterwards.

    ``y`` and ``top`` are platen steps below where the sheet began. The whole of
    the arithmetic is this: the staircase puts a bar at ``top + j`` in block j,
    the rewind is meant to bring the paper back to ``top``, and the rule is then
    typed half the staircase's span lower. So the rule lands at
    ``top + half_range * step + e`` for whatever ``e`` the rewind got wrong, and
    it meets block j exactly when ``(j - half_range) * step == e``. The number
    under the block is that value, so the sheet is read rather than measured.
    """
    per_line = 2 * ec.PLATEN_STEPS_PER_HALF_LINE
    _feed_down(enc, top - y)
    type_row(enc, [(0, header)])

    ladder = top + REWIND_LADDER_ROW * per_line
    _feed_down(enc, ladder - top)
    y = ladder + _rewind_staircase(enc, half_range, step)

    # Down the sheet and back up again: the one reversal the band is about, and
    # the only motion anywhere on the sheet that is not forward. The excursion
    # is a whole number of lines below where the staircase *started* so that the
    # return can be one too -- a rewind that also had to undo the staircase's
    # platen steps would be two mechanisms answering as one.
    _feed_down(enc, ladder + lines * per_line - y)
    if mechanism == "fine":
        enc.up_fine(lines * per_line)
    else:
        enc.up(2 * lines)
    y = ladder  # where the machine now believes the paper is

    _feed_down(enc, half_range * step)
    _rewind_rule(enc, half_range)
    y = ladder + half_range * step

    labels = top + REWIND_LABEL_ROW * per_line
    _feed_down(enc, labels - y)
    _rewind_labels(enc, half_range, step)
    return labels


def _ladder_check_band(enc: etp.Encoder, *, y: int, top: int,
                       half_range: int, step: int) -> int:
    """A band with no reversal in it: what the ladder itself is worth.

    Every reading on this sheet is in units of one ``down_fine`` step, so a
    ladder whose steps are uneven -- or whose steps the machine quietly drops --
    would move every null on the sheet without looking like a fault. This band
    types the same staircase and then a rule one whole line below the row it
    started on, so the gap between the two closes by exactly one ladder step per
    block: a straight wedge if the ladder is even, and a visible kink where it
    is not.

    It cannot be folded into a trial, tempting as a zero-line rewind sounds. The
    staircase ends below where it began, and the only way back up to the rule's
    row is a reversal -- so a band with no rewind in it has to be a band with
    different geometry.
    """
    per_line = 2 * ec.PLATEN_STEPS_PER_HALF_LINE
    _feed_down(enc, top - y)
    type_row(enc, [(0, f"LADDER CHECK  NO REWIND  GAP CLOSES {step} PER BLOCK")])

    ladder = top + REWIND_LADDER_ROW * per_line
    _feed_down(enc, ladder - top)
    y = ladder + _rewind_staircase(enc, half_range, step)

    _feed_down(enc, ladder + per_line - y)
    _rewind_rule(enc, half_range)
    y = ladder + per_line

    labels = top + REWIND_LABEL_ROW * per_line
    _feed_down(enc, labels - y)
    _rewind_labels(enc, half_range, step)
    return labels


def _parse_rewind_lines(text: str | None) -> tuple[int, ...]:
    """``--lines 40,20,10`` -> (40, 20, 10), longest first."""
    if text is None:
        return REWIND_DISTANCES
    try:
        values = tuple(int(part, 0) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise PlanError(f"--lines wants whole numbers of lines: {text!r}") from exc
    if not values:
        raise PlanError("--lines named no distances")
    if any(v < 1 for v in values):
        raise PlanError(
            "a rewind of less than one line has nothing to measure -- the "
            "staircase and the rule would sit inside the same line feed"
        )
    # Longest first so the deepest excursion happens while the band is still
    # near the top of the paper. The sheet is as tall as its first trial
    # reaches, not as tall as the sum of them.
    return tuple(sorted(set(values), reverse=True))


def cmd_rewind(args) -> int:
    """Measure whether the platen comes back to where it was after a rewind.

    The planner sorts every strike so the paper only ever feeds forward, because
    reversing the feed introduces backlash that shows up as banding. That rule
    is also what stands between this pipeline and a print typed with more than
    one type wheel: a wheel is changed by hand, so each wheel wants a complete
    pass over the whole sheet, and the second pass can only begin by winding the
    paper back to the top. Whether that is worth building turns on one number
    nobody here has measured -- how far off the paper lands when it comes back --
    and this sheet is how to measure it.

    The reading is a vernier, so no scanner is involved. Each band is typed in
    two passes with the rewind between them:

        pass 1   a staircase of bars, one 1/240" platen step lower each block
        ...      down `--lines` lines and back up: the reversal under test
        pass 2   a single level rule, half the staircase's span below the row
                 the staircase started on

    On a machine that returns exactly, the rule crosses the staircase at the
    middle block. Where it actually crosses is the error, and every block is
    labelled with what it would mean in platen steps -- positive when the paper
    came back short, which is the only direction backlash can err in.

    What the sheet does not measure is the carriage. Its return to the left
    margin is exercised by every row of every print already, and a wheel change
    does not ask anything new of it; the platen coming back up a whole image is
    a motion nothing in this pipeline has ever made.
    """
    per_line = 2 * ec.PLATEN_STEPS_PER_HALF_LINE
    limit = ec.MAX_COLUMNS[args.pitch]
    half, step = args.half_range, args.ladder_step
    if half < 1:
        raise PlanError("--half-range needs at least one block either side")
    if step < 1:
        raise PlanError("--ladder-step is a count of platen steps, so at least one")
    width = REWIND_LEFT_INSET + (2 * half + 1) * REWIND_BLOCK_COLS
    if width > limit:
        raise PlanError(
            f"a ladder of {2 * half + 1} blocks is {width} columns wide, past "
            f"the carriage's {limit} at pitch {args.pitch}. Lower --half-range "
            f"to {((limit - REWIND_LEFT_INSET) // REWIND_BLOCK_COLS - 1) // 2} "
            "or less"
            + (", or switch to pitch 12." if args.pitch == 10 else ".")
        )
    if args.repeat < 1:
        raise PlanError("--repeat needs at least one run of each distance")

    distances = _parse_rewind_lines(args.lines)
    mech = "FINE 1/240" if args.mechanism == "fine" else "HALF-LINE"

    # How deep the staircase walks, which is what every layout rule below is
    # about: it walks down, and the only way back is the reversal under test.
    # Also what keeps a label inside its block: the widest is "-" and two digits,
    # and a block is four columns. See _rewind_labels.
    depth = 2 * half * step
    room = (REWIND_LABEL_ROW - REWIND_LADDER_ROW) * per_line
    if depth >= room:
        raise PlanError(
            f"the ladder walks {depth} platen steps down, into the labels "
            f"{room} below it. Lower --ladder-step or --half-range so that "
            f"2 * half_range * step stays under {room}"
        )
    if depth > min(distances) * per_line:
        raise PlanError(
            f"the ladder walks {depth} platen steps down but the shortest "
            f"rewind is {min(distances)} line(s) = {min(distances) * per_line}. "
            "The excursion has to start below where the staircase ended, so a "
            "ladder deeper than the shortest distance would need a backwards "
            "feed to reach it -- drop the short distances from --lines, or "
            "lower --ladder-step"
        )
    if not args.no_ladder_check and depth >= per_line:
        raise PlanError(
            f"the ladder walks {depth} platen steps down and the ladder-check "
            f"band puts its rule one line ({per_line}) below the staircase's "
            "first rung, which the staircase has already passed. Lower "
            "--ladder-step, or pass --no-ladder-check and check the ladder on "
            "a separate sheet"
        )

    # Lay the sheet out before encoding any of it, so a sheet that would not fit
    # the paper is refused rather than typed off the bottom of it.
    row = 2  # under the caption, with a blank line between
    bands: list[tuple[int, int, int]] = []  # (top row, lines, run)
    if not args.no_ladder_check:
        bands.append((row, 0, 0))
        row += REWIND_BAND_ROWS
    for lines in distances:
        for run in range(1, args.repeat + 1):
            bands.append((row, lines, run))
            row += REWIND_BAND_ROWS
    typed_rows = row - 1  # the trailing blank of the last band is not typed on

    def reach(top: int, lines: int) -> int:
        """The row a band winds down to before it comes back.

        Bands are stacked down the page and each excursion starts from its own
        band, so *no two of them reach the same depth* -- the last trial of the
        longest distance goes furthest, and by more than the difference between
        the distances. That is invisible on the paper unless it is written
        there, and it is what tells a band that failed because the platen missed
        from one that failed because the sheet stopped being held.
        """
        return top + REWIND_LADDER_ROW + lines

    deepest = max(reach(top, lines) for top, lines, _ in bands) + 1
    needed = max(typed_rows, deepest)
    if needed > args.rows:
        raise PlanError(
            f"the sheet needs {needed} lines but --rows is {args.rows}. The "
            f"bands themselves take {typed_rows}; the rest is how far the "
            f"{max(distances)}-line excursion reaches below its own band. Drop a "
            "distance from --lines, lower --repeat, or raise --rows if the "
            "paper is longer than `area` says"
        )

    enc = etp.Encoder()
    type_row(enc, [(0, f"REWIND REGISTRATION  PITCH {args.pitch}  "
                       f"STEP {step}/240 IN  {mech}")])
    y = 0
    for top, lines, run in bands:
        top_steps = top * per_line
        if lines == 0:
            y = _ladder_check_band(enc, y=y, top=top_steps, half_range=half,
                                   step=step)
        else:
            y = _rewind_band(
                enc, y=y, top=top_steps, lines=lines, half_range=half, step=step,
                mechanism=args.mechanism,
                header=f"REWIND {lines} LINES  RUN {run}  {mech}  "
                       f"TO ROW {reach(top, lines)}",
            )

    enc.newline(2)  # roll the sheet clear of the platen, as planner.encode does
    enc.end()

    job = etp.Job(body=enc.body(), cols=width, rows=needed,
                  strikes=enc.strikes, pitch=args.pitch, home_each_row=True)
    out = args.out if os.path.isabs(args.out) else os.path.join(SRC_DIR, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    size = etp.save(out, job)
    trials = [b for b in bands if b[1]]
    print(f"rewind-registration job -> {out} ({size} bytes, {job.strikes} strikes)")
    print(f"  {len(trials)} trials at {', '.join(str(d) for d in distances)} lines"
          f", {args.repeat} run(s) each, reversed by "
          + ("0xA6 platen steps" if args.mechanism == "fine"
             else "half-line keystrokes"))
    print(f"  ladder +-{half * step} platen steps in {step}-step blocks = "
          f"+-{half * step / ec.PLATEN_STEPS_PER_INCH * 25.4:.2f} mm, "
          f"{width} columns, {needed} lines of paper")
    print("\nLoad a sheet as you would for a print and type this:")
    print(f"  python -m erika.send {os.path.relpath(out, SRC_DIR)} --print")
    print("\nDo not touch the paper release or the platen knob while it runs --")
    print("the sheet measures the machine's own return, and a hand on the knob")
    print("is a different experiment.")
    print("\nHow to read it:")
    print("  Each band is a descending staircase of bars with a level rule cutting")
    print("  across it. Find the block where the staircase bar and the rule beside")
    print("  it form one straight line, and read the number underneath.")
    print("    0        the platen came back exactly. Multi-wheel printing is a")
    print("             matter of planning, not of mechanism.")
    print("    positive the rewind fell short by that many 1/240 inch steps.")
    print("             The same reading at every distance is backlash, and a")
    print("             second pass can simply wind back that much further.")
    print("             A reading that grows with the distance is slip, and it")
    print("             cannot be corrected -- only measured again per distance.")
    print("    negative the paper overshot, which backlash cannot cause. Check")
    print("             the ladder-check band before believing it.")
    print("    nothing  no block lines up: the error is past the ladder's reach.")
    print(f"             Run it again with --ladder-step {step * 2}: double the")
    print("             range, half the resolution.")
    print("  Runs of the same distance that read differently are the answer that")
    print("  ends the idea: an error that is not repeatable cannot be planned")
    print("  around. Scatter of one step is workable; a whole cell is not.")
    print("\n  One band off and the rest at 0 is a different finding, and the")
    print("  paper is the first suspect rather than the platen. The bands are")
    print("  stacked down the page, so no two excursions reach the same depth --")
    print("  each is labelled on the paper with the row it winds down to. Read")
    print(f"  against a sheet at the machine's paper mark: "
          f"{ec.DEFAULT_TOP_MARGIN_MM:.0f} mm above the first")
    print(f"  line, {ec.PAPER_HEIGHT_MM:.0f} mm of paper. The mark is a default "
          "and not a limit -- clamp")
    print("  the sheet further through and every line below moves up with it.")
    trials = [(top, lines, run) for top, lines, run in bands if lines]
    floor = max(reach(top, lines) for top, lines, _ in trials)
    for top, lines, run in trials:
        at = reach(top, lines)
        print(f"    {lines:>3} lines run {run}  to line {at:>3}"
              f"  {ec.line_on_paper_mm(at):>4.0f} mm down,"
              f"{ec.paper_under_line_mm(at):>4.0f} mm under it"
              + ("   <- deepest" if at == floor else ""))
    print("  The last figure is the one that carries between sheets: the feed")
    print("  rollers are where they are, so a different paper length or a")
    print("  different loading moves the line numbers and not it. On this")
    print(f"  machine winding back stops working with about "
          f"{ec.REWIND_TAIL_MM:.0f} mm still under the")
    print(f"  print line -- while feeding *forward* goes on to about "
          f"{ec.FEED_TAIL_MM:.0f} mm, so a")
    print("  sheet nearly out can still be pushed along and not pulled back.")
    print("  Backlash misses on every band alike and slip misses more as the")
    print("  distance grows. A single band missing is neither, and the likeliest")
    print("  cause is the sheet's bottom edge leaving the feed rollers at the")
    print("  bottom of that one excursion -- which is a fact about your paper and")
    print("  how it was fed, not about the machine. `area` measures how far down")
    print("  the platen still holds it; a print that stays inside that can be")
    print("  wound back, and one that runs past it cannot.")
    print("\n  For scale: one step is 1/240 in. A layer offset of half a cell is "
          f"{ec.PLATEN_STEPS_PER_HALF_LINE} steps,")
    print(f"  and a whole line is {per_line}. An error of {ec.PLATEN_STEPS_PER_HALF_LINE}"
          " puts the second wheel's ink exactly where")
    print("  the first wheel's half-line layer already went.")
    print("\n  The ladder-check band has no rewind in it. Its gap must close by one")
    print("  step per block in a straight wedge; a kink there means the ladder is")
    print("  uneven and every other reading on the sheet is shifted with it.")
    return 0



def _feed_title_len(rows: int, sweeps: int, pitch: int) -> int:
    """How wide the `feed` sheet's own caption is, in cells."""
    return len(f"FORWARD FEED  PITCH {pitch}  {rows} LINES  {sweeps} SWEEPS")


def feed_layout(rows: int, sweeps: int, pitch: int) -> tuple[int, int, int]:
    """How the `feed` sheet lays out: (marks per sweep, columns per line, width).

    Every line from 0 to ``rows - 1`` carries a mark, line 0 included -- it shares
    its line with the title rather than giving the title a line of its own.

    That is not tidiness. Whoever reads this sheet takes the topmost mark as the
    datum and measures down from it, because it is the only thing on the sheet
    that could be one; an advice note calling some other line "the first line"
    is a note nobody can act on. So the topmost mark *is* line 0, and the count
    is still lines below the first thing typed -- the same count the rewind
    sheet's depths use, so a reading from one can be held against the other.
    """
    marks = rows
    if marks < 2:
        raise PlanError(
            f"--rows {rows} leaves {max(marks, 0)} line(s) to mark. A feed is a "
            "distance between two of them, so there is nothing to measure"
        )
    if sweeps < 1:
        raise PlanError("--sweeps needs at least one pass across the page")
    per_sweep = -(-marks // sweeps)  # ceil, so the last sweep is the short one
    if per_sweep < 2:
        raise PlanError(
            f"{sweeps} sweeps over {marks} marked lines is fewer than two marks "
            "in some of them, and two marks are a straight line whatever the "
            "feed did. Lower --sweeps"
        )
    span = ec.MAX_COLUMNS[pitch] - FEED_MARGIN_COLS - FEED_BAR_CELLS
    step = span // (per_sweep - 1)
    if step < 1:
        raise PlanError(
            f"{per_sweep} marks in a sweep will not fit across {span} columns at "
            f"pitch {pitch} with one column each. Raise --sweeps so that fewer "
            "lines share a pass"
        )
    # Wide enough for the marks or for the title beside the first of them,
    # whichever reaches further: at two sweeps the marks barely leave the margin
    # and the title is the widest thing on the sheet.
    width = max(
        FEED_MARGIN_COLS + (per_sweep - 1) * step + FEED_BAR_CELLS,
        FEED_MARGIN_COLS + FEED_BAR_CELLS + 2 + _feed_title_len(rows, sweeps, pitch),
    )
    return per_sweep, step, width


def cmd_feed(args) -> int:
    """Does the paper still feed true this far down the sheet?

    The companion to `rewind`, and the more consequential of the two. That sheet
    measured a reversal and found the platen exact -- but only while the paper
    was still held: past a depth that belongs to the sheet and how it was fed,
    its readings fell off the ladder entirely. A sheet that has lost its rollers
    can still be dragged roughly along in one direction, though, so whether a
    plain forward-feeding print is affected at the same depth is a separate
    question. It is also the larger one. Every print feeds down to its last line,
    so if the answer is no, the bottom of a tall print has been landing wherever
    the paper happened to go -- with the plan verified, the CRC passing, and
    nothing anywhere saying so.

    Nothing on this sheet ever feeds backwards. There is no vernier and no
    second pass: one bar per line, its column stepping across the page, so each
    run of lines lays its marks along a straight line. Three readings come out of
    the one pattern:

      - **Each sweep is straight.** A line the platen placed short or long lies
        off it, and a bend or a kink names the line it started at. Good for
        something like a third of a millimetre by eye, better with a straightedge.
      - **The leftmost marks are evenly spaced.** One per sweep, a known number
        of lines apart. A feed that shortens gradually never kinks anything but
        does close these up, and a gap out of step with the rest is that.
      - **The first mark to the last is a known distance.** Printed with the
        advice, for a steel rule: the absolute check the other two cannot make,
        since both compare the sheet against itself.

    Feed the paper the way it was fed for the rewind sheet or the two do not
    compare -- the depth at issue belongs to the sheet and its loading rather
    than to the machine.
    """
    per_sweep, step, width = feed_layout(args.rows, args.sweeps, args.pitch)

    bar = FEED_BAR_CHAR * FEED_BAR_CELLS
    title = f"FORWARD FEED  PITCH {args.pitch}  {args.rows} LINES  {args.sweeps} SWEEPS"
    assert len(title) == _feed_title_len(args.rows, args.sweeps, args.pitch)
    #: Lines whose mark stands in the margin column, which is the sheet's ruler:
    #: every sweep's first line, and the last line of all if it is not one
    #: already. That last one is there because the bottom of the paper is what
    #: the sheet is about, and a ruler that stopped one sweep short of it would
    #: leave the interesting part to be measured diagonally.
    ruler = set(range(0, args.rows, per_sweep)) | {args.rows - 1}

    enc = etp.Encoder()
    for line in range(args.rows):
        if line:
            enc.newline(1)  # the detented mechanism; the only motion on the sheet
        marks: list[tuple[int, str]] = []
        # Every fifth line, so a kink can be named rather than counted to -- and
        # every line on the ruler whether or not it falls on five, because those
        # are the marks the second reading measures between.
        if line % FEED_LABEL_EVERY == 0 or line in ruler:
            text = str(line)
            # Right-aligned to end one clear column short of the marks, for the
            # reason rewind's labels are: a digit in a different column from line
            # to line has to be read twice.
            marks.append((FEED_MARGIN_COLS - 1 - len(text), text))
        col = FEED_MARGIN_COLS + (line % per_sweep) * step
        if line in ruler and col != FEED_MARGIN_COLS:
            # The last line, carrying a ruler mark as well as its own.
            marks.append((FEED_MARGIN_COLS, bar))
        marks.append((col, bar))
        if line == 0:
            marks.append((col + FEED_BAR_CELLS + 2, title))
        type_row(enc, marks)

    enc.newline(2)  # roll the sheet clear of the platen, as planner.encode does
    enc.end()

    job = etp.Job(body=enc.body(), cols=width, rows=args.rows,
                  strikes=enc.strikes, pitch=args.pitch, home_each_row=True)
    out = args.out if os.path.isabs(args.out) else os.path.join(SRC_DIR, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    size = etp.save(out, job)

    last = args.rows - 1
    ruler = sorted(set(range(0, args.rows, per_sweep)) | {last})
    print(f"forward-feed job -> {out} ({size} bytes, {job.strikes} strikes)")
    print(f"  {args.rows} marked lines in {args.sweeps} sweeps of up to "
          f"{per_sweep}, {step} column(s) across per line down")
    print(f"  {width} columns, {args.rows} lines of paper "
          f"({args.rows * ec.LINE_HEIGHT_MM:.0f} mm)")
    print("\nLoad a sheet the way you fed it for the rewind sheet -- the depth at")
    print("issue belongs to the paper and its loading, not to the machine -- and")
    print("type this:")
    print(f"  python -m erika.send {os.path.relpath(out, SRC_DIR)} --print")
    print("\nNothing on this sheet feeds the paper backwards. It is the other half")
    print("of what `rewind` asks: that sheet found the platen exact while the")
    print("paper was held, and this one asks whether a print that only ever feeds")
    print("forward is placed truly at the same depths.")
    print("\nHow to read it:")
    print("  1. Each sweep should be one straight line of marks. A bend or a kink")
    print("     is a line the platen placed short or long, and the line number in")
    print(f"     the margin -- every {FEED_LABEL_EVERY} lines -- names where.")
    print("     Sight along it, or lay a straightedge on it.")
    print("  2. The left column is the sheet's ruler: the first mark of every")
    print("     sweep stands in it, and so does one on the very last line. Every")
    print("     measurement below runs straight down that one column, so none of")
    print("     them is diagonal. Line 0 is the topmost mark and the datum.")
    print(f"     Paper figures assume the machine's paper mark: "
          f"{ec.DEFAULT_TOP_MARGIN_MM:.0f} mm above line 0,")
    print(f"     {ec.PAPER_HEIGHT_MM:.0f} mm of paper. That is a default, not a "
          "limit -- the sheet can be")
    print("     clamped anywhere, up to the first line at its very top edge.")
    print("       line   from line 0   from the one above   paper under it")
    # The first ruler mark past where winding back stopped working, and only the
    # first: every mark below it is past it too, and a column of arrows says
    # less than one.
    crossing = next((at for at in ruler if at > ec.deepest_rewindable_line()), None)
    for i, at in enumerate(ruler):
        gap = (f"{(at - ruler[i - 1]) * ec.LINE_HEIGHT_MM:>8.1f} mm" if i
               else f"{'-':>11}")
        print(f"       {at:>4}   {at * ec.LINE_HEIGHT_MM:>8.1f} mm   {gap}"
              f"   {ec.paper_under_line_mm(at):>9.0f} mm"
              + ("   <- rewind gave up by here" if at == crossing else ""))
    print("     A feed that shortens gradually bends no sweep at all and closes")
    print("     these gaps up, so this is the reading that catches it. The last")
    print("     gap is the short one -- the bottom sweep has fewer lines in it.")
    print(f"  3. Line 0 to line {last}, down that same column, is "
          f"{last * ec.LINE_HEIGHT_MM:.1f} mm. It is the")
    print("     absolute check: the first two readings compare the sheet with")
    print("     itself, so a feed wrong by the same fraction throughout would")
    print("     pass them both, and a steel rule on this one would not.")
    print("     Do not measure between the topmost and bottommost *sweep* marks:")
    print("     they are in different columns, so a rule laid between them reads")
    print("     the diagonal and is tens of millimetres long.")
    print("\nWhat it decides:")
    print("  If every sweep is straight to where the rollers let go, the depth")
    print("  `rewind` found limits winding back and nothing else, and an ordinary")
    print("  print may use the paper down to the grip. That is how it came out")
    print("  the first time it was typed, which is why the two tails below differ.")
    print("  If instead the sweeps go wrong where `rewind` did, that depth is the")
    print("  ceiling on the height of *any* print on this paper -- and nothing in")
    print("  the pipeline would know: a taller plan verifies against the mockup,")
    print("  uploads, and types its last lines onto paper the platen is no longer")
    print("  placing.")
    print("\n  Measured once already, on A4 loaded at the paper mark:")
    print(f"    line {ec.deepest_printable_line():>2}  the rollers let go, about "
          f"{ec.FEED_TAIL_MM} mm of sheet left. The ceiling")
    print("          on an ordinary print.")
    print(f"    line {ec.deepest_rewindable_line():>2}  winding back stops "
          f"working, about {ec.REWIND_TAIL_MM} mm left. The ceiling")
    print("          on a print that has to be wound back -- more than one wheel.")
    print("  The millimetres are what carry to another paper length or loading;")
    print("  the line numbers are those figures on this one. Clamp the sheet")
    print(f"  further through and both gain: at the top edge they are "
          f"{ec.deepest_printable_line(top_margin_mm=0)} and "
          f"{ec.deepest_rewindable_line(top_margin_mm=0)}.")
    print("  See erika_codes\' \"Where the paper sits\".")
    return 0



def cmd_sheet(args) -> int:
    """Type the full charset so it can be scanned into a real charset.

    Every row carries a registration mark at each end, one blank cell clear of
    the glyphs. They are what lets `charset --from-scan` recover the cell grid
    from the scan instead of trusting the crop: two marks of the same glyph a
    known number of cells apart, so the distance between their ink centroids is
    that many cells whatever bearing the mark has. See make_charset's note on
    SHEET_MARK_CHAR for what cropping to the ink costs instead.
    """
    from erika.make_charset import (SHEET_MARK_CHAR, SHEET_MARK_GAP,
                                    sheet_mark_cells)

    glyphs = ec.all_glyphs(dead_keys=args.dead_keys)
    forces = _parse_forces_arg(args.forces)
    cols = args.sheet_cols
    mark = ec.glyph_for_char(SHEET_MARK_CHAR)
    if mark is None:
        raise PlanError(f"the Sigma has no key for the registration mark "
                        f"{SHEET_MARK_CHAR!r}")
    left_cell, first_cell, right_cell = sheet_mark_cells(cols)
    enc = etp.Encoder()

    # The marks are struck at the hardest force whatever the row is typed at. A
    # mark is a measurement, not a sample, and a light one on a light-force row
    # is a measurement that might not be found.
    hardest = forces[0] if forces else None
    current: int | None = None

    def strike_mark() -> None:
        nonlocal current
        if hardest is not None and current != hardest:
            enc.set_force(hardest)
            current = hardest
        enc.strike(mark.code, mark.advances)

    def restore_force() -> None:
        nonlocal current
        if force is not None and current != force:
            enc.set_force(force)
            current = force

    def close_row(typed: int) -> None:
        """Pad from the last glyph of a row out to the closing mark's cell.

        `typed` rather than `cols` because the final row is usually short, and
        the marks have to sit in the same column on every row or the grid they
        describe is not a grid.
        """
        enc.right(2 * (right_cell - first_cell - typed))
        strike_mark()

    # Contiguous, force block after force block, with no line break where one
    # block ends -- because make_charset lays the tiles out the same way, and it
    # has to: a blank tile between two glyphs would be dropped by chop_charset
    # and shift every index after it. See build_sheet.
    i = 0
    total = len(glyphs) * max(1, len(forces))
    for force in forces or [None]:
        if force is not None:
            enc.set_force(force)
            current = force
        for glyph in glyphs:
            if i % cols == 0:
                if i:
                    close_row(cols)
                    enc.newline(1)
                enc.carriage_return()
                enc.right(2 * left_cell)
                strike_mark()
                enc.right(2 * (first_cell - left_cell - 1))
                restore_force()
            enc.strike(glyph.code, glyph.advances)
            if not glyph.advances:
                enc.right(2)
            i += 1
    close_row(i % cols or cols)
    if forces:
        enc.set_force(forces[0])  # leave the machine as it was found
    enc.newline(3)
    enc.carriage_return()
    enc.end()

    tiles = len(glyphs) * max(1, len(forces))
    rows = (tiles + cols - 1) // cols
    job = etp.Job(body=enc.body(), cols=right_cell + 1, rows=rows,
                  strikes=enc.strikes, pitch=args.pitch, home_each_row=True)
    out = args.out if os.path.isabs(args.out) else os.path.join(SRC_DIR, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    size = etp.save(out, job)
    forces_note = (f" at {len(forces)} strike forces "
                   f"({', '.join(f'0x{f:02X}' for f in forces)})" if forces else "")
    print(f"charset sheet -> {out} ({size} bytes, {len(glyphs)} glyphs{forces_note}, "
          f"{cols} per line)")
    print(f"\nEvery row carries a registration mark ({SHEET_MARK_CHAR!r}) at each end,")
    print(f"{SHEET_MARK_GAP} blank cell clear of the glyphs. Scan the sheet square-on with")
    print("white paper visible all round -- do NOT crop to the ink. The marks are")
    print("what the grid is measured from, and cropping to the outermost ink puts")
    print("the edge inside the outer cell by one side bearing, which on this")
    print("machine is about an eighth of a cell and depends on which glyph")
    print("happens to sit in that column.")
    print("\nFirst thing to look at: whether the glyphs are clear of each other.")
    print(f"This sheet is typed for {args.pitch} characters per inch, which is a slide")
    print("switch on the machine -- and nothing in the machine's command set can be")
    print("asked where that switch is set. A sheet typed with it at the other")
    print("setting has every glyph overlapping its neighbour, and it still scans,")
    print("still slices to a grid and still builds a charset: one that describes")
    print("type nobody will print. The firmware pins the pitch from the job's own")
    print("header before it types (IMG PREPARE PITCH, which is its default); if")
    print(f"that has been turned off, set the switch to {args.pitch} by hand.")
    print("\nA small skew is corrected when the scan is read, so square-on means")
    print("close rather than perfect -- but square it up on the platen anyway.")
    print("The grid is sliced axis-aligned, and every degree left in the scan is")
    print("a tile contaminated by its neighbour, worst at the corners of the")
    print("sheet, which is the part a spot check does not look at.")
    print("\nThen build a charset from the real type:")
    print(f"  python -m erika.pipeline charset --pitch {args.pitch} "
          f"--name sigma-scanned --from-scan /path/to/scan.png"
          + (f" --forces {args.forces}" if forces else ""))
    if forces:
        print("\nThe --forces there must match, and in the same order: the scan is")
        print("sliced to a grid, and the grid is what says which tile is which.")
    return 0


def _read_force_scan(args, blocks: dict[str, list[int]], run: int) -> int:
    """Read a typed probe sheet back as a density curve.

    The sweep has to be the one the sheet was typed with -- the same
    `--from/--to/--step/--run` -- which is why this lives inside `forces` rather
    than beside it: the command that typed the sheet, with `--from-scan` added,
    is the command that reads it, and there is no second place to get the
    arguments wrong.
    """
    from erika import force_scan  # noqa: PLC0415

    try:
        readings = force_scan.read_scan(args.scan, blocks, run, args.pitch,
                                        args.deskew)
    except (ValueError, FileNotFoundError) as exc:
        raise PlanError(str(exc)) from exc

    print(f"strike-force probe <- {force_scan.relative_path(args.scan)}")
    total = sum(len(v) for v in blocks.values())
    print(f"  {total} force(s) swept, {run} glyphs per row, pitch {args.pitch}")
    print(force_scan.report(readings, args.levels))
    return 0


def cmd_forces(args) -> int:
    """Sweep the strike-force command, so the machine can say what it accepts.

    The manual gives the code and says the next character is the strength. It
    does not say which strengths exist, nor whether "character" means a small
    integer or the ASCII digit for one -- and that cannot be settled from a
    desk. So this types every candidate and lets the paper answer.

    Two properties make the sheet readable whatever the machine does:

    - The first row is typed before the command is ever sent, so it shows the
      force the machine powers up with. Every row after states its own force,
      which means it does not matter whether a setting persists.
    - Every candidate value is one the wheel could have typed. On a machine that
      does not implement the command the value byte arrives as an ordinary
      character; inside the wheel's range the worst case is a visible stray
      character, which is itself the answer -- the command did nothing. Above the
      range it would be a *motion*, shifting the rest of the line and making the
      sheet unreadable exactly where it needs reading, or worse a command that
      eats the byte after it and takes the rest of the sheet with it.

    The published control code table (erika_ai/ressources/steuercodes.md) gave a
    weak steer on which hypothesis to read first: 0xA5, 0xA6 and 0xA7 all spell
    their operand out as a raw count, and 0xA3 uses the same phrasing for the
    force. That pointed at the `raw` block, and the paper agreed -- the sheet
    types one continuous ramp across the value space rather than marking only at
    0x30..0x39. erika_codes.SET_STRIKE_FORCE carries the measured scale.

    The command survives that answer because the answer is one machine's. A
    different wheel, a different ribbon or a different Sigma moves the threshold
    and the saturation point, and both are what a charset gets built against.
    """
    from erika import force_scan  # noqa: PLC0415 -- also the scan reader

    blocks = _force_probe_blocks(args)
    run = args.run
    if args.scan:
        return _read_force_scan(args, blocks, run)

    glyph = ec.glyph_for_char(args.char)
    if glyph is None:
        raise PlanError(f"the Sigma has no key for {args.char!r}")

    enc = etp.Encoder()

    # Walked rather than written out, because force_scan reads the sheet back
    # off the same list: a row's index in it is the row's y on the paper, and
    # the reader counts rows it cannot see (a force below the ink threshold
    # prints nothing, not even its own label) rather than looking for them.
    for line in force_scan.probe_lines(blocks, run):
        if line.kind == "blank":
            enc.newline(1)
        elif line.kind in ("title", "heading"):
            type_line(enc, line.text)
        else:
            # The label is typed at the row's own force, so a row that printed
            # nothing is blank end to end. That is deliberate: it is the same
            # answer either way, and a label at full force on a row that took no
            # ink would read as a force the machine accepted and then ignored.
            if line.value is not None:
                enc.set_force(line.value)
            type_row(enc, [(0, line.text)])
            enc.right(2 * (len(line.text) + 1))
            for _ in range(line.run_cells):
                enc.strike(glyph.code, glyph.advances)
            enc.newline(1)

    enc.newline(2)
    enc.carriage_return()
    enc.end()

    rows = 4 + sum(len(v) + 2 for v in blocks.values())
    job = etp.Job(body=enc.body(), cols=40, rows=rows, strikes=enc.strikes,
                  pitch=args.pitch, home_each_row=True)
    out = args.out if os.path.isabs(args.out) else os.path.join(SRC_DIR, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    size = etp.save(out, job)

    stride = "" if args.step == 1 else f", in steps of {args.step}"
    print(f"strike-force probe -> {out} ({size} bytes, {job.strikes} strikes)")
    for name, values in blocks.items():
        print(f"  {name:<6} 0x{values[0]:02X}..0x{values[-1]:02X} "
              f"({len(values)} values{stride})")
    print("\nWhy this sheet exists: a charset with more than one strike force is")
    print("the single largest quality factor in the paper this pipeline")
    print("implements (section 5.5), and nothing in the code knows yet whether")
    print("this machine can do it or how to ask.")
    print("\nWhat to check on the printed sheet:")
    print("  - Compare each row of glyphs with the 'AS FOUND' row at the top.")
    print("    Rows that print LIGHTER or DARKER are forces the machine accepted.")
    print("  - A row that also shows a stray character before the glyphs is a")
    print("    value the machine did not take as a force: it typed it instead.")
    print("    Those rows are answers too -- they rule the value out.")
    if args.step != 1:
        print(f"  - This was a coarse pass, every {args.step}th value. A row that")
        print("    differs brackets a neighbourhood rather than naming a value:")
        print("    sweep it again with --step 1 to find where it starts and stops.")
    print("  - If no row differs from 'AS FOUND' anywhere on the sheet, then this")
    print(f"    machine either does not honour 0x{ec.SET_STRIKE_FORCE:02X} or wants "
          "the force spelled")
    print("    some other way. Try another range with --from/--to, and --step to")
    print("    keep a wide one down to a sheet of paper.")
    print("\nFor reference, what this sheet said on a Sigma SM 8200i with a")
    print("Courier 10 wheel -- the machine the pipeline was written for:")
    print("  0          solid; full strike")
    print("  1..39      no ink at all")
    print("  40         first ink, isolated dots, not a character")
    print("  43         first legible character -- the practical floor")
    print("  55         fully formed characters")
    print("  95..103    saturated, indistinguishable from each other")
    print("So: a raw count, harder for larger, and a usable ladder of 43..95")
    print("that is compressed at the bottom. Your wheel and ribbon will differ;")
    print("that is what this sheet is for.")
    print("\nThen, with the values that worked, hardest first:")
    print("  python -m erika.pipeline sheet --forces 0,60,50,43      # type it")
    print("  python -m erika.pipeline charset --forces 0,60,50,43 \\")
    print("      --name sigma-forces --from-scan /path/to/scan.png")
    print("\nPick those by how the ink looks on this sheet, not by even")
    print("arithmetic: the values are a lever position, and the ramp between the")
    print("floor and saturation is not linear in them. Or scan the sheet and let")
    print("it say so itself -- every row is the same glyph struck the same number")
    print("of times, so the ink per row is the curve:")
    print("  python -m erika.pipeline forces --from-scan /path/to/scan.png \\")
    print("      --levels 4        # plus the same sweep this sheet was typed with")
    print("\nOne piece of advice from the paper that does NOT transfer: its")
    print("figure 20 found medium plus light beat dark plus medium, because its")
    print("typewriter could already reach black. This one cannot, so keep the")
    print("hardest strike and add lighter ones below it.")
    return 0


def melody_defaults():
    """Defaults for the melody parser, without importing melody at module load.

    `erika.melody` imports `erika_codes` and `etp` and nothing heavier, but the
    argument parser is built on every invocation of every subcommand and this
    keeps that honest.
    """
    from erika import melody as mel

    return mel.DEFAULT_GATE, mel.DEFAULT_MORSE_UNIT_MS


def cmd_melody(args) -> int:
    """Put a rhythm on the beeper.

    The machine's only output that costs neither paper nor ribbon, and the only
    one that can be tried at three in the morning. It is also the only place in
    this pipeline where nothing can be verified against a mockup afterwards --
    a melody leaves no trace -- so the command prints the score it is about to
    play, which is the closest thing there is to reading it back.
    """
    from erika import melody as mel

    if args.list:
        print("tunes:")
        for name in sorted(mel.TUNES):
            notation, tempo = mel.TUNES[name]
            print(f"  {name:10s} {tempo:5.0f} BPM  {notation}")
        print()
        print("Anything else is written with --notes, in the same notation:")
        print("  w h q e s  whole/half/quarter/eighth/sixteenth")
        print("  q.         dotted; -q a rest; 350ms an exact slot; | a bar line")
        return 0

    if args.probe:
        job = mel.probe_job()
        melody = None
    else:
        # Everything from reading the notation to fitting it into the device's
        # timing raises the same error, and all of it is the user's typing
        # rather than a defect -- so it reports as a message and an exit code,
        # not as a traceback.
        try:
            if args.morse:
                melody = mel.morse(args.morse, unit_ms=args.unit)
            elif args.notes:
                melody = mel.parse(
                    args.notes,
                    tempo=mel.DEFAULT_TEMPO if args.tempo_given is None
                    else args.tempo_given,
                    gate=args.gate, name="notes")
            else:
                melody = mel.tune(args.tune, tempo=args.tempo_given,
                                  gate=args.gate)
            job = mel.to_job(melody)
        except mel.MelodyError as exc:
            print(f"cannot play that: {exc}")
            return 2

    out = args.out if os.path.isabs(args.out) else os.path.join(SRC_DIR, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    size = etp.save(out, job)

    if melody is None:
        print(f"bell length sweep -> {out} ({size} bytes)")
        print(f"  {len(mel.PROBE_UNITS)} beeps: "
              f"{', '.join(str(n) for n in mel.PROBE_UNITS)} units")
        print()
        print("Listen, do not look -- nothing is typed. Twelve beeps, each")
        print("longer than the last, then silence.")
        print()
        print("This sweep has been run, and it is what the two constants come")
        print("from: time a long beep and divide by its unit count, and the")
        print(f"answer was {ec.BELL_UNIT_MS} ms rather than the table's 20 -- "
              "the Sigma's")
        print("beeper is about twice as quick as the S3004's is written down as.")
        print("Every length past 127 came out longer than the one before it, so")
        print("the operand is a plain unsigned byte and not a signed one; the")
        print(f"longest beep the machine has is {ec.MAX_BELL_UNITS} units, "
              f"{ec.MAX_BELL_UNITS * ec.BELL_UNIT_MS / 1000:.1f} s.")
        print()
        print("Worth re-running on a machine that has been serviced, on another")
        print("Sigma, or after the firmware's pacing changes -- it is the only")
        print("thing that would notice. A gap between two early beeps should")
        print("come out near 1.6 s; that is the host's timing model, not the")
        print("bell, and it is the half that would drift.")
    else:
        print(mel.score(melody))
        print()
        print(f"-> {out} ({size} bytes, nothing typed)")
        print(f"  {mel.RAW_BYTE_COST_MS} ms of that is the firmware's own pacing "
              f"per bell byte,")
        print(f"  which is why a note cannot be shorter than {mel.MIN_SLOT_MS} ms: "
              f"{mel.max_tempo_for('q'):.0f} BPM in quarters,")
        print(f"  {mel.max_tempo_for('e'):.0f} in eighths. A beep is "
              f"{ec.BELL_UNIT_MS} ms per unit, measured, and at most")
        print(f"  {ec.MAX_BELL_UNITS} units -- `melody --probe` is the sweep "
              "that timed it.")
    return 0


def cmd_codes(args) -> int:
    """Put the control codes the pipeline does not use yet on paper.

    The interface answers to about sixty control codes and this pipeline sends
    eleven. The rest are published -- see erika_ai/ressources/steuercodes.md and
    docs/control-codes.md in the workspace -- but published for an Erika S3004,
    and this is a Sigma SM 8200i. They share an interface, which is a claim
    about a family of machines and not a measurement of this one.

    So: send them and look at the paper. Every section has a reference typed
    beside it with codes that are already known to work, because "did it move a
    twelfth of an inch" is not a question anyone can answer by looking at a mark
    on its own.

    Two properties keep the sheet readable when a code does nothing, or does
    something else:

    - Every row begins with a carriage return, so a section that displaces the
      head cannot carry the error into the next one.
    - The sections are ordered by how much they cost if they misbehave. The bell
      first, because it says whether operand-carrying commands are understood at
      all without marking the paper; pitch last, because it is the one setting
      that would silently rescale everything after it.

    One question this sheet cannot answer is 0x96, the completion report. Its
    effect is on *timing* -- RTS held until the character is actually printed --
    and timing lives in the firmware's delay table, not in the opcode stream.
    """
    per_half = ec.carriage_steps_per_half_step(args.pitch)
    cell = 2 * per_half  # 1/120" steps to one character cell
    n = 20

    enc = etp.Encoder()
    type_line(enc, "CONTROL CODE PROBE")
    enc.newline(1)

    # 1. The bell. No ink, and it is the cheapest possible answer to "does this
    #    machine take a command with an operand at all" -- which every section
    #    below depends on.
    type_line(enc, "1 BELL 0xAA (LISTEN -- NOTHING IS TYPED)")
    enc.raw_command(0xAA, 10)  # ~200 ms
    enc.delay_ms(500)
    enc.newline(1)

    # 2. Carriage steps against the escapement. Two combs, one cell apart in
    #    the same columns: if 0xA5 moves 1/120" the lower bars sit under the
    #    upper ones.
    type_line(enc, f"2 CARRIAGE STEPS 0xA5 ({cell} STEPS = ONE CELL AT PITCH {args.pitch})")
    for _ in range(n):
        enc.strike(ec.glyph_for_char(RULER_CHAR).code)
        enc.right(2)
    enc.newline(1)
    for _ in range(n):
        enc.strike(ec.glyph_for_char(RULER_CHAR).code)
        enc.carriage_steps(cell)
    enc.newline(2)

    # 3. One long move against ten short ones. This is the case 0xA5 is worth
    #    having for -- ten SPACE bytes and ten escapement steps become one
    #    command -- and it is also where the operand's range gets exercised.
    type_line(enc, "3 ONE LONG STEP AGAINST TEN SPACES")
    enc.strike(ec.glyph_for_char(RULER_CHAR).code)
    enc.right(2 * 9)  # nine more cells, the strike having taken one
    enc.strike(ec.glyph_for_char(RULER_CHAR).code)
    enc.newline(1)
    enc.strike(ec.glyph_for_char(RULER_CHAR).code)
    enc.carriage_steps(9 * cell)
    enc.delay_ms(400)  # an inch of carriage travel, and the firmware cannot know
    enc.strike(ec.glyph_for_char(RULER_CHAR).code)
    enc.newline(2)

    # 4. Platen steps against the line-feed mechanism. Four gaps each way, in
    #    one column, so a feed that disturbs the carriage shows as a sideways
    #    shift the way it does on the calibration sheet.
    ladder = 18
    full_line = 2 * ec.PLATEN_STEPS_PER_HALF_LINE
    type_line(enc, f"4 PLATEN STEPS 0xA6 ({full_line} STEPS = ONE LINE)")
    for _ in range(4):
        enc.carriage_return()
        enc.right(2 * ladder)
        enc.strike(ec.glyph_for_char("_").code)
        enc.newline(1)  # the reference: the detented line-feed mechanism
    for _ in range(4):
        enc.carriage_return()
        enc.right(2 * ladder)
        enc.strike(ec.glyph_for_char("_").code)
        enc.platen_steps(full_line)  # the same distance, by motor steps
    enc.newline(2)

    # 5. The forbidden counts. Eight feeds of five steps come to one line, and
    #    five is a count the table says the mechanism refuses -- so the encoder
    #    splits it, and this is whether the split lands where the arithmetic
    #    says it does.
    #
    #    Nine marks and eight feeds, so the first and last are exactly a line
    #    apart -- eight marks would span seven gaps and land seven eighths of the
    #    way down, which is not a distance anyone can check by eye.
    fine = full_line // 8
    type_line(enc, f"5 FINE FEED (EIGHT FEEDS OF {fine} STEPS = ONE LINE)")
    for i in range(9):
        enc.carriage_return()
        enc.right(2 * ladder)
        enc.strike(ec.glyph_for_char("-").code)
        if i < 8:
            enc.platen_steps(fine)
    enc.newline(2)

    # 6. Doppeldruck, in exactly the order a plan would use it: the code, then
    #    every glyph in the stack but the last, then the last one to advance.
    type_line(enc, "6 NO ADVANCE 0xA9 (O SHOULD BE STRUCK THROUGH)")
    for _ in range(8):
        enc.raw(0xA9)
        enc.strike(ec.glyph_for_char("-").code)
        enc.strike(ec.glyph_for_char("O").code)
        enc.right(2)
    enc.newline(2)

    # 7. Backward printing. Five letters from a known column: forwards they run
    #    away from it, backwards they run into it.
    start = 20
    type_line(enc, "7 BACKWARD PRINT 0x8E (ABCDE FROM COLUMN 20)")
    enc.carriage_return()
    enc.right(2 * start)
    enc.raw(0x8E)
    type_text(enc, "ABCDE")
    enc.raw(0x8D)
    enc.newline(2)

    # 8. The correction ribbon. Both halves on one line and one of them left
    #    alone, because what 0x8C does depends on the tape fitted and the
    #    difference between "erased" and "never typed" is the reference.
    type_line(enc, "8 CORRECTION 0x8C (RIGHT GROUP RETYPED THROUGH IT)")
    type_row(enc, [(0, "MMMMM"), (10, "MMMMM")])
    enc.carriage_return()
    enc.right(2 * 10)
    enc.raw(0x8C)
    type_text(enc, "MMMMM")
    enc.raw(0x8B)
    enc.newline(2)

    # 9. Pitch. Second to last, because a pitch that sticks rescales everything
    #    after it -- and section 10 being after it is what checks the restore,
    #    which nothing did while this was the final section.
    pitch15 = 0x89
    restore = 0x87 if args.pitch == 10 else 0x88
    type_line(enc, "9 PITCH 15 0x89 (LOWER COMB SHOULD BE NARROWER)")
    for _ in range(15):
        enc.strike(ec.glyph_for_char(RULER_CHAR).code)
        enc.right(2)
    enc.newline(1)
    enc.raw(pitch15)
    for _ in range(15):
        enc.strike(ec.glyph_for_char(RULER_CHAR).code)
        enc.right(2)
    enc.raw(restore)
    enc.newline(2)

    # 10. Whether the *type* fits the pitch, which section 9 cannot say.
    #
    #     Section 9's comb is typed with RULER_CHAR, and RULER_CHAR was chosen
    #     for being a narrow vertical mark -- which makes it exactly the one
    #     character whose spacing can shrink without the glyphs colliding. It
    #     confirms the escapement and is blind to the wheel.
    #
    #     The slugs on a daisy wheel are a fixed physical size. If the wheel is
    #     cut for pitch 10, a cell at pitch 15 is two thirds as wide as the
    #     character standing in it, so neighbours overlap by about a third. That
    #     is ruinous for text and might be useful for a picture -- more ink per
    #     unit area, on a machine that cannot reach black -- but either way it is
    #     a different charset, not a wider one, and nothing should be built on
    #     pitch 15 before the paper has said which.
    #
    #     Three rows: the same groups at the working pitch and at 15, and then
    #     the wide glyph spaced out so that each one has a whole cell to itself.
    #     Between them they separate "the type is narrower at this pitch" from
    #     "the type is the same and now collides".
    wide, thin = "M", RULER_CHAR
    run = 15
    type_line(enc, "10 TYPE AT PITCH 15 (DOES THE WHEEL FIT THE STEP?)")

    def group(char: str, count: int, gap: int = 0) -> None:
        for _ in range(count):
            enc.strike(ec.glyph_for_char(char).code)
            if gap:
                enc.right(2 * gap)

    enc.carriage_return()
    group(wide, run)
    enc.right(2 * 3)
    group(thin, run)
    enc.newline(1)

    enc.raw(pitch15)
    enc.carriage_return()
    group(wide, run)
    enc.right(2 * 3)
    group(thin, run)
    enc.newline(1)

    enc.carriage_return()
    group(wide, run, gap=1)  # one clear cell between each
    enc.raw(restore)
    enc.newline(2)

    enc.carriage_return()
    enc.end()

    job = etp.Job(body=enc.body(), cols=2 * run + 3, rows=40, strikes=enc.strikes,
                  pitch=args.pitch, home_each_row=True)
    out = args.out if os.path.isabs(args.out) else os.path.join(SRC_DIR, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    size = etp.save(out, job)
    print(f"control code probe -> {out} ({size} bytes, {job.strikes} strikes)")
    print(f"  pitch {args.pitch}: one cell is {cell} carriage steps, "
          f"one line is {full_line} platen steps")
    print()
    print("None of these codes has been on paper on this machine. They are")
    print("published for the Erika S3004 (erika_ai/ressources/steuercodes.md);")
    print("the Sigma shares that interface, which is a claim about a family of")
    print("machines rather than a measurement of this one. This sheet is the")
    print("measurement.")
    print()
    print("What to check on the printed sheet:")
    print("  1  A beep, roughly a fifth of a second, and nothing typed. That is")
    print("     one command plus an operand byte understood as a pair. If the")
    print("     machine types a character instead, it did not understand the")
    print("     command and took the operand as text -- and 0xA5 and 0xA6 below")
    print("     will not work either. If nothing at all happens, the command was")
    print("     swallowed silently, which is the harder case: read part 2.")
    print("  2  Two combs of bars, the lower under the upper. The upper is typed")
    print("     with the escapement, the lower by asking for the same distance")
    print(f"     in {cell} carriage steps. Bars that line up mean 0xA5 moves")
    print("     1/120 inch and this machine can be driven in absolute units")
    print("     rather than in fractions of whatever the pitch switch is set to.")
    print("     A lower comb that is there but at the wrong spacing means the")
    print("     step is a different size -- measure it and say so. A lower row")
    print("     that is a single bar means the moves did nothing.")
    print("  3  Two bars, one cell and nine apart, then the same pair produced")
    print("     by a single command. They must coincide. This is where the code")
    print("     earns its place: ten bytes and ten escapement steps become two.")
    print("  4  Eight marks in ONE column, so seven gaps: the first FOUR are")
    print("     driven by the detented line-feed mechanism and are the")
    print(f"     reference, the last THREE are asked for as {full_line} platen")
    print("     steps and are the thing under test. All seven must be equal.")
    print("     Which gap is odd decides what it means: one of the last three")
    print("     and 0xA6 does not move 1/240 inch; one of the first four and the")
    print("     platen or the paper is at fault and this sheet is only the")
    print("     messenger -- `calibrate` part 3 tests the same mechanism and")
    print("     would show it too. Any sideways shift means a motor-step feed")
    print("     nudges the carriage, which the detented feed does not.")
    print(f"  5  Nine marks {fine} steps apart, so the first and last are exactly")
    print("     one line apart and the seven between are evenly spaced within it.")
    print(f"     {fine} is one of the counts the table says the mechanism refuses,")
    print("     so this also checks that splitting it into 2 + 2 + 1 comes to the")
    print("     same distance. Uneven gaps mean the split does not, and a first")
    print("     and last more or less than a line apart means the step is not")
    print("     1/240 inch.")
    print("  6  Eight O struck through, with no backspace anywhere in the plan.")
    print("     That is 0xA9 printing a character where the head stands. It")
    print("     matters because overstriking is how layers stack, and doing it")
    print("     with a backspace spends the escapement's repeatability on every")
    print("     stacked character. Plain O with the hyphens beside them means")
    print("     0xA9 did nothing; hyphens one cell left means it did something")
    print("     else.")
    print("  7  Five letters, from column 20. Reading ABCDE to the right of")
    print("     column 20 means backward print did nothing. Reading EDCBA and")
    print("     *ending* at column 20 means it works -- and a serpentine pass")
    print("     can then cost one byte per cell instead of three.")
    print("  8  Two groups of five M. The left group is the reference and is")
    print("     never touched. The right group was typed, then typed again with")
    print("     the correction ribbon selected. Blank means a lift-off tape and")
    print("     an erase; white or pale means a cover-up tape, which is white")
    print("     ink and a thing this machine could not do before; unchanged")
    print("     means 0x8C did nothing or there is no correction tape fitted.")
    print("  9  Two combs of fifteen bars. If the lower is about two-thirds the")
    print("     width of the upper, this machine takes 15 characters per inch --")
    print("     which the slide switch does not offer, and which is about 97")
    print("     escapement positions instead of 65. Same width means 0x89 did")
    print("     nothing. What this cannot tell you is whether the *type* fits")
    print("     that step, which is what part 10 is for -- the bar is a narrow")
    print("     mark on purpose, and a narrow mark is exactly the one whose")
    print("     spacing can shrink without the glyphs touching.")
    print(f"  10 Three rows. The first is {run} {wide} then {run} {thin} at")
    print(f"     pitch {args.pitch}, and it is the reference: look at how much")
    print(f"     clear paper stands between neighbouring {wide}. The second row")
    print("     is the same two groups at pitch 15. The third is at pitch 15 as")
    print(f"     well, with a whole empty cell between each {wide}.")
    print()
    print("     The wheel's slugs are a fixed physical size. If the wheel is cut")
    print(f"     for pitch {args.pitch}, the step at 15 is two thirds of the one")
    print(f"     the {wide} was drawn for, so neighbours must overlap by about a")
    print("     third. Four things the rows can say:")
    print()
    print(f"     - Row 2's {wide} merge into a band and row 3's are clean and")
    print("       separate. The escapement shrank and the type did not. Pitch 15")
    print("       is then not 97 columns of the same picture but a different")
    print("       charset -- one whose glyphs bleed into their neighbours' cells,")
    print("       which make_charset cannot model today. Still possibly worth")
    print("       having: overlap is ink, and this machine cannot reach black.")
    print()
    print("     - Row 2 looks like row 1, only narrower. The wheel really is")
    print("       narrower at this pitch, and pitch 15 is a clean half again as")
    print("       much horizontal resolution -- rebuild the charset at the new")
    print("       cell and nothing else has to change.")
    print()
    print(f"     - Row 2's {thin} merge as well. Something other than the type is")
    print("       wrong: the thin mark is the control here and should survive")
    print("       any pitch the machine offers.")
    print()
    print(f"     - Row 1 is already narrow. The restore (0x{restore:02X}) at the end of")
    print("       part 9 did not take and everything here is at pitch 15. Set the")
    print("       pitch switch by hand, or send 0x95 to reset the machine.")
    print()
    print("0x96, the completion report, is not on this sheet: it changes when")
    print("RTS is released rather than what is typed, so it is answered by the")
    print("firmware's pacing and not by ink. See docs/control-codes.md.")
    return 0

def _parse_forces_arg(text: str | None) -> list[int]:
    from erika.make_charset import parse_forces

    forces = list(parse_forces(text))
    bad = [f for f in forces if not ec.is_usable_force(f)]
    if bad:
        raise PlanError(
            f"strike force(s) {', '.join(f'0x{f:02X}' for f in bad)} are "
            "commands -- a machine that ignores the force command would type "
            "them, and they would move the head or swallow the byte after "
            f"them. Pick values of 0x{ec.MAX_FORCE:02X} or below."
        )
    if len(set(forces)) != len(forces):
        raise PlanError(f"duplicate strike force in {text!r}")
    return forces


def _force_probe_blocks(args) -> dict[str, list[int]]:
    """What the probe sweeps: either an explicit range, or both hypotheses.

    ``--step`` strides the *value space* and the unusable values are dropped
    after, not before. That is the order that makes a coarse pass mean what it says: every
    Nth candidate byte, at even intervals, with a gap where the sweep happens to
    land on a code that would move the head. Filtering first would renumber the
    values and hand back N usable ones at uneven spacing, which is unreadable as a
    sweep and is the opposite of what a step is for.

    A stride from ``first``, and deliberately not "and also the last": rows at even
    intervals are what makes a change in the ink obvious down the sheet, and every
    row labels the value it was typed at, so nothing about the coverage is a guess.
    """
    step = args.step
    if step < 1:
        raise PlanError(f"--step {step} makes no sense; it is a stride, so 1 or more")

    if args.first is not None or args.last is not None:
        first = args.first if args.first is not None else 0
        last = args.last if args.last is not None else first
        if last < first:
            raise PlanError(f"--from 0x{first:02X} is above --to 0x{last:02X}")
        asked = range(first, last + 1, step)
        values = [v for v in asked if ec.is_usable_force(v)]
        skipped = len(asked) - len(values)
        if not values:
            raise PlanError(
                f"every value a sweep of 0x{first:02X}..0x{last:02X} in steps of "
                f"{step} lands on is a command rather than a force the machine "
                f"could type -- those stop at 0x{ec.MAX_FORCE:02X}"
            )
        if skipped:
            print(
                f"note: skipping {skipped} value(s) above 0x{ec.MAX_FORCE:02X} in "
                "the requested range; a machine that ignores the force command "
                "would type them as motions or commands"
            )
        return {"custom": values}

    blocks = {
        name: [v for v in range(lo, hi + 1, step) if ec.is_usable_force(v)]
        for name, (lo, hi) in ec.FORCE_PROBE_BLOCKS.items()
    }
    # A block whose every stepped-to value is unusable contributes nothing but
    # a heading. None of the blocks in erika_codes can do that -- the stride always
    # includes the first value and both begin on a usable one -- but a block added
    # later could, and a heading with no rows under it reads as a failed sweep.
    kept = {name: values for name, values in blocks.items() if values}
    if not kept:
        raise PlanError(
            f"a sweep in steps of {step} lands on no usable force in any probe block"
        )
    return kept


# ---------------------------------------------------------------------------


def _add_charset_arg(p, default="sigma-10"):
    p.add_argument("--charset", "-c", default=default,
                   help=f"charset folder under src/charsets (default {default})")


def _add_ink_args(p):
    """The ink model and strike forces, from make_charset's own parser.

    Imported rather than restated: unlike the --grey defaults, which are spelled
    twice on purpose because importing the converter would drag numba into every
    `--help`, make_charset costs nothing to import here.
    """
    from erika.make_charset import add_ink_args

    add_ink_args(p)


def _add_grey_args(p):
    """How the photograph is turned into black and white before optimizing.

    Defaults to `luma`, which is what the optimizer would have done on its own,
    so a run without these flags is the run it always was.
    """
    p.add_argument("--grey", "-g", default="luma", choices=GREY_METHODS,
                   help="colour to grey conversion: luma (Rec. 601, the default), "
                        "average (flat RGB mean), c2g (STRESS decolorization -- keeps "
                        "differences a formula merges), stress (STRESS local "
                        "enhancement, then luma). See erika/stress.py")
    p.add_argument("--stress-radius", type=float, default=stress_defaults("radius"),
                   help="spray radius as a fraction of the longest side (default 1.0)")
    p.add_argument("--stress-samples", type=int, default=stress_defaults("samples"),
                   help="sample points per iteration (default 5)")
    p.add_argument("--stress-iterations", type=int, default=stress_defaults("iterations"),
                   help="iterations per pixel; more is less noisy and slower (default 20)")
    p.add_argument("--stress-seed", type=int, default=stress_defaults("seed"),
                   help="fixes the spray, so a conversion can be repeated")
    p.add_argument("--stress-shadows", action="store_true",
                   help="--grey stress only: normalise against the lower envelope too, "
                        "which opens up the shadows at the cost of looking synthetic")


#: The conversions `--grey` accepts and the STRESS defaults, spelled here rather
#: than imported from `erika.stress`.
#:
#: Deliberate, and the one duplication in this package that is not a mistake:
#: building the parser is what every subcommand and every `--help` does, and
#: what erika-studio does to read these values back out for its own form -- and
#: importing `erika.stress` for it would drag numba, two seconds of import, into
#: a `calibrate` run that never touches a photograph. `test_stress.py` compares
#: both halves against the converter's own.
GREY_METHODS = ("luma", "average", "c2g", "stress")
_STRESS_DEFAULTS = {"radius": 1.0, "samples": 5, "iterations": 20, "seed": 1}


def stress_defaults(name: str):
    return _STRESS_DEFAULTS[name]


def _add_plan_args(p):
    p.add_argument("--out", "-o", default="results/photo.etp", help="output .etp path")
    p.add_argument("--no-home", action="store_true",
                   help="do not carriage-return between passes (faster, less accurate)")
    p.add_argument("--no-serpentine", action="store_true",
                   help="always sweep left to right (only affects --no-home)")
    p.add_argument("--indent", type=int, default=0, metavar="COLUMNS",
                   help="blank columns left of the print, to move it right on "
                        "the sheet. The paper cannot always be moved instead -- "
                        "it is fed against the guides -- and the carriage can")
    p.add_argument("--offset", type=int, default=0, metavar="LINES",
                   help="blank lines above the print, to move it down the sheet. "
                        "Down only: where the first line lands was decided when "
                        "the sheet was clamped and the platen cannot wind back "
                        "above it, so to move a print *up*, load the paper "
                        "higher. Nothing refuses an offset that runs off the "
                        "paper -- the machine has no vertical limit, only the "
                        "sheet does; `pipeline feed` is what measures yours")
    p.add_argument("--settle-ms", type=int, default=0,
                   help="pause after each paper feed, in ms")
    p.add_argument("--cr-delay-ms", type=int, default=0,
                   help="pause after each carriage return, in ms")
    p.add_argument("--jitter", type=float, default=0.05,
                   help="registration error for the shaky preview, in cells (default 0.05)")
    # Both of these are on. They name the mechanism to *stop* using, because the
    # control-code sheet came back positive on 0xA5, 0xA6 and 0xA9 and there is
    # no longer a reason to ask for them by name -- only a reason to be able to
    # go back to the keystrokes when a sheet comes out wrong and the question is
    # which mechanism did it.
    p.add_argument("--keystrokes-only", dest="fine", action="store_false",
                   help="refuse a layer offset the keyboard cannot reach, rather "
                        "than placing it with the machine's own motor steps "
                        "(1/120 inch across, 1/240 down). Those steps are what "
                        "make 16x1 and daisy_full typeable, so this rules them "
                        "out; it is here for isolating a fault to a mechanism")
    p.add_argument("--backspace-overstrike", dest="no_advance",
                   action="store_false",
                   help="type a stack of glyphs in one cell by striking and "
                        "backspacing, as before 0xA9 was confirmed, instead of "
                        "with Doppeldruck. Spends the escapement's repeatability "
                        "on every stacked character; here for the same reason")
    p.add_argument("--backspace-sweep", dest="backward", action="store_false",
                   help="type a serpentine's reverse passes by striking and "
                        "backspacing, as before 0x8E was confirmed, instead of "
                        "with Rückwärtsdruck. Three bytes a cell instead of one, "
                        "and it does nothing at all without --no-home, which is "
                        "what makes a pass run right to left in the first place")
    p.add_argument("--no-preview", action="store_true")
    p.add_argument("--ops-per-second", type=float, default=10.0,
                   help="head operations per second, for the time estimate")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m erika.pipeline",
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("charset", help="build the Sigma charset")
    c.add_argument("--name", "-N", default=None)
    c.add_argument("--pitch", "-p", type=int, default=10, choices=(10, 12))
    c.add_argument("--cell-height", type=int, default=40)
    c.add_argument("--font", "-f", default=None)
    c.add_argument("--bleed", "-b", type=float, default=0.2)
    c.add_argument("--dead-keys", action="store_true")
    # On, like the mechanisms above: a scan that needs it and does not get it
    # produces a charset that builds, verifies and prints with every tone
    # slightly wrong. Named for turning off so a suspect straightening can be
    # ruled out when a charset comes back odd -- not because it is optional.
    c.add_argument("--no-deskew", dest="deskew", action="store_false",
                   help="use the scan exactly as it was given, rather than "
                        "squaring it up first. The grid is sliced axis-aligned, "
                        "so a scan a degree off contaminates every tile with its "
                        "neighbour, worst at the corners of the sheet")
    c.add_argument("--from-scan", default=None)
    # The grid the sheet was typed at, which is what identifies a tile when the
    # scan is sliced back up. It has to be given here because `sheet` takes it
    # too and the two are one number: a sheet typed at anything but the default
    # and rebuilt without saying so is sliced on the wrong grid, and every tile
    # is then a blend of two glyphs -- which nothing downstream can detect,
    # because a blend is still a plausible tile.
    c.add_argument("--sheet-cols", type=int, default=20,
                   help="glyphs per line the sheet was typed at; pass the same "
                        "value `sheet --sheet-cols` was given")
    _add_ink_args(c)
    c.set_defaults(func=cmd_charset)

    r = sub.add_parser("print", help="image -> optimize -> .etp")
    _add_charset_arg(r)
    r.add_argument("--target", "-t", default="images/mwdog_crop.png",
                   help="input photograph, relative to src/")
    r.add_argument("--row_length", "-r", type=int, default=40,
                   help="characters per row; sets the printed size (default 40)")
    r.add_argument("--num_loops", "-n", type=int, default=15)
    r.add_argument("--layers", "-l", default="4x1",
                   help=f"layer scheme. Typeable by keystroke alone are "
                        f"{', '.join(TYPEABLE_LAYER_SCHEMES)}; the motor steps "
                        f"reach finer ones, which depends on the charset's pitch "
                        f"(--keystrokes-only rules those out)")
    r.add_argument("--init_mode", "-i", default="random")
    r.add_argument("--asymmetry", "-a", type=float, default=0.1)
    r.add_argument("--search", "-s", default="simAnneal")
    r.add_argument("--init_temp", "-temp", type=float, default=0.001)
    r.add_argument("--shuffle", "-sh", type=bool, default=True)
    r.add_argument("--match-blur", type=float, default=0.0,
                   help="0..1: how much of the score comes from local average "
                        "tone rather than per-pixel error (default 0, off). "
                        "Lets the optimizer halftone, which is where the "
                        "highlights on this machine come from -- at the cost of "
                        "pixel-level accuracy. See erika/softmatch.py")
    r.add_argument("--match-block", type=int, default=None,
                   help="block side in cell pixels for --match-blur "
                        "(default 8; must divide the cell)")
    r.add_argument("--align", action="store_true",
                   help="search for the crop that best lines the picture up with "
                        "the character grid before optimizing (the paper's "
                        "section 3.4). Buys shape matching, not tone, and costs a "
                        "visible fraction of a run -- see erika/align.py")
    r.add_argument("--align-steps", type=int, default=None,
                   help="subdivisions per axis for --align (default 4, the "
                        "paper's quarters, which is 64 crops). Below 3 buys "
                        "little: a half-cell shift is one the layer scheme "
                        "already provides")
    r.add_argument("--align-loops", type=int, default=None,
                   help="greedy cycles per candidate crop (default 2). One is the "
                        "paper's and is too noisy here to rank crops reliably")
    _add_grey_args(r)
    _add_plan_args(r)
    r.set_defaults(func=cmd_print)

    pl = sub.add_parser("plan", help="re-plan an existing choices.json")
    _add_charset_arg(pl)
    pl.add_argument("--choices", default="results/choices.json")
    _add_plan_args(pl)
    pl.set_defaults(func=cmd_plan)

    v = sub.add_parser("verify", help="diff a plan render against the mockup")
    _add_charset_arg(v)
    v.add_argument("--choices", default="results/choices.json")
    v.add_argument("--reference", default="results/final.png")
    v.set_defaults(func=cmd_verify)

    cal = sub.add_parser("calibrate", help="motion-code test pattern")
    cal.add_argument("--out", "-o", default="results/calibrate.etp")
    cal.add_argument("--pitch", "-p", type=int, default=10, choices=(10, 12))
    cal.set_defaults(func=cmd_calibrate)

    ar = sub.add_parser("area", help="mark the four corners of the printable area")
    ar.add_argument("--out", "-o", default="results/print_area.etp")
    ar.add_argument("--pitch", "-p", type=int, default=10, choices=(10, 12))
    ar.add_argument("--rows", type=int, default=DEFAULT_PAPER_ROWS,
                    help=f"lines to mark out; the machine has no vertical limit, "
                         f"only the paper does (default {DEFAULT_PAPER_ROWS}, "
                         f"every line that lands on A4 below the "
                         f"{ec.DEFAULT_TOP_MARGIN_MM:.0f} mm the paper mark "
                         f"leaves -- clamp the sheet higher and more fit). The "
                         f"rollers let go around line "
                         f"{ec.deepest_printable_line()}, and a sheet that has "
                         f"to be wound back is done at line "
                         f"{ec.deepest_rewindable_line()} -- see `feed` and "
                         "`rewind`")
    ar.add_argument("--columns", type=int, default=None,
                    help="width to mark out; default is the carriage limit "
                         f"({ec.MAX_COLUMNS[10]} at pitch 10, "
                         f"{ec.MAX_COLUMNS[12]} at pitch 12). Set it to an -r "
                         "value to see where that print would land")
    ar.set_defaults(func=cmd_area)

    rw = sub.add_parser("rewind", help="measure what a reverse paper feed costs "
                                       "in registration")
    rw.add_argument("--out", "-o", default="results/rewind.etp")
    rw.add_argument("--pitch", "-p", type=int, default=10, choices=(10, 12))
    rw.add_argument("--lines", default=None,
                    help="rewind distances to try, in whole lines, comma "
                         "separated (default "
                         f"{','.join(str(d) for d in REWIND_DISTANCES)}). Several "
                         "because backlash reads the same at every distance and "
                         "slip grows with it")
    rw.add_argument("--repeat", type=int, default=REWIND_REPEATS,
                    help=f"runs of each distance (default {REWIND_REPEATS}). An "
                         "error that is not repeatable cannot be planned around, "
                         "so one run of each answers nothing")
    rw.add_argument("--half-range", type=int, default=REWIND_HALF_RANGE,
                    help=f"blocks of ladder either side of zero (default "
                         f"{REWIND_HALF_RANGE}). The carriage caps it: "
                         f"{((ec.MAX_COLUMNS[10] - REWIND_LEFT_INSET) // REWIND_BLOCK_COLS - 1) // 2} "
                         "at pitch 10")
    rw.add_argument("--ladder-step", type=int, default=1,
                    help="platen steps per block (default 1 = 1/240 in, a "
                         f"reach of +-{REWIND_HALF_RANGE / ec.PLATEN_STEPS_PER_INCH * 25.4:.2f}"
                         " mm). The carriage caps how many blocks there are, so "
                         "this is the only way to reach further: --ladder-step "
                         "2 doubles the range and halves the resolution. Run it "
                         "coarse first if nothing nulls")
    rw.add_argument("--mechanism", default="half-line", choices=("half-line", "fine"),
                    help="what winds the paper back: half-line keystrokes, or "
                         "the platen's own 1/240 inch steps via 0xA6. They are "
                         "different mechanisms and need not have the same "
                         "backlash")
    rw.add_argument("--rows", type=int, default=DEFAULT_PAPER_ROWS,
                    help=f"lines of paper available (default {DEFAULT_PAPER_ROWS}; "
                         "`area` is what measures this for your paper)")
    rw.add_argument("--no-ladder-check", action="store_true",
                    help="skip the band that checks the ladder's own step size")
    rw.set_defaults(func=cmd_rewind)

    fd = sub.add_parser("feed", help="how far down the sheet the paper still "
                                     "feeds true")
    fd.add_argument("--out", "-o", default="results/feed.etp")
    fd.add_argument("--pitch", "-p", type=int, default=10, choices=(10, 12))
    fd.add_argument("--rows", type=int, default=DEFAULT_PAPER_ROWS,
                    help=f"lines of paper to mark, top to bottom (default "
                         f"{DEFAULT_PAPER_ROWS}). Ask for more than you believe "
                         "fits: where it stops coming out right is the answer")
    fd.add_argument("--sweeps", type=int, default=FEED_SWEEPS,
                    help=f"times the pattern crosses the page (default "
                         f"{FEED_SWEEPS}). More sweeps put fewer lines in each, "
                         "which sends the marks further across for every line "
                         "down -- a shallower run shows more of a misplaced "
                         "line, and a shorter one is harder to sight along")
    fd.set_defaults(func=cmd_feed)

    sh = sub.add_parser("sheet", help="type the charset, for scanning back in")
    sh.add_argument("--out", "-o", default="results/charset_sheet.etp")
    sh.add_argument("--pitch", "-p", type=int, default=10, choices=(10, 12))
    sh.add_argument("--sheet-cols", type=int, default=20)
    sh.add_argument("--dead-keys", action="store_true")
    sh.add_argument("--forces", default=None,
                    help="type the whole set once per strike force, hardest first "
                         "(e.g. 0,60,50,43). Pass the same list to `charset "
                         "--from-scan` -- the grid is what identifies the tiles")
    sh.set_defaults(func=cmd_sheet)

    co = sub.add_parser("codes", help="probe the control codes the pipeline does "
                                     "not use yet")
    co.add_argument("--out", "-o", default="results/control_codes.etp")
    co.add_argument("--pitch", "-p", type=int, default=10, choices=(10, 12))
    co.set_defaults(func=cmd_codes)

    fo = sub.add_parser("forces", help="sweep the strike-force command on paper")
    fo.add_argument("--out", "-o", default="results/strike_forces.etp")
    fo.add_argument("--pitch", "-p", type=int, default=10, choices=(10, 12))
    fo.add_argument("--from", dest="first", type=lambda v: int(v, 0), default=None,
                    help="first value to try (accepts 0x..); default sweeps both "
                         "of the ranges in erika_codes.FORCE_PROBE_BLOCKS")
    fo.add_argument("--to", dest="last", type=lambda v: int(v, 0), default=None,
                    help="last value to try")
    fo.add_argument("--char", default="M",
                    help="glyph to sample with; wants to be a dense one so a "
                         "change in force is obvious (default M)")
    fo.add_argument("--run", type=int, default=20,
                    help="glyphs per row (default 20)")
    fo.add_argument("--step", type=lambda v: int(v, 0), default=1,
                    help="sweep every Nth value rather than every one (default 1, "
                         "accepts 0x..). What makes a wide --from/--to readable: "
                         f"the usable space is {ec.MAX_FORCE + 1} values and more "
                         "than one sheet of paper, and at --step 16 it is seven "
                         "lines. Find the neighbourhood coarsely, then sweep it")
    fo.add_argument("--from-scan", dest="scan", default=None,
                    help="read a scan of this sheet instead of typing one: the "
                         "ink per row is the transfer curve of the force "
                         "command, which is what says where to put three or four "
                         "forces so they land evenly in tone rather than evenly "
                         "in number. Give the same sweep the sheet was typed with")
    fo.add_argument("--no-deskew", dest="deskew", action="store_false",
                    help="read the scan exactly as it was given; see the same "
                         "flag on `charset`")
    fo.add_argument("--levels", type=int, default=4,
                    help="how many forces --from-scan should suggest (default 4)")
    fo.set_defaults(func=cmd_forces)

    me = sub.add_parser("melody", help="play a rhythm on the machine's beeper")
    me.add_argument("--tune", default="shave",
                    help="one of the built-in rhythms (--list to see them)")
    me.add_argument("--notes", default=None,
                    help="a rhythm of your own, e.g. \"q e e q | -q h\"")
    me.add_argument("--morse", default=None, help="spell this out in Morse")
    me.add_argument("--list", action="store_true", help="list the built-in tunes")
    me.add_argument("--probe", action="store_true",
                    help="sweep the bell's length operand instead of playing "
                         "anything, to time the unit and find where it stops")
    me.add_argument("--tempo", "-t", dest="tempo_given", type=float, default=None,
                    help="beats per minute; a built-in tune keeps its own "
                         "unless this says otherwise")
    me.add_argument("--gate", "-g", type=float, default=melody_defaults()[0],
                    help="fraction of each slot a note sounds for. The rest is "
                         "the silence that makes it a separate note -- with one "
                         "pitch there is nothing else to separate them "
                         f"(default {melody_defaults()[0]})")
    me.add_argument("--unit", type=int, default=melody_defaults()[1],
                    help="milliseconds in one Morse unit "
                         f"(default {melody_defaults()[1]})")
    me.add_argument("--out", "-o", default="results/melody.etp")
    me.set_defaults(func=cmd_melody)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except PlanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
