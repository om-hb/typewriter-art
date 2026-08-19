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
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")  # optimize.py builds a figure

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from erika import SRC_DIR, erika_codes as ec
from erika import etp, planner, preview
from erika.planner import Charset, PlanError

#: Layer schemes in layers.json whose offsets are all multiples of 0.5.
TYPEABLE_LAYER_SCHEMES = ("1x1", "1x2", "1x4", "2Hx1", "2Hx2", "2Vx1", "2Vx2", "4x1", "4x2")

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

#: Lines the `area` sheet marks out by default. 60 lines at Zeilenschaltung 1 is
#: 254 mm -- an A4 sheet with roughly 20 mm spare at each end. Unlike the width
#: this is not a machine limit, just a useful default; see cmd_area.
DEFAULT_AREA_ROWS = 60


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
        scan=args.from_scan,
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

    if args.layers not in TYPEABLE_LAYER_SCHEMES:
        raise PlanError(
            f"layer scheme '{args.layers}' uses offsets the Sigma cannot hit. "
            f"Pick one of: {', '.join(TYPEABLE_LAYER_SCHEMES)}"
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
    )
    job = planner.encode(plan, settle_ms=args.settle_ms, cr_delay_ms=args.cr_delay_ms,
                         no_advance=args.no_advance)
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


def cmd_sheet(args) -> int:
    """Type the full charset so it can be scanned into a real charset."""
    glyphs = ec.all_glyphs(dead_keys=args.dead_keys)
    forces = _parse_forces_arg(args.forces)
    cols = args.sheet_cols
    enc = etp.Encoder()

    # Contiguous, force block after force block, with no line break where one
    # block ends -- because make_charset lays the tiles out the same way, and it
    # has to: a blank tile between two glyphs would be dropped by chop_charset
    # and shift every index after it. See build_sheet.
    i = 0
    for force in forces or [None]:
        if force is not None:
            enc.set_force(force)
        for glyph in glyphs:
            if i and i % cols == 0:
                enc.newline(1)
            enc.strike(glyph.code, glyph.advances)
            if not glyph.advances:
                enc.right(2)
            i += 1
    if forces:
        enc.set_force(forces[0])  # leave the machine as it was found
    enc.newline(3)
    enc.carriage_return()
    enc.end()

    tiles = len(glyphs) * max(1, len(forces))
    rows = (tiles + cols - 1) // cols
    job = etp.Job(body=enc.body(), cols=cols, rows=rows, strikes=enc.strikes,
                  pitch=args.pitch, home_each_row=True)
    out = args.out if os.path.isabs(args.out) else os.path.join(SRC_DIR, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    size = etp.save(out, job)
    forces_note = (f" at {len(forces)} strike forces "
                   f"({', '.join(f'0x{f:02X}' for f in forces)})" if forces else "")
    print(f"charset sheet -> {out} ({size} bytes, {len(glyphs)} glyphs{forces_note}, "
          f"{cols} per line)")
    print("\nType it, scan the block of glyphs square-on cropped to the outermost")
    print("ink, then build a charset from the real type:")
    print(f"  python -m erika.pipeline charset --pitch {args.pitch} "
          f"--name sigma-scanned --from-scan /path/to/scan.png"
          + (f" --forces {args.forces}" if forces else ""))
    if forces:
        print("\nThe --forces there must match, and in the same order: the scan is")
        print("sliced to a grid, and the grid is what says which tile is which.")
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

    The published control code table (erika_ai/ressources/steuercodes.md) gives
    a weak steer on which hypothesis to read first: 0xA5, 0xA6 and 0xA7 all
    spell their operand out as a raw count, and 0xA3 uses the same phrasing for
    the force. That points at the `raw` block rather than the `ascii` one.
    """
    blocks = _force_probe_blocks(args)
    run = args.run
    glyph = ec.glyph_for_char(args.char)
    if glyph is None:
        raise PlanError(f"the Sigma has no key for {args.char!r}")

    enc = etp.Encoder()
    type_line(enc, "STRIKE FORCE PROBE")
    enc.newline(1)

    def sample(label: str) -> None:
        type_row(enc, [(0, label)])
        enc.right(2 * (len(label) + 1))
        for _ in range(run):
            enc.strike(glyph.code, glyph.advances)
        enc.newline(1)

    # Before any force command: what the machine does on its own.
    sample("AS FOUND ")

    for name, values in blocks.items():
        enc.newline(1)
        type_line(enc, f"{name.upper()}:")
        for value in values:
            enc.set_force(value)
            sample(f"{value:>3} 0x{value:02X}")

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
    print("\nThen, with the values that worked, hardest first:")
    print("  python -m erika.pipeline sheet --forces 0,3,6      # type it")
    print("  python -m erika.pipeline charset --forces 0,3,6 \\")
    print("      --name sigma-forces --from-scan /path/to/scan.png")
    print("\nOne piece of advice from the paper that does NOT transfer: its")
    print("figure 20 found medium plus light beat dark plus medium, because its")
    print("typewriter could already reach black. This one cannot, so keep the")
    print("hardest strike and add lighter ones below it.")
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

    # 9. Pitch, last, because a pitch that sticks rescales everything after it.
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

    enc.carriage_return()
    enc.end()

    job = etp.Job(body=enc.body(), cols=n + 2, rows=34, strikes=enc.strikes,
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
    print("  4  Eight marks in ONE column: four gaps driven by the line-feed")
    print(f"     mechanism, then four asked for as {full_line} platen steps. The")
    print("     two sets of gaps must be equal. Any sideways shift means a")
    print("     motor-step feed nudges the carriage, which the detented feed")
    print("     does not.")
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
    print("     columns of carriage instead of 65. Same width means 0x89 did")
    print("     nothing. If everything *after* this sheet comes out narrow, the")
    print(f"     restore (0x{restore:02X}) did not take: set the pitch switch by")
    print("     hand, or send 0x95 to reset the machine.")
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
    p.add_argument("--settle-ms", type=int, default=0,
                   help="pause after each paper feed, in ms")
    p.add_argument("--cr-delay-ms", type=int, default=0,
                   help="pause after each carriage return, in ms")
    p.add_argument("--jitter", type=float, default=0.05,
                   help="registration error for the shaky preview, in cells (default 0.05)")
    p.add_argument("--fine", action="store_true",
                   help="place strikes with the machine's own motor steps "
                        "(1/120 inch across, 1/240 down) where a layer offset "
                        "falls between the half-cell grid points -- which is what "
                        "makes the quarter-cell schemes (16x1) and daisy_full "
                        "typeable. UNCONFIRMED: needs 0xA5 and 0xA6, so type "
                        "`erika.pipeline codes` first")
    p.add_argument("--no-advance", action="store_true",
                   help="type a stack of glyphs in one cell with Doppeldruck "
                        "(0xA9) instead of a backspace between each pair, so the "
                        "escapement never moves. UNCONFIRMED on this machine -- "
                        "type `erika.pipeline codes` and read part 6 first")
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
    c.add_argument("--from-scan", default=None)
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
                   help=f"layer scheme; typeable ones are {', '.join(TYPEABLE_LAYER_SCHEMES)}")
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
    ar.add_argument("--rows", type=int, default=DEFAULT_AREA_ROWS,
                    help=f"lines to mark out; the machine has no vertical limit, "
                         f"only the paper does (default {DEFAULT_AREA_ROWS}, "
                         f"= {DEFAULT_AREA_ROWS * ec.LINE_HEIGHT_MM:.0f} mm)")
    ar.add_argument("--columns", type=int, default=None,
                    help="width to mark out; default is the carriage limit "
                         f"({ec.MAX_COLUMNS[10]} at pitch 10, "
                         f"{ec.MAX_COLUMNS[12]} at pitch 12). Set it to an -r "
                         "value to see where that print would land")
    ar.set_defaults(func=cmd_area)

    sh = sub.add_parser("sheet", help="type the charset, for scanning back in")
    sh.add_argument("--out", "-o", default="results/charset_sheet.etp")
    sh.add_argument("--pitch", "-p", type=int, default=10, choices=(10, 12))
    sh.add_argument("--sheet-cols", type=int, default=20)
    sh.add_argument("--dead-keys", action="store_true")
    sh.add_argument("--forces", default=None,
                    help="type the whole set once per strike force, hardest first "
                         "(e.g. 0,3,6). Pass the same list to `charset "
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
    fo.set_defaults(func=cmd_forces)

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
