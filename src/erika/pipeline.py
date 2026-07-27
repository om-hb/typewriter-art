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
    from erika.make_charset import make_charset

    make_charset(
        name=args.name,
        pitch=args.pitch,
        cell_height=args.cell_height,
        font=args.font,
        bleed=args.bleed,
        dead_keys=args.dead_keys,
        scan=args.from_scan,
        base_path=SRC_DIR,
    )
    return 0


def run_optimizer(args) -> str:
    """Run optimize.py and return the path to the choices it wrote."""
    import optimize

    if args.layers not in TYPEABLE_LAYER_SCHEMES:
        raise PlanError(
            f"layer scheme '{args.layers}' uses offsets the Sigma cannot hit. "
            f"Pick one of: {', '.join(TYPEABLE_LAYER_SCHEMES)}"
        )

    choices_path = os.path.join(SRC_DIR, "results", "choices.json")
    if os.path.exists(choices_path):
        os.remove(choices_path)

    print(f"optimizing {args.target} at {args.row_length} columns, "
          f"{args.layers} layers, {args.num_loops} loops ...")
    try:
        optimize.kword(
            charset=args.charset,
            target=args.target,
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
    )
    job = planner.encode(plan, settle_ms=args.settle_ms, cr_delay_ms=args.cr_delay_ms)
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
    cols = args.sheet_cols
    enc = etp.Encoder()
    for i, glyph in enumerate(glyphs):
        if i and i % cols == 0:
            enc.newline(1)
        enc.strike(glyph.code, glyph.advances)
        if not glyph.advances:
            enc.right(2)
    enc.newline(3)
    enc.carriage_return()
    enc.end()

    rows = (len(glyphs) + cols - 1) // cols
    job = etp.Job(body=enc.body(), cols=cols, rows=rows, strikes=enc.strikes,
                  pitch=args.pitch, home_each_row=True)
    out = args.out if os.path.isabs(args.out) else os.path.join(SRC_DIR, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    size = etp.save(out, job)
    print(f"charset sheet -> {out} ({size} bytes, {len(glyphs)} glyphs, "
          f"{cols} per line)")
    print("\nType it, scan the block of glyphs square-on cropped to the outermost")
    print("ink, then build a charset from the real type:")
    print(f"  python -m erika.pipeline charset --pitch {args.pitch} "
          f"--name sigma-scanned --from-scan /path/to/scan.png")
    return 0


# ---------------------------------------------------------------------------


def _add_charset_arg(p, default="sigma-10"):
    p.add_argument("--charset", "-c", default=default,
                   help=f"charset folder under src/charsets (default {default})")


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
    sh.set_defaults(func=cmd_sheet)

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
