"""Read a scanned strike-force probe sheet back as a density curve.

`pipeline forces` types a sheet and the eye reads it: which values marked the
paper, which typed a stray character, where the ink starts and where it stops
changing. That answers what the sheet was built to answer -- whether the machine
honours the command at all, and how the operand is spelled.

It does not answer the question that comes next. A multi-force charset wants
three or four values spaced evenly *in tone*, and the force scale is a lever
position rather than a quantity of ink: on the machine this was written for the
gap between the first legible character and a fully formed one is twelve values,
and the forty above that are the top of the range. Picking by even arithmetic
puts three of four samples in the same place.

The sheet already holds the answer. Every row is the same glyph struck the same
number of times, so the ink per row *is* the transfer curve of the force
command, and a scan turns reading it off into arithmetic. Which is the same
argument as `charset --from-scan`: a measurement of the machine beats a model of
it, and the machine has already written the measurement down.

**The grid is computed, not detected.** A row typed at a force below the ink
threshold is blank -- and so is its label, which is typed at the row's own force
-- so the rows cannot be found by looking for them. Blank rows are data (they
are what says where the threshold is), and a reader that skipped them would
renumber every row after. So the geometry comes from the title line, which is
typed before any force command and is therefore always there: its width gives
the cell, the pitch gives the line, and every row after is counted off the grid.
That is also why `probe_lines` is shared with the builder rather than repeated
here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from erika import deskew, erika_codes as ec

#: The title, and the reference row typed before any force command. The title
#: doubles as the scan's ruler, so its width is load-bearing: it is measured in
#: cells to find the cell.
PROBE_TITLE = "STRIKE FORCE PROBE"
AS_FOUND_LABEL = "AS FOUND "

#: Ink below this fraction of the darkest row is treated as paper rather than a
#: mark. Scanner noise and paper texture land around a couple of percent.
INK_FLOOR = 0.04


@dataclass(frozen=True)
class ProbeLine:
    """One line of the probe sheet, whether or not anything printed on it.

    ``kind`` is 'title', 'blank', 'heading' or 'sample'. A sample carries the
    force it was typed at (None for the reference row), the label typed at the
    left margin, and the column the run of glyphs starts at.
    """

    kind: str
    text: str = ""
    value: int | None = None
    run_col: int = 0
    run_cells: int = 0


def sample_label(value: int) -> str:
    """The label a sample row carries. Fixed width, so the runs line up."""
    return f"{value:>3} 0x{value:02X}"


def run_column(label: str) -> int:
    """Where the run of glyphs starts, given the label typed before it.

    The builder types the label from the left margin, which leaves the head at
    ``len(label)``, then moves ``len(label) + 1`` cells further. So the run
    begins at ``2 * len(label) + 1`` and not, as the arithmetic looks like it
    ought to say, one cell past the label. It has been that way since the sheet
    was first typed; this function is the single place that knows it, and the
    reader asks it rather than repeating the sum.
    """
    return 2 * len(label) + 1


def probe_lines(blocks: dict[str, list[int]], run: int) -> list[ProbeLine]:
    """Every line of the sheet in order, so a row's index is its y on paper.

    Shared with ``pipeline.cmd_forces``, which walks this to type the sheet.
    One list, one order: a reader that counted lines for itself would be a
    fourth hand-mirrored table, and a sheet whose rows are misidentified reads
    as a machine with a strange force curve rather than as a bug.
    """
    lines = [
        ProbeLine("title", PROBE_TITLE),
        ProbeLine("blank"),
        ProbeLine("sample", AS_FOUND_LABEL, None,
                  run_column(AS_FOUND_LABEL), run),
    ]
    for name, values in blocks.items():
        lines.append(ProbeLine("blank"))
        lines.append(ProbeLine("heading", f"{name.upper()}:"))
        for value in values:
            label = sample_label(value)
            lines.append(ProbeLine("sample", label, value, run_column(label), run))
    return lines


@dataclass(frozen=True)
class Reading:
    """One row of the sheet, as the scan found it."""

    value: int | None  #: None for the reference row typed before any command
    label: str
    ink: float  #: mean ink over the run, 0 (paper) to 1 (the darkest row)
    marked: bool  #: did anything reach the paper at all


def _grid(ink: np.ndarray, pitch: int) -> tuple[float, float, float, float]:
    """Origin and cell from the title line: (x0, y0, cell_px, line_px).

    The title is the topmost ink on the sheet and is a known number of cells
    wide, which makes it a ruler that every scan carries. Its extent is measured
    to the outermost ink, so it is short of the full cell box by one side
    bearing at each end -- about an eighth of a cell on this machine, the same
    figure ``make_charset`` allows for when it declines to crop a charset sheet
    to the ink.
    """
    rows = ink.sum(axis=1)
    inked = np.flatnonzero(rows > rows.max() * 0.02)
    if inked.size == 0:
        raise ValueError("no ink anywhere on the scan")

    # The title band: from the first inked row down to the first clear one.
    top = int(inked[0])
    clear = np.flatnonzero(rows[top:] <= rows.max() * 0.02)
    if clear.size == 0:
        raise ValueError("the scan is ink from the title down; is it cropped?")
    bottom = top + int(clear[0])

    cols = ink[top:bottom].sum(axis=0)
    lit = np.flatnonzero(cols > cols.max() * 0.02)
    if lit.size == 0:
        raise ValueError("the title line has no ink in it")
    left, right = int(lit[0]), int(lit[-1])

    bearings = 0.25  # a side bearing at each end, in cells
    cell_px = (right - left + 1) / (len(PROBE_TITLE) - bearings)
    x0 = left - cell_px * bearings / 2
    line_px = cell_px * ec.cell_aspect(pitch)
    return x0, float(top), cell_px, line_px


def read_scan(
    scan_path: str,
    blocks: dict[str, list[int]],
    run: int,
    pitch: int = 10,
    deskew_scan: bool = True,
) -> list[Reading]:
    """Measure the ink on every sample row of a scanned probe sheet.

    ``blocks`` and ``run`` must be what the sheet was typed with -- the same
    ``--from/--to/--step/--run`` -- for the same reason ``charset --from-scan``
    needs the sheet's ``--forces`` and ``--sheet-cols``: the grid is the only
    thing that says which row is which, and a row misidentified still reads as a
    perfectly plausible curve.
    """
    import cv2  # noqa: PLC0415 -- a heavy import for one code path

    im = cv2.imread(scan_path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(scan_path)

    # Square the sheet up first. This grid is computed rather than detected, so
    # rotation does not blur a measurement here -- it silently reads each row a
    # little further along the one below it, which is worse.
    if deskew_scan:
        im, angle, note = deskew.straighten(im)
        if angle:
            print(f"deskewed the scan by {angle:+.2f} deg")
        if note:
            print(f"WARNING: {note}")

    # Only the paper is normalised, and only for scanner exposure. The ink is
    # the measurement -- stretching the darkest row to black here would be
    # measuring the scanner's black point instead of the machine's hardest
    # strike, which is make_charset's mistake from section 5.5 in a new place.
    paper = float(np.percentile(im, 95))
    if paper <= 0:
        raise ValueError("the scan has no paper in it")
    ink = np.clip(1.0 - im.astype("float32") / paper, 0.0, 1.0)

    x0, y0, cell_px, line_px = _grid(ink, pitch)
    height, width = ink.shape

    readings: list[Reading] = []
    for index, line in enumerate(probe_lines(blocks, run)):
        if line.kind != "sample":
            continue
        top = int(round(y0 + index * line_px))
        bottom = int(round(y0 + (index + 1) * line_px))
        # Half a cell in at each end, so a cell width that is out by a percent
        # or two still samples the run and nothing but the run.
        left = int(round(x0 + (line.run_col + 0.5) * cell_px))
        right = int(round(x0 + (line.run_col + line.run_cells - 0.5) * cell_px))
        if top >= height or left >= width or right <= left or bottom <= top:
            raise ValueError(
                f"row {index} ({line.text.strip() or 'reference'}) falls outside "
                "the scan. The sheet is longer than the image: check that the "
                "sweep given here is the one it was typed with, and that the "
                "whole sheet was scanned."
            )
        window = ink[top:min(bottom, height), left:min(right, width)]
        readings.append(Reading(line.value, line.text.strip() or "as found",
                                float(window.mean()), False))

    darkest = max((r.ink for r in readings), default=0.0)
    if darkest <= 0:
        raise ValueError("no row on the sheet has any ink on it")
    return [
        Reading(r.value, r.label, r.ink / darkest, r.ink / darkest >= INK_FLOOR)
        for r in readings
    ]


def suggest(readings: list[Reading], levels: int) -> list[int]:
    """Force values whose ink is spaced evenly, hardest first.

    Even in *tone*, which is the whole point: the values are a lever position
    and the ramp between the threshold and saturation is not linear in them.
    The darkest row is kept whatever else is -- this machine cannot reach black,
    so the paper's advice to trade the hardest strike for two lighter ones does
    not transfer (its figure 20; see ``cmd_forces``).
    """
    if levels < 1:
        raise ValueError(f"levels must be 1 or more, got {levels}")
    marked = [r for r in readings if r.value is not None and r.marked]
    if not marked:
        raise ValueError("no row with a force on it printed anything")

    darkest = max(r.ink for r in marked)
    chosen: list[int] = []
    # Ties go to the first value swept, which is the lowest -- so a saturated
    # top of the scale is represented by 0 rather than by whichever of the
    # identical values happened to be sampled. Hardest first is by *ink*: 0 is
    # full strike, so the list this returns usually opens with its smallest
    # number, and sorting it numerically afterwards would invert it.
    for step in range(levels):
        want = darkest * (levels - step) / levels
        pick = min(
            (r for r in marked if r.value not in chosen),
            key=lambda r: abs(r.ink - want),
            default=None,
        )
        if pick is not None:
            chosen.append(pick.value)
    return chosen


def _ends(readings: list[Reading]) -> list[str]:
    """Where the ramp starts and where it stops -- the two numbers to quote.

    Zero is stepped over on purpose. It is full strike on this machine, so it
    marks, and reporting it as "the first value that reached the paper" would
    name the top of the scale as the bottom of it. The threshold is the lowest
    value *above* zero that took ink, and everything between the two is the
    dead band.
    """
    marked = [r for r in readings if r.value is not None and r.marked]
    if not marked:
        return ["nothing on this sheet reached the paper."]

    lines = []
    ramp = [r for r in marked if r.value > 0]
    if ramp:
        first = min(ramp, key=lambda r: r.value)
        lines.append(f"ink begins at {first.value} (0x{first.value:02X})")

        # Saturation: the lowest value from which nothing gets darker. Anything
        # above it is a force block spent twice on the same tone.
        top = max(r.ink for r in ramp)
        ordered = sorted(ramp, key=lambda r: r.value)
        flat = [
            r.value for i, r in enumerate(ordered)
            if all(o.ink >= top * 0.99 for o in ordered[i:])
        ]
        if flat and flat[0] != ordered[-1].value:
            lines.append(f"and stops changing from {flat[0]} (0x{flat[0]:02X}) "
                         "up -- a force above that is a block spent twice")
    if any(r.value == 0 and r.marked for r in readings):
        lines.append("0 prints solid, so it is the top of this scale and not "
                     "the bottom of it")
    return lines


def report(readings: list[Reading], levels: int) -> str:
    """The curve, and what to do with it. Printed by ``pipeline forces``."""
    out = ["", "ink per row, as a fraction of the darkest (the run only, not "
           "the label):", ""]
    for r in readings:
        bar = "#" * int(round(r.ink * 40))
        note = "" if r.marked else "   (no ink)"
        out.append(f"  {r.label:<9} {r.ink:5.3f}  {bar}{note}")

    out += ["", *_ends(readings)]
    picked = suggest(readings, levels)
    if picked:
        by_value = {r.value: r for r in readings}
        top = by_value[picked[0]].ink
        densities = [f"{by_value[v].ink / top:.2f}" for v in picked[1:]]
        out += [
            "",
            f"{levels} forces spaced evenly in ink, hardest first:",
            f"  --forces {','.join(str(v) for v in picked)}",
        ]
        if densities:
            out += [
                "",
                "and the ink each of the lighter ones transfers, if you want a",
                "modelled charset to match the measured one:",
                f"  --force-density {','.join(densities)}",
            ]
    out += [
        "",
        "A value that only just marks is worth the most tonally and is trusted",
        "least: it sits on a threshold, so ribbon wear and wheel position move",
        "it. Look at its tiles on the charset scan before believing the charset.",
    ]
    return "\n".join(out)


def relative_path(path: str) -> str:
    """A path as the printout should show it.

    Relative only when that is actually shorter: a scan in a temporary
    directory is a dozen ``..`` segments away from a checkout, and the absolute
    path is the one a reader can paste back.
    """
    try:
        relative = os.path.relpath(path)
    except ValueError:  # a different drive, on Windows
        return path
    return relative if len(relative) < len(path) else path
