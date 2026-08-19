"""Generate a typewriter-art charset for the Sigma SM 8200i.

The charsets shipped with typewriter-art are photographic scans of *other*
machines (Hermes, Smith-Corona, Daisywriter). They are fine for producing a
mockup, but they carry no mapping from a character index back to a key -- so
there is no way to tell the Sigma what to type.

This module builds a charset that does carry that mapping:

  charsets/<name>/sigma.png     glyph sheet, one cell per typeable character
  charsets/<name>/config.json   the config chop_charset() consumes
  charsets/<name>/glyphs.json   index -> {char, code, advances}   <- the point
  charsets/<name>/preview.png   labelled contact sheet, for eyeballing

Cell geometry is the machine's real geometry: cell width = one pitch step,
cell height = one line advance at Zeilenschaltung 1. That makes a 0.5 layer
offset in layers.json exactly one half-step / one half-line on the machine.

The glyph shapes come from a monospace outline font (Courier New by default),
which is an approximation of the Sigma's own typeface. **A scan is the real
answer**: type the charset sheet (`pipeline.py sheet`), scan it, and build from
it with `--from-scan`. What the font path can only guess at is how much ink
reaches the paper, and that turns out to decide the picture:

  - An outline glyph is *pure black with hard edges*, which no ribbon produces.
    It matters more than it sounds, because the optimizer scores candidates per
    **pixel**: a stroke at grey 0 laid through a mid-grey cell costs more
    squared error than leaving the cell empty, so the optimizer declines to mark
    it at all. Measured on the sample photograph, half of every cell in a
    default run came out blank while the picture as a whole was 39 grey levels
    too light. `--ink` and `--spread` exist to stop the model being impossible;
    they do not make it accurate.
  - Which direction to err in, when guessing: a charset modelled *lighter* than
    the machine makes the optimizer ask for more ink than the paper needs, and
    the print comes out dark. Modelled *darker* -- which is what pure black
    does -- it declines to mark midtones and the print comes out blotchy and
    pale, losing detail that cannot be recovered afterwards. So the defaults
    lean light.

`--forces` is the other half of the same problem, and the more valuable one. The
machine can strike at more than one force (see erika_codes.SET_STRIKE_FORCE),
and the paper's section 5.5 calls a charset with varying strike force "the
largest factor in obtaining a good tonal range in the midtones and highlights".
Repeating every glyph at each force is how a charset offers that. On the font
path the lighter copies are *modelled* by `--force-density` and are a guess of
unknown quality; on the scan path they are measured, which is the whole reason
to prefer it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

if __package__ in (None, ""):  # allow `python erika/make_charset.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from erika import SRC_DIR, erika_codes as ec

DEFAULT_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "/Library/Fonts/Courier New.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Courier_New.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "C:/Windows/Fonts/cour.ttf",
)

#: Tiles whose mean brightness is >= this are treated as blank by
#: chop_charset() and silently dropped. Keep it high enough that even '´'
#: survives; _verify_mapping() asserts that nothing was actually dropped.
WHITE_THRESHOLD = 0.999

SUPERSAMPLE = 8

#: Grey the densest ink reaches, 0 = pure black. Typed characters do not reach
#: black -- the paper's section 5.5 makes the point that this is a *feature*,
#: "even the darkest typewriter characters are less than fully black, which
#: yields a greater tonal range when characters are allowed to overlap" -- and
#: the charsets it scanned bottom out around 25/255. Hence 0.10 rather than 0.
DEFAULT_INK = 0.10

#: Point spread of the ink, in cell pixels. Ribbon fabric, impact and paper all
#: soften the edge of a mark; an outline font has no edge softness at all beyond
#: its antialiasing. Applied per cell, so ink never leaks into a neighbour.
DEFAULT_SPREAD = 0.6

#: How much ink each force after the first transfers, as a fraction of the
#: first. Pure guesswork -- the numbers a real machine gives have to be scanned
#: -- but the *shape* is what the paper's figure 20 shows: usable extra levels
#: come from adding lighter strikes below the full one, not from spreading
#: evenly. Only consulted when --forces names more than one force and the
#: glyphs come from a font.
DEFAULT_FORCE_DENSITY = (0.55, 0.30, 0.18)


def find_font(explicit: str | None = None) -> str:
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(explicit)
        return explicit
    for path in DEFAULT_FONT_CANDIDATES:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "No monospace font found. Pass --font /path/to/a/monospace.ttf"
    )


def _fit_font(font_path: str, target_advance: float) -> ImageFont.FreeTypeFont:
    """Pick the point size whose advance width is closest to target_advance."""
    lo, hi = 1, 4000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ImageFont.truetype(font_path, mid).getlength("M") <= target_advance:
            lo = mid
        else:
            hi = mid - 1
    return ImageFont.truetype(font_path, lo)


def _draw(font, char, big_w, big_h, pad, baseline) -> np.ndarray:
    """Draw one glyph on a padded canvas, pen origin at (pad, baseline)."""
    canvas = Image.new("L", (big_w + 2 * pad, big_h + 2 * pad), 255)
    ImageDraw.Draw(canvas).text((pad, baseline), char, font=font, fill=0, anchor="ls")
    return np.asarray(canvas, dtype=np.uint8)


def _ink_bbox(arr: np.ndarray) -> tuple[int, int] | None:
    """(top, bottom) rows containing ink, or None if the glyph is blank."""
    rows = np.flatnonzero((arr < 250).any(axis=1))
    return (int(rows[0]), int(rows[-1]) + 1) if rows.size else None


def apply_ink_model(
    tile: np.ndarray, ink: float, spread: float, density: float = 1.0
) -> np.ndarray:
    """Turn a black-on-white glyph into something a ribbon could have made.

    ``ink`` is the grey the densest ink reaches, ``density`` scales that for a
    lighter strike force, and ``spread`` is the point spread in cell pixels. The
    blur is padded with paper and cropped back, so a cell never lends ink to its
    neighbour -- overlapping is the planner's business and has to stay exact.
    """
    floor = 1.0 - (1.0 - ink) * density
    out = floor + (1.0 - floor) * tile
    if spread > 0:
        k = max(3, int(round(spread * 6)) | 1)
        h, w = out.shape
        padded = cv2.copyMakeBorder(out, k, k, k, k, cv2.BORDER_CONSTANT, value=1.0)
        padded = cv2.GaussianBlur(padded, (k, k), spread)
        out = padded[k : k + h, k : k + w]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


class CellLayout:
    """Shared type placement for a whole glyph set.

    A typeface cut for a given pitch fits all of its glyphs -- accents at the
    top, descenders and the underscore at the bottom -- inside one line. An
    outline font's ascent/descent metrics are usually taller than that, so
    centring on them clips the extremes. Instead we measure the union of the
    actual ink extents across the set and centre *that* in the cell, with one
    baseline shared by every glyph so their relative heights stay correct.
    """

    def __init__(self, font_path: str, cell_w: int, cell_h: int, glyphs):
        s = SUPERSAMPLE
        self.s = s
        self.big_w, self.big_h = cell_w * s, cell_h * s
        self.pad = max(self.big_w, self.big_h)
        self.font = _fit_font(font_path, self.big_w)
        self.shrunk = 1.0

        top, bottom = self._measure(self.font, glyphs)
        extent = bottom - top
        if extent > self.big_h:
            # Glyph set is taller than the line: scale the type down to fit.
            self.shrunk = self.big_h / extent
            size = max(1, int(self.font.size * self.shrunk))
            self.font = ImageFont.truetype(font_path, size)
            top, bottom = self._measure(self.font, glyphs)
            extent = bottom - top
        self.extent = extent
        # Place the union ink box centred in the cell.
        self.baseline = (self.big_h - extent) / 2.0 - top + self.pad

    def _measure(self, font, glyphs) -> tuple[int, int]:
        """Union ink extent relative to the baseline, in supersampled px."""
        top, bottom = 0, 0
        for g in glyphs:
            arr = _draw(font, g.char, self.big_w, self.big_h, self.pad, self.pad)
            bbox = _ink_bbox(arr)
            if bbox is None:
                raise ValueError(
                    f"{g.char!r} renders blank in this font -- pick another --font"
                )
            top = min(top, bbox[0] - self.pad)
            bottom = max(bottom, bbox[1] - self.pad)
        return top, bottom

    def render(
        self,
        glyph: ec.Glyph,
        bleed: float,
        ink: float = 0.0,
        spread: float = 0.0,
        density: float = 1.0,
    ) -> tuple[np.ndarray, float]:
        """Render one glyph to a float32 cell in [0, 1]; 1.0 == bare paper.

        The three arguments after ``bleed`` are the ink model, and the order
        they are applied in is the order the physical process happens in:
        ``bleed`` spreads the *type*, ``ink``/``density`` decide how dark the
        mark is, and ``spread`` softens its edge on the paper. Doing ink before
        spread is what leaves the edges lighter than the middle; the other order
        would give a soft edge at full density, which is not what a mark looks
        like.
        """
        arr = _draw(self.font, glyph.char, self.big_w, self.big_h, self.pad, self.baseline)
        if bleed > 0:
            k = max(1, int(round(bleed * self.s)))
            k += 1 - (k % 2)  # erode wants an odd kernel
            # On a white ground, erode == min filter == ink spread.
            arr = cv2.erode(arr, np.ones((k, k), np.uint8))
        spill = self._ink_outside_cell(arr)
        cell = arr[self.pad : self.pad + self.big_h, self.pad : self.pad + self.big_w]
        small = cv2.resize(
            cell, (self.big_w // self.s, self.big_h // self.s), interpolation=cv2.INTER_AREA
        )
        tile = small.astype(np.float32) / 255.0
        return apply_ink_model(tile, ink, spread, density), spill

    def _ink_outside_cell(self, arr: np.ndarray) -> float:
        """Fraction of a glyph's ink that falls outside its own cell."""
        ink = 255 - arr.astype(np.int32)
        total = ink.sum()
        if total == 0:
            return 0.0
        inside = ink[
            self.pad : self.pad + self.big_h, self.pad : self.pad + self.big_w
        ].sum()
        return float(total - inside) / float(total)


def build_sheet(
    glyphs: tuple[ec.Glyph, ...],
    font_path: str,
    cell_w: int,
    cell_h: int,
    bleed: float,
    cols: int,
    ink: float = 0.0,
    spread: float = 0.0,
    densities: tuple[float, ...] = (1.0,),
) -> tuple[np.ndarray, int, int, list[float], CellLayout]:
    """Lay the glyph set out once per strike force, all in one grid.

    Tiles run contiguously, so a force block can begin part-way through a row.
    That looks untidy on the contact sheet and is deliberate: ``chop_charset``
    drops any tile it judges blank, and a blank tile *between* two glyphs shifts
    every index after it. Contiguous means the only gap is at the very end,
    where dropping it shifts nothing. ``pipeline sheet`` types the glyphs in the
    same contiguous order for the same reason, so a scan slices to this grid.
    """
    layout = CellLayout(font_path, cell_w, cell_h, glyphs)
    total = len(glyphs) * len(densities)
    rows = (total + cols - 1) // cols
    sheet = np.ones((rows * cell_h, cols * cell_w), dtype=np.float32)
    spills = []
    i = 0
    for density in densities:
        for glyph in glyphs:
            tile, spill = layout.render(glyph, bleed, ink, spread, density)
            spills.append(spill)
            r, c = divmod(i, cols)
            sheet[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w] = tile
            i += 1
    return sheet, cols, rows, spills, layout


def _labelled_preview(sheet: np.ndarray, entries, cell_w, cell_h, cols) -> np.ndarray:
    """Contact sheet with index/char/code annotations, scaled up for reading.

    ``entries`` is the (glyph, force) list the charset is actually built from, so
    a multi-force sheet labels which force each tile was struck at -- the one
    place the two blocks can be told apart by eye.
    """
    scale = max(2, 96 // cell_w)
    label_h = 34
    rows = (len(entries) + cols - 1) // cols
    out_w = cols * cell_w * scale
    out_h = rows * (cell_h * scale + label_h)
    out = np.full((out_h, out_w), 255, dtype=np.uint8)
    for i, (glyph, force) in enumerate(entries):
        r, c = divmod(i, cols)
        tile = sheet[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
        big = cv2.resize(
            (tile * 255).astype(np.uint8),
            (cell_w * scale, cell_h * scale),
            interpolation=cv2.INTER_NEAREST,
        )
        y0 = r * (cell_h * scale + label_h)
        x0 = c * cell_w * scale
        out[y0 : y0 + cell_h * scale, x0 : x0 + cell_w * scale] = big
        cv2.rectangle(
            out, (x0, y0), (x0 + cell_w * scale - 1, y0 + cell_h * scale - 1), 200, 1
        )
        # Index 0 is the blank tile that chop_charset prepends, so glyph i
        # ends up at charset index i + 1.
        label = f"{i + 1}:{glyph.code:02X}"
        if force is not None:
            label += f"f{force:02X}"
        cv2.putText(
            out,
            label,
            (x0 + 2, y0 + cell_h * scale + 14),
            cv2.FONT_HERSHEY_PLAIN,
            0.8,
            0,
            1,
            cv2.LINE_AA,
        )
    return out


def _verify_mapping(charset_dir: str, base_path: str, expected: int) -> None:
    """Re-run the upstream loader and confirm our index mapping is what it sees.

    chop_charset() silently drops tiles it considers blank, which would shift
    every index after the dropped one and quietly mistype the whole image.
    Rather than trusting that it won't, we run it and check the count.
    """
    from utils import prep_charset

    chars, x_change, y_change = prep_charset(os.path.basename(charset_dir), base_path)
    if len(chars) != expected:
        raise AssertionError(
            f"chop_charset() returned {len(chars)} tiles, expected {expected}. "
            "A glyph was probably dropped as blank -- raise whiteThreshold or "
            "increase --bleed."
        )
    if abs(x_change - 1.0) > 1e-6 or abs(y_change - 1.0) > 1e-6:
        raise AssertionError(
            f"chop_charset() rescaled the sheet (xChange={x_change}, "
            f"yChange={y_change}); cell dimensions must be even integers."
        )


def resolve_densities(
    n_forces: int, densities: tuple[float, ...] | None
) -> tuple[float, ...]:
    """One ink density per force, the first always 1.0 (full measure).

    Only the font path uses these; a scan brings its own.
    """
    if n_forces <= 1:
        return (1.0,)
    if densities is None:
        densities = DEFAULT_FORCE_DENSITY[: n_forces - 1]
    if len(densities) != n_forces - 1:
        raise ValueError(
            f"{n_forces} forces need {n_forces - 1} densities for the "
            f"lighter ones, got {len(densities)}"
        )
    if not all(0 < d <= 1 for d in densities):
        raise ValueError(f"densities must be in (0, 1], got {densities}")
    return (1.0,) + tuple(densities)


def make_charset(
    name: str | None = None,
    pitch: int = 10,
    cell_height: int = 40,
    font: str | None = None,
    bleed: float = 0.2,
    dead_keys: bool = False,
    sheet_cols: int = 20,
    base_path: str | None = None,
    scan: str | None = None,
    ink: float = DEFAULT_INK,
    spread: float = DEFAULT_SPREAD,
    forces: tuple[int, ...] = (),
    force_densities: tuple[float, ...] | None = None,
) -> str:
    base_path = base_path or SRC_DIR
    if pitch not in ec.PITCH_WIDTH_MM:
        raise ValueError(f"pitch must be 10 or 12, got {pitch}")
    bad = [f for f in forces if not ec.is_usable_force(f)]
    if bad:
        raise ValueError(
            f"strike force(s) {', '.join(f'0x{f:02X}' for f in bad)} are motion "
            "codes. A machine that ignores the force command types the byte "
            "instead, and those bytes move the head -- pick values outside "
            "0x71..0x82. `pipeline forces` sweeps two ranges that are safe."
        )
    if len(set(forces)) != len(forces):
        raise ValueError(f"duplicate strike force in {forces}")
    densities = resolve_densities(len(forces), force_densities)

    aspect = ec.cell_aspect(pitch)
    cell_h = cell_height + (cell_height % 2)  # chop_charset requires even cells
    cell_w = int(round(cell_h / aspect))
    cell_w += cell_w % 2
    actual_aspect = cell_h / cell_w
    name = name or f"sigma-{pitch}"

    glyphs = ec.all_glyphs(dead_keys=dead_keys)
    charset_dir = os.path.join(base_path, "charsets", name)
    os.makedirs(charset_dir, exist_ok=True)

    # (glyph, force) for every tile, in sheet order: the whole glyph set at the
    # first force, then again at the next. Force blocks are appended, so the
    # indices a single-force charset already had keep their meaning.
    force_list = list(forces) or [None]
    entries = [(g, f) for f in force_list for g in glyphs]

    layout = None
    if scan:
        sheet, cols, rows = _sheet_from_scan(
            scan, len(entries), sheet_cols, cell_w, cell_h
        )
        spills = [0.0] * len(entries)
        font_used = f"scan:{os.path.basename(scan)}"
    else:
        font_used = find_font(font)
        sheet, cols, rows, spills, layout = build_sheet(
            glyphs, font_used, cell_w, cell_h, bleed, sheet_cols,
            ink=ink, spread=spread, densities=densities,
        )

    image_name = "sigma.png"
    cv2.imwrite(os.path.join(charset_dir, image_name), (sheet * 255).astype(np.uint8))
    cv2.imwrite(
        os.path.join(charset_dir, "preview.png"),
        _labelled_preview(sheet, entries, cell_w, cell_h, cols),
    )

    config = {
        "charset_name": f"Sigma SM 8200i (pitch {pitch})",
        "image_path": image_name,
        "slicesX": cols,
        "slicesY": rows,
        "excludeChars": [],
        "whiteThreshold": WHITE_THRESHOLD,
        "blankSpace": True,
    }
    with open(os.path.join(charset_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    # Index 0 is the blank tile chop_charset prepends; glyph i lands at i + 1.
    mapping = {
        "charset_name": config["charset_name"],
        "pitch": pitch,
        "cell_width_px": cell_w,
        "cell_height_px": cell_h,
        "cell_width_mm": ec.PITCH_WIDTH_MM[pitch],
        "cell_height_mm": ec.LINE_HEIGHT_MM,
        "aspect": actual_aspect,
        "max_columns": ec.MAX_COLUMNS[pitch],
        "font": font_used,
        "bleed": bleed,
        "ink": ink,
        "spread": spread,
        "dead_keys": dead_keys,
        # The typing order, hardest first. planner.Charset reads it as the order
        # a line is typed in, and refuses a glyph whose force is not named here.
        "forces": list(forces),
        "force_densities": list(densities[1:]) if forces else [],
        "glyphs": [
            {"index": 0, **asdict(ec.BLANK), "name": "space", "force": None},
            *[
                {"index": i + 1, **asdict(g), "force": f}
                for i, (g, f) in enumerate(entries)
            ],
        ],
    }
    # encoding= is not optional: the glyph names carry non-ASCII characters
    # (£ § ° ä ...) and Python's default text encoding is the locale's, which on
    # Windows is cp1252. Without it this file is written in cp1252 while every
    # reader opens it as UTF-8, and loading dies on the first £.
    with open(os.path.join(charset_dir, "glyphs.json"), "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    _verify_mapping(charset_dir, base_path, expected=len(entries) + 1)

    worst = max(zip(spills, (g for g, _ in entries)),
                key=lambda sg: sg[0], default=(0.0, None))
    print(f"charset '{name}' -> {charset_dir}")
    if forces:
        print(f"  {len(glyphs)} glyphs x {len(forces)} strike forces "
              f"({', '.join(f'0x{f:02X}' for f in forces)}) + blank, "
              f"sheet {cols}x{rows} cells")
    else:
        print(f"  {len(glyphs)} glyphs + blank, sheet {cols}x{rows} cells")
    print(f"  cell {cell_w}x{cell_h}px  aspect {actual_aspect:.4f} "
          f"(ideal {aspect:.4f}, {abs(actual_aspect / aspect - 1) * 100:.2f}% off)")
    print(f"  glyph source: {font_used}")
    if scan:
        print("  ink measured from the scan")
    else:
        print(f"  ink modelled: densest grey {ink * 255:.0f}, spread {spread} px"
              + (f", lighter forces at {densities[1:]} of full" if forces else ""))
    if layout is not None:
        print(f"  type fills {layout.extent / layout.big_h * 100:.0f}% of the line height"
              + (f", scaled to {layout.shrunk * 100:.0f}% to fit" if layout.shrunk < 1 else ""))
    if worst[1] is not None and worst[0] > 0.005:
        print(f"  warning: {worst[0] * 100:.1f}% of {worst[1].char!r}'s ink falls "
              f"outside its cell")
    print("  verified against chop_charset(): index mapping is stable")
    return charset_dir


def _sheet_from_scan(scan_path, n_tiles, cols, cell_w, cell_h):
    """Slice a scanned charset sheet into the same cell grid.

    The charset sheet (see pipeline.py sheet) types the glyphs in GLYPHS order,
    `cols` per line, contiguously -- once per strike force if it was asked for
    more than one. Scan it square-on, crop to the outermost ink, and pass it
    here.

    Only the paper is normalised, and that is the point of this function. It used
    to stretch the 1st percentile to 0 as well, which took the darkest ink on the
    sheet and made it pure black -- destroying, in the one place that measures
    real ink, exactly the property the paper's section 5.5 says the tonal range
    depends on: "even the darkest typewriter characters are less than fully
    black, which yields a greater tonal range when characters are allowed to
    overlap". Scanner exposure needs correcting; ink density is the measurement.
    """
    im = cv2.imread(scan_path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(scan_path)
    rows = (n_tiles + cols - 1) // cols
    im = cv2.resize(im, (cols * cell_w, rows * cell_h), interpolation=cv2.INTER_AREA)
    # The sheet is mostly paper, so a high percentile *is* the paper -- a robust
    # white point that a stray bright speck cannot drag around.
    paper = float(np.percentile(im, 95))
    return np.clip(im.astype(np.float32) / max(paper, 1e-6), 0, 1), cols, rows


def parse_forces(text: str | None) -> tuple[int, ...]:
    """`--forces 0,3,6` -> (0, 3, 6). Accepts hex with 0x."""
    if not text:
        return ()
    return tuple(int(part, 0) for part in text.split(",") if part.strip())


def parse_densities(text: str | None) -> tuple[float, ...] | None:
    if not text:
        return None
    return tuple(float(part) for part in text.split(",") if part.strip())


def add_ink_args(p) -> None:
    """The ink model and the strike forces, shared with pipeline.py's parser."""
    p.add_argument("--ink", type=float, default=DEFAULT_INK,
                   help=f"grey the densest ink reaches, 0-1 (default {DEFAULT_INK}). "
                        "0 is pure black, which no ribbon produces -- see the module "
                        "docstring for why that matters more than it sounds")
    p.add_argument("--spread", type=float, default=DEFAULT_SPREAD,
                   help=f"ink point spread in cell pixels (default {DEFAULT_SPREAD})")
    p.add_argument("--forces", default=None,
                   help="strike forces to repeat the glyph set at, hardest first, "
                        "e.g. 0,3,6. Empty (the default) builds a single-force "
                        "charset. Run `pipeline forces` first: nothing here is "
                        "confirmed on hardware yet")
    p.add_argument("--force-density", default=None,
                   help="ink each lighter force transfers, as a fraction of the "
                        "first, e.g. 0.55,0.3. Font path only -- a scan measures "
                        f"it. Default {','.join(str(d) for d in DEFAULT_FORCE_DENSITY)}"
                        " truncated to fit")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--name", "-N", default=None, help="charset folder name (default sigma-<pitch>)")
    p.add_argument("--pitch", "-p", type=int, default=10, choices=(10, 12),
                   help="Schriftteilung: 10 or 12 characters per inch (default 10)")
    p.add_argument("--cell-height", type=int, default=40,
                   help="cell height in pixels; width follows from the pitch (default 40)")
    p.add_argument("--font", "-f", default=None, help="path to a monospace TTF")
    p.add_argument("--bleed", "-b", type=float, default=0.2,
                   help="ink spread in cell pixels, approximating ribbon bleed (default 0.2)")
    p.add_argument("--dead-keys", action="store_true",
                   help="include the four non-advancing accent keys (¨ ^ ´ `)")
    p.add_argument("--sheet-cols", type=int, default=20, help="cells per row on the sheet")
    p.add_argument("--from-scan", default=None,
                   help="build from a scanned charset sheet instead of a font")
    add_ink_args(p)
    a = p.parse_args(argv)
    make_charset(
        name=a.name, pitch=a.pitch, cell_height=a.cell_height, font=a.font,
        bleed=a.bleed, dead_keys=a.dead_keys, sheet_cols=a.sheet_cols, scan=a.from_scan,
        ink=a.ink, spread=a.spread, forces=parse_forces(a.forces),
        force_densities=parse_densities(a.force_density),
    )


if __name__ == "__main__":
    main()
