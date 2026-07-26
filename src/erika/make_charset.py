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
which is an approximation of the Sigma's own typeface. It is good enough for
tone -- the optimizer only cares about per-cell ink coverage. For a faithful
result, type the calibration sheet (`pipeline.py calibrate`), scan it, and
build the charset from the scan with `--from-scan`.
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

    def render(self, glyph: ec.Glyph, bleed: float) -> tuple[np.ndarray, float]:
        """Render one glyph to a float32 cell in [0, 1]; 1.0 == bare paper."""
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
        return small.astype(np.float32) / 255.0, spill

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
) -> tuple[np.ndarray, int, int, list[float], CellLayout]:
    layout = CellLayout(font_path, cell_w, cell_h, glyphs)
    rows = (len(glyphs) + cols - 1) // cols
    sheet = np.ones((rows * cell_h, cols * cell_w), dtype=np.float32)
    spills = []
    for i, glyph in enumerate(glyphs):
        tile, spill = layout.render(glyph, bleed)
        spills.append(spill)
        r, c = divmod(i, cols)
        sheet[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w] = tile
    return sheet, cols, rows, spills, layout


def _labelled_preview(sheet: np.ndarray, glyphs, cell_w, cell_h, cols) -> np.ndarray:
    """Contact sheet with index/char/code annotations, scaled up for reading."""
    scale = max(2, 96 // cell_w)
    label_h = 34
    rows = (len(glyphs) + cols - 1) // cols
    out_w = cols * cell_w * scale
    out_h = rows * (cell_h * scale + label_h)
    out = np.full((out_h, out_w), 255, dtype=np.uint8)
    for i, glyph in enumerate(glyphs):
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
        cv2.putText(
            out,
            f"{i + 1}:{glyph.code:02X}",
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
) -> str:
    base_path = base_path or SRC_DIR
    if pitch not in ec.PITCH_WIDTH_MM:
        raise ValueError(f"pitch must be 10 or 12, got {pitch}")

    aspect = ec.cell_aspect(pitch)
    cell_h = cell_height + (cell_height % 2)  # chop_charset requires even cells
    cell_w = int(round(cell_h / aspect))
    cell_w += cell_w % 2
    actual_aspect = cell_h / cell_w
    name = name or f"sigma-{pitch}"

    glyphs = ec.all_glyphs(dead_keys=dead_keys)
    charset_dir = os.path.join(base_path, "charsets", name)
    os.makedirs(charset_dir, exist_ok=True)

    layout = None
    if scan:
        sheet, cols, rows = _sheet_from_scan(scan, len(glyphs), sheet_cols, cell_w, cell_h)
        spills = [0.0] * len(glyphs)
        font_used = f"scan:{os.path.basename(scan)}"
    else:
        font_used = find_font(font)
        sheet, cols, rows, spills, layout = build_sheet(
            glyphs, font_used, cell_w, cell_h, bleed, sheet_cols
        )

    image_name = "sigma.png"
    cv2.imwrite(os.path.join(charset_dir, image_name), (sheet * 255).astype(np.uint8))
    cv2.imwrite(
        os.path.join(charset_dir, "preview.png"),
        _labelled_preview(sheet, glyphs, cell_w, cell_h, cols),
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
    with open(os.path.join(charset_dir, "config.json"), "w") as f:
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
        "dead_keys": dead_keys,
        "glyphs": [
            {"index": 0, **asdict(ec.BLANK), "name": "space"},
            *[{"index": i + 1, **asdict(g)} for i, g in enumerate(glyphs)],
        ],
    }
    with open(os.path.join(charset_dir, "glyphs.json"), "w") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    _verify_mapping(charset_dir, base_path, expected=len(glyphs) + 1)

    worst = max(zip(spills, glyphs), key=lambda sg: sg[0], default=(0.0, None))
    print(f"charset '{name}' -> {charset_dir}")
    print(f"  {len(glyphs)} glyphs + blank, sheet {cols}x{rows} cells")
    print(f"  cell {cell_w}x{cell_h}px  aspect {actual_aspect:.4f} "
          f"(ideal {aspect:.4f}, {abs(actual_aspect / aspect - 1) * 100:.2f}% off)")
    print(f"  glyph source: {font_used}")
    if layout is not None:
        print(f"  type fills {layout.extent / layout.big_h * 100:.0f}% of the line height"
              + (f", scaled to {layout.shrunk * 100:.0f}% to fit" if layout.shrunk < 1 else ""))
    if worst[1] is not None and worst[0] > 0.005:
        print(f"  warning: {worst[0] * 100:.1f}% of {worst[1].char!r}'s ink falls "
              f"outside its cell")
    print("  verified against chop_charset(): index mapping is stable")
    return charset_dir


def _sheet_from_scan(scan_path, n_glyphs, cols, cell_w, cell_h):
    """Slice a scanned calibration sheet into the same cell grid.

    The calibration sheet (see pipeline.py calibrate) types the glyphs in
    GLYPHS order, `cols` per line, one line per row. Scan it square-on, crop
    to the outermost ink, and pass it here.
    """
    im = cv2.imread(scan_path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(scan_path)
    rows = (n_glyphs + cols - 1) // cols
    im = cv2.resize(im, (cols * cell_w, rows * cell_h), interpolation=cv2.INTER_AREA)
    # Normalise paper to white and the darkest ink to black.
    lo, hi = np.percentile(im, 1), np.percentile(im, 99)
    im = np.clip((im.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    return im, cols, rows


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
                   help="build from a scanned calibration sheet instead of a font")
    a = p.parse_args(argv)
    make_charset(
        name=a.name, pitch=a.pitch, cell_height=a.cell_height, font=a.font,
        bleed=a.bleed, dead_keys=a.dead_keys, sheet_cols=a.sheet_cols, scan=a.from_scan,
    )


if __name__ == "__main__":
    main()
