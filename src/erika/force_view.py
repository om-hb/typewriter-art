"""One glyph, every strike force, side by side -- the check the contact sheet cannot make.

The labelled contact sheet shows a scan sliced out of step, because every glyph
then sits against its neighbour's label. It does not show a grid that is a
*little* out of step, and on a multi-force sheet that is the failure worth
catching: the force blocks are typed in order down the page, so a row pitch that
is a fraction short lands hardest on the last block, which is the lightest one
and the one with the least ink to spare. The tiles still look like their
characters. They are simply cut a little lower each time.

Put the same glyph's four strikes in a row and it is obvious in a second. A
period that sits at the same height in every column is a grid that holds; one
that sinks column by column is one that drifts. That was a real bug, found this
way and not by the contact sheet, and this exists so the next one is found the
same way rather than on paper.

Only written for a charset with more than one force, because with one force
there is nothing to compare.
"""

from __future__ import annotations

import cv2
import numpy as np

#: How many glyphs to show, spread from the lightest mark on the wheel to the
#: heaviest. Enough that a drift is unmistakable, few enough to take in at once.
ROWS_SHOWN = 8

#: Pixels per tile pixel. The tiles are two dozen pixels across; a reader needs
#: to see where the ink sits inside the cell, not merely that there is some.
SCALE = 5

LABEL_W = 116
HEAD_H = 26
GAP = 8


def _pick(sheet, entries, cols, cell_w, cell_h, block, count):
    """Which glyphs to show: a spread from least ink to most, at full force.

    Chosen from the sheet rather than named here, because the interesting ones
    differ by wheel -- and because the glyphs that expose a drifting grid are the
    small ones that sit high or low in the cell, which is exactly what the light
    end of this spread picks up.
    """
    ink = []
    for i in range(min(block, len(entries))):
        r, c = divmod(i, cols)
        tile = sheet[r * cell_h: (r + 1) * cell_h, c * cell_w: (c + 1) * cell_w]
        ink.append(1.0 - float(tile.mean()))
    order = np.argsort(np.array(ink))
    if len(order) <= count:
        return list(order)
    # Evenly spaced through the ranking, ends included.
    picks = np.linspace(0, len(order) - 1, count).round().astype(int)
    return [int(order[p]) for p in dict.fromkeys(picks)]


def build(sheet, entries, cols, cell_w, cell_h, forces, count=ROWS_SHOWN):
    """The montage, or None when the charset has only one strike force."""
    if len(forces) < 2:
        return None
    block = len(entries) // len(forces)
    if block <= 0:
        return None

    chosen = _pick(sheet, entries, cols, cell_w, cell_h, block, count)
    tile_w, tile_h = cell_w * SCALE, cell_h * SCALE
    width = LABEL_W + len(forces) * (tile_w + GAP) + GAP
    height = HEAD_H + len(chosen) * (tile_h + GAP) + GAP
    out = np.full((height, width), 245, dtype=np.uint8)

    for j, force in enumerate(forces):
        x = LABEL_W + j * (tile_w + GAP)
        cv2.putText(out, f"force {force}" if force is not None else "one force",
                    (x, 18), cv2.FONT_HERSHEY_PLAIN, 1.0, 60, 1, cv2.LINE_AA)

    for i, glyph_index in enumerate(chosen):
        y = HEAD_H + i * (tile_h + GAP)
        glyph = entries[glyph_index][0]
        name = glyph.name or repr(glyph.char)
        cv2.putText(out, f"{name}  #{glyph_index + 1}", (6, y + tile_h // 2),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, 60, 1, cv2.LINE_AA)
        for j in range(len(forces)):
            index = j * block + glyph_index
            if index >= len(entries):
                continue
            r, c = divmod(index, cols)
            tile = sheet[r * cell_h: (r + 1) * cell_h, c * cell_w: (c + 1) * cell_w]
            x = LABEL_W + j * (tile_w + GAP)
            out[y: y + tile_h, x: x + tile_w] = cv2.resize(
                (tile * 255).astype(np.uint8), (tile_w, tile_h),
                interpolation=cv2.INTER_NEAREST,
            )
            cv2.rectangle(out, (x - 1, y - 1), (x + tile_w, y + tile_h), 175, 1)
    return out


def drift(sheet, entries, cols, cell_w, cell_h, forces):
    """How far a glyph's ink moves up or down between the two heaviest forces.

    The number behind the picture, in tile pixels. Those two blocks are the ones
    where every glyph is fully formed, so what is left is registration rather
    than a light strike losing part of a letter. On the sheets this was built
    against it was 2.8px of a 40px cell with the grid taken from the ink, and
    1.0px with it taken from the marks.
    """
    if len(forces) < 2:
        return None
    block = len(entries) // len(forces)
    rows_y = np.arange(cell_h, dtype="float64")

    def centre(index):
        r, c = divmod(index, cols)
        tile = 1.0 - sheet[r * cell_h: (r + 1) * cell_h, c * cell_w: (c + 1) * cell_w]
        total = tile.sum()
        if total < 1e-3:
            return None
        return float((tile.sum(axis=1) * rows_y).sum() / total)

    moved = []
    for i in range(block):
        a, b = centre(i), centre(block + i)
        if a is not None and b is not None:
            moved.append(abs(a - b))
    return float(np.median(moved)) if moved else None
