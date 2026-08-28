"""Straighten a scanned sheet before anything measures a grid on it.

Both sheets this pipeline reads back -- the charset sheet and the strike-force
probe -- are sliced into an *axis-aligned* grid. The registration marks fix
where that grid starts, which is a different problem from which way it points,
and nothing was fixing the second one. A scan a degree off builds, verifies and
prints, and every tile is contaminated by its neighbour by an amount that grows
toward the corners of the sheet: measured on a synthetic charset sheet with
realistic ink extents, a degree of rotation moves the worst tile by 27 grey
levels out of 255, and one and a half degrees by 42. Two degrees is the first
angle that *fails* -- ``chop_charset`` drops tiles as blank and the mapping
check refuses the build -- so the dangerous band is the shallow one, where the
charset comes out looking entirely reasonable and every tone in it is wrong.

The angle is found from the ink itself rather than from the marks. The marks
would give it -- two of them per row, and the line through one column of them is
the skew -- but finding a mark on a rotated sheet means first solving the
problem the marks are being asked to solve: ``_find_sheet_marks`` identifies them
by projecting the whole image onto one axis, and rotation is precisely what
smears that projection. So this maximises the sharpness of the projection
instead, which is the same measurement one step earlier and needs nothing
identified first. A typed sheet is a grid of ink with paper between the rows and
between the columns, and at the true angle both projections are at their most
corrugated; off it, every row smears into the next.

Scored on both axes together because either alone has a way to be fooled: a
sheet of one very dark row scores well on columns at any angle, and a single
column of marks scores well on rows. A typed sheet has structure in both, and
demanding both is what makes the maximum sharp.
"""

from __future__ import annotations

import numpy as np

#: How far from square a scan may be and still be straightened. Wider than any
#: plausible flatbed slip, and narrow enough that the search cannot wander off
#: to an angle that happens to line the glyph *diagonals* up.
DESKEW_LIMIT_DEG = 5.0

#: Two passes: every quarter degree across the range, then every fiftieth of one
#: around the winner. The fine pass is what the tolerance actually needs -- a
#: quarter of a degree is already 5 grey levels on the worst tile.
COARSE_STEP_DEG = 0.25
FINE_STEP_DEG = 0.02

#: The search runs on a copy no larger than this on its long side. The angle is a
#: property of the ink's arrangement rather than of its detail, and a 300 dpi A4
#: scan is 30 megapixels rotated forty times otherwise.
SEARCH_MAX_PX = 900

#: Below this the image has no grid to speak of and the angle is noise. Sharpness
#: is a normalised variance, so it is comparable between images.
MIN_STRUCTURE = 0.02

#: Angles smaller than this are left alone, and the scan is handed on untouched.
#:
#: Rotating costs a resample, and a resample costs half a pixel of blur on every
#: edge in the image whatever the angle is. On a sheet whose rows are two dozen
#: pixels tall that is a few percent of a row's ink -- which is more than a
#: hundredth of a degree of skew was ever going to take away. So the correction
#: has to be worth more than the resample that applies it, and below a twentieth
#: of a degree it is not: a quarter of a degree moves the worst tile of a charset
#: by five grey levels out of 255, so a twentieth moves it by about one.
MIN_ANGLE_DEG = 0.05


def _paper(im: np.ndarray) -> float:
    return float(np.percentile(im, 95))


def _sharpness(im: np.ndarray, paper: float) -> float:
    """How corrugated the two ink projections are, scale-free.

    Measured on the middle of the image. A rotation fills the corners with
    whatever border colour it was given, and while that is paper here, the
    *shape* of the filled region changes with the angle -- scoring it would be
    scoring the border rather than the sheet.
    """
    height, width = im.shape
    view = im[height // 10: height - height // 10, width // 10: width - width // 10]
    if view.size == 0:
        return 0.0
    ink = np.clip(paper - view.astype(np.float32), 0.0, None)

    total = 0.0
    for axis in (0, 1):
        profile = ink.sum(axis=axis)
        mean = float(profile.mean())
        if mean <= 1e-6:
            return 0.0
        total += float(profile.var()) / (mean * mean)
    return total


def _rotate(im: np.ndarray, angle: float, fill: float) -> np.ndarray:
    import cv2  # noqa: PLC0415

    if angle == 0.0:
        return im
    height, width = im.shape
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    return cv2.warpAffine(
        im, matrix, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(fill),
    )


def find_angle(im: np.ndarray, limit: float = DESKEW_LIMIT_DEG) -> float:
    """The rotation that would square this sheet up, in degrees.

    Positive means the sheet is turned anticlockwise on the scanner and wants
    turning back. Returns 0.0 when the image has no grid structure to measure,
    which is the honest answer and leaves the scan untouched.
    """
    import cv2  # noqa: PLC0415

    small = im
    longest = max(im.shape)
    if longest > SEARCH_MAX_PX:
        scale = SEARCH_MAX_PX / longest
        small = cv2.resize(im, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
    paper = _paper(small)
    if _sharpness(small, paper) < MIN_STRUCTURE:
        return 0.0

    def best_of(angles):
        return max(angles, key=lambda a: _sharpness(_rotate(small, a, paper), paper))

    coarse = np.arange(-limit, limit + COARSE_STEP_DEG / 2, COARSE_STEP_DEG)
    around = float(best_of(coarse))
    fine = np.arange(around - COARSE_STEP_DEG, around + COARSE_STEP_DEG + FINE_STEP_DEG / 2,
                     FINE_STEP_DEG)
    return float(best_of(fine))


def straighten(
    im: np.ndarray, limit: float = DESKEW_LIMIT_DEG
) -> tuple[np.ndarray, float, str | None]:
    """Square a scan up. Returns the image, the angle applied, and a warning.

    The warning is not an error and the image is straightened anyway: an angle
    at the edge of the search means the true one may be outside it, and the
    caller is the one that can say so where a person will read it.
    """
    angle = find_angle(im, limit)
    if abs(angle) < MIN_ANGLE_DEG:
        return im, 0.0, None

    note = None
    if abs(angle) >= limit - COARSE_STEP_DEG:
        note = (
            f"the scan is {angle:+.2f} deg off square, which is the edge of what "
            f"this looks for -- it may be further. Re-scan it against the "
            f"platen's registration corner rather than trusting this."
        )
    return _rotate(im, angle, _paper(im)), angle, note
