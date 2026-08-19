"""Optimised cropping: line the photograph up with the character grid.

Characters can only land on a fixed grid, so where the picture sits relative to
that grid decides how well an edge in the photograph can be matched by an edge in
the type. The paper's section 3.4 searches for the best placement by trying 64
slightly different crops and scoring each with a quick approximation:

    x = -a x charWidth
    y = -b x charHeight
    s = (c + n) / n          n = characters per row
    a, b, c in {0, 0.25, 0.5, 0.75}

Section 5.3.1 shows it sharpening a checkerboard dramatically and a portrait
visibly, and calls alignment "one of the dominant factors in the quality of the
result" at small image sizes.

Measured here it is real but much smaller than that: at 20 columns SSIM goes from
0.444 to 0.457, at 40 from 0.454 to 0.464, and RMSE does not move at all. It stays
small with a good charset too (0.313 to 0.330 with strike forces). The reason is
worth knowing before reaching for this: alignment buys *shape* matching, and on
this machine shape matching is not the binding constraint -- ink is. The paper's
dominant-factor claim is about a checkerboard test pattern and a 20-column
portrait, where there is nothing else left to get wrong.

So it is off by default, and it is a pre-pass rather than a change to the search:
it picks a crop, writes the cropped target, and hands the path on. Nothing
downstream knows it happened. What it buys, measured on the sample photograph
with the charset as it ships:

    20 columns, 4x1   SSIM 0.388 -> 0.402   RMSE 65.2 -> 64.0    12 s
    20 columns, 4x2   SSIM 0.407 -> 0.419   RMSE 52.4 -> 51.4    24 s
    40 columns, 4x1   SSIM 0.372 -> 0.378   RMSE 66.5 -> 65.9    42 s
    40 columns, 4x2   SSIM 0.384 -> 0.389   RMSE 52.9 -> 52.4    85 s

Nothing regresses, the gain is larger at 20 columns than at 40 -- which is the
one part of the paper's claim that does carry over -- and the search can cost
several times the optimizer run it precedes. It scales with the charset, so a set
carrying three strike forces costs about three times these numbers.

One consequence of the layer schemes is worth knowing before choosing
``steps``. ``4x1`` places layers at 0 and 0.5 of a cell, so the placements are
already half-cell periodic and a half-cell shift is close to a relabelling rather
than a better fit. The effective period is therefore half a cell, the paper's
quarters sample it twice over, and ``steps`` below 3 probes little but the scale
and the border.

**Two greedy cycles per candidate, not the paper's one.** This is the one
deliberate departure and it is not a refinement, it is what makes the search work
at all. A single cycle's score carries enough run-to-run noise to swamp the
difference between crops: measured at 40 columns, the noise range across seeds is
0.20 against a spread of 0.32 across all 64 candidates, and the search *disagrees
with itself* -- three runs of the same search pick two different crops, on scores
0.001 apart. A second cycle re-evaluates every cell against a settled background
and drops the noise range to 0.06, which is where three runs agree. The noise is
irreducible from Python: the inner loop draws candidate order from numba's
per-thread RNG, and greedy selection is order-dependent whenever two glyphs tie.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from erika import SRC_DIR

#: Offsets and scales tried in each dimension. The paper's quarters.
DEFAULT_STEPS = 4

#: Greedy cycles per candidate. See the module docstring: one is the paper's and
#: is too noisy at the resolutions this pipeline prints at.
DEFAULT_LOOPS = 2

#: Where the chosen crop is written, relative to src/. Derived and disposable,
#: and a fixed name so a run does not leave a trail of them behind -- the same
#: arrangement as pipeline.GREY_TARGET, which this usually reads.
ALIGNED_TARGET = "results/target-aligned.png"


@dataclass(frozen=True)
class Crop:
    """One member of the search family, in the paper's own parameters."""

    a: float  #: horizontal shift, in character widths
    b: float  #: vertical shift, in character heights
    c: float  #: scale, as (c + row_length) / row_length

    @property
    def is_identity(self) -> bool:
        return self.a == 0 and self.b == 0 and self.c == 0

    def scale(self, row_length: int) -> float:
        return (self.c + row_length) / row_length

    def __str__(self) -> str:
        return f"a={self.a} b={self.b} c={self.c}"


@dataclass
class Result:
    """What the search chose, and what it cost."""

    crop: Crop
    score: float
    identity_score: float
    candidates: int
    seconds: float

    @property
    def gain(self) -> float:
        return self.score - self.identity_score


def transform(
    image: np.ndarray,
    crop: Crop,
    row_length: int,
    cell: tuple[int, int],
    aspect_change: tuple[float, float],
) -> np.ndarray:
    """Apply one crop to a source image, keeping its size.

    The paper's parameters are in character cells, which live in the *resized*
    frame; this warps the source instead, so the picture is resampled once rather
    than twice. The conversion is exact because ``resizeTarget`` scales by a
    constant in each axis, and a uniform scale commutes with that:

        out = diag(sx, sy) . src, and shifting src by (dx/sx, dy/sy)
        shifts out by exactly (dx, dy)

    An identity crop is a genuine no-op -- warpAffine with an integer translation
    and unit scale copies pixel for pixel -- so a search that finds nothing costs
    the picture nothing.

    Enlarging (every ``c`` above 0) pushes content past the frame, which is what
    makes this a *crop*; the vacated edge is filled with paper, as
    ``resizeTarget`` pads with paper for the same reason.
    """
    cell_h, cell_w = cell
    x_change, y_change = aspect_change
    height, width = image.shape[:2]

    # The scaling resizeTarget will apply, so the shift can be expressed in the
    # frame the paper's parameters are written in.
    out_w = row_length * cell_w
    out_h = round((out_w / width) * height * (x_change / y_change))
    scale_x = out_w / width
    scale_y = out_h / height

    s = crop.scale(row_length)
    matrix = np.array(
        [[s, 0.0, -crop.a * cell_w / scale_x],
         [0.0, s, -crop.b * cell_h / scale_y]],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        image, matrix, (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM over the image, gaussian 11x11 sigma 1.5.

    Written out rather than imported: the metric is four gaussian blurs, and
    scikit-image is not a dependency of this fork -- the parallel rewrite of the
    optimizer dropped the only other thing that used it.
    """
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    kernel, sigma = (11, 11), 1.5
    mu_a = cv2.GaussianBlur(a, kernel, sigma)
    mu_b = cv2.GaussianBlur(b, kernel, sigma)
    var_a = cv2.GaussianBlur(a * a, kernel, sigma) - mu_a ** 2
    var_b = cv2.GaussianBlur(b * b, kernel, sigma) - mu_b ** 2
    cov = cv2.GaussianBlur(a * b, kernel, sigma) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return float(np.mean(num / den))


def quality(mockup: np.ndarray, target: np.ndarray) -> float:
    """The paper's crop score: SSIM x 4 + PSNR.

    Two metrics because they disagree in a useful way -- section 5.1 finds SSIM
    "working much like an edge detector with little respect for tone matching" --
    and alignment is the one decision where the edge detector is the point.
    """
    mse = float(np.mean((mockup - target) ** 2))
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 99.0
    return ssim(mockup, target) * 4 + psnr


def _greedy_cycles(chars, target, layer_offsets, loops, asymmetry):
    """Composite `loops` greedy passes over every layer, from a blank page.

    A reimplementation of the middle of ``optimize.kword`` -- deliberately, and
    this is the one duplication in this package that is not guarded by comparing
    two copies. Calling ``kword`` itself would mean 64 rounds of writing
    ``choices.json``, four layer PNGs and a matplotlib figure, and would clobber
    the results the real run is about to write.

    What makes the duplication tolerable is what it is used for: this scores
    candidates *relative to each other*. A divergence from upstream changes the
    ranking slightly, not the correctness of anything printed. The hazard that
    would matter -- getting the offsets or the indices wrong -- is caught by
    ``test_align_composites_the_way_the_optimizer_does``, which recomposites the
    returned choices independently and requires the same mockup.
    """
    from utils import layer_optimization_pass

    cell_h, cell_w = chars.shape[1], chars.shape[2]
    num_rows = target.shape[0] // cell_h
    num_cols = target.shape[1] // cell_w
    pad = ((0, max(o[0] for o in layer_offsets)),
           (0, max(o[1] for o in layer_offsets)))
    padded = np.pad(target, pad, "constant", constant_values=1).astype("float32")

    layers = np.array(
        [np.ones_like(padded) for _ in layer_offsets], dtype="float32"
    )
    mockup = np.ones_like(padded)
    choices = np.zeros((len(layer_offsets), num_rows * num_cols), dtype="uint16")
    blank = np.ones_like(mockup) if len(layer_offsets) == 1 else None

    def paint(layer_num, offset):
        for i, choice in enumerate(choices[layer_num]):
            row, col = divmod(i, num_cols)
            layers[layer_num][
                row * cell_h + offset[0]:(row + 1) * cell_h + offset[0],
                col * cell_w + offset[1]:(col + 1) * cell_w + offset[1],
            ] = chars[choice]

    for _ in range(loops):
        for layer_num, offset in enumerate(layer_offsets):
            # "Every other layer", composited the same way kword does it, and
            # with the fork's own fix for a single layer having no others.
            if blank is not None:
                background = blank
            else:
                background = layers[(layer_num + 1) % len(layer_offsets)]
                for i in range(2, len(layer_offsets)):
                    background = background * layers[(layer_num + i) % len(layer_offsets)]
            choices[layer_num], mockup, _, _ = layer_optimization_pass(
                background, mockup, padded, chars, choices[layer_num],
                np.array(offset), asymmetry=asymmetry, mode="greedy",
                temperature=0.0,
            )
            paint(layer_num, offset)

    return mockup, choices, padded


def candidates(steps: int = DEFAULT_STEPS) -> list[Crop]:
    """The search family, identity first.

    Identity first so the report can say what alignment actually bought, and so a
    tie leaves the picture untouched rather than resampled for nothing.
    """
    if steps < 1:
        raise ValueError(f"--align-steps must be at least 1, got {steps}")
    grid = [i / steps for i in range(steps)]
    crops = [Crop(a, b, c) for a, b, c in itertools.product(grid, repeat=3)]
    crops.sort(key=lambda crop: not crop.is_identity)
    return crops


def search(
    target_path: str,
    charset: str,
    row_length: int,
    layers: str,
    steps: int = DEFAULT_STEPS,
    loops: int = DEFAULT_LOOPS,
    asymmetry: float = 0.1,
    base_path: str | None = None,
    progress=None,
) -> tuple[Result, np.ndarray]:
    """Score every crop and return the best, with the image it produced.

    ``target_path`` is resolved the way ``kword`` resolves it -- against
    ``base_path`` -- so the same string that would have been handed to the
    optimizer can be handed here.
    """
    from utils import prep_charset, resizeTarget

    base_path = base_path or SRC_DIR
    path = target_path if os.path.isabs(target_path) else os.path.join(base_path, target_path)
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)

    chars, x_change, y_change = prep_charset(charset, base_path)
    cell = (chars.shape[1], chars.shape[2])
    with open(os.path.join(base_path, "layers.json"), encoding="utf-8") as f:
        fractional = json.load(f)[layers]
    layer_offsets = [
        (int(cell[0] * o[0]), int(cell[1] * o[1])) for o in fractional
    ]

    started = time.perf_counter()
    best: tuple[float, Crop, np.ndarray] | None = None
    identity_score = float("nan")
    family = candidates(steps)
    for index, crop in enumerate(family):
        warped = transform(image, crop, row_length, cell, (x_change, y_change))
        prepared, _ = resizeTarget(warped, row_length, cell, (x_change, y_change))
        mockup, _, padded = _greedy_cycles(
            chars, prepared.astype("float32") / 255, layer_offsets, loops, asymmetry
        )
        score = quality(mockup, padded)
        if crop.is_identity:
            identity_score = score
        if best is None or score > best[0]:
            best = (score, crop, warped)
        if progress is not None:
            progress(index + 1, len(family), crop, score)

    score, crop, warped = best
    result = Result(
        crop=crop,
        score=score,
        identity_score=identity_score,
        candidates=len(family),
        seconds=time.perf_counter() - started,
    )
    return result, warped


def apply_to_file(
    target_path: str,
    charset: str,
    row_length: int,
    layers: str,
    steps: int = DEFAULT_STEPS,
    loops: int = DEFAULT_LOOPS,
    asymmetry: float = 0.1,
    base_path: str | None = None,
    destination: str = ALIGNED_TARGET,
    progress=None,
) -> tuple[Result, str]:
    """Search, write the chosen crop, and return the path to hand the optimizer.

    The file is written even when identity wins. It costs one PNG and keeps the
    caller's contract simple -- and because an identity warp is pixel for pixel,
    what lands on disk is the picture that came in.
    """
    base_path = base_path or SRC_DIR
    source = target_path if os.path.isabs(target_path) else os.path.join(base_path, target_path)
    out = destination if os.path.isabs(destination) else os.path.join(base_path, destination)
    # Both have to exist before samefile can be asked -- it stats them, so a
    # destination that is not there yet raises rather than answering False. Which
    # is the normal case on a first run.
    if os.path.exists(source) and os.path.exists(out) and os.path.samefile(source, out):
        # Aligning an already-aligned target is not idempotent -- each pass
        # resamples, and the score carries noise -- so silently replacing the
        # input with a second-generation copy is the wrong thing to do quietly.
        raise ValueError(
            f"--align would overwrite its own input ({destination}). Point it at "
            "the original picture; the crop is chosen from scratch each time and "
            "aligning twice resamples for nothing."
        )

    result, warped = search(
        target_path, charset, row_length, layers, steps=steps, loops=loops,
        asymmetry=asymmetry, base_path=base_path, progress=progress,
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    cv2.imwrite(out, warped)
    return result, destination
