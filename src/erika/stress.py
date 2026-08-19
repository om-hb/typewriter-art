"""Colour to black and white, the step before the optimizer sees anything.

The paper this repository implements says the quiet part in section 5.3: "since
many source images are colour, the black and white conversion method has a
substantial effect on the result", and then declares it out of scope. It is not
out of scope here, because this machine makes the choice matter far more than a
printer does. A photograph arrives spanning 0..255; the Sigma answers with some
fifty distinguishable greys per cell. Whatever the conversion throws away is
thrown away before the only stage that could have spent the range well.

Rec. 601 luma -- what ``cv2.IMREAD_GRAYSCALE`` does, and the default here -- is a
*global* map: every pixel of the same colour becomes the same grey, wherever it
sits in the picture. So two adjacent areas that differ in hue but not in
luminance merge into one flat tone, and on this machine a flat tone is a region
with no characters in it at all. A red sign on green foliage is the standard
example; both come out at about grey 76.

STRESS answers that by measuring each pixel against its *own surroundings*
instead of against a fixed formula. Around every pixel it throws a random spray
of sample points, takes the local minimum and maximum, and reports where the
pixel sits between them. Colour differences that no global formula would keep
survive as tonal differences, because locally there is nothing else for the
range to be spent on.

    Kolås, Farup and Rizzi, "STRESS: A framework for spatial color algorithms",
    Journal of Imaging Science and Technology 55:040503, 2011 -- reference [12]
    of the typewriter-art paper, cited exactly for this.

Two operations come out of the one framework, and both are here because they
answer different complaints:

``c2g``
    Colour to grey (GEGL's ``gegl:c2g``): the pixel projected onto the axis
    between the local envelopes, which is the decolorization the paper's
    Figure 12 shows against an RGB average.
``stress``
    The local enhancement (GEGL's ``gegl:stress``): each channel rescaled by its
    own envelope, which is a retinex-like white balance and tone map, converted
    to grey by luma afterwards.

Reimplemented rather than shelled out to GEGL or GIMP. This pipeline is
developed on three machines with different toolchains and already asks for a
Python 3.10, numba and a PlatformIO; a native image-processing stack that has to
be found, versioned and matched on each of them would cost more than 200 lines
of numpy. The port is faithful to GEGL's ``envelopes.h`` in what it computes and
deliberately different in three places, each noted where it happens.

**The kernel is deliberately bit-reproducible**, which the optimizer it feeds is
not. A run of ``kword`` draws from numba's per-thread RNG and cannot be pinned
from Python, so two runs of it differ; a conversion should not add a second
source of that, because a photograph is usually converted several times while
its tone is being settled and the comparison is only worth anything if the
conversion is the same one each time. Three things buy it: the sampling is
seeded per pixel rather than from a stream the threads share, so ``prange``
cannot reorder the result; the spray is drawn with multiplications and a square
root and no trigonometry or ``pow``, all of them exactly rounded in IEEE 754, so
the answer does not depend on whose libm is installed -- this project is
developed on three machines; and there is no ``fastmath`` here although
``utils.py`` uses it two files away, because it licenses exactly the
reassociation that would undo the other two.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

#: The conversions this module offers, in the order they are worth trying.
METHODS = ("luma", "average", "c2g", "stress")

#: Sample points per iteration, and iterations per pixel.
#:
#: Both trade the same thing -- more of either means less sampling noise and a
#: proportionally longer run -- so the samples stay at GEGL's default and the
#: iterations are the knob. GEGL's default of 5 is aimed at a live preview in an
#: image editor and is far too noisy here; its own reference composition uses 30.
#:
#: 20 is where the noise stops mattering to *this* machine, measured on
#: ``images/mwdog_crop.png`` as the difference between two seeds. Per pixel that
#: is still 9 grey levels, which sounds fatal and is not: the optimizer averages
#: each cell down to one figure, and per cell -- on the 40-column grid, so about
#: a hundred pixels each -- the same difference is 0.87 mean and 2.3 at the 95th
#: percentile, against a gamut whose own steps are 4.5 apart. So the spray's
#: noise lands below the machine's own quantisation. Doubling to 40 halves it
#: again for twice the time, and buys nothing anyone can print.
DEFAULT_SAMPLES = 5
DEFAULT_ITERATIONS = 20

#: Spray radius, as a fraction of the picture's longest side.
#:
#: GEGL asks for pixels and advises "close to the longest side of the image",
#: which is the same number written in the units that make it depend on the
#: resolution. A fraction is what keeps a preview and an export agreeing: the
#: browser renders the picture at two sizes and both have to look the same, and
#: a radius in pixels would silently mean two different neighbourhoods.
DEFAULT_RADIUS = 1.0

#: Seed for the spray. Fixed rather than drawn from the clock so a run can be
#: repeated: the sampling is stochastic and two seeds differ by a fraction of a
#: grey level, but "a fraction" is not "nothing" when a job takes half an hour.
DEFAULT_SEED = 1

#: Attempts to land a sample inside the unit disc before giving up on the last
#: pair drawn. Rejection sampling accepts pi/4 of the time, so eight failures in
#: a row has probability 1e-5; the bound exists only so the loop is provably
#: finite -- and a bound rather than a retry loop without one, so that the number
#: of draws a pixel makes cannot depend on the run.
_DISC_ATTEMPTS = 8


# ---------------------------------------------------------------------------
# sRGB <-> linear light
# ---------------------------------------------------------------------------
#
# The envelopes are computed in linear light, as GEGL does (its buffers are
# "RGBA float", which babl defines as linear), and the answer is encoded back to
# sRGB before anything downstream sees it. Both halves matter. Linear is where a
# minimum and a maximum are a statement about light rather than about a display
# curve; sRGB is the space the rest of this pipeline is written in -- the
# optimizer compares glyph coverage against 8-bit greys, and a linear grey handed
# to it would print about two stops too dark.


def _srgb_to_linear_lut() -> np.ndarray:
    values = np.arange(256, dtype=np.float64) / 255.0
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


_TO_LINEAR = _srgb_to_linear_lut()


def _linear_to_srgb(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    return np.where(
        clipped <= 0.0031308, clipped * 12.92, 1.055 * clipped ** (1 / 2.4) - 0.055
    )


#: Rec. 601 luma, the weights ``cv2.IMREAD_GRAYSCALE`` uses. Applied to *encoded*
#: values, which is what makes it the same conversion the rest of the pipeline
#: would have done on its own.
LUMA_WEIGHTS = (0.299, 0.587, 0.114)


# ---------------------------------------------------------------------------
# the spray
# ---------------------------------------------------------------------------


@njit(cache=True)
def _next_random(state):
    """xorshift32. Returns the new state and a double in [0, 1).

    A hand-rolled generator rather than numba's, for the reason the module
    docstring gives: numba's is per thread and cannot be seeded from Python, so
    a conversion using it would differ between runs of the same settings.
    """
    x = state
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= x >> 17
    x ^= (x << 5) & 0xFFFFFFFF
    # 24 bits, which a double holds exactly, taken from the high end: xorshift's
    # low bits are its weakest and the spray is what they would show up in.
    return x, (x >> 8) * (1.0 / 16777216.0)


@njit(cache=True)
def _seed_for(seed, index):
    """A generator state for one pixel, from the run's seed and the pixel index.

    Per pixel rather than one stream walked across the image, which is the first
    of the three deliberate differences from GEGL. It buys two things worth more
    than the fidelity: the result no longer depends on the order the pixels
    happen to be visited, so ``prange`` cannot change it, and it does not depend
    on where in the image a pixel is either, so a crop of the input gives the
    same answer for the pixels it kept.
    """
    x = ((seed & 0xFFFFFFFF) * 0x9E3779B1 + (index & 0xFFFFFFFF) * 0x85EBCA77) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & 0xFFFFFFFF
    x ^= x >> 16
    # xorshift32 is stuck at zero, and index 0 with seed 0 gets there.
    if x == 0:
        return 0x9E3779B9
    return x


@njit(cache=True)
def _sample_offset(state, radius):
    """One spray offset: a point in the disc of ``radius``, biased inwards.

    GEGL draws an angle and a radius from precomputed tables and raises the
    radius to ``RGAMMA`` -- 2.0, hardcoded there, with the property that would
    expose it commented out. So the distance is ``radius * u**2`` for a uniform
    ``u``, which crowds the samples towards the centre: near neighbours describe
    the local envelope, far ones only place it against the picture at large.

    Reproducing that with a sine and a power would put a libm in the middle of a
    result that is meant to be identical on every machine this is developed on.
    A point rejection-sampled from the unit disc gives the direction without
    trigonometry, and ``s = a^2 + b^2`` is itself uniform on (0, 1] -- so it *is*
    the ``u`` the radius wants, and the whole offset comes out as
    ``radius * a * s * sqrt(s)``: multiplications and a square root, both exactly
    rounded in IEEE 754 arithmetic everywhere.
    """
    a = 0.0
    b = 0.0
    s = 0.0
    for _ in range(_DISC_ATTEMPTS):
        state, u = _next_random(state)
        state, v = _next_random(state)
        a = 2.0 * u - 1.0
        b = 2.0 * v - 1.0
        s = a * a + b * b
        if s > 0.0 and s <= 1.0:
            break
    if s <= 0.0 or s > 1.0:
        # Nothing landed in the disc. Fall back to the centre rather than to a
        # square-shaped spray; one pixel in a hundred thousand loses one sample.
        return state, 0.0, 0.0
    scale = radius * s * np.sqrt(s)
    return state, a * scale, b * scale


@njit(cache=True)
def _envelopes(image, x, y, radius, samples, iterations, seed):
    """The local minimum and maximum around one pixel, per channel.

    This is GEGL's ``compute_envelopes``. Each iteration sprays ``samples``
    points, notes the smallest and largest value seen in each channel, and
    records two things: how wide that range was, and where the centre pixel sat
    inside it. Averaging *those* over the iterations rather than averaging the
    minima and maxima themselves is the whole trick -- it is what lets five
    samples describe a neighbourhood of a hundred thousand pixels without the
    envelopes collapsing onto the extremes of the picture.

    The envelopes are then reconstructed around the centre pixel from the mean
    range and the mean relative position, so the centre always lies between
    them.
    """
    height = image.shape[0]
    width = image.shape[1]
    state = _seed_for(seed, y * width + x)

    centre0 = image[y, x, 0]
    centre1 = image[y, x, 1]
    centre2 = image[y, x, 2]

    range0 = 0.0
    range1 = 0.0
    range2 = 0.0
    relative0 = 0.0
    relative1 = 0.0
    relative2 = 0.0

    for _ in range(iterations):
        min0 = centre0
        min1 = centre1
        min2 = centre2
        max0 = centre0
        max1 = centre1
        max2 = centre2

        for _ in range(samples):
            state, dx, dy = _sample_offset(state, radius)
            # Clamped to the edge, which is the second deliberate difference:
            # GEGL lets the spray fall outside and leans on its abyss policy.
            # Clamping means a pixel near an edge is measured against a spray
            # that is denser on the inside, which is preferable to measuring it
            # against pixels that do not exist.
            sx = int(np.floor(x + dx + 0.5))
            sy = int(np.floor(y + dy + 0.5))
            if sx < 0:
                sx = 0
            elif sx >= width:
                sx = width - 1
            if sy < 0:
                sy = 0
            elif sy >= height:
                sy = height - 1

            value0 = image[sy, sx, 0]
            value1 = image[sy, sx, 1]
            value2 = image[sy, sx, 2]
            if value0 < min0:
                min0 = value0
            elif value0 > max0:
                max0 = value0
            if value1 < min1:
                min1 = value1
            elif value1 > max1:
                max1 = value1
            if value2 < min2:
                min2 = value2
            elif value2 > max2:
                max2 = value2

        span0 = max0 - min0
        span1 = max1 - min1
        span2 = max2 - min2
        range0 += span0
        range1 += span1
        range2 += span2
        relative0 += (centre0 - min0) / span0 if span0 > 0.0 else 0.5
        relative1 += (centre1 - min1) / span1 if span1 > 0.0 else 0.5
        relative2 += (centre2 - min2) / span2 if span2 > 0.0 else 0.5

    inverse = 1.0 / iterations
    range0 *= inverse
    range1 *= inverse
    range2 *= inverse
    relative0 *= inverse
    relative1 *= inverse
    relative2 *= inverse

    return (
        centre0 - relative0 * range0,
        centre1 - relative1 * range1,
        centre2 - relative2 * range2,
        centre0 + (1.0 - relative0) * range0,
        centre1 + (1.0 - relative1) * range1,
        centre2 + (1.0 - relative2) * range2,
    )


@njit(cache=True, parallel=True)
def _c2g(image, radius, samples, iterations, seed):
    """Colour to grey: the pixel projected onto the local envelope axis.

    GEGL's ``gegl:c2g``, and the one the paper's Figure 12 is about. The
    envelopes span a line through colour space; where the pixel falls along that
    line is its grey. Two colours of identical luminance sitting side by side put
    the envelopes *along the axis that separates them*, so the projection gives
    them different greys -- which is exactly the failure of a fixed luma formula
    that this exists to answer.
    """
    height = image.shape[0]
    width = image.shape[1]
    out = np.empty((height, width), dtype=np.float64)

    for y in prange(height):
        for x in range(width):
            min0, min1, min2, max0, max1, max2 = _envelopes(
                image, x, y, radius, samples, iterations, seed
            )
            span0 = max0 - min0
            span1 = max1 - min1
            span2 = max2 - min2
            numerator = (
                (image[y, x, 0] - min0) * span0
                + (image[y, x, 1] - min1) * span1
                + (image[y, x, 2] - min2) * span2
            )
            denominator = span0 * span0 + span1 * span1 + span2 * span2
            if denominator > 0.0:
                out[y, x] = numerator / denominator
            else:
                # A neighbourhood with no variation at all has no axis to project
                # onto. GEGL hands back the red channel; luma is the same
                # decision made in a way that still reads as the picture, which
                # matters because a flat region large enough to swallow the spray
                # is a sky, not a rounding case.
                out[y, x] = (
                    LUMA_WEIGHTS[0] * image[y, x, 0]
                    + LUMA_WEIGHTS[1] * image[y, x, 1]
                    + LUMA_WEIGHTS[2] * image[y, x, 2]
                )
    return out


@njit(cache=True, parallel=True)
def _stress(image, radius, samples, iterations, seed, enhance_shadows):
    """The local enhancement: each channel rescaled by its own envelope.

    GEGL's ``gegl:stress``. Unlike ``c2g`` this stays in colour -- it is a
    retinex, correcting a cast and opening up local contrast -- so a grey
    conversion still has to follow it.

    With ``enhance_shadows`` off (GEGL's default) only the upper envelope is
    used, which lifts the picture towards white without touching what the
    shadows do; with it on, both envelopes are, which normalises every
    neighbourhood to the full range and looks correspondingly synthetic.
    """
    height = image.shape[0]
    width = image.shape[1]
    out = np.empty((height, width, 3), dtype=np.float64)

    for y in prange(height):
        for x in range(width):
            min0, min1, min2, max0, max1, max2 = _envelopes(
                image, x, y, radius, samples, iterations, seed
            )
            for channel in range(3):
                value = image[y, x, channel]
                if channel == 0:
                    low, high = min0, max0
                elif channel == 1:
                    low, high = min1, max1
                else:
                    low, high = min2, max2
                if not enhance_shadows:
                    low = 0.0
                span = high - low
                if span > 0.0:
                    out[y, x, channel] = (value - low) / span
                else:
                    out[y, x, channel] = 0.5
    return out


# ---------------------------------------------------------------------------
# the conversions
# ---------------------------------------------------------------------------


def _as_rgb(image: np.ndarray) -> np.ndarray:
    """An 8-bit RGB array, from whatever shape the caller had."""
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    elif array.ndim == 3 and array.shape[2] == 4:
        array = array[:, :, :3]
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected a greyscale or RGB image, got shape {array.shape}")
    return np.ascontiguousarray(array)


def radius_pixels(width: int, height: int, radius: float = DEFAULT_RADIUS) -> int:
    """The spray radius in pixels, from the fraction of the longest side.

    Never below two: a radius of one pixel samples the pixel itself, every
    envelope collapses and the picture comes out mid grey.
    """
    return max(2, int(round(radius * max(width, height))))


def to_grey(
    image: np.ndarray,
    method: str = "luma",
    *,
    radius: float = DEFAULT_RADIUS,
    samples: int = DEFAULT_SAMPLES,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = DEFAULT_SEED,
    enhance_shadows: bool = False,
) -> np.ndarray:
    """Convert an RGB image to 8-bit grey by one of :data:`METHODS`.

    ``luma`` and ``average`` are the two global formulas, and the second is the
    RGB average the paper's Figure 12 puts STRESS against. The other two are the
    spatial ones described at the top of this module.
    """
    if method not in METHODS:
        raise ValueError(f"unknown conversion '{method}'; pick one of {', '.join(METHODS)}")

    rgb = _as_rgb(image)
    if method == "luma":
        weighted = (
            LUMA_WEIGHTS[0] * rgb[:, :, 0].astype(np.float64)
            + LUMA_WEIGHTS[1] * rgb[:, :, 1].astype(np.float64)
            + LUMA_WEIGHTS[2] * rgb[:, :, 2].astype(np.float64)
        )
        return np.clip(np.round(weighted), 0, 255).astype(np.uint8)
    if method == "average":
        return np.round(rgb.astype(np.float64).mean(axis=2)).astype(np.uint8)

    height, width = rgb.shape[:2]
    linear = np.ascontiguousarray(_TO_LINEAR[rgb])
    pixels = float(radius_pixels(width, height, radius))
    if method == "c2g":
        result = _c2g(linear, pixels, int(samples), int(iterations), int(seed))
        return _to_byte(_linear_to_srgb(result))

    enhanced = _stress(
        linear, pixels, int(samples), int(iterations), int(seed), bool(enhance_shadows)
    )
    encoded = _linear_to_srgb(enhanced)
    # Luma over the *encoded* channels, so this last step is the same conversion
    # `luma` would have done -- the enhancement is what differs, not the way the
    # three channels are folded into one.
    weighted = (
        LUMA_WEIGHTS[0] * encoded[:, :, 0]
        + LUMA_WEIGHTS[1] * encoded[:, :, 1]
        + LUMA_WEIGHTS[2] * encoded[:, :, 2]
    )
    return _to_byte(weighted)


def enhance(
    image: np.ndarray,
    *,
    radius: float = DEFAULT_RADIUS,
    samples: int = DEFAULT_SAMPLES,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = DEFAULT_SEED,
    enhance_shadows: bool = False,
) -> np.ndarray:
    """``gegl:stress`` on its own: an 8-bit RGB image, still in colour."""
    rgb = _as_rgb(image)
    height, width = rgb.shape[:2]
    linear = np.ascontiguousarray(_TO_LINEAR[rgb])
    result = _stress(
        linear,
        float(radius_pixels(width, height, radius)),
        int(samples),
        int(iterations),
        int(seed),
        bool(enhance_shadows),
    )
    return _to_byte(_linear_to_srgb(result))


def _to_byte(values: np.ndarray) -> np.ndarray:
    return np.clip(np.round(values * 255.0), 0, 255).astype(np.uint8)


def convert_file(source: str, destination: str, method: str = "luma", **params) -> str:
    """Read an image, convert it to grey and write it out. Returns the path.

    The optimizer opens its target with ``cv2.IMREAD_GRAYSCALE``, which is the
    luma formula and no way to ask for another. So a different conversion is a
    file: this writes one and the caller points the optimizer at it.
    """
    import cv2

    image = cv2.imread(source, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read {source}")
    grey = to_grey(image[:, :, ::-1], method, **params)
    if not cv2.imwrite(destination, grey):
        raise ValueError(f"could not write {destination}")
    return destination
