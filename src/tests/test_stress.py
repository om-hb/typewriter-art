"""Tests for the colour to black and white conversions.

The two that carry the argument are the first two: they build the failure case
the paper's section 5.3 is about -- two colours a luma formula maps to the same
grey -- and assert that luma does merge them and that ``c2g`` does not.
Everything else guards a property the implementation is free to lose quietly:
that a conversion is reproducible, that the radius means the same thing at any
resolution, and that the numbers ``erika.pipeline`` advertises are the ones
``erika.stress`` actually uses.
"""

from __future__ import annotations

import numpy as np
import pytest

from erika import pipeline, stress


def two_colour_image(left=(200, 60, 60), right=(60, 130, 60), size=(64, 96)) -> np.ndarray:
    """Two blocks of equal-ish luma, side by side."""
    height, width = size
    image = np.zeros((height, width, 3), np.uint8)
    image[:, : width // 2] = left
    image[:, width // 2 :] = right
    return image


def block_means(grey: np.ndarray) -> tuple[float, float]:
    width = grey.shape[1]
    return float(grey[:, : width // 2].mean()), float(grey[:, width // 2 :].mean())


# ---------------------------------------------------------------------------
# what the conversions are for
# ---------------------------------------------------------------------------


def test_luma_merges_two_colours_of_equal_luminance():
    """The failure the rest of this module exists to answer.

    Red 200,60,60 and green 60,130,60 differ about as much as two colours can
    and still weigh the same under Rec. 601. On paper that is one flat area
    where the photograph had an edge -- and no tone control downstream can put
    the edge back, because by then there is nothing left to tell the two apart.
    """
    left, right = block_means(stress.to_grey(two_colour_image(), "luma"))
    assert abs(left - right) < 2


def test_c2g_separates_colours_that_luma_merges():
    """The decolorization, doing the one thing a global formula cannot.

    Measured on this pair it opens them to about 85 grey levels apart, which is
    a third of the range and some twenty steps of what the machine can print.

    Only ``c2g`` is claimed to do this. ``stress`` is a local *tone* map rather
    than a decolorization -- it separates the same two by about ten levels, and
    earns its place further down instead.
    """
    left, right = block_means(stress.to_grey(two_colour_image(), "c2g", iterations=8))
    assert abs(left - right) > 50


def test_average_is_the_paper_s_baseline():
    """Figure 12's left panel: a flat mean, which separates these two a little."""
    grey = stress.to_grey(two_colour_image(), "average")
    left, right = block_means(grey)
    assert left == pytest.approx((200 + 60 + 60) / 3, abs=0.5)
    assert right == pytest.approx((60 + 130 + 60) / 3, abs=0.5)


def test_c2g_keeps_a_smooth_ramp_in_order():
    """A monotone input has to come out monotone.

    A local method is free to be non-monotone over the picture as a whole --
    that is what makes it local -- but on a picture that is nothing *but* one
    global gradient there is no local structure to spend the range on, and an
    inverted patch would be an artefact rather than a decision.
    """
    ramp = np.tile(np.linspace(0, 255, 128, dtype=np.uint8), (256, 1))
    grey = stress.to_grey(np.dstack([ramp] * 3), "c2g", iterations=30)
    # Averaged down the 256 rows, which leaves the column means carrying about a
    # grey level of the spray's noise; the bound is what that noise cannot cross,
    # not a claim that the sequence is exactly monotone.
    columns = grey.mean(axis=0)
    assert np.all(np.diff(columns) > -3.0)
    # And it should still use the range rather than collapsing towards mid grey.
    assert columns[-1] - columns[0] > 200


def test_stress_lifts_a_flat_cast_towards_white():
    """The enhancement half: a picture behind a colour cast comes back lighter.

    With the shadows left alone (GEGL's default) every channel is divided by its
    own local maximum, which is a white balance -- so a uniformly dim, tinted
    image is pushed up towards the top of the range, where this machine has
    almost all of its usable levels.
    """
    dim = np.zeros((48, 48, 3), np.uint8)
    dim[:, :] = (90, 70, 40)
    dim[12:36, 12:36] = (140, 110, 60)
    plain = stress.to_grey(dim, "luma")
    lifted = stress.to_grey(dim, "stress", iterations=8)
    assert lifted.mean() > plain.mean() + 20


# ---------------------------------------------------------------------------
# properties the implementation must not lose
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", stress.METHODS)
def test_every_method_returns_an_8_bit_image_of_the_same_size(method):
    grey = stress.to_grey(two_colour_image(size=(24, 32)), method, iterations=3)
    assert grey.shape == (24, 32)
    assert grey.dtype == np.uint8


@pytest.mark.parametrize("method", ("c2g", "stress"))
def test_a_seed_makes_a_run_repeatable(method):
    """Sampling is stochastic; a job is half an hour of typing.

    Unlike the optimizer -- whose inner loop draws from numba's per-thread RNG
    and is out of reach of any seed -- this is reproducible outright, because
    the spray is seeded per pixel rather than from a stream the threads share.
    So it is also independent of how numba schedules the rows.
    """
    image = two_colour_image(size=(40, 40))
    once = stress.to_grey(image, method, iterations=4, seed=3)
    again = stress.to_grey(image, method, iterations=4, seed=3)
    assert np.array_equal(once, again)
    assert not np.array_equal(once, stress.to_grey(image, method, iterations=4, seed=4))


def test_more_iterations_means_less_noise():
    """The knob does what its help text says, which is the only reason it exists.

    Two seeds disagree by exactly the sampling noise, so their difference is the
    measurement. It falls as one over the square root of the iterations, and the
    default is chosen where it lands below the machine's own quantisation -- see
    ``DEFAULT_ITERATIONS``.
    """
    image = two_colour_image(size=(64, 64))

    def noise(iterations):
        a = stress.to_grey(image, "c2g", iterations=iterations, seed=1).astype(int)
        b = stress.to_grey(image, "c2g", iterations=iterations, seed=2).astype(int)
        return np.abs(a - b).mean()

    assert noise(20) < noise(4)


def test_the_radius_is_a_fraction_of_the_longest_side():
    """The property that makes a resized copy of a picture convert like the picture.

    GEGL asks for the radius in pixels, and advises setting it to about the
    longest side -- which is the same number written in units that depend on the
    resolution. ``erika-studio`` converts a scaled-down copy of the photograph,
    and the answer has to be the one the full-size original would have given.
    """
    assert stress.radius_pixels(800, 600, 1.0) == 800
    assert stress.radius_pixels(400, 300, 1.0) == 400
    assert stress.radius_pixels(800, 600, 0.25) == 200
    # A degenerate radius would sample only the pixel itself, every envelope
    # would collapse and the picture would come out flat mid grey.
    assert stress.radius_pixels(8, 8, 0.0) == 2


def test_a_greyscale_or_rgba_input_is_accepted():
    """Whatever cv2 or a PNG with alpha hands over has to work."""
    flat = np.full((16, 16), 120, np.uint8)
    assert stress.to_grey(flat, "luma").shape == (16, 16)
    with_alpha = np.full((16, 16, 4), 200, np.uint8)
    assert stress.to_grey(with_alpha, "c2g", iterations=2).shape == (16, 16)


def test_an_image_smaller_than_the_spray_still_converts():
    """The clamped spray must survive a picture only a few pixels across."""
    tiny = np.arange(3 * 3 * 3, dtype=np.uint8).reshape(3, 3, 3)
    assert stress.to_grey(tiny, "c2g", iterations=2).shape == (3, 3)


def test_an_unknown_method_is_refused_by_name():
    with pytest.raises(ValueError, match="c2g"):
        stress.to_grey(two_colour_image(), "retinex")


def test_enhance_returns_colour():
    """``gegl:stress`` on its own is still an RGB image; only ``c2g`` is grey."""
    out = stress.enhance(two_colour_image(size=(24, 24)), iterations=3)
    assert out.shape == (24, 24, 3)
    assert out.dtype == np.uint8


# ---------------------------------------------------------------------------
# the duplication, and its guard
# ---------------------------------------------------------------------------


def test_the_cli_advertises_the_methods_and_defaults_the_converter_has():
    """``pipeline.py`` spells the methods and defaults out rather than importing them.

    Deliberately: building the parser is what every subcommand and every
    ``--help`` does, and what ``erika-studio`` does to read the defaults back
    out, and importing ``erika.stress`` for it would drag numba -- two seconds
    of import -- into runs that never touch a photograph. This is the price of
    that, and this test is what keeps it honest.
    """
    assert pipeline.GREY_METHODS == stress.METHODS
    assert pipeline.stress_defaults("radius") == stress.DEFAULT_RADIUS
    assert pipeline.stress_defaults("samples") == stress.DEFAULT_SAMPLES
    assert pipeline.stress_defaults("iterations") == stress.DEFAULT_ITERATIONS
    assert pipeline.stress_defaults("seed") == stress.DEFAULT_SEED


def test_the_print_command_defaults_to_the_conversion_the_optimizer_would_do():
    """`--grey luma` is `cv2.IMREAD_GRAYSCALE`, so an unflagged run is unchanged."""
    args = pipeline.build_parser().parse_args(["print", "-t", "images/mwdog_crop.png"])
    assert args.grey == "luma"
    assert pipeline.preprocess_target(args) == "images/mwdog_crop.png"


def test_a_directory_target_is_refused_rather_than_half_converted(tmp_path):
    """`kword` also takes a directory of frames; this conversion does not."""
    args = pipeline.build_parser().parse_args(
        ["print", "-t", str(tmp_path), "--grey", "c2g"]
    )
    with pytest.raises(pipeline.PlanError, match="single image"):
        pipeline.preprocess_target(args)


def test_an_unknown_grey_method_is_refused_by_the_parser(capsys):
    """argparse names the alternatives, which is the whole value of `choices`."""
    with pytest.raises(SystemExit):
        pipeline.build_parser().parse_args(["print", "--grey", "sepia"])
    assert "c2g" in capsys.readouterr().err


def test_an_unknown_grey_method_is_refused_before_the_optimizer_runs():
    """And again where it is used, for a caller that did not come via the CLI."""
    args = pipeline.build_parser().parse_args(["print", "-t", "images/mwdog_crop.png"])
    args.grey = "sepia"
    with pytest.raises(pipeline.PlanError, match="sepia"):
        pipeline.preprocess_target(args)


def test_convert_file_writes_a_greyscale_png(tmp_path):
    cv2 = pytest.importorskip("cv2")
    source = tmp_path / "in.png"
    cv2.imwrite(str(source), two_colour_image(size=(24, 32))[:, :, ::-1])
    out = tmp_path / "out.png"
    stress.convert_file(str(source), str(out), "c2g", iterations=3)
    written = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert written.shape == (24, 32)
