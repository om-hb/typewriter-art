"""Score candidate characters on local average tone as well as per pixel.

The optimizer compares a candidate against the target pixel by pixel. With a
charset of real typed characters that is the right thing to do: their ink is
soft and comes at several strike forces, so a cell can be filled to very nearly
any tone and the pixels agree with the tone. With a charset of hard marks it is
not, and the failure is specific: a *sparse* mark that is right on average is
wrong at every pixel it covers and every pixel it leaves bare, so the optimizer
declines to place it. It will not halftone. On this machine that costs the
highlights and midtones -- exactly where a printer would dither.

The paper proposes the remedy itself, in section 6:

    "the algorithm could be made to discourage reliance on precise character
    placement for tone matching, including by adding blur or reducing
    resolution during selection"

That is what this is. Loss becomes

    (1 - weight) * per-pixel AMSE  +  weight * AMSE over block x block means

so at weight 0 nothing changes, and at weight 1 only local tone is scored.

Two things it is *not*:

- It is not the charset being softened. The mockup stays a faithful composite of
  the glyphs as they are, which is what keeps `pipeline print`'s check
  meaningful: the plan is still diffed against a mockup that shows exactly what
  the machine will type. Blurring the *charset* to get the same effect would
  make the mockup a picture of a machine we do not have.
- It is not free. Pixel-level error goes up as tone goes down; measured at 40
  columns on the sample photograph with `4x2`, per-pixel RMSE moves 73 -> 82
  while the same error measured over half a cell -- roughly what an eye does
  with a typed sheet at arm's length -- moves 34 -> 15. Which of those you want
  depends on whether the result is to be looked at or zoomed into, so this is a
  flag and not a default.

Installed over ``optimize.layer_optimization_pass`` in ``optimize``'s own
namespace, which is the same seam erika-studio's progress display and variety
pass use, and for the same reason: ``optimize`` binds the name at import, so
patching ``utils`` would have no effect. Nothing upstream is edited.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

#: Weight above which only local tone is scored. Kept as a name because the
#: argument parser quotes it.
MAX_WEIGHT = 1.0

#: Block side, in cell pixels, over which tone is averaged. 8 on a 24x40 cell is
#: roughly a third of the cell width -- large enough that a thin mark is judged
#: by the tone it contributes rather than by the pixels it hits, small enough
#: that a cell still holds several independent blocks and shape survives.
DEFAULT_BLOCK = 8


@njit(parallel=True, fastmath=True)
def _pass(bg, mockup, target, chars, choices, layer_offset,
          asymmetry, mode, temperature, weight, block):
    """A copy of utils.layer_optimization_pass with a second loss term.

    Deliberately a copy rather than a wrapper: the block means have to be
    computed inside the candidate loop, which is the whole of the inner loop,
    and numba will not inline a Python-level helper into it.
    """
    num_cols = target.shape[1] // chars.shape[2]
    comparisons = np.zeros(choices.shape[0], dtype="uint32")
    total_err = 0.0
    char_h, char_w = chars.shape[1], chars.shape[2]
    n_by, n_bx = char_h // block, char_w // block

    for i in prange(choices.shape[0]):
        prev_choice = choices[i]
        row = i // num_cols
        col = i % num_cols
        y0 = row * char_h + layer_offset[0]
        x0 = col * char_w + layer_offset[1]
        target_slice = target[y0:y0 + char_h, x0:x0 + char_w]
        mockup_slice = mockup[y0:y0 + char_h, x0:x0 + char_w]
        bg_slice = bg[y0:y0 + char_h, x0:x0 + char_w]

        # The target's block means do not depend on the candidate, so once.
        target_blocks = np.zeros((n_by, n_bx), dtype=np.float32)
        for by in range(n_by):
            for bx in range(n_bx):
                acc = 0.0
                for yy in range(by * block, (by + 1) * block):
                    for xx in range(bx * block, (bx + 1) * block):
                        acc += target_slice[yy, xx]
                target_blocks[by, bx] = acc / (block * block)

        cand_blocks = np.zeros((n_by, n_bx), dtype=np.float32)

        cur_composite = mockup_slice
        err = target_slice - cur_composite
        asym = np.where(err > 0, err * (1 + asymmetry), err)
        fine = np.mean(np.square(asym))
        for by in range(n_by):
            for bx in range(n_bx):
                acc = 0.0
                for yy in range(by * block, (by + 1) * block):
                    for xx in range(bx * block, (bx + 1) * block):
                        acc += cur_composite[yy, xx]
                cand_blocks[by, bx] = acc / (block * block)
        block_err = target_blocks - cand_blocks
        block_asym = np.where(block_err > 0, block_err * (1 + asymmetry), block_err)
        cur_loss = (1.0 - weight) * fine + weight * np.mean(np.square(block_asym))

        for new_choice in np.random.permutation(chars.shape[0]):
            comparisons[i] += 1
            if new_choice == prev_choice:
                continue
            new_composite = bg_slice * chars[new_choice]
            err = target_slice - new_composite
            asym = np.where(err > 0, err * (1 + asymmetry), err)
            fine = np.mean(np.square(asym))
            for by in range(n_by):
                for bx in range(n_bx):
                    acc = 0.0
                    for yy in range(by * block, (by + 1) * block):
                        for xx in range(bx * block, (bx + 1) * block):
                            acc += new_composite[yy, xx]
                    cand_blocks[by, bx] = acc / (block * block)
            block_err = target_blocks - cand_blocks
            block_asym = np.where(
                block_err > 0, block_err * (1 + asymmetry), block_err
            )
            new_loss = (1.0 - weight) * fine + weight * np.mean(np.square(block_asym))

            if mode == "greedy":
                if new_loss < cur_loss:
                    choices[i] = new_choice
                    cur_loss = new_loss
                    cur_composite = new_composite
            else:
                delta = cur_loss - new_loss
                if delta > 0:
                    choices[i] = new_choice
                    cur_loss = new_loss
                    cur_composite = new_composite
                    break
                # Same acceptance rule as upstream, including that a loss of
                # exactly 0 makes the exponent infinite -- numba has no
                # exception to catch there, so guard the division instead.
                if temperature > 0.0 and np.exp(delta / temperature) > np.random.rand():
                    choices[i] = new_choice
                    cur_loss = new_loss
                    cur_composite = new_composite
                    break

        # The mockup has to be updated exactly as upstream does it, or the plan
        # would be checked against a picture no longer made of these glyphs.
        mockup[y0:y0 + char_h, x0:x0 + char_w] = cur_composite
        total_err += cur_loss

    return choices, mockup, np.sum(comparisons), total_err / choices.shape[0]


def validate(weight: float, block: int, cell: tuple[int, int]) -> None:
    """Refuse settings that would silently do nothing or divide badly."""
    if not 0.0 <= weight <= MAX_WEIGHT:
        raise ValueError(f"match blur weight must be 0..{MAX_WEIGHT}, got {weight}")
    if block < 1:
        raise ValueError(f"match block must be at least 1, got {block}")
    cell_h, cell_w = cell
    if cell_h % block or cell_w % block:
        # Partial blocks at the right and bottom edge would weight the middle of
        # every cell more than its edges, which is a bias in the one direction
        # this whole idea is trying to remove.
        raise ValueError(
            f"match block {block} does not divide the {cell_w}x{cell_h} cell. "
            "Pick a block that divides both, or rebuild the charset with a "
            "--cell-height that it divides."
        )


def install(weight: float, block: int = DEFAULT_BLOCK) -> None:
    """Replace optimize's reference to the layer pass with the blurred one."""
    import optimize

    def wrapper(bg, mockup, target, chars, choices, layer_offset,
                asymmetry=0.1, mode="greedy", temperature=0.001):
        return _pass(bg, mockup, target, chars, choices, layer_offset,
                     asymmetry, mode, temperature, float(weight), int(block))

    optimize.layer_optimization_pass = wrapper


def uninstall() -> None:
    """Put upstream's own function back. Only the tests need this."""
    import optimize
    import utils

    optimize.layer_optimization_pass = utils.layer_optimization_pass
