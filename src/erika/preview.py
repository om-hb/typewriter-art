"""Re-render a motion plan back into an image.

This closes the loop. ``optimize.py`` produces a mockup by compositing its
layer arrays; the planner throws those arrays away and keeps only a list of
"strike glyph N at half-step x, half-line y". Rendering that list back out
and comparing it to the optimizer's own mockup proves the flattening, the
ordering and the index mapping are all faithful -- no typewriter required.

It also renders a *jittered* version, which applies a random registration
error to every strike. That is the honest preview: it shows how much of the
detail survives the mechanics before you spend an hour of ribbon finding out.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from erika.planner import Plan


def _canvas_shape(plan: Plan) -> tuple[int, int]:
    """Match the padded mockup shape optimize.py produces."""
    ch, cw = plan.charset.cell_h, plan.charset.cell_w
    pad_v = ch // 2 if any(ov for ov, _ in plan.layer_offsets) else 0
    pad_h = cw // 2 if any(oh for _, oh in plan.layer_offsets) else 0
    return plan.rows * ch + pad_v, plan.cols * cw + pad_h


def render(
    plan: Plan,
    tiles: np.ndarray,
    jitter: float = 0.0,
    seed: int = 0,
    upto: int | None = None,
) -> np.ndarray:
    """Composite the plan's strikes onto bare paper.

    ``tiles`` is the charset array from ``utils.prep_charset`` -- the same
    array optimize.py used, so a jitter-free render is bit-comparable to its
    mockup. Strikes multiply, exactly as overlapping ink does.

    ``jitter`` is the standard deviation of per-strike registration error, in
    cell widths/heights (0.05 is a realistic well-adjusted machine).
    """
    ch, cw = plan.charset.cell_h, plan.charset.cell_w
    if tiles.shape[1:] != (ch, cw):
        raise ValueError(
            f"charset tiles are {tiles.shape[1:]}, plan expects ({ch}, {cw}) -- "
            "the charset folder and glyphs.json have diverged"
        )
    height, width = _canvas_shape(plan)
    pad = max(ch, cw) if jitter else 0
    canvas = np.ones((height + 2 * pad, width + 2 * pad), dtype=np.float32)

    rng = np.random.default_rng(seed)
    strikes = plan.strikes[:upto] if upto is not None else plan.strikes
    for s in strikes:
        top = s.y * (ch // 2) + pad
        left = s.x * (cw // 2) + pad
        if jitter:
            top += int(round(rng.normal(0, jitter * ch)))
            left += int(round(rng.normal(0, jitter * cw)))
        canvas[top : top + ch, left : left + cw] *= tiles[s.index]

    return canvas[pad : pad + height, pad : pad + width]


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def compare(rendered: np.ndarray, reference_path: str) -> dict:
    """Diff a render against optimize.py's own mockup."""
    ref = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    if ref is None:
        raise FileNotFoundError(reference_path)
    ours = to_uint8(rendered)
    if ours.shape != ref.shape:
        return {"ok": False, "reason": f"shape {ours.shape} vs {ref.shape}"}
    diff = np.abs(ours.astype(np.int32) - ref.astype(np.int32))
    return {
        "ok": True,
        "max_abs": int(diff.max()),
        "mean_abs": float(diff.mean()),
        "differing_px": int((diff > 1).sum()),
        "total_px": int(diff.size),
    }


def save_previews(
    plan: Plan,
    tiles: np.ndarray,
    out_dir: str,
    jitter: float = 0.05,
    seed: int = 0,
) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = {}

    clean = render(plan, tiles)
    paths["plan"] = os.path.join(out_dir, "erika_plan.png")
    cv2.imwrite(paths["plan"], to_uint8(clean))

    if jitter > 0:
        shaky = render(plan, tiles, jitter=jitter, seed=seed)
        paths["jitter"] = os.path.join(out_dir, "erika_plan_jitter.png")
        cv2.imwrite(paths["jitter"], to_uint8(shaky))

    return paths
