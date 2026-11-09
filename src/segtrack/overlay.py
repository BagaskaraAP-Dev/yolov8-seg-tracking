# segtrack - YOLO instance segmentation with multi-object tracking
# Copyright (C) 2026 Bagaskara Amukti Palapa
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Rewritten from the GPL-3.0 project by Muhammad Moin Faisal; see NOTICE.

"""Mask compositing, in NumPy only.

Kept free of OpenCV and of ultralytics so the blending maths can be
tested on small arrays without a model, a video, or a GPU.

The project this replaces drew masks through its own annotator using a
per-class palette, while boxes came from a second palette, so the mask
and the box for one object disagreed on color. Here the caller passes
one color per instance and uses the same one for both.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["blend_masks", "mask_to_bool"]

BGR = tuple[int, int, int]


def mask_to_bool(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Normalise a mask of any dtype to a boolean array.

    Accepts the float confidence masks ultralytics returns as well as
    already-binary masks.
    """
    array = np.asarray(mask)
    if array.dtype == bool:
        return array
    if np.issubdtype(array.dtype, np.floating):
        return array > threshold
    return array > 0


def blend_masks(
    frame: np.ndarray,
    masks: Sequence[np.ndarray],
    colors: Sequence[BGR],
    alpha: float = 0.45,
    threshold: float = 0.5,
) -> np.ndarray:
    """Tint every masked region with its own color and return a new frame.

    Earlier masks win where two overlap, because `masks` arrives sorted
    by descending confidence: the more certain instance stays visible
    instead of being painted over by whatever happens to come last.

    The whole frame is blended in one pass, so cost does not grow with
    the number of overlapping instances.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    if len(masks) != len(colors):
        raise ValueError(f"got {len(masks)} masks but {len(colors)} colors")

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame must be HxWx3, got shape {frame.shape}")

    if not masks or alpha == 0.0:
        return frame.copy()

    height, width = frame.shape[:2]
    tint = np.zeros_like(frame, dtype=np.float32)
    claimed = np.zeros((height, width), dtype=bool)

    for mask, color in zip(masks, colors, strict=False):
        binary = mask_to_bool(mask, threshold)
        if binary.shape != (height, width):
            raise ValueError(
                f"mask shape {binary.shape} does not match frame {(height, width)}; "
                "resize masks to frame resolution first"
            )
        fresh = binary & ~claimed
        if not fresh.any():
            continue
        tint[fresh] = color
        claimed |= binary

    if not claimed.any():
        return frame.copy()

    blended = frame.astype(np.float32)
    blended[claimed] = blended[claimed] * (1.0 - alpha) + tint[claimed] * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)
