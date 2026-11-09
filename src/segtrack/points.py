# segtrack - YOLO instance segmentation with multi-object tracking
# Copyright (C) 2026 Bagaskara Amukti Palapa
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Rewritten from the GPL-3.0 project by Muhammad Moin Faisal; see NOTICE.

"""Reducing a box to the one point that represents the object."""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["anchor_point"]

Point = tuple[int, int]


def anchor_point(xyxy: Sequence[float], mode: str = "bottom") -> Point:
    """The point used for an object's trail and for line counting.

    Upstream wrote this as int((y2 + y2) / 2), which is the bottom edge
    expressed in a way that reads like a typo for the centre. Both are
    defensible, so it is a named option here instead of a puzzle:

    - "bottom" follows where an object meets the ground, which is what a
      counting line drawn across a road wants.
    - "center" is steadier when objects are partly occluded, because the
      bottom edge jumps as soon as the feet leave the frame.
    """
    x1, y1, x2, y2 = xyxy
    mid_x = round((x1 + x2) / 2)
    if mode == "bottom":
        return (mid_x, round(y2))
    if mode in ("center", "centre"):
        return (mid_x, round((y1 + y2) / 2))
    raise ValueError(f"unknown anchor {mode!r}; expected 'bottom' or 'center'")
