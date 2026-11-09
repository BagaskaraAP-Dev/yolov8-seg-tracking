# segtrack - YOLO instance segmentation with multi-object tracking
# Copyright (C) 2026 Bagaskara Amukti Palapa
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Rewritten from the GPL-3.0 project by Muhammad Moin Faisal; see NOTICE.

"""Frame annotation: boxes, labels, trails, the counting line, the HUD
and the authorship watermark.

Everything that needs OpenCV lives here, so the geometry and colour
modules stay testable without it.

Three upstream drawing problems are fixed here:

- Labels were always drawn above the box, so an object touching the top
  of the frame had its id rendered off-screen.
- Label text was always white, which vanished on light box colours.
- Trail thickness was computed as sqrt(64 / (i + i)) * 1.5 with 64
  hard-coded, so changing the trail length changed nothing about how it
  tapered, and the first segment was drawn thicker than the line width
  of the box it belonged to.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from .colors import text_color_for

__all__ = [
    "draw_box",
    "draw_counting_line",
    "draw_hud",
    "draw_trail",
    "draw_watermark",
    "format_counts",
    "resize_masks",
]

BGR = tuple[int, int, int]
Point = tuple[int, int]

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _scaled(frame: np.ndarray, base: float) -> float:
    """Scale a size by frame height so a 4K render is not covered in 12px text."""
    return max(base * frame.shape[0] / 720.0, base * 0.6)


def draw_box(
    frame: np.ndarray,
    xyxy: Sequence[float],
    label: str,
    color: BGR,
    thickness: int = 2,
) -> None:
    """Draw one bounding box with a readable label. Modifies frame in place."""
    x1, y1, x2, y2 = (round(value) for value in xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    if not label:
        return

    font_scale = _scaled(frame, 0.5)
    font_thickness = max(1, round(font_scale * 2))
    (text_width, text_height), baseline = cv2.getTextSize(
        label, _FONT, font_scale, font_thickness
    )
    pad = max(2, round(font_scale * 6))
    plate_height = text_height + baseline + pad

    # Above the box by default; inside it when there is no room above, so
    # objects at the top edge of the frame keep their label.
    top = y1 - plate_height
    if top < 0:
        top = min(y1, frame.shape[0] - plate_height)
    top = max(0, top)

    left = max(0, min(x1, frame.shape[1] - (text_width + 2 * pad)))
    right = min(frame.shape[1], left + text_width + 2 * pad)

    cv2.rectangle(frame, (left, top), (right, top + plate_height), color, -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        label,
        (left + pad, top + text_height + pad // 2),
        _FONT,
        font_scale,
        text_color_for(color),
        font_thickness,
        cv2.LINE_AA,
    )


def draw_trail(
    frame: np.ndarray,
    points: Sequence[Point],
    color: BGR,
    max_thickness: int = 3,
) -> None:
    """Draw a track's path, newest point first, tapering as it gets older.

    The taper is relative to the trail's own length, so a longer trail
    fades over its whole length instead of over a fixed 64 frames.
    """
    if len(points) < 2:
        return
    span = len(points) - 1
    for index in range(1, len(points)):
        newer, older = points[index - 1], points[index]
        if newer is None or older is None:
            continue
        weight = 1.0 - (index - 1) / span
        thickness = max(1, round(max_thickness * weight))
        cv2.line(frame, newer, older, color, thickness, cv2.LINE_AA)


def draw_counting_line(
    frame: np.ndarray,
    line: tuple[tuple[float, float], tuple[float, float]],
    color: BGR = (60, 200, 255),
    thickness: int = 2,
) -> None:
    """Draw the counting line so the viewer can see what is being counted."""
    start = (round(line[0][0]), round(line[0][1]))
    end = (round(line[1][0]), round(line[1][1]))
    cv2.line(frame, start, end, color, thickness, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    lines: Sequence[str],
    origin: Point = (12, 12),
    alpha: float = 0.55,
) -> None:
    """Draw one translucent panel of status text at the top left.

    One panel, not one per metric: the text is the content, the panel
    exists only to keep it readable over a bright frame.
    """
    rows = [row for row in lines if row]
    if not rows:
        return

    font_scale = _scaled(frame, 0.52)
    font_thickness = max(1, round(font_scale * 2))
    pad = max(6, round(font_scale * 10))

    sizes = [cv2.getTextSize(row, _FONT, font_scale, font_thickness) for row in rows]
    row_height = max(size[0][1] for size in sizes) + max(size[1] for size in sizes)
    gap = max(3, round(font_scale * 6))

    panel_width = max(size[0][0] for size in sizes) + 2 * pad
    panel_height = len(rows) * row_height + (len(rows) - 1) * gap + 2 * pad

    left, top = origin
    right = min(frame.shape[1], left + panel_width)
    bottom = min(frame.shape[0], top + panel_height)
    if right <= left or bottom <= top:
        return

    region = frame[top:bottom, left:right]
    panel = np.full_like(region, (24, 24, 24))
    cv2.addWeighted(panel, alpha, region, 1.0 - alpha, 0, dst=region)

    text_y = top + pad + sizes[0][0][1]
    for row, ((_, text_height), baseline) in zip(rows, sizes, strict=False):
        cv2.putText(
            frame,
            row,
            (left + pad, text_y),
            _FONT,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )
        text_y += text_height + baseline + gap


def draw_watermark(
    frame: np.ndarray,
    text: str,
    alpha: float = 0.6,
) -> None:
    """Stamp an authorship and licence line into the bottom-right corner.

    Rendered output travels away from this repository, so the frame
    itself carries who made it and under what licence. One line, corner
    placement, no box and no glow: it should be findable, not loud.
    """
    if not text:
        return

    font_scale = _scaled(frame, 0.42)
    font_thickness = max(1, round(font_scale * 2))
    (text_width, text_height), baseline = cv2.getTextSize(
        text, _FONT, font_scale, font_thickness
    )
    margin = max(8, round(font_scale * 14))

    x = frame.shape[1] - text_width - margin
    y = frame.shape[0] - margin - baseline
    if x < 0 or y - text_height < 0:
        return

    stamped = frame.copy()
    # A dark pass under a light pass, so the line survives both a bright
    # and a dark background without needing a plate behind it.
    cv2.putText(
        stamped, text, (x + 1, y + 1), _FONT, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA
    )
    cv2.putText(
        stamped, text, (x, y), _FONT, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA
    )
    cv2.addWeighted(stamped, alpha, frame, 1.0 - alpha, 0, dst=frame)


def resize_masks(masks: np.ndarray, size: tuple[int, int]) -> list[np.ndarray]:
    """Resize an (N, h, w) mask stack to (height, width) frame resolution.

    Nearest-neighbour keeps the mask binary, which is what blend_masks
    expects; interpolating would soften the edge into partial alpha.
    """
    height, width = size
    resized = []
    for mask in np.asarray(masks):
        array = np.asarray(mask)
        if array.shape == (height, width):
            resized.append(array)
            continue
        resized.append(
            cv2.resize(
                array.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        )
    return resized


def format_counts(
    totals: dict[str, int], per_class: dict[str, dict[str, int]] | None = None
) -> list[str]:
    """Turn counter output into HUD rows, busiest class first."""
    rows = [f"in {totals.get('in', 0)}  out {totals.get('out', 0)}"]
    if per_class:
        merged: dict[str, int] = {}
        for counts in per_class.values():
            for name, value in counts.items():
                merged[name] = merged.get(name, 0) + value
        for name, value in sorted(merged.items(), key=lambda item: (-item[1], item[0])):
            rows.append(f"{name}: {value}")
    return rows
