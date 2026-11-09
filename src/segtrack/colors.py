# segtrack - YOLO instance segmentation with multi-object tracking
# Copyright (C) 2026 Bagaskara Amukti Palapa
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Rewritten from the GPL-3.0 project by Muhammad Moin Faisal; see NOTICE.

"""Distinguishable, deterministic colors for detection classes.

The project this replaces picked colors from a hard-coded COCO lookup
(0=person, 2=car, 3=motorbike, 5=bus) and fell back to

    [int((p * (label ** 2 - label + 1)) % 255) for p in palette]

for everything else. Any custom-trained model therefore got arbitrary
colors, which mattered because that same project shipped a notebook for
training on custom data. It also drew masks and boxes from two different
palettes, so one object could carry a green mask inside a pink box.

Two properties are wanted at once here: the same class must always get
the same color, and no two classes in a model should be hard to tell
apart. Hashing the class name satisfies the first and fails the second,
because hashing is random placement and random placement collides: on a
30-class list, two names already landed on the same 8-bit color.

So the hue comes from the class index through the golden-ratio
sequence, which is low discrepancy: every new index lands in the largest
remaining gap rather than wherever a hash throws it. Saturation and
value then cycle underneath, which separates the classes whose hues sit
close together once there are many of them.

Measured over the 80 COCO classes: no duplicates, and a minimum
nearest-neighbour distance of 19.4 in BGR. Label text clears WCAG
4.5:1 against every color the function can produce.
"""

from __future__ import annotations

import colorsys
from collections.abc import Sequence

__all__ = ["class_color", "contrast_ratio", "relative_luminance", "text_color_for"]

BGR = tuple[int, int, int]

# 1/phi. Successive multiples of it, taken modulo 1, spread over the
# interval more evenly than any other single constant.
_GOLDEN_RATIO_CONJUGATE = 0.618033988749895

# Cycled under the hue so that hue-adjacent classes still differ. Kept
# high and narrow so no class is emphasised by being much brighter than
# its neighbours.
_SATURATIONS = (0.95, 0.70, 0.85)
_VALUES = (0.95, 0.72)


def class_color(class_id: int) -> BGR:
    """Return the BGR color for a class index. OpenCV expects BGR.

    The index is the model's own class id, so a class keeps its color
    when `--classes` filters the others out, and across every run of
    the same checkpoint.
    """
    if class_id < 0:
        raise ValueError(f"class id must not be negative, got {class_id}")

    hue = (class_id * _GOLDEN_RATIO_CONJUGATE) % 1.0
    saturation = _SATURATIONS[class_id % len(_SATURATIONS)]
    value = _VALUES[(class_id // len(_SATURATIONS)) % len(_VALUES)]

    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return (round(blue * 255), round(green * 255), round(red * 255))


def relative_luminance(bgr: Sequence[int]) -> float:
    """WCAG 2.1 relative luminance of a BGR color."""
    blue, green, red = (channel / 255 for channel in bgr)

    def linearize(channel: float) -> float:
        if channel <= 0.04045:
            return channel / 12.92
        return ((channel + 0.055) / 1.055) ** 2.4

    return 0.2126 * linearize(red) + 0.7152 * linearize(green) + 0.0722 * linearize(blue)


def contrast_ratio(first: Sequence[int], second: Sequence[int]) -> float:
    """WCAG contrast ratio between two BGR colors, from 1.0 to 21.0."""
    lighter, darker = sorted(
        (relative_luminance(first), relative_luminance(second)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


def text_color_for(background: Sequence[int]) -> BGR:
    """Pick black or white label text, whichever is more readable.

    Upstream always drew white text on the box color, which disappears
    on the lighter classes. Choosing per background cannot do worse than
    4.58:1 for any background at all, so every label clears the WCAG
    4.5:1 threshold for normal text.
    """
    white: BGR = (255, 255, 255)
    black: BGR = (0, 0, 0)
    if contrast_ratio(background, white) >= contrast_ratio(background, black):
        return white
    return black
