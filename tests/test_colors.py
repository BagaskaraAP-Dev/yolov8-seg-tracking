# segtrack - YOLO instance segmentation with multi-object tracking
# Copyright (C) 2026 Bagaskara Amukti Palapa
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Rewritten from the GPL-3.0 project by Muhammad Moin Faisal; see NOTICE.

"""Colour assignment has to be stable, distinguishable and readable.

The first version of this module hashed the class name. These tests are
what caught it: two of thirty COCO names landed on the same 8-bit
colour, because hashing is random placement and random placement
collides. The golden-ratio sequence replaced it.
"""

from __future__ import annotations

import math
from itertools import combinations

import pytest

from segtrack.colors import class_color, contrast_ratio, relative_luminance, text_color_for

COCO_CLASS_COUNT = 80


def test_color_is_stable_for_an_index():
    assert class_color(2) == class_color(2)


def test_every_coco_class_gets_a_distinct_color():
    colors = [class_color(index) for index in range(COCO_CLASS_COUNT)]
    assert len(set(colors)) == COCO_CLASS_COUNT


def test_coco_classes_stay_visually_apart():
    # Distinct bytes is not enough: two colours one unit apart are the
    # same colour to a viewer. 15 is the floor this palette holds at 80
    # classes; it measures 19.4.
    colors = [class_color(index) for index in range(COCO_CLASS_COUNT)]
    closest = min(math.dist(a, b) for a, b in combinations(colors, 2))
    assert closest >= 15.0


def test_small_class_lists_are_spread_further_apart():
    colors = [class_color(index) for index in range(10)]
    closest = min(math.dist(a, b) for a, b in combinations(colors, 2))
    assert closest >= 40.0


def test_color_channels_are_in_range():
    for index in range(COCO_CLASS_COUNT):
        for channel in class_color(index):
            assert isinstance(channel, int)
            assert 0 <= channel <= 255


def test_a_custom_model_with_two_classes_works():
    # Upstream's COCO lookup gave these the bit-shift fallback palette.
    assert class_color(0) != class_color(1)


def test_a_model_with_many_classes_does_not_raise():
    assert len(class_color(999)) == 3


def test_negative_class_id_is_rejected():
    with pytest.raises(ValueError, match="must not be negative"):
        class_color(-1)


def test_label_text_is_readable_on_every_class_color():
    # WCAG 4.5:1 is the threshold for normal-size text. Picking the
    # better of black and white can never do worse than 4.58:1, so this
    # holds for any colour the palette can produce.
    for index in range(COCO_CLASS_COUNT):
        background = class_color(index)
        assert contrast_ratio(background, text_color_for(background)) >= 4.5


def test_text_color_flips_with_background_lightness():
    assert text_color_for((255, 255, 255)) == (0, 0, 0)
    assert text_color_for((0, 0, 0)) == (255, 255, 255)


@pytest.mark.parametrize(
    ("color", "expected"),
    [((0, 0, 0), 0.0), ((255, 255, 255), 1.0)],
)
def test_luminance_endpoints(color, expected):
    assert relative_luminance(color) == pytest.approx(expected, abs=1e-6)


def test_contrast_ratio_is_symmetric_and_bounded():
    assert contrast_ratio((0, 0, 0), (255, 255, 255)) == pytest.approx(21.0, abs=0.01)
    assert contrast_ratio((10, 20, 30), (10, 20, 30)) == pytest.approx(1.0)
    assert contrast_ratio((255, 0, 0), (0, 255, 0)) == pytest.approx(
        contrast_ratio((0, 255, 0), (255, 0, 0))
    )
