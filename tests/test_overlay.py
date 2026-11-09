# segtrack - YOLO instance segmentation with multi-object tracking
# Copyright (C) 2026 Bagaskara Amukti Palapa
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Rewritten from the GPL-3.0 project by Muhammad Moin Faisal; see NOTICE.

"""Mask compositing, checked on arrays small enough to reason about."""

from __future__ import annotations

import numpy as np
import pytest

from segtrack.overlay import blend_masks, mask_to_bool

RED = (0, 0, 255)
BLUE = (255, 0, 0)


def frame(height=4, width=4, value=0):
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_mask_to_bool_handles_float_confidence():
    mask = np.array([[0.1, 0.9], [0.5, 0.51]])
    assert mask_to_bool(mask).tolist() == [[False, True], [False, True]]


def test_mask_to_bool_passes_through_bool():
    mask = np.array([[True, False]])
    assert mask_to_bool(mask) is mask


def test_mask_to_bool_handles_integer_masks():
    assert mask_to_bool(np.array([[0, 1, 255]], dtype=np.uint8)).tolist() == [
        [False, True, True]
    ]


def test_unmasked_pixels_are_untouched():
    base = frame(value=10)
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    out = blend_masks(base, [mask], [RED], alpha=1.0)
    assert out[1, 1].tolist() == [10, 10, 10]


def test_masked_pixels_take_the_color_at_full_alpha():
    mask = np.ones((2, 2), dtype=bool)
    out = blend_masks(frame(2, 2), [mask], [RED], alpha=1.0)
    assert out[0, 0].tolist() == list(RED)


def test_blend_is_a_weighted_average():
    mask = np.ones((1, 1), dtype=bool)
    out = blend_masks(frame(1, 1, value=100), [mask], [(200, 200, 200)], alpha=0.5)
    assert out[0, 0].tolist() == [150, 150, 150]


def test_first_mask_wins_where_two_overlap():
    # Masks arrive sorted by descending confidence, so the more certain
    # instance must stay visible instead of being painted over.
    full = np.ones((2, 2), dtype=bool)
    out = blend_masks(frame(2, 2), [full, full], [RED, BLUE], alpha=1.0)
    assert out[0, 0].tolist() == list(RED)


def test_second_mask_still_paints_where_it_does_not_overlap():
    left = np.zeros((2, 2), dtype=bool)
    left[:, 0] = True
    right = np.zeros((2, 2), dtype=bool)
    right[:, 1] = True
    out = blend_masks(frame(2, 2), [left, right], [RED, BLUE], alpha=1.0)
    assert out[0, 0].tolist() == list(RED)
    assert out[0, 1].tolist() == list(BLUE)


def test_input_frame_is_not_modified():
    base = frame(2, 2, value=5)
    blend_masks(base, [np.ones((2, 2), dtype=bool)], [RED], alpha=1.0)
    assert base[0, 0].tolist() == [5, 5, 5]


def test_no_masks_returns_a_copy():
    base = frame(2, 2, value=7)
    out = blend_masks(base, [], [])
    assert out is not base
    assert np.array_equal(out, base)


def test_zero_alpha_leaves_the_frame_alone():
    base = frame(2, 2, value=7)
    out = blend_masks(base, [np.ones((2, 2), dtype=bool)], [RED], alpha=0.0)
    assert np.array_equal(out, base)


def test_mismatched_mask_and_color_counts_are_rejected():
    with pytest.raises(ValueError, match="2 masks but 1 colors"):
        blend_masks(frame(2, 2), [np.ones((2, 2), bool)] * 2, [RED])


def test_mask_at_the_wrong_resolution_is_rejected():
    # Upstream drew masks that could silently fall out of step with the
    # frame. A loud error beats a mask painted onto the wrong object.
    with pytest.raises(ValueError, match="resize masks"):
        blend_masks(frame(4, 4), [np.ones((2, 2), bool)], [RED])


def test_non_image_frame_is_rejected():
    with pytest.raises(ValueError, match="HxWx3"):
        blend_masks(np.zeros((4, 4), dtype=np.uint8), [], [])


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_alpha_outside_the_unit_range_is_rejected(alpha):
    with pytest.raises(ValueError, match="between 0 and 1"):
        blend_masks(frame(2, 2), [], [], alpha=alpha)


def test_output_stays_uint8():
    out = blend_masks(frame(2, 2, 255), [np.ones((2, 2), bool)], [(255, 255, 255)], alpha=1.0)
    assert out.dtype == np.uint8
    assert out.max() <= 255
