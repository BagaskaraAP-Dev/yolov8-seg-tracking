# segtrack - YOLO instance segmentation with multi-object tracking
# Copyright (C) 2026 Bagaskara Amukti Palapa
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Rewritten from the GPL-3.0 project by Muhammad Moin Faisal; see NOTICE.

"""The anchor point, which upstream wrote as int((y2 + y2) / 2)."""

from __future__ import annotations

import pytest

from segtrack.points import anchor_point

BOX = (10.0, 20.0, 30.0, 60.0)


def test_bottom_anchor_sits_on_the_bottom_edge():
    assert anchor_point(BOX, "bottom") == (20, 60)


def test_center_anchor_sits_in_the_middle():
    assert anchor_point(BOX, "center") == (20, 40)


def test_british_spelling_is_accepted():
    assert anchor_point(BOX, "centre") == anchor_point(BOX, "center")


def test_bottom_is_the_default():
    assert anchor_point(BOX) == anchor_point(BOX, "bottom")


def test_coordinates_are_rounded_not_truncated():
    assert anchor_point((0.0, 0.0, 3.0, 3.0), "center") == (2, 2)


def test_returns_integers_for_opencv():
    for value in anchor_point(BOX):
        assert isinstance(value, int)


def test_unknown_anchor_is_rejected():
    with pytest.raises(ValueError, match="unknown anchor"):
        anchor_point(BOX, "top")
