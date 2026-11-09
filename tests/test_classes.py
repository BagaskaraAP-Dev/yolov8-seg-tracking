# segtrack - YOLO instance segmentation with multi-object tracking
# Copyright (C) 2026 Bagaskara Amukti Palapa
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Rewritten from the GPL-3.0 project by Muhammad Moin Faisal; see NOTICE.

"""The class filter upstream left as a TODO."""

from __future__ import annotations

import pytest

from segtrack.classes import parse_classes

COCO = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def test_none_means_every_class():
    assert parse_classes(None, COCO) is None


def test_empty_string_means_every_class():
    assert parse_classes("", COCO) is None
    assert parse_classes("  ,  ", COCO) is None


def test_indices_are_parsed():
    assert parse_classes("0,2,3", COCO) == [0, 2, 3]


def test_names_are_parsed():
    assert parse_classes("car,bus", COCO) == [2, 5]


def test_names_and_indices_can_be_mixed():
    assert parse_classes("person,2,bus", COCO) == [0, 2, 5]


def test_names_are_case_insensitive():
    assert parse_classes("Car,BUS", COCO) == [2, 5]


def test_whitespace_and_semicolons_are_tolerated():
    assert parse_classes(" car ; bus ", COCO) == [2, 5]


def test_duplicates_collapse_and_output_is_sorted():
    assert parse_classes("bus,2,car,5", COCO) == [2, 5]


def test_a_sequence_works_as_well_as_a_string():
    assert parse_classes(["car", 5], COCO) == [2, 5]


def test_unknown_name_fails_loudly_and_lists_the_options():
    # Silently tracking nothing is the worst outcome here: the render
    # finishes, looks broken, and gives no reason.
    with pytest.raises(ValueError, match="unknown class name 'cat'") as caught:
        parse_classes("cat", COCO)
    assert "car" in str(caught.value)


def test_index_outside_the_model_is_rejected():
    with pytest.raises(ValueError, match="not in this model"):
        parse_classes("99", COCO)


def test_gap_in_the_index_space_is_respected():
    # This model has no class 4. Accepting it would filter everything out.
    with pytest.raises(ValueError, match="not in this model"):
        parse_classes("4", COCO)


def test_negative_index_is_rejected():
    with pytest.raises(ValueError, match="negative"):
        parse_classes("-1", COCO)


def test_indices_pass_without_a_model_mapping():
    assert parse_classes("0,2", None) == [0, 2]


def test_names_cannot_be_resolved_without_a_model_mapping():
    with pytest.raises(ValueError, match="before the model is loaded"):
        parse_classes("car", None)


def test_custom_dataset_names_resolve():
    custom = {0: "daun_sehat", 1: "daun_sakit"}
    assert parse_classes("daun_sakit", custom) == [1]
