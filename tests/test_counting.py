# segtrack - YOLO instance segmentation with multi-object tracking
# Copyright (C) 2026 Bagaskara Amukti Palapa
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Rewritten from the GPL-3.0 project by Muhammad Moin Faisal; see NOTICE.

"""Counting is the feature the upstream README advertised and shipped
without, so it carries the heaviest tests in this repository."""

from __future__ import annotations

import pytest

from segtrack.counting import IN, OUT, LineCounter, parse_line, segments_intersect, side_of

HORIZONTAL = ((0.0, 100.0), (200.0, 100.0))


def test_parse_line_reads_four_numbers():
    assert parse_line("0,600,1280,600") == ((0.0, 600.0), (1280.0, 600.0))


def test_parse_line_tolerates_spacing_and_semicolons():
    assert parse_line(" 0 ; 600 ; 10 ; 600 ") == ((0.0, 600.0), (10.0, 600.0))


@pytest.mark.parametrize("spec", ["0,600,1280", "0,600,1280,600,7", "", "a,b,c,d"])
def test_parse_line_rejects_bad_input(spec):
    with pytest.raises(ValueError):
        parse_line(spec)


def test_parse_line_rejects_zero_length():
    with pytest.raises(ValueError, match="zero length"):
        parse_line("5,5,5,5")


def test_side_of_is_opposite_across_the_line():
    above = side_of(HORIZONTAL, (100.0, 50.0))
    below = side_of(HORIZONTAL, (100.0, 150.0))
    assert above == -below
    assert above != 0


def test_side_of_is_zero_on_the_line():
    assert side_of(HORIZONTAL, (50.0, 100.0)) == 0


def test_segments_intersect_crossing():
    assert segments_intersect((1.0, 0.0), (1.0, 2.0), (0.0, 1.0), (2.0, 1.0))


def test_segments_do_not_intersect_when_short():
    # The moving object passes beyond the end of the counting line.
    assert not segments_intersect((9.0, 0.0), (9.0, 2.0), (0.0, 1.0), (2.0, 1.0))


def test_segments_intersect_touching_endpoint():
    assert segments_intersect((1.0, 1.0), (1.0, 5.0), (0.0, 1.0), (2.0, 1.0))


def test_segments_intersect_collinear_overlap():
    assert segments_intersect((0.0, 1.0), (1.0, 1.0), (0.5, 1.0), (2.0, 1.0))


def test_first_sighting_is_never_counted():
    # An object already past the line when it appears has not been seen
    # to cross it. Counting it would be inventing an event.
    counter = LineCounter(HORIZONTAL)
    assert counter.update(1, "car", (100.0, 150.0)) is None
    assert counter.totals() == {IN: 0, OUT: 0}


def test_crossing_one_way_counts_in():
    counter = LineCounter(HORIZONTAL)
    counter.update(1, "car", (100.0, 50.0))
    assert counter.update(1, "car", (100.0, 150.0)) == IN
    assert counter.totals()[IN] == 1


def test_crossing_the_other_way_counts_out():
    counter = LineCounter(HORIZONTAL)
    counter.update(1, "car", (100.0, 150.0))
    assert counter.update(1, "car", (100.0, 50.0)) == OUT
    assert counter.totals()[OUT] == 1


def test_staying_on_one_side_never_counts():
    counter = LineCounter(HORIZONTAL)
    for y in (10.0, 20.0, 30.0, 90.0, 20.0):
        counter.update(1, "car", (100.0, y))
    assert counter.totals() == {IN: 0, OUT: 0}


def test_loitering_on_the_line_counts_once_per_crossing():
    counter = LineCounter(HORIZONTAL)
    counter.update(1, "person", (100.0, 50.0))
    assert counter.update(1, "person", (100.0, 100.0)) is None  # exactly on it
    assert counter.update(1, "person", (100.0, 150.0)) == IN
    assert counter.update(1, "person", (100.0, 160.0)) is None
    assert counter.totals()[IN] == 1


def test_back_and_forth_counts_both_directions():
    counter = LineCounter(HORIZONTAL)
    counter.update(1, "person", (100.0, 50.0))
    counter.update(1, "person", (100.0, 150.0))
    counter.update(1, "person", (100.0, 50.0))
    assert counter.totals() == {IN: 1, OUT: 1}


def test_counts_are_split_per_class():
    counter = LineCounter(HORIZONTAL)
    for track_id, label in ((1, "car"), (2, "car"), (3, "truck")):
        counter.update(track_id, label, (100.0, 50.0))
        counter.update(track_id, label, (100.0, 150.0))
    assert counter.per_class(IN) == {"car": 2, "truck": 1}
    assert counter.per_class(OUT) == {}


def test_per_class_rejects_an_unknown_direction():
    with pytest.raises(KeyError):
        LineCounter(HORIZONTAL).per_class("sideways")


def test_tracks_are_counted_independently():
    counter = LineCounter(HORIZONTAL)
    counter.update(1, "car", (10.0, 50.0))
    counter.update(2, "car", (20.0, 150.0))
    assert counter.update(1, "car", (10.0, 150.0)) == IN
    assert counter.update(2, "car", (20.0, 50.0)) == OUT


def test_strict_mode_ignores_a_pass_beyond_the_line_ends():
    # The line spans x in [0, 200]. This track crosses y=100 at x=900,
    # which is off the end of the line, so it is not this line's traffic.
    counter = LineCounter(HORIZONTAL)
    assert counter.crossed(1, "car", (900.0, 50.0), (900.0, 150.0)) is None
    assert counter.totals()[IN] == 0


def test_loose_mode_does_count_a_pass_beyond_the_line_ends():
    counter = LineCounter(HORIZONTAL)
    counter.update(1, "car", (900.0, 50.0))
    assert counter.update(1, "car", (900.0, 150.0)) == IN


def test_strict_mode_counts_a_fast_jump_over_the_line():
    # A vehicle moving faster than the frame rate never lands on the line.
    # Testing the segment, not the distance, is what catches it.
    counter = LineCounter(HORIZONTAL)
    counter.update(1, "car", (100.0, 10.0))
    assert counter.crossed(1, "car", (100.0, 10.0), (100.0, 400.0)) == IN


def test_forget_releases_gone_tracks():
    counter = LineCounter(HORIZONTAL)
    counter.update(1, "car", (100.0, 50.0))
    counter.update(2, "car", (100.0, 50.0))
    assert counter.forget([1]) == 1


def test_forgotten_track_is_treated_as_new():
    counter = LineCounter(HORIZONTAL)
    counter.update(1, "car", (100.0, 50.0))
    counter.forget([])
    assert counter.update(1, "car", (100.0, 150.0)) is None


def test_reset_clears_counts_and_sides():
    counter = LineCounter(HORIZONTAL)
    counter.update(1, "car", (100.0, 50.0))
    counter.update(1, "car", (100.0, 150.0))
    counter.reset()
    assert counter.totals() == {IN: 0, OUT: 0}
    assert counter.update(1, "car", (100.0, 50.0)) is None


def test_swapping_endpoints_swaps_the_labels():
    forward = LineCounter(HORIZONTAL)
    backward = LineCounter((HORIZONTAL[1], HORIZONTAL[0]))
    for counter in (forward, backward):
        counter.update(1, "car", (100.0, 50.0))
        counter.update(1, "car", (100.0, 150.0))
    assert forward.totals()[IN] == 1
    assert backward.totals()[OUT] == 1


def test_diagonal_line_counts_correctly():
    counter = LineCounter(((0.0, 0.0), (100.0, 100.0)))
    counter.update(1, "car", (80.0, 20.0))
    assert counter.update(1, "car", (20.0, 80.0)) in (IN, OUT)
    assert sum(counter.totals().values()) == 1
