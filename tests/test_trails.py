# segtrack - YOLO instance segmentation with multi-object tracking
# Copyright (C) 2026 Bagaskara Amukti Palapa
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Rewritten from the GPL-3.0 project by Muhammad Moin Faisal; see NOTICE.

"""Trail state must be per instance and must release dead tracks.

These cover the two upstream defects directly: state that outlived the
video it belonged to, and a prune path that crashed on its own default
argument.
"""

from __future__ import annotations

import pytest

from segtrack.trails import TrailStore


def test_newest_point_comes_first():
    store = TrailStore(maxlen=8)
    for x in range(4):
        store.update(1, (x, 0))
    assert store.points(1) == [(3, 0), (2, 0), (1, 0), (0, 0)]


def test_trail_is_capped_at_maxlen():
    store = TrailStore(maxlen=3)
    for x in range(10):
        store.update(7, (x, 0))
    points = store.points(7)
    assert len(points) == 3
    assert points[0] == (9, 0)


def test_unknown_track_has_no_points():
    assert TrailStore().points(999) == []


def test_prune_drops_only_inactive_tracks():
    store = TrailStore()
    for track_id in (1, 2, 3):
        store.update(track_id, (0, 0))

    dropped = store.prune([1, 3])

    assert dropped == 1
    assert 2 not in store
    assert 1 in store and 3 in store


def test_prune_with_no_active_tracks_clears_everything():
    # Upstream passed its identity array straight through, and the default
    # was None, so this call raised TypeError instead of clearing.
    store = TrailStore()
    store.update(5, (1, 1))
    assert store.prune([]) == 1
    assert len(store) == 0


def test_prune_accepts_any_iterable():
    store = TrailStore()
    store.update(5, (1, 1))
    store.update(6, (1, 1))
    assert store.prune(iter([5])) == 1


def test_state_does_not_leak_between_instances():
    first = TrailStore()
    first.update(1, (0, 0))
    second = TrailStore()
    assert len(second) == 0


def test_reset_clears_state_for_the_next_source():
    store = TrailStore()
    store.update(1, (0, 0))
    store.reset()
    assert len(store) == 0
    assert store.points(1) == []


def test_maxlen_below_two_is_rejected():
    # One point cannot describe a segment, so a trail of 1 is a silent
    # no-op rather than a short trail.
    with pytest.raises(ValueError, match="at least 2"):
        TrailStore(maxlen=1)
