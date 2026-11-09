# segtrack - YOLO instance segmentation with multi-object tracking
# Copyright (C) 2026 Bagaskara Amukti Palapa
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Rewritten from the GPL-3.0 project by Muhammad Moin Faisal; see NOTICE.

"""Per-track motion trails.

The project this replaces kept trails in a module-level dict named
data_deque. Two consequences: trails from one video leaked into the next
run inside the same process, and the only place stale ids were released
was inside the drawing call, which crashed when it was handed the
default identities=None.

A trail store is an object here, so a pipeline owns its own state and
can reset it between sources.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

__all__ = ["TrailStore"]

Point = tuple[int, int]


class TrailStore:
    """Fixed-length history of positions, keyed by track id."""

    def __init__(self, maxlen: int = 64) -> None:
        if maxlen < 2:
            raise ValueError(f"maxlen must be at least 2 to draw a segment, got {maxlen}")
        self._maxlen = maxlen
        self._points: dict[int, deque] = {}

    @property
    def maxlen(self) -> int:
        return self._maxlen

    def update(self, track_id: int, point: Point) -> None:
        """Record where a track is now. Newest point first."""
        trail = self._points.get(track_id)
        if trail is None:
            trail = deque(maxlen=self._maxlen)
            self._points[track_id] = trail
        trail.appendleft(point)

    def points(self, track_id: int) -> list[Point]:
        """Positions for one track, newest first. Empty for an unknown id."""
        return list(self._points.get(track_id, ()))

    def prune(self, active_ids: Iterable[int]) -> int:
        """Drop trails whose track is gone. Returns how many were dropped.

        Taking a set here is both a correctness fix and a speed one:
        upstream tested membership against a NumPy array of ids, which
        is a linear scan per key, every frame.
        """
        active = set(active_ids)
        stale = [track_id for track_id in self._points if track_id not in active]
        for track_id in stale:
            del self._points[track_id]
        return len(stale)

    def reset(self) -> None:
        """Forget every track. Call this between input sources."""
        self._points.clear()

    def __len__(self) -> int:
        return len(self._points)

    def __contains__(self, track_id: object) -> bool:
        return track_id in self._points
