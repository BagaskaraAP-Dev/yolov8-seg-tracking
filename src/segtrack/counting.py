# segtrack - YOLO instance segmentation with multi-object tracking
# Copyright (C) 2026 Bagaskara Amukti Palapa
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Rewritten from the GPL-3.0 project by Muhammad Moin Faisal; see NOTICE.

"""Line-crossing counter.

The README of the project this replaces advertised vehicle counting and
showed a screenshot of it, but the counting code was not in the
repository: the counting line was a commented-out cv2.line call and
nothing consumed it. This module is that missing feature.

The geometry is exact rather than a distance threshold. A track counts
when the segment between its previous and current position actually
intersects the finite counting line, so a fast object cannot jump the
line between two frames without being counted, and an object loitering
near the line is not counted repeatedly.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

__all__ = ["IN", "OUT", "LineCounter", "parse_line", "segments_intersect", "side_of"]

Point = tuple[float, float]
Line = tuple[Point, Point]

IN = "in"
OUT = "out"


def parse_line(spec: str) -> Line:
    """Parse "x1,y1,x2,y2" in pixels into a counting line."""
    parts = [part.strip() for part in str(spec).replace(";", ",").split(",")]
    parts = [part for part in parts if part]
    if len(parts) != 4:
        raise ValueError(
            f"counting line needs exactly 4 numbers as x1,y1,x2,y2, got {len(parts)}: {spec!r}"
        )
    try:
        x1, y1, x2, y2 = (float(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"counting line must be numeric, got {spec!r}") from exc
    if (x1, y1) == (x2, y2):
        raise ValueError(f"counting line has zero length: {spec!r}")
    return ((x1, y1), (x2, y2))


def _cross(origin: Point, first: Point, second: Point) -> float:
    """Z component of the cross product of origin->first and origin->second."""
    return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (
        second[0] - origin[0]
    )


def side_of(line: Line, point: Point) -> int:
    """Which side of the directed line a point lies on: 1, -1, or 0 when on it."""
    value = _cross(line[0], line[1], point)
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _on_segment(start: Point, end: Point, point: Point) -> bool:
    """True when a collinear point lies within the segment's bounding box."""
    return min(start[0], end[0]) <= point[0] <= max(start[0], end[0]) and min(
        start[1], end[1]
    ) <= point[1] <= max(start[1], end[1])


def segments_intersect(a_start: Point, a_end: Point, b_start: Point, b_end: Point) -> bool:
    """True when segment a and segment b share at least one point.

    Collinear overlap counts as an intersection, which is what we want:
    an object sliding exactly along the counting line has reached it.
    """
    d1 = _cross(b_start, b_end, a_start)
    d2 = _cross(b_start, b_end, a_end)
    d3 = _cross(a_start, a_end, b_start)
    d4 = _cross(a_start, a_end, b_end)

    if ((d1 > 0) != (d2 > 0) or (d1 < 0) != (d2 < 0)) and (
        (d3 > 0) != (d4 > 0) or (d3 < 0) != (d4 < 0)
    ):
        return True

    if d1 == 0 and _on_segment(b_start, b_end, a_start):
        return True
    if d2 == 0 and _on_segment(b_start, b_end, a_end):
        return True
    if d3 == 0 and _on_segment(a_start, a_end, b_start):
        return True
    return bool(d4 == 0 and _on_segment(a_start, a_end, b_end))


class LineCounter:
    """Counts tracks crossing one line, split by direction and class.

    Direction is relative to the line as it was given. A track moving
    from the negative side to the positive side of the directed line
    p1 -> p2 counts as IN, the opposite as OUT. Swap the two endpoints
    on the command line to swap the labels.
    """

    def __init__(self, line: Line) -> None:
        self.line = line
        self._last_side: dict[int, int] = {}
        self._counts: dict[str, Counter] = {IN: Counter(), OUT: Counter()}

    def update(self, track_id: int, class_name: str, point: Point) -> str | None:
        """Feed one track's current position. Returns IN, OUT, or None.

        The first sighting of a track only records which side it is on:
        an object that is already past the line when it appears has not
        been seen to cross it, so counting it would be an invention.
        """
        previous_side = self._last_side.get(track_id)
        current_side = side_of(self.line, point)

        if previous_side is None:
            # A track first seen exactly on the line has no side yet; wait
            # for the next frame rather than guessing one.
            if current_side != 0:
                self._last_side[track_id] = current_side
            return None

        if current_side == 0 or current_side == previous_side:
            return None

        self._last_side[track_id] = current_side

        direction = IN if previous_side < 0 else OUT
        self._counts[direction][class_name] += 1
        return direction

    def crossed(
        self, track_id: int, class_name: str, previous: Point, current: Point
    ) -> str | None:
        """Strict variant that also requires hitting the finite segment.

        update() alone treats the line as infinite, which over-counts an
        object passing wide of a short line. Use this when the line only
        spans part of the frame, for example one lane of a road.
        """
        if not segments_intersect(previous, current, self.line[0], self.line[1]):
            self._last_side[track_id] = side_of(self.line, current) or self._last_side.get(
                track_id, 0
            )
            return None
        return self.update(track_id, class_name, current)

    def forget(self, active_ids: Iterable[int]) -> int:
        """Drop remembered sides for tracks that are gone."""
        active = set(active_ids)
        stale = [track_id for track_id in self._last_side if track_id not in active]
        for track_id in stale:
            del self._last_side[track_id]
        return len(stale)

    def per_class(self, direction: str) -> dict[str, int]:
        """Counts for one direction, keyed by class name."""
        if direction not in self._counts:
            raise KeyError(f"unknown direction {direction!r}; expected {IN!r} or {OUT!r}")
        return dict(self._counts[direction])

    def totals(self) -> dict[str, int]:
        """Total crossings per direction."""
        return {direction: sum(counts.values()) for direction, counts in self._counts.items()}

    def reset(self) -> None:
        """Clear counts and remembered sides. Call this between sources."""
        self._last_side.clear()
        for counts in self._counts.values():
            counts.clear()
