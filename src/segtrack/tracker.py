# segtrack - YOLO instance segmentation with multi-object tracking
# Copyright (C) 2026 Bagaskara Amukti Palapa
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Rewritten from the GPL-3.0 project by Muhammad Moin Faisal; see NOTICE.

"""The tracking pipeline.

This is the part that replaces the 13,369 lines of vendored ultralytics
8.0.3 in the project this descends from. That snapshot existed because
the upstream author needed to edit a predictor class to inject DeepSORT
into it. Modern ultralytics does tracking itself, through BoT-SORT and
ByteTrack, so the integration is an API call and this file can stay
small enough to read in one sitting.

What that buys, besides the line count:

- No deep_sort_pytorch download from a third-party Google Drive folder,
  and no ReID checkpoint loaded through an unguarded torch.load.
- No hydra. The CLI is argparse, so it is not pinned to an ultralytics
  version that stopped existing in early 2023.
- Masks and boxes come from one Results object, so a mask is attached to
  the track id that owns it instead of being drawn from a separate list
  that could fall out of step with the detections.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np

from . import draw
from .classes import parse_classes
from .colors import class_color
from .counting import IN, OUT, LineCounter, parse_line
from .device import resolve_device
from .overlay import blend_masks
from .points import anchor_point
from .trails import TrailStore

__all__ = ["FrameResult", "SegTracker", "Settings", "Track", "anchor_point"]

Point = tuple[int, int]


@dataclass
class Settings:
    """Everything the pipeline needs. The CLI fills this in."""

    source: str
    model: str = "yolov8n-seg.pt"
    device: str | None = None
    conf: float = 0.25
    iou: float = 0.7
    imgsz: int = 640
    classes: str | None = None
    tracker: str = "botsort.yaml"
    trail: int = 64
    draw_trails: bool = True
    anchor: str = "bottom"
    masks: bool = True
    mask_alpha: float = 0.45
    line: str | None = None
    strict_line: bool = True
    watermark: str | None = None
    max_frames: int | None = None
    box_thickness: int = 2
    hud: bool = True


@dataclass
class Track:
    """One tracked instance in one frame."""

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    color: tuple[int, int, int]
    anchor: Point


@dataclass
class FrameResult:
    """An annotated frame plus what was in it."""

    index: int
    frame: np.ndarray
    tracks: list[Track] = field(default_factory=list)
    crossings: list[tuple[int, str, str]] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    fps: float = 0.0


class SegTracker:
    """Runs segmentation plus tracking over a source and annotates frames."""

    def __init__(self, settings: Settings, log=None) -> None:
        self.settings = settings
        self._log = log or (lambda message: None)

        self.device = resolve_device(settings.device, warn=self._log)

        # Imported here, not at module import, so --help and the unit
        # tests do not pay for loading torch.
        from ultralytics import YOLO

        self.model = YOLO(settings.model)
        self.names: dict[int, str] = dict(self.model.names)
        self.classes = parse_classes(settings.classes, self.names)

        self.trails = TrailStore(maxlen=settings.trail)
        self.counter = LineCounter(parse_line(settings.line)) if settings.line else None
        self._warned_mask_mismatch = False

    def _color_for(self, class_id: int) -> tuple[int, int, int]:
        """One colour per class index, used for this track's mask and box."""
        return class_color(class_id)

    def _label(self, track_id: int, class_name: str) -> str:
        return f"{track_id} {class_name}"

    def reset(self) -> None:
        """Clear per-source state. State is per instance, not module level."""
        self.trails.reset()
        if self.counter:
            self.counter.reset()

    def run(self) -> Iterator[FrameResult]:
        """Yield one annotated FrameResult per frame of the source."""
        settings = self.settings
        stream = self.model.track(
            source=settings.source,
            stream=True,
            tracker=settings.tracker,
            conf=settings.conf,
            iou=settings.iou,
            imgsz=settings.imgsz,
            device=self.device,
            classes=self.classes,
            retina_masks=True,
            verbose=False,
        )

        started = time.perf_counter()
        for index, result in enumerate(stream):
            if settings.max_frames is not None and index >= settings.max_frames:
                break

            frame = result.orig_img.copy()
            tracks = self._tracks_from(result)

            if settings.masks and tracks:
                frame = self._paint_masks(frame, result, tracks)

            crossings = self._count(tracks)

            if settings.draw_trails:
                for track in tracks:
                    draw.draw_trail(frame, self.trails.points(track.track_id), track.color)

            for track in tracks:
                draw.draw_box(
                    frame,
                    track.xyxy,
                    self._label(track.track_id, track.class_name),
                    track.color,
                    settings.box_thickness,
                )

            if self.counter:
                draw.draw_counting_line(frame, self.counter.line)

            elapsed = time.perf_counter() - started
            fps = (index + 1) / elapsed if elapsed > 0 else 0.0

            counts = self.counter.totals() if self.counter else {}
            if settings.hud:
                draw.draw_hud(frame, self._hud_rows(len(tracks), fps, counts))
            if settings.watermark:
                draw.draw_watermark(frame, settings.watermark)

            yield FrameResult(
                index=index,
                frame=frame,
                tracks=tracks,
                crossings=crossings,
                counts=counts,
                fps=fps,
            )

    def _hud_rows(self, visible: int, fps: float, counts: dict[str, int]) -> list[str]:
        rows = [f"frame fps {fps:.1f}", f"tracking {visible}"]
        if self.counter:
            per_class = {
                direction: self.counter.per_class(direction) for direction in (IN, OUT)
            }
            rows.extend(draw.format_counts(counts, per_class))
        return rows

    def _tracks_from(self, result) -> list[Track]:
        """Pull tracked instances out of one ultralytics Results object."""
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.id is None or len(boxes) == 0:
            # No tracks confirmed this frame. Release every trail rather
            # than leaving stale paths on screen.
            self.trails.prune(())
            if self.counter:
                self.counter.forget(())
            return []

        ids = [int(value) for value in boxes.id.int().cpu().tolist()]
        class_ids = [int(value) for value in boxes.cls.int().cpu().tolist()]
        confidences = [float(value) for value in boxes.conf.cpu().tolist()]
        coordinates = boxes.xyxy.cpu().numpy()

        tracks: list[Track] = []
        for position, track_id in enumerate(ids):
            class_id = class_ids[position]
            class_name = self.names.get(class_id, str(class_id))
            xyxy = tuple(float(value) for value in coordinates[position])
            tracks.append(
                Track(
                    track_id=track_id,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidences[position],
                    xyxy=xyxy,
                    color=self._color_for(class_id),
                    anchor=anchor_point(xyxy, self.settings.anchor),
                )
            )

        self.trails.prune(ids)
        if self.counter:
            self.counter.forget(ids)
        return tracks

    def _paint_masks(self, frame: np.ndarray, result, tracks: Sequence[Track]) -> np.ndarray:
        """Tint each instance mask with the colour of the track that owns it."""
        masks = getattr(result, "masks", None)
        if masks is None or masks.data is None:
            return frame

        data = masks.data.cpu().numpy()
        if len(data) != len(tracks):
            # Index alignment between masks and tracked boxes is the whole
            # basis for colouring a mask by its track. If a version ever
            # breaks that, drawing nothing beats drawing the wrong mask on
            # the wrong object.
            if not self._warned_mask_mismatch:
                self._log(
                    f"masks ({len(data)}) and tracks ({len(tracks)}) are not aligned; "
                    "skipping mask overlay"
                )
                self._warned_mask_mismatch = True
            return frame

        resized = draw.resize_masks(data, frame.shape[:2])
        colors = [track.color for track in tracks]
        return blend_masks(frame, resized, colors, alpha=self.settings.mask_alpha)

    def _count(self, tracks: Sequence[Track]) -> list[tuple[int, str, str]]:
        """Update trails, then count crossings. Returns (id, class, direction)."""
        crossings: list[tuple[int, str, str]] = []
        for track in tracks:
            history = self.trails.points(track.track_id)
            previous = history[0] if history else None
            self.trails.update(track.track_id, track.anchor)

            if not self.counter or previous is None:
                continue

            if self.settings.strict_line:
                direction = self.counter.crossed(
                    track.track_id, track.class_name, previous, track.anchor
                )
            else:
                direction = self.counter.update(
                    track.track_id, track.class_name, track.anchor
                )
            if direction:
                crossings.append((track.track_id, track.class_name, direction))
        return crossings
