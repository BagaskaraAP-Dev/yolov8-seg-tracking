# segtrack - YOLO instance segmentation with multi-object tracking
# Copyright (C) 2026 Bagaskara Amukti Palapa
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Rewritten from the GPL-3.0 project by Muhammad Moin Faisal; see NOTICE.

"""Command line entry point.

The project this replaces drove its predictor through hydra, with a
130-line default.yaml that mixed training, validation and prediction
settings together, and had to be launched from inside the
ultralytics/yolo/v8/segment directory because it loaded a config by
relative path. hydra was removed from ultralytics weeks after that
snapshot was taken, which is a large part of why that code cannot be
moved to a current version.

argparse is in the standard library and will not do that.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import AUTHOR, DEFAULT_WATERMARK, LICENSE, __version__
from .tracker import SegTracker, Settings

__all__ = ["build_parser", "main"]

_FOURCC = {".mp4": "mp4v", ".m4v": "mp4v", ".avi": "MJPG", ".mkv": "mp4v", ".mov": "mp4v"}


def _log(message: str) -> None:
    print(f"segtrack: {message}", file=sys.stderr)


def _normalise_source(source: str) -> str | int:
    """A bare number means a camera index, anything else is a path or URL."""
    return int(source) if source.isdigit() else source


def _source_fps(source: str | int, fallback: float = 30.0) -> float:
    """Read the source frame rate so a render is not silently resampled."""
    import cv2

    capture = cv2.VideoCapture(source)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        capture.release()
    return fps if 0.0 < fps < 240.0 else fallback


class VideoWriter:
    """Writes annotated frames, opening on the first one to learn its size."""

    def __init__(self, path: Path, fps: float) -> None:
        self.path = path
        self.fps = fps
        self._writer = None

    def write(self, frame) -> None:
        import cv2

        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*_FOURCC.get(self.path.suffix.lower(), "mp4v"))
            height, width = frame.shape[:2]
            self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (width, height))
            if not self._writer.isOpened():
                raise RuntimeError(
                    f"could not open {self.path} for writing; "
                    f"try a .mp4 or .avi path on a local disk"
                )
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="segtrack",
        description=(
            "Segment, track and count objects in a video, a camera stream or an image. "
            "Weights are downloaded by ultralytics on first use."
        ),
        epilog=(
            f"(c) {AUTHOR}, licensed {LICENSE}. "
            "Credit for the original design: see NOTICE."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"segtrack {__version__}")

    source = parser.add_argument_group("input")
    source.add_argument(
        "--source",
        required=True,
        help="video path, image path, directory, stream URL, or a camera index such as 0",
    )
    source.add_argument(
        "--model",
        default="yolov8n-seg.pt",
        help=(
            "any ultralytics segmentation checkpoint, "
            "for example yolov8n-seg.pt or yolo11n-seg.pt"
        ),
    )
    source.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:1, mps, or a GPU index; auto falls back to cpu",
    )
    source.add_argument("--max-frames", type=int, default=None, help="stop after N frames")

    detection = parser.add_argument_group("detection and tracking")
    detection.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    detection.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    detection.add_argument("--imgsz", type=int, default=640, help="inference image size")
    detection.add_argument(
        "--classes",
        default=None,
        help="restrict to these classes, by index or name: '0,2,3' or 'person,car'",
    )
    detection.add_argument(
        "--tracker",
        default="botsort.yaml",
        choices=["botsort.yaml", "bytetrack.yaml"],
        help="botsort.yaml keeps ids through longer occlusions, bytetrack.yaml is faster",
    )

    drawing = parser.add_argument_group("drawing")
    drawing.add_argument(
        "--trail", type=int, default=64, help="trail length in frames, 0 to disable"
    )
    drawing.add_argument(
        "--anchor",
        default="bottom",
        choices=["bottom", "center"],
        help="the point that represents an object for its trail and for counting",
    )
    drawing.add_argument("--no-masks", action="store_true", help="draw boxes only, no mask tint")
    drawing.add_argument("--mask-alpha", type=float, default=0.45, help="mask tint strength")
    drawing.add_argument("--box-thickness", type=int, default=2, help="box line width in pixels")
    drawing.add_argument("--no-hud", action="store_true", help="hide the status panel")
    drawing.add_argument(
        "--watermark",
        default=DEFAULT_WATERMARK,
        help="authorship and licence line stamped into rendered frames",
    )
    drawing.add_argument(
        "--no-watermark", action="store_true", help="render without the authorship line"
    )

    counting = parser.add_argument_group("counting")
    counting.add_argument(
        "--line",
        default=None,
        help="counting line as x1,y1,x2,y2 in pixels, for example 0,600,1280,600",
    )
    counting.add_argument(
        "--loose-line",
        action="store_true",
        help="count any side change, even wide of the line's endpoints",
    )

    output = parser.add_argument_group("output")
    output.add_argument("--save", type=Path, default=None, help="write the annotated video here")
    output.add_argument("--show", action="store_true", help="preview in a window, q to quit")
    output.add_argument("--quiet", action="store_true", help="only print the final summary")

    return parser


def _settings_from(args: argparse.Namespace) -> Settings:
    return Settings(
        source=str(args.source),
        model=args.model,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        classes=args.classes,
        tracker=args.tracker,
        # The store is always kept: counting needs the previous position
        # even when the trail is not drawn.
        trail=max(2, args.trail or 2),
        draw_trails=bool(args.trail and args.trail >= 2),
        anchor=args.anchor,
        masks=not args.no_masks,
        mask_alpha=args.mask_alpha,
        line=args.line,
        strict_line=not args.loose_line,
        watermark=None if args.no_watermark else args.watermark,
        max_frames=args.max_frames,
        box_thickness=args.box_thickness,
        hud=not args.no_hud,
    )


def _summary(frames: int, tracker: SegTracker) -> list[str]:
    rows = [f"frames processed: {frames}"]
    if tracker.counter:
        totals = tracker.counter.totals()
        rows.append(f"crossings in: {totals.get('in', 0)}   out: {totals.get('out', 0)}")
        for direction in ("in", "out"):
            per_class = tracker.counter.per_class(direction)
            for name, value in sorted(per_class.items(), key=lambda item: (-item[1], item[0])):
                rows.append(f"  {direction:<3} {name}: {value}")
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.trail and args.trail < 0:
        print("segtrack: --trail cannot be negative", file=sys.stderr)
        return 2

    try:
        tracker = SegTracker(_settings_from(args), log=_log)
    except ValueError as exc:
        print(f"segtrack: {exc}", file=sys.stderr)
        return 2
    except ImportError as exc:
        print(
            f"segtrack: {exc}\nInstall the dependencies with: pip install -e .",
            file=sys.stderr,
        )
        return 1

    if not args.quiet:
        _log(f"model {args.model} on {tracker.device}, tracker {args.tracker}")
        if tracker.classes is not None:
            names = ", ".join(tracker.names[index] for index in tracker.classes)
            _log(f"restricted to: {names}")

    writer = None
    if args.save:
        writer = VideoWriter(args.save, _source_fps(_normalise_source(str(args.source))))

    frames = 0
    interrupted = False
    try:
        import cv2

        for result in tracker.run():
            frames += 1
            if writer:
                writer.write(result.frame)
            if args.show:
                cv2.imshow("segtrack", result.frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if not args.quiet:
                for track_id, class_name, direction in result.crossings:
                    _log(f"frame {result.index}: {class_name} {track_id} crossed {direction}")
    except KeyboardInterrupt:
        interrupted = True
        _log("interrupted, finishing the file that is already written")
    finally:
        if writer:
            writer.close()
        if args.show:
            import cv2

            cv2.destroyAllWindows()

    for row in _summary(frames, tracker):
        print(row)
    if args.save and frames:
        print(f"saved: {args.save}")

    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
