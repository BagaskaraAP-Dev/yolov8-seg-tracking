# segtrack - YOLO instance segmentation with multi-object tracking
# Copyright (C) 2026 Bagaskara Amukti Palapa
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Rewritten from the GPL-3.0 project by Muhammad Moin Faisal; see NOTICE.

"""Argument wiring.

These need OpenCV, because the drawing module imports it, so they skip
rather than fail on a bare checkout. They do not need a model, weights,
a video or a GPU: nothing here runs inference.
"""

from __future__ import annotations

import pytest

pytest.importorskip("cv2", reason="opencv-python is needed to import segtrack.draw")

from segtrack.cli import build_parser
from segtrack.tracker import Settings


def parse(*argv):
    return build_parser().parse_args(["--source", "clip.mp4", *argv])


def test_source_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_defaults_are_a_runnable_configuration():
    args = parse()
    assert args.model == "yolov8n-seg.pt"
    assert args.device == "auto"
    assert args.tracker == "botsort.yaml"
    assert args.line is None


def test_watermark_is_on_by_default():
    # The authorship line is the default because rendered video leaves
    # the repository and should still say who made it.
    args = parse()
    assert "AGPL-3.0" in args.watermark
    assert not args.no_watermark


def test_watermark_can_be_replaced_or_removed():
    assert parse("--watermark", "my studio").watermark == "my studio"
    assert parse("--no-watermark").no_watermark is True


def test_unknown_tracker_is_rejected_at_parse_time():
    with pytest.raises(SystemExit):
        parse("--tracker", "deepsort.yaml")


def test_unknown_anchor_is_rejected_at_parse_time():
    with pytest.raises(SystemExit):
        parse("--anchor", "top")


def test_settings_accepts_every_parsed_field():
    # Guards against the parser and the dataclass drifting apart.
    from segtrack.cli import _settings_from

    settings = _settings_from(parse("--classes", "car", "--line", "0,10,100,10"))
    assert isinstance(settings, Settings)
    assert settings.classes == "car"
    assert settings.line == "0,10,100,10"
    assert settings.strict_line is True


def test_loose_line_flips_strict_mode():
    from segtrack.cli import _settings_from

    assert _settings_from(parse("--loose-line")).strict_line is False


def test_trail_zero_keeps_the_store_but_stops_drawing():
    # Counting needs the previous position even with trails switched off,
    # so the store stays alive at its minimum usable length.
    from segtrack.cli import _settings_from

    settings = _settings_from(parse("--trail", "0"))
    assert settings.draw_trails is False
    assert settings.trail >= 2


def test_trail_length_is_passed_through():
    from segtrack.cli import _settings_from

    settings = _settings_from(parse("--trail", "120"))
    assert settings.trail == 120
    assert settings.draw_trails is True
