# segtrack - YOLO instance segmentation with multi-object tracking
# Copyright (C) 2026 Bagaskara Amukti Palapa
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Rewritten from the GPL-3.0 project by Muhammad Moin Faisal; see NOTICE.

"""segtrack: YOLO instance segmentation with multi-object tracking.

A small, readable rewrite of the DeepSORT segmentation-tracking demo by
Muhammad Moin Faisal, rebuilt on the public ultralytics API. See NOTICE
for the full attribution chain and docs/CARA-KERJA.md for a walkthrough
of how a single frame travels through the pipeline.
"""

__all__ = [
    "AUTHOR",
    "COPYRIGHT_YEAR",
    "DEFAULT_WATERMARK",
    "LICENSE",
    "__version__",
]

__version__ = "1.0.0"

AUTHOR = "Bagaskara Amukti Palapa"
COPYRIGHT_YEAR = 2026
LICENSE = "AGPL-3.0"

# Stamped into rendered video by default. ASCII only: the OpenCV Hershey
# fonts have no glyph for (c) or for a middle dot, and draw a hollow box
# in their place.
DEFAULT_WATERMARK = f"segtrack v{__version__} - (c) {COPYRIGHT_YEAR} {AUTHOR} - {LICENSE}"
