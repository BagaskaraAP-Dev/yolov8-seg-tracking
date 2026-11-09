# segtrack - YOLO instance segmentation with multi-object tracking
# Copyright (C) 2026 Bagaskara Amukti Palapa
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Rewritten from the GPL-3.0 project by Muhammad Moin Faisal; see NOTICE.

"""Allows `python -m segtrack` as well as the installed `segtrack` command."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
