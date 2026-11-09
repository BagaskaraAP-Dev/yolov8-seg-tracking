# segtrack - YOLO instance segmentation with multi-object tracking
# Copyright (C) 2026 Bagaskara Amukti Palapa
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Rewritten from the GPL-3.0 project by Muhammad Moin Faisal; see NOTICE.

"""Compute-device selection.

The project this replaces constructed its tracker with use_cuda=True
hard-coded, so it raised on any machine without a CUDA GPU even though
its own config file exposed a device option. Resolution here is
explicit, honours an Apple Silicon GPU, and falls back to CPU.
"""

from __future__ import annotations

from collections.abc import Callable

__all__ = ["DeviceProbe", "available_devices", "resolve_device"]


class DeviceProbe:
    """What hardware is usable. Separated out so tests can fake it."""

    def has_cuda(self) -> bool:
        try:
            import torch
        except ImportError:
            return False
        return bool(torch.cuda.is_available())

    def has_mps(self) -> bool:
        try:
            import torch
        except ImportError:
            return False
        backend = getattr(torch.backends, "mps", None)
        return bool(backend is not None and backend.is_available())


def available_devices(probe: DeviceProbe | None = None) -> list[str]:
    """Devices this machine can actually run on, best first."""
    probe = probe or DeviceProbe()
    found = []
    if probe.has_cuda():
        found.append("cuda")
    if probe.has_mps():
        found.append("mps")
    found.append("cpu")
    return found


def resolve_device(
    requested: str | None = None,
    probe: DeviceProbe | None = None,
    warn: Callable[[str], None] | None = None,
) -> str:
    """Turn a --device value into a device string ultralytics accepts.

    None or "auto" picks the best device present. An explicit request is
    honoured when the hardware is there; when it is not, this falls back
    to CPU and reports it through warn rather than raising, so a long
    render does not die on a machine mismatch.

    Accepts "cpu", "cuda", "cuda:1", "mps", and a bare GPU index ("0").
    """
    probe = probe or DeviceProbe()
    choice = (requested or "auto").strip().lower()

    if choice in ("auto", ""):
        return available_devices(probe)[0]

    if choice == "cpu":
        return "cpu"

    if choice.isdigit():
        choice = f"cuda:{choice}"

    family = choice.split(":", 1)[0]

    if family not in ("cuda", "mps"):
        raise ValueError(
            f"unknown device {requested!r}; expected auto, cpu, cuda[:N], mps or a GPU index"
        )

    if family == "cuda" and not probe.has_cuda():
        if warn:
            warn(f"device {choice!r} requested but no CUDA GPU is visible, falling back to cpu")
        return "cpu"

    if family == "mps" and not probe.has_mps():
        if warn:
            warn(f"device {choice!r} requested but MPS is unavailable, falling back to cpu")
        return "cpu"

    return choice
