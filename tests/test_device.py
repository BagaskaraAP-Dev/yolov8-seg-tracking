# segtrack - YOLO instance segmentation with multi-object tracking
# Copyright (C) 2026 Bagaskara Amukti Palapa
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Rewritten from the GPL-3.0 project by Muhammad Moin Faisal; see NOTICE.

"""Device resolution, with the hardware faked.

This is the regression test for the upstream bug that made the whole
project unusable without an NVIDIA GPU: the tracker was built with
use_cuda=True hard-coded.
"""

from __future__ import annotations

import pytest

from segtrack.device import DeviceProbe, available_devices, resolve_device


class FakeProbe(DeviceProbe):
    def __init__(self, cuda=False, mps=False):
        self._cuda = cuda
        self._mps = mps

    def has_cuda(self):
        return self._cuda

    def has_mps(self):
        return self._mps


CPU_ONLY = FakeProbe()
WITH_CUDA = FakeProbe(cuda=True)
WITH_MPS = FakeProbe(mps=True)


def test_auto_prefers_cuda_when_present():
    assert resolve_device("auto", WITH_CUDA) == "cuda"


def test_auto_uses_mps_when_there_is_no_cuda():
    assert resolve_device("auto", WITH_MPS) == "mps"


def test_auto_falls_back_to_cpu():
    assert resolve_device("auto", CPU_ONLY) == "cpu"


def test_none_behaves_like_auto():
    assert resolve_device(None, CPU_ONLY) == "cpu"


def test_empty_string_behaves_like_auto():
    assert resolve_device("  ", CPU_ONLY) == "cpu"


def test_explicit_cpu_is_honoured_even_with_a_gpu():
    assert resolve_device("cpu", WITH_CUDA) == "cpu"


def test_bare_index_means_a_cuda_device():
    assert resolve_device("1", WITH_CUDA) == "cuda:1"


def test_cuda_index_is_preserved():
    assert resolve_device("cuda:2", WITH_CUDA) == "cuda:2"


def test_case_is_ignored():
    assert resolve_device("CUDA", WITH_CUDA) == "cuda"


def test_cuda_request_on_a_cpu_machine_falls_back_instead_of_raising():
    # A long render should not die at frame zero because the machine
    # changed. Upstream raised here, with no way to opt out.
    warnings = []
    assert resolve_device("cuda", CPU_ONLY, warn=warnings.append) == "cpu"
    assert len(warnings) == 1
    assert "falling back to cpu" in warnings[0]


def test_mps_request_without_mps_falls_back():
    warnings = []
    assert resolve_device("mps", CPU_ONLY, warn=warnings.append) == "cpu"
    assert warnings


def test_fallback_without_a_warn_callback_is_silent_not_fatal():
    assert resolve_device("cuda", CPU_ONLY) == "cpu"


@pytest.mark.parametrize("requested", ["tpu", "gpu", "cuda!", "-1"])
def test_unknown_device_names_are_rejected(requested):
    with pytest.raises(ValueError, match="unknown device"):
        resolve_device(requested, CPU_ONLY)


def test_available_devices_always_ends_with_cpu():
    assert available_devices(WITH_CUDA) == ["cuda", "cpu"]
    assert available_devices(FakeProbe(cuda=True, mps=True)) == ["cuda", "mps", "cpu"]
    assert available_devices(CPU_ONLY) == ["cpu"]


def test_real_probe_never_raises_without_torch():
    # The probe is imported at module load on machines that may not have
    # torch yet, so it has to answer rather than explode.
    probe = DeviceProbe()
    assert isinstance(probe.has_cuda(), bool)
    assert isinstance(probe.has_mps(), bool)
