"""Tests for FakeBelem backend loader and BackendInfo abstraction."""

import pytest
from qcirc_time_estimator.backends.fake_belem import load_fake_belem_info


def test_fake_belem_loader():
    backend, info = load_fake_belem_info()

    assert backend.name == "fake_belem"
    assert info.num_qubits == 5
    assert info.dt > 0
    assert "cx" in info.basis_gates
    assert "sx" in info.basis_gates
    assert "rz" in info.basis_gates

    # Verify topology shortest paths
    assert info.get_distance(0, 0) == 0
    assert info.get_distance(0, 1) == 1
    assert info.get_distance(0, 4) == 3  # 0 -> 1 -> 3 -> 4
    assert info.get_distance(2, 4) == 3  # 2 -> 1 -> 3 -> 4

    # Verify duration extraction
    assert info.avg_1q_duration_sec > 0
    assert info.avg_cx_duration_sec > 0
    assert info.avg_measure_duration_sec > 0
    assert info.avg_cx_duration_sec > info.avg_1q_duration_sec

