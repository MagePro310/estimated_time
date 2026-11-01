"""Unit tests for backend specification extractor and FakeBelem properties."""

import pytest
from qcirc_time_estimator.backends import load_fake_belem_info


def test_fake_belem_loader():
    """Verify that FakeBelem backend info is correctly loaded and populated."""
    backend, info = load_fake_belem_info()

    assert backend.name == "fake_belem"
    assert info.num_qubits == 5
    assert info.dt > 0
    assert "cx" in info.basis_gates or "ecr" in info.basis_gates or "cz" in info.basis_gates

    # Topology checks
    assert len(info.coupling_edges) > 0
    assert (0, 1) in info.coupling_edges or (1, 0) in info.coupling_edges

    # Timing metrics checks
    assert info.avg_cx_duration_sec > 0
    assert info.avg_1q_duration_sec > 0
    assert info.avg_measure_duration_sec > 0

    # Shortest path checks
    assert info.get_distance(0, 0) == 0
    assert info.get_distance(0, 1) == 1
    assert info.get_distance(0, 4) >= 2  # Belem T-shape: 0-1-2-3-4
