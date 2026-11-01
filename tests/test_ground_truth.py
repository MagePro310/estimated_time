"""Unit tests for exact duration calculation using Qiskit scheduler on FakeBelem."""

import pytest
from qiskit import QuantumCircuit
from qcirc_time_estimator.backends import load_fake_belem_info
from qcirc_time_estimator.ground_truth import compute_exact_duration


def test_compute_exact_duration_bell_state():
    """Verify ground truth computation for Bell state."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    backend, _ = load_fake_belem_info()
    gt = compute_exact_duration(qc, backend=backend, optimization_level=1)

    assert gt.duration_sec > 0
    assert gt.duration_dt > 0
    assert gt.transpile_time_sec > 0
    assert gt.depth > 0
    assert "cx" in gt.gate_counts or "ecr" in gt.gate_counts or "cz" in gt.gate_counts


def test_scheduling_method_asap():
    """Verify scheduling with ASAP method produces consistent durations."""
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.cx(0, 1)

    backend, _ = load_fake_belem_info()
    gt = compute_exact_duration(qc, backend=backend, scheduling_method="asap")

    assert gt.duration_dt > 0
    assert gt.scheduling_method == "asap"
