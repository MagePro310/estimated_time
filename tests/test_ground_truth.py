"""Tests for Qiskit ground truth transpilation and scheduling duration."""

import pytest
from qiskit import QuantumCircuit
from qcirc_time_estimator.backends.fake_belem import load_fake_belem_info
from qcirc_time_estimator.ground_truth.qiskit_scheduler import compute_exact_duration


def test_compute_exact_duration_bell_state():
    backend, info = load_fake_belem_info()

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    gt = compute_exact_duration(qc, backend=backend, optimization_level=1)

    assert gt.num_qubits == 2
    assert gt.duration_dt > 0
    assert gt.duration_sec > 0.0
    assert gt.transpile_time_sec > 0.0
    assert gt.depth > 0
    assert "cx" in gt.gate_counts or "rz" in gt.gate_counts


def test_scheduling_method_asap():
    backend, _ = load_fake_belem_info()
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)

    gt = compute_exact_duration(qc, backend=backend, optimization_level=1, scheduling_method="asap")
    assert gt.duration_sec > 0.0

