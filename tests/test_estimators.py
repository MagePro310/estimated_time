"""Tests for Analytical, Routing-Aware, and ML duration estimators."""

import pytest
from qiskit import QuantumCircuit
from qcirc_time_estimator.backends.fake_belem import load_fake_belem_info
from qcirc_time_estimator.estimators.analytical import AnalyticalCriticalPathEstimator
from qcirc_time_estimator.estimators.routing_aware import RoutingAwareEstimator
from qcirc_time_estimator.estimators.ml_estimator import MLDurationEstimator


def test_analytical_critical_path():
    _, info = load_fake_belem_info()
    estimator = AnalyticalCriticalPathEstimator()

    # Parallel vs Sequential
    qc_par = QuantumCircuit(2)
    qc_par.x(0)
    qc_par.x(1)

    qc_seq = QuantumCircuit(2)
    qc_seq.x(0)
    qc_seq.x(0)

    res_par = estimator.estimate(qc_par, info)
    res_seq = estimator.estimate(qc_seq, info)

    assert res_par.estimated_duration_sec > 0
    assert res_seq.estimated_duration_sec > res_par.estimated_duration_sec
    assert res_par.estimation_time_sec < 0.1  # Fast execution (< 100ms)


def test_routing_aware_distance_penalty():
    _, info = load_fake_belem_info()
    routing_est = RoutingAwareEstimator()

    # Adjacent CX: qubits (0, 1) have distance 1 on FakeBelem
    qc_adj = QuantumCircuit(5)
    qc_adj.cx(0, 1)

    # Distant CX: qubits (0, 4) have distance 3 on FakeBelem (0-1-3-4)
    qc_dist = QuantumCircuit(5)
    qc_dist.cx(0, 4)

    res_adj = routing_est.estimate(qc_adj, info)
    res_dist = routing_est.estimate(qc_dist, info)

    # Distant interaction must incur higher estimated duration due to SWAP penalty
    assert res_dist.estimated_duration_sec > res_adj.estimated_duration_sec
    assert res_dist.metadata["estimated_swaps"] == 2  # dist - 1 = 3 - 1 = 2 swaps


def test_ml_estimator_training_and_prediction():
    _, info = load_fake_belem_info()
    ml_est = MLDurationEstimator()

    qc1 = QuantumCircuit(2)
    qc1.h(0)
    qc1.cx(0, 1)

    qc2 = QuantumCircuit(3)
    qc2.h(0)
    qc2.cx(0, 1)
    qc2.cx(1, 2)

    circuits = [qc1, qc2]
    durations = [1e-6, 2.5e-6]

    ml_est.fit(circuits, durations, info)
    assert ml_est.is_fitted

    res = ml_est.estimate(qc1, info)
    assert res.estimated_duration_sec > 0.0
    assert res.estimated_duration_dt > 0

