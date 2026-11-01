"""Unit tests for analytical, routing-aware, and ML surrogate duration estimators."""

import pytest
from qiskit import QuantumCircuit
from qcirc_time_estimator.backends import load_fake_belem_info
from qcirc_time_estimator.estimators.analytical import AnalyticalCriticalPathEstimator
from qcirc_time_estimator.estimators.routing_aware import RoutingAwareEstimator
from qcirc_time_estimator.estimators.ml_estimator import MLDurationEstimator


def test_analytical_critical_path():
    """Verify analytical critical path calculation for a simple 2-qubit Bell circuit."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    _, backend_info = load_fake_belem_info()
    estimator = AnalyticalCriticalPathEstimator()
    result = estimator.estimate(qc, backend_info)

    assert result.estimated_duration_sec > 0
    assert result.estimated_duration_dt > 0
    assert result.estimation_time_sec < 0.05  # Must be fast (< 50ms)
    assert result.estimated_duration_sec > backend_info.avg_cx_duration_sec


def test_routing_aware_distance_penalty():
    """Verify that distant 2-qubit interactions incur additional SWAP overhead."""
    _, backend_info = load_fake_belem_info()
    routing_est = RoutingAwareEstimator()

    # Adjacent interaction on FakeBelem (e.g. 0 and 1, dist=1)
    qc_adj = QuantumCircuit(5)
    qc_adj.cx(0, 1)

    # Distant interaction on FakeBelem (e.g. 0 and 4, dist >= 2)
    qc_dist = QuantumCircuit(5)
    qc_dist.cx(0, 4)

    res_adj = routing_est.estimate(qc_adj, backend_info)
    res_dist = routing_est.estimate(qc_dist, backend_info)

    assert res_dist.estimated_duration_sec > res_adj.estimated_duration_sec


def test_ml_estimator_training_and_prediction():
    """Verify that MLDurationEstimator can fit on a minimal circuit set and predict."""
    _, backend_info = load_fake_belem_info()
    ml_est = MLDurationEstimator()

    circuits = []
    durations = []
    for n in range(2, 4):
        qc = QuantumCircuit(n)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        circuits.append(qc)
        durations.append(1.0e-6 * n)

    ml_est.fit(circuits, durations, backend_info)
    assert ml_est.is_fitted

    test_qc = QuantumCircuit(2)
    test_qc.cx(0, 1)
    pred = ml_est.estimate(test_qc, backend_info)

    assert pred.estimated_duration_sec > 0
    assert pred.estimation_time_sec < 0.05
