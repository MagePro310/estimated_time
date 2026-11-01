"""Unit tests for the CLOPSAwareDepthEstimator."""

import pytest
from qiskit import QuantumCircuit
from qcirc_time_estimator.backends import load_fake_belem_info
from qcirc_time_estimator.estimators.clops_depth_estimator import CLOPSAwareDepthEstimator


def test_clops_depth_components_extraction():
    _, info = load_fake_belem_info()
    estimator = CLOPSAwareDepthEstimator()

    # 1. Test topology penalty: Adjacent CX (0, 1) vs Distant CX (0, 4)
    qc_adj = QuantumCircuit(5)
    qc_adj.cx(0, 1)

    qc_dist = QuantumCircuit(5)
    qc_dist.cx(0, 4)  # Distance on FakeBelem is 3 hops (0-1-3-4)

    comp_adj = estimator.extract_depth_components(qc_adj, info)
    comp_dist = estimator.extract_depth_components(qc_dist, info)

    assert comp_adj["origin_depth"] == 1.0
    assert comp_dist["origin_depth"] == 1.0
    assert comp_adj["diff_topo"] == 0.0
    # dist = 3 -> 3 * (3 - 1) = 6.0
    assert comp_dist["diff_topo"] == 6.0


def test_clops_depth_1q_and_multi_components():
    _, info = load_fake_belem_info()
    estimator = CLOPSAwareDepthEstimator()

    # 2. Test 1Q pulse difference: Virtual Z (diff = -1) vs U gate (diff = +1)
    qc_rz = QuantumCircuit(1)
    qc_rz.rz(0.5, 0)
    comp_rz = estimator.extract_depth_components(qc_rz, info)
    assert comp_rz["diff_1q"] == -1.0

    qc_u = QuantumCircuit(1)
    qc_u.u(0.1, 0.2, 0.3, 0)
    comp_u = estimator.extract_depth_components(qc_u, info)
    assert comp_u["diff_1q"] == 1.0

    # 3. Test Multi-qubit gate: Toffoli (CCX) has extra CX layers (+5)
    qc_toff = QuantumCircuit(3)
    qc_toff.ccx(0, 1, 2)
    comp_toff = estimator.extract_depth_components(qc_toff, info)
    assert comp_toff["diff_multi"] == 5.0

    # 4. Test Measurement overhead
    qc_meas = QuantumCircuit(1, 1)
    qc_meas.measure(0, 0)
    comp_meas = estimator.extract_depth_components(qc_meas, info)
    assert comp_meas["diff_meas"] > 0.0


def test_clops_depth_estimation_physics_informed():
    _, info = load_fake_belem_info()
    estimator = CLOPSAwareDepthEstimator()

    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)

    res = estimator.estimate(qc, info)

    assert res.estimator_name == "clops_depth"
    assert res.estimated_duration_sec > 0.0
    assert res.estimated_duration_dt > 0
    assert res.estimation_time_sec < 0.1  # Ultra-fast (< 100ms)
    assert res.metadata["mode"] == "physics_informed"
    assert res.metadata["new_depth"] >= qc.depth()
    assert res.metadata["effective_clops"] > 0.0


def test_clops_depth_custom_clops_and_weights():
    _, info = load_fake_belem_info()
    custom_clops = 5000.0  # 5,000 layers per second -> tau_layer = 200 microseconds
    estimator = CLOPSAwareDepthEstimator(
        clops=custom_clops,
        alpha=0.5,
        beta=2.0,
    )

    qc = QuantumCircuit(2)
    qc.cx(0, 1)

    res = estimator.estimate(qc, info)

    expected_tau = 1.0 / custom_clops
    assert pytest.approx(res.metadata["tau_layer_sec"]) == expected_tau
    assert pytest.approx(res.metadata["effective_clops"]) == custom_clops
    assert res.metadata["weights"]["alpha"] == 0.5
    assert res.metadata["weights"]["beta"] == 2.0


def test_clops_depth_calibration():
    _, info = load_fake_belem_info()
    estimator = CLOPSAwareDepthEstimator(name="clops_calibrated")

    qc1 = QuantumCircuit(2)
    qc1.h(0)
    qc1.cx(0, 1)

    qc2 = QuantumCircuit(3)
    qc2.h(0)
    qc2.cx(0, 1)
    qc2.cx(1, 2)

    circuits = [qc1, qc2]
    durations = [1.2e-6, 2.8e-6]

    estimator.calibrate(circuits, durations, info)

    assert estimator.is_calibrated
    assert "w_origin" in estimator.calibrated_weights
    assert all(w >= 0.0 for w in estimator.calibrated_weights.values())

    res = estimator.estimate(qc1, info)
    assert res.metadata["mode"] == "calibrated"
    assert res.estimated_duration_sec > 0.0

