"""Unit tests for high-level circuit and hardware feature extraction."""

import pytest
from qiskit import QuantumCircuit
from qcirc_time_estimator.features import (
    extract_circuit_features,
    extract_hardware_features,
    extract_interaction_features,
)
from qcirc_time_estimator.backends import load_backend_and_info


def test_circuit_features_bell_state():
    """Verify circuit features on a 2-qubit Bell state."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    feats = extract_circuit_features(qc)
    assert feats["circ_num_qubits"] == 2
    assert feats["circ_depth_pre"] == 2
    assert feats["circ_total_gates_pre"] == 2
    assert feats["circ_1q_gates_pre"] == 1
    assert feats["circ_2q_gates_pre"] == 1
    assert feats["circ_multi_q_gates_pre"] == 0
    assert feats["circ_program_comm"] == 1.0  # Fully connected 2-qubit graph
    assert feats["circ_critical_depth_ratio"] == 1.0
    assert feats["circ_entanglement_ratio"] == 0.5
    assert feats["circ_parallelism"] == 0.5


def test_circuit_features_multi_qubit_gate():
    """Verify circuit features with 3-qubit Toffoli (CCX) gate."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.h(1)
    qc.ccx(0, 1, 2)

    feats = extract_circuit_features(qc)
    assert feats["circ_num_qubits"] == 3
    assert feats["circ_1q_gates_pre"] == 2
    assert feats["circ_2q_gates_pre"] == 0
    assert feats["circ_multi_q_gates_pre"] == 1
    assert feats["circ_total_gates_pre"] == 3
    assert feats["circ_entanglement_ratio"] > 0.0


def test_hardware_features_extraction():
    """Verify hardware feature extraction on FakeBelem and FakeJakarta."""
    _, belem_info = load_backend_and_info("fake_belem")
    belem_hw = extract_hardware_features(belem_info)

    assert belem_hw["hw_num_qubits"] == 5
    assert belem_hw["hw_coupling_edges"] == 4  # T-shape 5-qubit has 4 undirected edges
    assert belem_hw["hw_graph_diameter"] == 3
    assert belem_hw["hw_dt_sec"] > 0
    assert belem_hw["hw_avg_1q_duration_sec"] > 0
    assert belem_hw["hw_avg_2q_duration_sec"] > 0

    _, jakarta_info = load_backend_and_info("fake_jakarta")
    jakarta_hw = extract_hardware_features(jakarta_info)
    assert jakarta_hw["hw_num_qubits"] == 7
    assert jakarta_hw["hw_coupling_edges"] == 6


def test_interaction_features():
    """Verify interaction metrics between circuit and backend topology."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 2)

    _, belem_info = load_backend_and_info("fake_belem")
    inter = extract_interaction_features(qc, belem_info)

    assert 0.0 < inter["qubit_utilization"] <= 1.0
    assert inter["estimated_routing_swaps"] >= 0
