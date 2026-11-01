"""Tests for MQT Bench circuit loader."""

import pytest
from qiskit import QuantumCircuit
from qcirc_time_estimator.circuits.mqt_loader import (
    load_mqt_circuit,
    get_benchmark_suite,
    list_available_benchmarks,
)


def test_list_available_benchmarks():
    b_names = list_available_benchmarks()
    assert len(b_names) > 0
    assert "ghz" in b_names
    assert "qft" in b_names


def test_load_indep_circuit():
    qc_ghz = load_mqt_circuit("ghz", circuit_size=3, level="indep")
    assert isinstance(qc_ghz, QuantumCircuit)
    assert qc_ghz.num_qubits == 3

    qc_qft = load_mqt_circuit("qft", circuit_size=4, level="indep")
    assert isinstance(qc_qft, QuantumCircuit)
    assert qc_qft.num_qubits == 4


def test_get_benchmark_suite():
    suite = get_benchmark_suite(benchmarks=["ghz", "dj"], qubit_range=(2, 3), level="indep")
    assert len(suite) >= 2
    for name, n_q, qc in suite:
        assert isinstance(qc, QuantumCircuit)
        assert 2 <= n_q <= 3

