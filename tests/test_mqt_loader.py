"""Unit tests for MQT Bench circuit loader integration."""

import pytest
from qcirc_time_estimator.circuits import (
    list_available_benchmarks,
    load_mqt_circuit,
    get_benchmark_suite,
)


def test_list_available_benchmarks():
    """Verify that MQT bench returns standard algorithm names."""
    benchmarks = list_available_benchmarks()
    assert len(benchmarks) > 0
    assert "ghz" in benchmarks
    assert "qft" in benchmarks
    assert "grover" in benchmarks


def test_load_indep_circuit():
    """Verify loading a hardware-independent circuit with specific qubit size."""
    qc = load_mqt_circuit("ghz", circuit_size=3, level="indep")
    assert qc.num_qubits == 3
    assert qc.depth() > 0


def test_get_benchmark_suite():
    """Verify generating benchmark suite over specified algorithms and qubit sizes."""
    algs = ["ghz", "qft"]
    suite = get_benchmark_suite(benchmarks=algs, qubit_range=(2, 3), level="indep")

    assert len(suite) == 4  # 2 algorithms * 2 qubit sizes (2, 3)
    for name, n_q, qc in suite:
        assert name in algs
        assert n_q in (2, 3)
        assert qc.num_qubits == n_q
