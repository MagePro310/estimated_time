"""Integration with MQT Bench (Munich Quantum Toolkit) to load benchmark circuits."""

import os
from typing import List, Tuple, Optional
from qiskit import QuantumCircuit
from qiskit.qasm2 import dump
from mqt.bench import get_benchmark, BenchmarkLevel
from mqt.bench.benchmarks import get_available_benchmark_names


def list_available_benchmarks() -> List[str]:
    """Return all available benchmark algorithm names in mqt.bench."""
    return get_available_benchmark_names()


def _resolve_level(level_str: str) -> BenchmarkLevel:
    """Map string level name to mqt.bench BenchmarkLevel enum."""
    clean = level_str.strip().lower()
    if clean == "indep":
        return BenchmarkLevel.INDEP
    elif clean == "alg":
        return BenchmarkLevel.ALG
    elif clean == "nativegates":
        return BenchmarkLevel.NATIVEGATES
    elif clean == "mapped":
        return BenchmarkLevel.MAPPED
    else:
        raise ValueError(f"Unknown benchmark level: '{level_str}'. Expected 'indep' or 'alg'.")


def load_mqt_circuit(
    benchmark_name: str,
    circuit_size: int,
    level: str = "indep",
    cache_dir: Optional[str] = None,
) -> QuantumCircuit:
    """
    Load a quantum circuit from MQT Bench.

    Args:
        benchmark_name: Name of algorithm (e.g. 'ghz', 'qft', 'dj', 'grover').
        circuit_size: Number of qubits.
        level: BenchmarkLevel ('indep' by default, or 'alg').
        cache_dir: Optional directory to save/cache exported QASM file.

    Returns:
        QuantumCircuit object.
    """
    lvl_enum = _resolve_level(level)
    qc = get_benchmark(benchmark=benchmark_name, level=lvl_enum, circuit_size=circuit_size)

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        qasm_filename = os.path.join(cache_dir, f"{benchmark_name}_{level}_n{circuit_size}.qasm")
        try:
            with open(qasm_filename, "w", encoding="utf-8") as f:
                dump(qc, f)
        except Exception:
            pass

    return qc


def get_benchmark_suite(
    benchmarks: Optional[List[str]] = None,
    qubit_range: Tuple[int, int] = (2, 5),
    level: str = "indep",
    cache_dir: Optional[str] = None,
) -> List[Tuple[str, int, QuantumCircuit]]:
    """
    Generate a suite of benchmark circuits for a specified range of qubits.

    Args:
        benchmarks: List of benchmark names. Defaults to standard 8 algorithms.
        qubit_range: (min_qubits, max_qubits) inclusive.
        level: 'indep' or 'alg'.
        cache_dir: Optional directory to cache circuits.

    Returns:
        List of tuples: (benchmark_name, num_qubits, QuantumCircuit)
    """
    if benchmarks is None:
        benchmarks = ["ghz", "qft", "dj", "bv", "grover", "graphstate", "wstate", "ae"]

    min_q, max_q = qubit_range
    suite: List[Tuple[str, int, QuantumCircuit]] = []

    for name in benchmarks:
        for size in range(min_q, max_q + 1):
            try:
                qc = load_mqt_circuit(
                    benchmark_name=name,
                    circuit_size=size,
                    level=level,
                    cache_dir=cache_dir,
                )
                if qc.num_qubits <= max_q:
                    suite.append((name, size, qc))
            except Exception:
                continue

    return suite

