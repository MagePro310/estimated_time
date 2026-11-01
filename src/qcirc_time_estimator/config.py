"""Centralized experiment configuration in code for quantum circuit duration estimation."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from .backends import SAMPLE_BACKENDS, list_available_backends


@dataclass
class ExperimentConfig:
    """
    All experiment configurations are defined directly in Python code.
    Modify this object or choose one of the predefined presets to run your experiment.
    """
    # Mode: "multi_backend", "benchmark", or "single_circuit"
    mode: str = "multi_backend"

    # --- Target Backends Settings ---
    # List of backend names (e.g. from FakeProviderForBackendV2)
    target_backends: List[str] = field(default_factory=lambda: list(SAMPLE_BACKENDS))
    backend_name: str = "fake_belem"  # Used when mode == "benchmark" or "single_circuit"

    # --- MQT Bench Benchmark Suite Settings ---
    benchmarks: List[str] = field(
        default_factory=lambda: [
            "ghz", "qft", "dj", "bv", "grover", "graphstate", "wstate", "ae"
        ]
    )
    qubit_range: Tuple[int, int] = (2, 5)
    level: str = "indep"  # "indep" (hardware-independent) or "alg" (algorithmic)

    # --- Qiskit Scheduling & Optimization Settings ---
    optimization_level: int = 1  # 0, 1, 2, 3
    scheduling_method: str = "asap"  # "asap" or "alap"

    # --- Output CSV Dataset Settings ---
    output_dir: str = "data/results"
    dataset_csv_path: str = "data/results/ground_truth_dataset.csv"
    backends_csv_path: str = "data/results/backends_metadata.csv"

    # --- Single Circuit Evaluation Settings (used when mode == "single_circuit") ---
    single_circuit_name: Optional[str] = "qft"
    single_circuit_qubits: int = 3
    single_qasm_file: Optional[str] = None

    # --- CLOPS & NewDepth Custom Hyperparameters ---
    custom_clops: Optional[float] = None  # None uses hardware layer time (1 / t_cx)
    custom_alpha: Optional[float] = None  # None uses hardware t_1q / t_cx ratio
    custom_beta: Optional[float] = None   # None uses default 0.90 SWAP penalty


# ==============================================================================
# PREDEFINED EXPERIMENT PRESETS (EDIT OR SELECT DIRECTLY IN CODE)
# ==============================================================================

# Preset 1: Representative Multi-Backend Dataset Generation (5q, 7q, 27q, 127q)
MULTI_BACKEND_SAMPLE_CONFIG = ExperimentConfig(
    mode="multi_backend",
    target_backends=["fake_belem", "fake_athens", "fake_manila", "fake_jakarta", "fake_cairo", "fake_kyoto"],
    benchmarks=["ghz", "qft", "dj", "bv", "grover", "graphstate", "wstate", "ae"],
    qubit_range=(2, 5),
    optimization_level=1,
    dataset_csv_path="data/results/ground_truth_dataset.csv",
    backends_csv_path="data/results/backends_metadata.csv",
)

# Preset 2: Fast Micro Multi-Backend Test (3 backends, 3 algorithms, 2..3 qubits)
FAST_MULTI_BACKEND_CONFIG = ExperimentConfig(
    mode="multi_backend",
    target_backends=["fake_belem", "fake_athens", "fake_jakarta"],
    benchmarks=["ghz", "qft", "grover"],
    qubit_range=(2, 3),
    optimization_level=1,
    dataset_csv_path="data/results/ground_truth_dataset.csv",
    backends_csv_path="data/results/backends_metadata.csv",
)

# Preset 3: Full 68 Backends Batch Generation (Runs across all FakeProvider backends >= 2 qubits)
FULL_68_BACKENDS_CONFIG = ExperimentConfig(
    mode="multi_backend",
    target_backends=list_available_backends(min_qubits=2),
    benchmarks=["ghz", "qft", "dj", "bv", "grover", "wstate", "ae"],
    qubit_range=(2, 5),
    optimization_level=1,
    dataset_csv_path="data/results/full_68_backends_ground_truth.csv",
    backends_csv_path="data/results/full_68_backends_metadata.csv",
)

# Preset 4: Legacy Single Backend Benchmark (FakeBelem only)
FULL_MQT_BENCHMARK_CONFIG = ExperimentConfig(
    mode="benchmark",
    backend_name="fake_belem",
    benchmarks=["ghz", "qft", "dj", "bv", "grover", "graphstate", "wstate", "ae"],
    qubit_range=(2, 5),
    optimization_level=1,
    output_dir="data/results/full_benchmark",
)

# Preset 5: Legacy Fast Single Backend Test
FAST_TEST_CONFIG = ExperimentConfig(
    mode="benchmark",
    backend_name="fake_belem",
    benchmarks=["ghz", "qft", "grover"],
    qubit_range=(2, 3),
    optimization_level=1,
    output_dir="data/results/fast_test",
)

# Preset 6: Single Circuit Inspection
SINGLE_CIRCUIT_CONFIG = ExperimentConfig(
    mode="single_circuit",
    single_circuit_name="qft",
    single_circuit_qubits=3,
    optimization_level=1,
)

# ==============================================================================
# ACTIVE CONFIGURATION (Change this variable to switch active experiment)
# ==============================================================================
ACTIVE_CONFIG = MULTI_BACKEND_SAMPLE_CONFIG
