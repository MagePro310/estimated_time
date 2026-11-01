"""Quantum circuit execution duration estimator package."""

from .config import (
    ExperimentConfig,
    ACTIVE_CONFIG,
    MULTI_BACKEND_SAMPLE_CONFIG,
    FAST_MULTI_BACKEND_CONFIG,
    FULL_68_BACKENDS_CONFIG,
    FAST_TEST_CONFIG,
    FULL_MQT_BENCHMARK_CONFIG,
    SINGLE_CIRCUIT_CONFIG,
)
from .backends import (
    BackendInfo,
    load_backend_and_info,
    load_fake_belem_info,
    get_backend,
    list_available_backends,
    SAMPLE_BACKENDS,
)
from .circuits import (
    load_mqt_circuit,
    get_benchmark_suite,
    list_available_benchmarks,
)
from .ground_truth import (
    compute_exact_duration,
    GroundTruthResult,
)
from .features import (
    extract_circuit_features,
    extract_hardware_features,
    extract_interaction_features,
)
from .dataset import (
    CSVDataManager,
    DATASET_FIELDNAMES,
    BACKENDS_FIELDNAMES,
)
from .estimators import (
    BaseDurationEstimator,
    EstimationResult,
    AnalyticalCriticalPathEstimator,
    RoutingAwareEstimator,
    CLOPSAwareDepthEstimator,
    MLDurationEstimator,
)
from .evaluation import (
    BenchmarkRunner,
    MultiBackendRunner,
    compute_benchmark_metrics,
    plot_benchmark_results,
)
from .main import main

__all__ = [
    # Configuration
    "ExperimentConfig",
    "ACTIVE_CONFIG",
    "MULTI_BACKEND_SAMPLE_CONFIG",
    "FAST_MULTI_BACKEND_CONFIG",
    "FULL_68_BACKENDS_CONFIG",
    "FAST_TEST_CONFIG",
    "FULL_MQT_BENCHMARK_CONFIG",
    "SINGLE_CIRCUIT_CONFIG",
    # Hardware Backends
    "BackendInfo",
    "load_backend_and_info",
    "load_fake_belem_info",
    "get_backend",
    "list_available_backends",
    "SAMPLE_BACKENDS",
    # Circuits
    "load_mqt_circuit",
    "get_benchmark_suite",
    "list_available_benchmarks",
    # Ground Truth
    "compute_exact_duration",
    "GroundTruthResult",
    # Features
    "extract_circuit_features",
    "extract_hardware_features",
    "extract_interaction_features",
    # Dataset
    "CSVDataManager",
    "DATASET_FIELDNAMES",
    "BACKENDS_FIELDNAMES",
    # Estimators
    "BaseDurationEstimator",
    "EstimationResult",
    "AnalyticalCriticalPathEstimator",
    "RoutingAwareEstimator",
    "CLOPSAwareDepthEstimator",
    "MLDurationEstimator",
    # Evaluation
    "BenchmarkRunner",
    "MultiBackendRunner",
    "compute_benchmark_metrics",
    "plot_benchmark_results",
    # Main Entry Point
    "main",
]
