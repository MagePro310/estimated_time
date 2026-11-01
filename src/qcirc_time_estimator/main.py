"""Main programmatic entrypoint for executing quantum circuit time estimation experiments."""

import sys
from typing import Optional
from qiskit import QuantumCircuit

from .config import ExperimentConfig, ACTIVE_CONFIG
from .evaluation import BenchmarkRunner, MultiBackendRunner, plot_benchmark_results
from .backends import load_backend_and_info
from .circuits import load_mqt_circuit
from .ground_truth import compute_exact_duration
from .estimators import (
    AnalyticalCriticalPathEstimator,
    RoutingAwareEstimator,
    CLOPSAwareDepthEstimator,
)


def run_single_circuit(config: ExperimentConfig) -> None:
    """Execute ground truth scheduling and estimation for a single quantum circuit."""
    backend_name = getattr(config, "backend_name", "fake_belem")
    backend, backend_info = load_backend_and_info(backend_name)
    print("=" * 65)
    print(f"EVALUATING SINGLE CIRCUIT ON {backend.name.upper()} ({backend.num_qubits} QUBITS)")
    print("=" * 65)

    if config.single_circuit_name:
        print(f"Loading MQT Bench circuit '{config.single_circuit_name}' (qubits={config.single_circuit_qubits}, level={config.level})...")
        qc = load_mqt_circuit(
            config.single_circuit_name,
            circuit_size=config.single_circuit_qubits,
            level=config.level,
        )
    elif config.single_qasm_file:
        print(f"Loading circuit from QASM file: {config.single_qasm_file}...")
        qc = QuantumCircuit.from_qasm_file(config.single_qasm_file)
    else:
        print("Error: No circuit specified. Set single_circuit_name or single_qasm_file in config.")
        sys.exit(1)

    print(f"Circuit '{qc.name}': {qc.num_qubits} qubits, depth={qc.depth()}")

    # 1. Ground Truth
    print(f"\nTranspiling & scheduling with Qiskit on {backend.name} (Ground Truth)...")
    gt = compute_exact_duration(qc, backend=backend, optimization_level=config.optimization_level)
    print(f"  Ground Truth Duration:  {gt.duration_sec * 1e6:.3f} μs ({gt.duration_dt:,} dt)")
    print(f"  Qiskit Transpile Time:  {gt.transpile_time_sec * 1e3:.2f} ms")

    # 2. Estimators
    analytical = AnalyticalCriticalPathEstimator()
    routing = RoutingAwareEstimator()
    clops_est = CLOPSAwareDepthEstimator(
        clops=config.custom_clops,
        alpha=config.custom_alpha,
        beta=config.custom_beta,
    )

    res_an = analytical.estimate(qc, backend_info)
    res_rt = routing.estimate(qc, backend_info)
    res_clops = clops_est.estimate(qc, backend_info)

    err_an = abs(res_an.estimated_duration_sec - gt.duration_sec) / gt.duration_sec * 100.0 if gt.duration_sec > 0 else 0
    err_rt = abs(res_rt.estimated_duration_sec - gt.duration_sec) / gt.duration_sec * 100.0 if gt.duration_sec > 0 else 0
    err_clops = abs(res_clops.estimated_duration_sec - gt.duration_sec) / gt.duration_sec * 100.0 if gt.duration_sec > 0 else 0

    print("\n" + "-" * 65)
    print("ESTIMATION COMPARISON")
    print("-" * 65)
    print(f"1. Analytical Critical Path : {res_an.estimated_duration_sec * 1e6:8.3f} μs | Error: {err_an:5.2f}% | Latency: {res_an.estimation_time_sec * 1e3:.4f} ms")
    print(f"2. Routing-Aware Estimator  : {res_rt.estimated_duration_sec * 1e6:8.3f} μs | Error: {err_rt:5.2f}% | Latency: {res_rt.estimation_time_sec * 1e3:.4f} ms")
    print(f"3. CLOPS-Aware NewDepth     : {res_clops.estimated_duration_sec * 1e6:8.3f} μs | Error: {err_clops:5.2f}% | Latency: {res_clops.estimation_time_sec * 1e3:.4f} ms")

    comp = res_clops.metadata["depth_components"]
    print(f"   -> Breakdown: D_0={comp['origin_depth']:.0f}, Δ1Q={comp['diff_1q']:+.1f}, ΔTopo={comp['diff_topo']:.1f}, ΔMulti={comp['diff_multi']:.1f}")
    print(f"   -> Result   : NewDepth={res_clops.metadata['new_depth']:.2f}, Effective CLOPS={res_clops.metadata['effective_clops']:,.0f} layer/s")
    print("=" * 65)


def run_benchmark(config: ExperimentConfig) -> None:
    """Execute the single-backend benchmark experiment suite according to configuration."""
    print("=" * 65)
    print(f"RUNNING BENCHMARK SUITE (Backend: {config.backend_name}, Qubits {config.qubit_range})")
    print("=" * 65)

    runner = BenchmarkRunner(config)
    df, metrics = runner.run()

    print("\n" + "=" * 65)
    print("BENCHMARK EXECUTION COMPLETED")
    print("=" * 65)
    for est_name, m in metrics.items():
        print(f"\n[{est_name.upper()} ESTIMATOR]")
        print(f"  MAPE:            {m.get('mape_percent', 0.0):.2f}%")
        print(f"  MAE:             {m.get('mae_sec', 0.0) * 1e6:.2f} μs")
        print(f"  RMSE:            {m.get('rmse_sec', 0.0) * 1e6:.2f} μs")
        print(f"  Pearson r:       {m.get('pearson_r', 0.0):.4f}")
        print(f"  Avg Speedup:     {m.get('avg_speedup_factor', 1.0):.1f}x")

    plot_paths = plot_benchmark_results(df, metrics, config.output_dir)
    print("\nGenerated visualization plots:")
    for p in plot_paths:
        print(f"  -> {p}")


def run_multi_backend(config: ExperimentConfig) -> None:
    """Execute multi-backend ground truth data generation and store in CSV dataset."""
    runner = MultiBackendRunner(config)
    runner.run()


def main(config: Optional[ExperimentConfig] = None) -> None:
    """Entry point using in-code configuration."""
    cfg = config or ACTIVE_CONFIG
    if cfg.mode == "multi_backend":
        run_multi_backend(cfg)
    elif cfg.mode == "single_circuit":
        run_single_circuit(cfg)
    else:
        run_benchmark(cfg)


if __name__ == "__main__":
    main()
