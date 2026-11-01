"""Command-line interface (CLI) for running quantum circuit duration benchmarks and single estimations."""

import argparse
import sys
from .backends.fake_belem import load_fake_belem_info
from .circuits.mqt_loader import load_mqt_circuit
from .ground_truth.qiskit_scheduler import compute_exact_duration
from .estimators.analytical import AnalyticalCriticalPathEstimator
from .estimators.routing_aware import RoutingAwareEstimator
from .evaluation.benchmark_runner import BenchmarkRunner, BenchmarkRunConfig
from .evaluation.visualizer import plot_benchmark_results


def run_mqt_benchmark_cli(args: argparse.Namespace) -> None:
    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    config = BenchmarkRunConfig(
        benchmarks=benchmarks,
        qubit_range=(args.min_qubits, args.max_qubits),
        level=args.level,
        optimization_level=args.opt_level,
        scheduling_method=args.scheduling_method,
        output_dir=args.output_dir,
    )

    runner = BenchmarkRunner(config)
    df, metrics = runner.run()

    print("\n" + "=" * 60)
    print("BENCHMARK EXECUTION COMPLETED")
    print("=" * 60)
    for est_name, m in metrics.items():
        print(f"\n[{est_name.upper()} ESTIMATOR]")
        print(f"  MAPE:            {m.get('mape_percent', 0.0):.2f}%")
        print(f"  MAE:             {m.get('mae_sec', 0.0) * 1e6:.2f} μs")
        print(f"  RMSE:            {m.get('rmse_sec', 0.0) * 1e6:.2f} μs")
        print(f"  Pearson r:       {m.get('pearson_r', 0.0):.4f}")
        print(f"  Avg Speedup:     {m.get('avg_speedup_factor', 1.0):.1f}x")

    plot_paths = plot_benchmark_results(df, metrics, args.output_dir)
    print("\nGenerated visualization plots:")
    for p in plot_paths:
        print(f"  -> {p}")


def estimate_single_circuit_cli(args: argparse.Namespace) -> None:
    backend, backend_info = load_fake_belem_info()
    print(f"Target backend: {backend.name} ({backend.num_qubits} qubits, dt={backend.dt:.4e}s)")

    if args.benchmark_name:
        print(f"Loading MQT Bench circuit '{args.benchmark_name}' (size={args.num_qubits}, level={args.level})...")
        qc = load_mqt_circuit(args.benchmark_name, circuit_size=args.num_qubits, level=args.level)
    elif args.qasm_file:
        from qiskit import QuantumCircuit
        print(f"Loading circuit from QASM file: {args.qasm_file}...")
        qc = QuantumCircuit.from_qasm_file(args.qasm_file)
    else:
        print("Error: Must specify either --benchmark-name or --qasm-file")
        sys.exit(1)

    print(f"Circuit '{qc.name}': {qc.num_qubits} qubits, depth={qc.depth()}")

    # 1. Ground Truth
    print("Transpiling and scheduling with Qiskit on FakeBelem (Ground Truth)...")
    gt = compute_exact_duration(qc, backend=backend, optimization_level=args.opt_level)
    print(f"Ground Truth Duration:  {gt.duration_sec * 1e6:.3f} μs ({gt.duration_dt:,} dt)")
    print(f"Transpilation Time:     {gt.transpile_time_sec * 1e3:.2f} ms")

    # 2. Estimators
    analytical = AnalyticalCriticalPathEstimator()
    routing = RoutingAwareEstimator()

    res_an = analytical.estimate(qc, backend_info)
    res_rt = routing.estimate(qc, backend_info)

    err_an = abs(res_an.estimated_duration_sec - gt.duration_sec) / gt.duration_sec * 100.0 if gt.duration_sec > 0 else 0
    err_rt = abs(res_rt.estimated_duration_sec - gt.duration_sec) / gt.duration_sec * 100.0 if gt.duration_sec > 0 else 0

    print("\n--- Estimation Results ---")
    print(f"Analytical Critical Path: {res_an.estimated_duration_sec * 1e6:.3f} μs | Error: {err_an:.2f}% | Latency: {res_an.estimation_time_sec * 1e3:.4f} ms")
    print(f"Routing-Aware Estimator:  {res_rt.estimated_duration_sec * 1e6:.3f} μs | Error: {err_rt:.2f}% | Latency: {res_rt.estimation_time_sec * 1e3:.4f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantum Circuit Duration Estimator on IBM FakeBelem")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: run-mqt-benchmark
    p_bench = subparsers.add_parser("run-mqt-benchmark", help="Run benchmark suite on MQT Bench circuits")
    p_bench.add_argument("--benchmarks", type=str, default="ghz,qft,dj,bv,grover,graphstate,wstate,ae", help="Comma-separated benchmark names")
    p_bench.add_argument("--min-qubits", type=int, default=2, help="Minimum qubit count")
    p_bench.add_argument("--max-qubits", type=int, default=5, help="Maximum qubit count (<= 5 for FakeBelem)")
    p_bench.add_argument("--level", type=str, default="indep", choices=["indep", "alg"], help="MQT Bench level: 'indep' (default) or 'alg'")
    p_bench.add_argument("--opt-level", type=int, default=1, choices=[0, 1, 2, 3], help="Qiskit transpile optimization level")
    p_bench.add_argument("--scheduling-method", type=str, default="asap", choices=["asap", "alap"], help="Scheduling pass method")
    p_bench.add_argument("--output-dir", type=str, default="data/results/run", help="Output directory for CSV and plots")
    p_bench.set_defaults(func=run_mqt_benchmark_cli)

    # Subcommand: estimate
    p_est = subparsers.add_parser("estimate", help="Estimate duration for a single circuit")
    p_est.add_argument("--benchmark-name", type=str, help="Name of MQT benchmark (e.g. 'qft', 'ghz')")
    p_est.add_argument("--num-qubits", type=int, default=3, help="Number of qubits for benchmark")
    p_est.add_argument("--level", type=str, default="indep", choices=["indep", "alg"])
    p_est.add_argument("--qasm-file", type=str, help="Path to input QASM file")
    p_est.add_argument("--opt-level", type=int, default=1, choices=[0, 1, 2, 3])
    p_est.set_defaults(func=estimate_single_circuit_cli)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

