"""Execution runners for single-backend benchmark analysis and multi-backend dataset generation."""

import os
import time
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit

from ..config import ExperimentConfig
from ..backends import load_backend_and_info, SAMPLE_BACKENDS
from ..circuits import load_mqt_circuit, get_benchmark_suite
from ..features import extract_circuit_features, extract_hardware_features, extract_interaction_features
from ..ground_truth import compute_exact_duration, GroundTruthResult
from ..dataset import CSVDataManager
from ..estimators.base import BaseDurationEstimator
from ..estimators.analytical import AnalyticalCriticalPathEstimator
from ..estimators.routing_aware import RoutingAwareEstimator
from ..estimators.ml_estimator import MLDurationEstimator
from ..estimators.clops_depth_estimator import CLOPSAwareDepthEstimator
from .metrics import compute_benchmark_metrics


class BenchmarkRunner:
    """Orchestrates single-backend benchmark evaluation, model fitting, and metrics reporting."""

    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or ExperimentConfig()
        backend_name = getattr(self.config, "backend_name", "fake_belem")
        self.backend, self.backend_info = load_backend_and_info(backend_name)
        self.estimators: Dict[str, BaseDurationEstimator] = {
            "analytical": AnalyticalCriticalPathEstimator(),
            "routing_aware": RoutingAwareEstimator(),
            "clops_depth": CLOPSAwareDepthEstimator(
                name="clops_depth",
                clops=getattr(self.config, "custom_clops", None),
                alpha=getattr(self.config, "custom_alpha", None),
                beta=getattr(self.config, "custom_beta", None),
            ),
            "clops_calibrated": CLOPSAwareDepthEstimator(
                name="clops_calibrated",
                clops=getattr(self.config, "custom_clops", None),
            ),
            "ml_regressor": MLDurationEstimator(),
        }

    def run(self) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
        """
        Execute benchmark evaluation across all target circuits and estimators.

        Returns:
            Tuple of (results_dataframe, summary_metrics_dict).
        """
        cache_dir = os.path.join(self.config.output_dir, "cached_circuits")
        suite = get_benchmark_suite(
            benchmarks=self.config.benchmarks,
            qubit_range=self.config.qubit_range,
            level=self.config.level,
            cache_dir=cache_dir,
        )

        rows: List[Dict[str, Any]] = []
        gt_circuits: List[Tuple[QuantumCircuit, GroundTruthResult]] = []

        print(f"Executing Ground Truth compilation on {len(suite)} benchmark circuits...")
        for name, size, qc in suite:
            gt_res = compute_exact_duration(
                circuit=qc,
                backend=self.backend,
                optimization_level=self.config.optimization_level,
                scheduling_method=self.config.scheduling_method,
            )
            gt_circuits.append((qc, gt_res))

            row: Dict[str, Any] = {
                "benchmark_name": name,
                "qubits": size,
                "circuit_depth_pre": qc.depth(),
                "circuit_ops_pre": len(qc.data),
                "gt_duration_sec": gt_res.duration_sec,
                "gt_duration_dt": gt_res.duration_dt,
                "gt_transpile_time_sec": gt_res.transpile_time_sec,
                "gt_depth_post": gt_res.depth_post,
                "gt_total_gates_post": gt_res.total_gates_post,
                "gt_2q_gates_post": gt_res.gates_2q_post,
                "gt_swap_count_post": gt_res.swap_count_post,
                "gt_depth_expansion": gt_res.depth_expansion_ratio,
            }
            rows.append(row)

        # Optional: calibrate/train data-driven estimators (clops_calibrated, ml_regressor)
        if len(gt_circuits) >= 4:
            train_circs = [item[0] for item in gt_circuits]
            train_durs = [item[1].duration_sec for item in gt_circuits]

            if "clops_calibrated" in self.estimators:
                try:
                    self.estimators["clops_calibrated"].calibrate(train_circs, train_durs, self.backend_info)
                except Exception as e:
                    print(f"Warning: Calibration for CLOPSAwareDepthEstimator failed: {e}")

            if "ml_regressor" in self.estimators:
                try:
                    self.estimators["ml_regressor"].fit(train_circs, train_durs, self.backend_info)
                except Exception as e:
                    print(f"Warning: ML model training failed: {e}")

        # Run estimators and measure latency
        print("Running estimation models...")
        for est_name, estimator in self.estimators.items():
            for idx, (qc, gt_res) in enumerate(gt_circuits):
                res = estimator.estimate(qc, self.backend_info)
                rows[idx][f"{est_name}_duration_sec"] = res.estimated_duration_sec
                rows[idx][f"{est_name}_time_sec"] = res.estimation_time_sec
                gt_dur = gt_res.duration_sec
                err_pct = abs(res.estimated_duration_sec - gt_dur) / gt_dur * 100.0 if gt_dur > 0 else 0.0
                rows[idx][f"{est_name}_error_pct"] = err_pct

        df = pd.DataFrame(rows)

        # Compute summary metrics
        metrics: Dict[str, Dict[str, float]] = {}
        for est_name in self.estimators.keys():
            y_pred = df[f"{est_name}_duration_sec"].tolist()
            y_true = df["gt_duration_sec"].tolist()
            t_trans = df["gt_transpile_time_sec"].tolist()
            t_est = df[f"{est_name}_time_sec"].tolist()
            metrics[est_name] = compute_benchmark_metrics(y_true, y_pred, t_trans, t_est)

        # Save results to output_dir
        os.makedirs(self.config.output_dir, exist_ok=True)
        csv_path = os.path.join(self.config.output_dir, "benchmark_results.csv")
        df.to_csv(csv_path, index=False)

        metrics_path = os.path.join(self.config.output_dir, "summary_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        return df, metrics


class MultiBackendRunner:
    """
    Executes ground truth transpilation, hardware feature extraction, and estimator
    predictions across multiple target backends, appending 51-feature rows into a CSV dataset.
    """

    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or ExperimentConfig()
        self.data_manager = CSVDataManager(
            dataset_csv_path=self.config.dataset_csv_path,
            backends_csv_path=self.config.backends_csv_path,
        )
        self.analytical_est = AnalyticalCriticalPathEstimator()
        self.routing_est = RoutingAwareEstimator()
        self.clops_est = CLOPSAwareDepthEstimator()

    def run(self) -> None:
        """Run data generation loop across target backends and benchmark algorithms."""
        target_backends = self.config.target_backends or SAMPLE_BACKENDS
        benchmarks = self.config.benchmarks
        min_q, max_q = self.config.qubit_range

        print("=" * 70)
        print(f"MULTI-BACKEND DATASET GENERATION")
        print(f"Target Backends ({len(target_backends)}): {target_backends}")
        print(f"Benchmarks ({len(benchmarks)}): {benchmarks}")
        print(f"Qubit range: {min_q}..{max_q}, Optimization Level: {self.config.optimization_level}")
        print(f"Dataset path: {self.config.dataset_csv_path}")
        print("=" * 70)

        total_processed = 0

        for b_name in target_backends:
            print(f"\n---> Initializing target backend: {b_name}...")
            try:
                backend, b_info = load_backend_and_info(b_name)
            except Exception as e:
                print(f"Error loading backend '{b_name}': {e}. Skipping.")
                continue

            # Extract and record machine hardware metadata
            hw_features = extract_hardware_features(b_info)
            backend_meta_record = {
                "backend_name": b_name,
                "hw_num_qubits": b_info.num_qubits,
                "hw_coupling_edges": len(b_info.coupling_edges),
                "hw_avg_node_degree": hw_features["hw_avg_node_degree"],
                "hw_graph_diameter": hw_features["hw_graph_diameter"],
                "hw_avg_shortest_path": hw_features["hw_avg_shortest_path"],
                "hw_edge_density": hw_features["hw_edge_density"],
                "hw_dt_sec": b_info.dt,
                "hw_avg_1q_duration_sec": b_info.avg_1q_duration_sec,
                "hw_avg_2q_duration_sec": b_info.avg_cx_duration_sec,
                "hw_avg_meas_duration_sec": b_info.avg_measure_duration_sec,
                "hw_basis_gates": ";".join(b_info.basis_gates),
            }
            self.data_manager.save_backend_metadata(backend_meta_record)

            # Cap circuit size by backend qubit capacity
            effective_max_q = min(max_q, b_info.num_qubits)
            if effective_max_q < min_q:
                print(f"Backend '{b_name}' has only {b_info.num_qubits} qubits (requires >= {min_q}). Skipping.")
                continue

            suite = get_benchmark_suite(
                benchmarks=benchmarks,
                qubit_range=(min_q, effective_max_q),
                level=self.config.level,
            )

            print(f"Processing {len(suite)} circuits on {b_name}...")
            batch_records = []

            for bench_name, size, qc in suite:
                try:
                    # 1. High-level circuit features
                    circ_features = extract_circuit_features(qc)

                    # 2. Circuit x Hardware interaction features
                    inter_features = extract_interaction_features(qc, b_info)

                    # 3. Ground truth compilation
                    gt_res = compute_exact_duration(
                        circuit=qc,
                        backend=backend,
                        optimization_level=self.config.optimization_level,
                        scheduling_method=self.config.scheduling_method,
                    )

                    # 4. Baseline estimations
                    an_res = self.analytical_est.estimate(qc, b_info)
                    rt_res = self.routing_est.estimate(qc, b_info)
                    clops_res = self.clops_est.estimate(qc, b_info)

                    gt_dur = gt_res.duration_sec
                    err_an = abs(an_res.estimated_duration_sec - gt_dur) / gt_dur * 100.0 if gt_dur > 0 else 0.0
                    err_rt = abs(rt_res.estimated_duration_sec - gt_dur) / gt_dur * 100.0 if gt_dur > 0 else 0.0
                    err_clops = abs(clops_res.estimated_duration_sec - gt_dur) / gt_dur * 100.0 if gt_dur > 0 else 0.0
                    clops_speedup = (gt_res.transpile_time_sec / clops_res.estimation_time_sec) if clops_res.estimation_time_sec > 0 else 1.0

                    sample_id = f"{b_name}_{bench_name}_n{size}_{self.config.level}_opt{self.config.optimization_level}"

                    record: Dict[str, Any] = {
                        "sample_id": sample_id,
                        "benchmark_name": bench_name,
                        "circuit_name": qc.name or bench_name,
                        "circuit_level": self.config.level,
                        "backend_name": b_name,
                        "optimization_level": self.config.optimization_level,
                        "scheduling_method": self.config.scheduling_method,
                        # Circuit features
                        **circ_features,
                        # Hardware features
                        **hw_features,
                        # Interaction features
                        **inter_features,
                        # Ground Truth
                        "gt_duration_sec": gt_res.duration_sec,
                        "gt_duration_dt": gt_res.duration_dt,
                        "gt_transpile_time_sec": gt_res.transpile_time_sec,
                        "gt_depth_post": gt_res.depth_post,
                        "gt_total_gates_post": gt_res.total_gates_post,
                        "gt_2q_gates_post": gt_res.gates_2q_post,
                        "gt_swap_count_post": gt_res.swap_count_post,
                        "gt_depth_expansion_ratio": gt_res.depth_expansion_ratio,
                        "gt_gate_expansion_ratio": gt_res.gate_expansion_ratio,
                        # Baseline Estimations
                        "pred_analytical_sec": an_res.estimated_duration_sec,
                        "err_analytical_pct": round(err_an, 3),
                        "time_analytical_sec": an_res.estimation_time_sec,
                        "pred_routing_aware_sec": rt_res.estimated_duration_sec,
                        "err_routing_aware_pct": round(err_rt, 3),
                        "time_routing_aware_sec": rt_res.estimation_time_sec,
                        "pred_clops_depth_sec": clops_res.estimated_duration_sec,
                        "err_clops_depth_pct": round(err_clops, 3),
                        "time_clops_depth_sec": clops_res.estimation_time_sec,
                        "clops_speedup_factor": round(clops_speedup, 1),
                        "timestamp": datetime.now().isoformat(),
                    }

                    batch_records.append(record)
                    total_processed += 1

                except Exception as e:
                    print(f"Error processing circuit {bench_name} (n={size}) on {b_name}: {e}")
                    continue

            # Batch append to dataset CSV
            self.data_manager.append_records_batch(batch_records)
            print(f"Saved {len(batch_records)} records for backend '{b_name}'.")

        print("\n" + "=" * 70)
        print(f"DATASET GENERATION FINISHED: {total_processed} total records created/updated.")
        stats = self.data_manager.get_summary_stats()
        print(f"Summary: {stats['total_samples']} samples across {stats['distinct_backends']} backends.")
        print("=" * 70)
        return self.data_manager.load_dataset()
