"""Benchmark runner orchestrating dataset generation, ground truth scheduling, and estimation."""

import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from qiskit import QuantumCircuit

from ..backends.backend_info import BackendInfo
from ..backends.fake_belem import load_fake_belem_info
from ..circuits.mqt_loader import get_benchmark_suite
from ..ground_truth.qiskit_scheduler import compute_exact_duration, GroundTruthResult
from ..estimators.base import BaseDurationEstimator, EstimationResult
from ..estimators.analytical import AnalyticalCriticalPathEstimator
from ..estimators.routing_aware import RoutingAwareEstimator
from ..estimators.ml_estimator import MLDurationEstimator
from .metrics import compute_benchmark_metrics


@dataclass
class BenchmarkRunConfig:
    """Configuration for a benchmark experiment run."""
    benchmarks: List[str] = field(default_factory=lambda: ["ghz", "qft", "dj", "bv", "grover", "graphstate", "wstate", "ae"])
    qubit_range: Tuple[int, int] = (2, 5)
    level: str = "indep"
    optimization_level: int = 1
    scheduling_method: str = "asap"
    output_dir: str = "data/results/run"


class BenchmarkRunner:
    """Orchestrates the entire estimation benchmark workflow."""

    def __init__(self, config: Optional[BenchmarkRunConfig] = None):
        self.config = config or BenchmarkRunConfig()
        self.backend, self.backend_info = load_fake_belem_info()
        self.estimators: Dict[str, BaseDurationEstimator] = {
            "analytical": AnalyticalCriticalPathEstimator(),
            "routing_aware": RoutingAwareEstimator(),
            "ml_regressor": MLDurationEstimator(),
        }

    def run(self) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
        """
        Execute the benchmark suite:
        1. Load MQT Bench circuits
        2. Compute Ground Truth (Qiskit transpile + schedule)
        3. Train ML Estimator (if included)
        4. Run Estimators
        5. Aggregate and report metrics
        """
        os.makedirs(self.config.output_dir, exist_ok=True)
        print(f"Loading MQT Bench circuits ({self.config.level} level, qubits {self.config.qubit_range})...")
        suite = get_benchmark_suite(
            benchmarks=self.config.benchmarks,
            qubit_range=self.config.qubit_range,
            level=self.config.level,
            cache_dir=os.path.join(self.config.output_dir, "cached_circuits"),
        )
        print(f"Loaded {len(suite)} benchmark circuits.")

        if not suite:
            raise RuntimeError("No benchmark circuits could be loaded for the specified configuration.")

        # 1. Compute Ground Truth
        print("Computing exact ground-truth durations with Qiskit scheduler on FakeBelem...")
        gt_results: List[GroundTruthResult] = []
        circuits: List[QuantumCircuit] = []
        metadata_list: List[Dict[str, Any]] = []

        for b_name, n_q, qc in suite:
            gt = compute_exact_duration(
                circuit=qc,
                backend=self.backend,
                optimization_level=self.config.optimization_level,
                scheduling_method=self.config.scheduling_method,
            )
            gt_results.append(gt)
            circuits.append(qc)
            metadata_list.append({
                "benchmark_name": b_name,
                "circuit_name": qc.name,
                "num_qubits": n_q,
                "depth_pre_transpile": qc.depth(),
                "depth_post_transpile": gt.depth,
                "gt_duration_dt": gt.duration_dt,
                "gt_duration_sec": gt.duration_sec,
                "transpile_time_sec": gt.transpile_time_sec,
            })

        # 2. Train ML estimator if present
        ml_est = self.estimators.get("ml_regressor")
        if ml_est is not None and isinstance(ml_est, MLDurationEstimator):
            # Train on the benchmark set
            true_secs = [gt.duration_sec for gt in gt_results]
            ml_est.fit(circuits, true_secs, self.backend_info)

        # 3. Evaluate each estimator
        records = []
        for i, (qc, meta, gt) in enumerate(zip(circuits, metadata_list, gt_results)):
            row = dict(meta)
            for est_key, est in self.estimators.items():
                res = est.estimate(qc, self.backend_info)
                row[f"{est_key}_duration_sec"] = res.estimated_duration_sec
                row[f"{est_key}_duration_dt"] = res.estimated_duration_dt
                row[f"{est_key}_estimation_time_sec"] = res.estimation_time_sec
                # Relative percentage error
                if gt.duration_sec > 0:
                    err_pct = abs(res.estimated_duration_sec - gt.duration_sec) / gt.duration_sec * 100.0
                else:
                    err_pct = 0.0
                row[f"{est_key}_error_pct"] = err_pct
            records.append(row)

        df = pd.DataFrame(records)

        # 4. Calculate summary metrics per estimator
        metrics_summary: Dict[str, Dict[str, float]] = {}
        y_true = [gt.duration_sec for gt in gt_results]
        t_trans = [gt.transpile_time_sec for gt in gt_results]

        for est_key in self.estimators.keys():
            y_pred = df[f"{est_key}_duration_sec"].tolist()
            t_est = df[f"{est_key}_estimation_time_sec"].tolist()
            m = compute_benchmark_metrics(
                y_true=y_true,
                y_pred=y_pred,
                transpile_times=t_trans,
                estimate_times=t_est,
            )
            metrics_summary[est_key] = m

        # 5. Save outputs
        csv_path = os.path.join(self.config.output_dir, "benchmark_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved benchmark results to {csv_path}")

        json_path = os.path.join(self.config.output_dir, "summary_metrics.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"Saved summary metrics to {json_path}")

        return df, metrics_summary

