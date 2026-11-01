"""CSV dataset manager for ground truth records and hardware backend metadata."""

import os
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd

DATASET_FIELDNAMES: List[str] = [
    # Identifiers & Config
    "sample_id",
    "benchmark_name",
    "circuit_name",
    "circuit_level",
    "backend_name",
    "optimization_level",
    "scheduling_method",
    # Circuit Features (phi_circuit)
    "circ_num_qubits",
    "circ_depth_pre",
    "circ_total_gates_pre",
    "circ_1q_gates_pre",
    "circ_2q_gates_pre",
    "circ_multi_q_gates_pre",
    "circ_meas_ops_pre",
    "circ_program_comm",
    "circ_critical_depth_ratio",
    "circ_entanglement_ratio",
    "circ_parallelism",
    "circ_gate_density",
    # Hardware Topology & Timing (phi_machine)
    "hw_num_qubits",
    "hw_coupling_edges",
    "hw_avg_node_degree",
    "hw_graph_diameter",
    "hw_avg_shortest_path",
    "hw_edge_density",
    "hw_dt_sec",
    "hw_avg_1q_duration_sec",
    "hw_avg_2q_duration_sec",
    "hw_avg_meas_duration_sec",
    # Circuit x Hardware Interaction
    "qubit_utilization",
    "estimated_routing_swaps",
    # Exact Ground Truth (Target Y)
    "gt_duration_sec",
    "gt_duration_dt",
    "gt_transpile_time_sec",
    "gt_depth_post",
    "gt_total_gates_post",
    "gt_2q_gates_post",
    "gt_swap_count_post",
    "gt_depth_expansion_ratio",
    "gt_gate_expansion_ratio",
    # Baseline Estimator Predictions
    "pred_analytical_sec",
    "err_analytical_pct",
    "time_analytical_sec",
    "pred_routing_aware_sec",
    "err_routing_aware_pct",
    "time_routing_aware_sec",
    "pred_clops_depth_sec",
    "err_clops_depth_pct",
    "time_clops_depth_sec",
    "clops_speedup_factor",
    # Metadata
    "timestamp",
]

BACKENDS_FIELDNAMES: List[str] = [
    "backend_name",
    "hw_num_qubits",
    "hw_coupling_edges",
    "hw_avg_node_degree",
    "hw_graph_diameter",
    "hw_avg_shortest_path",
    "hw_edge_density",
    "hw_dt_sec",
    "hw_avg_1q_duration_sec",
    "hw_avg_2q_duration_sec",
    "hw_avg_meas_duration_sec",
    "hw_basis_gates",
    "timestamp",
]


class CSVDataManager:
    """Manages appending, batching, and querying ground truth records in CSV format."""

    def __init__(
        self,
        dataset_csv_path: str = "data/results/ground_truth_dataset.csv",
        backends_csv_path: str = "data/results/backends_metadata.csv",
    ):
        self.dataset_csv_path = dataset_csv_path
        self.backends_csv_path = backends_csv_path

        os.makedirs(os.path.dirname(os.path.abspath(self.dataset_csv_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(self.backends_csv_path)), exist_ok=True)

        self._init_csv_headers()

    def _init_csv_headers(self) -> None:
        """Initialize CSV files with header rows if they do not yet exist."""
        if not os.path.exists(self.dataset_csv_path) or os.path.getsize(self.dataset_csv_path) == 0:
            with open(self.dataset_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=DATASET_FIELDNAMES)
                writer.writeheader()

        if not os.path.exists(self.backends_csv_path) or os.path.getsize(self.backends_csv_path) == 0:
            with open(self.backends_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=BACKENDS_FIELDNAMES)
                writer.writeheader()

    def append_record(self, record: Dict[str, Any]) -> None:
        """Append a single record into the ground truth dataset CSV."""
        if "timestamp" not in record:
            record["timestamp"] = datetime.now().isoformat()

        row = {k: record.get(k, "") for k in DATASET_FIELDNAMES}
        with open(self.dataset_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=DATASET_FIELDNAMES)
            writer.writerow(row)

    def append_records_batch(self, records: List[Dict[str, Any]]) -> None:
        """Append a batch of records efficiently into the dataset CSV."""
        if not records:
            return
        now_str = datetime.now().isoformat()
        rows = []
        for r in records:
            if "timestamp" not in r:
                r["timestamp"] = now_str
            rows.append({k: r.get(k, "") for k in DATASET_FIELDNAMES})

        with open(self.dataset_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=DATASET_FIELDNAMES)
            writer.writerows(rows)

    def save_backend_metadata(self, metadata: Dict[str, Any], backend_name: Optional[str] = None) -> None:
        """Record or update backend specifications in backends_metadata.csv."""
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now().isoformat()

        b_name = backend_name or metadata.get("backend_name", "")
        metadata["backend_name"] = b_name

        # Avoid duplicate backend rows if already present
        if os.path.exists(self.backends_csv_path) and os.path.getsize(self.backends_csv_path) > 0:
            try:
                df = pd.read_csv(self.backends_csv_path)
                if not df.empty and b_name in df["backend_name"].values:
                    return
            except Exception:
                pass

        row = {k: metadata.get(k, "") for k in BACKENDS_FIELDNAMES}
        with open(self.backends_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=BACKENDS_FIELDNAMES)
            writer.writerow(row)

    def load_dataset(self) -> pd.DataFrame:
        """Load the ground truth dataset into a pandas DataFrame."""
        if not os.path.exists(self.dataset_csv_path) or os.path.getsize(self.dataset_csv_path) == 0:
            return pd.DataFrame(columns=DATASET_FIELDNAMES)
        return pd.read_csv(self.dataset_csv_path)

    def load_backends_metadata(self) -> pd.DataFrame:
        """Load backend specifications into a pandas DataFrame."""
        if not os.path.exists(self.backends_csv_path) or os.path.getsize(self.backends_csv_path) == 0:
            return pd.DataFrame(columns=BACKENDS_FIELDNAMES)
        return pd.read_csv(self.backends_csv_path)

    def get_dataset_summary(self) -> Dict[str, Any]:
        """Return dataset statistics: total records, distinct backends, distinct benchmarks."""
        df = self.load_dataset()
        if df.empty:
            return {
                "total_records": 0,
                "total_samples": 0,
                "backends": [],
                "distinct_backends": 0,
                "distinct_benchmarks": 0,
                "qubit_range": (0, 0),
                "avg_ground_truth_us": 0.0,
            }

        return {
            "total_records": len(df),
            "total_samples": len(df),
            "backends": df["backend_name"].unique().tolist(),
            "distinct_backends": df["backend_name"].nunique(),
            "distinct_benchmarks": df["benchmark_name"].nunique(),
            "qubit_range": (int(df["circ_num_qubits"].min()), int(df["circ_num_qubits"].max())),
            "avg_ground_truth_us": float(df["gt_duration_sec"].mean() * 1e6),
        }

    get_summary_stats = get_dataset_summary
