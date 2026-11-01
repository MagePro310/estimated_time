"""Unit tests for CSVDataManager functionality and persistence."""

import os
import tempfile
import pytest
from qcirc_time_estimator.dataset import (
    CSVDataManager,
    DATASET_FIELDNAMES,
    BACKENDS_FIELDNAMES,
)


def test_csv_manager_init():
    """Verify CSV initialization with exact headers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        d_csv = os.path.join(tmpdir, "dataset.csv")
        b_csv = os.path.join(tmpdir, "backends.csv")
        mgr = CSVDataManager(d_csv, b_csv)

        assert os.path.exists(d_csv)
        assert os.path.exists(b_csv)

        df_d = mgr.load_dataset()
        assert list(df_d.columns) == DATASET_FIELDNAMES

        df_b = mgr.load_backends_metadata()
        assert list(df_b.columns) == BACKENDS_FIELDNAMES


def test_csv_manager_append_and_batch():
    """Verify appending single record and batch of records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        d_csv = os.path.join(tmpdir, "dataset.csv")
        b_csv = os.path.join(tmpdir, "backends.csv")
        mgr = CSVDataManager(d_csv, b_csv)

        rec1 = {
            "sample_id": "test_s1",
            "benchmark_name": "ghz",
            "backend_name": "fake_belem",
            "gt_duration_sec": 1.5e-6,
            "gt_duration_dt": 6750,
            "circ_num_qubits": 3,
        }
        mgr.append_record(rec1)

        batch = [
            {
                "sample_id": f"test_s{i}",
                "benchmark_name": "qft",
                "backend_name": "fake_athens",
                "gt_duration_sec": 2.0e-6 * i,
                "circ_num_qubits": 4,
            }
            for i in range(2, 5)
        ]
        mgr.append_records_batch(batch)

        df = mgr.load_dataset()
        assert len(df) == 4
        assert df["sample_id"].tolist() == ["test_s1", "test_s2", "test_s3", "test_s4"]

        summary = mgr.get_dataset_summary()
        assert summary["total_records"] == 4
        assert "fake_belem" in summary["backends"]
        assert "fake_athens" in summary["backends"]


def test_csv_manager_backend_metadata_deduplication():
    """Verify backend metadata saves without duplicate entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        d_csv = os.path.join(tmpdir, "dataset.csv")
        b_csv = os.path.join(tmpdir, "backends.csv")
        mgr = CSVDataManager(d_csv, b_csv)

        hw_feats = {
            "hw_num_qubits": 5,
            "hw_coupling_edges": 4,
            "hw_avg_node_degree": 1.6,
            "hw_graph_diameter": 3,
            "hw_avg_shortest_path": 1.8,
            "hw_edge_density": 0.4,
            "hw_dt_sec": 2.22e-10,
            "hw_avg_1q_duration_sec": 3.5e-8,
            "hw_avg_2q_duration_sec": 5.5e-7,
            "hw_avg_meas_duration_sec": 5.3e-6,
            "hw_basis_gates": "cx,id,rz,sx,x",
        }
        mgr.save_backend_metadata(hw_feats, "fake_belem")
        mgr.save_backend_metadata(hw_feats, "fake_belem")

        df_b = mgr.load_backends_metadata()
        assert len(df_b) == 1
        assert df_b.iloc[0]["backend_name"] == "fake_belem"
