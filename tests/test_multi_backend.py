"""Unit and integration tests for multi-backend pipeline and registry."""

import os
import tempfile
import pytest
from qcirc_time_estimator.backends import (
    list_available_backends,
    get_backend,
)
from qcirc_time_estimator.evaluation import MultiBackendRunner
from qcirc_time_estimator.config import ExperimentConfig


def test_registry_backend_lookup():
    """Verify backend lookup by various name formats."""
    b1 = get_backend("fake_athens")
    assert b1.name == "fake_athens"
    assert b1.num_qubits == 5

    b2 = get_backend("FakeJakartaV2")
    assert b2.name == "fake_jakarta"
    assert b2.num_qubits == 7

    b3 = get_backend("cairo")
    assert b3.name == "fake_cairo"
    assert b3.num_qubits == 27


def test_registry_available_backends():
    """Verify registry lists available backends."""
    backends = list_available_backends(min_qubits=2)
    assert len(backends) > 50
    assert "fake_belem" in backends
    assert "fake_kyoto" in backends


def test_micro_multi_backend_runner_execution():
    """Execute a fast end-to-end multi-backend generation test on 2 backends."""
    with tempfile.TemporaryDirectory() as tmpdir:
        d_csv = os.path.join(tmpdir, "test_dataset.csv")
        b_csv = os.path.join(tmpdir, "test_backends.csv")

        config = ExperimentConfig(
            mode="multi_backend",
            target_backends=["fake_belem", "fake_athens"],
            benchmarks=["ghz", "qft"],
            qubit_range=(2, 3),
            optimization_level=1,
            dataset_csv_path=d_csv,
            backends_csv_path=b_csv,
        )

        runner = MultiBackendRunner(config)
        df = runner.run()

        assert os.path.exists(d_csv)
        assert os.path.exists(b_csv)
        assert not df.empty
        assert len(df) == 8

        assert (df["gt_duration_sec"] > 0).all()
        assert (df["circ_num_qubits"] >= 2).all()
        assert (df["circ_total_gates_pre"] > 0).all()
        assert (df["gt_depth_post"] > 0).all()
        assert (df["pred_clops_depth_sec"] > 0).all()
        assert (df["clops_speedup_factor"] > 0).all()
