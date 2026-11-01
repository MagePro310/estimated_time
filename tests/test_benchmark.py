"""Integration test for end-to-end benchmark execution and visualizer."""

import os
import tempfile
import pytest
from qcirc_time_estimator.evaluation import BenchmarkRunner, plot_benchmark_results
from qcirc_time_estimator.config import ExperimentConfig as BenchmarkRunConfig


def test_micro_benchmark_pipeline():
    """Verify execution of full pipeline on a micro suite (2 algorithms, qubits 2..3)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = BenchmarkRunConfig(
            mode="benchmark",
            benchmarks=["ghz", "qft"],
            qubit_range=(2, 3),
            optimization_level=1,
            output_dir=tmpdir,
        )

        runner = BenchmarkRunner(config)
        df, metrics = runner.run()

        assert not df.empty
        assert len(df) == 4
        assert "analytical_duration_sec" in df.columns
        assert "routing_aware_duration_sec" in df.columns
        assert "clops_depth_duration_sec" in df.columns

        # Verify summary metrics
        assert "analytical" in metrics
        assert "clops_depth" in metrics
        assert "mape_percent" in metrics["clops_depth"]

        # Verify visualizer
        plots = plot_benchmark_results(df, metrics, tmpdir)
        assert len(plots) >= 2
        for p in plots:
            assert os.path.exists(p)
