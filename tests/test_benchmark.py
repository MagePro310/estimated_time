"""End-to-end integration test for benchmark runner and visualization."""

import os
import shutil
import pytest
from qcirc_time_estimator.evaluation.benchmark_runner import BenchmarkRunner, BenchmarkRunConfig
from qcirc_time_estimator.evaluation.visualizer import plot_benchmark_results


def test_micro_benchmark_pipeline(tmp_path):
    out_dir = str(tmp_path / "test_run")
    config = BenchmarkRunConfig(
        benchmarks=["ghz"],
        qubit_range=(2, 3),
        level="indep",
        optimization_level=1,
        output_dir=out_dir,
    )

    runner = BenchmarkRunner(config)
    df, metrics = runner.run()

    assert len(df) == 2  # ghz n=2 and ghz n=3
    assert "gt_duration_sec" in df.columns
    assert "analytical_duration_sec" in df.columns
    assert "routing_aware_duration_sec" in df.columns
    assert "ml_regressor_duration_sec" in df.columns

    for est in ["analytical", "routing_aware", "ml_regressor"]:
        assert est in metrics
        assert "mape_percent" in metrics[est]
        assert "mae_sec" in metrics[est]

    # Verify CSV file output
    csv_file = os.path.join(out_dir, "benchmark_results.csv")
    assert os.path.exists(csv_file)

    # Verify plots generation
    plot_files = plot_benchmark_results(df, metrics, out_dir)
    assert len(plot_files) >= 2
    for p in plot_files:
        assert os.path.exists(p)

