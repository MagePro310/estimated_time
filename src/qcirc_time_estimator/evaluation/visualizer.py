"""Visualization utilities for quantum circuit duration estimation benchmarks."""

import os
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_benchmark_results(df: pd.DataFrame, metrics: Dict[str, Dict[str, float]], output_dir: str) -> List[str]:
    """
    Generate evaluation plots:
    1. Predicted vs. Ground Truth Duration Scatter Plot
    2. Percentage Error by Benchmark Algorithm Bar Plot
    3. Transpiler Speedup Comparison Bar Plot

    Returns:
        List of generated figure file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_plots = []

    estimators = [k for k in metrics.keys()]
    colors = {"analytical": "#1f77b4", "routing_aware": "#2ca02c", "ml_regressor": "#d62728"}

    # --- Plot 1: Correlation (Predicted vs Ground Truth) ---
    fig, axes = plt.subplots(1, len(estimators), figsize=(5.5 * len(estimators), 5), squeeze=False)
    y_true_us = df["gt_duration_sec"] * 1e6  # convert to microseconds for human-friendly axis

    for idx, est_key in enumerate(estimators):
        ax = axes[0, idx]
        y_pred_us = df[f"{est_key}_duration_sec"] * 1e6
        color = colors.get(est_key, "blue")

        ax.scatter(y_true_us, y_pred_us, color=color, alpha=0.7, edgecolors="k", s=45, label="Circuits")

        # Identity reference line
        all_vals = np.concatenate([y_true_us.values, y_pred_us.values])
        min_v, max_v = (min(all_vals) * 0.9, max(all_vals) * 1.1) if len(all_vals) else (0, 1)
        ax.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1.5, label="Perfect Match (y=x)")

        m = metrics.get(est_key, {})
        mape_val = m.get("mape_percent", 0.0)
        r_val = m.get("pearson_r", 0.0)
        ax.set_title(f"{est_key.replace('_', ' ').title()}\nMAPE: {mape_val:.1f}%, r: {r_val:.3f}")
        ax.set_xlabel("Ground Truth Duration (μs)")
        ax.set_ylabel("Estimated Duration (μs)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left")

    plt.tight_layout()
    corr_path = os.path.join(output_dir, "correlation_plot.png")
    fig.savefig(corr_path, dpi=300)
    plt.close(fig)
    generated_plots.append(corr_path)

    # --- Plot 2: Error across Benchmark Algorithms ---
    if "benchmark_name" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        bench_groups = df.groupby("benchmark_name")[[f"{k}_error_pct" for k in estimators]].mean()
        
        x = np.arange(len(bench_groups.index))
        width = 0.8 / len(estimators)

        for i, est_key in enumerate(estimators):
            ax.bar(
                x + i * width - 0.4 + width / 2,
                bench_groups[f"{est_key}_error_pct"],
                width,
                label=est_key.replace("_", " ").title(),
                color=colors.get(est_key, None),
            )

        ax.set_xticks(x)
        ax.set_xticklabels(bench_groups.index, rotation=30, ha="right")
        ax.set_ylabel("Mean Percentage Error (MAPE %)")
        ax.set_title("Estimation Error Across MQT Benchmark Algorithms on FakeBelem")
        ax.legend()
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)

        plt.tight_layout()
        err_path = os.path.join(output_dir, "error_by_algorithm.png")
        fig.savefig(err_path, dpi=300)
        plt.close(fig)
        generated_plots.append(err_path)

    # --- Plot 3: Speedup Comparison ---
    fig, ax = plt.subplots(figsize=(6, 4.5))
    speedups = [metrics[k].get("avg_speedup_factor", 1.0) for k in estimators]
    labels = [k.replace("_", " ").title() for k in estimators]

    bars = ax.bar(labels, speedups, color=[colors.get(k, "teal") for k in estimators], width=0.5)
    ax.set_ylabel("Speedup vs Qiskit Transpiler + Scheduler (x)")
    ax.set_title("Inference Speedup Factor")
    ax.set_yscale("log")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h * 1.1, f"{h:,.0f}x", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    speedup_path = os.path.join(output_dir, "speedup_plot.png")
    fig.savefig(speedup_path, dpi=300)
    plt.close(fig)
    generated_plots.append(speedup_path)

    return generated_plots

