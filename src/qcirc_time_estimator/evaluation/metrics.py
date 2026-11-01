"""Evaluation metrics for quantum circuit execution time estimation."""

from typing import List, Dict, Any, Optional
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_benchmark_metrics(
    y_true: List[float],
    y_pred: List[float],
    transpile_times: Optional[List[float]] = None,
    estimate_times: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Compute regression and profiling performance metrics.

    Args:
        y_true: Ground truth execution durations (seconds).
        y_pred: Estimated execution durations (seconds).
        transpile_times: Optional list of Qiskit transpilation times (seconds).
        estimate_times: Optional list of estimator execution times (seconds).

    Returns:
        Dictionary containing MAE, MAPE (%), RMSE, R2, Pearson_r, Spearman_rho, and Speedup.
    """
    y_t = np.array(y_true, dtype=np.float64)
    y_p = np.array(y_pred, dtype=np.float64)

    if len(y_t) == 0:
        return {}

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))

    # MAPE: avoid division by zero
    non_zero = y_t > 0
    if np.any(non_zero):
        mape = float(np.mean(np.abs((y_t[non_zero] - y_p[non_zero]) / y_t[non_zero])) * 100.0)
    else:
        mape = 0.0

    # R2 score (if variance exists)
    if len(y_t) > 1 and np.var(y_t) > 1e-18:
        r2 = float(r2_score(y_t, y_p))
    else:
        r2 = 1.0 if np.allclose(y_t, y_p) else 0.0

    # Pearson and Spearman correlation
    if len(y_t) > 1 and np.std(y_t) > 1e-12 and np.std(y_p) > 1e-12:
        pr, _ = pearsonr(y_t, y_p)
        sr, _ = spearmanr(y_t, y_p)
    else:
        pr = 1.0 if np.allclose(y_t, y_p) else 0.0
        sr = 1.0 if np.allclose(y_t, y_p) else 0.0

    metrics = {
        "mae_sec": mae,
        "mape_percent": mape,
        "rmse_sec": rmse,
        "r2_score": r2,
        "pearson_r": float(pr),
        "spearman_rho": float(sr),
    }

    if transpile_times and estimate_times and len(transpile_times) == len(estimate_times):
        t_trans = np.array(transpile_times, dtype=np.float64)
        t_est = np.array(estimate_times, dtype=np.float64)
        valid = t_est > 0
        if np.any(valid):
            speedup = float(np.mean(t_trans[valid] / t_est[valid]))
            metrics["avg_speedup_factor"] = speedup

    return metrics
