"""Statistical and benchmark metric calculations."""

from typing import List, Dict, Optional
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_benchmark_metrics(
    y_true: List[float],
    y_pred: List[float],
    transpile_times: Optional[List[float]] = None,
    estimate_times: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Compute standard statistical and speedup metrics comparing predictions against ground truth."""
    y_t = np.array(y_true, dtype=np.float64)
    y_p = np.array(y_pred, dtype=np.float64)

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))

    # Relative percentage error
    valid_mask = y_t > 0
    if np.any(valid_mask):
        mape = float(np.mean(np.abs((y_t[valid_mask] - y_p[valid_mask]) / y_t[valid_mask])) * 100.0)
    else:
        mape = 0.0

    r2 = float(r2_score(y_t, y_p)) if len(y_t) > 1 and np.var(y_t) > 0 else 0.0

    # Pearson & Spearman correlations
    if len(y_t) > 1 and np.std(y_t) > 0 and np.std(y_p) > 0:
        p_r, _ = pearsonr(y_t, y_p)
        s_r, _ = spearmanr(y_t, y_p)
        pearson_val = float(p_r)
        spearman_val = float(s_r)
    else:
        pearson_val = 0.0
        spearman_val = 0.0

    metrics: Dict[str, float] = {
        "mae_sec": mae,
        "rmse_sec": rmse,
        "mape_percent": mape,
        "r2_score": r2,
        "pearson_r": pearson_val,
        "spearman_rho": spearman_val,
    }

    if transpile_times and estimate_times and len(transpile_times) == len(estimate_times):
        t_trans = np.array(transpile_times, dtype=np.float64)
        t_est = np.array(estimate_times, dtype=np.float64)
        valid_est = t_est > 0
        if np.any(valid_est):
            speedups = t_trans[valid_est] / t_est[valid_est]
            metrics["avg_speedup_factor"] = float(np.mean(speedups))
            metrics["median_speedup_factor"] = float(np.median(speedups))

    return metrics

