"""Machine Learning surrogate model for quantum circuit duration estimation."""

import time
from typing import List
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from qiskit import QuantumCircuit
from .base import BaseDurationEstimator, EstimationResult
from .analytical import AnalyticalCriticalPathEstimator
from ..backends import BackendInfo


class MLDurationEstimator(BaseDurationEstimator):
    """
    Predicts circuit execution duration using a regression model trained
    on circuit topological features and analytical baseline estimates.
    """

    def __init__(self, name: str = "ml_surrogate_regressor"):
        super().__init__(name=name)
        self.model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
        self.is_fitted = False
        self._fallback_estimator = AnalyticalCriticalPathEstimator()

    def extract_features(self, circuit: QuantumCircuit, backend_info: BackendInfo) -> np.ndarray:
        """
        Extract numerical structural and topological features from the circuit.
        """
        num_qubits = circuit.num_qubits
        depth = circuit.depth()

        n_1q = 0
        n_2q = 0
        n_meas = 0
        distances = []

        for inst in circuit.data:
            q_count = len(inst.qubits)
            op_name = inst.operation.name.lower()
            if op_name == "measure":
                n_meas += 1
            elif q_count == 1:
                if op_name not in ("barrier", "delay", "snapshot"):
                    n_1q += 1
            elif q_count == 2:
                n_2q += 1
                q0 = circuit.find_bit(inst.qubits[0]).index
                q1 = circuit.find_bit(inst.qubits[1]).index
                dist = backend_info.get_distance(q0, q1)
                distances.append(dist)

        ratio_2q = (n_2q / (n_1q + n_2q)) if (n_1q + n_2q) > 0 else 0.0
        avg_dist = float(np.mean(distances)) if distances else 1.0
        max_dist = float(np.max(distances)) if distances else 1.0

        # Also use analytical baseline estimate as a high-value anchor feature
        analytical_res = self._fallback_estimator.estimate(circuit, backend_info)
        analytical_dur = analytical_res.estimated_duration_sec

        features = [
            float(num_qubits),
            float(depth),
            float(n_1q),
            float(n_2q),
            float(n_meas),
            float(ratio_2q),
            avg_dist,
            max_dist,
            analytical_dur,
        ]
        return np.array(features, dtype=np.float32)

    def fit(
        self,
        circuits: List[QuantumCircuit],
        true_durations_sec: List[float],
        backend_info: BackendInfo,
    ) -> None:
        """Train the regressor on circuit features and ground truth durations."""
        X = [self.extract_features(qc, backend_info) for qc in circuits]
        y = np.array(true_durations_sec, dtype=np.float32)
        self.model.fit(X, y)
        self.is_fitted = True

    def estimate(self, circuit: QuantumCircuit, backend_info: BackendInfo) -> EstimationResult:
        start_time = time.perf_counter()

        if not self.is_fitted:
            # Fall back to analytical estimator if ML model has not been trained
            fallback = self._fallback_estimator.estimate(circuit, backend_info)
            fallback.estimator_name = f"{self.name} (unfitted fallback)"
            return fallback

        features = self.extract_features(circuit, backend_info).reshape(1, -1)
        pred_sec = float(self.model.predict(features)[0])
        pred_dt = int(round(pred_sec / backend_info.dt)) if backend_info.dt > 0 else 0
        est_time_sec = time.perf_counter() - start_time

        return EstimationResult(
            estimator_name=self.name,
            estimated_duration_sec=pred_sec,
            estimated_duration_dt=pred_dt,
            estimation_time_sec=est_time_sec,
            metadata={"fitted": True},
        )

