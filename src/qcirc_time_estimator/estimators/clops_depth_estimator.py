"""CLOPS and NewDepth duration estimator based on hardware-informed depth expansion."""

import time
from typing import Dict, List, Tuple, Optional
import numpy as np
from qiskit import QuantumCircuit

from .base import BaseDurationEstimator, EstimationResult
from ..backends import BackendInfo


class CLOPSAwareDepthEstimator(BaseDurationEstimator):
    """
    Estimates quantum circuit execution time using the NewDepth * CLOPS formula:
        Estimated Time = NewDepth * tau_layer = NewDepth / CLOPS
    where:
        NewDepth = origin_depth + alpha * diff_1q + beta * diff_topo + gamma * diff_multi + mu * diff_meas - lambda * diff_opt
    """

    def __init__(
        self,
        name: str = "clops_depth",
        clops: Optional[float] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        mu: Optional[float] = None,
        lam: Optional[float] = None,
    ):
        super().__init__(name=name)
        self.clops = clops
        self.user_alpha = alpha
        self.user_beta = beta
        self.user_gamma = gamma
        self.user_mu = mu
        self.user_lam = lam

        # Fitted calibration weights (if calibrated)
        self.is_calibrated: bool = False
        self.calibrated_weights: Dict[str, float] = {}

    def extract_depth_components(
        self, circuit: QuantumCircuit, backend_info: BackendInfo
    ) -> Dict[str, float]:
        """
        Extract physical depth expansion components from a target-independent circuit:
        - origin_depth (D_0): Logic depth before physical compilation
        - diff_1q (Delta_1Q): Basis gate mismatch (virtual Z vs 1-pulse vs 2-pulse 1Q gates)
        - diff_topo (Delta_topo): Coupling map SWAP routing overhead on critical path
        - diff_multi (Delta_multi): Extra CX layers needed for multi-qubit gates (CCX, CSWAP)
        - diff_meas (Delta_meas): Delay of measurement operations relative to CX layer
        - diff_opt (Delta_opt): Redundancy reduction from gate cancellations
        """
        num_qubits = max(backend_info.num_qubits, circuit.num_qubits)
        v2p = {i: i for i in range(circuit.num_qubits)}

        wire_depth: Dict[int, float] = {q: 0.0 for q in range(num_qubits)}
        wire_1q_diff: Dict[int, float] = {q: 0.0 for q in range(num_qubits)}
        wire_topo_diff: Dict[int, float] = {q: 0.0 for q in range(num_qubits)}
        wire_multi_diff: Dict[int, float] = {q: 0.0 for q in range(num_qubits)}

        has_measure = False

        for instruction in circuit.data:
            op_name = instruction.operation.name.lower()
            qargs = tuple(circuit.find_bit(q).index for q in instruction.qubits)

            if len(qargs) == 0:
                continue

            if op_name in ("barrier", "snapshot", "delay", "id"):
                continue

            if op_name == "measure":
                has_measure = True

            if len(qargs) == 1:
                q = qargs[0]
                p_q = v2p.get(q, q)
                wire_depth[p_q] += 1.0

                # 1Q gate pulse count mismatch relative to a standard 1Q layer:
                # - Virtual Z (0 pulses): -1.0 (doesn't take a physical time layer)
                # - Standard native/1-pulse (X, SX, H, RX, RY): 0.0
                # - 2-pulse (U, U3): +1.0 (needs 2 SX pulses + virtual RZ)
                if op_name in ("rz", "z", "s", "t", "sdg", "tdg", "p"):
                    pulse_diff = -1.0
                elif op_name in ("u", "u3"):
                    pulse_diff = 1.0
                elif op_name in ("measure", "reset"):
                    pulse_diff = 0.0
                else:
                    pulse_diff = 0.0

                wire_1q_diff[p_q] += pulse_diff

            elif len(qargs) == 2:
                q1, q2 = qargs[0], qargs[1]
                p1, p2 = v2p.get(q1, q1), v2p.get(q2, q2)

                # 2Q gate critical path synchronization
                cur_depth = max(wire_depth[p1], wire_depth[p2]) + 1.0
                wire_depth[p1] = cur_depth
                wire_depth[p2] = cur_depth

                cur_1q = max(wire_1q_diff[p1], wire_1q_diff[p2])
                if op_name in ("cz", "ecr"):
                    # CZ / ECR needs 2 additional 1Q pulses
                    ratio_1q = (
                        backend_info.avg_1q_duration_sec
                        / max(backend_info.avg_cx_duration_sec, 1e-9)
                    )
                    cur_1q += 2.0 * ratio_1q
                wire_1q_diff[p1] = cur_1q
                wire_1q_diff[p2] = cur_1q

                # Topology SWAP penalty: dist - 1 SWAPs, each SWAP = 3 CX layers
                dist = backend_info.get_distance(p1, p2)
                swap_penalty = 3.0 * float(max(dist - 1, 0))
                cur_topo = max(wire_topo_diff[p1], wire_topo_diff[p2]) + swap_penalty
                wire_topo_diff[p1] = cur_topo
                wire_topo_diff[p2] = cur_topo

                cur_multi = max(wire_multi_diff[p1], wire_multi_diff[p2])
                wire_multi_diff[p1] = cur_multi
                wire_multi_diff[p2] = cur_multi

            else:
                # Multi-qubit (3+) gate
                p_qubits = [v2p.get(q, q) for q in qargs]
                cur_depth = max(wire_depth[p] for p in p_qubits) + 1.0
                for p in p_qubits:
                    wire_depth[p] = cur_depth

                cur_1q = max(wire_1q_diff[p] for p in p_qubits)
                for p in p_qubits:
                    wire_1q_diff[p] = cur_1q

                cur_topo = max(wire_topo_diff[p] for p in p_qubits)
                for p in p_qubits:
                    wire_topo_diff[p] = cur_topo

                # Extra layers for decomposition into 2Q basis:
                # Toffoli (CCX) = 6 CX layers (+5 extra); CSWAP = 8 CX layers (+7 extra)
                if op_name in ("ccx", "toffoli"):
                    extra_layers = 5.0
                elif op_name in ("cswap", "fredkin"):
                    extra_layers = 7.0
                else:
                    extra_layers = float(len(qargs) * 2)

                cur_multi = max(wire_multi_diff[p] for p in p_qubits) + extra_layers
                for p in p_qubits:
                    wire_multi_diff[p] = cur_multi

        active_wires = [q for q in range(num_qubits) if wire_depth[q] > 0]

        origin_depth = (
            float(max(wire_depth[q] for q in active_wires))
            if active_wires
            else float(circuit.depth())
        )
        diff_1q = float(max(wire_1q_diff[q] for q in active_wires)) if active_wires else 0.0
        diff_topo = float(max(wire_topo_diff[q] for q in active_wires)) if active_wires else 0.0
        diff_multi = float(max(wire_multi_diff[q] for q in active_wires)) if active_wires else 0.0

        # Measurement delay in units of 2Q layers
        ratio_meas = backend_info.avg_measure_duration_sec / max(
            backend_info.avg_cx_duration_sec, 1e-9
        )
        diff_meas = float(ratio_meas) if has_measure else 0.0

        # Optimization cancellation heuristic (cancellations at opt_level >= 1)
        diff_opt = 0.0

        return {
            "origin_depth": origin_depth,
            "diff_1q": diff_1q,
            "diff_topo": diff_topo,
            "diff_multi": diff_multi,
            "diff_meas": diff_meas,
            "diff_opt": diff_opt,
        }

    def _get_active_weights(self, backend_info: BackendInfo) -> Dict[str, float]:
        """Compute the weights: either from calibrated data or first-principles physics."""
        if self.is_calibrated and self.calibrated_weights:
            return dict(self.calibrated_weights)

        # Physics-informed default weights
        # alpha: ratio of 1Q pulse duration to 2Q pulse duration
        t_1q = backend_info.avg_1q_duration_sec
        t_cx = max(backend_info.avg_cx_duration_sec, 1e-9)
        default_alpha = t_1q / t_cx
        default_beta = 0.90  # Effective SWAP critical-path expansion factor
        default_gamma = 1.0  # Multi-qubit decomposition layer factor
        default_mu = 1.0  # Measurement layer factor
        default_lam = 0.0  # Gate cancellation reduction

        return {
            "w_origin": 1.0,
            "alpha": self.user_alpha if self.user_alpha is not None else default_alpha,
            "beta": self.user_beta if self.user_beta is not None else default_beta,
            "gamma": self.user_gamma if self.user_gamma is not None else default_gamma,
            "mu": self.user_mu if self.user_mu is not None else default_mu,
            "lambda": self.user_lam if self.user_lam is not None else default_lam,
        }

    def compute_new_depth(
        self, circuit: QuantumCircuit, backend_info: BackendInfo
    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """Calculate NewDepth and return its component breakdown and weights."""
        components = self.extract_depth_components(circuit, backend_info)
        weights = self._get_active_weights(backend_info)

        w_origin = weights.get("w_origin", 1.0)
        alpha = weights.get("alpha", 0.12)
        beta = weights.get("beta", 0.9)
        gamma = weights.get("gamma", 1.0)
        mu = weights.get("mu", 1.0)
        lam = weights.get("lambda", 0.0)

        new_depth = (
            w_origin * components["origin_depth"]
            + alpha * components["diff_1q"]
            + beta * components["diff_topo"]
            + gamma * components["diff_multi"]
            + mu * components["diff_meas"]
            - lam * components["diff_opt"]
        )

        # Depth must be non-negative
        new_depth = max(new_depth, 0.0)
        return new_depth, components, weights

    def calibrate(
        self,
        circuits: List[QuantumCircuit],
        ground_truth_durations_sec: List[float],
        backend_info: BackendInfo,
    ) -> None:
        """
        Calibrate weights using Non-Negative Least Squares (NNLS) on empirical ground truth.
        Fits: Target (y_i = Duration_i / tau_layer) ~ w_0 * D_0 + alpha * diff_1q + beta * diff_topo + ...
        """
        from scipy.optimize import nnls

        tau_layer = self._get_tau_layer(backend_info)
        X_rows = []
        y_vals = []

        for qc, dur_sec in zip(circuits, ground_truth_durations_sec):
            comp = self.extract_depth_components(qc, backend_info)
            target_depth = dur_sec / max(tau_layer, 1e-12)
            X_rows.append([
                comp["origin_depth"],
                comp["diff_1q"],
                comp["diff_topo"],
                comp["diff_multi"],
                comp["diff_meas"],
            ])
            y_vals.append(target_depth)

        X = np.array(X_rows, dtype=float)
        y = np.array(y_vals, dtype=float)

        # Fit non-negative weights
        weights, _ = nnls(X, y)

        self.calibrated_weights = {
            "w_origin": float(weights[0]),
            "alpha": float(weights[1]),
            "beta": float(weights[2]),
            "gamma": float(weights[3]),
            "mu": float(weights[4]),
            "lambda": 0.0,
        }
        self.is_calibrated = True

    def _get_tau_layer(self, backend_info: BackendInfo) -> float:
        """Return layer duration tau_layer = 1 / CLOPS in seconds."""
        if self.clops is not None and self.clops > 0:
            return 1.0 / self.clops
        return max(backend_info.avg_cx_duration_sec, 1e-9)

    def estimate(
        self, circuit: QuantumCircuit, backend_info: BackendInfo
    ) -> EstimationResult:
        """
        Estimate circuit execution duration using:
            Estimated Time = NewDepth * tau_layer
        """
        start_time = time.perf_counter()

        new_depth, components, weights = self.compute_new_depth(circuit, backend_info)
        tau_layer = self._get_tau_layer(backend_info)
        effective_clops = 1.0 / tau_layer

        estimated_sec = new_depth * tau_layer
        estimated_dt = (
            int(round(estimated_sec / backend_info.dt)) if backend_info.dt > 0 else 0
        )
        est_time_sec = time.perf_counter() - start_time

        mode = "calibrated" if self.is_calibrated else "physics_informed"

        return EstimationResult(
            estimator_name=self.name,
            estimated_duration_sec=estimated_sec,
            estimated_duration_dt=estimated_dt,
            estimation_time_sec=est_time_sec,
            metadata={
                "mode": mode,
                "new_depth": new_depth,
                "tau_layer_sec": tau_layer,
                "effective_clops": effective_clops,
                "depth_components": components,
                "weights": weights,
            },
        )
