"""Analytical critical path duration estimator based on circuit DAG traversal."""

import time
from typing import Dict, Tuple
from qiskit import QuantumCircuit
from .base import BaseDurationEstimator, EstimationResult
from ..backends import BackendInfo


class AnalyticalCriticalPathEstimator(BaseDurationEstimator):
    """
    Estimates circuit execution time by traversing the dependency DAG along qubit wires
    and accumulating decomposed native gate durations along the critical path.
    """

    def __init__(self, name: str = "analytical_critical_path"):
        super().__init__(name=name)

    def _decompose_gate_duration(self, op_name: str, qubits: Tuple[int, ...], backend_info: BackendInfo) -> float:
        """Estimate the physical duration of an elementary or decomposed gate."""
        op = op_name.lower()
        t_1q = backend_info.avg_1q_duration_sec
        t_cx = backend_info.avg_cx_duration_sec
        t_meas = backend_info.avg_measure_duration_sec

        # Zero-duration virtual / metadata operations
        if op in ("barrier", "snapshot", "delay", "id"):
            return 0.0
        if op in ("rz", "z", "s", "t", "sdg", "tdg", "p"):
            return 0.0

        # Native / single 1Q pulses
        if op in ("x", "sx"):
            return backend_info.get_gate_duration_sec(op, qubits)
        if op in ("h", "rx", "ry", "u1", "u2"):
            return t_1q  # 1 pulse of SX surrounded by virtual RZ
        if op in ("u", "u3"):
            return 2 * t_1q  # 2 SX pulses + RZ

        # Two-qubit gates
        if op == "cx":
            return backend_info.get_gate_duration_sec(op, qubits)
        if op in ("cz", "ecr"):
            return t_cx + 2 * t_1q
        if op in ("cp", "crz"):
            return 2 * t_cx + 2 * t_1q
        if op == "swap":
            return 3 * t_cx

        # Multi-qubit gates
        if op in ("ccx", "toffoli"):
            return 6 * t_cx + 8 * t_1q
        if op in ("cswap", "fredkin"):
            return 8 * t_cx + 10 * t_1q

        # Measurement & Reset
        if op == "measure":
            return backend_info.get_gate_duration_sec(op, qubits)
        if op == "reset":
            return t_meas

        # Fallback default based on qubit count
        return t_cx if len(qubits) >= 2 else t_1q

    def estimate(self, circuit: QuantumCircuit, backend_info: BackendInfo) -> EstimationResult:
        start_time = time.perf_counter()

        # If circuit contains complex nested composite gates, decompose once
        qc = circuit
        has_composite = any(inst.operation.num_qubits > 2 and inst.operation.name not in ("ccx", "cswap") for inst in circuit.data)
        if has_composite:
            try:
                qc = circuit.decompose()
            except Exception:
                qc = circuit

        wire_times: Dict[int, float] = {i: 0.0 for i in range(backend_info.num_qubits)}

        for instruction in qc.data:
            op_name = instruction.operation.name
            qargs = tuple(qc.find_bit(q).index for q in instruction.qubits)

            gate_dur = self._decompose_gate_duration(op_name, qargs, backend_info)

            if len(qargs) == 0:
                continue
            elif len(qargs) == 1:
                q = qargs[0]
                wire_times[q] = wire_times.get(q, 0.0) + gate_dur
            else:
                # Multi-qubit gate: earliest start time is max of all operand wires
                start_wire_time = max(wire_times.get(q, 0.0) for q in qargs)
                end_wire_time = start_wire_time + gate_dur
                for q in qargs:
                    wire_times[q] = end_wire_time

        estimated_sec = max(wire_times.values()) if wire_times else 0.0
        estimated_dt = int(round(estimated_sec / backend_info.dt)) if backend_info.dt > 0 else 0
        est_time_sec = time.perf_counter() - start_time

        return EstimationResult(
            estimator_name=self.name,
            estimated_duration_sec=estimated_sec,
            estimated_duration_dt=estimated_dt,
            estimation_time_sec=est_time_sec,
            metadata={"wire_times": wire_times, "decomposed": has_composite},
        )

