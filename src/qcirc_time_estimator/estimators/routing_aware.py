"""Routing-aware duration estimator incorporating coupling graph SWAP overhead."""

import time
from typing import Dict
from qiskit import QuantumCircuit
from .base import BaseDurationEstimator, EstimationResult
from .analytical import AnalyticalCriticalPathEstimator
from ..backends import BackendInfo


class RoutingAwareEstimator(BaseDurationEstimator):
    """
    Estimates circuit execution time by combining DAG critical-path gate durations
    with topology-aware SWAP insertion penalties on the FakeBelem coupling map.
    """

    def __init__(self, name: str = "routing_aware"):
        super().__init__(name=name)
        self._analytical = AnalyticalCriticalPathEstimator()

    def estimate(self, circuit: QuantumCircuit, backend_info: BackendInfo) -> EstimationResult:
        start_time = time.perf_counter()

        qc = circuit
        has_composite = any(inst.operation.num_qubits > 2 and inst.operation.name not in ("ccx", "cswap") for inst in circuit.data)
        if has_composite:
            try:
                qc = circuit.decompose()
            except Exception:
                qc = circuit

        wire_times: Dict[int, float] = {i: 0.0 for i in range(backend_info.num_qubits)}
        total_estimated_swaps = 0
        t_cx = backend_info.avg_cx_duration_sec

        # Initial trivial virtual-to-physical mapping: virtual qubit i -> physical qubit i
        v2p = {i: i for i in range(backend_info.num_qubits)}

        for instruction in qc.data:
            op_name = instruction.operation.name
            v_qargs = tuple(qc.find_bit(q).index for q in instruction.qubits)

            if len(v_qargs) == 0:
                continue
            elif len(v_qargs) == 1:
                p_q = v2p.get(v_qargs[0], v_qargs[0])
                dur = self._analytical._decompose_gate_duration(op_name, (p_q,), backend_info)
                wire_times[p_q] = wire_times.get(p_q, 0.0) + dur
            elif len(v_qargs) == 2:
                p1 = v2p.get(v_qargs[0], v_qargs[0])
                p2 = v2p.get(v_qargs[1], v_qargs[1])

                dist = backend_info.get_distance(p1, p2)
                routing_delay = 0.0
                if dist > 1:
                    # Minimum number of SWAPs needed to bring qubits adjacent
                    num_swaps = dist - 1
                    total_estimated_swaps += num_swaps
                    # 1 SWAP = 3 CX gates
                    routing_delay = num_swaps * 3 * t_cx

                base_dur = self._analytical._decompose_gate_duration(op_name, (p1, p2), backend_info)
                total_gate_dur = base_dur + routing_delay

                start_wire_time = max(wire_times.get(p1, 0.0), wire_times.get(p2, 0.0))
                end_wire_time = start_wire_time + total_gate_dur
                wire_times[p1] = end_wire_time
                wire_times[p2] = end_wire_time
            else:
                # Multi-qubit (3+) gate
                p_qargs = [v2p.get(q, q) for q in v_qargs]
                base_dur = self._analytical._decompose_gate_duration(op_name, tuple(p_qargs), backend_info)
                start_wire_time = max(wire_times.get(q, 0.0) for q in p_qargs)
                end_wire_time = start_wire_time + base_dur
                for q in p_qargs:
                    wire_times[q] = end_wire_time

        estimated_sec = max(wire_times.values()) if wire_times else 0.0
        estimated_dt = int(round(estimated_sec / backend_info.dt)) if backend_info.dt > 0 else 0
        est_time_sec = time.perf_counter() - start_time

        return EstimationResult(
            estimator_name=self.name,
            estimated_duration_sec=estimated_sec,
            estimated_duration_dt=estimated_dt,
            estimation_time_sec=est_time_sec,
            metadata={
                "wire_times": wire_times,
                "estimated_swaps": total_estimated_swaps,
            },
        )

