"""Calculates exact quantum circuit duration via Qiskit transpilation and scheduling on IBM backends."""

import time
import warnings
from dataclasses import dataclass
from typing import Dict, Optional
from qiskit import QuantumCircuit, transpile
from qiskit.providers.backend import BackendV2
from qiskit_ibm_runtime.fake_provider import FakeBelemV2


@dataclass
class GroundTruthResult:
    """Stores exact circuit execution timing and physical transpilation metrics."""
    circuit_name: str
    num_qubits: int
    optimization_level: int
    scheduling_method: str
    duration_dt: int
    duration_sec: float
    transpile_time_sec: float
    depth: int  # Post-transpile depth
    gate_counts: Dict[str, int]
    depth_post: int = 0
    total_gates_post: int = 0
    gates_2q_post: int = 0
    gates_1q_post: int = 0
    meas_gates_post: int = 0
    swap_count_post: int = 0
    depth_expansion_ratio: float = 1.0
    gate_expansion_ratio: float = 1.0
    transpiled_circuit: Optional[QuantumCircuit] = None


def compute_exact_duration(
    circuit: QuantumCircuit,
    backend: Optional[BackendV2] = None,
    optimization_level: int = 1,
    scheduling_method: str = "asap",
    keep_transpiled_circuit: bool = False,
) -> GroundTruthResult:
    """
    Transpile and schedule a circuit on any IBM BackendV2 to obtain exact execution duration.

    Args:
        circuit: Input QuantumCircuit (e.g. from MQT Bench).
        backend: BackendV2 instance (defaults to FakeBelemV2).
        optimization_level: Qiskit optimization level (0, 1, 2, or 3).
        scheduling_method: 'asap' (as soon as possible) or 'alap' (as late as possible).
        keep_transpiled_circuit: Whether to store the transpiled QuantumCircuit object.

    Returns:
        GroundTruthResult containing duration_sec, duration_dt, post-transpile depth,
        gate counts, and physical compilation metrics.
    """
    if backend is None:
        backend = FakeBelemV2()

    dt = backend.dt if getattr(backend, "dt", None) is not None else 2.2222222222222221e-10

    # Measure CPU wall-clock time for transpilation and pulse scheduling
    start_time = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        transpiled_qc = transpile(
            circuit,
            backend=backend,
            optimization_level=optimization_level,
            scheduling_method=scheduling_method,
        )

        unit = getattr(transpiled_qc, "unit", "dt")
        duration_raw = getattr(transpiled_qc, "duration", None)
        if duration_raw is None:
            duration_raw = getattr(transpiled_qc, "_duration", 0)

    transpile_time_sec = time.perf_counter() - start_time

    if unit == "s":
        duration_sec = float(duration_raw) if duration_raw is not None else 0.0
        duration_dt = int(round(duration_sec / dt)) if dt > 0 else 0
    else:
        duration_dt = int(duration_raw) if duration_raw is not None else 0
        duration_sec = float(duration_dt * dt)

    gate_counts = dict(transpiled_qc.count_ops())
    depth_post = transpiled_qc.depth()

    total_gates_post = sum(v for k, v in gate_counts.items() if k not in ("barrier", "delay", "snapshot"))
    gates_2q_post = sum(gate_counts.get(op, 0) for op in ("cx", "ecr", "cz", "swap"))
    gates_1q_post = sum(gate_counts.get(op, 0) for op in ("x", "sx", "rz", "id", "h", "u", "u1", "u2", "u3"))
    meas_gates_post = gate_counts.get("measure", 0)

    initial_depth = circuit.depth()
    initial_gates = sum(1 for inst in circuit.data if inst.operation.name.lower() not in ("barrier", "delay", "snapshot"))
    initial_2q = sum(1 for inst in circuit.data if len(inst.qubits) == 2 and inst.operation.name.lower() not in ("barrier", "delay", "snapshot"))

    depth_expansion = (depth_post / initial_depth) if initial_depth > 0 else 1.0
    gate_expansion = (total_gates_post / initial_gates) if initial_gates > 0 else 1.0
    estimated_swaps_inserted = max(0, (gates_2q_post - initial_2q) // 3)

    return GroundTruthResult(
        circuit_name=circuit.name or "circuit",
        num_qubits=circuit.num_qubits,
        optimization_level=optimization_level,
        scheduling_method=scheduling_method,
        duration_dt=duration_dt,
        duration_sec=duration_sec,
        transpile_time_sec=transpile_time_sec,
        depth=depth_post,
        gate_counts=gate_counts,
        depth_post=depth_post,
        total_gates_post=total_gates_post,
        gates_2q_post=gates_2q_post,
        gates_1q_post=gates_1q_post,
        meas_gates_post=meas_gates_post,
        swap_count_post=estimated_swaps_inserted,
        depth_expansion_ratio=round(depth_expansion, 4),
        gate_expansion_ratio=round(gate_expansion, 4),
        transpiled_circuit=transpiled_qc if keep_transpiled_circuit else None,
    )

