"""Calculates exact quantum circuit duration via Qiskit transpilation and scheduling on FakeBelem."""

import time
import warnings
from dataclasses import dataclass
from typing import Dict, Any, Optional
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeBelemV2


@dataclass
class GroundTruthResult:
    """Stores exact circuit execution timing and transpilation metrics."""
    circuit_name: str
    num_qubits: int
    optimization_level: int
    scheduling_method: str
    duration_dt: int
    duration_sec: float
    transpile_time_sec: float
    depth: int
    gate_counts: Dict[str, int]
    transpiled_circuit: Optional[QuantumCircuit] = None


def compute_exact_duration(
    circuit: QuantumCircuit,
    backend: Optional[FakeBelemV2] = None,
    optimization_level: int = 1,
    scheduling_method: str = "asap",
    keep_transpiled_circuit: bool = False,
) -> GroundTruthResult:
    """
    Transpile and schedule a circuit on FakeBelem to obtain the ground truth execution duration.

    Args:
        circuit: Input QuantumCircuit (e.g. from MQT Bench).
        backend: FakeBelemV2 backend instance. If None, instantiates a new one.
        optimization_level: Qiskit optimization level (0, 1, 2, or 3).
        scheduling_method: 'asap' (as soon as possible) or 'alap' (as late as possible).
        keep_transpiled_circuit: Whether to store the transpiled QuantumCircuit object.

    Returns:
        GroundTruthResult with exact duration_dt and duration_sec.
    """
    if backend is None:
        backend = FakeBelemV2()

    dt = backend.dt if backend.dt is not None else 2.2222222222222221e-10

    # Measure wall-clock transpilation and scheduling time
    start_time = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        transpiled_qc = transpile(
            circuit,
            backend=backend,
            optimization_level=optimization_level,
            scheduling_method=scheduling_method,
        )
    transpile_time_sec = time.perf_counter() - start_time

    # Extract duration in dt
    duration_dt = getattr(transpiled_qc, "duration", None)
    if duration_dt is None:
        duration_dt = getattr(transpiled_qc, "_duration", 0)

    duration_dt = int(duration_dt) if duration_dt is not None else 0
    duration_sec = float(duration_dt * dt)

    gate_counts = dict(transpiled_qc.count_ops())
    depth = transpiled_qc.depth()

    return GroundTruthResult(
        circuit_name=circuit.name or "circuit",
        num_qubits=circuit.num_qubits,
        optimization_level=optimization_level,
        scheduling_method=scheduling_method,
        duration_dt=duration_dt,
        duration_sec=duration_sec,
        transpile_time_sec=transpile_time_sec,
        depth=depth,
        gate_counts=gate_counts,
        transpiled_circuit=transpiled_qc if keep_transpiled_circuit else None,
    )

