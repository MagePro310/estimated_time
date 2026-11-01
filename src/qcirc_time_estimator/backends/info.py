"""Standardized representation of target backend timing and topology."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class BackendInfo:
    """Standardized representation of target backend timing and topology."""
    name: str
    num_qubits: int
    dt: float  # clock cycle time in seconds (e.g. ~2.222e-10)
    basis_gates: List[str]
    coupling_edges: List[Tuple[int, int]]
    gate_durations_sec: Dict[Tuple[str, Tuple[int, ...]], float]
    gate_durations_dt: Dict[Tuple[str, Tuple[int, ...]], int]
    shortest_path_matrix: Dict[Tuple[int, int], int] = field(default_factory=dict)
    avg_1q_duration_sec: float = 0.0
    avg_cx_duration_sec: float = 0.0
    avg_measure_duration_sec: float = 0.0

    def get_gate_duration_sec(self, gate_name: str, qubits: Tuple[int, ...]) -> float:
        """Retrieve duration in seconds for a specific gate on given qubits, or fall back to averages."""
        key = (gate_name.lower(), tuple(qubits))
        if key in self.gate_durations_sec:
            return self.gate_durations_sec[key]

        # Fallbacks based on gate type
        op = gate_name.lower()
        if op in ("cx", "ecr", "cz"):
            return self.avg_cx_duration_sec
        elif op in ("measure",):
            return self.avg_measure_duration_sec
        elif op in ("rz", "z", "s", "t", "sdg", "tdg"):
            return 0.0  # Virtual Z gates take 0 duration
        elif op in ("x", "sx", "y", "h"):
            return self.avg_1q_duration_sec
        return self.avg_1q_duration_sec

    def get_distance(self, q1: int, q2: int) -> int:
        """Return shortest path distance (number of hops) between two qubits on coupling map."""
        if q1 == q2:
            return 0
        return self.shortest_path_matrix.get((q1, q2), 1)

