"""Data structures representing hardware backend properties and timing specifications."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


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
        if gate_name.lower() in ("cx", "ecr", "cz"):
            return self.avg_cx_duration_sec
        elif gate_name.lower() in ("measure",):
            return self.avg_measure_duration_sec
        elif gate_name.lower() in ("rz", "z", "s", "t", "sdg", "tdg"):
            return 0.0  # Virtual Z gates take 0 duration
        elif gate_name.lower() in ("x", "sx", "y", "h"):
            return self.avg_1q_duration_sec
        return self.avg_1q_duration_sec

    def get_distance(self, q1: int, q2: int) -> int:
        """Return shortest path distance (number of hops) between two qubits on the coupling map."""
        if q1 == q2:
            return 0
        return self.shortest_path_matrix.get((q1, q2), 1)

