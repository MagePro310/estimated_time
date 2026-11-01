"""Abstract base class and data structures for duration estimators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any
from qiskit import QuantumCircuit
from ..backends import BackendInfo


@dataclass
class EstimationResult:
    """Stores the predicted execution duration and estimation profiling."""
    estimator_name: str
    estimated_duration_sec: float
    estimated_duration_dt: int
    estimation_time_sec: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseDurationEstimator(ABC):
    """Abstract interface for all duration estimation models."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def estimate(self, circuit: QuantumCircuit, backend_info: BackendInfo) -> EstimationResult:
        """
        Estimate the execution duration of a quantum circuit on the target backend.

        Args:
            circuit: Un-transpiled or target-independent quantum circuit.
            backend_info: Hardware specifications and gate timings.

        Returns:
            EstimationResult containing estimated duration in seconds and dt clock cycles.
        """
        pass

