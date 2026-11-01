"""Loader and specification extractor for IBM FakeBelem."""

from typing import Tuple, Dict, List
from collections import deque
from qiskit_ibm_runtime.fake_provider import FakeBelemV2
from .backend_info import BackendInfo


def _compute_shortest_paths(num_qubits: int, edges: List[Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
    """Compute all-pairs shortest path distance matrix on coupling graph via BFS."""
    adj: Dict[int, List[int]] = {i: [] for i in range(num_qubits)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    dist: Dict[Tuple[int, int], int] = {}
    for src in range(num_qubits):
        q = deque([src])
        visited = {src: 0}
        while q:
            curr = q.popleft()
            d = visited[curr]
            for nxt in adj[curr]:
                if nxt not in visited:
                    visited[nxt] = d + 1
                    q.append(nxt)
        for dst, d in visited.items():
            dist[(src, dst)] = d

    # Fill unreachable if any (connected graph for Belem, but good practice)
    for i in range(num_qubits):
        for j in range(num_qubits):
            if (i, j) not in dist:
                dist[(i, j)] = 0 if i == j else 999
    return dist


def load_fake_belem_info() -> Tuple[FakeBelemV2, BackendInfo]:
    """Instantiate FakeBelemV2 and extract structured BackendInfo with durations and topology."""
    backend = FakeBelemV2()
    dt = backend.dt if backend.dt is not None else 2.2222222222222221e-10
    num_qubits = backend.num_qubits

    coupling_edges: List[Tuple[int, int]] = []
    if backend.coupling_map:
        coupling_edges = [tuple(edge) for edge in backend.coupling_map]

    gate_durations_sec: Dict[Tuple[str, Tuple[int, ...]], float] = {}
    gate_durations_dt: Dict[Tuple[str, Tuple[int, ...]], int] = {}

    cx_durations: List[float] = []
    one_q_durations: List[float] = []
    meas_durations: List[float] = []

    target = backend.target
    for op_name in target.operation_names:
        for qargs, props in target[op_name].items():
            if props is not None and props.duration is not None:
                dur_s = float(props.duration)
                dur_dt = int(round(dur_s / dt)) if dt > 0 else 0
                gate_durations_sec[(op_name, tuple(qargs))] = dur_s
                gate_durations_dt[(op_name, tuple(qargs))] = dur_dt

                if op_name in ("cx", "ecr", "cz"):
                    cx_durations.append(dur_s)
                elif op_name in ("x", "sx"):
                    one_q_durations.append(dur_s)
                elif op_name == "measure":
                    meas_durations.append(dur_s)

    avg_cx = sum(cx_durations) / len(cx_durations) if cx_durations else 8.1e-7
    avg_1q = sum(one_q_durations) / len(one_q_durations) if one_q_durations else 3.5e-8
    avg_meas = sum(meas_durations) / len(meas_durations) if meas_durations else 5.3e-6

    shortest_paths = _compute_shortest_paths(num_qubits, coupling_edges)

    info = BackendInfo(
        name=backend.name,
        num_qubits=num_qubits,
        dt=dt,
        basis_gates=list(target.operation_names),
        coupling_edges=coupling_edges,
        gate_durations_sec=gate_durations_sec,
        gate_durations_dt=gate_durations_dt,
        shortest_path_matrix=shortest_paths,
        avg_1q_duration_sec=avg_1q,
        avg_cx_duration_sec=avg_cx,
        avg_measure_duration_sec=avg_meas,
    )

    return backend, info

