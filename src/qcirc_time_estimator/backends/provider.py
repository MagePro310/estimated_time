"""IBM FakeProvider integration and backend property extraction."""

from typing import Dict, List, Tuple, Optional
from collections import deque
from qiskit.providers.backend import BackendV2
from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2, FakeBelemV2

from .info import BackendInfo

# Global provider instance cache
_GLOBAL_PROVIDER: Optional[FakeProviderForBackendV2] = None

# Representative sample backends spanning 5, 7, 27, and 127 qubits
SAMPLE_BACKENDS: List[str] = [
    "fake_belem",      # 5q T-shape
    "fake_athens",     # 5q Linear
    "fake_manila",     # 5q Linear
    "fake_jakarta",    # 7q Heavy-Hex
    "fake_cairo",      # 27q Falcon
    "fake_kyoto",      # 127q Eagle
]


def get_fake_provider() -> FakeProviderForBackendV2:
    """Retrieve or initialize the singleton FakeProviderForBackendV2."""
    global _GLOBAL_PROVIDER
    if _GLOBAL_PROVIDER is None:
        _GLOBAL_PROVIDER = FakeProviderForBackendV2()
    return _GLOBAL_PROVIDER


def list_available_backends(min_qubits: int = 1) -> List[str]:
    """List all available backend names in FakeProviderForBackendV2."""
    provider = get_fake_provider()
    return [b.name for b in provider.backends() if b.num_qubits >= min_qubits]


def get_backend(name: str) -> BackendV2:
    """
    Resolve and instantiate a backend by name (case-insensitive).
    Examples: 'fake_belem', 'FakeBelemV2', 'jakarta', 'fake_kyoto', 'FakeJakartaV2'.
    """
    provider = get_fake_provider()
    clean = name.strip().lower()
    if clean.endswith("v2"):
        clean = clean[:-2]
    if clean.startswith("fake_"):
        clean = clean[5:]
    elif clean.startswith("fake"):
        clean = clean[4:]

    canonical = f"fake_{clean}"

    # Try direct lookup by canonical name
    try:
        return provider.backend(canonical)
    except Exception:
        # Fall back to matching name in provider backend list
        for b in provider.backends():
            b_norm = b.name.lower().replace("_", "")
            if b.name.lower() == canonical or b_norm == clean:
                return b
        raise ValueError(f"Backend '{name}' (resolved as '{canonical}') not found in FakeProviderForBackendV2.")


def extract_backend_info(backend: BackendV2) -> BackendInfo:
    """
    Extract structured BackendInfo from any BackendV2 instance:
    - Basis gates & coupling map
    - BFS all-pairs shortest path matrix
    - Native gate durations (in seconds and dt)
    - Hardware averages for 1Q, 2Q, and measurement pulses
    """
    dt = getattr(backend, "dt", None)
    ref_dt = float(dt) if dt is not None else 2.2222222222222221e-10
    num_qubits = backend.num_qubits

    coupling_edges: List[Tuple[int, int]] = []
    if backend.coupling_map:
        try:
            coupling_edges = [tuple(edge) for edge in backend.coupling_map.get_edges()]
        except Exception:
            coupling_edges = [tuple(edge) for edge in backend.coupling_map]

    gate_durations_sec: Dict[Tuple[str, Tuple[int, ...]], float] = {}
    gate_durations_dt: Dict[Tuple[str, Tuple[int, ...]], int] = {}
    cx_durations: List[float] = []
    one_q_durations: List[float] = []
    meas_durations: List[float] = []

    target = backend.target
    if target is not None:
        for op_name in target.operation_names:
            for qargs, props in target[op_name].items():
                if props is not None and props.duration is not None:
                    dur_s = float(props.duration)
                    dur_dt = int(round(dur_s / ref_dt)) if ref_dt > 0 else 0
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

    # BFS all-pairs shortest paths on coupling graph
    adj: Dict[int, List[int]] = {i: [] for i in range(num_qubits)}
    for u, v in coupling_edges:
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

    # Fill unreachable
    for i in range(num_qubits):
        for j in range(num_qubits):
            if (i, j) not in dist:
                dist[(i, j)] = 0 if i == j else 999

    return BackendInfo(
        name=backend.name,
        num_qubits=num_qubits,
        dt=ref_dt,
        basis_gates=list(target.operation_names) if target else [],
        coupling_edges=coupling_edges,
        gate_durations_sec=gate_durations_sec,
        gate_durations_dt=gate_durations_dt,
        shortest_path_matrix=dist,
        avg_1q_duration_sec=avg_1q,
        avg_cx_duration_sec=avg_cx,
        avg_measure_duration_sec=avg_meas,
    )


def load_backend_and_info(name: str) -> Tuple[BackendV2, BackendInfo]:
    """Resolve backend by name and extract both backend object and BackendInfo."""
    backend = get_backend(name)
    info = extract_backend_info(backend)
    return backend, info


def load_fake_belem_info() -> Tuple[FakeBelemV2, BackendInfo]:
    """Instantiate FakeBelemV2 and extract structured BackendInfo (backward compatibility)."""
    backend = FakeBelemV2()
    return backend, extract_backend_info(backend)

