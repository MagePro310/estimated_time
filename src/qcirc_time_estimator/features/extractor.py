"""High-level circuit features, hardware topology features, and interaction features."""

from collections import defaultdict, deque
from typing import Dict, Any, List, Tuple, Set
import numpy as np
from qiskit import QuantumCircuit
from ..backends import BackendInfo


def extract_circuit_features(circuit: QuantumCircuit) -> Dict[str, Any]:
    """
    Extract high-level, hardware-independent circuit features (phi_circuit)
    prior to any transpilation or physical mapping:
    - n_qubits, depth, total gates, 1Q, 2Q, multi-Q, measurement counts.
    - Program Communication (PC): Density of interaction graph between logic qubits.
    - Critical Depth ratio (CD): Proportion of 2Q gates along the critical path.
    - Entanglement Ratio (ER): Ratio of 2Q+ operations to total operations.
    - Parallelism (PL): Gate concurrency factor.
    - Gate density: Average operations per qubit.
    """
    n_q = circuit.num_qubits
    depth = circuit.depth()

    n_1q = 0
    n_2q = 0
    n_multi = 0
    n_meas = 0
    total_ops = 0

    interactions: Dict[int, Set[int]] = defaultdict(set)
    wire_depth = {i: 0 for i in range(n_q)}
    wire_2q_count = {i: 0 for i in range(n_q)}

    for inst in circuit.data:
        op_name = inst.operation.name.lower()
        if op_name in ("barrier", "delay", "snapshot"):
            continue

        total_ops += 1
        q_indices = [circuit.find_bit(q).index for q in inst.qubits]
        n_op_q = len(q_indices)

        if op_name == "measure":
            n_meas += 1
        elif n_op_q == 1:
            n_1q += 1
        elif n_op_q == 2:
            n_2q += 1
            q0, q1 = q_indices[0], q_indices[1]
            interactions[q0].add(q1)
            interactions[q1].add(q0)
        elif n_op_q > 2:
            n_multi += 1
            for i in range(n_op_q):
                for j in range(i + 1, n_op_q):
                    interactions[q_indices[i]].add(q_indices[j])
                    interactions[q_indices[j]].add(q_indices[i])

        if q_indices:
            current_max_d = max(wire_depth[q] for q in q_indices)
            current_max_2q = max(wire_2q_count[q] for q in q_indices)
            new_d = current_max_d + 1
            new_2q = current_max_2q + (1 if n_op_q >= 2 else 0)
            for q in q_indices:
                wire_depth[q] = new_d
                wire_2q_count[q] = new_2q

    sum_deg = sum(len(interactions[i]) for i in range(n_q))
    pc = (sum_deg / (n_q * (n_q - 1))) if n_q > 1 else 0.0

    deepest_wire = max(wire_depth, key=wire_depth.get) if wire_depth else 0
    crit_2q = wire_2q_count.get(deepest_wire, 0)
    cd = (crit_2q / n_2q) if n_2q > 0 else 0.0

    er = ((n_2q + n_multi) / total_ops) if total_ops > 0 else 0.0
    pl = (total_ops / (n_q * depth)) if (n_q * depth) > 0 else 0.0
    density = (total_ops / n_q) if n_q > 0 else 0.0

    return {
        "circ_num_qubits": int(n_q),
        "circ_depth_pre": int(depth),
        "circ_total_gates_pre": int(total_ops),
        "circ_1q_gates_pre": int(n_1q),
        "circ_2q_gates_pre": int(n_2q),
        "circ_multi_q_gates_pre": int(n_multi),
        "circ_meas_ops_pre": int(n_meas),
        "circ_program_comm": float(round(pc, 6)),
        "circ_critical_depth_ratio": float(round(cd, 6)),
        "circ_entanglement_ratio": float(round(er, 6)),
        "circ_parallelism": float(round(pl, 6)),
        "circ_gate_density": float(round(density, 6)),
    }


def extract_hardware_features(backend_info: BackendInfo) -> Dict[str, Any]:
    """
    Extract hardware graph topology, timing, and pulse characteristics (phi_machine):
    - hw_num_qubits, hw_coupling_edges, hw_avg_node_degree, hw_graph_diameter,
      hw_avg_shortest_path, hw_edge_density, dt, and average gate durations.
    """
    n_q = backend_info.num_qubits
    edges = backend_info.coupling_edges

    # Degree per qubit
    deg: Dict[int, int] = defaultdict(int)
    for u, v in edges:
        deg[u] += 1
    avg_deg = sum(deg.values()) / n_q if n_q > 0 else 0.0

    # Shortest paths analysis
    dist_matrix = backend_info.shortest_path_matrix
    valid_distances: List[int] = []
    diameter = 0

    for i in range(n_q):
        for j in range(i + 1, n_q):
            d = dist_matrix.get((i, j), dist_matrix.get((j, i), 999))
            if d < 900:  # Reachable
                valid_distances.append(d)
                if d > diameter:
                    diameter = d

    avg_sp = float(np.mean(valid_distances)) if valid_distances else 0.0
    undirected_edges = len(edges) // 2
    max_possible_edges = (n_q * (n_q - 1)) // 2
    edge_density = undirected_edges / max_possible_edges if max_possible_edges > 0 else 0.0

    return {
        "hw_num_qubits": int(n_q),
        "hw_coupling_edges": int(undirected_edges),
        "hw_avg_node_degree": float(round(avg_deg, 4)),
        "hw_graph_diameter": int(diameter),
        "hw_avg_shortest_path": float(round(avg_sp, 4)),
        "hw_edge_density": float(round(edge_density, 4)),
        "hw_dt_sec": float(backend_info.dt),
        "hw_avg_1q_duration_sec": float(backend_info.avg_1q_duration_sec),
        "hw_avg_2q_duration_sec": float(backend_info.avg_cx_duration_sec),
        "hw_avg_meas_duration_sec": float(backend_info.avg_measure_duration_sec),
    }


def extract_interaction_features(circuit: QuantumCircuit, backend_info: BackendInfo) -> Dict[str, Any]:
    """
    Extract interaction features linking circuit structure to target hardware:
    - qubit_utilization: circuit qubits / machine physical qubits.
    - estimated_routing_swaps: estimated SWAP overhead given distance between interacting qubits.
    """
    circ_q = circuit.num_qubits
    hw_q = backend_info.num_qubits

    qubit_utilization = (circ_q / hw_q) if hw_q > 0 else 1.0

    # Estimate SWAP overhead under naive linear mapping (qubit i -> physical i)
    total_routing_swaps = 0
    for inst in circuit.data:
        if inst.operation.name.lower() in ("barrier", "delay", "snapshot"):
            continue
        qargs = [circuit.find_bit(q).index for q in inst.qubits]
        if len(qargs) == 2:
            q0, q1 = qargs[0], qargs[1]
            if q0 < hw_q and q1 < hw_q:
                d = backend_info.get_distance(q0, q1)
                total_routing_swaps += max(0, d - 1)

    return {
        "qubit_utilization": float(round(qubit_utilization, 4)),
        "estimated_routing_swaps": int(total_routing_swaps),
    }
