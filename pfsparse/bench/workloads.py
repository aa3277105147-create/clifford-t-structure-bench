"""Deterministic, fixed-amplitude workloads for the structure benchmarks."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

NATIVE_GATES = ("H", "X", "Z", "S", "SDG", "T", "TDG", "CNOT", "CZ")
_QISKIT_BASIS = ["h", "x", "z", "s", "sdg", "t", "tdg", "cx", "cz"]
_QISKIT_TO_NATIVE = {
    "h": "H", "x": "X", "z": "Z", "s": "S", "sdg": "SDG",
    "t": "T", "tdg": "TDG", "cx": "CNOT", "cz": "CZ",
}


def _append(circuit: list, gate: str, *qubits: int) -> None:
    circuit.append([gate, list(qubits)])


def _lower_qiskit(circuit, *, seed: int) -> tuple[int, list, float]:
    """Lower a Qiskit circuit exactly to the supported discrete gate basis."""
    from qiskit import transpile

    lowered = transpile(
        circuit,
        basis_gates=_QISKIT_BASIS,
        optimization_level=0,
        seed_transpiler=seed,
    )
    gates = []
    for item in lowered.data:
        name = item.operation.name
        if name in ("barrier", "id"):
            continue
        if name not in _QISKIT_TO_NATIVE:
            raise ValueError(f"Qiskit left unsupported gate {name!r} after exact lowering")
        qubits = [lowered.find_bit(q).index for q in item.qubits]
        gates.append([_QISKIT_TO_NATIVE[name], qubits])
    return lowered.num_qubits, gates, float(lowered.global_phase)


def _grover(scale: int, seed: int) -> tuple[int, list, int, int, dict]:
    from qiskit import QuantumCircuit

    n = scale
    # Consecutive seeds enumerate distinct marked states at a fixed scale.
    marked = seed % (1 << n)

    # Build each high-control X from clean helper qubits and exact Toffolis.
    # This stays in Clifford+T; no generic arbitrary-angle synthesis is used.
    helpers = list(range(n, n + max(0, n - 3)))
    qc = QuantumCircuit(n + len(helpers))

    def controlled_x(controls: list[int], target: int) -> None:
        if len(controls) == 1:
            qc.cx(controls[0], target)
            return
        if len(controls) == 2:
            qc.ccx(controls[0], controls[1], target)
            return
        carries = []
        qc.ccx(controls[0], controls[1], helpers[0])
        carries.append(helpers[0])
        for index in range(2, len(controls) - 1):
            carry = helpers[index - 1]
            qc.ccx(carries[-1], controls[index], carry)
            carries.append(carry)
        qc.ccx(carries[-1], controls[-1], target)
        for index in range(len(carries) - 1, 0, -1):
            qc.ccx(carries[index - 1], controls[index + 1], carries[index])
        qc.ccx(controls[0], controls[1], carries[0])

    qc.h(range(n))
    controls, target = list(range(n - 1)), n - 1
    for q in range(n):
        if not ((marked >> q) & 1):
            qc.x(q)
    qc.h(target)
    controlled_x(controls, target)
    qc.h(target)
    for q in range(n):
        if not ((marked >> q) & 1):
            qc.x(q)
    for q in range(n):
        qc.h(q)
        qc.x(q)
    qc.h(target)
    controlled_x(controls, target)
    qc.h(target)
    for q in range(n):
        qc.x(q)
        qc.h(q)

    physical_n, gates, phase = _lower_qiskit(qc, seed=seed)
    return physical_n, gates, 0, marked, {
        "source": "standard marked-state Grover circuit with seeded iteration count",
        "marked_state": marked,
        "iterations": 1,
        "workspace_qubits": len(helpers),
        "qiskit_lowering": "transpile to exact Clifford+T basis, optimization_level=0",
    }


def _adder(scale: int, seed: int) -> tuple[int, list, int, int, dict]:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import VBERippleCarryAdder

    width = scale // 2 + 1
    adder = VBERippleCarryAdder(width, kind="fixed")
    qc = QuantumCircuit(adder.num_qubits)
    seed_rng = random.Random(seed)
    chosen_a = seed_rng.randrange(1 << width)
    phase_pattern = seed % (1 << (2 * width))
    qc.h(range(width))
    for q in range(width):
        code = (phase_pattern >> (2 * q)) & 3
        if code == 1:
            qc.s(q)
        elif code == 2:
            qc.z(q)
        elif code == 3:
            qc.sdg(q)
    qc.append(adder, range(adder.num_qubits))

    # The b register starts at zero, so its sum is the selected value of a.
    y = chosen_a | (chosen_a << width)
    physical_n, gates, phase = _lower_qiskit(qc, seed=seed)
    return physical_n, gates, 0, y, {
        "source": "Qiskit VBE ripple-carry adder (fixed carry-in), exact lowering",
        "qiskit_component": "VBERippleCarryAdder",
        "input_phase_pattern": phase_pattern,
        "adder_width": width,
        "selected_superposition_branch": chosen_a,
        "qiskit_lowering": "transpile to exact Clifford+T basis, optimization_level=0",
    }


def _iqp(scale: int, seed: int) -> tuple[int, list, int, int, dict]:
    rng = random.Random(seed)
    n = scale
    circuit: list = []
    for q in range(n):
        _append(circuit, "H", q)

    for q in range(n):
        _append(circuit, "T" if rng.randrange(2) else "TDG", q)
    edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    rng.shuffle(edges)
    for u, v in edges[:min(len(edges), 2 * n)]:
        _append(circuit, "CNOT", u, v)
        _append(circuit, "T" if rng.randrange(2) else "TDG", v)
        _append(circuit, "CNOT", u, v)

    for q in range(n):
        _append(circuit, "H", q)
    x, y = rng.randrange(1 << n), rng.randrange(1 << n)
    return n, circuit, x, y, {
        "source": "seeded sparse IQP phase circuit; parity phases use CNOT-T-CNOT",
        "diagonal_edges": edges[:min(len(edges), 2 * n)],
    }


def _graphstate(scale: int, seed: int) -> tuple[int, list, int, int, dict]:
    rng = random.Random(seed)
    roots = list(range(4))
    rng.shuffle(roots)
    root_edges = [(roots[i], roots[(i + 1) % len(roots)]) for i in range(len(roots))]
    internal_per_edge = scale - 4
    n = len(roots) + len(root_edges) * internal_per_edge
    circuit: list = []
    for q in range(n):
        _append(circuit, "H", q)

    signs = []
    for q in roots:
        gate = "T" if rng.randrange(2) else "TDG"
        _append(circuit, gate, q)
        signs.append(gate)

    edges = []
    next_internal = len(roots)
    for u, v in root_edges:
        path = [u]
        path.extend(range(next_internal, next_internal + internal_per_edge))
        next_internal += internal_per_edge
        path.append(v)
        edges.extend((path[i], path[i + 1]) for i in range(len(path) - 1))
    for u, v in edges:
        _append(circuit, "CZ", u, v)
    for q in range(n):
        _append(circuit, "H", q)
    x, y = rng.randrange(1 << n), rng.randrange(1 << n)
    return n, circuit, x, y, {
        "source": "H^N D_G T_roots H^N graph-state circuit with subdivided root edges",
        "root_count": len(roots),
        "root_vertices": roots,
        "root_edges": root_edges,
        "root_t_gates": signs,
        "internal_path_vertices_per_edge": internal_per_edge,
        "clifford_graph_edges": edges,
    }


def _brickwork_1d(scale: int, seed: int) -> tuple[int, list, int, int, dict]:
    rng = random.Random(seed)
    n = scale
    circuit: list = []
    for q in range(n):
        _append(circuit, "H", q)
    depth = 4
    for layer in range(depth):
        start = layer % 2
        for u in range(start, n - 1, 2):
            v = u + 1
            if rng.randrange(2):
                _append(circuit, "CNOT", u, v)
            else:
                _append(circuit, "CNOT", v, u)
            if rng.randrange(2):
                _append(circuit, "CZ", u, v)
        for q in range(n):
            if rng.random() < 0.4:
                _append(circuit, "T" if rng.randrange(2) else "TDG", q)
            elif rng.random() < 0.2:
                _append(circuit, "S", q)
    for q in range(n):
        _append(circuit, "H", q)
    x, y = rng.randrange(1 << n), rng.randrange(1 << n)
    return n, circuit, x, y, {
        "source": "seeded four-layer 1D nearest-neighbor Clifford+T brickwork",
        "layers": depth,
    }


def _brickwork_2d(scale: int, seed: int) -> tuple[int, list, int, int, dict]:
    rng = random.Random(seed)
    rows = max(2, int(math.sqrt(scale)))
    cols = max(2, math.ceil(scale / rows))
    n = rows * cols
    circuit: list = []
    for q in range(n):
        _append(circuit, "H", q)

    layers = (
        [(r * cols + c, r * cols + c + 1) for r in range(rows) for c in range(cols - 1) if c % 2 == 0],
        [(r * cols + c, r * cols + c + 1) for r in range(rows) for c in range(cols - 1) if c % 2 == 1],
        [(r * cols + c, (r + 1) * cols + c) for r in range(rows - 1) for c in range(cols) if r % 2 == 0],
        [(r * cols + c, (r + 1) * cols + c) for r in range(rows - 1) for c in range(cols) if r % 2 == 1],
    )
    for edges in layers:
        for u, v in edges:
            _append(circuit, "CNOT" if rng.randrange(2) else "CZ", u, v)
        for q in range(n):
            if rng.random() < 0.3:
                _append(circuit, "T" if rng.randrange(2) else "TDG", q)
    for q in range(n):
        _append(circuit, "H", q)
    x, y = rng.randrange(1 << n), rng.randrange(1 << n)
    return n, circuit, x, y, {
        "source": "seeded four-matching-layer 2D grid Clifford+T brickwork",
        "grid_shape": [rows, cols],
        "layers": len(layers),
    }


_BUILDERS = {
    "grover": _grover,
    "arithmetic": _adder,
    "iqp": _iqp,
    "graphstate": _graphstate,
    "brickwork1d": _brickwork_1d,
    "brickwork2d": _brickwork_2d,
}


def task_hash(task: dict[str, Any]) -> str:
    """Hash the semantic circuit and query, excluding descriptive metadata."""
    body = {
        "n": task["n"],
        "circuit": task["circuit"],
        "global_phase": format(float(task.get("global_phase", 0.0)), ".17g"),
        "x": task["x"],
        "y": task["y"],
    }
    raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _require_exact_qasm_gates(qc) -> None:
    """Reject gates outside the exact finite subgroup before Qiskit's synthesis."""
    exact_names = {
        "h", "x", "z", "s", "sdg", "t", "tdg", "cx", "cz",
        "ccx", "swap", "id", "barrier",
    }
    allowed_u = {
        (math.pi / 2, 0.0, math.pi),  # H
        (math.pi, 0.0, math.pi),      # X
    }
    for item in qc.data:
        operation = item.operation
        name = operation.name
        if (name in exact_names and getattr(operation, "condition", None) is None):
            continue
        if name in ("p", "rz") and len(operation.params) == 1:
            theta = float(operation.params[0])
            quarter_turns = round(theta / (math.pi / 4))
            if abs(theta - quarter_turns * math.pi / 4) < 1e-12:
                continue
        if name == "u" and len(operation.params) == 3:
            angles = tuple(float(value) for value in operation.params)
            if any(all(abs(a - b) < 1e-12 for a, b in zip(angles, pattern))
                   for pattern in allowed_u):
                continue
        raise ValueError(
            f"QASM gate {name!r} is outside the supported exact Clifford+T input subset"
        )


def generate_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = []
    for family in config["families"]:
        if family not in _BUILDERS:
            raise ValueError(f"unknown workload family {family!r}")
        for scale in config["scales"]:
            for seed in config["seeds"]:
                n, gates, x, y, metadata = _BUILDERS[family](int(scale), int(seed))
                task = {
                    "id": f"{family}-n{scale}-s{seed}",
                    "family": family,
                    "group_id": family,
                    "n": n,
                    "circuit": gates,
                    "global_phase": metadata.pop("global_phase", 0.0),
                    "x": x,
                    "y": y,
                    "seed": int(seed),
                    "scale": int(scale),
                    "metadata": metadata,
                }
                task["sha256"] = task_hash(task)
                tasks.append(task)
    return tasks


def generate_batch_queries(task: dict[str, Any], count: int, seed: int) -> list[dict[str, Any]]:
    """Choose fixed-input/output T-slot sign fillings before any timing begins."""
    slots = sum(gate in ("T", "TDG") for gate, _ in task["circuit"])
    rng = random.Random(seed)
    queries = []
    for index in range(count):
        signs = [rng.choice((-1, 1)) for _ in range(slots)]
        queries.append({"id": f"{task['id']}-batch{index}", "slot_signs": signs})
    return queries


def load_qasm(path: str | Path, *, x: int = 0, y: int = 0, seed: int = 0) -> dict[str, Any]:
    """Load static QASM2/3, remove final measurements, and lower exactly to Clifford+T."""
    from qiskit import transpile
    from qiskit import qasm2, qasm3

    path = Path(path)
    text = path.read_text()
    if "openqasm 3" in text[:200].lower():
        qc = qasm3.loads(text)
    else:
        qc = qasm2.loads(text, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    qc.remove_final_measurements(inplace=True)
    if any(item.operation.name in ("reset", "measure", "if_else", "while_loop", "for_loop", "switch_case")
           or getattr(item.operation, "condition", None) is not None for item in qc.data):
        raise ValueError("QASM task must be static and contain no mid-circuit measurement or control flow")
    _require_exact_qasm_gates(qc)
    n, gates, phase = _lower_qiskit(qc, seed=seed)
    task = {
        "id": path.stem,
        "family": "imported_qasm",
        "group_id": f"imported_qasm:{task_hash({'n': n, 'circuit': gates, 'global_phase': phase, 'x': x, 'y': y})}",
        "n": n,
        "circuit": gates,
        "global_phase": phase,
        "x": int(x),
        "y": int(y),
        "seed": int(seed),
        "metadata": {"source": "QASM import", "filename": path.name,
                     "qasm_sha256": hashlib.sha256(text.encode()).hexdigest()},
    }
    task["sha256"] = task_hash(task)
    return task
