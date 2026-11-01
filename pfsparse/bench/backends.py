"""Amplitude backends for the Clifford+T structure experiments.

The public ``evaluate`` function measures preparation, contraction planning,
and evaluation separately. Quimb and cotengra are imported lazily so the
upstream pfsparse package remains usable without the benchmark extras.
"""
from __future__ import annotations

import cmath
import math
import random
import time
from typing import Any

import numpy as np

from ..graph_form import OMEGA, GraphNormalForm, circuit_graph_normal_form
from ..pathsum import build_pathsum
from ..reduce import reduce_pathsum


_DEFAULTS = {
    "planning_repeats": 8,
    "local_rounds": 3,
    "local_samples": 100,
    "memory_bytes": 2 * 1024**3,
    "seed": 0,
    "max_half": 22,
    "simplify_sequence": "ADCRS",
}


def _circuit(task: dict[str, Any]) -> list[tuple[str, tuple[int, ...]]]:
    return [(str(gate).upper(), tuple(int(q) for q in qs))
            for gate, qs in task["circuit"]]


def _phase(task: dict[str, Any]) -> complex:
    return cmath.exp(1j * float(task.get("global_phase", 0.0)))


def _pair(value: complex) -> list[float]:
    return [float(complex(value).real), float(complex(value).imag)]


def _parity_form_tn(form):
    """Export a constraint-free ParityForm with only constant-size factors.

    An affine parity is represented by an XOR chain. This keeps the largest
    parity gadget at rank three even when its support is large.
    """
    import quimb.tensor as qtn

    ps = form.ps
    if ps.constraints:
        raise ValueError("tensor export requires a constraint-free ParityForm")
    if ps.zero:
        tensor = qtn.Tensor(np.asarray(0.0, dtype=np.complex128), inds=())
        return qtn.TensorNetwork([tensor], virtual=True)

    omega = OMEGA
    tensors = []
    alive = sorted(ps.alive)
    variable_inds = {j: f"v{j}" for j in alive}
    tensors.append(qtn.Tensor(
        np.asarray((2.0 ** (ps.half_log2 / 2)) * omega ** (ps.c % 8),
                   dtype=np.complex128), inds=()))

    for j, coefficient in ps.a.items():
        data = np.asarray([omega ** 0, omega ** ((2 * coefficient) % 8)],
                          dtype=np.complex128)
        tensors.append(qtn.Tensor(data, inds=(variable_inds[j],)))

    edge_data = np.asarray([[1, 1], [1, -1]], dtype=np.complex128)
    for j, k in ps.b:
        tensors.append(qtn.Tensor(edge_data, inds=(variable_inds[j], variable_inds[k])))

    chain_index = 0
    for parity in ps.parities:
        support = parity.support()
        if not support:
            raise ValueError("ParityForm must fold constant parity factors before export")
        if len(support) == 1:
            data = np.asarray([omega ** parity.const, omega ** (1 ^ parity.const)],
                              dtype=np.complex128)
            tensors.append(qtn.Tensor(data, inds=(variable_inds[support[0]],)))
            continue

        left_ind = variable_inds[support[0]]
        right_ind = variable_inds[support[1]]
        parity_ind = f"xor{chain_index}"
        chain_index += 1
        xor = np.zeros((2, 2, 2), dtype=np.complex128)
        for a in (0, 1):
            for b in (0, 1):
                xor[a, b, a ^ b] = 1.0
        tensors.append(qtn.Tensor(xor, inds=(left_ind, right_ind, parity_ind)))
        for j in support[2:]:
            next_ind = f"xor{chain_index}"
            chain_index += 1
            tensors.append(qtn.Tensor(
                xor, inds=(parity_ind, variable_inds[j], next_ind)))
            parity_ind = next_ind
        final = np.asarray([omega ** parity.const, omega ** (1 ^ parity.const)],
                           dtype=np.complex128)
        tensors.append(qtn.Tensor(final, inds=(parity_ind,)))

    tn = qtn.TensorNetwork(tensors, virtual=True)
    return tn


def _graph_form_tn(form: GraphNormalForm, signs=None):
    import quimb.tensor as qtn

    weights = form.weights(None if signs is None else [OMEGA ** s for s in signs])
    tensors = [qtn.Tensor(
        np.asarray(form.prefactor, dtype=np.complex128), inds=())]
    for j, pair in enumerate(weights):
        tensors.append(qtn.Tensor(np.asarray(pair, dtype=np.complex128), inds=(f"g{j}",)))
    edge_data = np.asarray([[1, 1], [1, -1]], dtype=np.complex128)
    for j, k in form.edges:
        tensors.append(qtn.Tensor(edge_data, inds=(f"g{j}", f"g{k}")))
    tn = qtn.TensorNetwork(tensors, virtual=True)
    return tn


def build_graph_form(task: dict[str, Any]) -> GraphNormalForm:
    """Build the shared T/T† graph form for a fixed input/output amplitude."""
    return circuit_graph_normal_form(
        _circuit(task), int(task["n"]), int(task["x"]), int(task["y"]))


def mitm_value(task: dict[str, Any], signs=None, max_half: int = 22) -> complex:
    """Evaluate one fixed-amplitude query through the graph-form MITM route."""
    form = build_graph_form(task)
    slot_values = None if signs is None else [OMEGA ** int(sign) for sign in signs]
    return form.mitm_value(slot_values, max_half=max_half) * _phase(task)


def build_quimb_circuit(task: dict[str, Any]):
    """Construct a complex128 Quimb circuit, including the selected input state."""
    import quimb.tensor as qtn

    n = int(task["n"])
    circ = qtn.Circuit(n, dtype="complex128")
    for q in range(n):
        if (int(task["x"]) >> q) & 1:
            circ.x(q)
    methods = {
        "H": "h", "X": "x", "Z": "z", "S": "s", "SDG": "sdg",
        "T": "t", "TDG": "tdg", "CNOT": "cnot", "CZ": "cz",
    }
    for gate, qubits in _circuit(task):
        try:
            method = getattr(circ, methods[gate])
        except KeyError as exc:
            raise ValueError(f"unsupported gate {gate!r}") from exc
        method(*qubits)
    return circ


def _output_bits(y: int, n: int) -> str:
    # pfsparse uses qubit 0 as the least-significant bit; Quimb's amplitude
    # interface takes the subsystem string in circuit order.
    return format(y, f"0{n}b")[::-1]


def make_optimizer(repeats: int = 8, seed: int = 0, reusable: bool = False):
    """Construct the shared cotengra search policy, optionally with path caching."""
    import cotengra as ctg

    optimizer = ctg.ReusableHyperOptimizer if reusable else ctg.HyperOptimizer
    return optimizer(
        methods=("greedy", "labels"),
        constants={"greedy": {}, "labels": {"super_optimize": "greedy"}},
        minimize="combo",
        max_repeats=repeats,
        parallel=False,
        progbar=False,
        optlib="random",
        seed=seed,
    )


def _tree_metrics(tree: Any, tn=None) -> dict[str, float | int]:
    peak_elements = int(tree.max_size())
    resident_bytes = 0 if tn is None else sum(
        int(np.asarray(tensor.data).nbytes) for tensor in tn)
    return {
        "flops": float(tree.total_flops()),
        "write": float(tree.total_write()),
        "combo": (float(tree.combo_cost()) if tree.total_flops() or tree.total_write()
                  else 0.0),
        "peak_elements": peak_elements,
        "resident_bytes_estimate": resident_bytes,
        "peak_bytes_estimate": (peak_elements * np.dtype(np.complex128).itemsize
                                + resident_bytes),
    }


def quimb_rehearse(circ, y: int, *, simplify_sequence="ADCRS", optimize=None,
                   n: int | None = None, options: dict[str, Any] | None = None):
    """Build/simplify a Quimb amplitude network and plan its contraction.

    ``planning_s`` covers amplitude network construction, local simplification,
    and cotengra path search. The returned tree can be reused across networks
    with the same contraction geometry.
    """
    opts = dict(options or {})
    n = int(circ.N if n is None else n)
    start = time.perf_counter()
    import quimb.tensor as qtn
    with qtn.contract_strategy("greedy"):
        rehs = circ.amplitude_rehearse(
            _output_bits(int(y), n),
            simplify_sequence=simplify_sequence,
            simplify_atol=0.0,
            optimize=optimize if optimize is not None else make_optimizer(
                int(opts.get("planning_repeats", _DEFAULTS["planning_repeats"])),
                int(opts.get("seed", _DEFAULTS["seed"]))),
            dtype="complex128",
        )
    planning_s = time.perf_counter() - start
    tree = rehs["tree"]
    metrics = _tree_metrics(tree, rehs["tn"])
    return {
        "tn": rehs["tn"],
        "tree": tree,
        "planning_s": planning_s,
        "W": float(tree.contraction_width()),
        "C": metrics["flops"],
        **metrics,
    }


def quimb_contract(rehs: dict[str, Any]) -> dict[str, Any]:
    """Contract a rehearsed network using its chosen tree."""
    start = time.perf_counter()
    value = rehs["tn"].contract(all, optimize=rehs["tree"], output_inds=())
    return {"value": complex(value), "contraction_s": time.perf_counter() - start}


def _statevector(task: dict[str, Any], memory_bytes: int):
    n = int(task["n"])
    estimate = 32 * (1 << n)
    if estimate > memory_bytes:
        raise MemoryError(f"statevector estimate {estimate} bytes exceeds budget {memory_bytes}")
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    prepare_start = time.perf_counter()
    qc = QuantumCircuit(n)
    for q in range(n):
        if (int(task["x"]) >> q) & 1:
            qc.x(q)
    methods = {
        "H": "h", "X": "x", "Z": "z", "S": "s", "SDG": "sdg",
        "T": "t", "TDG": "tdg", "CNOT": "cx", "CZ": "cz",
    }
    for gate, qubits in _circuit(task):
        try:
            getattr(qc, methods[gate])(*qubits)
        except KeyError as exc:
            raise ValueError(f"unsupported gate {gate!r}") from exc
    prepare_s = time.perf_counter() - prepare_start
    evaluate_start = time.perf_counter()
    state = Statevector.from_instruction(qc)
    evaluate_s = time.perf_counter() - evaluate_start
    return complex(state.data[int(task["y"])]), {
        "statevector_bytes_estimate": estimate,
        "n": n,
    }, prepare_s, evaluate_s


def _candidate_forms(task, options):
    from ..basis import local_search
    from ..basis import compress_rank

    circuit = _circuit(task)
    ps = build_pathsum(circuit, int(task["n"]), int(task["x"]), int(task["y"]))
    reduced = reduce_pathsum(ps, order="min_degree")
    ranked = compress_rank(reduced)
    local = local_search(
        ranked,
        max_rounds=int(options["local_rounds"]),
        sample=int(options["local_samples"]),
        rng=random.Random(int(options["seed"])),
    )
    return ps, {"reduced": reduced, "rank": ranked, "local": local}


def _form_structure(form) -> dict[str, Any]:
    ps = form.ps
    degrees = {j: 0 for j in ps.alive}
    for a, b in ps.b:
        degrees[a] = degrees.get(a, 0) + 1
        degrees[b] = degrees.get(b, 0) + 1
    return {
        "r": form.r,
        "t": form.t,
        "support_sum": form.support_sum,
        "q_size": form.q_size,
        "L": form.L,
        "quadratic_edges": len(ps.b),
        "quadratic_maxdegree": max(degrees.values(), default=0),
        "zero": bool(ps.zero),
    }


def _run_quimb_candidate(tn, options):
    import quimb.tensor as qtn

    optimizer = make_optimizer(int(options["planning_repeats"]), int(options["seed"]))
    start = time.perf_counter()
    with qtn.contract_strategy("greedy"):
        if options["simplify_sequence"]:
            tn.full_simplify_(seq=options["simplify_sequence"], atol=0.0,
                              equalize_norms=False)
    tree = tn.contraction_tree(optimize=optimizer, output_inds=())
    planning_s = time.perf_counter() - start
    metrics = _tree_metrics(tree, tn)
    return {"tn": tn, "tree": tree, "planning_s": planning_s,
            **metrics}


def _contract_candidate(rehs: dict[str, Any]) -> tuple[complex, float]:
    result = quimb_contract(rehs)
    return result["value"], float(result["contraction_s"])


def _estimate_candidate(name: str, rehs: dict[str, Any], memory_bytes: int):
    metrics = {key: rehs[key] for key in (
        "flops", "write", "combo", "peak_elements", "peak_bytes_estimate")}
    over = metrics["peak_bytes_estimate"] > memory_bytes
    return {
        "name": name,
        "status": "memory_limit" if over else "ok",
        "reason": "estimated intermediate exceeds memory budget" if over else None,
        **metrics,
    }


def _quimb_candidates(task, strategy, options):
    prepare_start = time.perf_counter()
    if strategy == "native":
        circ = build_quimb_circuit(task)
        prepared = time.perf_counter()
        rehs = quimb_rehearse(
            circ, int(task["y"]), n=int(task["n"]),
            simplify_sequence=options["simplify_sequence"], options=options)
        estimate = _estimate_candidate("native", rehs, int(options["memory_bytes"]))
        return [{"name": "native", "rehs": rehs, "estimate": estimate,
                 "prepare_s": prepared - prepare_start,
                 "plan_s": float(rehs["planning_s"])}], []

    if strategy not in ("fixed", "adaptive"):
        raise ValueError(f"unknown Quimb strategy {strategy!r}")
    forms_start = time.perf_counter()
    _, forms = _candidate_forms(task, options)
    forms_prepare_s = time.perf_counter() - forms_start
    candidate_defs = []
    if strategy == "adaptive":
        native_start = time.perf_counter()
        native_circuit = build_quimb_circuit(task)
        native_prepare_s = time.perf_counter() - native_start
        native_rehs = quimb_rehearse(
            native_circuit, int(task["y"]), n=int(task["n"]),
            simplify_sequence=options["simplify_sequence"], options=options)
        candidate_defs.append(("native", native_rehs, native_prepare_s))
        for name, form in forms.items():
            start = time.perf_counter()
            tn = _parity_form_tn(form)
            prepare_s = time.perf_counter() - start
            rehs = _run_quimb_candidate(tn, options)
            candidate_defs.append((
                name, rehs, prepare_s + (forms_prepare_s if name == "reduced" else 0.0)))
        start = time.perf_counter()
        graph = build_graph_form(task)
        tn = _graph_form_tn(graph)
        graph_prepare_s = time.perf_counter() - start
        graph_rehs = _run_quimb_candidate(tn, options)
        candidate_defs.append(("graph", graph_rehs, graph_prepare_s))
    else:
        ranked = forms["local"]
        tn = _parity_form_tn(ranked)
        prepare_s = time.perf_counter() - prepare_start
        rehs = _run_quimb_candidate(tn, options)
        candidate_defs.append(("fixed", rehs, prepare_s))

    result = []
    for name, rehs, prepare_s in candidate_defs:
        estimate = _estimate_candidate(name, rehs, int(options["memory_bytes"]))
        result.append({"name": name, "rehs": rehs, "estimate": estimate,
                       "prepare_s": prepare_s,
                       "plan_s": float(rehs.get("planning_s", 0.0))})
    structures = {name: _form_structure(form) for name, form in forms.items()}
    return result, structures


def evaluate(task: dict[str, Any], backend: str, strategy: str = "native",
             options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate ``<y|C|x>`` and return JSON-safe timings and route metadata.

    Backends are ``quimb``, ``statevector``, and ``mitm``. Quimb strategies
    are ``native`` (circuit TN), ``fixed`` (reduce/rank/local pipeline), and
    ``adaptive`` (the planner chooses from all five representations).
    """
    opts = dict(_DEFAULTS)
    opts.update(options or {})
    backend = str(backend).lower()
    n, x, y = int(task["n"]), int(task["x"]), int(task["y"])
    if not 0 <= x < (1 << n) or not 0 <= y < (1 << n):
        raise ValueError("x and y must be n-bit computational basis indices")
    started = time.perf_counter()
    structures: dict[str, Any] = {}
    candidates: list[dict[str, Any]] = []

    if backend == "statevector":
        try:
            value, state_meta, prepare_s, evaluate_s = _statevector(
                task, int(opts["memory_bytes"]))
        except MemoryError as exc:
            return {
                "backend": backend, "strategy": strategy, "status": "memory_limit",
                "reason": str(exc), "amplitude": None,
                "timings_s": {"prepare": 0.0, "plan": 0.0, "evaluate": 0.0},
                "total_s": time.perf_counter() - started,
                "selected_candidate": None, "candidates": [],
                "structure": {"statevector_bytes_estimate": 32 * (1 << n)},
            }
        value *= _phase(task)
        return {
            "backend": backend, "strategy": strategy, "status": "ok",
            "amplitude": _pair(value),
            "timings_s": {"prepare": prepare_s, "plan": 0.0,
                          "evaluate": evaluate_s},
            "total_s": time.perf_counter() - started,
            "selected_candidate": "qiskit_statevector", "candidates": [],
            "structure": state_meta,
        }

    if backend == "mitm":
        prep_start = time.perf_counter()
        form = build_graph_form(task)
        prepared = time.perf_counter()
        try:
            value = form.mitm_value(max_half=int(opts["max_half"]))
        except ValueError as exc:
            return {
                "backend": backend, "strategy": strategy, "status": "unsupported",
                "reason": str(exc), "amplitude": None,
                "timings_s": {"prepare": prepared - prep_start, "plan": 0.0,
                              "evaluate": 0.0},
                "total_s": time.perf_counter() - started,
                "selected_candidate": None, "candidates": [],
                "structure": {"graph_vertices": form.n, "slots": form.t},
            }
        finished = time.perf_counter()
        value *= _phase(task)
        return {
            "backend": backend, "strategy": strategy, "status": "ok",
            "amplitude": _pair(value),
            "timings_s": {"prepare": prepared - prep_start, "plan": 0.0,
                          "evaluate": finished - prepared},
            "total_s": time.perf_counter() - started,
            "selected_candidate": "graph_mitm", "candidates": [],
            "structure": {"graph_vertices": form.n, "slots": form.t},
        }

    if backend != "quimb":
        raise ValueError(f"unknown backend {backend!r}")
    candidates, structures = _quimb_candidates(task, strategy, opts)
    feasible = [candidate for candidate in candidates
                if candidate["estimate"]["status"] == "ok"]
    if not feasible:
        return {
            "backend": backend, "strategy": strategy, "status": "memory_limit",
            "reason": "all candidates exceed the estimated intermediate memory budget",
            "amplitude": None, "timings_s": {
                "prepare": sum(c["prepare_s"] for c in candidates),
                "plan": sum(c["plan_s"] for c in candidates), "evaluate": 0.0},
            "total_s": time.perf_counter() - started,
            "selected_candidate": None,
            "candidates": [c["estimate"] for c in candidates], "structure": structures,
        }
    selected = min(feasible, key=lambda c: (c["estimate"]["combo"], c["name"]))
    value, contraction_s = _contract_candidate(selected["rehs"])
    value *= _phase(task)
    return {
        "backend": backend, "strategy": strategy, "status": "ok",
        "amplitude": _pair(value),
        "timings_s": {
            "prepare": sum(c["prepare_s"] for c in candidates),
            "plan": sum(c["plan_s"] for c in candidates),
            "evaluate": contraction_s,
        },
        "total_s": time.perf_counter() - started,
        "selected_candidate": selected["name"],
        "selected_estimate": selected["estimate"],
        "candidates": [c["estimate"] for c in candidates],
        "structure": structures,
    }

