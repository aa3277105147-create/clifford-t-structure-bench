"""Repeated amplitude queries on one Clifford+T circuit skeleton.

The third method measures a single reusable graph normal form over independently
selected T/T† slot values. The first method deliberately rebuilds that form for
every query. Both use the same MITM evaluator.
"""
from __future__ import annotations

import cmath
import random
import time
from typing import Any, Iterable

from ..graph_form import OMEGA, circuit_graph_normal_form


def generate_sign_vectors(task: dict[str, Any], count: int, seed: int = 0) -> list[list[int]]:
    """Create reproducible nested T/T† fillings; row zero matches the input circuit."""
    if count < 1:
        raise ValueError("count must be positive")
    slots = [gate for gate, _ in task["circuit"] if gate in ("T", "TDG")]
    default = [1 if gate == "T" else -1 for gate in slots]
    rows = [default]
    rng = random.Random(seed)
    while len(rows) < count:
        rows.append([rng.choice((-1, 1)) for _ in slots])
    return rows


def _validate_sign_vectors(task: dict[str, Any], sign_vectors: Iterable[Iterable[int]]) -> list[tuple[int, ...]]:
    expected = sum(gate in ("T", "TDG") for gate, _ in task["circuit"])
    rows = [tuple(row) for row in sign_vectors]
    if not rows:
        raise ValueError("at least one sign vector is required")
    for row in rows:
        if len(row) != expected or any(sign not in (-1, 1) for sign in row):
            raise ValueError(f"each sign vector must contain {expected} values in {{-1, 1}}")
    return rows


def _complex_pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def _global_phase(task: dict[str, Any]) -> complex:
    return cmath.exp(1j * float(task.get("global_phase", 0.0)))


def _check_mitm_size(vertices: int, max_half: int) -> None:
    if vertices - vertices // 2 > max_half:
        raise NotImplementedError(
            f"graph has {vertices} vertices; MITM supports a right half of at most {max_half}"
        )


def _quimb_reusable(task: dict[str, Any], rows: list[tuple[int, ...]], options: dict[str, Any]):
    """Run per-query Quimb contractions through one cotengra path cache."""
    from .backends import make_optimizer, build_quimb_circuit, quimb_contract, quimb_rehearse

    setup_started = time.perf_counter()
    optimizer = make_optimizer(
        repeats=int(options.get("planning_repeats", 8)),
        seed=int(options.get("seed", 0)),
        reusable=True,
    )
    setup_s = time.perf_counter() - setup_started
    amplitudes, per_query = [], []
    prepare_s = evaluate_s = 0.0
    plan_s = setup_s
    for row in rows:
        t0 = time.perf_counter()
        variant = dict(task)
        slot = iter(row)
        variant["circuit"] = [
            ["T" if next(slot) == 1 else "TDG", qs] if gate in ("T", "TDG") else [gate, qs]
            for gate, qs in task["circuit"]
        ]
        circ = build_quimb_circuit(variant)
        t1 = time.perf_counter()
        # The helper performs tensor-network construction and path search; its
        # reusable optimizer caches the same contraction geometry across rows.
        rehs = quimb_rehearse(
            circ, int(task["y"]),
            simplify_sequence=str(options.get("simplify_sequence", "ADCRS")),
            optimize=optimizer,
        )
        contraction = quimb_contract(rehs)
        t3 = time.perf_counter()
        prepare_s += t1 - t0
        plan_s += float(rehs["planning_s"])
        evaluate_s += float(contraction["contraction_s"])
        value = contraction["value"] * _global_phase(task)
        phase_finished = time.perf_counter()
        evaluate_s += phase_finished - t3
        per_query.append(phase_finished - t0)
        amplitudes.append(_complex_pair(value))
    per_query = [seconds + setup_s / len(rows) for seconds in per_query]
    return amplitudes, {"prepare": prepare_s, "plan": plan_s, "evaluate": evaluate_s}, per_query


def evaluate_batch(
    task: dict[str, Any],
    method: str,
    sign_vectors: Iterable[Iterable[int]],
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate fixed input/output amplitudes over T/T† fillings.

    ``independent`` rebuilds the graph form for every row; ``shared_graph``
    builds it once. ``quimb_cached`` rebuilds each tensor network while sharing
    cotengra's reusable contraction-path cache.
    """
    rows = _validate_sign_vectors(task, sign_vectors)
    options = dict(options or {})
    if method not in {"independent", "shared_graph", "quimb_cached"}:
        raise ValueError(f"unknown batch method {method!r}")
    max_half = int(options.get("max_half", 22))
    total_started = time.perf_counter()

    if method == "quimb_cached":
        amplitudes, timings, per_query = _quimb_reusable(task, rows, options)
        graph_vertices = None
    elif method == "independent":
        amplitudes = []
        per_query = []
        prepare_s = evaluate_s = 0.0
        graph_vertices = None
        for row in rows:
            started = time.perf_counter()
            form = circuit_graph_normal_form(
                [(gate, tuple(qs)) for gate, qs in task["circuit"]],
                int(task["n"]), int(task["x"]), int(task["y"]),
            )
            prepared = time.perf_counter()
            _check_mitm_size(form.n, max_half)
            value = form.mitm_value([OMEGA ** sign for sign in row], max_half=max_half)
            value *= _global_phase(task)
            finished = time.perf_counter()
            prepare_s += prepared - started
            evaluate_s += finished - prepared
            per_query.append(finished - started)
            amplitudes.append(_complex_pair(value))
            graph_vertices = form.n
        timings = {"prepare": prepare_s, "plan": 0.0, "evaluate": evaluate_s}
    else:
        started = time.perf_counter()
        form = circuit_graph_normal_form(
            [(gate, tuple(qs)) for gate, qs in task["circuit"]],
            int(task["n"]), int(task["x"]), int(task["y"]),
        )
        prepared = time.perf_counter()
        _check_mitm_size(form.n, max_half)
        amplitudes, per_query = [], []
        evaluate_s = 0.0
        for row in rows:
            query_started = time.perf_counter()
            value = form.mitm_value([OMEGA ** sign for sign in row], max_half=max_half)
            value *= _global_phase(task)
            query_finished = time.perf_counter()
            evaluate_s += query_finished - query_started
            per_query.append(query_finished - query_started)
            amplitudes.append(_complex_pair(value))
        timings = {"prepare": prepared - started, "plan": 0.0, "evaluate": evaluate_s}
        per_query = [seconds + timings["prepare"] / len(rows) for seconds in per_query]
        graph_vertices = form.n

    total_s = time.perf_counter() - total_started
    return {
        "method": method,
        "amplitudes": amplitudes,
        "timings_s": timings,
        "per_query_s": per_query,
        "total_s": total_s,
        "metadata": {
            "query_count": len(rows),
            "unique_query_count": len(set(rows)),
            "slot_count": len(rows[0]),
            "graph_vertices": graph_vertices,
            "path_cache": "cotengra ReusableHyperOptimizer" if method == "quimb_cached" else None,
        },
    }
