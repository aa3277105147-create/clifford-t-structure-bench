"""Cost-based backend selection for fixed computational-basis amplitudes."""
from __future__ import annotations

from collections import defaultdict
from time import perf_counter

import numpy as np

from ..pathsum import build_pathsum
from ..reduce import reduce_pathsum


BACKENDS = ("quimb", "statevector", "mitm")
POLICIES = ("fixed", "raw", "effective")
RAW_FEATURES = ("n", "gates", "depth", "t", "interaction_edges", "maxdegree")
EFFECTIVE_FEATURES = (
    "r", "L", "support_sum", "quadratic_edges", "quadratic_maxdegree",
)


def extract_features(task: dict, effective: bool = True) -> dict:
    """Measure inexpensive circuit features and, optionally, the reduced parity form.

    The effective features end at ``reduce_pathsum``. They do not build a graph
    normal form, optimize a tensor network, or evaluate the amplitude.
    """
    start = perf_counter()
    n = int(task["n"])
    circuit = task["circuit"]
    layers = [0] * n
    edges = set()
    t_count = 0
    for gate, qubits in circuit:
        qs = tuple(int(q) for q in qubits)
        if gate in ("T", "TDG"):
            t_count += 1
        if gate in ("CNOT", "CZ"):
            a, b = qs
            edge = (min(a, b), max(a, b))
            edges.add(edge)
        if qs:
            layer = max(layers[q] for q in qs) + 1
            for q in qs:
                layers[q] = layer
    raw = {
        "n": n,
        "gates": len(circuit),
        "depth": max(layers, default=0),
        "t": t_count,
        "interaction_edges": len(edges),
        "maxdegree": max((sum(q in edge for edge in edges) for q in range(n)), default=0),
    }
    raw_seconds = perf_counter() - start

    effective_values = None
    effective_seconds = 0.0
    if effective:
        start = perf_counter()
        ps = build_pathsum(circuit, n, int(task["x"]), int(task["y"]))
        form = reduce_pathsum(ps, order="min_degree")
        reduced = form.ps
        quadratic_degree = defaultdict(int)
        for a, b in reduced.b:
            quadratic_degree[a] += 1
            quadratic_degree[b] += 1
        effective_values = {
            "r": form.r,
            "L": form.L,
            "support_sum": form.support_sum,
            "quadratic_edges": len(reduced.b),
            "quadratic_maxdegree": max(quadratic_degree.values(), default=0),
        }
        effective_seconds = perf_counter() - start
    return {
        "raw": raw,
        "effective": effective_values,
        "timings_s": {"raw_features": raw_seconds, "effective_features": effective_seconds},
    }


def _feature_vector(features: dict, policy: str) -> np.ndarray:
    names = RAW_FEATURES if policy == "raw" else RAW_FEATURES + EFFECTIVE_FEATURES
    values = [float(features["raw"][key]) for key in RAW_FEATURES]
    if policy == "effective":
        values.extend(float(features["effective"][key]) for key in EFFECTIVE_FEATURES)
    # Log scaling keeps gate counts and graph sizes on comparable ranges.
    return np.log1p(np.asarray(values, dtype=float))


def _fit_ridge(rows: list[tuple[np.ndarray, float]]) -> dict:
    x = np.vstack([row[0] for row in rows])
    y = np.asarray([row[1] for row in rows], dtype=float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    normalized = (x - mean) / scale
    y_mean = float(y.mean())
    # Intercept is unpenalized; alpha=1 is fixed for every family and fold.
    weights = np.linalg.solve(normalized.T @ normalized + np.eye(x.shape[1]),
                              normalized.T @ (y - y_mean))
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weights": weights.tolist(),
        "intercept": y_mean,
        "alpha": 1.0,
    }


def _predict_cost(model: dict, features: dict, policy: str) -> float:
    x = _feature_vector(features, policy)
    normalized = (x - np.asarray(model["mean"])) / np.asarray(model["scale"])
    return float(model["intercept"] + normalized @ np.asarray(model["weights"]))


def _record_succeeded(row: dict) -> bool:
    verification = row.get("verification", True)
    if isinstance(verification, dict):
        verification = verification.get("ok", verification.get("status") == "ok")
    return row.get("status") == "ok" and verification is not False


def fit_selector(tasks: list[dict], backend_records: list[dict], feature_records: list[dict],
                 timeout_s: float = 60.0) -> dict:
    """Fit per-backend ridge models and fixed-backend baselines by family holdout.

    Runtime labels are log10(seconds). A backend that fails any repeat is
    assigned the same two-timeout penalty as every other failing backend.
    Feature rows must already be available at routing time; their extraction
    costs are reported separately by ``evaluate_selectors``.
    """
    task_by_id = {task["id"]: task for task in tasks}
    feature_by_id = {row["task_id"]: row for row in feature_records
                     if row.get("status", "ok") == "ok"}
    grouped = defaultdict(list)
    for row in backend_records:
        grouped[(row["task_id"], row["backend"])].append(row)

    # Aggregate repeated measurements only when all repeats completed.
    aggregate = {}
    for (task_id, backend), rows in grouped.items():
        good = all(_record_succeeded(row) for row in rows)
        if good:
            seconds = float(np.median([float(row["total_s"]) for row in rows]))
        else:
            seconds = 2.0 * timeout_s
        aggregate[(task_id, backend)] = {"ok": good, "score_s": seconds, "rows": rows}

    families = sorted({task["family"] for task in tasks})
    folds = {}
    for heldout in families:
        training = [task for task in tasks
                    if task["family"] != heldout
                    and task["id"] in feature_by_id]
        scores = {backend: [] for backend in BACKENDS}
        train_rows = {policy: {backend: [] for backend in BACKENDS}
                      for policy in ("raw", "effective")}
        for task in training:
            fr = feature_by_id[task["id"]]
            for backend in BACKENDS:
                observed = aggregate.get((task["id"], backend))
                if observed is None:
                    continue
                log_cost = float(np.log10(max(observed["score_s"], 1e-12)))
                scores[backend].append(observed["score_s"])
                for policy in ("raw", "effective"):
                    if policy == "effective" and fr.get("effective") is None:
                        continue
                    train_rows[policy][backend].append((_feature_vector(fr, policy), log_cost))
        fixed_backend = min(BACKENDS, key=lambda b: (
            float(np.mean(scores[b])) if scores[b] else float("inf"), b))
        fold = {"fixed_backend": fixed_backend, "raw": {}, "effective": {},
                "training_families": sorted({task["family"] for task in training})}
        for policy in ("raw", "effective"):
            for backend in BACKENDS:
                rows = train_rows[policy][backend]
                if rows:
                    fold[policy][backend] = _fit_ridge(rows)
        folds[heldout] = fold
    return {
        "version": 1,
        "target": "log10(seconds)",
        "feature_names": {"raw": list(RAW_FEATURES),
                          "effective": list(RAW_FEATURES + EFFECTIVE_FEATURES)},
        "backends": list(BACKENDS),
        "timeout_s": timeout_s,
        "folds": folds,
    }


def predict_with_features(task: dict, model_bundle: dict, policy: str,
                          features: dict) -> tuple[str, dict]:
    """Return the held-out fold's route and predicted costs for one task."""
    if policy not in POLICIES:
        raise ValueError(f"unknown selector policy {policy!r}")
    fold = model_bundle["folds"][task["family"]]
    if policy == "fixed":
        return fold["fixed_backend"], {"fixed": 0.0}
    models = fold[policy]
    costs = {backend: _predict_cost(model, features, policy)
             for backend, model in models.items()}
    if not costs:
        raise ValueError(f"no trained models for policy {policy!r}, family {task['family']!r}")
    selected = min(costs, key=costs.get)
    return selected, costs


def select_backend(task: dict, model_bundle: dict, policy: str) -> dict:
    """Route a task through the model trained without its circuit family.

    ``fixed`` performs no feature extraction. Learned policies report feature
    and inference time so the caller can include routing in its task budget.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown selector policy {policy!r}")
    fold = model_bundle["folds"][task["family"]]
    if policy == "fixed":
        return {"selected_backend": fold["fixed_backend"], "policy": policy,
                "heldout_family": task["family"], "timings_s": {"features": 0.0, "inference": 0.0},
                "predicted_costs": {}}
    feature_start = perf_counter()
    features = extract_features(task, effective=policy == "effective")
    feature_s = perf_counter() - feature_start
    inference_start = perf_counter()
    backend, costs = predict_with_features(task, model_bundle, policy, features)
    inference_s = perf_counter() - inference_start
    return {"selected_backend": backend, "policy": policy,
            "heldout_family": task["family"],
            "timings_s": {"features": feature_s, "inference": inference_s},
            "predicted_costs": costs,
            "feature_data": features}


def evaluate_selectors(tasks: list[dict], backend_records: list[dict],
                       feature_records: list[dict], timeout_s: float = 60.0) -> dict:
    """Evaluate fixed, raw-feature, and effective-feature routes by held-out family.

    The result gives out-of-fold predictions and observed-cost estimates from
    the recorded backend runs. The calling benchmark should separately replay
    each selected route to measure its true online total time and peak RSS.
    """
    bundle = fit_selector(tasks, backend_records, feature_records, timeout_s)
    task_by_id = {task["id"]: task for task in tasks}
    feature_by_id = {row["task_id"]: row for row in feature_records
                     if row.get("status", "ok") == "ok"}
    outcomes = defaultdict(list)
    for row in backend_records:
        outcomes[(row["task_id"], row["backend"])].append(row)
    predictions = []
    aggregates = {policy: [] for policy in POLICIES}
    for task in tasks:
        fr = feature_by_id.get(task["id"])
        if fr is None:
            continue
        backend_costs = {}
        backend_status = {}
        for backend in BACKENDS:
            rows = outcomes.get((task["id"], backend), [])
            if rows and all(_record_succeeded(row) for row in rows):
                backend_costs[backend] = float(np.median([float(row["total_s"]) for row in rows]))
                backend_status[backend] = "ok"
            else:
                backend_costs[backend] = 2.0 * timeout_s
                backend_status[backend] = "failed"
        oracle_backend = min(backend_costs, key=backend_costs.get)
        for policy in POLICIES:
            if policy == "effective" and fr.get("effective") is None:
                continue
            backend, predicted = predict_with_features(task, bundle, policy, fr)
            feature_s = 0.0
            if policy == "raw":
                feature_s = float(fr.get("timings_s", {}).get("raw_features", 0.0))
            elif policy == "effective":
                feature_s = sum(float(v) for v in fr.get("timings_s", {}).values())
            selected_cost = backend_costs[backend]
            selector_total_s = feature_s + selected_cost
            record = {
                "task_id": task["id"],
                "family": task["family"],
                "heldout_family": task["family"],
                "policy": policy,
                "selected_backend": backend,
                "selected_status": backend_status[backend],
                "selected_backend_s": selected_cost,
                "feature_extraction_s": feature_s,
                "predicted_log10_costs": predicted,
                "selector_total_s_estimate": selector_total_s,
                "oracle_backend": oracle_backend,
                "oracle_regret": selected_cost / max(backend_costs[oracle_backend], 1e-12),
            }
            predictions.append(record)
            aggregates[policy].append(record)

    summary = {}
    for policy, rows in aggregates.items():
        if not rows:
            continue
        summary[policy] = {
            "tasks": len(rows),
            "completed": sum(row["selected_status"] == "ok" for row in rows),
            "median_estimated_total_s": float(np.median(
                [row["selector_total_s_estimate"] for row in rows])),
            "median_oracle_regret": float(np.median([row["oracle_regret"] for row in rows])),
        }
    return {"models": bundle, "predictions": predictions, "summary": summary}
