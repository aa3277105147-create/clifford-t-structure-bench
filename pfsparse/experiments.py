"""Reproducible experiment runner with explicit failures and output selection.

Examples:
    python -m pfsparse.experiments --family F3 --sizes 8 16 32 64 --seeds 10 --out f3.csv
    python -m pfsparse.experiments --family F3 --width-exponent 1.5 --orders natural min_degree min_fill --sizes 8 16 32 --out sparse_f3.csv

All orders use the same circuit and symbolically chosen output for a family/size/
seed. Such an output is only a candidate: exact cancellation is checked when the
rank-compressed form is small; larger outputs are explicitly marked unverified.
Rows with timeouts, exhausted retries, or resource limits remain in the CSV.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from functools import lru_cache
import math
import random
import signal
import threading
import time

from . import circuits as C
from .pathsum import build_pathsum
from .metrics import bounded_graph_metrics

ORDERS = ("natural", "min_degree", "min_fill")
STATS = ("r", "t", "support_sum", "q_size", "L", "parity_rank", "half_log2", "zero")
BASE_FIELDS = ("family", "size", "seed", "order", "width_exponent", "brick_depth",
               "n", "s", "t", "h", "ratio_s_t", "actual_depth", "interaction_max_degree",
               "status", "stage", "output_y_hex", "output_method", "output_verification",
               "tries", "exact_zero_rejections", "preparation_seconds", "preparation_reused",
               "reduce_seconds", "rank_seconds", "search_seconds", "seconds", "timeout_seconds",
               "ls_rounds_budget", "ls_sample_budget", "verify_max_r", "tw_max_nodes",
               "note", "tw_upper_raw", "tw_status_raw", "tw_upper_rank", "tw_status_rank")
FIELDS = list(BASE_FIELDS) + [stage + "_" + key for stage in ("red", "rank", "ls") for key in STATS]


def _qubits(family: str, t: int, width_exponent: float) -> int:
    if family in ("F0", "F1"):
        return max(4, t // 2)
    if family == "F2":
        return math.ceil(4 * t ** width_exponent)
    if family == "F3":
        return max(2, math.ceil((4 * t ** width_exponent) ** 0.5)) ** 2
    if family == "F4":
        return max(4, (2 * t) // 5 // 2 * 2)
    raise ValueError(f"unknown family {family}")


def make_circuit(family: str, size: int, rng: random.Random,
                 width_exponent: float = 1.0, depth: int = 4):
    """size is target T count; actual t is always recorded separately.

    F2/F3 default n=Theta(t) are controls with bounded s/t. Set width_exponent>1
    to test the intended sparse-T regime at fixed physical depth.
    """
    if size < 1 or width_exponent <= 0 or depth < 1:
        raise ValueError("size, width_exponent and depth must be positive")
    t, n = size, _qubits(family, size, width_exponent)
    if family == "F0":
        return C.random_clifford_t(n, 4 * t, t, rng), n
    if family == "F1":
        return C.phase_polynomial_circuit(n, t, n_cliff=max(t, t * t // 4), rng=rng), n
    if family == "F2":
        return C.brickwork_1d(n, depth=depth, t=t, rng=rng), n
    if family == "F3":
        side = math.isqrt(n)
        return C.brickwork_2d(side, side, depth=depth, t=t, rng=rng), n
    return C.qaoa_clifford_t(n, C.random_regular_graph_edges(n, 3, rng), 1, rng), n


def _circuit_metrics(circ, n):
    levels = [0] * n
    neighbours = [set() for _ in range(n)]
    for _, qs in circ:
        level = 1 + max((levels[q] for q in qs), default=0)
        for q in qs:
            levels[q] = level
        if len(qs) == 2:
            u, v = qs
            neighbours[u].add(v)
            neighbours[v].add(u)
    return {"actual_depth": max(levels, default=0),
            "interaction_max_degree": max(map(len, neighbours), default=0)}


@lru_cache(maxsize=1)
def _prepare_case(family, size, seed, width_exponent, depth, max_tries, verify_max_r):
    """Canonical preparation, independent of the elimination order under test."""
    from . import reduce as R
    start = time.perf_counter()
    circ, n = make_circuit(family, size, random.Random(seed), width_exponent, depth)
    output_rng = random.Random((seed << 32) ^ (size << 8) ^ ord(family[-1]) ^ 0x51CE)
    zero_rejections = 0
    for tries in range(1, max_tries + 1):
        y = R.find_support_output(circ, n, 0, output_rng)
        if y is None:
            # A unitary acting on a basis state has at least one supported output.
            raise AssertionError("symbolic output constraints inconsistent for a unitary circuit")
        ps = build_pathsum(circ, n, 0, y)
        canonical = R.compress_to_rank(R.reduce_pathsum(ps, order="min_degree"))
        if canonical.ps.zero:
            zero_rejections += 1
            continue
        if canonical.r <= verify_max_r:
            _, coefficients = canonical.ps.evaluate_exact(max_variables=verify_max_r)
            if not any(coefficients):
                zero_rejections += 1
                continue
            verification = "exact_nonzero"
        else:
            verification = "unverified_above_rank_budget"
        return {"circ": circ, "n": n, "ps": ps, "y": y, "tries": tries,
                "verification": verification, "zero_rejections": zero_rejections,
                "seconds": time.perf_counter() - start}
    return {"circ": circ, "n": n, "ps": None, "y": None, "tries": max_tries,
            "verification": "all_candidates_exactly_zero", "zero_rejections": zero_rejections,
            "seconds": time.perf_counter() - start}


class ExperimentTimeout(TimeoutError):
    pass


@contextmanager
def _deadline(seconds):
    if seconds is None or seconds <= 0:
        yield
        return
    if not hasattr(signal, "setitimer") or threading.current_thread() is not threading.main_thread():
        raise RuntimeError("strict per-case timeout requires POSIX main-thread execution; pass timeout=None otherwise")
    def expired(_signum, _frame):
        raise ExperimentTimeout("per-case wall-time budget exhausted")
    previous_handler = signal.signal(signal.SIGALRM, expired)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    start = time.perf_counter()
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, max(1e-6, previous_timer[0] - (time.perf_counter() - start)), previous_timer[1])


def run_one(family: str, size: int, seed: int, order: str, ls_rounds: int,
            max_tries: int = 20, *, width_exponent: float = 1.0, depth: int = 4,
            ls_sample: int = 200, timeout: float | None = 120.0,
            tw_max_nodes: int = 128, verify_max_r: int = 16,
            max_qubits: int | None = None):
    from . import reduce as R
    from . import basis as Bs
    if order not in ORDERS or max_tries < 1 or verify_max_r < 0:
        raise ValueError("invalid order or experiment budget")
    row = {"family": family, "size": size, "seed": seed, "order": order,
           "width_exponent": width_exponent, "brick_depth": depth,
           "timeout_seconds": timeout, "ls_rounds_budget": ls_rounds,
           "ls_sample_budget": ls_sample, "verify_max_r": verify_max_r,
           "tw_max_nodes": tw_max_nodes, "output_method": "symbolic_linear_candidate",
           "stage": "prepare", "status": "running"}
    start = time.perf_counter()
    n_est = _qubits(family, size, width_exponent)
    row["n"] = n_est
    if max_qubits is not None and n_est > max_qubits:
        row.update(status="resource_limit", n=n_est, note="declared qubit budget", seconds=0.0)
        return row
    try:
        with _deadline(timeout):
            hits = _prepare_case.cache_info().hits
            case = _prepare_case(family, size, seed, width_exponent, depth, max_tries, verify_max_r)
            circ, n = case["circ"], case["n"]
            actual_t = C.t_count(circ)
            row.update(n=n, s=len(circ), t=actual_t, h=sum(g == "H" for g, _ in circ),
                       ratio_s_t=len(circ) / actual_t if actual_t else None,
                       output_y_hex=hex(case["y"]) if case["y"] is not None else None,
                       output_verification=case["verification"], tries=case["tries"],
                       exact_zero_rejections=case["zero_rejections"],
                       preparation_seconds=round(case["seconds"], 6),
                       preparation_reused=_prepare_case.cache_info().hits > hits)
            row.update(_circuit_metrics(circ, n))
            if case["ps"] is None:
                row.update(status="zero_output_retries_exhausted", stage="prepare")
                return row
            ps = case["ps"]
            row["stage"] = "reduce"
            stage_start = time.perf_counter()
            pf = R.reduce_pathsum(ps, order=order)
            row["reduce_seconds"] = round(time.perf_counter() - stage_start, 6)
            row.update({"red_" + k: v for k, v in pf.stats().items()})
            row["stage"] = "rank"
            stage_start = time.perf_counter()
            pf2 = R.compress_to_rank(pf)
            row["rank_seconds"] = round(time.perf_counter() - stage_start, 6)
            row.update({"rank_" + k: v for k, v in pf2.stats().items()})
            row["stage"] = "local_search"
            stage_start = time.perf_counter()
            search_rng = random.Random((seed << 32) ^ (size << 8) ^ 0xB4515)
            pf3 = Bs.local_search(pf2, max_rounds=ls_rounds, sample=ls_sample, rng=search_rng)
            row["search_seconds"] = round(time.perf_counter() - stage_start, 6)
            row.update({"ls_" + k: v for k, v in pf3.stats().items()})
            if pf3.ps.zero and case["verification"] == "exact_nonzero":
                raise AssertionError("elimination contradicts exact nonzero preparation")
            known_zero = pf3.ps.zero
            if case["verification"] != "exact_nonzero" and pf3.r <= verify_max_r:
                row["stage"] = "verify_output"
                _, coefficients = pf3.ps.evaluate_exact(max_variables=verify_max_r)
                known_zero = not any(coefficients)
                row["output_verification"] = ("exact_zero_after_reduction" if known_zero
                                               else "exact_nonzero_after_reduction")
            elif known_zero:
                row["output_verification"] = "exact_zero_from_reduction"
            row["stage"] = "treewidth"
            for name, value in [("raw", ps), ("rank", pf2.ps)]:
                metrics = bounded_graph_metrics(value, max_nodes=tw_max_nodes)
                row["tw_upper_" + name] = metrics["tw_upper"]
                row["tw_status_" + name] = metrics["tw_status"]
            if known_zero:
                row["status"] = "zero_output_discovered"
            elif row["output_verification"].startswith("exact_nonzero"):
                row["status"] = "ok"
            else:
                row["status"] = "ok_unverified_output"
            row["stage"] = "complete"
    except ExperimentTimeout as exc:
        row.update(status="timeout", note=str(exc))
    finally:
        row["seconds"] = round(time.perf_counter() - start, 6)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", required=True, choices=["F0", "F1", "F2", "F3", "F4"])
    ap.add_argument("--sizes", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--order", choices=ORDERS)
    ap.add_argument("--orders", nargs="+", choices=ORDERS)
    ap.add_argument("--width-exponent", type=float, default=1.0)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--ls-rounds", type=int, default=5)
    ap.add_argument("--ls-sample", type=int, default=200)
    ap.add_argument("--timeout", type=float, default=120.0, help="wall seconds per row; 0 disables")
    ap.add_argument("--max-tries", type=int, default=20)
    ap.add_argument("--verify-max-r", type=int, default=16)
    ap.add_argument("--tw-max-nodes", type=int, default=128)
    ap.add_argument("--max-qubits", type=int)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.order and args.orders:
        ap.error("choose --order or --orders, not both")
    orders = args.orders or [args.order or "min_degree"]
    if args.seeds < 1 or any(t < 1 for t in args.sizes):
        ap.error("seeds and sizes must be positive")
    count = 0
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for size in args.sizes:
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                for order in orders:
                    row = run_one(args.family, size, seed, order, args.ls_rounds,
                                  max_tries=args.max_tries, width_exponent=args.width_exponent,
                                  depth=args.depth, ls_sample=args.ls_sample, timeout=args.timeout,
                                  tw_max_nodes=args.tw_max_nodes, verify_max_r=args.verify_max_r,
                                  max_qubits=args.max_qubits)
                    writer.writerow(row)
                    fh.flush()
                    count += 1
                    print(f"{args.family} size={size} seed={seed} order={order}: {row['status']} ({row['seconds']:.3f}s)", flush=True)
    print(f"wrote {count} rows to {args.out}")


if __name__ == "__main__":
    main()
