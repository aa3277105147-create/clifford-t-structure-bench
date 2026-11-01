"""Construct exact effective forms for the positive long-path family.

This table applies the proved path identity. It is explicitly NOT a record of
full-state-vector simulations of the large physical circuits. Independent
small-circuit tests of contraction are in tests/test_graph_contraction.py.
"""
from __future__ import annotations

import argparse
import csv
import random
import time

from .circuits import random_regular_graph_edges
from .obstructions import effective_graph_form


def connected_cubic_edges(t, rng):
    for attempt in range(1, 101):
        edges = random_regular_graph_edges(t, 3, rng)
        neighbours = [set() for _ in range(t)]
        for u, v in edges:
            neighbours[u].add(v)
            neighbours[v].add(u)
        seen, todo = {0}, [0]
        while todo:
            v = todo.pop()
            for u in neighbours[v]-seen:
                seen.add(u)
                todo.append(u)
        if len(seen) == t:
            return edges, attempt
    raise RuntimeError("connected cubic graph rejection sampler exhausted")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    fields = ["family", "t", "seed", "ell", "path_regime", "root_edges", "n", "s",
              "ratio_s_t", "physical_degree", "depth_upper", "boundary_size",
              "L", "support_sum", "q_size", "uniform_L_upper", "half_log2",
              "root_graph_attempts", "evaluation_method", "nonzero_verification", "seconds"]
    with open(args.out, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for t in args.sizes:
            if t < 4 or t % 2:
                raise ValueError("cubic examples require even t >= 4")
            for seed in range(args.seeds):
                edges, attempts = connected_cubic_edges(t, random.Random((t << 32) ^ seed ^ 0xB0A0D))
                m = len(edges)
                for ell, regime in [(1, "fixed_ell_1"), (t, "ell_equals_t")]:
                    start = time.perf_counter()
                    pf = effective_graph_form(t, edges)
                    # Same root sum as the obstruction, different exact scalar.
                    pf.ps.half_log2 = -2*t-2*ell*m
                    if t <= 8:
                        _, coeffs = pf.ps.evaluate_exact()
                        nonzero = "exact_nonzero" if any(coeffs) else "exact_zero"
                    else:
                        nonzero = "not_enumerated"
                    writer.writerow(dict(family="positive_even_paths", t=t, seed=seed,
                                         ell=ell, path_regime=regime, root_edges=m,
                                         n=t+2*ell*m, s=3*t+(6*ell+1)*m,
                                         ratio_s_t=(3*t+(6*ell+1)*m)/t,
                                         physical_degree=3, depth_upper=8, boundary_size=2,
                                         L=pf.L, support_sum=pf.support_sum, q_size=pf.q_size,
                                         uniform_L_upper=4*t+m, half_log2=pf.ps.half_log2,
                                         root_graph_attempts=attempts,
                                         evaluation_method="constructed_effective_form_from_proved_identity",
                                         nonzero_verification=nonzero,
                                         seconds=round(time.perf_counter()-start, 6)))
            stream.flush()
            print(f"positive paths t={t}: fixed ell and ell=t constructed", flush=True)


if __name__ == "__main__":
    main()
