"""Representation-size experiments for the proven depth-five obstruction.

We instantiate its exact contracted form, not an exponentially large state
vector. Counts N and s describe the explicit physical construction. Local
search is heuristic; the all-affine lower bound is supplied by the proof.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import random
import time

from .basis import local_search
from .obstructions import effective_graph_form


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256, 512])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--ls-rounds", type=int, default=3)
    parser.add_argument("--ls-sample", type=int, default=100)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    fields = ["family", "t", "seed", "logical_edges", "n", "s", "ratio_s_t",
              "physical_degree_upper", "depth_upper", "natural_L", "natural_support",
              "natural_q_size", "ls_L", "ls_support", "ls_q_size", "ls_r",
              "verification", "ls_rounds_budget", "ls_sample_budget", "seconds"]
    with open(args.out, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for t in args.sizes:
            for seed in range(args.seeds):
                start = time.perf_counter()
                rng = random.Random((t << 32) ^ seed ^ 0xD335E)
                edges = [e for e in itertools.combinations(range(t), 2) if rng.randrange(2)]
                pf = effective_graph_form(t, edges)
                out = local_search(pf, max_rounds=args.ls_rounds, sample=args.ls_sample,
                                   rng=random.Random(seed ^ (t << 16)))
                if t <= 12:
                    value = pf.ps.evaluate_exact()
                    assert out.ps.evaluate_exact() == value
                    verification = "exact_value_and_nonzero" if any(value[1]) else "exact_value_zero"
                else:
                    verification = "algebraic_identity_only_not_enumerated"
                row = dict(family="depth5_obstruction", t=t, seed=seed, logical_edges=len(edges),
                           n=t+4*len(edges), s=3*t+13*len(edges),
                           ratio_s_t=(3*t+13*len(edges))/t,
                           physical_degree_upper=3, depth_upper=5,
                           natural_L=pf.L, natural_support=pf.support_sum, natural_q_size=pf.q_size,
                           ls_L=out.L, ls_support=out.support_sum, ls_q_size=out.q_size, ls_r=out.r,
                           verification=verification, ls_rounds_budget=args.ls_rounds,
                           ls_sample_budget=args.ls_sample,
                           seconds=round(time.perf_counter()-start, 6))
                writer.writerow(row)
                stream.flush()
            print(f"obstruction t={t}: {args.seeds} instances complete", flush=True)


if __name__ == "__main__":
    main()
