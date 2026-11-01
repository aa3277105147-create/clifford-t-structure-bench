"""Exact finite checks for Walsh representations by nonnegative counts.

Exhaustively enumerate labelled simple graphs through six vertices by
default. Walsh coefficients are kept as integer numerators with denominator
2**n; no floating-point estimates are used. The separate CNF check verifies
the actual unique-extension XOR encodings of every parity predicate.

This is a finite verification of the representation identities, not a test
of the asymptotic high-probability query--mass lower bound.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pfsparse.q1_bounds import gf2_rank  # noqa: E402


def integer_walsh(values):
    """Unnormalized Walsh transform in the bit-packed character order."""
    result = list(values)
    size = len(result)
    if not size or size & (size - 1):
        raise ValueError("Expected a nonempty power-of-two vector")
    half = 1
    while half < size:
        for start in range(0, size, 2 * half):
            for offset in range(half):
                a, b = result[start + offset], result[start + offset + half]
                result[start + offset] = a + b
                result[start + offset + half] = a - b
        half *= 2
    return tuple(result)


def parity_cnf(n, mask, target):
    """A width-three CNF with unique extensions for mask dot x = target.

    The first n variables are declared slots, including unused slots. XOR
    gates introduce one auxiliary each; the final XOR is fixed directly.
    Thus there are at most max(n, 2*n-2) declared variables.
    """
    if not isinstance(n, int) or n < 0 or not 0 <= mask < 1 << n:
        raise ValueError("Invalid parity instance")
    if target not in (0, 1):
        raise ValueError("The target must be a bit")
    variables = [i + 1 for i in range(n) if mask >> i & 1]
    if not variables:
        return n, () if target == 0 else ((),)
    if len(variables) == 1:
        return n, ((variables[0] if target else -variables[0],),)
    clauses = []
    total = n
    current = variables[0]
    for variable in variables[1:-1]:
        total += 1
        output = total
        # output = current XOR variable, with all four forbidden triples.
        clauses.extend(((current, variable, -output),
                        (current, -variable, output),
                        (-current, variable, output),
                        (-current, -variable, -output)))
        current = output
    last = variables[-1]
    if target:
        clauses.extend(((current, last), (-current, -last)))
    else:
        clauses.extend(((-current, last), (current, -last)))
    return total, tuple(clauses)


def satisfies(clauses, assignment):
    return all(any(bool(assignment >> (abs(literal) - 1) & 1)
                   == (literal > 0) for literal in clause)
               for clause in clauses)


def parity_cnf_check(max_n=6):
    """Enumerate full assignments, checking multiplicities as well as support."""
    if not 0 <= max_n <= 6:
        raise ValueError("This exhaustive check is restricted to n <= 6")
    queries = full_assignments = zero_queries = 0
    maximum_variables = maximum_clauses = 0
    by_n = []
    for n in range(max_n + 1):
        n_queries = n_assignments = 0
        for mask in range(1 << n):
            complementary_counts = []
            for target in (0, 1):
                total, clauses = parity_cnf(n, mask, target)
                assert total <= max(n, 2 * n - 2)
                assert all(len(clause) <= 3 for clause in clauses)
                counts = [0] * (1 << n)
                for assignment in range(1 << total):
                    if satisfies(clauses, assignment):
                        counts[assignment & ((1 << n) - 1)] += 1
                expected = [int(((mask & x).bit_count() & 1) == target)
                            for x in range(1 << n)]
                assert counts == expected
                assert all(value in (0, 1) for value in counts)
                complementary_counts.append(counts)
                zero_queries += not any(counts)
                n_queries += 1
                n_assignments += 1 << total
                maximum_variables = max(maximum_variables, total)
                maximum_clauses = max(maximum_clauses, len(clauses))
            assert all(a + b == 1 for a, b in zip(*complementary_counts))
        queries += n_queries
        full_assignments += n_assignments
        by_n.append({"n": n, "parity_queries": n_queries,
                     "full_assignments": n_assignments})
    return {"all_pass": True, "max_n": max_n, "parity_queries": queries,
            "full_assignments": full_assignments,
            "zero_queries": zero_queries,
            "maximum_declared_variables": maximum_variables,
            "maximum_clause_count": maximum_clauses,
            "unique_extensions_and_complements_checked": True,
            "by_n": by_n}


def walsh_representation_check(max_n=6):
    """Check spectral rank formulas and two positive-query decompositions.

    Pair version: sum_a hat_sigma(a) (P[a,0] - P[a,1]).
    Constant version: 1 - 2 sum_{a != 0} hat_sigma(a) P[a,1].
    Every value and absolute mass is represented with denominator 2**n.
    """
    if not 0 <= max_n <= 6:
        raise ValueError("This exhaustive check is restricted to n <= 6")
    total_graphs = pointwise_checks = coefficient_checks = 0
    direct_transform_checks = missing_constant_graphs = 0
    by_n = []
    examples = {}
    for n in range(max_n + 1):
        size = 1 << n
        edges = tuple(combinations(range(n), 2))
        quadratic_masks = tuple(sum(1 << k for k, (i, j) in enumerate(edges)
                                    if (x >> i) & (x >> j) & 1)
                                for x in range(size))
        parity_ones = tuple(tuple(x for x in range(size)
                                 if (a & x).bit_count() & 1)
                           for a in range(size))
        rank_counts = Counter()
        n_missing_constant = 0
        for graph in range(1 << len(edges)):
            adjacency = [0] * n
            for k, (i, j) in enumerate(edges):
                if graph >> k & 1:
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i
            rank = gf2_rank(adjacency)
            assert rank % 2 == 0
            rank_counts[rank] += 1
            signs = tuple(-1 if (graph & row).bit_count() & 1 else 1
                          for row in quadratic_masks)
            numerators = integer_walsh(signs)
            support = tuple(a for a, value in enumerate(numerators) if value)
            assert len(support) == 1 << rank
            for a in support:
                assert numerators[a] ** 2 == 1 << (2 * n - rank)
                coefficient_checks += 1
            if n <= 4:
                direct = tuple(sum(value * (-1 if (a & x).bit_count() & 1 else 1)
                                   for x, value in enumerate(signs))
                               for a in range(size))
                assert numerators == direct
                direct_transform_checks += 1

            contains_zero = int(numerators[0] != 0)
            if not contains_zero:
                missing_constant_graphs += 1
                n_missing_constant += 1
            # All pure graph quadratics vanish at x=0.
            assert sum(numerators) == size
            pair_mass_numerator = sum(abs(value) for value in numerators)
            assert pair_mass_numerator == size * (1 << (rank // 2))
            pair_listed_queries = 2 * len(support)
            pair_nonzero_queries = pair_listed_queries - contains_zero
            constant_queries = 1 + len(support) - contains_zero
            assert pair_listed_queries == 1 << (rank + 1)
            assert constant_queries <= size

            # P[a,0] + P[a,1] = 1, verified on the actual CNFs separately.
            # Accumulate every nonconstant predicate's exact contribution.
            pair_target = [sum(numerators)] * size
            constant_target = [size] * size
            constant_mass = [size] * size
            for a in support:
                if a == 0:
                    continue
                signed = 2 * numerators[a]
                unsigned = abs(signed)
                for x in parity_ones[a]:
                    pair_target[x] -= signed
                    constant_target[x] -= signed
                    constant_mass[x] += unsigned
            assert pair_target == [size * value for value in signs]
            assert constant_target == pair_target

            kernel = tuple(x for x in range(size)
                           if all((row & x).bit_count() % 2 == 0
                                  for row in adjacency))
            assert len(kernel) == 1 << (n - rank)
            kernel_set = set(kernel)
            for x in kernel:
                q_x = int(signs[x] == -1)
                assert all(((a & x).bit_count() & 1) == q_x for a in support)
            for x, sign in enumerate(signs):
                if x not in kernel_set:
                    expected_mass = size * (1 + (1 << (rank // 2)))
                elif sign == 1:
                    expected_mass = size
                else:
                    expected_mass = size * (1 + (1 << (1 + rank // 2)))
                assert constant_mass[x] == expected_mass
                pointwise_checks += 1
            maximum_mass = max(constant_mass)
            assert maximum_mass <= size * (1 + (1 << (1 + rank // 2)))
            if rank == 0:
                assert constant_queries == pair_nonzero_queries == 1
                assert maximum_mass == pair_mass_numerator == size

            if (n, graph) in ((3, 0), (2, 1), (3, 7)):
                name = {(3, 0): "empty_three_vertices",
                        (2, 1): "one_edge_full_rank",
                        (3, 7): "triangle_affine_fourier_support"}[(n, graph)]
                examples[name] = {
                    "n": n, "rank": rank, "support_size": len(support),
                    "zero_in_fourier_support": bool(contains_zero),
                    "listed_pair_queries": pair_listed_queries,
                    "nonzero_pair_queries": pair_nonzero_queries,
                    "pair_mass": pair_mass_numerator // size,
                    "constant_queries": constant_queries,
                    "constant_maximum_mass": maximum_mass // size,
                }
            total_graphs += 1
        by_n.append({"n": n, "graphs": 1 << len(edges),
                     "rank_counts": dict(sorted(rank_counts.items())),
                     "graphs_without_constant_fourier_coefficient": n_missing_constant})
    return {"all_pass": True, "max_n": max_n,
            "graphs": total_graphs,
            "nonzero_fourier_coefficients_checked": coefficient_checks,
            "pointwise_representation_and_mass_checks": pointwise_checks,
            "direct_transform_cross_checks": direct_transform_checks,
            "graphs_without_constant_fourier_coefficient": missing_constant_graphs,
            "pair_query_count": "2^(r+1) listed; 2^(r+1)-1[0 in support] after removing zero functions",
            "constant_query_count": "1+2^r-1[0 in support] <= 2^n",
            "pair_mass": "Exactly 2^(r/2) at every x",
            "constant_mass": "1+2^(r/2) outside ker(A); 1 on ker(A) with q=0; 1+2^(1+r/2) on ker(A) with q=1",
            "examples": examples, "by_n": by_n}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-n", type=int, default=6)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "interference_checks.json")
    args = parser.parse_args()
    started = time.perf_counter()
    cnf = parity_cnf_check(args.max_n)
    walsh = walsh_representation_check(args.max_n)
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arithmetic": "exact integers throughout; normalized Fourier magnitudes checked by integer squares",
        "parity_cnf": cnf, "walsh_representations": walsh,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "all_pass": cnf["all_pass"] and walsh["all_pass"],
        "scope": "Finite exact representation checks, not an empirical proof of the asymptotic uniform query--mass bound. Walsh representations use exponentially many queries in general.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
