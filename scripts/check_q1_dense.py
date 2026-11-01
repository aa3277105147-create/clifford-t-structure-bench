"""Exact finite checks for dense formal queries and bounded cancellation.

This script does not implement the counting sparsification algorithm. It
checks the polynomial/mass identities used to transfer its disjoint leaves,
and the independent even-moment argument, by finite integer enumeration.
Run from the project root: python scripts/check_q1_dense.py
"""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from itertools import combinations, product
import json
from math import comb
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pfsparse.q1_bounds import quadratic_row  # noqa: E402


def satisfies(clauses, assignment):
    """Evaluate signed, one-based CNF literals on a bit-packed assignment."""
    return all(any(bool(assignment >> (abs(lit) - 1) & 1) == (lit > 0)
                   for lit in clause) for clause in clauses)


def formal_count(clauses, n, t):
    """Integer coefficients of W_F; the first t variables are slots."""
    if not 0 <= t <= n:
        raise ValueError("Require 0 <= slot count <= total variable count")
    if any(len(c) > 3 or any(not isinstance(v, int) or not 1 <= abs(v) <= n
                            for v in c) for c in clauses):
        raise ValueError("Expected clauses of width at most three")
    result = [0] * (1 << t)
    for assignment in range(1 << n):
        if satisfies(clauses, assignment):
            result[assignment & ((1 << t) - 1)] += 1
    return tuple(result)


def coordinate_refinement(clauses, variables):
    """A concrete disjoint refinement by unit clauses, not sparsification."""
    return tuple(tuple(clauses) + tuple((v if bit else -v,)
                                       for v, bit in zip(variables, bits))
                 for bits in product((0, 1), repeat=len(variables)))


def coefficient_identity_and_mass(counts, coefficients):
    target = tuple(sum(a * n[x] for a, n in zip(coefficients, counts))
                   for x in range(len(counts[0])))
    mass = tuple(sum(abs(a) * n[x] for a, n in zip(coefficients, counts))
                 for x in range(len(counts[0])))
    return target, mass


def refinement_check(random_cases=128):
    rng = random.Random(179)
    nontrivial_multiplicity = overlapping_projections = 0
    for _ in range(random_cases):
        n = rng.randint(3, 6)
        t = rng.randint(1, min(4, n - 1))
        clauses = []
        for _ in range(rng.randint(0, 2 * n)):
            variables = rng.sample(range(1, n + 1), rng.randint(1, min(n, 3)))
            clauses.append(tuple(v if rng.getrandbits(1) else -v for v in variables))
        clauses = tuple(clauses)
        variables = tuple(range(t + 1, min(n, t + 2) + 1))
        leaves = coordinate_refinement(clauses, variables)
        parent = formal_count(clauses, n, t)
        counts = [formal_count(f, n, t) for f in leaves]
        assert tuple(sum(c[x] for c in counts) for x in range(1 << t)) == parent
        for assignment in range(1 << n):
            assert sum(satisfies(f, assignment) for f in leaves) == satisfies(clauses, assignment)
        nontrivial_multiplicity += max(parent) > 1
        overlapping_projections += any(sum(c[x] > 0 for c in counts) > 1
                                       for x in range(1 << t))

    # A complete cancellation-free two-sign identity with four extensions
    # per slot assignment. Refinement leaves overlap after projection.
    t, n = 2, 4
    plus, minus = ((-1, -2),), ((1,), (2,))
    coefficients = (Fraction(1, 4), Fraction(-1, 4))
    parents = [formal_count(f, n, t) for f in (plus, minus)]
    target, mass = coefficient_identity_and_mass(parents, coefficients)
    assert target == (1, 1, 1, -1) and mass == (1, 1, 1, 1)
    leaves, alphas = [], []
    for f, alpha in zip((plus, minus), coefficients):
        for leaf in coordinate_refinement(f, (3, 4)):
            leaves.append(formal_count(leaf, n, t))
            alphas.append(alpha)
    assert coefficient_identity_and_mass(leaves, alphas) == (target, mass)
    assert all(sum(v[x] > 0 for v in leaves) == 4 for x in range(4))

    # Disjoint full assignments can project to overlapping supports.
    unrestricted = [formal_count(f, 2, 1)
                    for f in coordinate_refinement((), (2,))]
    assert unrestricted == [(1, 1), (1, 1)]
    # Under a unique extension y=x, this particular projection is disjoint.
    unique = ((-1, 2), (1, -2))
    projected = [formal_count(f, 2, 1)
                 for f in coordinate_refinement(unique, (2,))]
    assert projected == [(1, 0), (0, 1)]
    return {"random_cnf_refinements": random_cases,
            "parents_with_multiple_extensions": nontrivial_multiplicity,
            "refinements_with_overlapping_projections": overlapping_projections,
            "two_sign_identity_leaves": len(leaves),
            "unique_extension_projection_checked": True,
            "all_pass": True,
            "scope": "Coordinate partitions test the transfer identities; no generic sparsification implementation."}


def bounded_cancellation_check():
    plus, minus = ((-1, -2),), ((1,), (2,))
    counts = [formal_count(f, 2, 2) for f in ((), plus, minus)]
    eps = Fraction(1, 8)
    coefficients = (eps, 1 - eps, -1 - eps)
    target, mass = coefficient_identity_and_mass(counts, coefficients)
    assert target == (1, 1, 1, -1)
    assert mass == (1, 1, 1, Fraction(5, 4))
    assert counts[0] == (1, 1, 1, 1)  # mixed-sign support
    # All exact count functions are nonnegative; the cancellation is only
    # between queries, not in their auxiliary variables.
    return {"epsilon": str(eps), "maximum_mass": str(max(mass)),
            "query_support_contains_both_signs": True,
            "all_pass": True,
            "scope": "An illustrative identity, not an asymptotic upper bound."}


def flower_children(clauses, heart, petals):
    """One genuine flower OR split, without false-heart guards yet."""
    heart_variables = {abs(v) for v in heart}
    assert heart and len(heart_variables) == len(heart)
    used = set(heart_variables)
    for petal in petals:
        variables = {abs(v) for v in petal}
        assert petal and not used.intersection(variables)
        used.update(variables)
    remainder = list(clauses)
    for petal in petals:
        clause = tuple(heart) + tuple(petal)
        assert len(clause) <= 3
        remainder.remove(clause)
    return (tuple(remainder) + (tuple(heart),),
            tuple(remainder) + tuple(tuple(p) for p in petals))


def flower_guard_check():
    """Post-hoc guards on fixed one-/two-level flower trees, exhaustively.

    This checks the disjointization lemma's local mechanics, not the CIP
    leaf-count or occurrence theorem and not a complete sparsifier.
    """
    families = []
    for heart_size in (1, 2):
        for bits in product((-1, 1), repeat=heart_size + 2):
            heart = tuple(bits[i] * (i + 1) for i in range(heart_size))
            petals = ((bits[-2] * (heart_size + 1),),
                      (bits[-1] * (heart_size + 2),))
            formula = tuple(heart + petal for petal in petals)
            families.append((heart_size + 3, 2, formula, ((heart, petals),)))
    families.extend([
        # Two separate hearts, with one unused declared variable.
        (7, 2, ((1, 3), (1, 4), (2, 5), (2, 6)),
         (((1,), ((3,), (4,))), ((2,), ((5,), (6,))))),
        # Repeated false-heart guards must be deduplicated.
        (5, 2, ((1, 2), (1, 3), (1, 4), (1, 5)),
         (((1,), ((2,), (3,))), ((1,), ((4,), (5,))))),
        # Opposite false-heart guards force an empty leaf.
        (5, 2, ((1, 2), (1, 3), (-1, 4), (-1, 5)),
         (((1,), ((2,), (3,))), ((-1,), ((4,), (5,))))),
        # A heart on an auxiliary variable yields overlapping projections
        # even after the full assignments are partitioned.
        (3, 2, ((3, 1), (3, 2)), (((3,), ((1,), (2,))),)),
    ])
    raw_overlap_cases = projected_overlap_cases = duplicate_guard_paths = 0
    contradictory_guard_leaves = opposite_unit_leaves = 0
    raw_leaves_total = guarded_leaves_total = assignments_checked = 0
    for n, t, formula, steps in families:
        tree = [(formula, ())]
        for heart, petals in steps:
            expanded = []
            for current, guards in tree:
                left, right = flower_children(current, heart, petals)
                for assignment in range(1 << n):
                    parent_ok = satisfies(current, assignment)
                    left_ok, right_ok = satisfies(left, assignment), satisfies(right, assignment)
                    assert not left_ok or parent_ok
                    assert not right_ok or parent_ok
                    assert parent_ok == (left_ok or right_ok)
                expanded.extend([(left, guards),
                                 (right, guards + tuple(-v for v in heart))])
            tree = expanded
        raw_leaves_total += len(tree)
        raw_overlap_cases += any(sum(satisfies(f, a) for f, _ in tree) > 1
                                 for a in range(1 << n))
        guarded = []
        for leaf, guards in tree:
            duplicate_guard_paths += len(guards) != len(set(guards))
            guard_set = set(guards)
            if any(-v in guard_set for v in guard_set):
                contradictory_guard_leaves += 1
                continue
            assert len(guard_set) <= n
            assert len({abs(v) for v in guard_set}) == len(guard_set)
            original_units = {c[0] for c in leaf if len(c) == 1}
            all_units = original_units | guard_set
            if any(-v in all_units for v in all_units):
                opposite_unit_leaves += 1
                continue
            units_to_add = guard_set - original_units
            guarded.append(tuple(leaf) + tuple((v,) for v in sorted(units_to_add)))
            # At most one new occurrence per variable is appended.
            for variable in range(1, n + 1):
                assert sum(abs(v) == variable for v in units_to_add) <= 1
        guarded_leaves_total += len(guarded)
        for assignment in range(1 << n):
            assert sum(satisfies(f, assignment) for f in guarded) == satisfies(formula, assignment)
            assignments_checked += 1
        parent_count = formal_count(formula, n, t)
        leaf_counts = [formal_count(f, n, t) for f in guarded]
        assert tuple(sum(c[x] for c in leaf_counts) for x in range(1 << t)) == parent_count
        projected_overlap_cases += any(sum(c[x] > 0 for c in leaf_counts) > 1
                                       for x in range(1 << t))
        for alpha in (Fraction(3, 7), Fraction(-5, 11)):
            target, mass = coefficient_identity_and_mass(leaf_counts, [alpha] * len(leaf_counts))
            assert target == tuple(alpha * v for v in parent_count)
            assert mass == tuple(abs(alpha) * v for v in parent_count)
    assert raw_overlap_cases > 0 and projected_overlap_cases > 0
    assert duplicate_guard_paths > 0 and contradictory_guard_leaves > 0
    return {"flower_families": len(families), "maximum_tree_depth": 2,
            "raw_leaf_count": raw_leaves_total,
            "guarded_leaf_count_after_unit_conflict_removal": guarded_leaves_total,
            "full_assignments_checked": assignments_checked,
            "raw_covers_with_overlaps": raw_overlap_cases,
            "partitions_with_overlapping_projections": projected_overlap_cases,
            "paths_with_duplicate_guards": duplicate_guard_paths,
            "leaves_with_opposite_guards_removed": contradictory_guard_leaves,
            "other_opposite_unit_leaves_removed": opposite_unit_leaves,
            "coefficient_count_and_absolute_mass_preserved": True,
            "at_most_one_added_occurrence_per_variable": True,
            "all_pass": True,
            "scope": "Exact one-/two-level flower trees check post-hoc disjointization only; CIP asymptotic tree bounds remain a cited theorem."}


def graph_signs(t):
    rows = [quadratic_row(x, t) for x in range(1 << t)]
    return [tuple(1 - 2 * ((g & row).bit_count() & 1) for row in rows)
            for g in range(1 << comb(t, 2))]


def level_set_check(max_t=6):
    records = []
    for t in range(2, max_t + 1):
        signs = graph_signs(t)
        one_sizes = [sum(s == -1 for s in row) for row in signs[1:]]
        zero_sizes = [(1 << t) - k for k in one_sizes]
        assert min(one_sizes) >= 1 << (t - 2)
        assert min(zero_sizes) >= 1 << (t - 2)
        records.append({"t": t, "nonzero_graphs": len(one_sizes),
                        "minimum_one_level_size": min(one_sizes),
                        "minimum_zero_level_size": min(zero_sizes),
                        "lower_bound": 1 << (t - 2), "all_pass": True})
    return {"records": records,
            "nonzero_graphs": sum(r["nonzero_graphs"] for r in records),
            "all_pass": True}


def moment_domination_check():
    indicator_comparisons = rational_comparisons = 0
    for t in range(4):
        signs = graph_signs(t)
        reference = [sum(row) for row in signs]
        for support in range(1 << (1 << t)):
            values = [sum(s for x, s in enumerate(row) if support >> x & 1)
                      for row in signs]
            for h in range(1, 6):
                assert sum(v ** (2 * h) for v in values) <= sum(v ** (2 * h) for v in reference)
                indicator_comparisons += 1
    rng = random.Random(271)
    signs = graph_signs(4)
    reference = [sum(row) for row in signs]
    for _ in range(128):
        denominator = rng.randint(2, 9)
        numerators = [rng.randrange(denominator + 1) for _ in range(16)]
        values = [sum(a * s for a, s in zip(numerators, row)) for row in signs]
        for h in range(1, 7):
            assert sum(v ** (2 * h) for v in values) <= denominator ** (2 * h) * sum(v ** (2 * h) for v in reference)
            rational_comparisons += 1
    return {"indicator_supports": sum(1 << (1 << t) for t in range(4)),
            "indicator_moment_comparisons": indicator_comparisons,
            "rational_weight_functions": 128,
            "rational_moment_comparisons": rational_comparisons,
            "arithmetic": "integer numerators and exact powers; no floating-point moments",
            "all_pass": True}


def character_and_gauss_check(max_t=6):
    t = 3
    rows = [quadratic_row(x, t) for x in range(1 << t)]
    signs = graph_signs(t)
    tuples_checked = 0
    for h in (1, 2):
        surviving = 0
        for xs in product(range(1 << t), repeat=2 * h):
            mask = 0
            for x in xs:
                mask ^= rows[x]
            total = 0
            for sign in signs:
                value = 1
                for x in xs:
                    value *= sign[x]
                total += value
            assert total == (len(signs) if mask == 0 else 0)
            surviving += mask == 0
            tuples_checked += 1
        assert sum(sum(sign) ** (2 * h) for sign in signs) == len(signs) * surviving
    bound_checks = 0
    for t in range(max_t + 1):
        signs = graph_signs(t)
        unweighted = [sum(row) for row in signs]
        for h in range(1, 9):
            total = sum(v ** (2 * h) for v in unweighted)
            assert total <= len(signs) * 17 * (1 << (h * t + h * (h + 1) // 2))
            bound_checks += 1
    return {"character_tuples": tuples_checked,
            "gauss_moment_bound_checks": bound_checks,
            "gauss_max_t": max_t, "moment_orders": list(range(2, 17, 2)),
            "all_pass": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/q1_dense_checks.json")
    args = parser.parse_args()
    start = time.perf_counter()
    original = ROOT / "research_inputs/q1_dense/exact_mt_results.json"
    reproduced = ROOT / "results/q1_dense_reproduction/exact_mt_results.json"
    reproduction = {}
    if original.exists() and reproduced.exists():
        assert json.loads(original.read_text()) == json.loads(reproduced.read_text())
        reproduction = {"result": json.loads(reproduced.read_text()), "equal_json": True,
            "original_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
            "reproduced_sha256": hashlib.sha256(reproduced.read_bytes()).hexdigest()}
    sources = ["scripts/check_q1_dense.py", "tests/test_q1_dense.py",
               "research_inputs/q1_dense/exact_mt.py", "research_inputs/q1_dense/lemma_r.py"]
    result = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "original_script_reproduction": reproduction,
              "refinement": refinement_check(),
              "flower_guards": flower_guard_check(),
              "bounded_cancellation_example": bounded_cancellation_check(),
              "quadratic_level_sets": level_set_check(),
              "nonnegative_moments": moment_domination_check(),
              "character_and_gauss": character_and_gauss_check(),
              "source_sha256": {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in sources},
              "seconds": round(time.perf_counter() - start, 6),
              "scope": [
                  "These are finite exact checks; the asymptotic probability and query bounds are proved in the manuscript.",
                  "The script does not implement general counting sparsification or a generic CNF encoder.",
                  "The cancellation example only shows why phase compatibility fails above mass one."
              ]}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(output), "all_pass": True,
                      "original_checks": reproduction.get("result"),
                      "nonzero_graphs": result["quadratic_level_sets"]["nonzero_graphs"],
                      "moment_comparisons": sum(result["nonnegative_moments"][k] for k in
                         ("indicator_moment_comparisons", "rational_moment_comparisons")),
                      "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
