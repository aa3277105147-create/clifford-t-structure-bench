"""Exact regression checks for normalization, elimination and symbolic outputs."""
import random

import numpy as np

from pfsparse.affine import Affine
from pfsparse.pathsum import PathSum, build_pathsum, symbolic_output_pathsum
from pfsparse.reduce import (apply_constraint, find_support_output, gauss_eliminate,
                             reduce_pathsum, reduce_raw)
from pfsparse.reference import statevector
from pfsparse.circuits import random_clifford_t


def _random_sum(rng, variables=6):
    ps = PathSum()
    for _ in range(variables):
        ps.new_var()
    ps.c = rng.randrange(8)
    ps.half_log2 = rng.randrange(-8, 5)
    for j in ps.alive:
        ps.add_a(j, rng.randrange(4))
    for j in ps.alive:
        for k in ps.alive:
            if j < k and rng.randrange(2):
                ps.toggle_b(j, k)
    ps.parities = [Affine(rng.randrange(1 << variables), rng.randrange(2))
                   for _ in range(rng.randrange(5))]
    return ps


def test_r1_exact_cyclotomic_coefficients():
    rng = random.Random(1001)
    for _ in range(100):
        ps = _random_sum(rng)
        ps.constraints = [Affine(rng.randrange(1 << 6), rng.randrange(2))
                          for _ in range(rng.randrange(1, 5))]
        expected = ps.evaluate_exact()
        apply_constraint(ps, 0)
        assert ps.evaluate_exact() == expected


def test_r2_all_four_coefficients_exact():
    rng = random.Random(1002)
    for coefficient in range(4):
        for _ in range(30):
            ps = _random_sum(rng)
            ps.parities = [Affine(p.mask & ~1, p.const) for p in ps.parities]
            ps.a.pop(0, None)
            ps.add_a(0, coefficient)
            expected = ps.evaluate_exact()
            gauss_eliminate(ps, 0)
            assert ps.evaluate_exact() == expected


def test_random_circuits_all_orders_exact():
    rng = random.Random(1003)
    for _ in range(45):
        n = rng.randrange(1, 5)
        circ = random_clifford_t(n, rng.randrange(1, 20), rng.randrange(6), rng)
        x, y = rng.randrange(1 << n), rng.randrange(1 << n)
        ps = build_pathsum(circ, n, x, y)
        exact = ps.evaluate_exact()
        for order in ("natural", "min_degree", "min_fill"):
            pf = reduce_pathsum(ps, order)
            assert pf.ps.evaluate_exact() == exact
            assert np.isclose(pf.value(), statevector(circ, n, x)[y], atol=1e-10)


def test_symbolic_reduction_preserves_each_parameter_fibre():
    rng = random.Random(1004)
    for _ in range(12):
        n = rng.randrange(1, 4)
        circ = random_clifford_t(n, 12, 3, rng)
        x = rng.randrange(1 << n)
        symbolic, outputs = symbolic_output_pathsum(circ, n, x)
        reduced = reduce_raw(symbolic, allow_parameters=True)
        for y in range(1 << n):
            fibre = reduced.copy()
            for q, j in enumerate(outputs):
                fibre.alive.add(j)
                fibre.constraints.append(Affine.var(j) ^ Affine.constant((y >> q) & 1))
            value = reduce_pathsum(fibre).ps.evaluate_exact()
            assert value == build_pathsum(circ, n, x, y).evaluate_exact()


def test_symbolic_candidate_is_not_nonzero_guarantee():
    # H T^4 H = X. All formal output parameters pass the Clifford-only
    # conditions, but the unmeasured T terms cancel exactly at output zero.
    circ = [("H", (0,))] + [("T", (0,))] * 4 + [("H", (0,))]
    candidates = {find_support_output(circ, 1, 0, random.Random(seed))
                  for seed in range(16)}
    assert candidates == {0, 1}
    assert build_pathsum(circ, 1, 0, 0).evaluate_exact() == (0, (0, 0, 0, 0))
    assert build_pathsum(circ, 1, 0, 1).evaluate_exact() == (0, (1, 0, 0, 0))


def test_gaussian_fill_can_increase_L():
    degree = 6
    ps = PathSum()
    for _ in range(degree + 1):
        ps.new_var()
    ps.add_a(0, 1)
    for j in range(1, degree + 1):
        ps.toggle_b(0, j)
        ps.parities.append(Affine.var(j))
    initial_L = len(ps.alive) + len(ps.parities) + degree + len(ps.a) + len(ps.b)
    expected = ps.evaluate_exact()
    result = reduce_pathsum(ps)
    assert initial_L == 26
    assert result.L == 39
    assert result.ps.evaluate_exact() == expected
