import random

import numpy as np

from conftest import impl, small_circuits
from pfsparse import reduce as R
from pfsparse.pathsum import build_pathsum
from pfsparse.reference import statevector


def _instances(seed=2):
    rng = random.Random(seed)
    for circ, n in small_circuits(seed=seed):
        x = rng.randrange(1 << n)
        psi = statevector(circ, n, x)
        y = int(np.argmax(np.abs(psi))) if rng.random() < 0.7 else rng.randrange(1 << n)
        yield circ, n, x, y, psi[y]


def test_single_steps_preserve_value():
    for circ, n, x, y, ref in _instances():
        ps = impl(build_pathsum, circ, n, x, y)
        # one R1 step
        if ps.constraints:
            q = ps.copy()
            impl(R.apply_constraint, q, 0)
            assert np.isclose(q.evaluate_bruteforce(), ref, atol=1e-9)
        # one R2 step on some eliminable variable (after clearing constraints)
        q = ps.copy()
        while q.constraints and not q.zero:
            impl(R.apply_constraint, q, 0)
        free = [j for j in q.alive if not q.in_parity(j) and not q.in_constraint(j)]
        if free and not q.zero:
            impl(R.gauss_eliminate, q, free[0])
            assert np.isclose(q.evaluate_bruteforce(), ref, atol=1e-9)


def test_reduce_gives_valid_parity_form():
    for order in ("natural", "min_degree", "min_fill"):
        for circ, n, x, y, ref in _instances(seed=3):
            ps = impl(build_pathsum, circ, n, x, y)
            pf = impl(R.reduce_pathsum, ps, order=order)
            assert np.isclose(pf.value(), ref, atol=1e-9)
            if not pf.ps.zero:
                assert not pf.ps.constraints
                assert all(pf.ps.in_parity(j) for j in pf.ps.alive)


def test_compress_to_rank():
    for circ, n, x, y, ref in _instances(seed=4):
        pf = impl(R.reduce_pathsum, impl(build_pathsum, circ, n, x, y))
        if pf.ps.zero:
            continue
        pf2 = impl(R.compress_to_rank, pf)
        assert np.isclose(pf2.value(), ref, atol=1e-9)
        assert pf2.r <= max(pf2.t, 0) or pf2.ps.zero
