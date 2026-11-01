import random

import numpy as np

from conftest import impl, small_circuits
from pfsparse import basis as Bs
from pfsparse import reduce as R
from pfsparse.pathsum import build_pathsum
from pfsparse.reference import statevector


def test_xor_change_and_local_search():
    rng = random.Random(5)
    for circ, n in small_circuits(seed=5):
        x = rng.randrange(1 << n)
        psi = statevector(circ, n, x)
        y = int(np.argmax(np.abs(psi)))
        pf = impl(R.reduce_pathsum, impl(build_pathsum, circ, n, x, y))
        if pf.ps.zero or pf.r < 2:
            continue
        j, k = rng.sample(sorted(pf.ps.alive), 2)
        pf2 = impl(Bs.xor_change, pf, j, k)
        assert np.isclose(pf2.value(), psi[y], atol=1e-9)
        pf3 = impl(Bs.local_search, pf, max_rounds=5, rng=rng)
        assert pf3.L <= pf.L
        assert np.isclose(pf3.value(), psi[y], atol=1e-9)
