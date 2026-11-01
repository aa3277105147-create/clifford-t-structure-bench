import random

import numpy as np

from conftest import impl, small_circuits
from pfsparse.pathsum import build_pathsum
from pfsparse.reference import statevector


def test_build_pathsum_matches_statevector():
    rng = random.Random(1)
    for circ, n in small_circuits():
        x = rng.randrange(1 << n)
        psi = statevector(circ, n, x)
        for y in rng.sample(range(1 << n), min(4, 1 << n)):
            ps = impl(build_pathsum, circ, n, x, y)
            assert np.isclose(ps.evaluate_bruteforce(), psi[y], atol=1e-9), (circ, x, y)
            assert all(p.mask >> ps.nvars == 0 for p in ps.parities)
