import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pfsparse import circuits as C  # noqa: E402


def impl(fn, *args, **kwargs):
    """Required functions are implemented; a future stub is a test failure."""
    return fn(*args, **kwargs)


def small_circuits(count=40, seed=7):
    """Small instances (<= 5 qubits) from every family, for brute-force comparison."""
    rng = random.Random(seed)
    out = []
    for i in range(count):
        kind = i % 5
        if kind == 0:
            n = rng.randint(1, 4)
            out.append((C.random_clifford_t(n, rng.randint(1, 16), rng.randint(0, 5), rng), n))
        elif kind == 1:
            n = rng.randint(1, 4)
            out.append((C.phase_polynomial_circuit(n, rng.randint(0, 4), rng.randint(0, 8), rng), n))
        elif kind == 2:
            n = rng.randint(2, 5)
            out.append((C.brickwork_1d(n, rng.randint(1, 3), rng.randint(0, 4), rng), n))
        elif kind == 3:
            out.append((C.brickwork_2d(2, 2, rng.randint(1, 3), rng.randint(0, 4), rng), 4))
        else:
            n = 4
            out.append((C.qaoa_clifford_t(n, [(0, 1), (1, 2), (2, 3)], 1, rng), n))
    return out
