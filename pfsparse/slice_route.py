"""Optional (DESIGN.md 2.3): the magic-state slice route of Theorem 8.6(i).

Suggested implementation: pip install stim; build V by replacing the i-th T^{+-1} with CNOT(q_i -> n+i);
compute stabilizer generators of V^dagger |y, 0^t>; bring them to a standard form whose X-parts span the
support directions; read off the affine support A, then the fibre at x and its parameterisation
u = B w + b; the phase as a Z4 quadratic function of w follows from products of generators
(P|z> = i^s (-1)^{b.z} |z XOR a>).  The global phase is irrelevant for L.
"""
from __future__ import annotations


def parity_form_via_slice(circ, n: int, x: int, y: int):
    raise NotImplementedError("optional: DESIGN.md 2.3")
