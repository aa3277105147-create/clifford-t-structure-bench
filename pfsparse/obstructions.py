"""Explicit depth-five, degree-three obstruction to affine PF sparsification.

The theorem is about affine reparameterizations retaining the injected T
factors, not a lower bound on the time required to evaluate a scalar amplitude.
See docs/THEORY.md for the counting proof and precise scope.
"""
from __future__ import annotations

from dataclasses import dataclass

from .affine import Affine
from .parityform import ParityForm
from .pathsum import PathSum, _add_four_product


def _simple_edges(t: int, edges) -> tuple[tuple[int, int], ...]:
    if not isinstance(t, int) or t < 1:
        raise ValueError("t must be a positive integer")
    out = set()
    for edge in edges:
        if len(edge) != 2:
            raise ValueError("an edge must have two endpoints")
        u, v = edge
        if (not isinstance(u, int) or not isinstance(v, int)
                or not 0 <= u < t or not 0 <= v < t or u == v):
            raise ValueError("edges must be loop-free on vertices 0,...,t-1")
        e = (min(u, v), max(u, v))
        if e in out:
            raise ValueError("duplicate edge in a simple graph")
        out.add(e)
    return tuple(sorted(out))


@dataclass(frozen=True)
class ObstructionCircuit:
    t: int
    logical_edges: tuple[tuple[int, int], ...]
    n: int
    roots: tuple[int, ...]
    layers: tuple[tuple[tuple[str, tuple[int, ...]], ...], ...]

    @property
    def circuit(self):
        return [gate for layer in self.layers for gate in layer]

    @property
    def s(self):
        return sum(map(len, self.layers))

    @property
    def physical_edges(self):
        return tuple(qs for layer in self.layers for gate, qs in layer if gate == "CZ")

    def root_output(self, a: int = 0) -> int:
        if not isinstance(a, int) or not 0 <= a < (1 << self.t):
            raise ValueError("root output must have t bits")
        return sum(((a >> i) & 1) << q for i, q in enumerate(self.roots))


def bounded_degree_graph_circuit(t: int, edges) -> ObstructionCircuit:
    """Construct C_G with N=t+4m, s=3t+13m, degree<=3 and depth<=5.

    The t roots are qubits 0,...,t-1. Every incidence receives a copy, and
    every consecutive pair along a root/copy chain receives a Fourier
    equality auxiliary. The explicit layer list is part of the certificate.
    """
    edges = _simple_edges(t, edges)
    incidence = [[] for _ in range(t)]
    next_q = t
    copies = {}
    for e in edges:
        for i in e:
            copies[(i, e)] = next_q
            incidence[i].append(next_q)
            next_q += 1
    first, second = [], []
    for i, chain in enumerate(incidence):
        previous = i
        for copy in chain:
            auxiliary = next_q
            next_q += 1
            first.append(("CZ", (previous, auxiliary)))
            second.append(("CZ", (auxiliary, copy)))
            previous = copy
    cross = [("CZ", (copies[(u, e)], copies[(v, e)])) for e in edges for u, v in [e]]
    h_layer = tuple(("H", (q,)) for q in range(next_q))
    fourth = tuple(cross + [("T", (i,)) for i in range(t)])
    return ObstructionCircuit(t, edges, next_q, tuple(range(t)),
                              (h_layer, tuple(first), tuple(second), fourth, h_layer))


def effective_graph_form(t: int, edges, root_output: int = 0) -> ParityForm:
    """Exact reduced amplitude of C_G on output (root_output,0,...,0)."""
    edges = _simple_edges(t, edges)
    if not isinstance(root_output, int) or not 0 <= root_output < (1 << t):
        raise ValueError("root output must have t bits")
    ps = PathSum(nvars=t, alive=set(range(t)), half_log2=-2*t-4*len(edges))
    ps.parities = [Affine.var(i) for i in range(t)]
    ps.b = set(edges)
    ps.a = {i: 2 for i in range(t) if root_output & (1 << i)}
    return ParityForm.from_pathsum(ps)


def _rank(rows: list[int]) -> int:
    pivots = {}
    for row in rows:
        while row:
            lead = row.bit_length()-1
            if lead not in pivots:
                pivots[lead] = row
                break
            row ^= pivots[lead]
    return len(pivots)


def affine_graph_form(t: int, edges, rows: list[int], offset: int = 0,
                      root_output: int = 0) -> ParityForm:
    """Reparameterize x_i = rows[i] dot w XOR offset_i exactly.

    The returned size is 2t+wt(A)+mon(q_G(Aw+b)+a dot (Aw+b)).
    Constant phase is carried separately, including the affine output phase.
    """
    edges = _simple_edges(t, edges)
    if (len(rows) != t or any(not isinstance(row, int) or not 0 <= row < (1 << t)
                              for row in rows) or _rank(rows) != t):
        raise ValueError("rows must be an invertible t by t binary matrix")
    if (not isinstance(offset, int) or not 0 <= offset < (1 << t)
            or not isinstance(root_output, int) or not 0 <= root_output < (1 << t)):
        raise ValueError("offset and root_output must have t bits")
    ps = PathSum(nvars=t, alive=set(range(t)), half_log2=-2*t-4*len(edges))
    parities = [Affine(row, (offset >> i) & 1) for i, row in enumerate(rows)]
    ps.parities = parities
    for u, v in edges:
        _add_four_product(ps, parities[u], parities[v])
    for i in range(t):
        if root_output & (1 << i):
            ps.add_c(4*parities[i].const)
            for j in parities[i].support():
                ps.add_a(j, 2)
    return ParityForm.from_pathsum(ps)
