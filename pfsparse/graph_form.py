"""Polynomial-time, slot-preserving weighted graph normal forms.

The circuit entry point treats every physical T/TDG as diag(1,z_i) and retains
all slots, even constant factors. Its graph and normalization are independent
of the choice z_i=omega or omega.conjugate(). The older ParityForm entry point
is formal only in its surviving factors; it cannot recover discarded slots.
Numerical evaluators are explicitly separate from the exact integer oracle.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import cmath

from .basis import _xor_change_raw
from .pathsum import PathSum, build_pathsum
from .reduce import reduce_raw

OMEGA = cmath.exp(1j * cmath.pi / 4)


def bits(mask: int):
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def inverse_binary(rows: list[int]) -> tuple[int, ...]:
    """Invert a square binary matrix represented by row masks, without enumeration."""
    n = len(rows)
    if any(v < 0 or v >> n for v in rows):
        raise ValueError("matrix row exceeds its dimension")
    work = [[row, 1 << i] for i, row in enumerate(rows)]
    for col in range(n):
        pivot = next((i for i in range(col, n) if work[i][0] & (1 << col)), None)
        if pivot is None:
            raise ValueError("singular binary matrix")
        work[col], work[pivot] = work[pivot], work[col]
        for i in range(n):
            if i != col and work[i][0] & (1 << col):
                work[i][0] ^= work[col][0]
                work[i][1] ^= work[col][1]
    return tuple(row[1] for row in work)


def _mul(a, b):
    out = [0] * 4
    for i in range(4):
        for j in range(4):
            k = i + j
            out[k % 4] += a[i] * b[j] * (1 if k < 4 else -1)
    return tuple(out)


def _omega_power(k):
    k %= 8
    a = [0] * 4
    a[k % 4] = 1 if k < 4 else -1
    return tuple(a)


def _canonical(a, half_log2):
    a = list(a)
    if half_log2 & 1:
        a0, a1, a2, a3 = a
        a = [a1 - a3, a0 + a2, a1 + a3, a2 - a0]
    e = half_log2 // 2
    if e >= 0:
        a = [x << e for x in a]
        d = 0
    else:
        d = -e
    if not any(a):
        return (0, (0, 0, 0, 0))
    while d and all(x % 2 == 0 for x in a):
        a = [x // 2 for x in a]
        d -= 1
    return d, tuple(a)


@dataclass(frozen=True)
class GraphLabel:
    kind: str                      # direct or fourier
    phase: int                     # Z4 for direct; F2 for fourier
    slot: int


@dataclass(frozen=True)
class GraphNormalForm:
    labels: tuple[GraphLabel | None, ...]
    edges: frozenset[tuple[int, int]]
    half_log2: int = 0
    phase: int = 0
    default_signs: tuple[int, ...] = ()
    zero: bool = False

    def __post_init__(self):
        n = len(self.labels)
        if any(not 0 <= u < v < n for u, v in self.edges):
            raise ValueError("edges must be canonical pairs of distinct vertices")
        slots = sorted(label.slot for label in self.labels if label is not None)
        if slots != list(range(len(self.default_signs))):
            raise ValueError("every formal slot must occur exactly once")
        if any(s not in (-1, 1) for s in self.default_signs):
            raise ValueError("default slot signs must be +/-1")
        for label in self.labels:
            if label is not None and (label.kind not in ("direct", "fourier")
                    or label.phase not in range(4 if label.kind == "direct" else 2)):
                raise ValueError("invalid graph label")

    @property
    def n(self):
        return len(self.labels)

    @property
    def t(self):
        return len(self.default_signs)

    @property
    def prefactor(self):
        return 0j if self.zero else (2.0 ** (self.half_log2 / 2)) * OMEGA ** self.phase

    @property
    def order(self):
        return tuple(label.slot for label in self.labels if label is not None)

    @property
    def adjacency(self):
        adj = [0] * self.n
        for u, v in self.edges:
            adj[u] |= 1 << v
            adj[v] |= 1 << u
        return tuple(adj)

    def weights(self, slot_values=None):
        values = ([OMEGA ** s for s in self.default_signs]
                  if slot_values is None else list(slot_values))
        if len(values) != self.t:
            raise ValueError("one value is required per original slot")
        out = []
        for label in self.labels:
            if label is None:
                out.append((1, 1))
            elif label.kind == "direct":
                out.append((1, (1j ** label.phase) * values[label.slot]))
            else:
                z = values[label.slot]
                out.append(((1 + z) / 2, (-1) ** label.phase * (1 - z) / 2))
        return tuple(out)

    def value(self, slot_values=None, *, max_vertices=22):
        """Floating-point enumeration for small graphs; not an exact oracle."""
        if self.zero:
            return 0j
        if self.n > max_vertices:
            raise ValueError("graph exceeds enumeration budget")
        w = self.weights(slot_values)
        total = 0j
        for x in range(1 << self.n):
            sign = sum(((x >> i) & 1) * ((x >> j) & 1) for i, j in self.edges) & 1
            term = (-1) ** sign
            for i in range(self.n):
                term *= w[i][(x >> i) & 1]
            total += term
        return self.prefactor * total

    def evaluate_exact(self, signs=None, *, max_vertices=18):
        """Exact Q(omega) oracle with the same canonical tuple as PathSum.

        signs selects T (+1) or TDG (-1) independently at every original slot.
        Both sides of every Fourier label have denominator 2, so all arithmetic
        can be done with four integer coefficients and one final power of two.
        """
        if self.zero:
            return (0, (0, 0, 0, 0))
        if self.n > max_vertices:
            raise ValueError("graph exceeds exact-enumeration budget")
        signs = self.default_signs if signs is None else tuple(signs)
        if len(signs) != self.t or any(s not in (-1, 1) for s in signs):
            raise ValueError("one +/-1 sign is required per slot")
        one = (1, 0, 0, 0)
        w, f = [], 0
        for label in self.labels:
            if label is None:
                w.append((one, one))
            elif label.kind == "direct":
                w.append((one, _omega_power(signs[label.slot] + 2 * label.phase)))
            else:
                z = _omega_power(signs[label.slot])
                lo = tuple(one[j] + z[j] for j in range(4))
                hi = tuple((-1) ** label.phase * (one[j] - z[j]) for j in range(4))
                w.append((lo, hi))
                f += 1
        total = [0] * 4
        for x in range(1 << self.n):
            term = one
            for j in range(self.n):
                term = _mul(term, w[j][(x >> j) & 1])
            sign = (-1) ** (sum(((x >> i) & 1) * ((x >> j) & 1) for i, j in self.edges) & 1)
            for j in range(4):
                total[j] += sign * term[j]
        return _canonical(_mul(total, _omega_power(self.phase)), self.half_log2 - 2 * f)

    def mitm_value(self, slot_values=None, *, max_half=22):
        from .mitm import mitm_value
        if self.zero:
            return 0j
        return self.prefactor * mitm_value(self.n, self.adjacency, self.weights(slot_values), max_half=max_half)


def _zero_form(t, signs):
    return GraphNormalForm(tuple(GraphLabel("direct", 0, j) for j in range(t)),
                           frozenset(), default_signs=tuple(signs), zero=True)


def _compress_slots(ps, order):
    ps = reduce_raw(ps, order=order, preserve_parities=True)
    if ps.zero:
        return ps
    columns = dict.fromkeys(ps.alive, 0)
    for i, p in enumerate(ps.parities):
        for j in bits(p.mask):
            columns[j] |= 1 << i
    pivots = {}
    for k in sorted(ps.alive):
        vector, combination = columns[k], 0
        while vector:
            lead = vector.bit_length() - 1
            if lead not in pivots:
                pivots[lead] = (vector, combination ^ (1 << k))
                break
            other, other_combination = pivots[lead]
            vector ^= other
            combination ^= other_combination
        if not vector:
            for j in bits(combination):
                _xor_change_raw(ps, j, k)
    return reduce_raw(ps, order=order, preserve_parities=True)


def _from_full_rank(ps, signs):
    t = len(signs)
    if ps.zero:
        return _zero_form(t, signs)
    if len(ps.parities) != t or ps.constraints:
        raise ValueError("expected all slots and no remaining constraints")
    vs = sorted(ps.alive)
    r = len(vs)
    pos = {v: k for k, v in enumerate(vs)}
    parities = [(sum(1 << pos[j] for j in p.support()), p.const) for p in ps.parities]
    piv, basis = [], {}
    for i, (mask, _) in enumerate(parities):
        vector = mask
        while vector:
            lead = vector.bit_length() - 1
            if lead not in basis:
                basis[lead] = vector
                piv.append(i)
                break
            vector ^= basis[lead]
    if len(piv) != r:
        raise ValueError("graph conversion requires full column rank; compress first")
    inverse = inverse_binary([parities[i][0] for i in piv])
    shift = sum(parities[i][1] << k for k, i in enumerate(piv))

    def original(x):
        x ^= shift
        return sum((((row & x).bit_count() & 1) << vs[k]) for k, row in enumerate(inverse))

    def phase(x):
        z = original(x)
        q = sum(a * ((z >> j) & 1) for j, a in ps.a.items())
        q += 2 * sum(((z >> j) & 1) * ((z >> k) & 1) for j, k in ps.b)
        return (ps.c + 2 * q) % 8

    constant = phase(0)
    linear = [((phase(1 << j) - constant) % 8) // 2 for j in range(r)]
    edges = set()
    for j, k in combinations(range(r), 2):
        cross = (((phase((1 << j) | (1 << k)) - constant) % 8) // 2 - linear[j] - linear[k]) % 4
        if cross not in (0, 2):
            raise AssertionError("non-Clifford quadratic coefficient")
        if cross:
            edges.add((j, k))
    labels = [GraphLabel("direct", linear[k], i) for k, i in enumerate(piv)]
    piv_set = set(piv)
    for i, (mask, const) in enumerate(parities):
        if i in piv_set:
            continue
        coeff = 0
        for j in bits(mask):
            coeff ^= inverse[j]
        affine_const = const ^ ((coeff & shift).bit_count() & 1)
        node = len(labels)
        labels.append(GraphLabel("fourier", affine_const, i))
        edges.update((j, node) for j in bits(coeff))
    # Restore original slot order. Otherwise a pivot-first layout silently
    # permutes the physical T/TDG slots in downstream circuit realizations.
    relabel = [label.slot for label in labels]
    edges = {tuple(sorted((relabel[u], relabel[v]))) for u, v in edges}
    labels.sort(key=lambda label: label.slot)
    return GraphNormalForm(tuple(labels), frozenset(edges), ps.half_log2, constant, tuple(signs))


def graph_normal_form(pf):
    """Convert a rank-compressed ParityForm, formally in its surviving factors.

    Unlike the uploaded prototype this conversion never enumerates assignments.
    Use circuit_graph_normal_form when original physical slot identities matter.
    """
    return _from_full_rank(pf.ps, (1,) * pf.t)


def circuit_graph_normal_form(circ, n, x=0, y=0, *, order="min_degree"):
    """One common t-vertex graph for every filling of the circuit's T/TDG slots."""
    signs = tuple(1 if gate == "T" else -1 for gate, _ in circ if gate in ("T", "TDG"))
    skeleton = [("T" if gate == "TDG" else gate, qs) for gate, qs in circ]
    ps = build_pathsum(skeleton, n, x, y)
    ps = _compress_slots(ps, order)
    return _from_full_rank(ps, signs)
