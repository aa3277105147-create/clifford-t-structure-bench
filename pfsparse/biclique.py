"""Constructive edge-disjoint biclique compression with exact normalization.

This ordered-block construction is elementary and polynomial time. It is not
an implementation of the density-sensitive optimal STOC 2026 algorithm.
Every used biclique adds two unweighted vertices and one joining edge; report
both incidence weight W and the actual auxiliary graph edge count W+k.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

from .graph_form import GraphNormalForm, bits


def _validate(t, adj):
    if t < 0 or len(adj) != t:
        raise ValueError("wrong adjacency dimension")
    for i, row in enumerate(adj):
        if row < 0 or row >> t or (row >> i) & 1:
            raise ValueError("adjacency must be a loop-free binary graph")
        for j in bits(row):
            if not (adj[j] >> i) & 1:
                raise ValueError("adjacency must be symmetric")


def biclique_partition(t, adj, k):
    """Return an edge partition as pairs of disjoint vertex tuples.

    For every block U, later vertices with the same nonempty neighborhood S in
    U form one biclique (S,class). Internal block edges are partitioned by stars.
    No exponential assignment verification is part of the construction.
    """
    _validate(t, adj)
    if not isinstance(k, int) or k < 1:
        raise ValueError("positive integer block size required")
    result = []
    for start in range(0, t, k):
        end = min(t, start + k)
        mask = ((1 << end) - 1) ^ ((1 << start) - 1)
        for u in range(start, end):
            later = adj[u] & mask & ~((1 << (u + 1)) - 1)
            if later:
                result.append(((u,), tuple(bits(later))))
        classes = {}
        for v in range(end, t):
            pattern = adj[v] & mask
            if pattern:
                classes.setdefault(pattern, []).append(v)
        result.extend((tuple(bits(pattern)), tuple(vertices))
                      for pattern, vertices in sorted(classes.items()))
    return tuple(result)


@dataclass(frozen=True)
class BicliqueCompression:
    form: GraphNormalForm
    partition: tuple
    block_size: int
    incidence_weight: int
    candidate_edges: int
    original_edges: int
    used: bool

    def stats(self):
        return {"block_size": self.block_size, "bicliques": len(self.partition),
                "incidence_weight": self.incidence_weight,
                "candidate_edges": self.candidate_edges,
                "original_edges": self.original_edges, "used": self.used,
                "vertices": self.form.n, "edges": len(self.form.edges)}


def compress_bicliques(form, block_size=None, *, retain_if_larger=True):
    """Replace each block by the exact two-variable Fourier gadget.

    The new scalar is the old scalar times 2**(-number_of_bicliques).
    Default finite-size search minimizes actual edge count W+k, not only W.
    With retain_if_larger=True (default), a non-improving graph is left intact.
    """
    t, adj = form.n, form.adjacency
    candidates = ([block_size] if block_size is not None
                  else range(1, max(1, t.bit_length() - 1) + 1))
    best = None
    for k in candidates:
        partition = biclique_partition(t, adj, k)
        weight = sum(len(a) + len(b) for a, b in partition)
        key = (weight + len(partition), weight, k)
        if best is None or key < best[0]:
            best = (key, partition, weight, k)
    key, partition, weight, k = best
    edge_count = key[0]
    used = not retain_if_larger or edge_count < len(form.edges)
    if used:
        edges = set()
        for j, (a, b) in enumerate(partition):
            u, v = t + 2 * j, t + 2 * j + 1
            edges.add((u, v))
            edges.update((x, u) for x in a)
            edges.update((x, v) for x in b)
        out = GraphNormalForm(form.labels + (None,) * (2 * len(partition)),
                              frozenset(edges), form.half_log2 - 2 * len(partition),
                              form.phase, form.default_signs, form.zero)
    else:
        out = form
    return BicliqueCompression(out, partition, k, weight, edge_count, len(form.edges), used)


def block_weight_bound(t, k):
    """A simple uniform bound used in the proof (not an asymptotic estimate).

    Cross-block class sizes contribute <=t²/(2k)+t; pattern sides <=t*2**k;
    internal stars contribute <=t*(k+1)/2. Ceilings are included conservatively.
    """
    if t < 0 or k < 1:
        raise ValueError("invalid dimensions")
    return math.ceil(t * t / (2 * k)) + t + math.ceil(t / k) * k * (1 << max(0, k - 1)) + math.ceil(t * (k + 1) / 2)


@dataclass(frozen=True)
class DegreeThreeRealization:
    n: int
    layers: tuple
    t: int
    scale_half_log2: int
    phase: int
    zero: bool
    slot_qubits: tuple[int, ...]

    @property
    def circuit(self):
        return [gate for layer in self.layers for gate in layer]


def degree_three_realization(form):
    """Seven-layer degree-three circuit, one T/TDG per original formal slot.

    form.value() = 2**(scale_half_log2/2)*omega**phase*<0|circuit|0>.
    A formal zero form uses the zero scalar flag, not a claim of zero circuit
    amplitude. S^a is one of I,S,Z,SDG, all in the supported fixed gate set.
    """
    v, edges = form.n, sorted(form.edges)
    incidence = [[] for _ in range(v)]
    for ei, (a, b) in enumerate(edges):
        incidence[a].append(ei)
        incidence[b].append(ei)
    endpoints, next_vertex = {}, v
    first, second = [], []
    for root, incident_edges in enumerate(incidence):
        previous = root
        for ei in incident_edges:
            copy, auxiliary = next_vertex, next_vertex + 1
            next_vertex += 2
            first.append(("CZ", (previous, auxiliary)))
            second.append(("CZ", (auxiliary, copy)))
            endpoints[(root, ei)] = copy
            previous = copy
    cross = [("CZ", (endpoints[(a, ei)], endpoints[(b, ei)]))
             for ei, (a, b) in enumerate(edges)]
    fifth, sixth = [], []
    slot_qubits = [None] * form.t
    f = 0
    for root, label in enumerate(form.labels):
        if label is None:
            continue
        slot_qubits[label.slot] = root
        if label.kind == "fourier":
            f += 1
            if label.phase:
                cross.append(("Z", (root,)))
            fifth.append(("H", (root,)))
        elif label.phase:
            fifth.append(({1: "S", 2: "Z", 3: "SDG"}[label.phase], (root,)))
        sixth.append(("T" if form.default_signs[label.slot] == 1 else "TDG", (root,)))
    hadamards = tuple(("H", (q,)) for q in range(next_vertex))
    layers = (hadamards, tuple(first), tuple(second), tuple(cross),
              tuple(fifth), tuple(sixth), hadamards)
    return DegreeThreeRealization(next_vertex, layers, form.t,
                                  form.half_log2 + 2 * v + 4 * len(edges) - f,
                                  form.phase, form.zero, tuple(slot_qubits))
