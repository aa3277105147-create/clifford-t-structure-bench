"""Exact contraction for H^N (CZ graph + singleton T roots) H^N.

Deleting the T roots must leave components with nonsingular F2 adjacency.
Each component then contracts to a quadratic on its boundary, with an exact
Gaussian sign and a power-of-sqrt(2) normalization.  No enumeration is used by
the contraction algorithm.  A singular component means this sufficient
criterion does not apply; it does not prove that a small form is impossible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .affine import Affine
from .parityform import ParityForm
from .pathsum import PathSum


class SingularCliffordComponentError(ValueError):
    """A Clifford-only component fails the nonsingularity criterion."""

    def __init__(self, vertices: tuple[int, ...], rank: int):
        self.vertices = vertices
        self.rank = rank
        super().__init__(
            f"Clifford component {vertices!r} has singular F2 adjacency "
            f"(rank {rank} < {len(vertices)}); the nonsingular-boundary "
            "criterion does not apply"
        )


@dataclass(frozen=True)
class ComponentContraction:
    """Exact per-component certificate; vertices use physical qubit labels."""

    vertices: tuple[int, ...]
    boundary: tuple[int, ...]
    adjacency_rank: int
    gaussian_sign: int
    output_shift_sign: int
    linear_terms: int
    quadratic_terms: int

    @property
    def size(self) -> int:
        return len(self.vertices)


@dataclass(frozen=True)
class GraphCircuitSpecification:
    """A simple graph circuit; bit q of x or y refers to physical qubit q."""

    n: int
    edges: tuple[tuple[int, int], ...]
    roots: tuple[int, ...]

    @property
    def gate_count(self) -> int:
        return 2 * self.n + len(self.roots) + len(self.edges)

    def layers(self) -> list[list]:
        """Return an explicit schedule with disjoint gates in every layer.

        Greedy edge coloring uses at most max(0, 2*Delta-1) CZ layers.
        With a separate T layer and two Hadamard layers, depth is at most
        2*Delta+2 when Delta>=1.  An edgeless graph uses at most three layers.
        """
        used = [set() for _ in range(self.n)]
        cz_layers = []
        for edge in self.edges:
            j, k = edge
            unavailable = used[j] | used[k]
            color = 0
            while color in unavailable:
                color += 1
            if color == len(cz_layers):
                cz_layers.append([])
            cz_layers[color].append(("CZ", edge))
            used[j].add(color)
            used[k].add(color)
        hadamards = [("H", (j,)) for j in range(self.n)]
        layers = [hadamards] if hadamards else []
        layers.extend(cz_layers)
        if self.roots:
            layers.append([("T", (j,)) for j in self.roots])
        if hadamards:
            layers.append(list(hadamards))
        return layers

    def circuit(self) -> list:
        """Flatten the certified layers into build_pathsum/statevector format."""
        return [gate for layer in self.layers() for gate in layer]


@dataclass
class GraphContraction:
    """A singleton-root parity form and the parameters in its size bound.

    Parity-form variable j corresponds to physical vertex roots[j].  The
    integer half_log2 in pf.ps includes the entire Hadamard/Gaussian magnitude;
    pf.ps.c includes all Gaussian signs.  The final sum may still be zero.
    """

    pf: ParityForm
    specification: GraphCircuitSpecification
    components: tuple[ComponentContraction, ...]
    max_degree: int
    input_x: int
    output_y: int

    @property
    def n(self) -> int:
        return self.specification.n

    @property
    def roots(self) -> tuple[int, ...]:
        return self.specification.roots

    @property
    def t(self) -> int:
        return len(self.roots)

    @property
    def max_boundary(self) -> int:
        return max((len(c.boundary) for c in self.components), default=0)

    @property
    def boundary_incidences(self) -> int:
        return sum(len(c.boundary) for c in self.components)

    @property
    def L_bound(self) -> int:
        """Floor of [4 + Delta (b + 2)/2] t, valid for every input/output."""
        return 4 * self.t + self.max_degree * (self.max_boundary + 2) * self.t // 2

    @property
    def zero_io_L_bound(self) -> int | None:
        """The theorem's improved zero-input/zero-output bound, if applicable."""
        if self.input_x or self.output_y:
            return None
        return 3 * self.t + self.max_degree * (self.max_boundary + 2) * self.t // 2

    def stats(self) -> dict:
        return {
            "n": self.n, "t": self.t,
            "s": self.specification.gate_count,
            "edges": len(self.specification.edges),
            "max_degree": self.max_degree,
            "components": len(self.components),
            "max_component_size": max((c.size for c in self.components), default=0),
            "max_boundary": self.max_boundary,
            "boundary_incidences": self.boundary_incidences,
            "L": self.pf.L, "L_bound": self.L_bound,
            "half_log2": self.pf.ps.half_log2,
            "constant_phase": self.pf.ps.c,
        }


def _bits(mask: int):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def _validated_specification(n, edges, roots) -> GraphCircuitSpecification:
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    root_list = list(roots)
    if (len(set(root_list)) != len(root_list)
            or any(not isinstance(j, int) or not 0 <= j < n for j in root_list)):
        raise ValueError("roots must be distinct physical qubit indices")
    normalized = set()
    for edge in edges:
        if len(edge) != 2:
            raise ValueError("each edge must have two endpoints")
        j, k = edge
        if (not isinstance(j, int) or not isinstance(k, int)
                or not 0 <= j < n or not 0 <= k < n or j == k):
            raise ValueError(f"invalid simple-graph edge {edge!r}")
        item = (min(j, k), max(j, k))
        if item in normalized:
            raise ValueError(f"duplicate simple-graph edge {item!r}")
        normalized.add(item)
    return GraphCircuitSpecification(n, tuple(sorted(normalized)), tuple(sorted(root_list)))


def _inverse_binary(rows: list[int]) -> tuple[list[int], int]:
    """Gauss-Jordan elimination on bitsets; return inverse rows and rank."""
    size = len(rows)
    augmented = [row | (1 << (size + j)) for j, row in enumerate(rows)]
    rank = 0
    for column in range(size):
        pivot = next((j for j in range(rank, size)
                      if (augmented[j] >> column) & 1), None)
        if pivot is None:
            continue
        augmented[rank], augmented[pivot] = augmented[pivot], augmented[rank]
        for j in range(size):
            if j != rank and (augmented[j] >> column) & 1:
                augmented[j] ^= augmented[rank]
        rank += 1
    if rank != size:
        return [], rank
    return [row >> size for row in augmented], rank


def _matvec(rows: list[int], value: int) -> int:
    return sum(((row & value).bit_count() & 1) << j for j, row in enumerate(rows))


def _quadratic_value(upper_rows: list[int], value: int) -> int:
    return sum((upper_rows[j] & value).bit_count() for j in _bits(value)) & 1


def _gaussian_sign(adjacency: list[int]) -> int:
    """Sign of sum_z (-1)^q(z), for nonsingular graph quadratic q.

    Contract a coupled pair a,b using
        sum_{a,b} (-1)^(ab+a alpha+b beta) = 2 (-1)^(alpha beta).
    Linear and constant terms created by contraction are retained exactly.
    """
    rows = list(adjacency)
    alive = (1 << len(rows)) - 1
    linear = constant = 0
    while alive:
        a = (alive & -alive).bit_length() - 1
        neighbours = rows[a] & alive
        if not neighbours:
            raise ValueError("Gaussian pair contraction requires nonsingular adjacency")
        b = (neighbours & -neighbours).bit_length() - 1
        remaining = alive ^ (1 << a) ^ (1 << b)
        alpha, beta = rows[a] & remaining, rows[b] & remaining
        la, lb = (linear >> a) & 1, (linear >> b) & 1
        constant ^= la & lb
        linear = ((linear & remaining) ^ (beta if la else 0)
                  ^ (alpha if lb else 0) ^ (alpha & beta))
        for j in _bits(remaining):
            rows[j] &= remaining
            if (alpha >> j) & 1:
                rows[j] ^= beta
            if (beta >> j) & 1:
                rows[j] ^= alpha
        rows[a] = rows[b] = 0
        alive = remaining
    return -1 if constant else 1


def contract_graph(n: int, edges: Iterable[tuple[int, int]], roots: Iterable[int],
                   x: int = 0, y: int = 0) -> GraphContraction:
    """Construct the exact parity form for <y|H^N D_H T_roots H^N|x>.

    All linear algebra is over F2 and all phases/normalizations are integer
    data.  The algorithm is polynomial in the physical graph size.  Bounded
    physical degree and bounded component boundary are only needed to deduce
    L=O(t), not for correctness of contraction itself.

    Raises SingularCliffordComponentError whenever any component outside the
    roots has singular adjacency.  No claim about that case is inferred.
    """
    spec = _validated_specification(n, edges, roots)
    for name, value in (("x", x), ("y", y)):
        if not isinstance(value, int) or not 0 <= value < (1 << n):
            raise ValueError(f"{name} must be an n-bit computational-basis index")
    adjacency = [set() for _ in range(n)]
    for j, k in spec.edges:
        adjacency[j].add(k)
        adjacency[k].add(j)
    root_index = {physical: j for j, physical in enumerate(spec.roots)}
    root_set = set(spec.roots)
    t = len(spec.roots)
    eta = x ^ y
    ps = PathSum(nvars=t, alive=set(range(t)), half_log2=-(n + t))
    ps.parities = [Affine.var(j) for j in range(t)]
    for physical, j in root_index.items():
        if (eta >> physical) & 1:
            ps.add_a(j, 2)
    for j, k in spec.edges:
        if j in root_index and k in root_index:
            ps.toggle_b(root_index[j], root_index[k])

    components = []
    unseen = set(range(n)) - root_set
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, vertices, boundary = [start], [], set()
        while stack:
            j = stack.pop()
            vertices.append(j)
            for k in adjacency[j]:
                if k in root_set:
                    boundary.add(k)
                elif k in unseen:
                    unseen.remove(k)
                    stack.append(k)
        vertices = tuple(sorted(vertices))
        boundary = tuple(sorted(boundary))
        local = {physical: j for j, physical in enumerate(vertices)}
        rows = [sum(1 << local[k] for k in adjacency[j] if k in local)
                for j in vertices]
        inverse, rank = _inverse_binary(rows)
        if rank != len(vertices):
            raise SingularCliffordComponentError(vertices, rank)
        upper = [row & ~((1 << (j + 1)) - 1) for j, row in enumerate(rows)]
        eta_local = sum(((eta >> physical) & 1) << j
                        for j, physical in enumerate(vertices))
        shift = _matvec(inverse, eta_local)
        shift_phase = _quadratic_value(upper, shift)
        gaussian_sign = _gaussian_sign(rows)
        if (gaussian_sign == -1) ^ bool(shift_phase):
            ps.add_c(4)

        rhs = [sum(1 << local[k] for k in adjacency[root] if k in local)
               for root in boundary]
        images = [_matvec(inverse, column) for column in rhs]
        linear_terms = quadratic_terms = 0
        for j, (root, vector) in enumerate(zip(boundary, images)):
            # q(v0+vj)+q(v0) = q(vj)+eta^T vj.
            linear = _quadratic_value(upper, vector) ^ ((eta_local & vector).bit_count() & 1)
            if linear:
                ps.add_a(root_index[root], 2)
                linear_terms += 1
            for k in range(j + 1, len(boundary)):
                # Polar cross term vj^T A vk = R_j^T vk.
                if (rhs[j] & images[k]).bit_count() & 1:
                    ps.toggle_b(root_index[root], root_index[boundary[k]])
                    quadratic_terms += 1
        components.append(ComponentContraction(
            vertices, boundary, rank, gaussian_sign,
            -1 if shift_phase else 1, linear_terms, quadratic_terms,
        ))

    return GraphContraction(
        ParityForm.from_pathsum(ps), spec, tuple(components),
        max((len(row) for row in adjacency), default=0), x, y,
    )


def subdivide_edges_even(n_roots: int, root_edges: Iterable[tuple[int, int]],
                         internal_pairs: int = 1) -> GraphCircuitSpecification:
    """Replace each root edge by a path with 2*internal_pairs new vertices.

    Root labels are 0,...,n_roots-1.  With a connected bounded-degree root
    graph, this gives a connected bounded-degree constant-depth family whose
    gate-to-T ratio is unbounded as internal_pairs grows.  The general helper
    also accepts disconnected root graphs.  It returns the physical graph,
    roots, and explicit parallel layers, without enumerating any amplitudes.
    """
    if not isinstance(internal_pairs, int) or internal_pairs < 1:
        raise ValueError("internal_pairs must be a positive integer")
    original = _validated_specification(n_roots, root_edges, range(n_roots))
    edges, next_vertex = [], n_roots
    for left, right in original.edges:
        interior = list(range(next_vertex, next_vertex + 2 * internal_pairs))
        path = [left] + interior + [right]
        edges.extend((min(j, k), max(j, k)) for j, k in zip(path, path[1:]))
        next_vertex += len(interior)
    return GraphCircuitSpecification(next_vertex, tuple(sorted(edges)), original.roots)
