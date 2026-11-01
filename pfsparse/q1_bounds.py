"""Exact finite checks for the Q1 representation bounds.

These are small-instance combinatorial tools, not a quantum simulation
algorithm. All ranks, dimensions, and support bounds use integer arithmetic.
The enumerators intentionally restrict their exponential workloads.
"""

from itertools import combinations
from math import comb, isqrt


def gf2_rank(rows):
    """Rank of nonnegative bit-packed row vectors over GF(2)."""
    pivots = {}
    for row in rows:
        if not isinstance(row, int) or row < 0:
            raise ValueError("Rows must be nonnegative integers")
        while row:
            pivot = row.bit_length() - 1
            if pivot in pivots:
                row ^= pivots[pivot]
            else:
                pivots[pivot] = row
                break
    return len(pivots)


def _check_t(t):
    if not isinstance(t, int) or t < 0:
        raise ValueError("t must be a nonnegative integer")


def quadratic_row(x, t):
    """Evaluation of all square-free degree-two monomials at x."""
    _check_t(t)
    if not isinstance(x, int) or not 0 <= x < 1 << t:
        raise ValueError("Point lies outside the t-dimensional Boolean cube")
    row = 0
    for k, (i, j) in enumerate(combinations(range(t), 2)):
        if (x >> i) & (x >> j) & 1:
            row |= 1 << k
    return row


def restriction_rank(points, t):
    """Rank R(P) of pure quadratic monomials restricted to the given set."""
    return gf2_rank(quadratic_row(x, t) for x in points)


def colex_size_bound(t, rank):
    """Sharp support bound M_t(R), a consequence of Kruskal--Katona.

    Write R = C(a, 2) + b with 0 <= b < a. The result is
    t - a + 2**a + 2**b - 1. It is attained for every admissible rank.
    """
    _check_t(t)
    if not isinstance(rank, int) or not 0 <= rank <= comb(t, 2):
        raise ValueError("rank must lie between zero and C(t, 2)")
    if t == 0:
        return 1
    a = (1 + isqrt(1 + 8 * rank)) // 2
    b = rank - comb(a, 2)
    assert 0 <= b < a
    return t - a + (1 << a) + (1 << b) - 1


def colex_extremal_points(t, rank):
    """All cliques of the colex graph, including the empty clique.

    This explicit exponential witness is limited to t <= 16.
    """
    colex_size_bound(t, rank)
    if t > 16:
        raise ValueError("Explicit extremal-point enumeration requires t <= 16")
    if t == 0:
        return (0,)
    a = (1 + isqrt(1 + 8 * rank)) // 2
    b = rank - comb(a, 2)
    points = list(range(1 << a))
    if b:
        points.extend((1 << a) | y for y in range(1 << b))
        isolated_start = a + 1
    else:
        isolated_start = a
    points.extend(1 << i for i in range(isolated_start, t))
    return tuple(points)


def piece_count_bound(t, description_bits, confidence_bits):
    """Finite lower bound for a fixed description budget.

    If the dictionary has at most 2**description_bits descriptions, the
    theorem guarantees this many disjoint pieces with failure probability
    at most 2**(-confidence_bits). This function evaluates the bound only.
    """
    _check_t(t)
    if not isinstance(description_bits, int) or description_bits < 0:
        raise ValueError("description_bits must be a nonnegative integer")
    if not isinstance(confidence_bits, int) or confidence_bits < 1:
        raise ValueError("confidence_bits must be a positive integer")
    rank_cap = min(description_bits + confidence_bits, comb(t, 2))
    support_cap = colex_size_bound(t, rank_cap)
    return ((1 << t) + support_cap - 1) // support_cap


def symplectic_inner(v, w, t):
    """Pairing for bit vectors (x | z), with x in the low t bits."""
    _check_t(t)
    if not 0 <= v < 1 << (2 * t) or not 0 <= w < 1 << (2 * t):
        raise ValueError("Vector lies outside the symplectic space")
    mask = (1 << t) - 1
    return (((v & mask) & (w >> t)).bit_count()
            + ((w & mask) & (v >> t)).bit_count()) & 1


def _rref_subspaces(n, dimension):
    """Each row-space exactly once, in reduced row-echelon form."""
    for pivots in combinations(range(n), dimension):
        pivot_set = set(pivots)
        free = [(i, j) for i, p in enumerate(pivots)
                for j in range(p + 1, n) if j not in pivot_set]
        for assignment in range(1 << len(free)):
            rows = [1 << p for p in pivots]
            for k, (i, j) in enumerate(free):
                if (assignment >> k) & 1:
                    rows[i] |= 1 << j
            yield tuple(rows)


def lagrangian_subspaces(t):
    """Enumerate all Lagrangians, including non-transverse ones, for t<=4."""
    _check_t(t)
    if t > 4:
        raise ValueError("Complete Lagrangian enumeration requires t <= 4")
    for rows in _rref_subspaces(2 * t, t):
        if all(symplectic_inner(rows[i], rows[j], t) == 0
               for i in range(t) for j in range(i)):
            yield rows


def graph_lagrangian(t, graph_mask):
    """Basis e_i | A_G e_i for the simple graph encoded by graph_mask."""
    _check_t(t)
    if not isinstance(graph_mask, int) or not 0 <= graph_mask < 1 << comb(t, 2):
        raise ValueError("Invalid simple-graph mask")
    adjacency = [0] * t
    for k, (i, j) in enumerate(combinations(range(t), 2)):
        if (graph_mask >> k) & 1:
            adjacency[i] |= 1 << j
            adjacency[j] |= 1 << i
    return tuple((1 << i) | (adjacency[i] << t) for i in range(t))


def intersection_dimension(left, right):
    """Dimension of the intersection of two bit-packed row spans."""
    left, right = tuple(left), tuple(right)
    return gf2_rank(left) + gf2_rank(right) - gf2_rank(left + right)
