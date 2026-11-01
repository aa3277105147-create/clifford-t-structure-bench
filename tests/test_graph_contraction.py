"""Independent exact and statevector checks of the boundary contraction theorem."""
import itertools
import random

import numpy as np
import pytest

from pfsparse.affine import Affine
from pfsparse.graph_contraction import (
    SingularCliffordComponentError, contract_graph, subdivide_edges_even,
)
from pfsparse.pathsum import PathSum, build_pathsum
from pfsparse.reference import statevector


def _direct_graph_sum(n, edges, roots, x, y):
    """Independent unreduced N-variable expansion, exact in Q(omega)."""
    ps = PathSum(nvars=n, alive=set(range(n)), half_log2=-2 * n)
    ps.parities = [Affine.var(j) for j in roots]
    ps.b = {tuple(sorted(edge)) for edge in edges}
    for j in range(n):
        if ((x ^ y) >> j) & 1:
            ps.a[j] = 2
    return ps.evaluate_exact()


def test_k4_negative_gaussian_sign_and_all_output_shifts():
    # Internal K4 has G(0)=-4; two distinct root attachments also exercise
    # induced linear terms, cross terms, and output-dependent scalar signs.
    roots = (0, 1)
    edges = list(itertools.combinations(range(2, 6), 2)) + [(0, 2), (0, 3), (1, 4)]
    for eta in range(64):
        result = contract_graph(6, edges, roots, x=19, y=19 ^ eta)
        assert result.components[0].gaussian_sign == -1
        assert result.components[0].adjacency_rank == 4
        assert result.components[0].boundary == roots
        assert result.pf.ps.evaluate_exact() == _direct_graph_sum(6, edges, roots, 19, 19 ^ eta)
        assert result.pf.ps.half_log2 == -8
        assert result.pf.L <= result.L_bound


def test_tiny_graphs_match_independent_statevectors_and_gate_pathsum():
    cases = [
        (0, [], [], 0),
        (1, [], [0], 1),
        (3, [(0, 1), (1, 2)], [0], 3),
        (4, [(0, 2), (2, 3), (3, 1), (0, 1)], [0, 1], 5),
        (4, list(itertools.combinations(range(4), 2)), [], 6),
        (4, [(0, 1), (2, 3)], [0, 1], 9),
    ]
    for n, edges, roots, x in cases:
        first = contract_graph(n, edges, roots, x, 0)
        circuit = first.specification.circuit()
        reference = statevector(circuit, n, x)
        for y in range(1 << n):
            result = contract_graph(n, edges, roots, x, y)
            assert np.isclose(result.pf.value(), reference[y], atol=1e-12)
            assert result.pf.ps.evaluate_exact() == build_pathsum(circuit, n, x, y).evaluate_exact()
            assert result.pf.r == result.pf.t == len(roots)


def test_random_small_graphs_exact_cyclotomic_and_bound():
    rng = random.Random(4401)
    accepted = 0
    singular = 0
    # Enumerate the full unreduced graph sum only in this oracle, never in
    # contract_graph.  Labels and root subsets are deliberately not prefixes.
    for _ in range(180):
        n = rng.randrange(1, 9)
        edges = [edge for edge in itertools.combinations(range(n), 2) if rng.randrange(2)]
        roots = [j for j in range(n) if rng.randrange(2)]
        x, y = rng.randrange(1 << n), rng.randrange(1 << n)
        try:
            result = contract_graph(n, edges, roots, x, y)
        except SingularCliffordComponentError:
            singular += 1
            continue
        accepted += 1
        assert result.pf.ps.evaluate_exact() == _direct_graph_sum(n, edges, roots, x, y)
        assert result.pf.L <= result.L_bound
        assert result.boundary_incidences <= result.max_degree * len(roots)
        assert all(c.adjacency_rank == c.size for c in result.components)
    assert accepted >= 45
    assert singular >= 45


def test_singular_components_rejected_explicitly():
    for n, edges, roots, rank in [
        (2, [(0, 1)], [0], 0),
        (4, [(0, 1), (1, 2), (2, 3)], [0], 2),
        (3, list(itertools.combinations(range(3), 2)), [], 2),
    ]:
        with pytest.raises(SingularCliffordComponentError) as caught:
            contract_graph(n, edges, roots)
        assert caught.value.rank == rank
        assert "criterion does not apply" in str(caught.value)


def test_even_paths_reproduce_root_graph_and_gate_counts():
    root_edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
    t, m = 4, len(root_edges)
    for pairs in (1, 2, 5, 32):
        spec = subdivide_edges_even(t, root_edges, pairs)
        result = contract_graph(spec.n, spec.edges, spec.roots)
        assert spec.n == t + 2 * pairs * m
        assert len(spec.edges) == (2 * pairs + 1) * m
        assert spec.gate_count == 3 * t + (6 * pairs + 1) * m
        assert result.max_boundary == result.max_degree == 2
        layers = spec.layers()
        assert len(layers) <= 2 * result.max_degree + 2
        assert sum(len(layer) for layer in layers) == spec.gate_count
        for layer in layers:
            occupied = [q for _, qubits in layer for q in qubits]
            assert len(occupied) == len(set(occupied))
        assert result.pf.ps.b == set(root_edges)
        assert result.pf.ps.a == {}
        assert result.pf.ps.c == 0
        assert result.pf.ps.half_log2 == -2 * (t + pairs * m)
        assert result.pf.L == 3 * t + m
        assert all(c.gaussian_sign == 1 and c.size == 2 * pairs for c in result.components)
        assert result.pf.L <= result.zero_io_L_bound


def test_even_path_arbitrary_io_has_same_cross_terms():
    rng = random.Random(4402)
    for pairs in (1, 2, 3):
        spec = subdivide_edges_even(2, [(0, 1)], pairs)
        for _ in range(12):
            x, y = rng.randrange(1 << spec.n), rng.randrange(1 << spec.n)
            result = contract_graph(spec.n, spec.edges, spec.roots, x, y)
            assert result.pf.ps.b == {(0, 1)}
            assert result.pf.L <= 9  # 4t + m
            assert result.pf.ps.evaluate_exact() == _direct_graph_sum(
                spec.n, spec.edges, spec.roots, x, y)


def test_graph_input_validation():
    invalid = [
        (2, [(0, 0)], [0]), (2, [(0, 1), (1, 0)], [0, 1]),
        (2, [(0, 2)], [0]), (2, [], [0, 0]), (2, [], [2]),
    ]
    for n, edges, roots in invalid:
        with pytest.raises(ValueError):
            contract_graph(n, edges, roots)
    with pytest.raises(ValueError):
        subdivide_edges_even(2, [(0, 1)], 0)
