"""Independent small-circuit checks of the theorem's explicit construction."""
import itertools
import random

import numpy as np
import pytest

from pfsparse.circuits import random_regular_graph_edges, t_count
from pfsparse.obstructions import (bounded_degree_graph_circuit, effective_graph_form,
                                  affine_graph_form)
from pfsparse.pathsum import build_pathsum
from pfsparse.reference import statevector
from pfsparse.reduce import reduce_pathsum


def all_graphs(t):
    pairs = list(itertools.combinations(range(t), 2))
    for mask in range(1 << len(pairs)):
        yield [e for i, e in enumerate(pairs) if mask & (1 << i)]


def test_depth_degree_gate_count_certificate():
    rng = random.Random(71)
    for t in (1, 2, 3, 7, 16):
        edges = [e for e in itertools.combinations(range(t), 2) if rng.randrange(2)]
        gadget = bounded_degree_graph_circuit(t, edges)
        assert gadget.n == t+4*len(edges)
        assert gadget.s == 3*t+13*len(edges)
        assert t_count(gadget.circuit) == t
        assert len(gadget.layers) == 5
        for layer in gadget.layers:
            operands = [q for _, qs in layer for q in qs]
            assert len(operands) == len(set(operands))
        neighbours = [set() for _ in range(gadget.n)]
        for u, v in gadget.physical_edges:
            neighbours[u].add(v)
            neighbours[v].add(u)
        assert max(map(len, neighbours)) <= 3


def test_all_graphs_up_to_three_roots_all_root_outputs():
    # The largest independent state vector has N=15 (complete graph on 3 roots).
    for t in (1, 2, 3):
        for edges in all_graphs(t):
            gadget = bounded_degree_graph_circuit(t, edges)
            state = statevector(gadget.circuit, gadget.n, 0)
            for output in range(1 << t):
                expected = effective_graph_form(t, edges, output)
                y = gadget.root_output(output)
                assert np.allclose(state[y], expected.value(), rtol=1e-11, atol=1e-12)
                for order in ("natural", "min_degree", "min_fill"):
                    reduced = reduce_pathsum(build_pathsum(gadget.circuit, gadget.n, 0, y), order)
                    assert reduced.ps.evaluate_exact() == expected.ps.evaluate_exact()


def test_smallest_example_exact_constant():
    pf = effective_graph_form(2, [(0, 1)])
    assert pf.ps.evaluate_exact() == (4, (1, 2, -1, 0))
    assert pf.L == 7


def test_affine_graph_coordinates_preserve_exact_value():
    rng = random.Random(83)
    for t in range(1, 7):
        for _ in range(12):
            edges = [e for e in itertools.combinations(range(t), 2) if rng.randrange(2)]
            rows = [1 << i for i in range(t)]
            if t > 1:
                for _ in range(4*t):
                    i, j = rng.sample(range(t), 2)
                    rows[i] ^= rows[j]
            offset, output = rng.randrange(1 << t), rng.randrange(1 << t)
            changed = affine_graph_form(t, edges, rows, offset, output)
            original = effective_graph_form(t, edges, output)
            assert changed.ps.evaluate_exact() == original.ps.evaluate_exact()
            assert changed.parity_matrix_rank() == t
            assert changed.L == 2*t+sum(row.bit_count() for row in rows)+changed.q_size


def test_regular_graph_generator_does_not_change_ensemble():
    for n, d in ((4, 3), (16, 3), (31, 4), (7, 0)):
        edges = random_regular_graph_edges(n, d, random.Random(5))
        degrees = [0]*n
        for u, v in edges:
            assert 0 <= u < v < n
            degrees[u] += 1
            degrees[v] += 1
        assert degrees == [d]*n
        assert len(set(edges)) == n*d//2
    with pytest.raises(ValueError):
        random_regular_graph_edges(7, 3, random.Random(5))
