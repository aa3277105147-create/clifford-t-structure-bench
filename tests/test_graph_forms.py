import itertools
import random

import numpy as np
import pytest

from pfsparse.affine import Affine
from pfsparse.pathsum import PathSum, build_pathsum
from pfsparse.reduce import reduce_pathsum, compress_to_rank
from pfsparse.circuits import random_clifford_t
from pfsparse.graph_form import (GraphLabel, GraphNormalForm, circuit_graph_normal_form,
                                graph_normal_form, inverse_binary, _canonical, _mul, _omega_power)
from pfsparse.biclique import biclique_partition, compress_bicliques, degree_three_realization, block_weight_bound
from pfsparse.mitm import fwht, mitm_value


def test_formal_slots_all_sign_fillings_exact():
    """Physical identities, including TDG and varying inputs/outputs, not residual slots."""
    rng = random.Random(616)
    for case in range(18):
        n, t = 2 + case % 2, 1 + case % 4
        circuit = random_clifford_t(n, 12 + t, t, rng)
        x, y = rng.randrange(1 << n), rng.randrange(1 << n)
        form = circuit_graph_normal_form(circuit, n, x, y)
        assert form.n == form.t == t
        positions = [j for j, (g, _) in enumerate(circuit) if g in ("T", "TDG")]
        for signs in itertools.product((-1, 1), repeat=t):
            filled = list(circuit)
            for j, sign in zip(positions, signs):
                filled[j] = ("T" if sign == 1 else "TDG", filled[j][1])
            expected = build_pathsum(filled, n, x, y).evaluate_exact()
            assert form.evaluate_exact(signs) == expected


def test_constant_slots_and_shared_scalar():
    circuit = [("T", (0,)), ("TDG", (0,))]
    for value in (0, 1):
        form = circuit_graph_normal_form(circuit, 1, value, value)
        assert form.n == 2
        assert all(label.kind == "fourier" for label in form.labels)
        assert form.phase == 0
        assert form.half_log2 == 0
        assert np.allclose(form.value([2 + 3j, -1 + 2j]), (2 + 3j) * (-1 + 2j) if value else 1)


def test_original_slot_order_survives_pivot_and_realization():
    # Slot0 is constant, slot1 is the pivot: pivot-first output would swap them.
    circuit = [("T", (0,)), ("H", (0,)), ("TDG", (0,)), ("H", (0,))]
    form = circuit_graph_normal_form(circuit, 1, 1, 0)
    assert form.order == (0, 1)
    realization = degree_three_realization(form)
    assert realization.slot_qubits == (0, 1)
    for signs in itertools.product((-1, 1), repeat=2):
        filled = []
        for gate, qs in realization.circuit:
            if gate in ("T", "TDG"):
                slot = realization.slot_qubits.index(qs[0])
                gate = "T" if signs[slot] == 1 else "TDG"
            filled.append((gate, qs))
        ps = build_pathsum(filled, realization.n, 0, 0)
        ps.half_log2 += realization.scale_half_log2
        ps.add_c(realization.phase)
        assert ps.evaluate_exact() == form.evaluate_exact(signs)


def test_zero_kernel_distinct_from_numeric_cancellation():
    zero = circuit_graph_normal_form([("T", (0,))], 1, 0, 1)
    assert zero.zero and zero.n == 1 and zero.evaluate_exact() == (0, (0, 0, 0, 0))
    circuit = [("H", (0,))] + [("T", (0,))] * 4 + [("H", (0,))]
    cancelling = circuit_graph_normal_form(circuit, 1, 0, 0)
    assert not cancelling.zero and cancelling.n == 4
    assert cancelling.evaluate_exact() == (0, (0, 0, 0, 0))
    assert abs(cancelling.value([1] * 4) - 1) < 1e-12


def test_empty_graph_and_pure_clifford():
    assert circuit_graph_normal_form([], 0).evaluate_exact() == (0, (1, 0, 0, 0))
    for circuit, y in [([("H", (0,))], 0), ([("H", (0,)), ("S", (0,)), ("H", (0,))], 1)]:
        form = circuit_graph_normal_form(circuit, 1, 0, y)
        assert form.n == 0
        assert form.evaluate_exact() == build_pathsum(circuit, 1, 0, y).evaluate_exact()


def test_reduced_form_conversion_exact_and_polynomial_size():
    rng = random.Random(19)
    for _ in range(20):
        circuit = random_clifford_t(3, 20, 5, rng)
        pf = compress_to_rank(reduce_pathsum(build_pathsum(circuit, 3, 0, rng.randrange(8))))
        graph = graph_normal_form(pf)
        assert graph.evaluate_exact() == pf.ps.evaluate_exact()
    # The prototype's 2**r validation loop could not handle this size.
    ps = PathSum()
    for j in range(80):
        ps.new_var()
        ps.parities.append(Affine.var(j))
    from pfsparse.parityform import ParityForm
    graph = graph_normal_form(ParityForm(ps))
    assert graph.n == 80 and not graph.edges


def test_gf2_inverse_and_invalid_rank():
    rows = [0b011, 0b110, 0b111]
    inverse = inverse_binary(rows)
    for x in range(8):
        y = sum(((row & x).bit_count() & 1) << j for j, row in enumerate(rows))
        z = sum(((row & y).bit_count() & 1) << j for j, row in enumerate(inverse))
        assert z == x
    with pytest.raises(ValueError):
        inverse_binary([1, 1])


def test_partition_edge_multiplicity_and_bound():
    rng = random.Random(91)
    for n in (0, 1, 2, 7, 15, 31):
        edges = {(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < .5}
        form = GraphNormalForm((None,) * n, frozenset(edges))
        for k in (1, 2, 4, 8):
            part = biclique_partition(n, form.adjacency, k)
            observed = []
            for a, b in part:
                assert not set(a) & set(b)
                observed.extend(tuple(sorted((u, v))) for u in a for v in b)
            assert len(observed) == len(set(observed))
            assert set(observed) == edges
            assert sum(len(a) + len(b) for a, b in part) <= block_weight_bound(n, k)


def test_auxiliary_gadget_exact_with_complex_labels():
    # One K2,2 block; Fourier labels test denominators as well as phases.
    labels = (GraphLabel("direct", 1, 0), GraphLabel("fourier", 1, 1),
              GraphLabel("direct", 3, 2), GraphLabel("fourier", 0, 3))
    form = GraphNormalForm(labels, frozenset({(0, 2), (0, 3), (1, 2), (1, 3)}), -3, 5, (1, -1, 1, -1))
    comp = compress_bicliques(form, 2, retain_if_larger=False)
    assert comp.form.n == 6 and len(comp.form.edges) == 5
    for signs in itertools.product((-1, 1), repeat=4):
        assert comp.form.evaluate_exact(signs) == form.evaluate_exact(signs)
    z = [2 + 1j, -3j, .25, -.3 + .8j]
    assert abs(comp.form.value(z) - form.value(z)) < 1e-12


def test_nonimproving_compression_is_explicit():
    form = GraphNormalForm((None, None), frozenset({(0, 1)}))
    comp = compress_bicliques(form)
    assert not comp.used and comp.form is form
    assert comp.candidate_edges == 3


def test_degree_three_realization_exact_and_layer_legal():
    for label in [GraphLabel("direct", a, 0) for a in range(4)] + [GraphLabel("fourier", b, 0) for b in range(2)]:
        for sign in (-1, 1):
            form = GraphNormalForm((label, None), frozenset({(0, 1)}), -2, 3, (sign,))
            realized = degree_three_realization(form)
            assert len(realized.layers) == 7 and realized.n == 6
            neighbours = [set() for _ in range(realized.n)]
            for layer in realized.layers:
                seen = set()
                for gate, qs in layer:
                    assert not seen.intersection(qs)
                    seen.update(qs)
                    if gate == "CZ":
                        a, b = qs
                        neighbours[a].add(b)
                        neighbours[b].add(a)
            assert max(map(len, neighbours)) <= 3
            assert sum(g in ("T", "TDG") for g, _ in realized.circuit) == 1
            ps = build_pathsum(realized.circuit, realized.n, 0, 0)
            ps.half_log2 += realized.scale_half_log2
            ps.add_c(realized.phase)
            assert ps.evaluate_exact() == form.evaluate_exact()


def test_mitm_against_exact_and_formal_enumeration():
    rng = random.Random(74)
    for n in range(10):
        labels = tuple(GraphLabel("direct" if j % 2 else "fourier", j % (4 if j % 2 else 2), j) for j in range(n))
        edges = frozenset((i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < .5)
        form = GraphNormalForm(labels, edges, -3, 2, (1,) * n)
        z = [complex(rng.random(), rng.random()) for _ in range(n)]
        assert abs(form.mitm_value(z) - form.value(z)) < 1e-11
        d, coeff = form.evaluate_exact()
        oracle = 2.0 ** -d * sum(a * np.exp(1j * np.pi * j / 4) for j, a in enumerate(coeff))
        assert abs(form.mitm_value() - oracle) < 1e-11


def test_walsh_involution_and_resource_guard():
    a = np.array([1 + 2j, 3, -4j, -5])
    assert np.allclose(fwht(fwht(a)), 4 * a)
    with pytest.raises(ValueError):
        fwht([1, 2, 3])
    with pytest.raises(ValueError):
        mitm_value(50, [0] * 50, [(1, 1)] * 50)


def test_ising_rewrite_all_subsets():
    rng = random.Random(5)
    for n in (3, 6, 9):
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < .5]
        degrees = [sum(i in edge for edge in edges) for i in range(n)]
        for x in range(1 << n):
            inside = sum(((x >> i) & 1) & ((x >> j) & 1) for i, j in edges)
            cut = sum(((x >> i) ^ (x >> j)) & 1 for i, j in edges)
            degree_sum = sum(degrees[i] for i in range(n) if (x >> i) & 1)
            assert degree_sum - cut == 2 * inside
