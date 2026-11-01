from itertools import combinations
from math import comb, prod

import pytest

from pfsparse.q1_bounds import (
    colex_extremal_points, colex_size_bound, gf2_rank, graph_lagrangian,
    intersection_dimension, lagrangian_subspaces, piece_count_bound,
    quadratic_row, restriction_rank,
)


def test_restriction_bound_all_point_sets_through_three_bits():
    for t in range(4):
        rows = [quadratic_row(x, t) for x in range(1 << t)]
        for mask in range(1 << (1 << t)):
            rank = gf2_rank(row for x, row in enumerate(rows) if mask >> x & 1)
            assert mask.bit_count() <= colex_size_bound(t, rank)


def test_exact_equality_witness_for_every_rank_through_ten_bits():
    for t in range(11):
        for rank in range(comb(t, 2) + 1):
            points = colex_extremal_points(t, rank)
            assert len(set(points)) == len(points) == colex_size_bound(t, rank)
            assert restriction_rank(points, t) == rank


def test_lagrangian_enumerator_count_and_uniqueness():
    for t in range(4):
        spaces = list(lagrangian_subspaces(t))
        assert len(spaces) == prod((1 << j) + 1 for j in range(1, t + 1))
        assert len(set(spaces)) == len(spaces)
        assert all(gf2_rank(basis) == t for basis in spaces)


def test_graph_intersection_equals_kernel_of_adjacency_difference():
    t = 4
    for g in range(1 << comb(t, 2)):
        for h in (0, 1, 7, 31, 63):
            left, right = graph_lagrangian(t, g), graph_lagrangian(t, h)
            rank = gf2_rank((x ^ y) >> t for x, y in zip(left, right))
            assert intersection_dimension(left, right) == t - rank
    vertical = tuple(1 << (t + i) for i in range(t))
    assert all(intersection_dimension(vertical, graph_lagrangian(t, g)) == 0
               for g in range(1 << comb(t, 2)))


def test_stabilizer_overlap_formula_including_zero_phase_overlaps():
    t = 3
    pairs = list(combinations(range(t), 2))
    zero_cases = 0
    for g in range(1 << len(pairs)):
        for h in range(1 << len(pairs)):
            dimension = intersection_dimension(
                graph_lagrangian(t, g), graph_lagrangian(t, h))
            for phase in range(1 << t):
                total = sum(
                    (-1) ** (((quadratic_row(x, t) & (g ^ h)).bit_count()
                              + (phase & x).bit_count()) & 1)
                    for x in range(1 << t))
                if total == 0:
                    zero_cases += 1
                else:
                    assert total * total == 1 << (t + dimension)
    assert zero_cases > 0


def test_piece_count_finite_and_saturated_cases():
    assert piece_count_bound(0, 0, 1) == 1
    assert piece_count_bound(4, 0, 1) == 3
    assert piece_count_bound(4, 5, 1) == 1
    assert piece_count_bound(128, 128, 128) > 1 << 100
    assert colex_size_bound(5, 0) == 6
    assert colex_size_bound(5, 10) == 32


def test_q1_bound_input_and_enumeration_guards():
    with pytest.raises(ValueError):
        colex_size_bound(3, 4)
    with pytest.raises(ValueError):
        restriction_rank([8], 3)
    with pytest.raises(ValueError):
        list(lagrangian_subspaces(5))
    with pytest.raises(ValueError):
        piece_count_bound(5, 0, 0)
