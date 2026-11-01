from scripts.check_q1_dense import (
    bounded_cancellation_check, character_and_gauss_check, level_set_check,
    flower_guard_check, moment_domination_check, refinement_check,
)


def test_dense_refinement_preserves_counts_mass_and_projection_scope():
    result = refinement_check(32)
    assert result["all_pass"] and result["parents_with_multiple_extensions"] > 0
    assert result["refinements_with_overlapping_projections"] > 0


def test_bounded_mass_above_one_does_not_force_monochromatic_queries():
    result = bounded_cancellation_check()
    assert result["all_pass"] and result["maximum_mass"] == "5/4"


def test_both_nonzero_quadratic_level_sets_are_large():
    result = level_set_check(5)
    assert result["all_pass"] and result["nonzero_graphs"] == 1094


def test_even_moment_domination_for_nonnegative_weights_exactly():
    result = moment_domination_check()
    assert result["all_pass"] and result["indicator_moment_comparisons"] == 1390
    assert result["rational_moment_comparisons"] == 768


def test_character_expansion_and_gauss_moment_constant_exactly():
    result = character_and_gauss_check(4)
    assert result["all_pass"] and result["character_tuples"] == 4160
    assert result["gauss_moment_bound_checks"] == 40


def test_posthoc_flower_guards_partition_counts_without_changing_tree():
    result = flower_guard_check()
    assert result["all_pass"] and result["flower_families"] == 28
    assert result["raw_covers_with_overlaps"] > 0
    assert result["paths_with_duplicate_guards"] > 0
    assert result["leaves_with_opposite_guards_removed"] > 0
    assert result["partitions_with_overlapping_projections"] > 0
