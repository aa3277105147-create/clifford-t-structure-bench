from scripts.check_interference import parity_cnf_check, walsh_representation_check


def test_quadratic_walsh_spectra_and_positive_counting_representations():
    result = walsh_representation_check(5)
    assert result["all_pass"] and result["graphs"] == 1100
    assert result["examples"]["triangle_affine_fourier_support"] == {
        "n": 3, "rank": 2, "support_size": 4,
        "zero_in_fourier_support": False,
        "listed_pair_queries": 8, "nonzero_pair_queries": 8,
        "pair_mass": 2, "constant_queries": 5,
        "constant_maximum_mass": 5,
    }
    assert result["examples"]["one_edge_full_rank"]["constant_queries"] == 4
    assert result["examples"]["empty_three_vertices"]["constant_maximum_mass"] == 1


def test_all_small_parity_queries_have_unique_cnf_extensions():
    result = parity_cnf_check(6)
    assert result["all_pass"] and result["parity_queries"] == 254
    assert result["unique_extensions_and_complements_checked"]
    assert result["maximum_declared_variables"] == 10
    assert result["zero_queries"] == 7
