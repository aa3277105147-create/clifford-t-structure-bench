"""Exact basis invariants and scientific experiment bookkeeping regressions."""
import random

from pfsparse.affine import Affine
from pfsparse.basis import (_columns, _unreduced_delta, compress_rank,
                           local_search, xor_change)
from pfsparse.parityform import ParityForm
from pfsparse.pathsum import PathSum


def _random_form(rng, ids):
    ps = PathSum(nvars=max(ids) + 1, alive=set(ids))
    ps.c, ps.half_log2 = rng.randrange(8), rng.randrange(-9, 5)
    for j in ids:
        ps.add_a(j, rng.randrange(4))
        for k in ids:
            if j < k and rng.randrange(2):
                ps.toggle_b(j, k)
    # One dense row ensures every variable occurs, without forcing full rank.
    ps.parities = [Affine(sum(1 << j for j in ids), rng.randrange(2))]
    for _ in range(rng.randrange(len(ids) + 1)):
        mask = sum(1 << j for j in ids if rng.randrange(2))
        ps.parities.append(Affine(mask, rng.randrange(2)))
    return ParityForm.from_pathsum(ps)


def test_transvections_preserve_exact_values_with_noncontiguous_ids():
    rng = random.Random(2001)
    for _ in range(60):
        ids = sorted(rng.sample(range(16), rng.randrange(2, 8)))
        pf = _random_form(rng, ids)
        expected = pf.ps.evaluate_exact()
        original = pf.ps.copy()
        j, k = rng.sample(ids, 2)
        changed = xor_change(pf, j, k)
        assert changed.ps.evaluate_exact() == expected
        # The caller's object must remain reusable in comparisons of orders.
        assert pf.ps == original


def test_rank_compression_preserves_sum_and_removes_all_dependent_columns():
    rng = random.Random(2002)
    for _ in range(80):
        ids = sorted(rng.sample(range(20), rng.randrange(1, 8)))
        pf = _random_form(rng, ids)
        rank = pf.parity_matrix_rank()
        expected = pf.ps.evaluate_exact()
        changed = compress_rank(pf)
        assert changed.ps.evaluate_exact() == expected
        if not changed.ps.zero:
            assert changed.r <= rank
            assert changed.r == changed.parity_matrix_rank()
            assert changed.r <= changed.t


def test_xor_and_r2_can_increase_L_so_search_must_compare_final_form():
    ps = PathSum(nvars=7, alive=set(range(7)), a={1: 1},
                 b={(1, k) for k in range(2, 7)})
    ps.parities = [Affine.var(0) ^ Affine.var(1)] + [Affine.var(k) for k in range(2, 7)]
    pf = ParityForm.from_pathsum(ps)
    candidate = xor_change(pf, 0, 1)
    assert (pf.L, candidate.L) == (26, 33)
    assert candidate.ps.evaluate_exact() == ps.evaluate_exact()
    result = local_search(pf, max_rounds=10)
    assert result.L <= pf.L
    assert result.ps.evaluate_exact() == ps.evaluate_exact()


def test_local_search_exact_invariance_monotonicity_and_reproducibility():
    rng = random.Random(2003)
    for _ in range(35):
        pf = _random_form(rng, list(range(rng.randrange(2, 8))))
        expected = pf.ps.evaluate_exact()
        out = local_search(pf, max_rounds=5, sample=12, rng=random.Random(41))
        repeat = local_search(pf, max_rounds=5, sample=12, rng=random.Random(41))
        assert out.ps.evaluate_exact() == expected
        assert out.L <= pf.L
        assert out.ps == repeat.ps


def test_optimized_move_score_matches_full_reduction():
    rng = random.Random(2004)
    for _ in range(20):
        pf = _random_form(rng, [0, 2, 5, 7, 9])
        columns = _columns(pf.ps)
        neighbours = {j: set(pf.ps.neighbours(j)) for j in pf.ps.alive}
        for j in pf.ps.alive:
            for k in pf.ps.alive:
                if j != k and columns[j] != columns[k]:
                    predicted = pf.L + _unreduced_delta(pf.ps, j, k, neighbours)
                    assert xor_change(pf, j, k).L == predicted


def test_experiment_orders_share_output_and_keep_original_t_count():
    from pfsparse.experiments import _prepare_case, run_one
    _prepare_case.cache_clear()
    rows = [run_one("F1", 8, 2, order, 1, timeout=20, tw_max_nodes=0, verify_max_r=10)
            for order in ("natural", "min_degree", "min_fill")]
    assert all(row["status"] == "ok" for row in rows)
    assert len({row["output_y_hex"] for row in rows}) == 1
    assert all(row["t"] == 8 for row in rows)
    assert [row["preparation_reused"] for row in rows] == [False, True, True]
    assert all(row["ls_L"] <= row["rank_L"] for row in rows)
    assert all(row["output_verification"] == "exact_nonzero" for row in rows)


def test_resource_and_treewidth_budgets_are_explicit():
    from pfsparse.experiments import run_one
    from pfsparse.metrics import bounded_graph_metrics
    row = run_one("F3", 64, 0, "min_degree", 0, width_exponent=1.5, max_qubits=50)
    assert row["status"] == "resource_limit"
    assert row["n"] > 50 and "rank_L" not in row
    ps = PathSum(nvars=4, alive=set(range(4)), parities=[Affine(15)])
    skipped = bounded_graph_metrics(ps, max_nodes=3)
    assert skipped["tw_status"] == "skipped_size" and skipped["tw_upper"] is None
    measured = bounded_graph_metrics(ps, max_nodes=4)
    assert measured["tw_status"] == "computed_upper_bound"
    assert measured["tw_upper"] == 3


def test_timeout_retains_a_censored_row(monkeypatch):
    import pfsparse.experiments as experiments
    def timed_out(*args, **kwargs):
        raise experiments.ExperimentTimeout("test budget")
    # The timeout is raised after preparation, so deterministic instance fields
    # remain available even though a reduction did not finish.
    import pfsparse.reduce as reduction
    experiments._prepare_case.cache_clear()
    experiments._prepare_case("F1", 4, 0, 1.0, 4, 20, 10)
    monkeypatch.setattr(reduction, "reduce_pathsum", timed_out)
    row = experiments.run_one("F1", 4, 0, "natural", 0, verify_max_r=10, tw_max_nodes=0)
    assert row["status"] == "timeout"
    assert row["stage"] == "reduce"
    assert row["t"] == 4 and "red_L" not in row


def test_sparse_T_width_scaling_exposes_gate_ratio():
    from pfsparse.experiments import make_circuit
    from pfsparse.circuits import t_count
    ratios = []
    for t in (8, 32):
        circ, n = make_circuit("F2", t, random.Random(5), width_exponent=1.5, depth=4)
        assert n >= 4 * t ** 1.5
        assert t_count(circ) == t
        ratios.append(len(circ) / t)
    assert ratios[1] > ratios[0]
