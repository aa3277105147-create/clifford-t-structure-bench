"""Invertible F2 variable changes and heuristic minimization of parity-form size.

The raw transvection specializes R1 to z_j = w_j XOR w_k. It delays
Gaussian elimination so a rank-elimination plan keeps valid variable indices.
"""
from __future__ import annotations

import random
import time

from .affine import Affine
from .parityform import ParityForm


def _xor_change_raw(ps, j: int, k: int) -> None:
    """In place, substitute z_j = w_j XOR w_k without summing variables.

    Only parity-matrix column k changes (column k ^= column j).
    """
    if j == k or j not in ps.alive or k not in ps.alive:
        raise ValueError("xor_change requires two distinct summed variables")
    aj = ps.a.get(j, 0)
    neighbours = ps.neighbours(j)
    ps.add_a(k, aj + 2 * (k in neighbours))
    if aj & 1:
        ps.toggle_b(j, k)
    for ell in neighbours:
        if ell != k:
            ps.toggle_b(k, ell)
    bit_j, bit_k = 1 << j, 1 << k
    ps.parities = [Affine(p.mask ^ bit_k if p.mask & bit_j else p.mask, p.const)
                   for p in ps.parities]
    ps.constraints = [Affine(p.mask ^ bit_k if p.mask & bit_j else p.mask, p.const)
                      for p in ps.constraints]


def xor_change(pf: ParityForm, j: int, k: int) -> ParityForm:
    """Change z_j = w_j XOR w_k and fully re-reduce a copy.

    The sum is preserved, but L can increase through Gaussian clique fill.
    A heuristic must compare the fully reduced candidate with the old form.
    """
    from .reduce import reduce_pathsum
    ps = pf.ps.copy()
    _xor_change_raw(ps, j, k)
    return reduce_pathsum(ps)


def _columns(ps) -> dict[int, int]:
    columns = dict.fromkeys(ps.alive, 0)
    for row, parity in enumerate(ps.parities):
        mask = parity.mask
        while mask:
            low = mask & -mask
            j = low.bit_length() - 1
            if j not in columns:
                raise ValueError("parity references a variable outside alive")
            columns[j] |= 1 << row
            mask ^= low
    return columns


def compress_rank(pf: ParityForm) -> ParityForm:
    """Zero dependent parity columns, then sum those variables out.

    Output r is at most the input matrix rank (and hence at most t).
    This is a dimension guarantee, not monotonicity of L.
    """
    from .reduce import reduce_pathsum
    ps = pf.ps.copy()
    if ps.zero:
        return ParityForm.from_pathsum(ps)
    if ps.constraints:
        raise ValueError("rank compression requires a constraint-free form")
    columns = _columns(ps)
    # Each reduced column is stored with its expression in original pivot columns.
    pivots: dict[int, tuple[int, int]] = {}
    for k in sorted(ps.alive):
        vector, combination = columns[k], 0
        while vector:
            lead = vector.bit_length() - 1
            if lead not in pivots:
                pivots[lead] = (vector, combination ^ (1 << k))
                break
            pivot_vector, pivot_combination = pivots[lead]
            vector ^= pivot_vector
            combination ^= pivot_combination
        if vector == 0:
            while combination:
                low = combination & -combination
                _xor_change_raw(ps, low.bit_length() - 1, k)
                combination ^= low
    out = reduce_pathsum(ps)
    if not out.ps.zero and out.r > len(pivots):
        raise AssertionError("rank compression failed to remove dependent variables")
    return out


def _unreduced_delta(ps, j: int, k: int, neighbours) -> int:
    """Exact L change when no parity column vanishes."""
    bj, bk = 1 << j, 1 << k
    delta = sum((-1 if p.mask & bk else 1)
                for p in ps.parities if p.mask & bj)
    aj, ak = ps.a.get(j, 0), ps.a.get(k, 0)
    jk = k in neighbours[j]
    new_ak = (ak + aj + 2 * jk) % 4
    delta += int(new_ak != 0) - int(ak != 0)
    if aj & 1:
        delta += -1 if jk else 1
    for ell in neighbours[j]:
        if ell != k:
            delta += -1 if ell in neighbours[k] else 1
    return delta


def local_search(pf: ParityForm, max_rounds: int = 50, sample: int | None = None,
                 rng: random.Random | None = None,
                 time_budget: float | None = None) -> ParityForm:
    """Greedy best improvement over sampled or all elementary XOR moves.

    Only a strictly smaller fully reduced L is accepted. Budget exhaustion is
    not a certificate of a globally optimal basis. A supplied RNG reproduces
    the sampled moves; absent one, a fixed seed is used.
    """
    if max_rounds < 0 or (sample is not None and sample < 0):
        raise ValueError("search budgets must be nonnegative")
    if time_budget is not None and time_budget < 0:
        raise ValueError("time_budget must be nonnegative")
    rng = rng if rng is not None else random.Random(0)
    current = ParityForm.from_pathsum(pf.ps)
    start = time.perf_counter()
    for _ in range(max_rounds):
        if current.ps.zero or current.r < 2:
            break
        if time_budget is not None and time.perf_counter() - start >= time_budget:
            break
        variables = sorted(current.ps.alive)
        n = len(variables)
        count = n * (n - 1)
        choices = range(count) if sample is None or sample >= count else rng.sample(range(count), sample)
        columns = _columns(current.ps)
        neighbours = {j: set() for j in variables}
        for u, v in current.ps.b:
            neighbours[u].add(v)
            neighbours[v].add(u)
        best_L, best_pair, best_form = current.L, None, None
        for choice in choices:
            if time_budget is not None and time.perf_counter() - start >= time_budget:
                break
            ji, ki = divmod(choice, n - 1)
            if ki >= ji:
                ki += 1
            j, k = variables[ji], variables[ki]
            if columns[j] == columns[k]:
                candidate = xor_change(current, j, k)
                if candidate.L < best_L:
                    best_L, best_pair, best_form = candidate.L, (j, k), candidate
            else:
                new_L = current.L + _unreduced_delta(current.ps, j, k, neighbours)
                if new_L < best_L:
                    best_L, best_pair, best_form = new_L, (j, k), None
        if best_pair is None:
            break
        candidate = best_form if best_form is not None else xor_change(current, *best_pair)
        if candidate.L >= current.L:
            raise AssertionError("local-search score disagrees with reduced form")
        current = candidate
    return current
