"""Exact R1-R3 reductions, including symbolic-output necessary conditions.

R1 preserves a constrained sum; substitute alone is an algebraic substitution,
not an identity between unconstrained sums with different numbers of variables.
"""
from __future__ import annotations

from .affine import Affine
from .pathsum import PathSum, symbolic_output_pathsum
from .parityform import ParityForm


def substitute(ps: PathSum, p: int, mu: Affine) -> None:
    """In place: replace z_p by the affine function mu (mu must not contain p) in parities,
    constraints and Q, then delete variable p.  This is the core of R1 and of R4.

    Deleting p preserves a sum when justified by a unique-extension constraint
    (R1), or when a fresh replacement variable makes the change bijective (R4).
    """
    if p not in ps.alive:
        raise ValueError("the substituted variable must be summed")
    if mu.contains(p):
        raise ValueError("replacement contains its own pivot")
    if mu.mask < 0 or mu.mask >> ps.nvars or mu.const not in (0, 1):
        raise ValueError("invalid affine replacement")
    support = mu.support()
    coefficient = ps.a.pop(p, 0)
    neighbours = ps.neighbours(p)
    ps.b = {e for e in ps.b if p not in e}
    ps.alive.remove(p)
    ps.parities = [term.substitute(p, mu) for term in ps.parities]
    ps.constraints = [con.substitute(p, mu) for con in ps.constraints]
    ps.add_c(2 * coefficient * mu.const)
    signed = -coefficient if mu.const else coefficient
    for j in support:
        ps.add_a(j, signed)
    if coefficient & 1:
        for u, j in enumerate(support):
            for k in support[u + 1:]:
                ps.toggle_b(j, k)
    for k in neighbours:
        for j in support:
            if j == k:
                ps.add_a(k, 2)
            else:
                ps.toggle_b(j, k)
        if mu.const:
            ps.add_a(k, 2)


def apply_constraint(ps: PathSum, idx: int, choose_pivot=None) -> None:
    """R1: resolve ps.constraints[idx] (remove it; set ps.zero if it reads 1 = 0).

    Pivots are restricted to alive (summed) variables. A parameter-only
    nonconstant condition must instead be retained by reduce_raw.
    choose_pivot, if given, is a callable (ps, candidates) -> pivot.
    """
    con = ps.constraints[idx]
    if not con.mask:
        ps.constraints.pop(idx)
        ps.zero |= bool(con.const)
        return
    candidates = [j for j in con.support() if j in ps.alive]
    if not candidates:
        raise ValueError("constraint has no summed pivot; retain it as a parameter condition")
    p = (choose_pivot(ps, candidates) if choose_pivot is not None
         else _select_variable(ps, candidates, "min_degree", prefer_unused=True))
    if p not in candidates:
        raise ValueError("pivot must be a summed variable in the constraint")
    ps.constraints.pop(idx)
    substitute(ps, p, con ^ Affine.var(p))


def gauss_eliminate(ps: PathSum, z: int) -> None:
    """R2: sum out a variable z that occurs in no parity and no constraint.

    This can increase the number of quadratic terms through clique fill.
    """
    if z not in ps.alive or ps.in_parity(z) or ps.in_constraint(z):
        raise ValueError("R2 requires a summed variable absent from parities and constraints")
    a = ps.a.pop(z, 0)
    neighbours = ps.neighbours(z)
    ps.b = {e for e in ps.b if z not in e}
    ps.alive.remove(z)
    if a in (0, 2):
        ps.half_log2 += 2
        con = Affine(sum(1 << j for j in neighbours), a // 2)
        if con.mask:
            ps.constraints.append(con)
        elif con.const:
            ps.zero = True
    else:
        ps.half_log2 += 1
        sign = -1 if a == 1 else 1
        ps.add_c(-sign)
        for j in neighbours:
            ps.add_a(j, sign)
        for u, j in enumerate(neighbours):
            for k in neighbours[u + 1:]:
                ps.toggle_b(j, k)


def reduce_pathsum(ps: PathSum, order: str = "min_degree") -> ParityForm:
    """Main loop of DESIGN.md 2.2: R1 until no constraints, R2 on variables outside parities,
    repeat; stop early if ps.zero.  order in {"natural", "min_degree", "min_fill"}.
    Works on a copy; returns ParityForm.from_pathsum(result).
    """
    return ParityForm.from_pathsum(reduce_raw(ps, order=order))


def _bits(mask: int):
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def _parity_mask(ps: PathSum) -> int:
    mask = 0
    for term in ps.parities:
        mask |= term.mask
    return mask


def _interaction_masks(ps: PathSum) -> dict[int, int]:
    """Bitset adjacency for Q edges and affine-scope cliques, including parameters."""
    adj = {j: 0 for j in ps.alive}
    for j, k in ps.b:
        adj[j] = adj.get(j, 0) | (1 << k)
        adj[k] = adj.get(k, 0) | (1 << j)
    for term in ps.parities + ps.constraints:
        if term.mask.bit_count() < 2:
            continue
        for j in _bits(term.mask):
            adj[j] = adj.get(j, 0) | (term.mask ^ (1 << j))
    return adj


def _select_variable(ps: PathSum, candidates, order: str, prefer_unused=False) -> int:
    candidates = sorted(candidates)
    if not candidates:
        raise ValueError("empty elimination candidate set")
    if prefer_unused:
        used = _parity_mask(ps)
        unused = [j for j in candidates if not ((used >> j) & 1)]
        if unused:
            candidates = unused
    if len(candidates) == 1 or order == "natural":
        return candidates[0]
    adjacency = _interaction_masks(ps)
    if order == "min_degree":
        return min(candidates, key=lambda j: (adjacency.get(j, 0).bit_count(), j))
    if order != "min_fill":
        raise ValueError(f"unknown elimination order {order!r}")
    def score(j):
        neighbours = adjacency.get(j, 0)
        # Each missing unordered edge is counted twice in this sum.
        missing = sum((neighbours & ~(adjacency.get(k, 0) | (1 << k))).bit_count()
                      for k in _bits(neighbours)) // 2
        return missing, neighbours.bit_count(), j
    return min(candidates, key=score)


def _referenced_mask(ps: PathSum) -> int:
    mask = 0
    for j in ps.a:
        mask |= 1 << j
    for j, k in ps.b:
        mask |= (1 << j) | (1 << k)
    for term in ps.parities + ps.constraints:
        mask |= term.mask
    return mask


def _fold_constants(ps: PathSum, preserve_parities: bool = False) -> None:
    parities = []
    for term in ps.parities:
        if term.mask or preserve_parities:
            parities.append(term)
        else:
            ps.add_c(term.const)
    ps.parities = parities
    constraints = []
    for con in ps.constraints:
        if con.mask:
            constraints.append(con)
        elif con.const:
            ps.zero = True
    ps.constraints = constraints


def reduce_raw(ps: PathSum, order: str = "min_degree",
               allow_parameters: bool = False,
               preserve_parities: bool = False) -> PathSum:
    """Reduce a copy, retaining parameter-only constraints in symbolic mode.

    Every iteration either removes a summed variable or removes a constant
    constraint, so termination is guaranteed independently of the heuristic.
    No monotonicity of L is asserted for individual Gaussian eliminations.
    preserve_parities retains constant factors and their order, so each list
    entry can be an independent formal slot rather than the numerical T phase.
    """
    if order not in ("natural", "min_degree", "min_fill"):
        raise ValueError(f"unknown elimination order {order!r}")
    out = ps.copy()
    alive_mask = sum(1 << j for j in out.alive)
    if not allow_parameters and _referenced_mask(out) & ~alive_mask:
        raise ValueError("unsummed parameters require allow_parameters=True")
    while not out.zero:
        _fold_constants(out, preserve_parities=preserve_parities)
        if out.zero:
            break
        alive_mask = sum(1 << j for j in out.alive)
        idx = next((i for i, con in enumerate(out.constraints) if con.mask & alive_mask), None)
        if idx is not None:
            choose = lambda obj, vs: _select_variable(obj, vs, order, prefer_unused=True)
            apply_constraint(out, idx, choose)
            continue
        free = alive_mask & ~_parity_mask(out)
        if free:
            j = _select_variable(out, _bits(free), order)
            gauss_eliminate(out, j)
            continue
        break
    _fold_constants(out, preserve_parities=preserve_parities)
    if out.zero:
        out.alive.clear()
        out.a.clear()
        out.b.clear()
        out.parities.clear()
        out.constraints.clear()
        out.c = out.half_log2 = 0
    return out


def find_support_output(circ, n: int, x: int, rng) -> int | None:
    """Sample y uniformly from the affine necessary conditions found symbolically.

    This is a candidate-output heuristic, not an exact sampler from C's output
    distribution and not a guarantee of nonzero amplitude. Remaining T-phase
    terms can cancel. Returns None only if those linear conditions are inconsistent.
    """
    ps, outputs = symbolic_output_pathsum(circ, n, x)
    reduced = reduce_raw(ps, allow_parameters=True)
    if reduced.zero:
        return None
    parameters = set(outputs)
    rows = {}
    for con in reduced.constraints:
        mask, rhs = con.mask, con.const
        if any(j not in parameters for j in _bits(mask)):
            raise RuntimeError("symbolic reduction left a summed-variable constraint")
        while mask:
            pivot = mask.bit_length() - 1
            if pivot not in rows:
                rows[pivot] = (mask, rhs)
                break
            other, value = rows[pivot]
            mask ^= other
            rhs ^= value
        if not mask and rhs:
            return None
    assignment = 0
    for j in outputs:
        if j not in rows and rng.randrange(2):
            assignment |= 1 << j
    # Highest-bit pivots have only lower coordinates to their right.
    for pivot in sorted(rows):
        mask, rhs = rows[pivot]
        value = (mask & assignment).bit_count() % 2 ^ rhs
        if value:
            assignment |= 1 << pivot
    return sum(((assignment >> j) & 1) << q for q, j in enumerate(outputs))


def compress_to_rank(pf: ParityForm) -> ParityForm:
    """R3: change variables (via basis.xor_change) so that parities depend on rank-many variables,
    then sum out the others with gauss_eliminate and reduce again.  Result has r <= t.
    """
    from .basis import compress_rank
    return compress_rank(pf)
