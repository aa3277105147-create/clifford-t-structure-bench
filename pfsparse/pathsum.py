"""Path sums  kappa * sum_z omega^{P(z)} * prod_k [alpha_k(z) = 0],
P(z) = c + 2 Q(z) + sum_i [lambda_i(z)]  (mod 8),
Q(z) = sum_j a_j z_j + 2 sum_{j<k} b_jk z_j z_k  (mod 4),  kappa = 2^{half_log2 / 2}.
See DESIGN.md section 1 and 2.1.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import itertools
import numpy as np

from .affine import Affine

W8 = np.exp(1j * np.pi / 4)


@dataclass
class PathSum:
    nvars: int = 0                                   # next fresh variable index
    alive: set = field(default_factory=set)          # indices of variables still summed over
    c: int = 0                                       # mod 8
    half_log2: int = 0                               # kappa = 2^{half_log2/2}
    a: dict = field(default_factory=dict)            # j -> a_j in Z4 (zero entries removed)
    b: set = field(default_factory=set)              # {(j, k)} with j < k, b_jk = 1
    parities: list = field(default_factory=list)     # list[Affine]
    constraints: list = field(default_factory=list)  # list[Affine], each must equal 0
    zero: bool = False                               # True once the sum is known to vanish

    # ---------------------------------------------------------------- bookkeeping (implemented)
    def new_var(self) -> int:
        j = self.nvars
        self.nvars += 1
        self.alive.add(j)
        return j

    def new_param(self) -> int:
        """A symbolic parameter (e.g. an output bit y_q): a variable index that is NOT summed over,
        so it is kept out of self.alive.  Used by reduce.find_support_output (DESIGN.md 2.4)."""
        j = self.nvars
        self.nvars += 1
        return j

    def add_c(self, v: int) -> None:
        self.c = (self.c + v) % 8

    def add_a(self, j: int, v: int) -> None:
        val = (self.a.get(j, 0) + v) % 4
        if val:
            self.a[j] = val
        else:
            self.a.pop(j, None)

    def toggle_b(self, j: int, k: int) -> None:
        if j == k:
            raise ValueError("b_jj is not a quadratic term; use add_a(j, 2) for 4 z_j in P")
        e = (min(j, k), max(j, k))
        if e in self.b:
            self.b.remove(e)
        else:
            self.b.add(e)

    def neighbours(self, j: int) -> list[int]:
        return [k for e in self.b if j in e for k in e if k != j]

    def in_parity(self, j: int) -> bool:
        return any(p.contains(j) for p in self.parities)

    def in_constraint(self, j: int) -> bool:
        return any(a.contains(j) for a in self.constraints)

    def copy(self) -> "PathSum":
        return PathSum(self.nvars, set(self.alive), self.c, self.half_log2, dict(self.a), set(self.b),
                       list(self.parities), list(self.constraints), self.zero)

    # ---------------------------------------------------------------- ground truth (implemented)
    def phase(self, z: int) -> int:
        """P(z) mod 8 for an assignment z (bit j = z_j)."""
        q = sum(v * ((z >> j) & 1) for j, v in self.a.items())
        q += 2 * sum(((z >> j) & 1) * ((z >> k) & 1) for j, k in self.b)
        return (self.c + 2 * (q % 4) + sum(p.evaluate(z) for p in self.parities)) % 8

    def evaluate_bruteforce(self) -> complex:
        """Numerical value by enumeration (floating point; use only for small instances)."""
        if self.zero:
            return 0.0
        vs = sorted(self.alive)
        if len(vs) > 22:
            raise ValueError("too many variables for brute force")
        total = 0j
        for bits in itertools.product((0, 1), repeat=len(vs)):
            z = sum(bit << j for bit, j in zip(bits, vs))
            if all(con.evaluate(z) == 0 for con in self.constraints):
                total += W8 ** self.phase(z)
        return (2.0 ** (self.half_log2 / 2)) * total

    def evaluate_exact(self, max_variables: int = 22) -> tuple[int, tuple[int, int, int, int]]:
        """Return (d, a), meaning 2**(-d) * sum(a[j] * omega**j, j=0..3).

        This integer oracle is genuinely exact, unlike evaluate_bruteforce.
        Unsummed symbolic parameters, if any, are fixed to zero. Common powers
        of two are removed so equality of returned tuples is equality of values.
        """
        if self.zero:
            return (0, (0, 0, 0, 0))
        vs = sorted(self.alive)
        if len(vs) > max_variables:
            raise ValueError("too many variables for exact enumeration")
        counts = [0] * 8
        for bits in itertools.product((0, 1), repeat=len(vs)):
            z = sum(bit << j for bit, j in zip(bits, vs))
            if all(con.evaluate(z) == 0 for con in self.constraints):
                counts[self.phase(z)] += 1
        a = [counts[j] - counts[j + 4] for j in range(4)]
        if self.half_log2 & 1:
            # sqrt(2) = omega - omega**3 in Z[omega], omega**4 = -1.
            a0, a1, a2, a3 = a
            a = [a1 - a3, a0 + a2, a1 + a3, a2 - a0]
        exponent = self.half_log2 // 2
        if exponent >= 0:
            a = [v << exponent for v in a]
            d = 0
        else:
            d = -exponent
        if not any(a):
            return (0, (0, 0, 0, 0))
        while d and all(v % 2 == 0 for v in a):
            a = [v // 2 for v in a]
            d -= 1
        return (d, tuple(a))


def build_pathsum(circ, n: int, x: int, y: int) -> PathSum:
    """Construct the exact path-sum representation of <y|C|x>."""
    if not isinstance(y, int) or not 0 <= y < (1 << n):
        raise ValueError("y must be an n-bit computational-basis index")
    ps, wires = _build_pathsum_wires(circ, n, x)
    ps.constraints.extend(wire ^ Affine.constant((y >> q) & 1)
                          for q, wire in enumerate(wires))
    return ps


def symbolic_output_pathsum(circ, n: int, x: int) -> tuple[PathSum, list[int]]:
    """Construct <y|C|x> with unsummed output parameters; return their indices."""
    ps, wires = _build_pathsum_wires(circ, n, x)
    outputs = [ps.new_param() for _ in range(n)]
    ps.constraints.extend(wire ^ Affine.var(j) for wire, j in zip(wires, outputs))
    return ps, outputs


def _add_affine_quadratic(ps: PathSum, ell: Affine, coefficient: int) -> None:
    """Add 2 * coefficient * [ell] to the phase modulo eight."""
    coefficient %= 4
    support = ell.support()
    ps.add_c(2 * coefficient * ell.const)
    signed = -coefficient if ell.const else coefficient
    for j in support:
        ps.add_a(j, signed)
    if coefficient & 1:
        for u, j in enumerate(support):
            for k in support[u + 1:]:
                ps.toggle_b(j, k)


def _add_four_product(ps: PathSum, left: Affine, right: Affine) -> None:
    """Add 4 * ([left] [right] mod 2), accumulating F2 duplicates."""
    ls, rs = left.support(), right.support()
    ps.add_c(4 * left.const * right.const)
    if left.const:
        for k in rs:
            ps.add_a(k, 2)
    if right.const:
        for j in ls:
            ps.add_a(j, 2)
    for j in ls:
        for k in rs:
            if j == k:
                ps.add_a(j, 2)
            else:
                ps.toggle_b(j, k)


def _build_pathsum_wires(circ, n: int, x: int) -> tuple[PathSum, list[Affine]]:
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    if not isinstance(x, int) or not 0 <= x < (1 << n):
        raise ValueError("x must be an n-bit computational-basis index")
    ps = PathSum()
    wires = [Affine.constant((x >> q) & 1) for q in range(n)]
    arity = {"H": 1, "X": 1, "Z": 1, "S": 1, "SDG": 1,
             "T": 1, "TDG": 1, "CNOT": 2, "CZ": 2}
    for gate, qubits in circ:
        if gate not in arity:
            raise ValueError(f"unsupported gate {gate!r}")
        if (len(qubits) != arity[gate] or len(set(qubits)) != len(qubits)
                or any(not isinstance(q, int) or q < 0 or q >= n for q in qubits)):
            raise ValueError(f"invalid operands for {gate}: {qubits!r}")
        q = qubits[0]
        ell = wires[q]
        if gate == "X":
            wires[q] = ell.flip()
        elif gate == "CNOT":
            wires[qubits[1]] = wires[qubits[1]] ^ ell
        elif gate == "H":
            j = ps.new_var()
            _add_four_product(ps, ell, Affine.var(j))
            wires[q] = Affine.var(j)
            ps.half_log2 -= 1
        elif gate == "T":
            ps.parities.append(ell)
        elif gate == "TDG":
            ps.parities.append(ell.flip())
            ps.add_c(-1)
        elif gate == "CZ":
            _add_four_product(ps, ell, wires[qubits[1]])
        else:
            _add_affine_quadratic(ps, ell, {"Z": 2, "S": 1, "SDG": -1}[gate])
    return ps, wires
