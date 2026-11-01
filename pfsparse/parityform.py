"""Parity forms: path sums without constraints in which every variable occurs in some parity."""
from __future__ import annotations
from dataclasses import dataclass

from .pathsum import PathSum


@dataclass
class ParityForm:
    ps: PathSum

    @staticmethod
    def from_pathsum(ps: PathSum, check: bool = True) -> "ParityForm":
        """Wrap a fully reduced path sum. Constant parities are folded into c."""
        ps = ps.copy()
        kept = []
        for p in ps.parities:
            if p.mask == 0:
                ps.add_c(p.const)
            else:
                kept.append(p)
        ps.parities = kept
        if check and not ps.zero:
            if ps.constraints:
                raise ValueError("not reduced: constraints remain")
            free = [j for j in ps.alive if not ps.in_parity(j)]
            if free:
                raise ValueError(f"not reduced: variables {free[:5]} occur in no parity")
        return ParityForm(ps)

    # sizes (DESIGN.md section 1)
    @property
    def r(self) -> int:
        return len(self.ps.alive)

    @property
    def t(self) -> int:
        return len(self.ps.parities)

    @property
    def support_sum(self) -> int:
        return sum(p.weight() for p in self.ps.parities)

    @property
    def q_size(self) -> int:
        return len(self.ps.a) + len(self.ps.b)

    @property
    def L(self) -> int:
        return self.r + self.t + self.support_sum + self.q_size

    def parity_matrix_rank(self) -> int:
        """Rank over F2 of the t x r coefficient matrix of the parities."""
        rows, rank = [p.mask for p in self.ps.parities], 0
        for bit in sorted(self.ps.alive):
            piv = next((i for i in range(rank, len(rows)) if (rows[i] >> bit) & 1), None)
            if piv is None:
                continue
            rows[rank], rows[piv] = rows[piv], rows[rank]
            for i in range(len(rows)):
                if i != rank and (rows[i] >> bit) & 1:
                    rows[i] ^= rows[rank]
            rank += 1
        return rank

    def stats(self) -> dict:
        return {"r": self.r, "t": self.t, "support_sum": self.support_sum, "q_size": self.q_size,
                "L": self.L, "parity_rank": self.parity_matrix_rank(),
                "half_log2": self.ps.half_log2, "zero": self.ps.zero}

    def value(self) -> complex:
        return self.ps.evaluate_bruteforce()
