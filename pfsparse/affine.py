"""Affine functions over F2 on variables z_0, z_1, ... (bitset representation)."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Affine:
    """l(z) = XOR_{j in mask} z_j  XOR  const."""
    mask: int = 0
    const: int = 0

    @staticmethod
    def var(j: int) -> "Affine":
        return Affine(1 << j, 0)

    @staticmethod
    def constant(b: int) -> "Affine":
        return Affine(0, b & 1)

    def __xor__(self, other: "Affine") -> "Affine":
        return Affine(self.mask ^ other.mask, self.const ^ other.const)

    def flip(self) -> "Affine":
        return Affine(self.mask, self.const ^ 1)

    def weight(self) -> int:
        return bin(self.mask).count("1")

    def support(self) -> list[int]:
        m, out, j = self.mask, [], 0
        while m:
            if m & 1:
                out.append(j)
            m >>= 1
            j += 1
        return out

    def contains(self, j: int) -> bool:
        return bool((self.mask >> j) & 1)

    def evaluate(self, z: int) -> int:
        """Value at the assignment whose j-th bit is z_j."""
        return (bin(self.mask & z).count("1") + self.const) & 1

    def substitute(self, p: int, mu: "Affine") -> "Affine":
        """Replace z_p by the affine function mu (mu must not contain p)."""
        if not self.contains(p):
            return self
        rest = Affine(self.mask & ~(1 << p), self.const)
        return rest ^ mu
