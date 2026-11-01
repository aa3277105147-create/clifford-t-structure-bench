"""Reference state-vector simulator (ground truth for tests). Bit q of a basis index = qubit q."""
from __future__ import annotations
import numpy as np

W8 = np.exp(1j * np.pi / 4)
GATES = ("H", "X", "CNOT", "Z", "S", "SDG", "T", "TDG", "CZ")


def statevector(circ, n: int, x: int) -> np.ndarray:
    N = 1 << n
    idx = np.arange(N)
    psi = np.zeros(N, complex)
    psi[x] = 1.0
    for g, qs in circ:
        bits = [(idx >> q) & 1 for q in qs]
        if g == "H":
            q = qs[0]
            i0, i1 = idx & ~(1 << q), idx | (1 << q)
            psi = (psi[i0] + np.where(bits[0] == 1, -1, 1) * psi[i1]) / np.sqrt(2)
        elif g == "X":
            psi = psi[idx ^ (1 << qs[0])]
        elif g == "CNOT":
            psi = psi[idx ^ (bits[0] << qs[1])]
        else:
            ph = {"Z": 4 * bits[0], "S": 2 * bits[0], "SDG": 6 * bits[0], "T": bits[0],
                  "TDG": 7 * bits[0], "CZ": 4 * bits[0] * (bits[1] if len(bits) > 1 else 0)}[g]
            psi = psi * np.power(W8, ph)
    return psi


def amplitude(circ, n: int, x: int, y: int) -> complex:
    return complex(statevector(circ, n, x)[y])
