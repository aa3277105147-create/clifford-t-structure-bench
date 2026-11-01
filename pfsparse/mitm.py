"""Numerical split-and-Walsh evaluation in O*(2**ceil(n/2)) operations.

Works for arbitrary complex unary weights. It is not a new algorithm, a
lower-bound certificate, or a floating-point guarantee of relative accuracy.
"""
from __future__ import annotations
import numpy as np


def fwht(values):
    a = np.asarray(values, dtype=np.complex128).copy()
    n = a.size
    if a.ndim != 1 or n == 0 or n & (n - 1):
        raise ValueError("Walsh input length must be a positive power of two")
    h = 1
    while h < n:
        blocks = a.reshape(-1, 2 * h)
        left, right = blocks[:, :h].copy(), blocks[:, h:].copy()
        blocks[:, :h] = left + right
        blocks[:, h:] = left - right
        h *= 2
    return a


def mitm_value(t, adj, weights, *, max_half=22):
    from .biclique import _validate
    _validate(t, adj)
    if len(weights) != t or any(len(w) != 2 for w in weights):
        raise ValueError("one binary weight pair is required per vertex")
    left, right = t // 2, t - t // 2
    if right > max_half:
        raise ValueError("half-graph exceeds declared array budget")

    def half_vector(start, size):
        xs = np.arange(1 << size, dtype=np.int64)
        out = np.ones(1 << size, dtype=np.complex128)
        parity = np.zeros(1 << size, dtype=np.int8)
        for j in range(size):
            bj = (xs >> j) & 1
            out *= np.where(bj, weights[start + j][1], weights[start + j][0])
            for k in range(j + 1, size):
                if adj[start + j] & (1 << (start + k)):
                    parity ^= (bj & ((xs >> k) & 1)).astype(np.int8)
        return xs, out * (1 - 2 * parity)

    _, g = half_vector(left, right)
    transform = fwht(g)
    xs, f = half_vector(0, left)
    indices = np.zeros(xs.size, dtype=np.int64)
    for k in range(right):
        mask = adj[left + k] & ((1 << left) - 1)
        parity = np.zeros(xs.size, dtype=np.int64)
        while mask:
            low = mask & -mask
            parity ^= (xs >> (low.bit_length() - 1)) & 1
            mask ^= low
        indices |= parity << k
    return complex(np.sum(f * transform[indices]))
