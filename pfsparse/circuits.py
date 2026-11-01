"""Circuit families over {H, X, CNOT, Z, S, SDG, T, TDG, CZ}. A circuit is a list of (gate, qubits)."""
from __future__ import annotations
import random

ONE_Q_CLIFF = ["H", "S", "SDG", "X", "Z"]
TWO_Q_CLIFF = ["CNOT", "CZ"]


def t_count(circ) -> int:
    return sum(1 for g, _ in circ if g in ("T", "TDG"))


def _rand_t(rng: random.Random) -> str:
    return rng.choice(["T", "TDG"])


def random_two_qubit_clifford(a: int, b: int, rng: random.Random, length: int = 4):
    """A short random word over {H, S, CNOT, CZ} on the pair (a, b)."""
    out = []
    for _ in range(length):
        g = rng.choice(["H", "S", "CNOT", "CZ"])
        if g in ("H", "S"):
            out.append((g, (rng.choice([a, b]),)))
        elif g == "CNOT":
            out.append(("CNOT", (a, b) if rng.random() < 0.5 else (b, a)))
        else:
            out.append(("CZ", (a, b)))
    return out


def random_clifford_t(n: int, s: int, t: int, rng: random.Random):
    """F0: s gates, t of them T/TDG at random positions, the rest random Cliffords."""
    pos = set(rng.sample(range(s), min(t, s)))
    circ = []
    for k in range(s):
        if k in pos:
            circ.append((_rand_t(rng), (rng.randrange(n),)))
        else:
            g = rng.choice(["H", "H"] + ONE_Q_CLIFF + (TWO_Q_CLIFF * 2 if n >= 2 else []))
            qs = tuple(rng.sample(range(n), 2)) if g in TWO_Q_CLIFF else (rng.randrange(n),)
            circ.append((g, qs))
    return circ


def phase_polynomial_circuit(n: int, t: int, n_cliff: int, rng: random.Random):
    """F1: H-layer, then a Hadamard-free word with t T/TDG and n_cliff CNOT/S/Z/X gates, then H-layer."""
    body = [(_rand_t(rng), (rng.randrange(n),)) for _ in range(t)]
    for _ in range(n_cliff):
        g = rng.choice(["CNOT", "CNOT", "S", "Z", "X"]) if n >= 2 else rng.choice(["S", "Z", "X"])
        qs = tuple(rng.sample(range(n), 2)) if g == "CNOT" else (rng.randrange(n),)
        body.append((g, qs))
    rng.shuffle(body)
    return [("H", (q,)) for q in range(n)] + body + [("H", (q,)) for q in range(n)]


def brickwork_1d(n: int, depth: int, t: int, rng: random.Random):
    """F2: depth layers of random 2-qubit Cliffords on alternating neighbours of a line, T gates at
    t random (layer, qubit) slots inserted after the corresponding layer.

    Each pair uses a four-gate word. Thus ``depth`` is the number of matching
    blocks, not elementary gate depth; elementary depth is at most 1+5*depth.
    """
    slots = rng.sample([(l, q) for l in range(depth) for q in range(n)], min(t, depth * n))
    by_layer = {}
    for l, q in slots:
        by_layer.setdefault(l, []).append(q)
    circ = [("H", (q,)) for q in range(n)]
    for l in range(depth):
        for a in range(l % 2, n - 1, 2):
            circ += random_two_qubit_clifford(a, a + 1, rng)
        for q in by_layer.get(l, []):
            circ.append((_rand_t(rng), (q,)))
    return circ


def brickwork_2d(lx: int, ly: int, depth: int, t: int, rng: random.Random):
    """F3: grid matching blocks cycle through four directions.

    As in F2, elementary gate depth is at most 1+5*depth, not ``depth``.
    """
    n = lx * ly
    qid = lambda i, j: i * ly + j
    patterns = []
    for parity in (0, 1):
        patterns.append([(qid(i, j), qid(i + 1, j)) for i in range(parity, lx - 1, 2) for j in range(ly)])
        patterns.append([(qid(i, j), qid(i, j + 1)) for i in range(lx) for j in range(parity, ly - 1, 2)])
    slots = rng.sample([(l, q) for l in range(depth) for q in range(n)], min(t, depth * n))
    by_layer = {}
    for l, q in slots:
        by_layer.setdefault(l, []).append(q)
    circ = [("H", (q,)) for q in range(n)]
    for l in range(depth):
        for a, b in patterns[l % 4]:
            circ += random_two_qubit_clifford(a, b, rng)
        for q in by_layer.get(l, []):
            circ.append((_rand_t(rng), (q,)))
    return circ


def qaoa_clifford_t(n: int, edges, p: int, rng: random.Random):
    """F4: QAOA-like gate word with independently signed pi/8 rotations.

    The implemented gates are CNOT-T^{+-1}-CNOT and H-T^{+-1}-H.
    They equal the associated ZZ and X rotations only up to global phase.
    Independently sampled signs make this QAOA-like, rather than ordinary
    shared-angle QAOA. Amplitudes refer to the exact gate word returned here.
    """
    circ = [("H", (q,)) for q in range(n)]
    for _ in range(p):
        for u, v in edges:
            circ += [("CNOT", (u, v)), (_rand_t(rng), (v,)), ("CNOT", (u, v))]
        for q in range(n):
            circ += [("H", (q,)), (_rand_t(rng), (q,)), ("H", (q,))]
    return circ


def random_regular_graph_edges(n: int, d: int, rng: random.Random):
    """A simple d-regular graph; never silently replace it by another ensemble."""
    if not isinstance(n, int) or not isinstance(d, int) or not 0 <= d < n:
        raise ValueError("require integer n,d with 0 <= d < n")
    if n * d % 2:
        raise ValueError("n*d must be even for a regular graph")
    import networkx as nx
    g = nx.random_regular_graph(d, n, seed=rng.randrange(1 << 30))
    return sorted(tuple(sorted(e)) for e in g.edges())


def sample_output(circ, n: int, x: int, rng: random.Random) -> int:
    """A basis string y lying on some classical path of the circuit: propagate x through the
    reversible gates and replace a wire by a fresh random bit at every Hadamard.  Such y satisfy
    all path-sum constraints for at least one path. Destructive interference
    can still make the amplitude zero; this is not a support certificate.
    """
    w = [(x >> q) & 1 for q in range(n)]
    for g, qs in circ:
        if g == "H":
            w[qs[0]] = rng.randrange(2)
        elif g == "X":
            w[qs[0]] ^= 1
        elif g == "CNOT":
            w[qs[1]] ^= w[qs[0]]
    return sum(b << q for q, b in enumerate(w))
