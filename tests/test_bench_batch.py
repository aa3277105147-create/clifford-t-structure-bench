import cmath
import unittest

from pfsparse.bench.batch import evaluate_batch, generate_sign_vectors
from pfsparse.pathsum import build_pathsum


class BatchAmplitudeTests(unittest.TestCase):
    def setUp(self):
        self.task = {
            "id": "batch-phase-check",
            "family": "iqp",
            "n": 2,
            "circuit": [
                ["H", [0]], ["T", [0]], ["CNOT", [0, 1]],
                ["TDG", [1]], ["H", [1]], ["S", [0]],
            ],
            "x": 0,
            "y": 1,
            "global_phase": 0.37,
            "seed": 3,
            "metadata": {},
        }
        self.rows = [[1, -1], [-1, 1], [-1, -1]]

    def test_sign_vectors_are_seeded_and_start_with_circuit_fill(self):
        first = generate_sign_vectors(self.task, 8, seed=17)
        second = generate_sign_vectors(self.task, 8, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(first[0], [1, -1])
        self.assertEqual(len(first), 8)
        self.assertTrue(all(len(row) == 2 for row in first))

    def test_reused_and_independent_graph_match_exact_path_sum(self):
        independent = evaluate_batch(self.task, "independent", self.rows)
        shared = evaluate_batch(self.task, "shared_graph", self.rows)
        self.assertEqual(len(shared["amplitudes"]), len(self.rows))
        self.assertEqual(shared["metadata"]["unique_query_count"], len(self.rows))
        for row, a, b in zip(self.rows, independent["amplitudes"], shared["amplitudes"]):
            self.assertAlmostEqual(a[0], b[0], places=11)
            self.assertAlmostEqual(a[1], b[1], places=11)
            circuit = [
                ["T" if sign == 1 else "TDG", qs] if gate in ("T", "TDG") else [gate, qs]
                for (gate, qs), sign in zip(
                    self.task["circuit"],
                    iter_signs(self.task["circuit"], row),
                )
            ]
            exact = build_pathsum(
                [(gate, tuple(qs)) for gate, qs in circuit],
                self.task["n"], self.task["x"], self.task["y"],
            ).evaluate_exact()
            reference = (sum(
                coefficient * cmath.exp(1j * cmath.pi * index / 4)
                for index, coefficient in enumerate(exact[1])
            ) / (2 ** exact[0])) * cmath.exp(1j * self.task["global_phase"])
            self.assertAlmostEqual(a[0], reference.real, places=10)
            self.assertAlmostEqual(a[1], reference.imag, places=10)

    def test_quimb_cached_matches_shared_graph(self):
        shared = evaluate_batch(self.task, "shared_graph", self.rows)
        cached = evaluate_batch(self.task, "quimb_cached", self.rows, {"planning_repeats": 2})
        for a, b in zip(shared["amplitudes"], cached["amplitudes"]):
            self.assertAlmostEqual(a[0], b[0], places=9)
            self.assertAlmostEqual(a[1], b[1], places=9)
        self.assertEqual(cached["metadata"]["query_count"], len(self.rows))

    def test_mitm_half_size_limit_is_reported_as_unsupported(self):
        with self.assertRaises(NotImplementedError):
            evaluate_batch(self.task, "shared_graph", self.rows[:1], {"max_half": 0})


def iter_signs(circuit, signs):
    it = iter(signs)
    return [next(it) if gate in ("T", "TDG") else None for gate, _ in circuit]


if __name__ == "__main__":
    unittest.main()
