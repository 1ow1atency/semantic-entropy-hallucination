import itertools
import unittest

import numpy as np

from src.entropy.semantic_entropy import semantic_entropy
from src.evaluation.analyze import round_entropy_scores, score_arrays, selective_curve


def record(entropy: float, correct: bool) -> dict:
    return {"semantic_entropy": entropy, "naive_entropy": entropy, "normalized_string_entropy": entropy,
            "agreement_rate": 0.5, "judge_label": "CORRECT" if correct else "INCORRECT"}


class FloatTieTest(unittest.TestCase):
    def setUp(self):
        # 0.1 + 0.2 != 0.3 in floating point, but the two are meant to be the same value.
        self.assertNotEqual(0.1 + 0.2, 0.3)
        self.records = round_entropy_scores([record(0.1 + 0.2, True), record(0.3, False), record(0.0, True)])
        self.correct = np.array([True, False, True])

    def test_tied_in_auroc_input(self):
        for key in ("semantic_entropy", "naive_entropy", "normalized_string_entropy"):
            scores = score_arrays(self.records)[key]
            self.assertEqual(scores[0], scores[1], key)

    def test_tied_in_selective_curve(self):
        curve = selective_curve(self.correct, score_arrays(self.records)["semantic_entropy"])
        # One point for the exact zero, then both tied questions are kept together.
        self.assertEqual([round(c, 6) for _, c, _ in curve], [round(1 / 3, 6), 1.0])

    def test_unrounded_scores_would_split_the_tie(self):
        raw = np.array([0.1 + 0.2, 0.3, 0.0])
        self.assertEqual(len(selective_curve(self.correct, raw)), 3)


class ClusterOrderTest(unittest.TestCase):
    def test_same_cluster_sizes_give_identical_rounded_entropy(self):
        sizes = [2, 2, 2, 1, 1, 1, 1]
        raw = set()
        for order in set(itertools.permutations(sizes)):
            cluster_ids = [cid for cid, size in enumerate(order) for _ in range(size)]
            raw.add(semantic_entropy(cluster_ids))
        self.assertGreater(len(raw), 1)  # summation order changes the last bits
        rounded = {r["semantic_entropy"] for r in round_entropy_scores([record(e, True) for e in raw])}
        self.assertEqual(len(rounded), 1)


if __name__ == "__main__":
    unittest.main()
