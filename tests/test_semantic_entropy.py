import math
import unittest

from src.entropy.semantic_entropy import (
    agreement_rate,
    naive_entropy,
    normalize_answer,
    normalized_string_entropy,
    semantic_entropy,
)


def ids_from_sizes(sizes: list[int]) -> list[int]:
    """Build cluster ids with the given cluster sizes, e.g. [2, 1] -> [0, 0, 1]."""
    return [cid for cid, size in enumerate(sizes) for _ in range(size)]


class SemanticEntropyTest(unittest.TestCase):
    def test_single_cluster_is_zero(self):
        self.assertAlmostEqual(semantic_entropy(ids_from_sizes([10])), 0.0)

    def test_two_equal_clusters_is_ln2(self):
        self.assertAlmostEqual(semantic_entropy(ids_from_sizes([5, 5])), math.log(2))
        self.assertAlmostEqual(semantic_entropy(ids_from_sizes([5, 5])), 0.6931, places=4)

    def test_all_distinct_is_ln10(self):
        self.assertAlmostEqual(semantic_entropy(list(range(10))), math.log(10))
        self.assertAlmostEqual(semantic_entropy(list(range(10))), 2.3026, places=4)

    def test_mixed_clusters(self):
        self.assertAlmostEqual(semantic_entropy(ids_from_sizes([5, 2, 1, 1, 1])), 1.3592, places=4)

    def test_single_sample(self):
        self.assertEqual(semantic_entropy([0]), 0.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            semantic_entropy([])


class NaiveEntropyTest(unittest.TestCase):
    def test_surface_variants_count_as_different(self):
        self.assertAlmostEqual(naive_entropy(["Paris"] * 5 + ["Paris."] * 5), math.log(2))

    def test_identical_strings_is_zero(self):
        self.assertAlmostEqual(naive_entropy(["Paris"] * 10), 0.0)

    def test_single_sample(self):
        self.assertEqual(naive_entropy(["Paris"]), 0.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            naive_entropy([])


class NormalizedStringEntropyTest(unittest.TestCase):
    def test_surface_variants_collapse(self):
        self.assertAlmostEqual(normalized_string_entropy(["Paris", "paris.", "The Paris"]), 0.0)

    def test_different_answers_stay_distinct(self):
        self.assertAlmostEqual(normalized_string_entropy(["Paris", "the paris", "Lyon", "Lyon!"]), math.log(2))

    def test_articles_only_removed_as_whole_words(self):
        self.assertEqual(normalize_answer("  The   Anthem of  a Nation. "), "anthem of nation")

    def test_single_sample(self):
        self.assertEqual(normalized_string_entropy(["Paris"]), 0.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            normalized_string_entropy([])


class AgreementRateTest(unittest.TestCase):
    def test_values(self):
        self.assertEqual(agreement_rate(ids_from_sizes([10])), 1.0)
        self.assertEqual(agreement_rate(ids_from_sizes([5, 5])), 0.5)
        self.assertEqual(agreement_rate(list(range(10))), 0.1)
        self.assertEqual(agreement_rate(ids_from_sizes([5, 2, 1, 1, 1])), 0.5)

    def test_single_sample(self):
        self.assertEqual(agreement_rate([0]), 1.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            agreement_rate([])


if __name__ == "__main__":
    unittest.main()
