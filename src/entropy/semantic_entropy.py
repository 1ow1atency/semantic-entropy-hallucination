"""Uncertainty scores over a set of sampled answers: semantic entropy and baselines."""

import math
import re
import string
from collections import Counter
from collections.abc import Hashable, Sequence


def _entropy(labels: Sequence[Hashable]) -> float:
    """Shannon entropy (natural log) of the empirical distribution of labels."""
    if not labels:
        raise ValueError("Cannot compute entropy of an empty list of samples.")
    n = len(labels)
    return sum(-(count / n) * math.log(count / n) for count in Counter(labels).values())


def semantic_entropy(cluster_ids: list[int]) -> float:
    """Discrete semantic entropy: entropy over the proportion of samples in each meaning cluster.

    Uses cluster frequencies rather than token probabilities, which Groq's API doesn't expose.
    A single sample gives 0.0; an empty list raises ValueError.
    """
    return _entropy(cluster_ids)


def naive_entropy(answers: list[str]) -> float:
    """Entropy over the raw answer strings, so "Paris" and "Paris." count as different answers."""
    return _entropy(answers)


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation and the articles "the", "a", "an", and collapse whitespace."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    return " ".join(text.split())


def normalized_string_entropy(answers: list[str]) -> float:
    """Entropy over normalized answer strings, so "Paris", "paris." and "The Paris" count as one answer."""
    return _entropy([normalize_answer(a) for a in answers])


def agreement_rate(cluster_ids: list[int]) -> float:
    """Fraction of samples in the largest cluster (1.0 means every sample agrees)."""
    if not cluster_ids:
        raise ValueError("Cannot compute agreement rate of an empty list of samples.")
    return Counter(cluster_ids).most_common(1)[0][1] / len(cluster_ids)


if __name__ == "__main__":
    from src.clustering.entailment import cluster_answers
    from src.sampling.sampler import sample_answers

    questions = {
        "France": "What is the capital of France?",
        "Funafuti": (
            "In what year was the Tuvaluan town of Funafuti first surveyed "
            "by the United States Exploring Expedition?"
        ),
    }
    rows = []
    for label, question in questions.items():
        answers = sample_answers(question)
        ids = cluster_answers(question, answers)
        rows.append((label, len(answers), len(set(ids)), semantic_entropy(ids),
                     naive_entropy(answers), agreement_rate(ids)))

    print(f"\n{'Question':<10} {'Samples':>7} {'Clusters':>8} {'Semantic H':>10} {'Naive H':>8} {'Agreement':>9}")
    for label, n, k, se, ne, ar in rows:
        print(f"{label:<10} {n:>7} {k:>8} {se:>10.4f} {ne:>8.4f} {ar:>9.2f}")
