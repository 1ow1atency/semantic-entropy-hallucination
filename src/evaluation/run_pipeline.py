"""Run sampling, clustering, uncertainty scoring and grading over a QA slice.

Results are appended to a JSONL file one question at a time, and questions already
in the file are skipped, so an interrupted run can simply be restarted.
"""

import argparse
import json
import logging
import random
from pathlib import Path

from src.clustering.entailment import cluster_answers
from src.entropy.semantic_entropy import (
    agreement_rate,
    naive_entropy,
    normalized_string_entropy,
    semantic_entropy,
)
from src.evaluation.data_loading import QAItem, load_triviaqa
from src.evaluation.judge import alias_match, judge_answer
from src.groq_chat import RateLimitExhausted
from src.sampling.sampler import sample_answers

RESULTS_PATH = Path("results/metrics/pipeline_results.jsonl")
N_SAMPLES = 10
MIN_SAMPLES = 8
SAMPLE_TEMPERATURE = 1.0
N_REVIEW = 15

logger = logging.getLogger(__name__)


def load_results(path: Path = RESULTS_PATH) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def process_item(item: QAItem) -> dict | None:
    """Return the result record for one question, or None if it has to be skipped."""
    samples = sample_answers(item.question, N_SAMPLES, SAMPLE_TEMPERATURE)
    if len(samples) < MIN_SAMPLES:
        logger.warning("Skipping %s: only %d of %d samples came back.", item.id, len(samples), N_SAMPLES)
        return None
    primary = sample_answers(item.question, 1, 0.0)
    if not primary:
        logger.warning("Skipping %s: the primary answer failed.", item.id)
        return None
    primary_answer = primary[0]

    cluster_ids = cluster_answers(item.question, samples)
    try:
        judge_label, judge_raw = judge_answer(item.question, item.gold_answers, primary_answer)
    except RateLimitExhausted:
        raise
    except Exception as e:
        logger.warning("Skipping %s: judge failed (%s).", item.id, e)
        return None

    return {
        "id": item.id,
        "question": item.question,
        "gold_answers": item.gold_answers,
        "primary_answer": primary_answer,
        "samples": samples,
        "cluster_ids": cluster_ids,
        "n_clusters": len(set(cluster_ids)),
        "semantic_entropy": semantic_entropy(cluster_ids),
        "naive_entropy": naive_entropy(samples),
        "normalized_string_entropy": normalized_string_entropy(samples),
        "agreement_rate": agreement_rate(cluster_ids),
        "judge_label": judge_label,
        "judge_raw": judge_raw,
        "alias_match": alias_match(item.gold_answers, primary_answer),
    }


def print_review(records: list[dict], seed: int) -> None:
    """Print a random sample of graded items and how often the judge and alias labels agree."""
    if not records:
        print("No results to review.")
        return
    rng = random.Random(seed)
    for r in rng.sample(records, min(N_REVIEW, len(records))):
        print(f"\nQ: {r['question']}")
        print(f"  gold:    {r['gold_answers'][0]}")
        print(f"  primary: {r['primary_answer']}")
        print(f"  judge: {r['judge_label']}   alias match: {r['alias_match']}")

    # NOT_ATTEMPTED has no alias-match counterpart, so compare only attempted answers.
    attempted = [r for r in records if r["judge_label"] != "NOT_ATTEMPTED"]
    agree = sum((r["judge_label"] == "CORRECT") == r["alias_match"] for r in attempted)
    print(f"\nJudge vs alias-match agreement: {agree}/{len(attempted)} attempted answers"
          + (f" ({agree / len(attempted):.0%})" if attempted else "")
          + f"; {len(records) - len(attempted)} NOT_ATTEMPTED excluded.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_questions", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    for noisy in ("httpx", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    items = load_triviaqa(args.n_questions, args.seed)
    done = {r["id"] for r in load_results()}
    todo = [item for item in items if item.id not in done]
    logger.info("%d questions in slice, %d already done, %d to run.", len(items), len(items) - len(todo), len(todo))

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(todo, 1):
        logger.info("[%d/%d] %s", i, len(todo), item.question)
        try:
            record = process_item(item)
        except RateLimitExhausted as e:
            logger.error("Stopping: %s. Rerun later to resume.", e)
            break
        if record is not None:
            with RESULTS_PATH.open("a") as f:
                f.write(json.dumps(record) + "\n")

    slice_ids = {item.id for item in items}
    print_review([r for r in load_results() if r["id"] in slice_ids], args.seed)


if __name__ == "__main__":
    main()
