"""Run sampling, clustering, uncertainty scoring and grading over a QA slice.

Results are written to a JSONL file one question at a time, and questions already in the
file with a non-empty primary answer are skipped, so an interrupted run can simply be
restarted. A question that fails is logged to a failures file and retried on later runs,
up to MAX_QUESTION_ATTEMPTS times.
"""

import argparse
import json
import logging
import random
import time
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
from src import groq_chat
from src.groq_chat import IncompleteResponse, RateLimitExhausted
from src.sampling.sampler import sample_with_metadata

RESULTS_PATH = Path("results/metrics/pipeline_results.jsonl")
FAILURES_PATH = Path("results/metrics/pipeline_failures.jsonl")
MAX_QUESTION_ATTEMPTS = 3
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


def is_done(record: dict) -> bool:
    return bool((record.get("primary_answer") or "").strip())


def failure_counts() -> dict[str, int]:
    counts = {}
    for f in load_results(FAILURES_PATH):
        counts[f["id"]] = counts.get(f["id"], 0) + 1
    return counts


def log_failure(qid: str, reason: str, finish_reason: str | None, attempt: int) -> None:
    logger.warning("Failed %s (attempt %d of %d): %s", qid, attempt, MAX_QUESTION_ATTEMPTS, reason)
    with FAILURES_PATH.open("a") as f:
        f.write(json.dumps({"id": qid, "reason": reason, "finish_reason": finish_reason,
                            "attempt": attempt}) + "\n")


def save_record(record: dict) -> None:
    """Append the record, or replace the existing line for the same question in place.

    Other lines are copied verbatim, so no other record changes.
    """
    line = json.dumps(record) + "\n"
    existing = RESULTS_PATH.read_text().splitlines(keepends=True) if RESULTS_PATH.exists() else []
    ids = [json.loads(l)["id"] if l.strip() else None for l in existing]
    if record["id"] not in ids:
        with RESULTS_PATH.open("a") as f:
            f.write(line)
        return
    existing[ids.index(record["id"])] = line
    tmp = RESULTS_PATH.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(existing))
    tmp.replace(RESULTS_PATH)


def process_item(item: QAItem) -> tuple[dict | None, dict | None]:
    """Return (record, None) for a finished question, or (None, failure) if it failed.

    A failure has a "reason" and the "finish_reason" of the last failed call, if known.
    """
    sampled, sample_failures = sample_with_metadata(item.question, N_SAMPLES, SAMPLE_TEMPERATURE)
    samples = [s.text for s in sampled]
    if len(samples) < MIN_SAMPLES:
        last = sample_failures[-1] if sample_failures else None
        return None, {"reason": f"only {len(samples)} of {N_SAMPLES} samples came back"
                                + (f" (last error: {last.error})" if last else ""),
                      "finish_reason": last.finish_reason if last else None}
    primary, primary_failures = sample_with_metadata(item.question, 1, 0.0)
    if not primary:
        last = primary_failures[-1]
        return None, {"reason": f"primary answer failed twice ({last.error})",
                      "finish_reason": last.finish_reason}
    primary_answer = primary[0].text

    cluster_ids = cluster_answers(item.question, samples)
    try:
        judge_label, judge_raw = judge_answer(item.question, item.gold_answers, primary_answer)
    except RateLimitExhausted:
        raise
    except Exception as e:
        return None, {"reason": f"judge failed ({e})",
                      "finish_reason": e.finish_reason if isinstance(e, IncompleteResponse) else None}

    return {
        "id": item.id,
        "question": item.question,
        "gold_answers": item.gold_answers,
        "primary_answer": primary_answer,
        "primary_finish_reason": primary[0].finish_reason,
        "primary_completion_tokens": primary[0].completion_tokens,
        "samples": samples,
        "samples_finish_reason": [s.finish_reason for s in sampled],
        "samples_completion_tokens": [s.completion_tokens for s in sampled],
        "cluster_ids": cluster_ids,
        "n_clusters": len(set(cluster_ids)),
        "semantic_entropy": semantic_entropy(cluster_ids),
        "naive_entropy": naive_entropy(samples),
        "normalized_string_entropy": normalized_string_entropy(samples),
        "agreement_rate": agreement_rate(cluster_ids),
        "judge_label": judge_label,
        "judge_raw": judge_raw,
        "alias_match": alias_match(item.gold_answers, primary_answer),
    }, None


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
    parser.add_argument("--ids", nargs="+", metavar="ID",
                        help="run only these question ids (they must be in the --n_questions slice)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    for noisy in ("httpx", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    start = time.monotonic()
    items = load_triviaqa(args.n_questions, args.seed)
    if args.ids:
        missing = set(args.ids) - {item.id for item in items}
        if missing:
            raise SystemExit(f"Not in the {args.n_questions}-question slice: {', '.join(sorted(missing))}")
        items = [item for item in items if item.id in set(args.ids)]
    done = {r["id"] for r in load_results() if is_done(r)}
    failures = failure_counts()
    gave_up = [item.id for item in items if item.id not in done and failures.get(item.id, 0) >= MAX_QUESTION_ATTEMPTS]
    todo = [item for item in items if item.id not in done and item.id not in gave_up]
    logger.info("%d questions selected, %d already done, %d given up after %d failed attempts, %d to run.",
                len(items), len(items) - len(todo) - len(gave_up), len(gave_up), MAX_QUESTION_ATTEMPTS, len(todo))
    if gave_up:
        logger.warning("Not retrying (see %s): %s", FAILURES_PATH, ", ".join(gave_up))

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(todo, 1):
        logger.info("[%d/%d] %s", i, len(todo), item.question)
        try:
            record, failure = process_item(item)
        except RateLimitExhausted as e:
            logger.error("Stopping: %s. Rerun later to resume.", e)
            break
        if record is not None:
            save_record(record)
        else:
            log_failure(item.id, failure["reason"], failure["finish_reason"], failures.get(item.id, 0) + 1)

    selected_ids = {item.id for item in items}
    print_review([r for r in load_results() if r["id"] in selected_ids], args.seed)

    elapsed = time.monotonic() - start
    print(f"\nElapsed: {elapsed / 60:.1f} min. API calls: {groq_chat.api_calls} "
          f"({groq_chat.rate_limited_calls} rate-limited and retried).")


if __name__ == "__main__":
    main()
