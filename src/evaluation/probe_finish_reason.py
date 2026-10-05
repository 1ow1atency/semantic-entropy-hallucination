"""Probe how often answer generation stops on the token limit (finish_reason="length").

Uses the sampler's current model, prompt and settings, with no retry on "length". Probes the
NOT_ATTEMPTED questions from the pipeline results plus the first other questions of the slice,
and records every call (10 samples plus 1 primary answer per question) to its own JSONL file.
A question counts as done only when all its calls succeeded; otherwise a rerun probes it again.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.evaluation.run_pipeline import (
    N_SAMPLES,
    RESULTS_PATH,
    SAMPLE_TEMPERATURE,
    load_results,
)
from src.groq_chat import RateLimitExhausted, chat
from src.sampling.sampler import MODEL, SYSTEM_PROMPT

PROBE_PATH = Path("results/metrics/probe_finish_reason.jsonl")
N_QUESTIONS = 20

logger = logging.getLogger(__name__)


def probe_call(record: dict, kind: str, index: int, temperature: float) -> dict:
    row = {"id": record["id"], "kind": kind, "index": index, "temperature": temperature,
           "not_attempted_in_pipeline": record["judge_label"] == "NOT_ATTEMPTED"}
    try:
        result = chat(MODEL, SYSTEM_PROMPT, record["question"], temperature)
        row.update(content=result.content, finish_reason=result.finish_reason,
                   completion_tokens=result.completion_tokens, error=None)
    except RateLimitExhausted:
        raise
    except Exception as e:
        row.update(content=None, finish_reason=None, completion_tokens=None, error=str(e))
    return row


def summarize(rows: list[dict]) -> None:
    ok = [r for r in rows if r["error"] is None]
    print(f"\nCalls: {len(rows)} ({len(rows) - len(ok)} errors)")
    for kind in ("all", "sample", "primary"):
        subset = [r for r in ok if kind == "all" or r["kind"] == kind]
        if not subset:
            continue
        tokens = np.array([r["completion_tokens"] for r in subset if r["completion_tokens"] is not None])
        n_length = sum(r["finish_reason"] == "length" for r in subset)
        n_empty = sum(not (r["content"] or "").strip() for r in subset)
        print(f"  {kind:<8} n={len(subset):<4} length: {n_length} ({n_length / len(subset):.1%})  "
              f"empty content: {n_empty}  completion tokens mean {tokens.mean():.0f}, max {tokens.max()}")
    reasons = sorted({str(r["finish_reason"]) for r in ok})
    print(f"  finish_reason values seen: {', '.join(reasons)}")

    print("\nPreviously NOT_ATTEMPTED questions:")
    for qid in sorted({r["id"] for r in rows if r["not_attempted_in_pipeline"]}):
        q_rows = [r for r in ok if r["id"] == qid]
        primary = [r for r in q_rows if r["kind"] == "primary"]
        n_length = sum(r["finish_reason"] == "length" for r in q_rows)
        p = primary[0] if primary else None
        print(f"  {qid}: primary finish_reason={p and p['finish_reason']}, "
              f"tokens={p and p['completion_tokens']}, content={p and p['content']!r}; "
              f"{n_length}/{len(q_rows)} calls hit length")


def load_probe_rows() -> list[dict]:
    if not PROBE_PATH.exists():
        return []
    with PROBE_PATH.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def completed_ids(rows: list[dict]) -> set[str]:
    """Questions whose calls all succeeded; questions with any errored call are probed again."""
    calls, failed = {}, set()
    for r in rows:
        calls[r["id"]] = calls.get(r["id"], 0) + 1
        if r["error"] is not None:
            failed.add(r["id"])
    return {qid for qid, n in calls.items() if n == N_SAMPLES + 1 and qid not in failed}


def save_question_rows(new_rows: list[dict]) -> None:
    """Write one question's rows, replacing any rows left from an earlier attempt at it."""
    qid = new_rows[0]["id"]
    kept = [r for r in load_probe_rows() if r["id"] != qid]
    tmp = PROBE_PATH.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        f.writelines(json.dumps(row) + "\n" for row in kept + new_rows)
    tmp.replace(PROBE_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="+", metavar="ID",
                        help="probe only these question ids from the pipeline results")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    records = load_results(RESULTS_PATH)
    not_attempted = [r for r in records if r["judge_label"] == "NOT_ATTEMPTED"]
    others = [r for r in records if r["judge_label"] != "NOT_ATTEMPTED"]
    if args.ids:
        by_id = {r["id"]: r for r in records}
        missing = [qid for qid in args.ids if qid not in by_id]
        if missing:
            raise SystemExit(f"Not in {RESULTS_PATH}: {', '.join(missing)}")
        chosen = [by_id[qid] for qid in args.ids]
    else:
        chosen = (not_attempted + others)[:N_QUESTIONS]

    done = completed_ids(load_probe_rows())
    PROBE_PATH.parent.mkdir(parents=True, exist_ok=True)
    for i, record in enumerate(chosen, 1):
        if record["id"] in done:
            continue
        logger.info("[%d/%d] %s", i, len(chosen), record["question"])
        try:
            rows = [probe_call(record, "sample", j, SAMPLE_TEMPERATURE) for j in range(N_SAMPLES)]
            rows.append(probe_call(record, "primary", 0, 0.0))
        except RateLimitExhausted as e:
            logger.error("Stopping: %s. Rerun later to resume.", e)
            break
        save_question_rows(rows)

    summarize(load_probe_rows())


if __name__ == "__main__":
    main()
