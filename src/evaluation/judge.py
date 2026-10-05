"""Grade an answer against gold answers with an LLM judge and with alias matching."""

import re

from src.entropy.semantic_entropy import normalize_answer
from src.groq_chat import RateLimitExhausted, chat, complete_text

JUDGE_MODEL = "openai/gpt-oss-120b"
LABELS = ("CORRECT", "INCORRECT", "NOT_ATTEMPTED")
MAX_GOLD_IN_PROMPT = 20  # some TriviaQA items have dozens of aliases
MAX_ATTEMPTS = 2  # one initial try plus one retry

JUDGE_SYSTEM_PROMPT = (
    "You grade answers to trivia questions. You are given a question, a list of acceptable "
    "gold answers, and a candidate answer. Reply with exactly one word:\n"
    "CORRECT if the candidate answer means the same as any gold answer,\n"
    "INCORRECT if it gives a different or wrong answer,\n"
    "NOT_ATTEMPTED if it declines, says it doesn't know, or gives no answer.\n"
    "Output nothing else."
)


def judge_answer(question: str, gold_answers: list[str], answer: str) -> tuple[str, str]:
    """Return (label, raw_reply) where label is one of LABELS.

    A failed call (including a cut-off or empty reply, or one without exactly one label)
    is retried once; the second failure is raised.
    """
    gold = "; ".join(gold_answers[:MAX_GOLD_IN_PROMPT])
    message = f"Question: {question}\nGold answers: {gold}\nCandidate answer: {answer}"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            raw = complete_text(chat(JUDGE_MODEL, JUDGE_SYSTEM_PROMPT, message, temperature=0.0))
            # Check longer labels first so "INCORRECT" isn't read as "CORRECT".
            found = set(re.findall(r"NOT_ATTEMPTED|INCORRECT|CORRECT", raw.upper()))
            if len(found) != 1:
                raise ValueError(f"Judge reply has no single label: {raw!r}")
            return found.pop(), raw
        except RateLimitExhausted:
            raise
        except Exception:
            if attempt == MAX_ATTEMPTS:
                raise


def alias_match(gold_answers: list[str], answer: str) -> bool:
    """True if any normalized gold answer appears as whole words in the normalized answer."""
    padded = f" {normalize_answer(answer)} "
    return any(f" {g} " in padded for g in map(normalize_answer, gold_answers) if g)
