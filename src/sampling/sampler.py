"""Sample multiple answers to the same question from an LLM via the Groq API."""

import logging
from dataclasses import dataclass

from src.groq_chat import IncompleteResponse, RateLimitExhausted, chat, complete_text

MODEL = "openai/gpt-oss-20b"
SYSTEM_PROMPT = (
    "Answer the question directly and concisely. "
    "Give only the answer, with no explanation."
)
MAX_ATTEMPTS = 2  # one initial try plus one retry

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    text: str
    finish_reason: str | None
    completion_tokens: int | None


@dataclass
class CallFailure:
    """A call that failed on both attempts."""
    error: str
    finish_reason: str | None  # of the last attempt, if the API answered at all


def sample_answers(question: str, n_samples: int = 10, temperature: float = 1.0) -> list[str]:
    """Ask the same question n_samples times and return the raw text answers."""
    samples, _ = sample_with_metadata(question, n_samples, temperature)
    return [s.text for s in samples]


def sample_with_metadata(
    question: str, n_samples: int = 10, temperature: float = 1.0
) -> tuple[list[Sample], list[CallFailure]]:
    """Like sample_answers, but each answer also carries its finish reason and token usage.

    Each request is retried once on failure, including a reply that hit the token limit
    or came back empty. A sample that fails twice is logged, skipped and returned in the
    failure list, so the sample list may have fewer than n_samples entries.
    Rate limits are waited out in chat(); RateLimitExhausted is raised to the caller.
    """
    answers, failures = [], []
    for i in range(n_samples):
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                result = chat(MODEL, SYSTEM_PROMPT, question, temperature)
                answers.append(Sample(complete_text(result), result.finish_reason, result.completion_tokens))
                break
            except RateLimitExhausted:
                raise
            except Exception as e:
                if attempt < MAX_ATTEMPTS:
                    logger.warning("Sample %d failed (%s); retrying.", i + 1, e)
                else:
                    logger.error("Sample %d failed twice (%s); skipping.", i + 1, e)
                    finish_reason = e.finish_reason if isinstance(e, IncompleteResponse) else None
                    failures.append(CallFailure(str(e), finish_reason))
    return answers, failures


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    questions = {
        "stable": "What is the capital of France?",
        "varied": "In what year was the Tuvaluan town of Funafuti first surveyed by the United States Exploring Expedition?",
    }
    for label, question in questions.items():
        answers = sample_answers(question)
        print(f"\n[{label}] {question}  ({len(answers)} samples)")
        for j, answer in enumerate(answers, 1):
            print(f"  {j:2d}. {answer}")
