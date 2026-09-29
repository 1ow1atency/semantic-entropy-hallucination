"""Sample multiple answers to the same question from an LLM via the Groq API."""

import logging
import os

from dotenv import load_dotenv
from groq import Groq

MODEL = "openai/gpt-oss-20b"
SYSTEM_PROMPT = (
    "Answer the question directly and concisely. "
    "Give only the answer, with no explanation."
)
MAX_ATTEMPTS = 2  # one initial try plus one retry

logger = logging.getLogger(__name__)

load_dotenv()
_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
        _client = Groq(api_key=api_key)
    return _client


def _query_once(question: str, temperature: float) -> str:
    response = _get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=temperature,
        reasoning_effort="low",
        reasoning_format="hidden",  # return only the final answer, not the reasoning trace
    )
    return response.choices[0].message.content.strip()


def sample_answers(question: str, n_samples: int = 10, temperature: float = 1.0) -> list[str]:
    """Ask the same question n_samples times and return the raw text answers.

    Each request is retried once on failure. A sample that fails twice is logged
    and skipped, so the returned list may have fewer than n_samples entries.
    """
    answers = []
    for i in range(n_samples):
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                answers.append(_query_once(question, temperature))
                break
            except Exception as e:
                if attempt < MAX_ATTEMPTS:
                    logger.warning("Sample %d failed (%s); retrying.", i + 1, e)
                else:
                    logger.error("Sample %d failed twice (%s); skipping.", i + 1, e)
    return answers


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
