"""Shared Groq chat client with rate-limit (HTTP 429) handling."""

import logging
import os
import time

from dotenv import load_dotenv
from groq import Groq, RateLimitError

MAX_RATE_LIMIT_WAITS = 10
MAX_WAIT_SECONDS = 300  # longer waits usually mean a daily quota; stop instead of hanging
DEFAULT_WAIT_SECONDS = 10.0

logger = logging.getLogger(__name__)

load_dotenv()
_client = None


class RateLimitExhausted(Exception):
    """Raised when Groq keeps rate-limiting us, or asks us to wait longer than MAX_WAIT_SECONDS."""


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
        # Retries are handled here and by callers, not by the SDK.
        _client = Groq(api_key=api_key, max_retries=0)
    return _client


def _retry_after_seconds(error: RateLimitError) -> float:
    try:
        return float(error.response.headers.get("retry-after"))
    except (TypeError, ValueError):
        return DEFAULT_WAIT_SECONDS


def chat(model: str, system_prompt: str, user_message: str, temperature: float) -> str:
    """Send one chat request and return the stripped reply text.

    Reasoning models run with low, hidden reasoning so only the final answer comes back.
    On a 429, waits for the server's retry-after time and tries again.
    """
    for _ in range(MAX_RATE_LIMIT_WAITS):
        try:
            response = _get_client().chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                reasoning_effort="low",
                reasoning_format="hidden",
            )
            return response.choices[0].message.content.strip()
        except RateLimitError as e:
            wait = _retry_after_seconds(e)
            if wait > MAX_WAIT_SECONDS:
                raise RateLimitExhausted(f"Groq asked to wait {wait:.0f}s ({e})") from e
            logger.warning("Rate limited by Groq; waiting %.1fs.", wait)
            time.sleep(wait)
    raise RateLimitExhausted(f"Still rate limited after {MAX_RATE_LIMIT_WAITS} waits.")
