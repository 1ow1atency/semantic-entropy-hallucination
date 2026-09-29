"""Load QA datasets as a generic list of questions with gold answers."""

from dataclasses import dataclass

from datasets import load_dataset

SHUFFLE_BUFFER = 2000


@dataclass
class QAItem:
    id: str
    question: str
    gold_answers: list[str]  # the first entry is the primary gold answer; the rest are aliases


def load_triviaqa(n_questions: int, seed: int = 0) -> list[QAItem]:
    """Stream a reproducible, shuffled slice of TriviaQA validation questions (rc.nocontext).

    For a fixed seed, the first n items are the same whatever n is, so a larger run
    extends a smaller one.
    """
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=SHUFFLE_BUFFER)
    items = []
    for row in ds.take(n_questions):
        answer = row["answer"]
        gold = [answer["value"]] + [a for a in answer["aliases"] if a != answer["value"]]
        items.append(QAItem(id=row["question_id"], question=row["question"], gold_answers=gold))
    return items
