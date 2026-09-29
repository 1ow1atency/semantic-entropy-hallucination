# Semantic Entropy Hallucination Detection

Detecting when a large language model (LLM) is likely hallucinating by sampling several answers to the same question and measuring how much they disagree **in meaning**.

## The problem

LLMs often produce fluent, confident answers that are factually wrong. One useful signal is the model's own uncertainty: if you ask the same question several times and get answers that mean different things, the model probably doesn't know the answer.

Measuring that uncertainty is not simple. Token-level metrics such as log-probability or naive entropy over output strings treat *"Paris"*, *"It's Paris."*, and *"The capital of France is Paris"* as three different answers, even though they mean the same thing. That inflates the apparent uncertainty and hides the signal we care about.

## Approach: semantic entropy

Semantic entropy (Kuhn, Gal & Farquhar, 2023; Farquhar et al., *Nature* 2024) measures uncertainty over **meanings** rather than strings:

1. **Sampling:** For a given question, sample *N* answers from the LLM at a non-zero temperature.
2. **Entailment clustering:** Group answers that mean the same thing. Two answers go in the same cluster if each entails the other (bidirectional entailment), as judged by an NLI model or an LLM.
3. **Entropy computation:** Estimate a probability for each meaning cluster (from sequence likelihoods, or from cluster sizes when likelihoods aren't available) and compute the entropy over clusters:

   $$SE(x) = -\sum_{c} p(c \mid x) \log p(c \mid x)$$

   Low entropy means the answers agree in meaning, so the model is probably confident. High entropy means the answers disagree in meaning, so the answer is likely a hallucination.
4. **Evaluation:** Check how well semantic entropy predicts incorrect answers on QA datasets with known ground truth (for example AUROC and accuracy–rejection curves), and compare it with baselines such as naive predictive entropy.

## Project structure

```
.
├── data/                 # QA datasets and cached model generations
├── src/
│   ├── groq_chat.py      # Shared Groq client with rate-limit (429) handling
│   ├── sampling/         # Generate multiple answers per question from the LLM
│   ├── clustering/       # Bidirectional-entailment clustering of answers
│   ├── entropy/          # Semantic entropy and baseline scores
│   └── evaluation/       # Data loading, judging, pipeline and analysis
├── tests/                # Unit tests for the entropy scores
├── results/
│   ├── metrics/          # Per-question results (JSONL) and metric tables
│   └── plots/            # Figures
├── requirements.txt
├── .env.example          # Template for required environment variables
└── README.md
```

## Status

The full pipeline is implemented and has been run on a 25-question pilot:

- **Sampling:** 10 answers per question at temperature 1.0, plus one primary answer at temperature 0 (the answer that gets graded), from `openai/gpt-oss-20b` via Groq. `llama-3.1-8b-instant` wasn't available on this account.
- **Clustering:** bidirectional entailment with `microsoft/deberta-large-mnli`.
- **Scores:** semantic entropy, plus three baselines: naive entropy over raw strings, normalized-string entropy (lowercased, punctuation and articles removed), and 1 − agreement rate.
- **Grading:** an LLM judge (`openai/gpt-oss-120b`, a larger model than the one being tested) labels each primary answer CORRECT, INCORRECT or NOT_ATTEMPTED. A normalized alias match against the gold answers is recorded alongside it.
- **Evaluation:** AUROC for predicting an incorrect answer, with 95% bootstrap confidence intervals over questions, and a selective-answering curve.

## Running it

1. **Set up your API key.** Copy the example env file and set `GROQ_API_KEY` in `.env`. `.env` is listed in `.gitignore`, so your key stays out of version control.

   ```bash
   cp .env.example .env
   ```

2. **Install the requirements** (Python 3.11):

   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

3. **Run the pipeline** on a reproducible slice of TriviaQA validation questions. Each question is written to `results/metrics/pipeline_results.jsonl` as soon as it finishes, and questions already in that file are skipped, so an interrupted run can simply be restarted. The first run downloads the NLI model (about 1.6 GB).

   ```bash
   .venv/bin/python -m src.evaluation.run_pipeline --n_questions 25 --seed 0
   ```

4. **Analyze the results.** This writes `summary.json`, `auroc.csv` and `selective_answering.csv` to `results/metrics/`, and the plot to `results/plots/`.

   ```bash
   .venv/bin/python -m src.evaluation.analyze
   ```

Run the unit tests with `.venv/bin/python -m unittest discover -s tests -t .`

## Results (25-question pilot)

> **These are pilot numbers from 25 TriviaQA questions.** The confidence intervals are wide and overlap heavily, so this run can't tell the scores apart. It shows that the pipeline works end to end, not which score is best.

The primary answer was correct on 12 of 25 questions (48%). No answers were NOT_ATTEMPTED, and the judge and alias-match labels agreed on all 25.

AUROC for predicting an incorrect answer (higher is better; 0.5 is chance), with 95% bootstrap CIs over questions:

| Score | AUROC | 95% CI |
|---|---|---|
| Semantic entropy | 0.891 | [0.729, 1.000] |
| Naive entropy | 0.869 | [0.702, 1.000] |
| Normalized-string entropy | 0.862 | [0.699, 0.987] |
| 1 − agreement rate | 0.904 | [0.763, 1.000] |

All four scores do much better than chance on this pilot. Their intervals overlap almost completely, so a much larger run is needed to compare them.

The selective-answering curve shows accuracy on the questions that are answered when the most uncertain questions are refused first. Coverage is the share of questions answered.

![Selective answering curve](results/plots/selective_answering.png)

## Limitations

- **Small sample.** 25 questions give wide confidence intervals; see above.
- **Clustering can split answers that mean the same thing.** The NLI model sometimes judges a short answer and a longer sentence carrying the same answer as not entailing each other. For "Which title character was named Dolores Haze?", the samples "Lolita." and "Dolores Haze is the title character in Nabokov's novel *Lolita*." ended up in different clusters. This inflates semantic entropy on questions the model answers correctly but at varying lengths.
- **Discrete semantic entropy.** Groq's API doesn't return token probabilities, so cluster probabilities are estimated from how often each cluster appears among the 10 samples.

## References

- Kuhn, L., Gal, Y., & Farquhar, S. (2023). *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation.* ICLR.
- Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024). *Detecting hallucinations in large language models using semantic entropy.* Nature, 630, 625–630.
