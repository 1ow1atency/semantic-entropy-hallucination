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
├── data/               # QA datasets and cached model generations
├── src/
│   ├── sampling/       # Generate multiple answers per question from the LLM
│   ├── clustering/     # Bidirectional-entailment clustering of answers
│   ├── entropy/        # Semantic entropy (and baseline) computation
│   └── evaluation/     # Correctness labeling, AUROC, and other metrics
├── results/
│   ├── metrics/        # Metric outputs (JSON/CSV)
│   └── plots/          # Figures
├── .env.example        # Template for required environment variables
└── README.md
```

## Setup

The LLM is accessed through the [Groq API](https://console.groq.com/). Copy the example env file and add your key:

```bash
cp .env.example .env
# then edit .env and set GROQ_API_KEY
```

`.env` is listed in `.gitignore`, so your key stays out of version control.

Answers are sampled from `openai/gpt-oss-20b` via Groq, since `llama-3.1-8b-instant` wasn't available on this account.

## Status

This is the initial scaffold. The pipeline (sampling, clustering, entropy, evaluation) is not implemented yet.

## References

- Kuhn, L., Gal, Y., & Farquhar, S. (2023). *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation.* ICLR.
- Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024). *Detecting hallucinations in large language models using semantic entropy.* Nature, 630, 625–630.
