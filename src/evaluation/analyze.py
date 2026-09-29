"""Analyze pipeline results: accuracy, AUROC with bootstrap CIs, and selective answering."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from src.evaluation.run_pipeline import load_results

METRICS_DIR = Path("results/metrics")
PLOTS_DIR = Path("results/plots")
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 0

# Scores where higher means "more likely incorrect".
SCORES = {
    "semantic_entropy": ("Semantic entropy", lambda r: r["semantic_entropy"]),
    "naive_entropy": ("Naive entropy", lambda r: r["naive_entropy"]),
    "normalized_string_entropy": ("Normalized-string entropy", lambda r: r["normalized_string_entropy"]),
    "one_minus_agreement": ("1 − agreement rate", lambda r: 1 - r["agreement_rate"]),
}
PLOTTED_SCORES = ["semantic_entropy", "naive_entropy", "normalized_string_entropy"]

# Reference palette: categorical slots 1-3, text and surface tokens (light mode).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def auroc_with_ci(incorrect: np.ndarray, scores: np.ndarray) -> tuple[float, float, float]:
    """AUROC for predicting an incorrect answer, with a 95% percentile bootstrap CI over questions."""
    if len(set(incorrect)) < 2:
        return float("nan"), float("nan"), float("nan")
    auroc = roc_auc_score(incorrect, scores)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(incorrect), len(incorrect))
        if len(set(incorrect[idx])) == 2:  # AUROC is undefined for a one-class resample
            boot.append(roc_auc_score(incorrect[idx], scores[idx]))
    low, high = np.percentile(boot, [2.5, 97.5])
    return auroc, low, high


def selective_curve(correct: np.ndarray, scores: np.ndarray) -> list[tuple[float, float, float]]:
    """(threshold, coverage, accuracy) when answering only questions with score <= threshold.

    The threshold sweeps the unique score values, so tied questions are kept or dropped together.
    """
    return [
        (t, float(np.mean(scores <= t)), float(np.mean(correct[scores <= t])))
        for t in np.unique(scores)
    ]


def plot_selective(curves: dict[str, list], overall_accuracy: float, n: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.axhline(overall_accuracy, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate(f"Answer everything: {overall_accuracy:.0%}", xy=(0.02, overall_accuracy),
                xycoords=("axes fraction", "data"), xytext=(0, -12), textcoords="offset points",
                color=TEXT_SECONDARY, fontsize=8.5)

    for (key, curve), color in zip(curves.items(), SERIES_COLORS):
        coverage = [c for _, c, _ in curve]
        accuracy = [a for _, _, a in curve]
        ax.plot(coverage, accuracy, color=color, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=SCORES[key][0], zorder=3)

    ax.set_xlim(0, 1.02)
    ax.set_ylim(min(0.0, overall_accuracy - 0.1), 1.04)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Coverage (share of questions answered)", color=TEXT_SECONDARY, fontsize=9.5)
    ax.set_ylabel("Accuracy on answered questions", color=TEXT_SECONDARY, fontsize=9.5)
    ax.set_title(f"Selective answering: refuse the most uncertain questions first (n = {n})",
                 loc="left", color=TEXT_PRIMARY, fontsize=11)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8.5, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    legend = ax.legend(loc="lower left", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(TEXT_PRIMARY)

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    records = load_results()
    if not records:
        raise SystemExit("No results found; run src.evaluation.run_pipeline first.")
    attempted = [r for r in records if r["judge_label"] != "NOT_ATTEMPTED"]
    n_excluded = len(records) - len(attempted)
    correct = np.array([r["judge_label"] == "CORRECT" for r in attempted])
    incorrect = (~correct).astype(int)
    accuracy = float(correct.mean())
    alias_accuracy = float(np.mean([r["alias_match"] for r in attempted]))

    print(f"Questions: {len(records)} total, {n_excluded} NOT_ATTEMPTED excluded, {len(attempted)} analyzed")
    print(f"Accuracy (judge): {accuracy:.1%} ({correct.sum()}/{len(correct)})")
    print(f"Accuracy (alias match, same questions): {alias_accuracy:.1%}")

    auroc_rows = []
    print(f"\n{'Score':<28} {'AUROC':>6}   95% CI")
    for key, (name, get) in SCORES.items():
        scores = np.array([get(r) for r in attempted], dtype=float)
        auroc, low, high = auroc_with_ci(incorrect, scores)
        auroc_rows.append({"score": key, "auroc": auroc, "ci_low": low, "ci_high": high})
        print(f"{name:<28} {auroc:>6.3f}   [{low:.3f}, {high:.3f}]")
    if not 0 < incorrect.sum() < len(incorrect):
        print("AUROC is undefined: every analyzed answer has the same label.")

    curves = {
        key: selective_curve(correct, np.array([SCORES[key][1](r) for r in attempted], dtype=float))
        for key in PLOTTED_SCORES
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    with (METRICS_DIR / "summary.json").open("w") as f:
        json.dump({"n_total": len(records), "n_not_attempted_excluded": n_excluded,
                   "n_analyzed": len(attempted), "accuracy_judge": accuracy,
                   "accuracy_alias_match": alias_accuracy,
                   "n_bootstrap": N_BOOTSTRAP}, f, indent=2)
    with (METRICS_DIR / "auroc.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["score", "auroc", "ci_low", "ci_high"])
        writer.writeheader()
        writer.writerows(auroc_rows)
    with (METRICS_DIR / "selective_answering.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["score", "threshold", "coverage", "accuracy"])
        for key, curve in curves.items():
            writer.writerows([key, *point] for point in curve)
    plot_selective(curves, accuracy, len(attempted), PLOTS_DIR / "selective_answering.png")
    print(f"\nWrote {METRICS_DIR}/summary.json, auroc.csv, selective_answering.csv "
          f"and {PLOTS_DIR}/selective_answering.png")


if __name__ == "__main__":
    main()
