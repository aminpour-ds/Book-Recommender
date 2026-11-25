"""
Draw charts from the CSV files that evaluate_models.py saves in the reports folder.

Run after evaluate_models.py:
    python -m src.plot_results
"""

import matplotlib.pyplot as plt
import pandas as pd
from src.config import REPORTS_DIR

CHARTS_DIR = REPORTS_DIR / "charts"


def plot_metrics_at_k(results, k=10):
    """Bar chart: every metric at one K, one bar per model."""
    metrics = ["precision", "recall", "hit_rate", "map", "mrr", "ndcg"]
    data = results[results["k"] == k].set_index("model")
    data = data[[f"{metric}_mean" for metric in metrics]]
    data.columns = metrics

    ax = data.T.plot(kind="bar", figsize=(9, 5), rot=0)
    ax.set_title(f"Model comparison at K={k}")
    ax.set_ylabel("score (higher is better)")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / f"metrics_at_{k}.png")
    plt.close()


def plot_metric_by_k(results, metric):
    """Line chart: how one metric changes when K grows, one line per model."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for model, model_df in results.groupby("model"):
        ax.plot(model_df["k"], model_df[f"{metric}_mean"], marker="o", label=model)

    ax.set_title(f"{metric} by K")
    ax.set_xlabel("K (number of recommendations)")
    ax.set_ylabel(metric)
    ax.set_xticks(sorted(results["k"].unique()))
    ax.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / f"{metric}_by_k.png")
    plt.close()


def plot_user_coverage(coverage):
    """Bar chart: percent of users who got at least one relevant book (from any seed)."""
    k_values = [int(col.split("_")[-1]) for col in coverage.columns if col.startswith("user_hit_any_seed_at_")]

    rows = {}
    for model, model_df in coverage.groupby("model"):
        rows[model] = [model_df[f"user_hit_any_seed_at_{k}"].mean() * 100 for k in k_values]

    data = pd.DataFrame(rows, index=[f"K={k}" for k in k_values])
    ax = data.plot(kind="bar", figsize=(7, 4), rot=0)
    ax.set_title("Users with at least one hit")
    ax.set_ylabel("% of users")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "user_coverage.png")
    plt.close()


def run():
    results = pd.read_csv(REPORTS_DIR / "evaluation_results.csv")
    coverage = pd.read_csv(REPORTS_DIR / "evaluation_user_coverage.csv")

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_metrics_at_k(results, k=10)
    plot_metric_by_k(results, "hit_rate")
    plot_metric_by_k(results, "ndcg")
    plot_user_coverage(coverage)

    print("Charts saved in:", CHARTS_DIR)


if __name__ == "__main__":
    run()
