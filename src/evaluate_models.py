"""Run offline recommender evaluation, print a short summary and save the full JSON report."""

import argparse
import json
import statistics
from pathlib import Path

import pandas as pd

from src.config import Books_clean, Ratings_clean
from src.evaluation import evaluate_recommenders

# The full report (including every recommended book) is saved here instead of being printed.
REPORT_PATH = Path("reports") / "evaluation_report.json"


def _std(values: list[float]) -> float:
    """Standard deviation across folds (0.0 if there are fewer than 2 folds)."""
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def print_summary(report: dict) -> None:
    """Print the performance measures of every model as small tables."""
    k = next(iter(report.values()))["k"]

    print(f"\n=== Model performance (top-{k} recommendations) ===\n")
    header = f"{'Model':<15}{'Precision@k':>13}{'(std)':>9}{'Recall@k':>11}{'(std)':>9}{'Avg score':>11}  Score type"
    print(header)
    print("-" * len(header))

    for name, result in report.items():
        folds = [f for f in result["folds"] if f["users_evaluated"]]
        precision_std = _std([f["precision_at_k"] for f in folds])
        recall_std = _std([f["recall_at_k"] for f in folds])

        recs = pd.DataFrame(result["recommendations"])
        avg_score = recs["score"].mean() if not recs.empty else float("nan")
        score_type = recs["score_type"].iloc[0] if not recs.empty else "-"

        print(
            f"{name:<15}{result['precision_at_k']:>13.4f}{precision_std:>9.4f}"
            f"{result['recall_at_k']:>11.4f}{recall_std:>9.4f}{avg_score:>11.3f}  {score_type}"
        )

    for name, result in report.items():
        print(f"\n{name} - per fold")
        print(f"{'Fold':<6}{'Users':>8}{'Precision@k':>13}{'Recall@k':>11}")
        for fold in result["folds"]:
            print(
                f"{fold['fold']:<6}{fold['users_evaluated']:>8}"
                f"{fold['precision_at_k']:>13.4f}{fold['recall_at_k']:>11.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the recommender models.")
    parser.add_argument("--k", type=int, default=10, help="length of each recommendation list")
    parser.add_argument("--max-users", type=int, default=1000, help="number of users to score (0 = all users)")
    args = parser.parse_args()

    # ISBN must be read as text, otherwise leading zeros are lost and books/ratings no longer match.
    books = pd.read_csv(Books_clean, dtype={"ISBN": str})
    ratings = pd.read_csv(Ratings_clean, dtype={"ISBN": str})

    print("Evaluating... (this can take a few minutes)")
    report = evaluate_recommenders(
        books,
        ratings,
        k=args.k,
        max_users=args.max_users or None,
    )

    print_summary(report)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull report (with every recommendation and its score) saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()