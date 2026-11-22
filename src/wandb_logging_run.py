"""Run and log the current recommender evaluation in Weights & Biases."""

import pandas as pd
import wandb
from src.wandb_logging import ExperimentLogger
from src.config import Books_clean, Ratings_clean, TOP_K, WANDB_MODE
from src.evaluate_models import K_VALUES, METRICS, RANDOM_STATE, RELEVANT_THRESHOLD, SEEDS_PER_USER, TEST_SIZE
from src.evaluate_models import run as evaluate_all_models


def dataset_config(ratings, books):
    """Create the dataset and evaluation metadata shared by model runs."""
    return {
        "n_ratings": int(len(ratings)),
        "n_users": int(ratings["User-ID"].nunique()),
        "n_books": int(len(books)),
        "n_unique_rated_books": int(ratings["ISBN"].nunique()),
        "top_k": TOP_K,
        "random_state": RANDOM_STATE,
        "relevance_threshold": RELEVANT_THRESHOLD,
        "test_size": TEST_SIZE,
        "seeds_per_user": SEEDS_PER_USER,
        "k_values": K_VALUES,
    }


def log_model_run(model_name, results_df, coverage_df, seed_details_df,
                  split_summary_df, base_config):
    """Create one W&B run for a model and log its evaluation reports."""
    model_results = results_df[results_df["model"] == model_name].copy()
    model_coverage = coverage_df[coverage_df["model"] == model_name].copy()
    model_seeds = seed_details_df[seed_details_df["model"] == model_name].copy()

    if model_results.empty:
        print(f"[{model_name}] No evaluation rows found; run skipped.")
        return

    run_config = dict(base_config)
    run_config["model"] = model_name

    logger = ExperimentLogger(
        name=model_name,
        config=run_config,
        tags=["recommender", "offline-evaluation", model_name],
        mode=WANDB_MODE,
    )

    for _, result_row in model_results.sort_values("k").iterrows():
        k = int(result_row["k"])
        metrics = {}

        for metric in METRICS:
            metrics[f"{metric}_mean_at_{k}"] = float(result_row[f"{metric}_mean"])
            metrics[f"{metric}_std_at_{k}"] = float(result_row[f"{metric}_std"])

        logger.log(metrics)

    final_row = model_results.sort_values("k").iloc[-1]
    logger.log({
        "n_users": int(final_row["n_users"]),
        "n_seed_queries": int(final_row["n_seed_queries"]),
    })

    logger.log({
        "evaluation_results": wandb.Table(dataframe=model_results),
        "user_coverage": wandb.Table(dataframe=model_coverage),
        "seed_details": wandb.Table(dataframe=model_seeds),
        "split_summary": wandb.Table(dataframe=split_summary_df),
    })

    logger.finish()

    print(
        f"[{model_name}] "
        f"precision_mean={final_row['precision_mean']:.4f}, "
        f"recall_mean={final_row['recall_mean']:.4f} "
        f"at k={int(final_row['k'])}"
    )


def main():
    books = pd.read_csv(Books_clean, dtype={"ISBN": str})
    ratings = pd.read_csv(Ratings_clean, dtype={"ISBN": str})
    base_config = dataset_config(ratings, books)

    # evaluate_all_models() reruns evaluate_models.py and recomputes all reports in memory. 
    # It does not read existing CSV files. It calls run() and Returns 4 DataFrames.
    results_df, coverage_df, seed_details_df, split_summary_df = evaluate_all_models()

    for model_name in results_df["model"].drop_duplicates().tolist():
        log_model_run(
            model_name=model_name,
            results_df=results_df,
            coverage_df=coverage_df,
            seed_details_df=seed_details_df,
            split_summary_df=split_summary_df,
            base_config=base_config,
        )

    print("\nAll evaluation experiments finished.")
    print("For offline runs, W&B results are saved in ./wandb/.")
    print("To sync them online later: run `wandb login`, then "
          "`wandb sync wandb/offline-run-*`.")


if __name__ == "__main__":
    main()