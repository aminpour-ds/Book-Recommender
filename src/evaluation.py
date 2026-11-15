"""Offline evaluation for the recommender models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.popularity import PopularityRecommender


# A class that just holds the results of one model. "frozen=True" makes it read-only after creation,
# so nobody can accidentally change the results.
@dataclass(frozen=True)
class EvaluationResult:
    """Aggregated and per-fold ranking metrics plus recommendation evidence."""

    model: str
    k: int
    folds: list[dict[str, float | int]]
    precision_at_k: float
    recall_at_k: float
    recommendations: list[dict[str, Any]]

    def object_as_dict(self) -> dict[str, Any]:
        """Converts the object to a normal dictionary, to save it as JSON or show it in Streamlit."""
        return {
            "model": self.model,
            "k": self.k,
            "folds": self.folds,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "recommendations": self.recommendations,
        }


def precision_recall_at_k(recommended_ids: list[str], high_rated_book_ids: set[str], k: int) -> tuple[float, float]:
    """
    Returns Precision@k and Recall@k for one user's recommendations.
    recommended_ids: list of book IDs that the model recommended
    high_rated_book_ids: only books the user rated highly (not low-rated books)
    hits = number of recommended books that the user actually liked
    """
    if k <= 0:
        raise ValueError("k must be greater than zero.")
    if not high_rated_book_ids:
        raise ValueError("high_rated_book_ids must not be empty.")

    recommended = recommended_ids[:k]
    hits = len(set(recommended) & high_rated_book_ids)
    precision = hits / k
    recall = hits / len(high_rated_book_ids)
    return precision, recall


def evaluate_recommenders(
    books: pd.DataFrame,
    ratings: pd.DataFrame,
    *,
    n_splits: int = 5,
    k: int = 10,
    relevance_threshold: int = 7,
    min_ratings_per_user: int = 10,
    max_seeds: int = 5,
    max_users: int | None = 1000,
    random_state: int = 42,
) -> dict[str, dict[str, Any]]:
    """
    Evaluate the models with user-aware K-fold holdouts.

    - Every user's ratings are spread over the folds, so each user has ratings in both train and test.
    - Popularity is fitted on the training ratings of each fold (all users, no leakage).
    - Content-based uses book metadata only. The user's highly rated TRAIN books are the query seeds.
    - Ratings >= relevance_threshold count as "liked". Held-out liked books are the ground truth.
    - Users with fewer than min_ratings_per_user ratings are removed (too little data to split).
    - At most max_users randomly chosen users are scored (None = all users). 
    - The same users and the same folds are used for every model, so the results are comparable.
    """
    _validate_inputs(books, ratings, n_splits, k, relevance_threshold, min_ratings_per_user, max_seeds, max_users)

    # Work with ISBNs as strings everywhere (avoids repeated astype(str) calls and dtype mismatches).
    books = books.assign(ISBN=books["ISBN"].astype(str))
    ratings = ratings.assign(ISBN=ratings["ISBN"].astype(str))

    # Keep only ratings of books that exist in the catalogue. Otherwise content-based can never hit them.
    ratings = ratings[ratings["ISBN"].isin(set(books["ISBN"]))]

    # Keep only users with enough ratings to be split into folds.
    ratings_per_user = ratings.groupby("User-ID")["ISBN"].transform("size")
    ratings = ratings[ratings_per_user >= min_ratings_per_user].reset_index(drop=True)
    if ratings.empty:
        raise ValueError("No users left after filtering. Lower min_ratings_per_user.")

    # Choose which users are scored (the models are still fitted on ALL users' training ratings).
    user_ids = np.sort(ratings["User-ID"].unique())
    if max_users is not None and len(user_ids) > max_users:
        user_ids = np.random.default_rng(random_state).choice(user_ids, size=max_users, replace=False)
    eval_users = set(user_ids.tolist())

    fold_ids = _user_aware_fold_ids(ratings, n_splits, random_state)
    isbn_to_title = dict(zip(books["ISBN"], books["Book-Title"]))

    results = {}
    for model_name in ("content-based", "popularity"):
        results[model_name] = _evaluate_model(
            model_name, books, ratings, isbn_to_title, fold_ids, eval_users,
            n_splits=n_splits, k=k, relevance_threshold=relevance_threshold, max_seeds=max_seeds,
        ).object_as_dict()

    return results


def _evaluate_model(
    model_name: str,
    books: pd.DataFrame,
    ratings: pd.DataFrame,
    isbn_to_title: dict[str, Any],
    fold_ids: np.ndarray,
    eval_users: set,
    *,
    n_splits: int,
    k: int,
    relevance_threshold: int,
    max_seeds: int,
) -> EvaluationResult:
    """
    For every fold:
    1. fit the model (popularity: on the training ratings; content-based: on book metadata, once),
    2. for every scored user, build the recommendations from the user's TRAIN data,
    3. compare them with the user's liked TEST books (Precision@k / Recall@k).
    """
    is_eval_user = ratings["User-ID"].isin(eval_users).to_numpy()
    fold_reports: list[dict[str, float | int]] = []
    recommendation_rows: list[dict[str, Any]] = []
    shared_model = None  # content-based does not depend on ratings, so it is fitted only once

    for fold_number in range(1, n_splits + 1):
        in_test = fold_ids == fold_number - 1
        train = ratings[~in_test]                      # all users: used to fit the model
        test = ratings[in_test & is_eval_user]         # scored users only
        train_eval = train[is_eval_user[~in_test]]     # scored users only: used for their history

        if model_name == "content-based":
            if shared_model is None:
                shared_model = _fit_model(model_name, books, train)
            model = shared_model
        else:
            model = _fit_model(model_name, books, train)

        fold_precisions = []
        fold_recalls = []
        train_by_user = dict(tuple(train_eval.groupby("User-ID")))

        for user_id, test_user in test.groupby("User-ID"):
            train_user = train_by_user.get(user_id)
            if train_user is None:
                continue

            # Ground truth: books the user rated >= relevance_threshold in the test part.
            high_rated_book_ids = set(test_user.loc[test_user["Book-Rating"] >= relevance_threshold, "ISBN"])
            if not high_rated_book_ids:
                continue

            # Seeds: the user's best train books. Low-rated books must not be used as queries.
            # The same eligibility rule is applied to both models, so both score the same users.
            liked_train = train_user[train_user["Book-Rating"] >= relevance_threshold]
            if liked_train.empty:
                continue
            seed_isbns = (
                liked_train.sort_values(["Book-Rating", "ISBN"], ascending=[False, True])
                .head(max_seeds)["ISBN"].tolist()
            )

            seen_ids = set(train_user["ISBN"])  # books already rated: must not be recommended again
            recommendations = _recommend_for_user(model_name, model, isbn_to_title, seed_isbns, k, seen_ids)
            recommended_ids = [str(row["ISBN"]) for row in recommendations]
            precision, recall = precision_recall_at_k(recommended_ids, high_rated_book_ids, k)
            fold_precisions.append(precision)
            fold_recalls.append(recall)

            for rank, row in enumerate(recommendations, start=1):
                recommendation_rows.append(
                    {
                        "model": model_name,
                        "fold": fold_number,
                        "user_id": str(user_id),
                        "rank": rank,
                        "ISBN": str(row["ISBN"]),
                        "score": float(row["score"]),
                        "score_type": row["score_type"],
                        "relevant": str(row["ISBN"]) in high_rated_book_ids,
                    }
                )

        fold_reports.append(
            {
                "fold": fold_number,
                "users_evaluated": len(fold_precisions),
                "precision_at_k": _mean(fold_precisions),
                "recall_at_k": _mean(fold_recalls),
            }
        )

    evaluated_folds = [report for report in fold_reports if report["users_evaluated"]]
    return EvaluationResult(
        model=model_name,
        k=k,
        folds=fold_reports,
        precision_at_k=_mean([float(report["precision_at_k"]) for report in evaluated_folds]),
        recall_at_k=_mean([float(report["recall_at_k"]) for report in evaluated_folds]),
        recommendations=recommendation_rows,
    )


def _user_aware_fold_ids(ratings: pd.DataFrame, n_splits: int, random_state: int) -> np.ndarray:
    """
    User-aware K-fold: the ratings of each user are shuffled and dealt to the folds one by one.
    Returns an array with the fold number (0 .. n_splits-1) for every rating row.
    That row is in the TEST part of that fold and in the TRAIN part of all other folds.
    A random starting fold per user keeps the folds balanced (otherwise fold 0 always gets the extra rows).
    """
    rng = np.random.default_rng(random_state)
    fold_ids = np.empty(len(ratings), dtype=int)

    for user_positions in ratings.groupby("User-ID").indices.values():
        shuffled = rng.permutation(user_positions)
        start = int(rng.integers(n_splits))
        fold_ids[shuffled] = (np.arange(len(shuffled)) + start) % n_splits

    return fold_ids


def _fit_model(model_name: str, books: pd.DataFrame, ratings: pd.DataFrame):
    if model_name == "content-based":
        return ContentBasedRecommender().fit(books)
    if model_name == "popularity":
        return PopularityRecommender().fit(ratings, books)
    raise ValueError(f"Unsupported model: {model_name}")


def _recommend_for_user(
    model_name: str,
    model,
    isbn_to_title: dict[str, Any],
    seed_isbns: list[str],
    k: int,
    seen_ids: set[str],
) -> list[dict[str, Any]]:
    """Returns up to k recommended books for a user, excluding books they have already rated."""
    # popularity branch
    if model_name == "popularity":
        # Ask for extra candidates, because some popular books may already be known by the user.
        candidates = model.recommend(top_k=k + len(seen_ids))
        return [
            {
                "ISBN": row["ISBN"],
                "score": row["average_rating"],
                # This is the book's average over ALL users, not a prediction for this user.
                "score_type": "average_rating",
            }
            for row in candidates
            if str(row["ISBN"]) not in seen_ids
        ][:k]

    # content-based branch: similar books to each seed, merged. The best similarity wins per book.
    if model_name == "content-based":
        by_isbn: dict[str, dict[str, Any]] = {}
        for seed in seed_isbns:
            title = isbn_to_title.get(seed)
            if not isinstance(title, str):
                continue
            for row in model.recommend(title, top_k=k):
                isbn = str(row["ISBN"])
                if isbn not in seen_ids and (isbn not in by_isbn or row["similarity"] > by_isbn[isbn]["score"]):
                    by_isbn[isbn] = {
                        "ISBN": row["ISBN"],
                        "score": row["similarity"],
                        "score_type": "similarity",
                    }
        return sorted(by_isbn.values(), key=lambda row: row["score"], reverse=True)[:k]

    raise ValueError(f"Unsupported model: {model_name}")


def _validate_inputs(
    books: pd.DataFrame,
    ratings: pd.DataFrame,
    n_splits: int,
    k: int,
    relevance_threshold: int,
    min_ratings_per_user: int,
    max_seeds: int,
    max_users: int | None,
) -> None:
    """Raises ValueError if the input data or the settings are invalid."""
    required_books = {"ISBN", "Book-Title", "Book-Author", "Publisher"}
    required_ratings = {"User-ID", "ISBN", "Book-Rating"}
    missing_books = required_books - set(books.columns)
    missing_ratings = required_ratings - set(ratings.columns)
    if missing_books:
        raise ValueError(f"Books data is missing columns: {missing_books}")
    if missing_ratings:
        raise ValueError(f"Ratings data is missing columns: {missing_ratings}")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if n_splits > len(ratings):
        raise ValueError("n_splits cannot exceed the number of ratings.")
    if k <= 0:
        raise ValueError("k must be greater than zero.")
    if not 1 <= relevance_threshold <= 10:
        raise ValueError("relevance_threshold must be between 1 and 10.")
    if min_ratings_per_user < n_splits:
        raise ValueError("min_ratings_per_user must be at least n_splits.")
    if max_seeds <= 0:
        raise ValueError("max_seeds must be greater than zero.")
    if max_users is not None and max_users <= 0:
        raise ValueError("max_users must be greater than zero or None.")


def _mean(values: list[float]) -> float:
    """Returns the average rounded to 4 decimals, or 0.0 for an empty list so there is no division by zero."""
    return round(sum(values) / len(values), 4) if values else 0.0