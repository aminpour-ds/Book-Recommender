
"""
Offline evaluation for the Book Recommendation System.
Models:
    1. Popularity
    2. Content-Based
    3. Collaborative Filtering

Evaluation protocol:
    - User-level train/test split
    - Relevance threshold = 8
    - 20% of liked titles held out per user
    - Two seed books per user
    - One query title per seed
    - All held-out titles are relevant for each user's queries
    - training titles are excluded from recommendations
    - Held-out interactions are excluded from popularity training
"""

import math
import numpy as np
import pandas as pd
from ranx import Qrels, Run, evaluate
from src.config import TOP_K, Books_clean, Ratings_clean, REPORTS_DIR
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.popularity import PopularityRecommender


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
RELEVANT_THRESHOLD = 8
TEST_SIZE = 0.20
SEEDS_PER_USER = 2
MIN_LIKED_TITLES = 4
MAX_EVAL_USERS = 1000
K_VALUES = sorted({5, 10, 20, int(TOP_K)})
METRICS = ["precision", "recall", "f1", "hit_rate", "map", "mrr", "ndcg"]

# ============================================================
# DATA PREPARATION
# ============================================================

def normalize_isbn(value):
    """Normalize ISBN while preserving it as a string."""
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def normalize_title(value):
    """Normalize title for consistent title-level matching."""
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().split())


def prepare_data(ratings, books):
    """Validate and prepare the datasets."""

    required_rating_columns = {"User-ID", "ISBN", "Book-Rating"}
    required_book_columns = {"ISBN", "Book-Title", "Book-Author", "Publisher"}

    missing_ratings = required_rating_columns - set(ratings.columns)
    missing_books = required_book_columns - set(books.columns)

    if missing_ratings:
        raise ValueError(f"Missing rating columns: {missing_ratings}")

    if missing_books:
        raise ValueError(f"Missing book columns: {missing_books}")

    ratings = ratings.copy()
    books = books.copy()

    ratings["User-ID"] = ratings["User-ID"].fillna("").astype(str).str.strip()
    ratings["ISBN"] = ratings["ISBN"].map(normalize_isbn)
    ratings["Book-Rating"] = pd.to_numeric(ratings["Book-Rating"], errors="coerce")

    books["ISBN"] = books["ISBN"].map(normalize_isbn)

    for column in ["Book-Title", "Book-Author", "Publisher"]:
        books[column] = books[column].fillna("").astype(str).str.strip()

    ratings = ratings.dropna(subset=["Book-Rating"])
    ratings = ratings[
        (ratings["User-ID"] != "")
        & (ratings["ISBN"] != "")
        & (ratings["Book-Rating"] >= 1)
        & (ratings["Book-Rating"] <= 10)
    ].copy()

    books = books[
        (books["ISBN"] != "")
        & (books["Book-Title"] != "")
    ].drop_duplicates(subset="ISBN", keep="first").copy()

    valid_isbns = set(books["ISBN"])
    ratings = ratings[ratings["ISBN"].isin(valid_isbns)].copy()
    ratings = ratings.drop_duplicates(subset=["User-ID", "ISBN"], keep="last")

    if ratings.empty:
        raise ValueError("No valid rating records remain.")

    if books.empty:
        raise ValueError("No valid books remain.")

    return ratings.reset_index(drop=True), books.reset_index(drop=True)


# ============================================================
# TITLE MAPPINGS
# ============================================================

def build_title_mappings(books):
    """
    Map ISBNs to normalized titles.
    If multiple ISBNs represent the same title, the first ISBN is used as ISBN.
    """

    isbn_to_title = {}
    title_to_isbn = {}

    for _, row in books.iterrows():
        isbn = normalize_isbn(row["ISBN"])
        title = normalize_title(row["Book-Title"])

        if not isbn or not title:
            continue

        isbn_to_title[isbn] = title
        title_to_isbn.setdefault(title, isbn)

    return isbn_to_title, title_to_isbn


# ============================================================
# USER-LEVEL TRAIN / TEST SPLIT
# ============================================================

def build_eval_cases(ratings, isbn_to_title, title_to_isbn, threshold=RELEVANT_THRESHOLD, 
                     test_size=TEST_SIZE, seeds_per_user=SEEDS_PER_USER, min_liked_titles=MIN_LIKED_TITLES, 
                     max_users=MAX_EVAL_USERS, random_state=RANDOM_STATE
                     ):
    """
    Split each user's highly rated titles into train and test.
    Training: Used to select seed books. Each seed becomes one independent evaluation query.
    Test: Relevant books that the models should recommend.    
    """

    if not 0 < test_size < 1:
        raise ValueError("TEST_SIZE must be between 0 and 1.")

    rng = np.random.default_rng(random_state)

    # select the titles that users liked them and rated them more than threshold
    liked = ratings[ratings["Book-Rating"] >= threshold].copy()
    liked["_title_key"] = liked["ISBN"].map(isbn_to_title)
    liked = liked[liked["_title_key"].notna() & (liked["_title_key"] != "")]
    liked = liked.drop_duplicates(subset=["User-ID", "_title_key"], keep="first")

    # groups liked titles for every user with "User-ID", selects only the title column, and produces a dictionary
    liked_by_user = liked.groupby("User-ID")["_title_key"].apply(list).to_dict()

    # Finds users who have enough liked titles. A user is eligible only when they have at least "MIN_LIKED_TITLES" unique liked titles.
    # remove the duplicates and count the likes, it should be more than "MIN_LIKED_TITLES" to be eligible_users.
    required_titles = max(min_liked_titles, seeds_per_user + 1)
    eligible_users = []
    for user_id, titles in liked_by_user.items():
        distinct_titles = set(titles)   # each row of dictionary is like  1: ["A", "B", "A", "C"] and with "set" we remove the duplicates.
        if len(distinct_titles) >= required_titles:
            eligible_users.append(user_id)

    eligible_users = sorted(eligible_users)


    # randomly select the eligible users in size=max_users
    if max_users is not None and len(eligible_users) > max_users:
        eligible_users = sorted(rng.choice(eligible_users, size=max_users, replace=False).tolist())

    cases = []
    held_out_pairs = set()
    split_rows = []

    # processes one eligible user at a time. 
    for user_id in eligible_users:
        user_titles = sorted(set(liked_by_user[user_id]))
        shuffled_titles = rng.permutation(user_titles).tolist()  # randomize the order of user's liked titles.

        # for each user decide how many liked titles will be hidden as test books.
        n_test = max(1, math.ceil(len(user_titles) * test_size))   # guarantees that at least one title enters the test set
        n_test = min(n_test, len(user_titles) - seeds_per_user)   # ensures at least seeds_per_user titles remain in the training set

        # seprate test and train data for each user
        test_titles = shuffled_titles[:n_test]
        train_titles = shuffled_titles[n_test:]

        if len(train_titles) < seeds_per_user or not test_titles:
            continue

        # select seed_titles randomly from the train data for each user
        seed_titles = rng.choice(train_titles, size=seeds_per_user, replace=False).tolist()

        # get the ISBNs for the user's test data
        relevant_isbns = set()
        for title in test_titles:
            if title in title_to_isbn:
                isbn = title_to_isbn[title]
                relevant_isbns.add(isbn)

        if not relevant_isbns:
            continue

        # keep user's test data as pair (user_id, title) to hold these out of train data
        for title in test_titles:
            held_out_pairs.add((user_id, title))


        # per-seed evaluation : This loop creates one independent evaluation query for every seed title.
        for seed_number, seed_title in enumerate(seed_titles, start=1):
            query_id = f"user_{user_id}_seed_{seed_number}"

            cases.append({
                "query_id": query_id,
                "user_id": user_id,
                "seed_number": seed_number,
                "query_title": seed_title,
                "query_isbn": title_to_isbn[seed_title],
                "relevant": relevant_isbns
            })

        # add one summary dictionary for the each user to split_rows. It is used to create the evaluation_split_summary.csv report.
        split_rows.append({
            "user_id": user_id,
            "total_liked_titles": len(user_titles),
            "training_titles": len(train_titles),
            "test_titles": len(test_titles),
            "number_of_seeds": len(seed_titles),
            "seed_titles": " | ".join(seed_titles),
            "test_titles_list": " | ".join(test_titles)
        })

    if not cases:
        raise ValueError("No evaluation cases found. Check the threshold and minimum liked titles.")

    return cases, held_out_pairs, pd.DataFrame(split_rows)


# ============================================================
# USER TRAINING HISTORY
# ============================================================

def build_user_seen_titles(ratings, isbn_to_title, held_out_pairs):
    """
    Build each user's training history. titles the user has interacted with except test data.
    In the training history: 
        - excludes held-out/test titles.
        - including low-rated titles

    At the end all previously rated titles are excluded from recommendations.
    """

    seen_by_user = {}

    for row in ratings[["User-ID", "ISBN"]].itertuples(index=False):
        user_id, isbn = row
        title = isbn_to_title.get(normalize_isbn(isbn))

        if not title:
            continue

        if (user_id, title) in held_out_pairs:
            continue

        if user_id not in seen_by_user:
            seen_by_user[user_id] = set()

        seen_by_user[user_id].add(title)

    return seen_by_user


# ============================================================
# RANX QRELS AND RUN
# ============================================================

def build_qrels(cases):
    """ Evaluation with RANX """
    qrels_dict = {}
    for case in cases:
        qrels_dict[case["query_id"]] = {isbn: 1 for isbn in case["relevant"]}

    return Qrels(qrels_dict)


def build_run(recommendations, model_name):
    """Convert ranked recommendations into ranx Run."""
    run_dict = {}
    for query_id, recommended_isbns in recommendations.items():
        unique_isbns = list(dict.fromkeys(recommended_isbns))

        if not unique_isbns:
            continue

        total = len(unique_isbns)
        run_dict[query_id] = {isbn: float(total - rank) for rank, isbn in enumerate(unique_isbns)}

    if not run_dict:
        raise ValueError(f"{model_name} generated no recommendations.")

    return Run(run_dict, name=model_name)


# ============================================================
# CONTENT-BASED RECOMMENDATIONS
# ============================================================

def generate_content_recommendations(cases, books, title_to_isbn, user_seen_titles, top_k):
    """
    Generate content-based recommendations. returns dictionaries containing ISBN and Book-Title.
    """

    Content_model = ContentBasedRecommender()
    Content_model.fit(books)

    recommendations_by_query = {}

    for case in cases:
        query_id = case["query_id"]
        user_id = case["user_id"]
        seed_title = normalize_title(case["query_title"])

        seen_titles = user_seen_titles.get(user_id, set())
        candidate_k = min(len(books), top_k + len(seen_titles) + 1)

        candidate_items = Content_model.recommend(title=case["query_title"], top_k=candidate_k)

        selected_isbns = []
        selected_titles = set(seen_titles)
        selected_titles.add(seed_title)

        for item in candidate_items:
            title = normalize_title(item.get("Book-Title"))

            if not title or title in selected_titles:
                continue

            canonical_isbn = title_to_isbn.get(title)

            if canonical_isbn is None:
                continue

            selected_titles.add(title)
            selected_isbns.append(canonical_isbn)

            if len(selected_isbns) >= top_k:
                break

        recommendations_by_query[query_id] = selected_isbns

    return recommendations_by_query


# ============================================================
# POPULARITY RECOMMENDATIONS
# ============================================================

def generate_popularity_recommendations(cases, ratings, books, isbn_to_title, title_to_isbn, 
                                        held_out_pairs, user_seen_titles, top_k
                                        ):
    """
    Fit popularity on training interactions only. All held-out user-title interactions are excluded 
    from training. The global popularity ranking is then filtered separately for each user.
    """
    # specify test data with true/false inside the ratings dataset and save Boolean pandas Series in is_held_out.
    is_held_out = []
    for user_id, isbn in zip(ratings["User-ID"], ratings["ISBN"]):
        title = isbn_to_title.get(normalize_isbn(isbn))
        if isinstance(title, str) and title:
            is_test_rating = (user_id, title) in held_out_pairs   # checks whether this exact user/title pair exists in the test/held-out set. if yes, then return True.
        else:
            is_test_rating = False

        is_held_out.append(is_test_rating)

    is_held_out = pd.Series(
        is_held_out,
        index=ratings.index,
        dtype=bool
    )

    train_ratings = ratings.loc[~is_held_out].copy()  #  seprate the train data

    if train_ratings.empty:
        raise ValueError("Popularity training data is empty.")

    Popularity_model = PopularityRecommender()
    Popularity_model.fit(train_ratings, books)

    # Request the full ranking, because we want to filter it separately for each user thats why top_k=len(books).
    popular_books = Popularity_model.recommend(top_k=len(books))

    global_ranked_titles = []
    global_seen_titles = set()

    for item in popular_books:
        isbn = normalize_isbn(item.get("ISBN"))
        title = isbn_to_title.get(isbn)

        if isbn is None:
            continue
        
        # remove duplicate titles from the popular_books ranking.
        if not title or title in global_seen_titles:
            continue

        global_seen_titles.add(title)
        global_ranked_titles.append((title, isbn))

    recommendations_by_query = {}

    for case in cases:
        query_id = case["query_id"]
        user_id = case["user_id"]
        seen_titles = user_seen_titles.get(user_id, set())

        selected = []
        # Remove the titles that the user has already seen.
        for title, canonical_isbn in global_ranked_titles:
            if title in seen_titles:
                continue

            selected.append(canonical_isbn)

            if len(selected) >= top_k:
                break

        recommendations_by_query[query_id] = selected

    return recommendations_by_query


# ============================================================
# RANX EVALUATION
# ============================================================

def evaluate_run(model_name, cases, recommendations, k_values=K_VALUES):
    """Calculate ranx metrics and their mean and standard deviation."""

    qrels = build_qrels(cases)
    run = build_run(recommendations, model_name)

    metric_names = [f"{metric}@{k}" for k in k_values for metric in METRICS]
    # evaluate() writes aggregate mean and standard-deviation values into the Run object
    scores = evaluate(qrels, run, metric_names, return_mean=True, return_std=True, make_comparable=True)

    rows = []

    n_users = len({case["user_id"] for case in cases})

    for k in k_values:
        row = {
            "model": model_name,
            "k": k,
            "n_users": n_users,
            "n_seed_queries": len(cases)
        }

        for metric in METRICS:
            name = f"{metric}@{k}"
            row[f"{metric}_mean"] = run.mean_scores[name]
            row[f"{metric}_std"] = run.std_scores[name]

        rows.append(row)

    return rows


# ============================================================
# USER-LEVEL COVERAGE
# ============================================================

def analyze_user_coverage(model_name, cases, recommendations, k_values=K_VALUES):
    """
    measures whether recommendations can cover relevant books for each individual user.
    """

    # creates one output row per user.
    cases_by_user = {}

    for case in cases:
        cases_by_user.setdefault(case["user_id"], []).append(case)

    rows = []
    for user_id, user_cases in cases_by_user.items():
        row = {
            "model": model_name,
            "user_id": user_id,
            "number_of_seeds": len(user_cases),
            "number_of_relevant_titles": len(user_cases[0]["relevant"])
        }

        for k in k_values:
            seed_hits = []   # relevant title -> hit | no relevant title -> miss

            for case in user_cases:
                recommended = recommendations.get(case["query_id"], [])[:k]
                hit = bool(set(recommended) & case["relevant"])
                seed_hits.append(hit)

            row[f"hit_seed_queries_at_{k}"] = int(sum(seed_hits))
            row[f"seed_hit_rate_at_{k}"] = float(np.mean(seed_hits))
            row[f"user_hit_any_seed_at_{k}"] = bool(any(seed_hits))
            row[f"user_no_hit_at_{k}"] = bool(not any(seed_hits))

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# SEED-LEVEL DETAILS
# ============================================================

def build_seed_details(model_name, cases, recommendations, isbn_to_title, k_values=K_VALUES):
    """Create a detailed report for each seed query."""
    rows = []
    for case in cases:
        query_id = case["query_id"]
        recommended = recommendations.get(query_id, [])

        recommended_titles = [isbn_to_title.get(isbn, "") for isbn in recommended]

        # converts the relevant ISBN set into a sorted list of unique book titles.
        relevant_titles = []
        for isbn in case["relevant"]:
            title = isbn_to_title.get(isbn, "")
            if title:
                relevant_titles.append(title)

        relevant_titles = sorted(set(relevant_titles))

        row = {
            "model": model_name,
            "query_id": query_id,
            "user_id": case["user_id"],
            "seed_number": case["seed_number"],
            "seed_title": case["query_title"],
            "number_of_relevant_titles": len(case["relevant"]),
            "number_of_recommendations": len(recommended),
            "relevant_titles": " | ".join(relevant_titles)
        }

        for k in k_values:
            top_k_isbns = recommended[:k]
            hit_isbns = set(top_k_isbns) & case["relevant"]

            row[f"hits_at_{k}"] = len(hit_isbns)
            row[f"hit_at_{k}"] = bool(hit_isbns)
            row[f"recommended_titles_at_{k}"] = " | ".join(recommended_titles[:k])
            row[f"hit_titles_at_{k}"] = " | ".join(
                isbn_to_title.get(isbn, "")
                for isbn in top_k_isbns
                if isbn in hit_isbns
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# VALIDATION
# ============================================================

def validate_recommendations(model_name, cases, recommendations, user_seen_titles, isbn_to_title, top_k):
    """
    Sanity check:
    - Check that recommendations contain no previously seen training titles
    - no duplicate title-level items.
    """

    case_by_query = {case["query_id"]: case for case in cases}

    for query_id, recommended_isbns in recommendations.items():
        case = case_by_query[query_id]
        user_id = case["user_id"]
        seen_titles = user_seen_titles.get(user_id, set())

        # Convert recommendation ISBNs to titles
        titles = [
            isbn_to_title.get(normalize_isbn(isbn), "")
            for isbn in recommended_isbns
        ]

        # Empty or unknown ISBN mappings are removed
        titles = [title for title in titles if title]
        # Check for duplicate titles
        if len(titles) != len(set(titles)):
            raise AssertionError(f"{model_name}: duplicate title in {query_id}")

        # Check for training leakage
        overlap = set(titles) & seen_titles

        # If a recommendation appears in the user's training-history titles, there is training leakage
        if overlap:
            raise AssertionError(f"{model_name}: training title leakage for {query_id}: {sorted(overlap)[:5]}")

        if len(recommended_isbns) > top_k:
            raise AssertionError(f"{model_name}: more than {top_k} recommendations for {query_id}")

    print(f"Sanity check passed: {model_name}")
    

# ============================================================
# COLLABORATIVE FILTERING
# ============================================================

#def evaluate_collaborative(cases, ratings, books, k_values=K_VALUES):
    """
    It must use the same train/test split and must not train on held-out interactions.
    """



# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(results_df):

    if results_df.empty:
        print("No evaluation results.")
        return

    columns = ["model", "k", "n_users", "n_seed_queries", "precision_mean", "recall_mean", "f1_mean", "hit_rate_mean", "map_mean", "mrr_mean", "ndcg_mean"]

    print("\n========== Evaluation Results ==========")
    print(results_df[columns].round(4).to_string(index=False))


def print_coverage(coverage_df):

    print("\n========== User Coverage ==========")

    for model in coverage_df["model"].unique():
        model_df = coverage_df[coverage_df["model"] == model]
        total_users = len(model_df)

        for k in K_VALUES:
            users_with_hit = int(model_df[f"user_hit_any_seed_at_{k}"].sum())
            users_without_hit = total_users - users_with_hit
            coverage = users_with_hit / total_users if total_users > 0 else 0

            print(
                f"{model:15} K={k:2} | "
                f"Users with hit: {users_with_hit}/{total_users} | "
                f"Without hit: {users_without_hit}/{total_users} | "
                f"Coverage: {coverage:.2%}"
            )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    print("Loading cleaned data...")

    ratings = pd.read_csv(Ratings_clean, dtype={"ISBN": str})
    books = pd.read_csv(Books_clean, dtype={"ISBN": str})

    ratings, books = prepare_data(ratings, books)

    isbn_to_title, title_to_isbn = build_title_mappings(books)

    print("\nBuilding user-level train/test splits...")

    cases, held_out_pairs, split_summary = build_eval_cases(
        ratings=ratings,
        isbn_to_title=isbn_to_title,
        title_to_isbn=title_to_isbn
    )

    n_users = split_summary["user_id"].nunique()

    print(f"\nEvaluation users: {n_users}")
    print(f"Seed queries: {len(cases)}")
    print(f"Seeds per user: {SEEDS_PER_USER}")
    print(f"Relevance threshold: {RELEVANT_THRESHOLD}")
    print(f"Test fraction: {TEST_SIZE:.0%}")
    print(f"K values: {K_VALUES}")

    print("Average liked titles:", round(split_summary["total_liked_titles"].mean(), 2))
    print("Average training titles:", round(split_summary["training_titles"].mean(), 2))
    print("Average test titles:", round(split_summary["test_titles"].mean(), 2))

    # Build each user's training history.
    user_seen_titles = build_user_seen_titles(ratings=ratings, isbn_to_title=isbn_to_title, held_out_pairs=held_out_pairs)

    all_results = []
    all_coverage = []
    all_seed_details = []

    max_k = max(K_VALUES)

    # --------------------------------------------------------
    # CONTENT-BASED
    # --------------------------------------------------------

    print("\nEvaluating Content-Based model...")

    content_recommendations = generate_content_recommendations(
        cases=cases,
        books=books,
        title_to_isbn=title_to_isbn,
        user_seen_titles=user_seen_titles,
        top_k=max_k
    )

    validate_recommendations(
        model_name="content_based",
        cases=cases,
        recommendations=content_recommendations,
        user_seen_titles=user_seen_titles,
        isbn_to_title=isbn_to_title,
        top_k=max_k
    )

    all_results.extend(evaluate_run("content_based", cases, content_recommendations))

    all_coverage.append(analyze_user_coverage("content_based", cases, content_recommendations))

    all_seed_details.append(build_seed_details("content_based", cases, content_recommendations, isbn_to_title))

    # --------------------------------------------------------
    # POPULARITY
    # --------------------------------------------------------

    print("\nEvaluating Popularity model...")

    popularity_recommendations = generate_popularity_recommendations(
        cases=cases,
        ratings=ratings,
        books=books,
        isbn_to_title=isbn_to_title,
        title_to_isbn=title_to_isbn,
        held_out_pairs=held_out_pairs,
        user_seen_titles=user_seen_titles,
        top_k=max_k
    )

    validate_recommendations(
        model_name="popularity",
        cases=cases,
        recommendations=popularity_recommendations,
        user_seen_titles=user_seen_titles,
        isbn_to_title=isbn_to_title,
        top_k=max_k
    )

    all_results.extend(evaluate_run("popularity", cases, popularity_recommendations))

    all_coverage.append(analyze_user_coverage("popularity", cases, popularity_recommendations))

    all_seed_details.append(build_seed_details("popularity", cases, popularity_recommendations, isbn_to_title))

    # --------------------------------------------------------
    # COLLABORATIVE FILTERING
    # --------------------------------------------------------
    '''.....'''


    # --------------------------------------------------------
    # SAVE REPORTS
    # --------------------------------------------------------

    results_df = pd.DataFrame(all_results)
    coverage_df = pd.concat(all_coverage, ignore_index=True)
    seed_details_df = pd.concat(all_seed_details, ignore_index=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(REPORTS_DIR / "evaluation_results.csv", index=False)
    coverage_df.to_csv(REPORTS_DIR / "evaluation_user_coverage.csv", index=False)
    seed_details_df.to_csv(REPORTS_DIR / "evaluation_seed_details.csv", index=False)
    split_summary.to_csv(REPORTS_DIR / "evaluation_split_summary.csv", index=False)

    print_results(results_df)
    print_coverage(coverage_df)

    print("\nAll reports saved in:", REPORTS_DIR)

    return results_df, coverage_df, seed_details_df, split_summary


if __name__ == "__main__":
    run()
