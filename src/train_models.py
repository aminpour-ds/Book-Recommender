import pandas as pd
from src.config import models_dir
from src.config import Books_clean, Ratings_clean
from src.recommenders.popularity import PopularityRecommender
from src.recommenders.content_based import ContentBasedRecommender


def main():
    books = pd.read_csv(Books_clean)
    ratings = pd.read_csv(Ratings_clean)

    models_dir.mkdir(parents=True, exist_ok=True)

    recommender = ContentBasedRecommender()
    recommender.fit(books)
    recommender.save(models_dir / "content_based.joblib")

    Popularity = PopularityRecommender()
    Popularity.fit(ratings, books)
    Popularity.save(models_dir / "popularity.joblib")
    
    print("Models saved to", models_dir)


if __name__ == "__main__":
    main()
