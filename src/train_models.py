import pandas as pd
from src.config import models_dir
from src.config import Books_clean
from src.recommenders.popularity import PopularityRecommender
from src.recommenders.content_based import ContentBasedRecommender


def main():
    books = pd.read_csv(Books_clean)

    models_dir.mkdir(parents=True, exist_ok=True)

    recommender = ContentBasedRecommender()
    recommender.fit(books)
    recommender.save(models_dir / "content_based.joblib")
    
    print("Content_based Models saved to", models_dir)


if __name__ == "__main__":
    main()
