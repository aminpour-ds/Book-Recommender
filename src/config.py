
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
raw_data_dir = project_root / "data" / "raw"
processed_data_dir = project_root / "data" / "processed"
models_dir = project_root / "models"

# 3 CSV Raw data files
Books = raw_data_dir / "Books.csv"
Users = raw_data_dir / "Users.csv"
Ratings = raw_data_dir / "Ratings.csv"

# 3 CSV Processed data files
Books_clean = processed_data_dir / "books_clean.csv"
Users_clean = processed_data_dir / "users_clean.csv"
Ratings_clean = processed_data_dir / "ratings_clean.csv"

# Default number of recommendations to return
TOP_K=10

# Path to the trained models file
content_based_model = models_dir / "content_based.joblib"