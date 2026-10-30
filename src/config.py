
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
raw_data_dir = project_root / "data" / "raw"
processed_data_dir = project_root / "data" / "processed"
models_dir = project_root / "models"

# 3 CSV Raw data files
Books = raw_data_dir / "Books.csv"
Users = raw_data_dir / "Users.csv"
Ratings = raw_data_dir / "Ratings.csv"