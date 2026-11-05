"""
Load the raw data from CSV files and return as pandas DataFrames.
"""

from src.config import Ratings, Users, Books 
import pandas as pd


def read_csv_file(file_path):
    """
    Read the raw data from CSV files and return as pandas DataFrames.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find '{file_path.name}'. ")
    dataframe = pd.read_csv(file_path, dtype=str)
    if dataframe.empty:
        raise ValueError(f"The file at {file_path} is empty.")
    return dataframe


def load_data():
    """
    Load all data.
    """
    users_df = read_csv_file(Users)
    books_df = read_csv_file(Books)
    ratings_df = read_csv_file(Ratings)

    return users_df, books_df, ratings_df



if __name__ == "__main__":
    users, books, ratings = load_data()
    print("Data loaded successfully.")
    print("Users DataFrame:")
    print(users.head(3))
    print("Books DataFrame:")
    print(books.head(3))
    print("Ratings DataFrame:")
    print(ratings.head(3))