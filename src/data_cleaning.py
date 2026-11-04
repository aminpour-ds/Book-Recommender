import pandas as pd
from src.data_Loading import load_data
from src.config import processed_data_dir

CURRENT_YEAR = 2026


def safe_url(value):
    """Return a clean string URL, or '' if the value is missing/NaN."""
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text


# ---------------------------------------------------------------------------
# Clean Books
# ---------------------------------------------------------------------------
def clean_books(raw_books):
    """
    Clean the books data.

    Rules:
    - Strip whitespace from column names (the headers) and column fields (the data).
    - Remove duplicate ISBNs.
    - Remove rows without ISBN or title.
    - Convert publication year to a numeric value.
    - Replace invalid and missing years as <NA>
    - Replace missing authors and publishers with "Unknown" placeholders.
    - missing/NaN URLs in the image columns become an empty string.
    """
    books = raw_books.copy()

    # Strip whitespace from column names (the headers)
    books.columns = books.columns.str.strip()
    # Strip whitespace from column fields (the data)
    books["ISBN"] = books["ISBN"].str.strip()
    books["Book-Title"] = books["Book-Title"].str.strip()
    books["Book-Author"] = books["Book-Author"].str.strip()
    books["Publisher"] = books["Publisher"].str.strip()
    books["Image-URL-S"] = books["Image-URL-S"].str.strip()
    books["Image-URL-M"] = books["Image-URL-M"].str.strip()
    books["Image-URL-L"] = books["Image-URL-L"].str.strip()
    books["Year-Of-Publication"] = books["Year-Of-Publication"].str.strip()

    # Remove duplicate ISBNs
    books = books.drop_duplicates(subset="ISBN", keep="first")

    # Remove rows without ISBN or title or with empty strings in those fields
    books = books.dropna(subset=["ISBN", "Book-Title"])
    books = books[books["ISBN"] != ""]
    books = books[books["Book-Title"] != ""]

    # Convert publication year to a numeric value.
    books["Year-Of-Publication"] = pd.to_numeric(books["Year-Of-Publication"], errors="coerce").astype("Int64") 

    # Replace invalid and missing years as <NA>
    invalid_year = (books["Year-Of-Publication"] < 1000) | (books["Year-Of-Publication"] > CURRENT_YEAR)
    books.loc[invalid_year, "Year-Of-Publication"] = pd.NA
    
    # Replace missing authors and publishers with "Unknown" placeholders.
    books["Book-Author"] = books["Book-Author"].replace("", pd.NA).fillna("Unknown")
    books["Publisher"] = books["Publisher"].replace("", pd.NA).fillna("Unknown")

    # missing/NaN URLs in the image columns become an empty string.
    books["Image-URL-S"] = books["Image-URL-S"].apply(safe_url)
    books["Image-URL-M"] = books["Image-URL-M"].apply(safe_url)
    books["Image-URL-L"] = books["Image-URL-L"].apply(safe_url)

    books = books.reset_index(drop=True)
    return books


# ---------------------------------------------------------------------------
# Clean Users
# ---------------------------------------------------------------------------
def clean_users(users):
    """
    Clean the users data.

    Rules:
    - Strip whitespace from column names and fields
    - Convert user IDs to integers.
    - Remove rows without user IDs or with invalid user IDs.
    - Remove duplicate user ID. 
    - Convert age to a numeric, round value.
    - Replace missing ages and invalid ages "below 5 or above 100" as <NA>
    - Replace missing locations with "Unknown" placeholders.
    """
    users = users.copy()

    # Strip whitespace from column names (the headers)
    users.columns = users.columns.str.strip()
    # Strip whitespace from column fields (the data)
    users["User-ID"] = users["User-ID"].str.strip()
    users["Location"] = users["Location"].str.strip()
    users["Age"] = users["Age"].str.strip()

    # Convert user IDs to integers
    users["User-ID"] = pd.to_numeric(users["User-ID"], errors="coerce").astype("Int64")
    
    # Remove rows without user IDs or with invalid user IDs
    users = users.dropna(subset=["User-ID"])
    users = users[users["User-ID"] > 0]

    # Remove duplicate user ID
    users = users.drop_duplicates(subset="User-ID", keep="first")

    # Convert age to a numeric, round value
    users["Age"] = pd.to_numeric(users["Age"], errors="coerce").round().astype("Int64")

    # Replace missing ages and invalid ages "below 5 or above 100" as <NA>
    invalid_age = (users["Age"] < 5) | (users["Age"] > 100)
    users.loc[invalid_age, "Age"] = pd.NA

    # Replace missing locations with "Unknown" placeholders.
    users["Location"] = users["Location"].replace("", pd.NA).fillna("Unknown")

    users = users.reset_index(drop=True)
    return users



def clean_all_data():
    """
    Clean all data and save to processed data directory.
    """
    raw_books, raw_ratings, raw_users = load_data()

    # Clean the data
    cleaned_books = clean_books(raw_books)
    cleaned_users = clean_users(raw_users)

    # Create the processed data directory if it doesn't exist and save the cleaned data.
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    cleaned_books.to_csv(processed_data_dir / "books_clean.csv", index=False)
    cleaned_users.to_csv(processed_data_dir / "users_clean.csv", index=False)
    # Print before-and-after cleaning.
    print("\nCleaning summary")
    print("=" * 60)
    print(f"Books: {raw_books.shape} -> {cleaned_books.shape}")
    print(f"Users: {raw_users.shape} -> {cleaned_users.shape}")
    print("\nCleaned files saved to:")
    print(processed_data_dir)



if __name__ == "__main__":
    clean_all_data()

