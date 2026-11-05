import pandas as pd
from src.data_Loading import load_data
from src.config import processed_data_dir

CURRENT_YEAR = 2026

# ---------------------------------------------------------------------------
# Clean Books
# ---------------------------------------------------------------------------
def clean_books(raw_books):
    """
    Clean the books data.

    Rules:
    - Strip whitespace from column names (the headers) and column fields (the data).
    - Remove rows without ISBN or title.
    - Remove duplicate ISBNs.(Deduplicate after removing bad rows)
    - Convert publication year to a numeric value.
    - Replace invalid and missing years as <NA>
    - Replace missing authors and publishers with "Unknown" placeholders.
    - missing/NaN URLs in the image columns become an empty string.
    """
    books = raw_books.copy()

    # Strip whitespace from column names (the headers)
    books.columns = books.columns.str.strip()
    # Strip whitespace from column fields (the data)
    books["ISBN"] = books["ISBN"].str.strip().str.upper()
    books["Book-Title"] = books["Book-Title"].str.strip()
    books["Book-Author"] = books["Book-Author"].str.strip()
    books["Publisher"] = books["Publisher"].str.strip()
    books["Image-URL-S"] = books["Image-URL-S"].str.strip()
    books["Image-URL-M"] = books["Image-URL-M"].str.strip()
    books["Image-URL-L"] = books["Image-URL-L"].str.strip()
    books["Year-Of-Publication"] = books["Year-Of-Publication"].str.strip()

    # Remove rows without ISBN or title or with empty strings in those fields
    books = books.dropna(subset=["ISBN", "Book-Title"])
    books = books[books["ISBN"] != ""]
    books = books[books["Book-Title"] != ""]

    # Remove duplicate ISBNs
    books = books.drop_duplicates(subset="ISBN", keep="first")

    # Convert publication year to a numeric value.
    books["Year-Of-Publication"] = pd.to_numeric(books["Year-Of-Publication"], errors="coerce").astype("Int64") 

    # Replace invalid and missing years as <NA>
    invalid_year = (books["Year-Of-Publication"] < 1000) | (books["Year-Of-Publication"] > CURRENT_YEAR)
    books.loc[invalid_year, "Year-Of-Publication"] = pd.NA
    
    # Replace missing authors and publishers with "Unknown" placeholders.
    books["Book-Author"] = books["Book-Author"].replace("", pd.NA).fillna("Unknown")
    books["Publisher"] = books["Publisher"].replace("", pd.NA).fillna("Unknown")

    # missing/NaN URLs in the image columns become an empty string.
    books["Image-URL-S"] = books["Image-URL-S"].fillna("")
    books["Image-URL-M"] = books["Image-URL-M"].fillna("")
    books["Image-URL-L"] = books["Image-URL-L"].fillna("")

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


# ---------------------------------------------------------------------------
# Clean Ratings
# ---------------------------------------------------------------------------
def clean_ratings(ratings, books, users):
    """
    Clean the Ratings table.

    Rules:
    - Strip whitespace from column names and fields
    - Convert User-ID and Book-Rating to numbers
    - Remove rows with invalid user IDs
    - Remove rows without user IDs or ISBN or Book-Rating
    - Remove duplicate rows with same "User-ID and ISBN", keep the last occurrence.
    - Remove rows with invalid ratings (below 1 or above 10).
    - Keep only ratings whose book (ISBN) and user actually exist.
    """
    ratings = ratings.copy()

    # Strip whitespace from column names (the headers)
    ratings.columns = ratings.columns.str.strip()
    # Strip whitespace from column fields (the data)
    ratings["User-ID"] = ratings["User-ID"].str.strip()
    ratings["ISBN"] = ratings["ISBN"].str.strip().str.upper()
    ratings["Book-Rating"] = ratings["Book-Rating"].str.strip()

    # Convert User-ID and Book-Rating to numbers
    ratings["User-ID"] = pd.to_numeric(ratings["User-ID"], errors="coerce").astype("Int64")
    ratings["Book-Rating"] = pd.to_numeric(ratings["Book-Rating"], errors="coerce").round().astype("Int64")

    # Remove rows with invalid user IDs
    ratings = ratings[ratings["User-ID"] > 0]

    # Remove rows without user IDs or ISBN or Book-Rating
    ratings = ratings.dropna(subset=["User-ID", "ISBN", "Book-Rating"])
    
    # Remove duplicate rows with same "User-ID and ISBN", keep the last occurrence.
    ratings = ratings.drop_duplicates(subset=["User-ID", "ISBN"], keep="last")
        
    # Remove rows with invalid ratings (below 1 or above 10)
    invalid_rating = (ratings["Book-Rating"] < 1) | (ratings["Book-Rating"] > 10)
    ratings = ratings[~invalid_rating]

    # Keep only ratings whose book(ISBN) and user actually exist
    ratings = ratings[ratings["ISBN"].isin(books["ISBN"]) & ratings["User-ID"].isin(users["User-ID"])]

    ratings = ratings.reset_index(drop=True)
    return ratings



# ---------------------------------------------------------------------------
# filter sparse matrix
# ---------------------------------------------------------------------------
def filter_sparse(ratings):
    """
    for solving the sparsity problem and optimizing the collaborative model, 
    filter out very inactive users and rarely rated books.

    Rules:
    - Remove users with fewer than 5 ratings.
    - Remove books with fewer than 5 ratings.
    - Repeat until no more users or books can be removed.
    """
    ratings = ratings.copy()
    min_user = 5   # a user must have rated at least 5 books, otherwise they're dropped.
    min_book = 5   # a book must have been rated by at least 5 users, otherwise it's dropped.

    while True:
        # Keep users with enough ratings.
        user_counts = ratings["User-ID"].value_counts()
        users_to_remove = user_counts[user_counts < min_user].index
        ratings = ratings[~ratings["User-ID"].isin(users_to_remove)]

        # Keep books with enough ratings.
        book_counts = ratings["ISBN"].value_counts()
        books_to_remove = book_counts[book_counts < min_book].index
        ratings = ratings[~ratings["ISBN"].isin(books_to_remove)]

        # If no more users or books to remove, break the loop
        if len(users_to_remove) == 0 and len(books_to_remove) == 0:
            break      
        

    ratings = ratings.reset_index(drop=True)
    return ratings



def clean_all_data():
    """
    Clean all data and save to processed data directory.
    """
    raw_users, raw_books, raw_ratings  = load_data()

    # Clean the data
    cleaned_books = clean_books(raw_books)
    cleaned_users = clean_users(raw_users)
    cleaned_ratings = clean_ratings(raw_ratings, cleaned_books, cleaned_users)
    cleaned_final_ratings = filter_sparse(cleaned_ratings)

    # Drop books and users left without ratings after filter_sparse, so the three tables stay consistent
    # and embedding tables have no unused entries that waste memory and never get trained
    cleaned_final_books = cleaned_books[cleaned_books["ISBN"].isin(cleaned_final_ratings["ISBN"])].reset_index(drop=True)
    cleaned_final_users = cleaned_users[cleaned_users["User-ID"].isin(cleaned_final_ratings["User-ID"])].reset_index(drop=True)

    # Create the processed data directory if it doesn't exist and save the cleaned data.
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    cleaned_final_books.to_csv(processed_data_dir / "books_clean.csv", index=False)
    cleaned_final_users.to_csv(processed_data_dir / "users_clean.csv", index=False)
    cleaned_final_ratings.to_csv(processed_data_dir / "ratings_clean.csv", index=False)

    # Print before-and-after cleaning.
    print("\nCleaning summary")
    print("=" * 60)
    print(f"Books:   {raw_books.shape} -> cleaned {cleaned_books.shape} -> final {cleaned_final_books.shape}")
    print(f"Users:   {raw_users.shape} -> cleaned {cleaned_users.shape} -> final {cleaned_final_users.shape}")
    print(f"Ratings: {raw_ratings.shape} -> cleaned {cleaned_ratings.shape} -> final {cleaned_final_ratings.shape}")
    print("\nCleaned files saved to:")
    print(processed_data_dir)



if __name__ == "__main__":
    clean_all_data()

