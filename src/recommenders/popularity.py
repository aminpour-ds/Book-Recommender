from src.config import TOP_K

class PopularityRecommender:
    """Recommend books according to their number of ratings."""

    def __init__(self):
        self.popular_books = None

    def fit(self, ratings, books):
        """
        Books are ranked by number of ratings. When two books have the same
        number of ratings, the one with the higher average rating comes first,
        and then the title (A to Z) is used to keep the order.
        """
        missing_rating_columns = {"ISBN", "Book-Rating"} - set(ratings.columns)
        if missing_rating_columns:
            raise ValueError(f"Ratings data is missing columns: {missing_rating_columns}")

        missing_book_columns = {"ISBN", "Book-Title", "Book-Author"} - set(books.columns)
        if missing_book_columns:
            raise ValueError(f"Books data is missing columns: {missing_book_columns}")

        # adding Number of ratings and average rating for each book(ISBN).
        rating_stats = (ratings.groupby("ISBN")["Book-Rating"].agg(rating_count="size", average_rating="mean").reset_index())
        rating_stats["average_rating"] = rating_stats["average_rating"].round(2)

        # merge rating_stats with books to add title and author of each book.
        popular_books = rating_stats.merge(books, on="ISBN", how="inner")

        # Most ratings first, books with the same rating sorted by average rating, then by title.
        popular_books = popular_books.sort_values(by=["rating_count", "average_rating", "Book-Title"], ascending=[False, False, True])

        self.popular_books = popular_books.reset_index(drop=True)

        return self


    def recommend(self, top_k=TOP_K):
        """
        Return the most popular top_k books.
        A list of dictionaries with ISBN, title, author, rating_count and average_rating 
        """
        if self.popular_books is None:
            raise RuntimeError("The recommender has not been fitted. Call fit() before recommend().")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        recommendations = self.popular_books.head(top_k)

        return recommendations[["ISBN", "Book-Title", "Book-Author", "rating_count", "average_rating"]].to_dict(orient="records")