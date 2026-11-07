import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import TOP_K

class ContentBasedRecommender:
    """Recommend books whose title, author and publisher text is similar."""

    def __init__(self):
        self.books = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.title_keys = None
        self.title_to_index = None


    @staticmethod
    def build_text(books):
        """
        This method doesn't need access to the recommender object or its class, 
        it is an independent utility and we define it as a static method. 
        Combine title (twice), author and publisher into one text per book. return a list of strings.
        """
        title = books["Book-Title"].fillna("")
        author = books["Book-Author"].fillna("")
        publisher = books["Publisher"].fillna("")
        return (title + " " + title + " " + author + " " + publisher).tolist()


    def fit(self, books):
        """
        Turn string list per book (title + title + author + publisher) into TF-IDF numbers/vectors.
        """
        required = {"ISBN", "Book-Title", "Book-Author", "Publisher"}
        missing = required - set(books.columns)
        if missing:
            raise ValueError(f"Books data is missing columns: {missing}")

        self.books = books.reset_index(drop=True)
        # stop_words="english": removes common English words(“the”, “and”, “of”,...)
        # min_df=2: ignore words that appear in only one book. 
        self.vectorizer = TfidfVectorizer(stop_words="english", min_df=2)        
        book_texts = self.build_text(self.books)                    # string list per book (title twice + author + publisher)
        self.vectorizer.fit(book_texts)                             # learn vocabulary and IDF values
        self.tfidf_matrix = self.vectorizer.transform(book_texts)   # convert texts to vectors. matrix is stored in sparse form(only non-zero values)

        # create a dictionary mapping Lowercased titles. Each key is a cleaned title and each value is the row number of that book. 
        self.title_keys = (self.books["Book-Title"].astype(str).str.strip().str.lower().tolist())
        self.title_to_index = {}
        for index, key in enumerate(self.title_keys):
            self.title_to_index.setdefault(key, index)

        return self


    def find_book_index(self, title):
        """Row number of a book: return the index of the book with the exact title match, otherwise the partial match will be used."""
        if title is None:
            return None
        
        key = str(title).strip().lower()

        if not key:
            return None

        if self.title_to_index is None:
            raise RuntimeError("The recommender has not been fitted.")

        # A dictionary lookup. Return the exact match.
        if key in self.title_to_index:
            return self.title_to_index[key]

        # The partial match will be used only when the exact match failed.
        for stored_title, index in self.title_to_index.items():
            if key in stored_title:
                return index

        # If no match is found, return None.
        return None


    def recommend(self, title, top_k=TOP_K):
        """
        Returns the books most similar to the given title, best first.
        Returns an empty list if the title is not in the catalogue.
        title is what the user typed. It can be partial, in other capitalization, or with extra spaces. 
        so we need to get the full cleaned title of the matched query book with "query_title".
        """
        if (self.tfidf_matrix is None or self.title_keys is None or self.books is None):
            raise RuntimeError("The recommender has not been fitted. Call fit() before recommend().")
        
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        # find the index of the book that we want to find similar books for it.
        index = self.find_book_index(title)
        if index is None:
            return []

        # find the similarity of this book with all other books. with flatten() converts the similarity matrix from 2D to 1D.
        similarities = cosine_similarity(self.tfidf_matrix.getrow(index), self.tfidf_matrix,).flatten()
        # ordered from the highest similarity to the lowest. 
        similarities = similarities.argsort()[::-1]

        # Get the full cleaned title of the matched query book.
        query_title = self.title_keys[index]   
        seen_titles = {query_title}

        # Walk through the books from most to least similar. Skip the query book and any books that have already been seen. Stop when we have enough results or when the similarity is zero or negative.
        results = []
        for i in similarities:
            # Stop the loop when you have enough results (top_k), or when the current book has no similarity at all (0)
            if len(results) == top_k or similarities[i] <= 0:
                break

            # If this book’s title was already used (the query itself, or an edition already chosen), skip it.
            if self.title_keys[i] in seen_titles:
                continue
            
            seen_titles.add(self.title_keys[i])
            book = self.books.iloc[i]
            results.append({
                "ISBN": book["ISBN"],
                "Book-Title": book["Book-Title"],
                "Book-Author": book["Book-Author"],
                "similarity": round(float(similarities[i]), 4),
            })
        
        return results


    def save(self, path):
        """Save the fitted ContentBasedRecommender object to a file."""
        joblib.dump(self, path)


    @staticmethod
    def load(path):
        """Load a previously saved recommender object from a file."""
        return joblib.load(path)