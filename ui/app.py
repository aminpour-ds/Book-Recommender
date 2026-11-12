import requests
import streamlit as st
from src.config import API_URL


# Width of the book-cover images in the UI.
COVER_WIDTH = 70

# Amazon serves a tiny 1x1 pixel instead of a real book cover. 
# Images smaller than 1,000 bytes are treated as invalid.
MIN_COVER_BYTES = 1000

# Amazon may reject requests that do not look like a browser. so we add User-Agent header 
# to the "get" http request and send for Amazon's image server.
COVER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}



# Streamlit decorator tells Streamlit to cache the function's returned data.
@st.cache_data(show_spinner=False, ttl=3600)
def resolve_cover(url):
    """
    Return image bytes from Amazon's image server to display for a book cover.

    url: is the book's image link that is downloadable from Amazon's image server.

    'requests.get': opens the url link, so Amazon receives the request and sends the image 
    bytes back. The FastAPI is not involved at all. This is why the User-Agent header 
    matters here: Amazon is the server that decides whether to accept the request.

    Here the URL is downloaded and checked to be a real image. If the URL is empty, unreachable, 
    an error, or the tiny "missing" pixel, the placeholder image from make_placeholder() is returned instead.
    """
    if not url:
        return None

    try:
        response = requests.get(url, headers=COVER_HEADERS, timeout=4)
    except requests.exceptions.RequestException:
        return None

    if response.status_code != 200:
        return None

    if not response.headers.get("Content-Type", "").startswith("image"):
        return None

    if len(response.content) < MIN_COVER_BYTES:
        return None

    return response.content



def get_recommendations(title, top_k):
    """
    Ask the backend for top_k similar books of title. 
    Returns two values when a network error occurs:
    1. None, because no recommendation data was received;
    2. An error message for the UI.
    """
    try:
        response = requests.post(f"{API_URL}/recommend", json={"title": title, "top_k": top_k}, timeout=30,)
    except requests.exceptions.RequestException:
        return None, (f"Could not reach the backend at {API_URL}. Make sure the FastAPI server is running.")

    if response.status_code == 404:
        return None, "No book was found with that title. Try a different spelling or part of the title."

    if response.status_code != 200:
        return None, f"Backend returned an error (status {response.status_code})."

    # response.json() converts the backend's JSON response into a Python dictionary; None means there is no error.
    return response.json(), None    



def show_recommendations(data):
    """
    Display the recommendation Books.
    data: contains the recommendation response returned by the FastAPI backend.
    """

    recommendations = data["recommendations"]

    if not recommendations:
        st.info("This book was found, but no similar books were returned.")
        return            # Stops the function immediately because there are no books to display.

    # start=1 makes the numbering begin at 1 instead of 0.
    for rank, book in enumerate(recommendations, start=1):
        cover_column, details_column = st.columns([1, 5])       # Creates two Streamlit columns. first column with width ratio 1 and the second column 5.

        with cover_column:
            cover = resolve_cover(book.get("image_url", ""))
            if cover is None:
                st.caption("No cover")
            else:
                st.image(cover, width=COVER_WIDTH)

        with details_column:
            st.markdown(f"**{rank}. {book['title']}**")
            st.write(f"by {book['author']}")
            st.write(f"Similarity: {book['similarity']:.4f}")      # :.4f formats it as a decimal number

        st.divider()     # Displays a horizontal dividing line after each recommendation for separating the books.



# ---------------------------------------------------------------------------
# Page layout.
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Book Recommender", layout="centered")

st.title("Book Recommender")
st.write("Find books similar to a title you already like.")

# Sidebar: settings.
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Number of recommendations", min_value=1, max_value=10, value=5)

with st.form("content_form"):
    title = st.text_input("Book title", placeholder="e.g. The Da Vinci Code")
    submitted = st.form_submit_button("Recommend similar books")   # If the user clicks the submitted, it becomes True.

if submitted:
    if not title.strip():
        st.warning("Please enter a book title.")
    else:
        with st.spinner("Searching for similar books..."):    # Displays a loading message while the backend request is running.
            data, error = get_recommendations(title.strip(), top_k)

        if error is not None:
            st.error(error)
        else:
            show_recommendations(data)