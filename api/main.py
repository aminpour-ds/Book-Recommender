import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from src.config import content_based_model, popularity_model
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.popularity import PopularityRecommender
from src.config import Ratings_clean


# Startup function. FastAPI calls this function once when the server starts.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """it loads models and user's Ratings data during API startup once, when the API starts."""
    if not content_based_model.exists():
        raise RuntimeError(f"Model file not found: {content_based_model}. Run `python -m src.train_models` in terminal first.")
    if not popularity_model.exists():
        raise RuntimeError(f"Model file not found: {popularity_model}. Run `python -m src.train_models` in terminal first.")
    app.state.content_model = ContentBasedRecommender.load(content_based_model)
    app.state.popularity_model = PopularityRecommender.load(popularity_model)
    ratings = pd.read_csv(Ratings_clean)
    # user lookup is fast for every request.
    app.state.rated_user_ids = set(ratings["User-ID"].astype(str))

    yield


# defines what JSON the client must send to the API when requesting recommendations.
class RecommendationRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True) # setting for cleanup the string_input inside the request model not with strip()
    user_id: str | None = Field(default=None, min_length=1)
    title: str | None = Field(default=None, min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


# The shape of one result of a list of recommendations. 
class BookRecommendation(BaseModel):
    ISBN: str
    title: str
    author: str
    similarity: float | None = None
    predicted_rating: float | None = None
    image_url: str = ""
    rating_count: int | None = None


# The shape of the JSON response returned by the API when recommendations are requested.
class RecommendationResponse(BaseModel):
    query: str | None
    user_id: str | None
    cold_start: bool
    source: str     # Possible values for source: content-based | popularity
    message: str | None = None
    recommendations: list[BookRecommendation]

# Creating the app on "/docs". The FastAPI app instance/The main entry point for the API.
app = FastAPI(
    title="Book Recommender API",
    description="Content-based book recommendations using TF-IDF.",
    version="1.0.0",
    lifespan=lifespan, # registers the startup function
)


# Endpoints
@app.get("/")
def root():
    """Return basic API information."""
    return {
        "message": "Book Recommender API is running.",
        "models": ["content-based", "popularity"],
    }


# Health-check endpoint are used by Docker and returns 200 if the API is running and the model is loaded.
@app.get("/health")
def health(request: Request):
    """
    Health-check endpoint. The health response becomes with "model_loaded": true if the models are loaded and "status": "ok".
    """
    return {
        "status": "ok",
        "model_loaded": (hasattr(request.app.state, "content_model") and hasattr(request.app.state, "popularity_model")),
    }


def build_API_recommendations(recommendations, source):
    """
    Convert recommendation results to the API response shape.
    """
    result = []

    for book in recommendations:
        if source == "popularity":
            recommendation = BookRecommendation(
                ISBN=str(book["ISBN"]),
                title=str(book["Book-Title"]),
                author=str(book["Book-Author"]),
                similarity=None,
                predicted_rating=float(book["average_rating"]),
                image_url=str(book.get("Image-URL-M", "")),
                rating_count=int(book["rating_count"]),
            )
        elif source == "content-based":
            recommendation = BookRecommendation(
                ISBN=str(book["ISBN"]),
                title=str(book["Book-Title"]),
                author=str(book["Book-Author"]),
                similarity=float(book["similarity"]),
                predicted_rating=None,
                image_url=str(book.get("Image-URL-M", "")),
            )
        else:
            raise ValueError(f"Unsupported recommendation source: {source}")

        result.append(recommendation)

    return result


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(payload: RecommendationRequest, request: Request):
    """
    Return content-based or popularity recommendations according to user and title state.
    """
    
    # Gets the model that was loaded at startup.
    content_model = request.app.state.content_model
    popularity_model = request.app.state.popularity_model
    rated_user_ids = request.app.state.rated_user_ids

    is_known_user = (payload.user_id is not None and payload.user_id in rated_user_ids)
    is_cold_start = not is_known_user
    title_exists = (payload.title is not None and content_model.find_book_index(payload.title) is not None)

    if is_cold_start and not title_exists:
        message = (
            "Cold start: showing popular books."
            if payload.title is None
            else (f"No book found for '{payload.title}'. Showing popular books instead.")
        )
        return RecommendationResponse(
            query=payload.title,
            user_id=payload.user_id,
            cold_start=True,
            source="popularity",
            message=message,
            recommendations=build_API_recommendations(popularity_model.recommend(top_k=payload.top_k), source="popularity",),
        )

    if is_known_user and not title_exists:
        raise HTTPException(
            status_code=400,
            detail="Personalized recommendations are not implemented yet.",
        )

    # If we reach this point, it means that the title exists. We can provide content-based recommendations.
    recommendations = content_model.recommend(title=payload.title, top_k=payload.top_k)
    # create the response object with the original query and the list of recommendations.
    return RecommendationResponse(
        query=payload.title,
        user_id=payload.user_id,
        cold_start=is_cold_start,
        source="content-based",
        message=None,
        recommendations=build_API_recommendations(recommendations, source="content-based",)
    )
