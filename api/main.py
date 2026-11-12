from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from src.config import content_based_model
from src.recommenders.content_based import ContentBasedRecommender


# Startup function. FastAPI calls this function once when the server starts.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model once, when the API starts."""
    if not content_based_model.exists():
        raise RuntimeError(f"Model file not found: {content_based_model}. Run `python -m src.train_models` in terminal first.")
    app.state.model = ContentBasedRecommender.load(content_based_model)
    yield


# defines what JSON the client must send to the API when requesting recommendations.
class RecommendationRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    title: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


# The shape of one result of a list of recommendations. 
class BookRecommendation(BaseModel):
    ISBN: str
    title: str
    author: str
    similarity: float
    image_url: str = ""


# The shape of the JSON response returned by the API when recommendations are requested.
class RecommendationResponse(BaseModel):
    query: str
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
        "model": "content-based",
    }


# Health-check endpoint are used by Docker and returns 200 if the API is running and the model is loaded.
@app.get("/health")
def health(request: Request):
    """Health-check endpoint."""
    return {
        "status": "ok",
        "model_loaded": hasattr(request.app.state, "model"),
    }


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(payload: RecommendationRequest, request: Request):
    """Return books similar to the requested title."""
    
    # Gets the model that was loaded at startup.
    model = request.app.state.model

    # Unknown title return 404. 
    if model.find_book_index(payload.title) is None:
        raise HTTPException(status_code=404, detail=f"No book found for title: {payload.title}",)

    recommendations = model.recommend(title=payload.title, top_k=payload.top_k)

    # convert every recommended-book dictionary into a BookRecommendation object and put all converted objects into a list.
    recommendation_objects = []
    for book in recommendations:
        recommendation = BookRecommendation(
            ISBN=str(book["ISBN"]),
            title=str(book["Book-Title"]),
            author=str(book["Book-Author"]),
            similarity=float(book["similarity"]),
            image_url=str(book.get("Image-URL-M", "")),
        )
        recommendation_objects.append(recommendation)

    # create the response object with the original query and the list of recommendations.
    response = RecommendationResponse(
        query=payload.title,
        recommendations=recommendation_objects,
    )

    return response