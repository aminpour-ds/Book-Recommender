# Book Recommender

The project will include:

- data loading and cleaning
- popularity-based recommendations
- content-based recommendations
- collaborative filtering
- FastAPI
- Streamlit
- evaluation
- robustness testing
- experiment logging


## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
```

## Data

This project uses the Book-Crossing dataset (book ratings, book metadata, and user information), as uploaded on Kaggle. The data is not included in this repository, you need to download it first.

- **Source:** https://www.kaggle.com/datasets/ra4u12/bookrecommendation
- **Version:** Version 1 (120.75 MB)
- **License:** GPL 2

### 1. Download the raw data

1. Open the Kaggle link above.
2. Click **Download** and unzip the archive.
3. Put the CSV files into `data/raw/`:
   - `BX-Book-Ratings.csv`
   - `BX-Books.csv`
   - `BX-Users.csv`

Rename the files to : Ratings, Books, Users

### 2. Create the processed data

Install the dependencies and run the data_cleaning script. This writes the cleaned files to `data/processed/`:

Run all commands from the project root (the folder that contains `src/` and `api/`).
```bash
pip install -r requirements.txt
python -m src.data_cleaning
```

### 3. Train / run the recommender

```bash
python -m src.train_models
```

**Make sure the model exists in the `models/` folder before running the server. You can check the contents of the folder with:**
```bash
Get-ChildItem models
```

### 4. Run the server/fastapi

```bash
uvicorn api.main:app --reload
```

The API is then available at `http://127.0.0.1:8000`.

### Interactive documentation

Open `http://127.0.0.1:8000/docs` for the full, interactive API documentation, where you can also try the endpoints in the browser.

### Endpoints

| Method | Path         | Description                                    |
|--------|--------------|------------------------------------------------|
| GET    | `/`          | Basic information about the API                |
| GET    | `/health`    | Health check (whether the model loaded)        |
| POST   | `/recommend` | Books similar to a given title                 |


### Example request:

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"title": "potter", "top_k": 2}'
```

### Example response:

```json
{
  "query": "potter",
  "recommendations": [
    {
      "ISBN": "059035342X",
      "title": "Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))",
      "author": "J. K. Rowling",
      "similarity": 8327
    },
    {
      "ISBN": "0590353403",
      "title": "Harry Potter and the Sorcerer's Stone (Book 1)",
      "author": "J. K. Rowling",
      "similarity": 8123
    }
  ]
}
```