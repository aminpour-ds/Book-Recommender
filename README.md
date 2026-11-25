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

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
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
```powershell
pip install -r requirements.txt
python -m src.data_cleaning
```

### 3. Train / run the recommender

```powershell
python -m src.train_models
```

**Make sure the model exists in the `models/` folder before running the server. You can check the contents of the folder with:**
```powershell
Get-ChildItem models
```

### 4. Run the server/fastapi

## Open one terminal and run:

```powershell
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

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/recommend `
  -ContentType "application/json" `
  -Body '{"title": "potter", "top_k": 2}'
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

### 5. Start Streamlit UI

## Open a second terminal in the project root and run: 

```powershell
python -m streamlit run ui/app.py
```

Streamlit should open a browser page:
http://localhost:8501


### 6. Evaluate the models

Evaluate the models on a per-user train/test split and print Precision@k, Recall@k, Hit Rate@k, MAP, MRR, and NDCG.
The reports are saved as CSV files in `reports/`:

```powershell
python -m src.evaluate_models
```


### 7. Log evaluation experiments with Weights & Biases (wandb)

Run the evaluation and W&B logging script:

```powershell
python -m src.wandb_logging_run
```

This reruns the evaluation pipeline and creates one W&B run for each model.
Each run logs the evaluation configuration, the metrics at every value in
`K_VALUES` (currently 5, 10, and 20), and the four evaluation reports as tables.

W&B runs in `offline` mode by default, so the data is saved only under `wandb/`.
To see the runs online, log in once and then either sync the offline runs:

```powershell
wandb login
wandb sync --sync-all
```

or log directly online:

```powershell
$env:WANDB_MODE="online"
python -m src.wandb_logging_run
```
