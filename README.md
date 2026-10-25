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

Install the dependencies and run the cleaning scriptThis writes the cleaned files to `data/processed/`:

```bash
pip install -r requirements.txt
python src/data_cleaning.py
```

### 3. Train / run the recommender

```bash
python src/train_models.py
```