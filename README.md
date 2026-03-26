# UFC Fight Predictor (Elo + XGBoost + FastAPI)

A UFC fight prediction API + minimal UI that returns the probability of the **red** fighter winning a matchup.
Built to showcase **ML + backend engineering** (feature engineering, evaluation, REST API, autocomplete UI).

## Features

- **Elo baseline** built from historical fight results
- **XGBoost model** using `elo_diff` + physical diffs (reach/height/weight)
- **FastAPI** backend with:
  - `POST /predict` → win probability + features used
  - `GET /fighters?q=` → autocomplete search for fighter names
  - `GET /health` → health check
- Minimal HTML UI at `/` + Swagger docs at `/docs`

## Model Performance (test set)

- Accuracy: ~0.77
- ROC AUC: ~0.84
- Brier score: ~0.15

## Project Structure

```txt
ufc-fight-predictor/
├── app/
│   ├── api.py        # FastAPI routes (/predict, /fighters, /health)
│   ├── predict.py    # model + feature assembly (production logic)
│   └── web.py        # minimal UI served at "/"
├── models/
│   ├── elo_ratings.json
│   ├── model_features.joblib
│   └── xgb_fight_predictor.joblib
├── notebooks/
│   └── ml_train.ipynb
├── data/             # local-only (ignored by git)
├── requirements.txt
└── README.md
```
## Local Setup
1) Create & activate virtual environment
python -m venv .venv
source .venv/bin/activate

2) Install dependencies
pip install -r requirements.txt

3) Add dataset (local only)

Place the CSV here:

data/UFC_full_data_golden.csv


Note: data/ is ignored by git and is not committed.

4) Run the server
uvicorn app.api:app --reload


Open in browser:

UI → http://127.0.0.1:8000/

Swagger Docs → http://127.0.0.1:8000/docs

## API Usage
GET /fighters?q=

Autocomplete fighter names.

Example:

curl "http://127.0.0.1:8000/fighters?q=isla&limit=10"

## POST /predict

Request:

curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"red":"Jon Jones","blue":"Islam Makhachev"}'


Response (example):

{
  "red": "Jon Jones",
  "blue": "Islam Makhachev",
  "prob_red_wins": 0.98,
  "features": {
    "elo_diff": 41.39,
    "reach_diff": 35.56,
    "height_diff": 15.24,
    "weight_diff": 93.0
  }
}

## Tests

Run:

pytest -q
