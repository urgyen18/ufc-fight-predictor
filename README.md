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
