from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from pathlib import Path
import pandas as pd

from app.predict import predict_fight
from app.web import router as web_router

app = FastAPI(title="UFC Fight Predictor")

# ---- Load fighter list once at startup ----
RAW = Path(__file__).resolve().parents[1] / "data" / "UFC_full_data_golden.csv"
COL_RED = "f_1_name"
COL_BLUE = "f_2_name"

_df_names = pd.read_csv(RAW, usecols=[COL_RED, COL_BLUE])

FIGHTERS = sorted(
    set(_df_names[COL_RED].dropna().astype(str).str.strip())
    | set(_df_names[COL_BLUE].dropna().astype(str).str.strip())
)
FIGHTERS_SET = set(FIGHTERS)  # fast membership check


@app.get("/fighters")
def fighters(q: str = Query("", max_length=50), limit: int = 15):
    qn = q.strip().lower()
    if not qn:
        return {"fighters": FIGHTERS[:limit]}
    matches = [name for name in FIGHTERS if qn in name.lower()]
    return {"fighters": matches[:limit]}


class PredictRequest(BaseModel):
    red: str
    blue: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    red = req.red.strip()
    blue = req.blue.strip()

    # Optional but recommended: validate names so users can’t type random strings
    if red not in FIGHTERS_SET:
        raise HTTPException(status_code=400, detail=f"Unknown fighter: '{red}'. Use /fighters to search.")
    if blue not in FIGHTERS_SET:
        raise HTTPException(status_code=400, detail=f"Unknown fighter: '{blue}'. Use /fighters to search.")

    try:
        return predict_fight(red, blue)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Serve the UI at "/"
app.include_router(web_router)
