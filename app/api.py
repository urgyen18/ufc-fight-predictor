from fastapi import FastAPI, HTTPException, Query
import pandas as pd
from pathlib import Path
from pydantic import BaseModel
from app.predict import predict_fight
from app.web import router as web_router

app = FastAPI(title="UFC Fight Predictor")

class PredictRequest(BaseModel):
    red: str
    blue: str

RAW = Path(__file__).resolve().parents[1] / "data" / "UFC_full_data_golden.csv"
COL_RED = "f_1_name"
COL_BLUE = "f_2_name"

# Load once at startup (fast)
_df = pd.read_csv(RAW, usecols=[COL_RED, COL_BLUE])
_FIGHTERS = sorted(set(_df[COL_RED].dropna().astype(str).str.strip())
                   | set(_df[COL_BLUE].dropna().astype(str).str.strip()))

@app.get("/fighters")
def fighters(q: str = Query("", max_length=50), limit: int = 10):
    qn = q.strip().lower()
    if not qn:
        return {"fighters": _FIGHTERS[:limit]}
    matches = [name for name in _FIGHTERS if qn in name.lower()]
    return {"fighters": matches[:limit]}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        return predict_fight(req.red, req.blue)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Serve the UI at "/"
app.include_router(web_router)
