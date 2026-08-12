from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import pandas as pd

from app.predict import predict_fight
from app.web import router as web_router

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGHTERS_PATH = ROOT / "models" / "fighters.json"

with open(FIGHTERS_PATH, "r") as f:
    FIGHTERS = json.load(f)

FIGHTERS_SET = set(FIGHTERS)


app = FastAPI(title="UFC Fight Predictor")


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


app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")

# Serve the UI at "/"
app.include_router(web_router)
