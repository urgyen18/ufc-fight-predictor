from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.predict import predict_fight
from app.web import router as web_router

app = FastAPI(title="UFC Fight Predictor")

class PredictRequest(BaseModel):
    red: str
    blue: str

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
