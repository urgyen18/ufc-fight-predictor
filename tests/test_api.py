from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_fighters_autocomplete():
    r = client.get("/fighters", params={"q": "isla", "limit": 20})
    assert r.status_code == 200
    fighters = r.json()["fighters"]
    assert any("Islam" in f for f in fighters)

def test_predict_rejects_unknown_fighter():
    r = client.post("/predict", json={"red": "Not A Fighter", "blue": "Islam Makhachev"})
    assert r.status_code == 400
