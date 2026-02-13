import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import math

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "xgb_fight_predictor.joblib"
FEATS_PATH = ROOT / "models" / "model_features.joblib"
ELO_PATH = ROOT / "models" / "elo_ratings.json"
FIGHTER_STATS_PATH = ROOT / "models" / "fighter_stats.json"

BASE_ELO = 1500.0


# ---------- Helpers ----------
def norm_name(s: str) -> str:
    """Normalize names for lookups (case/whitespace-insensitive)."""
    return str(s).strip().lower()


def _clean_json(x):
    """Convert numpy/scalars/NaN to JSON-safe Python values."""
    if isinstance(x, (np.floating, np.integer)):
        x = x.item()

    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None

    if isinstance(x, dict):
        return {k: _clean_json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean_json(v) for v in x]

    return x


# ---------- Load artifacts ----------
def load_model():
    model = joblib.load(MODEL_PATH)
    feats = joblib.load(FEATS_PATH)
    return model, feats


def load_elo():
    """
    Loads Elo ratings from models/elo_ratings.json
    Expected format: [{"fighter": "...", "elo": 1234.5}, ...]
    Returns dict: normalized_name -> elo_float
    """
    if not ELO_PATH.exists():
        return {}
    with open(ELO_PATH, "r") as f:
        items = json.load(f)
    out = {}
    for row in items:
        name = norm_name(row.get("fighter", ""))
        if not name:
            continue
        try:
            out[name] = float(row.get("elo", BASE_ELO))
        except Exception:
            out[name] = BASE_ELO
    return out


def load_fighter_stats():
    """
    Loads fighter stats from models/fighter_stats.json
    Expected format: { "Jon Jones": {"reach_cm":..., "height_cm":..., "weight_lbs":...}, ... }
    Returns dict with BOTH:
      - original keys (as-is)
      - normalized keys (lower/strip)
    so lookups work even if casing differs.
    """
    if not FIGHTER_STATS_PATH.exists():
        return {}

    with open(FIGHTER_STATS_PATH, "r") as f:
        raw = json.load(f)

    # Build a normalized lookup too
    normed = {}
    for k, v in raw.items():
        nk = norm_name(k)
        if nk and nk not in normed:
            normed[nk] = v

    # merge: exact-key lookups still work, and normalized lookups work too
    merged = dict(raw)
    for nk, v in normed.items():
        merged[nk] = v

    return merged


def _get_num(stats: dict | None, key: str) -> float:
    """Return float value for stats[key], or np.nan if missing."""
    if not stats:
        return np.nan
    v = stats.get(key)
    if v is None:
        return np.nan
    try:
        return float(v)
    except Exception:
        return np.nan


# ---------- Main prediction ----------
def predict_fight(red_name: str, blue_name: str):
    """
    Returns dict with:
      - prob_red_wins
      - features used (elo_diff, reach/height/weight diffs)
    """
    model, feats = load_model()
    elo = load_elo()
    stats = load_fighter_stats()

    red_name = str(red_name).strip()
    blue_name = str(blue_name).strip()

    # Elo features (normalized lookup)
    elo_red = elo.get(norm_name(red_name), BASE_ELO)
    elo_blue = elo.get(norm_name(blue_name), BASE_ELO)

    features = {"elo_diff": elo_red - elo_blue}

    # Physical diffs (from fighter_stats.json)
    red_stats = stats.get(red_name) or stats.get(norm_name(red_name))
    blue_stats = stats.get(blue_name) or stats.get(norm_name(blue_name))

    features["reach_diff"] = _get_num(red_stats, "reach_cm") - _get_num(blue_stats, "reach_cm")
    features["height_diff"] = _get_num(red_stats, "height_cm") - _get_num(blue_stats, "height_cm")
    features["weight_diff"] = _get_num(red_stats, "weight_lbs") - _get_num(blue_stats, "weight_lbs")

    # Build X in the exact feature order used in training
    X = pd.DataFrame([{f: features.get(f, np.nan) for f in feats}])

    # Handle missing values (simple fill; later we can do better)
    X = X.fillna(0)

    prob_red = float(model.predict_proba(X)[0, 1])

    out = {
        "red": red_name,
        "blue": blue_name,
        "prob_red_wins": prob_red,
        "features": features,
    }

    return _clean_json(out)


# ---------- Quick local test ----------
if __name__ == "__main__":
    out = predict_fight("Jon Jones", "islam makhachev")
    print(out)
