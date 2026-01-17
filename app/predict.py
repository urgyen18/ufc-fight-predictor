import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import math

RAW = Path(__file__).resolve().parents[1] / "data" / "UFC_full_data_golden.csv"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "xgb_fight_predictor.joblib"
FEATS_PATH = Path(__file__).resolve().parents[1] / "models" / "model_features.joblib"
ELO_PATH = Path(__file__).resolve().parents[1] / "models" / "elo_ratings.json"

BASE_ELO = 1500.0

# -- Normalize function --
def norm_name(s: str) -> str:
    return str(s).strip().lower()


# --- columns in your dataset (you used these earlier) ---
COL_RED  = "f_1_name"
COL_BLUE = "f_2_name"

# stats columns used for diffs (only if present in your dataset)
PAIR_COLS = {
    "reach_diff":  ("f_1_fighter_reach_cm",  "f_2_fighter_reach_cm"),
    "height_diff": ("f_1_fighter_height_cm", "f_2_fighter_height_cm"),
    "weight_diff": ("f_1_fighter_weight_lbs","f_2_fighter_weight_lbs"),
}

# ---------- Load artifacts ----------
def load_model():
    model = joblib.load(MODEL_PATH)
    feats = joblib.load(FEATS_PATH)
    return model, feats

def load_elo():
    if not ELO_PATH.exists():
        return {}
    with open(ELO_PATH, "r") as f:
        items = json.load(f)
    return {row["fighter"]: float(row["elo"]) for row in items}

def load_raw_df():
    return pd.read_csv(RAW)

# ---------- Helper: get latest fighter stats ----------
def latest_stats_for_fighter(df, fighter_name: str):
    """
    Returns a dict of latest known stats for a fighter from the dataset.
    We search rows where fighter appears as red or blue, and pull the newest row.
    """
    fighter_name = str(fighter_name).strip()

    # rows where fighter is in either slot
    fighter_norm = norm_name(fighter_name)
    f1 = df[COL_RED].astype(str).str.strip().str.lower()
    f2 = df[COL_BLUE].astype(str).str.strip().str.lower()
    mask = (f1 == fighter_norm) | (f2 == fighter_norm)
    sub = df.loc[mask].copy()
    if len(sub) == 0:
        return {}

    # try to sort by event_date if present
    if "event_date" in sub.columns:
        sub["event_date"] = pd.to_datetime(sub["event_date"], errors="coerce")
        sub = sub.dropna(subset=["event_date"]).sort_values("event_date")
    else:
        # fallback: keep file order
        pass

    row = sub.iloc[-1]  # newest

    # If fighter is red in this row, use f_1_ columns, else f_2_ columns
    is_red = norm_name(row[COL_RED]) == fighter_norm

    stats = {}
    if is_red:
        stats["reach_cm"]  = row.get("f_1_fighter_reach_cm", np.nan)
        stats["height_cm"] = row.get("f_1_fighter_height_cm", np.nan)
        stats["weight_lbs"]= row.get("f_1_fighter_weight_lbs", np.nan)
    else:
        stats["reach_cm"]  = row.get("f_2_fighter_reach_cm", np.nan)
        stats["height_cm"] = row.get("f_2_fighter_height_cm", np.nan)
        stats["weight_lbs"]= row.get("f_2_fighter_weight_lbs", np.nan)

    # Convert to floats where possible
    for k in list(stats.keys()):
        try:
            stats[k] = float(stats[k])
        except Exception:
            stats[k] = np.nan

    return stats

def _clean_json(x):
    # Convert numpy scalars → Python scalars
    if isinstance(x, (np.floating, np.integer)):
        x = x.item()

    # Replace NaN / Inf with None (JSON-safe)
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None

    # Recurse through dicts/lists
    if isinstance(x, dict):
        return {k: _clean_json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean_json(v) for v in x]

    return x


# ---------- Main prediction ----------
def predict_fight(red_name: str, blue_name: str):
    """
    Returns: dict with red win probability and features used.
    """
    model, feats = load_model()
    elo_raw = load_elo()
    elo = {norm_name(k): v for k, v in elo_raw.items()}

    df = load_raw_df()

    red_name = str(red_name).strip()
    blue_name = str(blue_name).strip()

    elo_red  = elo.get(norm_name(red_name), BASE_ELO)
    elo_blue = elo.get(norm_name(blue_name), BASE_ELO)

    features = {"elo_diff": elo_red - elo_blue}

    # Physical diffs (from latest known stats in dataset)
    red_stats  = latest_stats_for_fighter(df, red_name)
    blue_stats = latest_stats_for_fighter(df, blue_name)

    # ---- DEBUG: inspect why stats are missing ----
    
    print("RED input:", red_name)
    print("BLUE input:", blue_name)

    print("RED stats dict:", red_stats)
    print("BLUE stats dict:", blue_stats)

    if red_stats:
        print(
        "RED reach/height/weight:",
        red_stats.get("reach_cm"),
        red_stats.get("height_cm"),
        red_stats.get("weight_lbs"),
        )

    if blue_stats:
        print(
        "BLUE reach/height/weight:",
        blue_stats.get("reach_cm"),
        blue_stats.get("height_cm"),
        blue_stats.get("weight_lbs"),
        )
# --------------------------------------------

    # If we have stats, compute diffs
    if red_stats and blue_stats:
        features["reach_diff"]  = red_stats.get("reach_cm", np.nan)  - blue_stats.get("reach_cm", np.nan)
        features["height_diff"] = red_stats.get("height_cm", np.nan) - blue_stats.get("height_cm", np.nan)
        features["weight_diff"] = red_stats.get("weight_lbs", np.nan)- blue_stats.get("weight_lbs", np.nan)
    else:
        # missing stats; leave as nan
        features["reach_diff"]  = np.nan
        features["height_diff"] = np.nan
        features["weight_diff"] = np.nan

    # Build X in the exact feature order used in training
    X = pd.DataFrame([{f: features.get(f, np.nan) for f in feats}])

    # Handle missing values (simple fill; later we can do better)
    X = X.fillna(0)

    prob_red = float(model.predict_proba(X)[0, 1])

    out = {
        "red": red_name,
        "blue": blue_name,
        "prob_red_wins": float(prob_red),
        "features": features
    }

    return _clean_json(out)


# ---------- Quick local test ----------
if __name__ == "__main__":
    out = predict_fight("Jon Jones", "Islam Makhachev")
    print(out)

