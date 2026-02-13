import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "UFC_full_data_golden.csv"
OUT_STATS = ROOT / "models" / "fighter_stats.json"
OUT_FIGHTERS = ROOT / "models" / "fighters.json"

COL_RED = "f_1_name"
COL_BLUE = "f_2_name"

def norm(s: str) -> str:
    return str(s).strip()

def to_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None

def main():
    df = pd.read_csv(RAW)

    # Use event_date if present to choose latest row per fighter
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    else:
        df["event_date"] = pd.NaT

    fighter_stats = {}

    def consider(fighter_name, reach, height, weight, event_date):
        name = norm(fighter_name)
        if not name:
            return

        # Normalize event_date to Timestamp or None
        if pd.isna(event_date):
            event_date = None

        prev = fighter_stats.get(name)

        # First time we see the fighter
        if prev is None:
            fighter_stats[name] = {
                "reach_cm": to_float(reach),
                "height_cm": to_float(height),
                "weight_lbs": to_float(weight),
                "_event_date": event_date,   # store Timestamp (or None)
            }
            return

        prev_date = prev.get("_event_date")

        # If we have dates, keep the newest record
        if event_date is not None and (prev_date is None or event_date > prev_date):
            fighter_stats[name] = {
                "reach_cm": to_float(reach),
                "height_cm": to_float(height),
                "weight_lbs": to_float(weight),
                "_event_date": event_date,
            }

    # Scan all rows
    for _, row in df.iterrows():
        consider(
            row.get(COL_RED),
            row.get("f_1_fighter_reach_cm"),
            row.get("f_1_fighter_height_cm"),
            row.get("f_1_fighter_weight_lbs"),
            row.get("event_date"),
        )
        consider(
            row.get(COL_BLUE),
            row.get("f_2_fighter_reach_cm"),
            row.get("f_2_fighter_height_cm"),
            row.get("f_2_fighter_weight_lbs"),
            row.get("event_date"),
        )

    # Remove internal timestamp before saving JSON
    for v in fighter_stats.values():
        v.pop("_event_date", None)

    fighters = sorted(fighter_stats.keys())

    OUT_STATS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_STATS, "w") as f:
        json.dump(fighter_stats, f, indent=2)
    with open(OUT_FIGHTERS, "w") as f:
        json.dump(fighters, f, indent=2)

    print(f"Wrote {OUT_STATS} with {len(fighter_stats)} fighters")
    print(f"Wrote {OUT_FIGHTERS} with {len(fighters)} fighter names")

if __name__ == "__main__":
    main()
