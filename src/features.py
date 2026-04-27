"""
Builds a unified player feature matrix from PGA Tour stats.
Accepts custom stat weights so the model can be tuned per course.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from pgatour_stats import fetch_all_stats, load_stats, save_stats

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# Default weights — approach is the strongest predictor of wins on most courses
DEFAULT_WEIGHTS = {
    "sg_approach":  0.35,
    "sg_total":     0.30,
    "sg_putting":   0.20,
    "sg_ott":       0.10,
    "sg_around":    0.05,
    "driving_dist": 0.00,
    "fairway_pct":  0.00,
    "gir_pct":      0.00,
    "scrambling":   0.00,
}

# Course type presets
PRESETS = {
    "default": {
        "label": "Balanced (Default)",
        "desc":  "General model — SG: Approach weighted highest, balanced across all categories.",
        "weights": DEFAULT_WEIGHTS,
    },
    "driving": {
        "label": "Driving Course",
        "desc":  "Long, open course where distance off the tee creates birdie opportunities. Think Augusta Par 5s, Riviera, Pebble Beach.",
        "weights": {"sg_ott": 0.40, "driving_dist": 0.20, "sg_approach": 0.20, "sg_total": 0.10, "sg_putting": 0.05, "sg_around": 0.05, "fairway_pct": 0.0, "gir_pct": 0.0, "scrambling": 0.0},
    },
    "approach": {
        "label": "Approach Course",
        "desc":  "Premium on iron play — small greens, tough pins, missing the green is costly. Think Harbour Town, Colonial, Muirfield Village.",
        "weights": {"sg_approach": 0.55, "sg_total": 0.20, "sg_putting": 0.10, "sg_ott": 0.05, "sg_around": 0.05, "gir_pct": 0.05, "driving_dist": 0.0, "fairway_pct": 0.0, "scrambling": 0.0},
    },
    "putting": {
        "label": "Putting Course",
        "desc":  "Tricky, undulating greens where putts decide everything. Think TPC Scottsdale, Augusta, Bermuda grass venues.",
        "weights": {"sg_putting": 0.50, "sg_approach": 0.20, "sg_total": 0.15, "sg_around": 0.10, "sg_ott": 0.05, "driving_dist": 0.0, "fairway_pct": 0.0, "gir_pct": 0.0, "scrambling": 0.0},
    },
    "tight_fairways": {
        "label": "Tight Fairways (FIR)",
        "desc":  "Narrow tree-lined course where hitting fairways is essential. Think Riviera, Torrey Pines, TPC Sawgrass.",
        "weights": {"fairway_pct": 0.30, "sg_approach": 0.30, "sg_total": 0.15, "sg_putting": 0.10, "sg_ott": 0.05, "sg_around": 0.05, "gir_pct": 0.05, "driving_dist": 0.0, "scrambling": 0.0},
    },
    "short_game": {
        "label": "Short Game Course",
        "desc":  "Firm, fast greens where missing in the right spots and scrambling are crucial. Think links golf, firm conditions.",
        "weights": {"sg_around": 0.30, "sg_putting": 0.30, "scrambling": 0.15, "sg_approach": 0.15, "sg_total": 0.05, "sg_ott": 0.05, "driving_dist": 0.0, "fairway_pct": 0.0, "gir_pct": 0.0},
    },
    "birdie_fest": {
        "label": "Birdie Fest",
        "desc":  "Low scoring, accessible course — tournament decided by who makes the most birdies. Think Kapalua, TPC Scottsdale.",
        "weights": {"sg_total": 0.35, "sg_approach": 0.25, "sg_putting": 0.20, "sg_ott": 0.10, "sg_around": 0.05, "driving_dist": 0.05, "fairway_pct": 0.0, "gir_pct": 0.0, "scrambling": 0.0},
    },
}


def _normalize_weights(w: dict) -> dict:
    """Ensure weights sum to 1.0."""
    total = sum(w.values())
    if total == 0:
        return DEFAULT_WEIGHTS
    return {k: v / total for k, v in w.items()}


def build_features(year: int = 2026, refresh: bool = False, weights: dict = None) -> pd.DataFrame:
    """
    Returns per-player feature DataFrame with composite model_score.

    weights: dict of stat_name -> weight (0-1). If None uses DEFAULT_WEIGHTS.
             Keys: sg_approach, sg_total, sg_putting, sg_ott, sg_around,
                   driving_dist, fairway_pct, gir_pct, scrambling
    """
    df = load_stats(year)
    if df.empty or refresh:
        print(f"Fetching fresh PGA Tour stats for {year}...")
        df = fetch_all_stats(year)
        if df.empty:
            return pd.DataFrame()
        save_stats(df, year)

    df = df.copy()
    df = df.dropna(subset=["sg_total"])

    w = _normalize_weights(weights or DEFAULT_WEIGHTS)

    # Normalize driving_dist to same scale as SG stats (~strokes)
    # Tour avg ~295yds, 1 SD ~12yds; convert to strokes gained equivalent
    if "driving_dist" in df.columns:
        dist_mean = df["driving_dist"].mean()
        dist_std  = max(float(df["driving_dist"].std()), 1.0)
        df["driving_dist_sg"] = (df["driving_dist"].fillna(dist_mean) - dist_mean) / dist_std * 0.5
    else:
        df["driving_dist_sg"] = 0.0

    # Normalize pct stats similarly
    for col in ["fairway_pct", "gir_pct", "scrambling"]:
        if col in df.columns:
            m = df[col].mean()
            s = max(float(df[col].std()), 0.01)
            df[f"{col}_sg"] = (df[col].fillna(m) - m) / s * 0.4
        else:
            df[f"{col}_sg"] = 0.0

    # Composite model score from weighted stats
    df["model_score"] = (
        df["sg_approach"].fillna(0)   * w.get("sg_approach",  0)
        + df["sg_total"].fillna(0)    * w.get("sg_total",     0)
        + df["sg_putting"].fillna(0)  * w.get("sg_putting",   0)
        + df["sg_ott"].fillna(0)      * w.get("sg_ott",       0)
        + df["sg_around"].fillna(0)   * w.get("sg_around",    0)
        + df["driving_dist_sg"]       * w.get("driving_dist", 0)
        + df["fairway_pct_sg"]        * w.get("fairway_pct",  0)
        + df["gir_pct_sg"]            * w.get("gir_pct",      0)
        + df["scrambling_sg"]         * w.get("scrambling",   0)
    )

    # Composite helper scores (used by fantasy model)
    df["scoring_power"] = (
        df["birdie_avg"].fillna(df["birdie_avg"].median()) * 2.0
        - df["bogey_avg"].fillna(df["bogey_avg"].median()) * 1.5
    )
    df["ball_striking"] = df["sg_ott"].fillna(0) + df["sg_approach"].fillna(0)
    df["short_game"]    = df["sg_around"].fillna(0) + df["sg_putting"].fillna(0)

    df["p_bogey_free"] = np.clip(
        0.05 + 0.25 * np.exp(-df["bogey_avg"].fillna(df["bogey_avg"].median())), 0.0, 0.50
    )
    df["ceiling_score"] = (
        df["birdie_avg"].fillna(0) / df["bogey_avg"].fillna(1).clip(lower=0.1)
    )

    df = df.sort_values("model_score", ascending=False).reset_index(drop=True)
    df["model_rank"] = df.index + 1

    return df


if __name__ == "__main__":
    df = build_features()
    print(f"Feature matrix: {len(df)} players\n")
    cols = ["player_name", "model_rank", "model_score", "sg_total", "sg_approach",
            "sg_putting", "birdie_avg", "bogey_avg"]
    print(df[cols].head(15).to_string(index=False))
