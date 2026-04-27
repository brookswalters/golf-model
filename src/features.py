"""
Builds a unified player feature matrix from PGA Tour stats.
Used as input to both the betting simulator and fantasy model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from pgatour_stats import fetch_all_stats, load_stats, save_stats

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def _normalize(series: pd.Series) -> pd.Series:
    """Z-score normalize, handling NaN."""
    mean, std = series.mean(), series.std()
    if std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - mean) / std


def build_features(year: int = 2026, refresh: bool = False) -> pd.DataFrame:
    """
    Returns per-player feature DataFrame with composite scores.
    Pulls fresh stats if refresh=True or no cached data exists.
    """
    df = load_stats(year)
    if df.empty or refresh:
        print(f"Fetching fresh PGA Tour stats for {year}...")
        df = fetch_all_stats(year)
        if df.empty:
            return pd.DataFrame()
        save_stats(df, year)

    df = df.copy()

    # Drop players with no SG data
    df = df.dropna(subset=["sg_total"])

    # Composite scoring power: birdie upside minus bogey damage
    # Weights calibrated so each component is roughly equal scale
    df["scoring_power"] = (
        df["birdie_avg"].fillna(df["birdie_avg"].median()) * 2.0
        - df["bogey_avg"].fillna(df["bogey_avg"].median()) * 1.5
    )

    # Ball striking composite (SG: OTT + Approach)
    df["ball_striking"] = (
        df["sg_ott"].fillna(0) + df["sg_approach"].fillna(0)
    )

    # Short game composite (SG: Around + Putting)
    df["short_game"] = (
        df["sg_around"].fillna(0) + df["sg_putting"].fillna(0)
    )

    # Overall model score — approach is the strongest predictor of wins
    df["model_score"] = (
        df["sg_approach"].fillna(0) * 0.35
        + df["sg_total"].fillna(0)  * 0.30
        + df["sg_putting"].fillna(0) * 0.20
        + df["sg_ott"].fillna(0)    * 0.10
        + df["sg_around"].fillna(0) * 0.05
    )

    # Bogey-free round probability estimate
    # Based on bogey avg: lower bogey rate → higher P(bogey-free round)
    # Empirically ~15% of rounds are bogey-free at tour average (0.9 bogeys/round)
    df["p_bogey_free"] = np.clip(
        0.05 + 0.25 * np.exp(-df["bogey_avg"].fillna(df["bogey_avg"].median())),
        0.0, 0.50
    )

    # Low-round potential: players who shoot birdies in clusters
    # Proxy: high birdie avg relative to bogey avg
    df["ceiling_score"] = (
        df["birdie_avg"].fillna(0) / df["bogey_avg"].fillna(1).clip(lower=0.1)
    )

    # Normalize ranks for quick reference
    df = df.sort_values("model_score", ascending=False).reset_index(drop=True)
    df["model_rank"] = df.index + 1

    return df


if __name__ == "__main__":
    df = build_features(refresh=False)
    print(f"Feature matrix: {len(df)} players\n")
    cols = ["player_name", "model_rank", "model_score", "sg_total", "sg_approach",
            "sg_putting", "birdie_avg", "bogey_avg", "p_bogey_free", "ceiling_score"]
    print(df[cols].head(15).to_string(index=False))
