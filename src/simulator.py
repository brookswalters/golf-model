"""
Monte Carlo tournament simulator.
Converts player SG averages into finish probability distributions.
"""

import numpy as np
import pandas as pd
from features import build_features

# Historical round-to-round SG standard deviation on PGA Tour (~1.5 strokes/round)
# This is the key variance parameter — higher = more upsets, lower = chalk wins
ROUND_STD   = 1.5
CUT_RANK    = 65     # top 65 make the cut (standard PGA Tour)
N_SIM       = 20_000 # number of simulations


def simulate_tournament(
    field: pd.DataFrame,
    n_sim: int = N_SIM,
    cut_rank: int = CUT_RANK,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """
    Monte Carlo simulation of a 4-round stroke play tournament.

    field: DataFrame with columns [player_id, player_name, model_score]
           model_score is the per-round SG average vs field average (baseline = 0)

    Returns DataFrame with finish probability columns added.
    """
    rng = np.random.default_rng(rng_seed)
    n_players = len(field)

    # Skill baseline: model_score is SG vs average, so average player = 0
    # Negative = strokes vs par (we simulate strokes below average)
    skills = field["model_score"].to_numpy()

    # Containers — track finish positions across simulations
    wins     = np.zeros(n_players, dtype=np.int32)
    top5s    = np.zeros(n_players, dtype=np.int32)
    top10s   = np.zeros(n_players, dtype=np.int32)
    top20s   = np.zeros(n_players, dtype=np.int32)
    cuts_made = np.zeros(n_players, dtype=np.int32)

    for _ in range(n_sim):
        # Simulate 4 rounds: (n_players, 4) noise matrix
        noise = rng.normal(0, ROUND_STD, size=(n_players, 4))
        # Total score = -(skill * 4) + noise sum  (more skill = lower score = better)
        scores = -(skills * 4) + noise.sum(axis=1)

        # Cut after round 2
        r2_scores = -(skills * 2) + noise[:, :2].sum(axis=1)
        cut_line_rank = np.argsort(r2_scores)[cut_rank - 1]
        cut_line_score = r2_scores[cut_line_rank]
        made_cut = r2_scores <= cut_line_score

        # Final order — players who missed cut get a huge penalty
        final = scores.copy()
        final[~made_cut] = 9999

        ranks = final.argsort().argsort() + 1  # 1 = winner

        wins[ranks == 1]     += 1
        top5s[ranks <= 5]    += 1
        top10s[ranks <= 10]  += 1
        top20s[ranks <= 20]  += 1
        cuts_made[made_cut]  += 1

    result = field[["player_id", "player_name", "model_score"]].copy()
    result["p_win"]      = np.round(wins     / n_sim, 5)
    result["p_top5"]     = np.round(top5s    / n_sim, 5)
    result["p_top10"]    = np.round(top10s   / n_sim, 5)
    result["p_top20"]    = np.round(top20s   / n_sim, 5)
    result["p_cut"]      = np.round(cuts_made / n_sim, 5)
    result = result.sort_values("p_win", ascending=False).reset_index(drop=True)
    return result


def run(field_names: list[str] = None, year: int = 2026, refresh: bool = False) -> pd.DataFrame:
    """
    Run simulator on a subset of players (the tournament field) or full ranked list.
    field_names: list of player name strings; if None uses top 100 by model score.
    """
    features = build_features(year=year, refresh=refresh)
    if features.empty:
        return pd.DataFrame()

    if field_names:
        # Match names (case-insensitive partial match)
        mask = features["player_name"].str.lower().isin([n.lower() for n in field_names])
        field = features[mask].copy()
        not_found = [n for n in field_names if not any(features["player_name"].str.lower() == n.lower())]
        if not_found:
            print(f"Warning: {len(not_found)} players not found in stats: {not_found[:5]}")
    else:
        field = features.head(100).copy()

    if field.empty:
        print("No players in field — check field_names")
        return pd.DataFrame()

    # Ensure player_id column exists
    if "player_id" not in field.columns:
        field = field.copy()
        field["player_id"] = field.index.astype(str)

    print(f"Simulating {len(field)} players x {N_SIM:,} tournaments...")
    return simulate_tournament(field)


if __name__ == "__main__":
    results = run()
    if not results.empty:
        print("\nSimulated finish probabilities (top 15):")
        print(results.head(15)[["player_name", "p_win", "p_top5", "p_top10", "p_top20", "p_cut"]].to_string(index=False))
