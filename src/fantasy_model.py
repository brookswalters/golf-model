"""
PGA Tour Fantasy Golf model.
Scoring: Eagle=5, Birdie=2, Par=1, Bogey=-1, Double=-3
Bonuses: Low round=10, 2nd low=5, Bogey-free=3, FedExCup 1/10th
Captain earns double on everything.

Optimizes: starters (4), bench (2), captain pick, respecting 3-start-per-segment limit.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from features import build_features

DATA_DIR = Path(__file__).parent.parent / "data"
USAGE_FILE = DATA_DIR / "processed" / "fantasy_usage.json"

# Tournament FedExCup point values (from the contest rules)
FEDEX_PTS = {
    "standard":  500,
    "elevated":  700,
    "major":     750,
    "players":   750,
    "playoff":   750,
}

# PGA Tour scoring: average holes per round breakdown
AVG_HOLES = 18
# Tour averages per round (approximate baselines)
BASE_BIRDIES  = 3.7   # tour avg birdies/round
BASE_BOGEYS   = 0.9   # tour avg bogeys/round
BASE_EAGLES   = 0.05  # tour avg eagles/round
BASE_DOUBLES  = 0.2   # tour avg doubles/round
BASE_PARS     = AVG_HOLES - BASE_BIRDIES - BASE_BOGEYS - BASE_EAGLES - BASE_DOUBLES


def _expected_round_pts(birdies: float, bogeys: float, eagles: float = None, doubles: float = None) -> float:
    """Expected fantasy points from hole-by-hole scoring for one round."""
    if eagles is None:
        eagles = BASE_EAGLES
    if doubles is None:
        doubles = BASE_DOUBLES
    pars = AVG_HOLES - birdies - bogeys - eagles - doubles
    return (eagles * 5) + (birdies * 2) + (pars * 1) + (bogeys * -1) + (doubles * -3)


def _p_bogey_free(bogeys_per_round: float) -> float:
    """Probability of a bogey-free round using Poisson approximation."""
    return np.exp(-bogeys_per_round)


def _p_low_round(ceiling_score: float, rank_in_field: int = 1) -> float:
    """
    Rough probability of shooting the low round of the day.
    Better players with higher ceilings have better low-round probability.
    """
    base = 1 / 70  # ~70 players in field
    multiplier = max(0.5, 3.0 - (rank_in_field - 1) * 0.05)
    return min(base * multiplier, 0.20)


def expected_fantasy_pts(
    player: pd.Series,
    fedex_pts: int = 500,
    is_captain: bool = False,
) -> float:
    """
    Expected fantasy points for a player over a full tournament (4 rounds).
    """
    birdies = player.get("birdie_avg", BASE_BIRDIES)
    bogeys  = player.get("bogey_avg",  BASE_BOGEYS)
    model_rank = int(player.get("model_rank", 50))

    if pd.isna(birdies): birdies = BASE_BIRDIES
    if pd.isna(bogeys):  bogeys  = BASE_BOGEYS

    # Per-round expected pts from scoring
    round_pts = _expected_round_pts(birdies, bogeys)

    # Bogey-free round bonus (3 pts) — expected contribution over 4 rounds
    p_bf = _p_bogey_free(bogeys)
    bogey_free_bonus = p_bf * 3 * 4  # ~4 rounds (assumes makes cut)

    # Low round bonus — top players have higher chance
    p_low = _p_low_round(player.get("ceiling_score", 1.0), model_rank)
    p_2nd_low = p_low * 1.5  # 2nd low is more likely than 1st
    low_round_bonus = (p_low * 10 + p_2nd_low * 5) * 4

    # Cut probability — only players who make cut earn round 3/4 points
    # Approximate from model_rank: top 65 make cut
    p_cut = max(0.1, 1.0 - (model_rank - 1) * 0.008)
    p_cut = min(p_cut, 0.97)

    # Expected rounds played: 2 guaranteed + 2 if cut made
    expected_rounds = 2 + (2 * p_cut)

    # Scale round pts to expected rounds
    scoring_pts = round_pts * expected_rounds
    bonus_pts   = (bogey_free_bonus + low_round_bonus) * (expected_rounds / 4)

    # FedExCup end-of-tournament bonus (1/10th of pts, based on finish position)
    # Expected value = weighted average across likely finish positions
    # Top 10 finishes earn most of the points; scale by p_top10
    p_top10 = max(0.01, 1.0 - (model_rank - 1) * 0.012)
    fedex_bonus = (fedex_pts / 10) * p_top10 * 0.4  # discount for not always top10

    total = scoring_pts + bonus_pts + fedex_bonus
    if is_captain:
        total *= 2

    return round(total, 2)


def load_usage() -> dict:
    if USAGE_FILE.exists():
        with open(USAGE_FILE) as f:
            return json.load(f)
    return {}


def save_usage(usage: dict) -> None:
    USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(USAGE_FILE, "w") as f:
        json.dump(usage, f, indent=2)
    print(f"Usage saved → {USAGE_FILE}")


def record_starts(player_names: list[str], segment: int, usage: dict = None) -> dict:
    """Record that these players were used as starters this week."""
    usage = usage or load_usage()
    seg_key = f"seg{segment}"
    usage.setdefault(seg_key, {})
    for name in player_names:
        usage[seg_key][name] = usage[seg_key].get(name, 0) + 1
    save_usage(usage)
    return usage


def get_remaining_starts(player_name: str, segment: int, usage: dict = None) -> int:
    """Returns how many starts a player has left this segment (max 3)."""
    usage = usage or load_usage()
    seg_key = f"seg{segment}"
    used = usage.get(seg_key, {}).get(player_name, 0)
    return max(0, 3 - used)


def build_lineup(
    features: pd.DataFrame,
    tournament_field: list[str] = None,
    fedex_pts: int = 500,
    segment: int = 2,
    n_starters: int = 4,
    n_bench: int = 2,
) -> dict:
    """
    Recommend optimal lineup: 4 starters, 2 bench, 1 captain.

    Returns dict with starters, bench, captain, and their projected points.
    """
    df = features.copy()

    # Filter to tournament field if provided
    if tournament_field:
        mask = df["player_name"].str.lower().isin([n.lower() for n in tournament_field])
        df = df[mask]

    if df.empty:
        print("No players available for lineup")
        return {}

    usage = load_usage()

    # Calculate expected fantasy pts for each player
    df["exp_pts"] = df.apply(
        lambda r: expected_fantasy_pts(r, fedex_pts=fedex_pts, is_captain=False), axis=1
    )
    df["remaining_starts"] = df["player_name"].apply(
        lambda n: get_remaining_starts(n, segment, usage)
    )

    # Only pick players with starts remaining
    eligible = df[df["remaining_starts"] > 0].sort_values("exp_pts", ascending=False)

    if len(eligible) < n_starters + n_bench:
        print("Warning: not enough eligible players — ignoring usage limits")
        eligible = df.sort_values("exp_pts", ascending=False)

    starters = eligible.head(n_starters)
    bench    = eligible.iloc[n_starters: n_starters + n_bench]

    # Captain = starter with highest ceiling (top projected pts, not necessarily #1 overall)
    captain_row = starters.sort_values("ceiling_score", ascending=False).iloc[0]
    captain_pts = expected_fantasy_pts(captain_row, fedex_pts=fedex_pts, is_captain=True)

    # Total projected pts (starters + bench contributes if a starter misses cut)
    starter_pts = starters["exp_pts"].sum() - captain_row["exp_pts"] + captain_pts
    bench_pts   = bench["exp_pts"].sum()

    return {
        "starters":    starters[["player_name", "model_rank", "exp_pts", "remaining_starts"]].to_dict("records"),
        "bench":       bench[["player_name",    "model_rank", "exp_pts", "remaining_starts"]].to_dict("records"),
        "captain":     captain_row["player_name"],
        "captain_pts": round(captain_pts, 2),
        "proj_total":  round(starter_pts + bench_pts * 0.15, 2),  # bench contributes ~15% in expectation
    }


if __name__ == "__main__":
    features = build_features()
    if features.empty:
        print("No features available")
    else:
        lineup = build_lineup(features, fedex_pts=700, segment=2)
        if lineup:
            print(f"\n=== RECOMMENDED LINEUP ===")
            print(f"Captain: {lineup['captain']}  (proj {lineup['captain_pts']} pts)\n")
            print("STARTERS:")
            for p in lineup["starters"]:
                cap = " ← CAPTAIN" if p["player_name"] == lineup["captain"] else ""
                print(f"  {p['player_name']:30s}  proj {p['exp_pts']:5.1f} pts  ({p['remaining_starts']} starts left){cap}")
            print("\nBENCH:")
            for p in lineup["bench"]:
                print(f"  {p['player_name']:30s}  proj {p['exp_pts']:5.1f} pts  ({p['remaining_starts']} starts left)")
            print(f"\nProjected total: {lineup['proj_total']} pts")
