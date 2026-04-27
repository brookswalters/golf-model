"""
Betting edge calculator.
Compares model finish probabilities to implied odds from Bovada/DraftKings.
Outputs ranked bet list with edge % and Kelly unit size.
"""

import pandas as pd
import numpy as np
from bovada_scraper import fetch_bovada_golf
from dk_scraper import fetch_dk_golf

MIN_EDGE      = 0.03   # minimum edge to flag a bet (3%)
KELLY_FRAC    = 0.25   # fractional Kelly (conservative — full Kelly is very aggressive)
MAX_KELLY     = 0.05   # cap any single bet at 5 units


def _american_to_prob(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def _american_to_decimal(odds: int) -> float:
    if odds > 0:
        return (odds / 100) + 1
    return (100 / abs(odds)) + 1


def _kelly(p_model: float, decimal_odds: float) -> float:
    """Fractional Kelly criterion. Returns recommended bet fraction."""
    b = decimal_odds - 1
    q = 1 - p_model
    kelly = (b * p_model - q) / b
    return max(0.0, round(kelly * KELLY_FRAC, 4))


def _normalize_name(name: str) -> str:
    """Lowercase + strip for fuzzy matching between model and book names."""
    return name.lower().strip()


def _build_odds_table(outrights: list, top10s: list, top20s: list, matchups: list) -> pd.DataFrame:
    rows = []
    for r in outrights:
        rows.append({"player": r["player"], "market": "outright", "odds": r["odds"],
                     "implied": r["implied_prob"], "book": r["book"],
                     "tournament": r.get("tournament", "")})
    for r in top10s:
        rows.append({"player": r["player"], "market": "top10", "odds": r["odds"],
                     "implied": r["implied_prob"], "book": r["book"],
                     "tournament": r.get("tournament", "")})
    for r in top20s:
        rows.append({"player": r["player"], "market": "top20", "odds": r["odds"],
                     "implied": r["implied_prob"], "book": r["book"],
                     "tournament": r.get("tournament", "")})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _build_matchup_table(matchups: list) -> pd.DataFrame:
    rows = []
    for r in matchups:
        rows.append({
            "player1": r["player1"], "player2": r["player2"],
            "odds1": r["odds1"], "odds2": r["odds2"],
            "implied1": r["implied1"], "implied2": r["implied2"],
            "market_label": r.get("market", "matchup"),
            "book": r["book"],
            "tournament": r.get("tournament", ""),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def find_edges(sim_results: pd.DataFrame, books: list[str] = ("bovada", "draftkings")) -> dict:
    """
    Compare model probabilities to market odds.

    sim_results: output of simulator.run()
    books: which books to pull odds from

    Returns dict with keys:
        value_bets   — outright/top10/top20 with positive edge
        matchup_bets — matchup bets with positive edge
    """
    # Build sim lookup
    sim = sim_results.copy()
    sim["_name"] = sim["player_name"].apply(_normalize_name)

    # Pull odds from requested books
    all_outrights, all_top10s, all_top20s, all_matchups = [], [], [], []

    if "bovada" in books:
        bov = fetch_bovada_golf()
        all_outrights += bov.get("outrights", [])
        all_top10s    += bov.get("top10s", [])
        all_top20s    += bov.get("top20s", [])
        all_matchups  += bov.get("matchups", [])

    if "draftkings" in books:
        try:
            dk = fetch_dk_golf()
            all_outrights += dk.get("outrights", [])
            all_top10s    += dk.get("top10s", [])
            all_top20s    += dk.get("top20s", [])
            all_matchups  += dk.get("matchups", [])
        except Exception as e:
            print(f"DK fetch failed: {e}")

    odds_df = _build_odds_table(all_outrights, all_top10s, all_top20s, [])
    matchup_df = _build_matchup_table(all_matchups)

    value_bets = []

    if not odds_df.empty:
        odds_df["_name"] = odds_df["player"].apply(_normalize_name)

        # Map market → sim probability column
        market_col = {"outright": "p_win", "top10": "p_top10", "top20": "p_top20"}

        for _, row in odds_df.iterrows():
            col = market_col.get(row["market"])
            if not col:
                continue
            match = sim[sim["_name"] == row["_name"]]
            if match.empty:
                continue
            p_model = float(match.iloc[0][col])
            p_implied = float(row["implied"])
            edge = p_model - p_implied
            if edge < MIN_EDGE:
                continue
            decimal_odds = _american_to_decimal(int(row["odds"]))
            kelly = _kelly(p_model, decimal_odds)
            kelly = min(kelly, MAX_KELLY)
            value_bets.append({
                "player":     row["player"],
                "market":     row["market"],
                "book":       row["book"],
                "tournament": row["tournament"],
                "odds":       int(row["odds"]),
                "p_model":    round(p_model, 4),
                "p_implied":  round(p_implied, 4),
                "edge":       round(edge, 4),
                "kelly_units": kelly,
            })

    # Matchup bets
    matchup_bets = []
    if not matchup_df.empty:
        matchup_df["_name1"] = matchup_df["player1"].apply(_normalize_name)
        matchup_df["_name2"] = matchup_df["player2"].apply(_normalize_name)

        for _, row in matchup_df.iterrows():
            m1 = sim[sim["_name"] == row["_name1"]]
            m2 = sim[sim["_name"] == row["_name2"]]
            if m1.empty or m2.empty:
                continue
            # P(player1 beats player2) from win probabilities ratio
            p1_win = float(m1.iloc[0]["p_win"])
            p2_win = float(m2.iloc[0]["p_win"])
            total  = p1_win + p2_win
            if total == 0:
                continue
            p_model_1 = p1_win / total
            p_model_2 = p2_win / total

            for player, p_model, p_implied, odds in [
                (row["player1"], p_model_1, row["implied1"], row["odds1"]),
                (row["player2"], p_model_2, row["implied2"], row["odds2"]),
            ]:
                edge = p_model - float(p_implied)
                if edge < MIN_EDGE:
                    continue
                decimal_odds = _american_to_decimal(int(odds))
                kelly = min(_kelly(p_model, decimal_odds), MAX_KELLY)
                matchup_bets.append({
                    "player":     player,
                    "opponent":   row["player2"] if player == row["player1"] else row["player1"],
                    "market":     row["market_label"],
                    "book":       row["book"],
                    "tournament": row["tournament"],
                    "odds":       int(odds),
                    "p_model":    round(p_model, 4),
                    "p_implied":  round(float(p_implied), 4),
                    "edge":       round(edge, 4),
                    "kelly_units": kelly,
                })

    value_bets = sorted(value_bets, key=lambda x: x["edge"], reverse=True)
    matchup_bets = sorted(matchup_bets, key=lambda x: x["edge"], reverse=True)

    return {"value_bets": value_bets, "matchup_bets": matchup_bets}


if __name__ == "__main__":
    from simulator import run as sim_run
    results = sim_run()
    if results.empty:
        print("No sim results")
    else:
        edges = find_edges(results)
        print(f"\n=== VALUE BETS ({len(edges['value_bets'])}) ===")
        for b in edges["value_bets"][:15]:
            print(f"  {b['player']:28s} {b['market']:8s} {b['odds']:+6d}  edge {b['edge']:.1%}  kelly {b['kelly_units']:.3f}u  [{b['book']}]")
        print(f"\n=== MATCHUP BETS ({len(edges['matchup_bets'])}) ===")
        for b in edges["matchup_bets"][:10]:
            print(f"  {b['player']:28s} vs {b['opponent']:20s}  {b['odds']:+6d}  edge {b['edge']:.1%}  [{b['book']}]")
