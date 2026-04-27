"""
Weekly golf model runner.
Run every Tuesday/Wednesday before tournament lock.

Usage:
    python weekly.py
    python weekly.py --refresh          # re-pull fresh stats
    python weekly.py --segment 2        # specify current segment (default: auto-detect)
    python weekly.py --fedex 700        # FedExCup points for this tournament
    python weekly.py --record           # record this week's picks to usage tracker
"""

import argparse
import sys
from datetime import date
from pathlib import Path

from features import build_features
from simulator import run as sim_run
from betting_model import find_edges
from fantasy_model import build_lineup, record_starts

DIVIDER = "=" * 65


def _detect_segment() -> int:
    """Auto-detect current segment based on date."""
    today = date.today()
    if today <= date(2026, 3, 15):
        return 1
    if today <= date(2026, 6, 7):
        return 2
    return 3


def _print_header(tournament: str, fedex_pts: int, segment: int) -> None:
    print(f"\n{DIVIDER}")
    print(f"  GOLF MODEL — {date.today()}")
    print(f"  Tournament : {tournament}")
    print(f"  FedExCup   : {fedex_pts} pts  |  Segment {segment}")
    print(DIVIDER)


def _print_betting(edges: dict) -> None:
    value_bets   = edges.get("value_bets", [])
    matchup_bets = edges.get("matchup_bets", [])

    print(f"\n{'─'*65}")
    print(f"  BETTING — VALUE BETS ({len(value_bets)} found)")
    print(f"{'─'*65}")
    if not value_bets:
        print("  No value bets above threshold this week.")
    else:
        print(f"  {'Player':<28} {'Market':<10} {'Odds':>6}  {'Edge':>6}  {'Kelly':>6}  Book")
        print(f"  {'─'*58}")
        for b in value_bets[:20]:
            print(
                f"  {b['player']:<28} {b['market']:<10} {b['odds']:>+6d}  "
                f"{b['edge']:>5.1%}  {b['kelly_units']:>5.3f}u  {b['book']}"
            )

    print(f"\n{'─'*65}")
    print(f"  BETTING — MATCHUPS ({len(matchup_bets)} found)")
    print(f"{'─'*65}")
    if not matchup_bets:
        print("  No matchup edges this week (may not be posted yet).")
    else:
        print(f"  {'Player':<28} {'vs Opponent':<24} {'Odds':>6}  {'Edge':>6}  Book")
        print(f"  {'─'*58}")
        for b in matchup_bets[:15]:
            print(
                f"  {b['player']:<28} vs {b['opponent']:<22} {b['odds']:>+6d}  "
                f"{b['edge']:>5.1%}  {b['book']}"
            )


def _print_fantasy(lineup: dict) -> None:
    print(f"\n{'─'*65}")
    print(f"  FANTASY LINEUP")
    print(f"{'─'*65}")
    if not lineup:
        print("  Could not generate lineup.")
        return

    print(f"  Captain: {lineup['captain']}")
    print(f"\n  STARTERS:")
    for p in lineup["starters"]:
        cap = " ◄ CAPTAIN" if p["player_name"] == lineup["captain"] else ""
        starts_left = p["remaining_starts"]
        print(f"    {p['player_name']:<30} proj {p['exp_pts']:5.1f} pts  ({starts_left} starts left){cap}")

    print(f"\n  BENCH (auto-swap if starter misses cut):")
    for p in lineup["bench"]:
        print(f"    {p['player_name']:<30} proj {p['exp_pts']:5.1f} pts  ({p['remaining_starts']} starts left)")

    print(f"\n  Projected total: {lineup['proj_total']} pts")


def _print_sim_top(sim_results, n: int = 15) -> None:
    print(f"\n{'─'*65}")
    print(f"  MODEL PROBABILITIES (top {n})")
    print(f"{'─'*65}")
    print(f"  {'Player':<28} {'Win':>6}  {'Top5':>6}  {'Top10':>7}  {'Top20':>7}  {'Cut':>6}")
    print(f"  {'─'*58}")
    for _, r in sim_results.head(n).iterrows():
        print(
            f"  {r['player_name']:<28} "
            f"{r['p_win']:>5.1%}  {r['p_top5']:>5.1%}  {r['p_top10']:>6.1%}  "
            f"{r['p_top20']:>6.1%}  {r['p_cut']:>5.1%}"
        )


def run(
    tournament: str = "This Week's Tournament",
    fedex_pts: int = 500,
    segment: int = None,
    refresh: bool = False,
    record: bool = False,
) -> None:
    segment = segment or _detect_segment()
    _print_header(tournament, fedex_pts, segment)

    # 1. Features
    print("\nLoading player stats...")
    features = build_features(refresh=refresh)
    if features.empty:
        print("ERROR: No player stats available.")
        sys.exit(1)
    print(f"  {len(features)} players loaded")

    # 2. Simulate
    print("\nRunning Monte Carlo simulation...")
    sim_results = sim_run(refresh=False)
    if sim_results.empty:
        print("ERROR: Simulation failed.")
        sys.exit(1)

    _print_sim_top(sim_results)

    # 3. Betting edges
    print("\nFetching odds and calculating edges...")
    edges = find_edges(sim_results)
    _print_betting(edges)

    # 4. Fantasy lineup
    print("\nBuilding fantasy lineup...")
    lineup = build_lineup(features, fedex_pts=fedex_pts, segment=segment)
    _print_fantasy(lineup)

    # 5. Optionally record picks
    if record and lineup:
        starter_names = [p["player_name"] for p in lineup["starters"]]
        record_starts(starter_names, segment)
        print(f"\n  Starts recorded for segment {segment}: {starter_names}")

    print(f"\n{DIVIDER}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weekly golf model report")
    parser.add_argument("--tournament", default="This Week's Tournament", help="Tournament name")
    parser.add_argument("--fedex",      type=int, default=500, help="FedExCup points (500/700/750)")
    parser.add_argument("--segment",    type=int, default=None, help="Contest segment (1/2/3)")
    parser.add_argument("--refresh",    action="store_true", help="Re-pull fresh stats from PGA Tour")
    parser.add_argument("--record",     action="store_true", help="Record this week's picks to usage tracker")
    args = parser.parse_args()

    run(
        tournament=args.tournament,
        fedex_pts=args.fedex,
        segment=args.segment,
        refresh=args.refresh,
        record=args.record,
    )
