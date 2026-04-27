"""
Bovada golf scraper — outrights, matchups, top 10, top 20.
No API key required — uses Bovada's public JSON feed.
"""

import json
import urllib.request
from datetime import datetime, timezone

BOVADA_GOLF_URL = (
    "https://www.bovada.lv/services/sports/event/coupon/events/A/description/golf"
    "?lang=en&preMatchOnly=true"
)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
}

MARKET_MAP = {
    "Winner":                 "outright",
    "Tournament Winner":      "outright",
    "To Win Tournament":      "outright",
    "Top 10 Finish":          "top10",
    "Top 10":                 "top10",
    "Top 20 Finish":          "top20",
    "Top 20":                 "top20",
    "2-Ball":                 "matchup",
    "3-Ball":                 "matchup",
    "Head To Head":           "matchup",
    "Matchup":                "matchup",
}

# These Bovada markets are season-long futures, not weekly — skip them
SKIP_EVENTS = {"presidents cup", "ryder cup", "solheim cup", "masters 2027"}


def _american_to_prob(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def _fetch() -> list:
    req = urllib.request.Request(BOVADA_GOLF_URL, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"Bovada golf: fetch failed — {e}")
        return []


def fetch_bovada_golf() -> dict[str, list]:
    """
    Returns dict with keys: outrights, top10, top20, matchups.
    Each is a list of dicts.
    """
    data = _fetch()
    results = {"outrights": [], "top10s": [], "top20s": [], "matchups": []}

    for group in data:
        tournament = group.get("description", "")
        for event in group.get("events", []):
            if any(s in event.get("description", "").lower() for s in SKIP_EVENTS):
                continue
            start_ms = event.get("startTime", 0)
            event_date = (
                datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                if start_ms else ""
            )

            for dg in event.get("displayGroups", []):
                for market in dg.get("markets", []):
                    desc = market.get("description", "")
                    market_type = next(
                        (v for k, v in MARKET_MAP.items() if k.lower() in desc.lower()), None
                    )
                    if not market_type:
                        continue

                    outcomes = market.get("outcomes", [])

                    if market_type in ("outright", "top10", "top20"):
                        for outcome in outcomes:
                            price = outcome.get("price", {})
                            try:
                                odds = int(price["american"])
                            except (KeyError, ValueError):
                                continue
                            player = outcome.get("description", "").strip()
                            if not player:
                                continue
                            row = {
                                "tournament": tournament,
                                "event_date": event_date,
                                "player":     player,
                                "odds":       odds,
                                "implied_prob": round(_american_to_prob(odds), 4),
                                "market":     market_type,
                                "book":       "bovada",
                            }
                            key = {"outright": "outrights", "top10": "top10s", "top20": "top20s"}[market_type]
                            results[key].append(row)

                    elif market_type == "matchup":
                        if len(outcomes) < 2:
                            continue
                        p1 = outcomes[0]
                        p2 = outcomes[1]
                        try:
                            odds1 = int(p1.get("price", {}).get("american", 0))
                            odds2 = int(p2.get("price", {}).get("american", 0))
                        except (ValueError, TypeError):
                            continue
                        results["matchups"].append({
                            "tournament": tournament,
                            "event_date": event_date,
                            "player1":    p1.get("description", "").strip(),
                            "player2":    p2.get("description", "").strip(),
                            "odds1":      odds1,
                            "odds2":      odds2,
                            "implied1":   round(_american_to_prob(odds1), 4),
                            "implied2":   round(_american_to_prob(odds2), 4),
                            "market":     desc,
                            "book":       "bovada",
                        })

    for k, v in results.items():
        print(f"Bovada golf: {len(v):>4} {k}")
    return results


if __name__ == "__main__":
    data = fetch_bovada_golf()
    print("\n--- OUTRIGHTS (top 10) ---")
    for r in sorted(data["outrights"], key=lambda x: x["implied_prob"], reverse=True)[:10]:
        print(f"  {r['player']:30s} {r['odds']:+6d}  ({r['implied_prob']:.1%})")
    print("\n--- MATCHUPS (first 5) ---")
    for r in data["matchups"][:5]:
        print(f"  {r['player1']:25s} {r['odds1']:+6d}  vs  {r['player2']:25s} {r['odds2']:+6d}")
