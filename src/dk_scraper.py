"""
DraftKings golf scraper — outrights, matchups, top 10, top 20.
No API key required — uses DK's public sportsbook JSON endpoints.
"""

import json
import urllib.request
import pandas as pd
from datetime import date
from pathlib import Path

HISTORY_DIR = Path(__file__).parent.parent / "data" / "odds_history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

DK_BASE       = "https://sportsbook.draftkings.com/sites/US-SB/api/v5"
GOLF_GROUP    = 2196   # PGA Tour event group
OUTRIGHT_CAT  = 1238   # Tournament Winner
TOP10_CAT     = 1239   # Top 10 Finish
TOP20_CAT     = 1240   # Top 20 Finish
MATCHUP_CAT   = 1242   # Matchups (2-ball / 3-ball)

CATEGORY_NAMES = {
    OUTRIGHT_CAT: "outright",
    TOP10_CAT:    "top10",
    TOP20_CAT:    "top20",
    MATCHUP_CAT:  "matchup",
}


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def _american_to_prob(odds) -> float:
    try:
        odds = int(odds)
    except (TypeError, ValueError):
        return 0.0
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def _scrape_category(category_id: int, market_type: str) -> list[dict]:
    url = f"{DK_BASE}/eventgroups/{GOLF_GROUP}/categories/{category_id}"
    try:
        data = _get(url)
    except Exception as e:
        print(f"DK golf [{market_type}]: fetch failed — {e}")
        return []

    rows = []
    eg = data.get("eventGroup", {})
    tournament = eg.get("name", "")

    for offer_cat in eg.get("offerCategories", []):
        for sub_desc in offer_cat.get("offerSubcategoryDescriptors", []):
            sub = sub_desc.get("offerSubcategory", {})
            for offer_group in sub.get("offers", []):
                for offer in offer_group:
                    outcomes = offer.get("outcomes", [])

                    if market_type in ("outright", "top10", "top20"):
                        for outcome in outcomes:
                            player = (outcome.get("participant") or outcome.get("label", "")).strip()
                            odds = outcome.get("oddsAmerican")
                            if not player or odds is None:
                                continue
                            rows.append({
                                "tournament":   tournament,
                                "player":       player,
                                "odds":         int(odds),
                                "implied_prob": round(_american_to_prob(odds), 4),
                                "market":       market_type,
                                "book":         "draftkings",
                                "scraped_date": str(date.today()),
                            })

                    elif market_type == "matchup":
                        if len(outcomes) < 2:
                            continue
                        p1, p2 = outcomes[0], outcomes[1]
                        name1 = (p1.get("participant") or p1.get("label", "")).strip()
                        name2 = (p2.get("participant") or p2.get("label", "")).strip()
                        odds1 = p1.get("oddsAmerican")
                        odds2 = p2.get("oddsAmerican")
                        if not name1 or not name2 or odds1 is None or odds2 is None:
                            continue
                        rows.append({
                            "tournament":   tournament,
                            "player1":      name1,
                            "player2":      name2,
                            "odds1":        int(odds1),
                            "odds2":        int(odds2),
                            "implied1":     round(_american_to_prob(odds1), 4),
                            "implied2":     round(_american_to_prob(odds2), 4),
                            "market":       offer.get("label", "matchup"),
                            "book":         "draftkings",
                            "scraped_date": str(date.today()),
                        })

    print(f"DK golf [{market_type}]: {len(rows)} rows")
    return rows


def fetch_dk_golf() -> dict[str, list]:
    """Returns dict with keys: outrights, top10s, top20s, matchups."""
    return {
        "outrights": _scrape_category(OUTRIGHT_CAT, "outright"),
        "top10s":    _scrape_category(TOP10_CAT,    "top10"),
        "top20s":    _scrape_category(TOP20_CAT,    "top20"),
        "matchups":  _scrape_category(MATCHUP_CAT,  "matchup"),
    }


def save_to_history(data: dict, scraped_date: str = None) -> None:
    d = scraped_date or str(date.today())
    for market, rows in data.items():
        if not rows:
            continue
        out = HISTORY_DIR / f"dk_{market}_{d}.parquet"
        pd.DataFrame(rows).to_parquet(out, index=False)
        print(f"Saved → {out}")


if __name__ == "__main__":
    data = fetch_dk_golf()
    print("\n--- OUTRIGHTS (top 10 by implied prob) ---")
    outrights = sorted(data["outrights"], key=lambda x: x["implied_prob"], reverse=True)
    for r in outrights[:10]:
        print(f"  {r['player']:30s} {r['odds']:+6d}  ({r['implied_prob']:.1%})")
    print("\n--- MATCHUPS (first 5) ---")
    for r in data["matchups"][:5]:
        print(f"  {r['player1']:25s} {r['odds1']:+6d}  vs  {r['player2']:25s} {r['odds2']:+6d}")
