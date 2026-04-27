"""
DataGolf free-tier data fetcher.
Pulls player rankings, SG stats, and current tournament field + predictions.
No API key required for most endpoints — key required for some advanced ones.
Set DATAGOLF_KEY in .env if you have one; gracefully degrades without it.
"""

import json
import urllib.request
import urllib.parse
import os
from pathlib import Path
from datetime import date

from dotenv import load_dotenv

load_dotenv()

KEY = os.getenv("DATAGOLF_KEY", "")
BASE = "https://feeds.datagolf.com"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
}


def _get(path: str, params: dict = None) -> dict | list:
    if KEY:
        params = params or {}
        params["key"] = KEY
    qs = ("?" + urllib.parse.urlencode(params)) if params else ""
    url = f"{BASE}{path}{qs}"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"DataGolf fetch failed [{path}]: {e}")
        return {}


def fetch_rankings() -> list[dict]:
    """
    OWGR-style DataGolf rankings with SG data.
    Free endpoint — no key needed.
    Returns list of player dicts with dg_id, player_name, country, owgr, sg_total, etc.
    """
    data = _get("/preds/get-dg-rankings", {"file_format": "json"})
    players = data.get("rankings", data) if isinstance(data, dict) else data
    print(f"DataGolf rankings: {len(players)} players")
    return players


def fetch_current_field() -> list[dict]:
    """
    Current tournament field with DataGolf predictions.
    Returns list of player dicts with dg_id, player_name, odds, projected finish, etc.
    """
    data = _get("/preds/pre-tournament", {"file_format": "json"})
    players = data.get("baseline", []) if isinstance(data, dict) else []
    print(f"DataGolf current field: {len(players)} players")
    return players


def fetch_skill_ratings() -> list[dict]:
    """
    SG skill ratings (approach, OTT, around green, putting) — free endpoint.
    Returns list of player dicts.
    """
    data = _get("/preds/skill-ratings", {"file_format": "json", "display": "value"})
    players = data.get("players", data) if isinstance(data, dict) else data
    print(f"DataGolf skill ratings: {len(players)} players")
    return players


def fetch_approach_skill() -> list[dict]:
    """
    Detailed approach shot stats — key SG predictor for outright wins.
    """
    data = _get("/preds/approach-skill", {"file_format": "json"})
    players = data.get("players", data) if isinstance(data, dict) else data
    print(f"DataGolf approach skill: {len(players)} players")
    return players


def save_raw(data, name: str) -> Path:
    out = RAW_DIR / f"{name}_{date.today()}.json"
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved → {out}")
    return out


if __name__ == "__main__":
    rankings = fetch_rankings()
    if rankings:
        save_raw(rankings, "dg_rankings")
        print("\nTop 10 by DataGolf ranking:")
        for p in rankings[:10]:
            name = p.get("player_name", p.get("name", "?"))
            rank = p.get("datagolf_rank", p.get("dg_rank", "?"))
            sg = p.get("sg_total", p.get("total", "?"))
            print(f"  #{rank:<4} {name:30s} SG: {sg}")

    field = fetch_current_field()
    if field:
        save_raw(field, "dg_field")
        print(f"\nCurrent field: {len(field)} players")

    skills = fetch_skill_ratings()
    if skills:
        save_raw(skills, "dg_skills")
