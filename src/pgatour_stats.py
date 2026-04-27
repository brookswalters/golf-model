"""
PGA Tour stats scraper via the public GraphQL API (same one the website uses).
Pulls all key Strokes Gained categories + birdie rate + cut %.
No API key required.
"""

import json
import urllib.request
import pandas as pd
from pathlib import Path
from datetime import date

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PGATOUR_GQL = "https://orchestrator.pgatour.com/graphql"
GQL_KEY     = "da2-gsrx5bibzbb4njvhl7t37prf7y"  # public key used by pgatour.com

# Stat IDs — verified against pgatour.com/stats
STATS = {
    "sg_total":       "02675",
    "sg_ott":         "02567",   # off the tee
    "sg_approach":    "02568",   # approach the green
    "sg_around":      "02569",   # around the green
    "sg_putting":     "02564",   # putting
    "birdie_ratio":   "02415",   # birdie:bogey ratio + total birdies/bogeys
    "bogey_avg":      "02419",   # bogeys per round (avg)
    "scoring_avg":    "108",     # scoring average
    "driving_dist":   "101",     # driving distance
    "fairway_pct":    "102",     # fairway %
    "gir_pct":        "103",     # greens in regulation
    "scrambling":     "130",     # scrambling %
}

STAT_QUERY = """
query StatDetails($tourCode: TourCode!, $statId: String!, $year: Int) {
    statDetails(tourCode: $tourCode, statId: $statId, year: $year) {
        statTitle
        rows {
            ... on StatDetailsPlayer {
                playerId
                playerName
                rank
                stats { statValue statName }
            }
        }
    }
}
"""


def _gql(stat_id: str, year: int = 2026) -> list[dict]:
    payload = json.dumps({
        "operationName": "StatDetails",
        "variables": {"tourCode": "R", "statId": stat_id, "year": year},
        "query": STAT_QUERY,
    }).encode()
    req = urllib.request.Request(
        PGATOUR_GQL,
        data=payload,
        headers={
            "User-Agent":   "Mozilla/5.0",
            "Content-Type": "application/json",
            "x-api-key":    GQL_KEY,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        if data.get("errors"):
            print(f"PGA GQL error [{stat_id}]: {data['errors'][0]['message']}")
            return []
        return data["data"]["statDetails"]["rows"]
    except Exception as e:
        print(f"PGA Tour fetch failed [{stat_id}]: {e}")
        return []


def _parse_stat_value(stats: list[dict], name_hint: str = "Avg") -> float | None:
    """Extract the primary value from the stats array (usually 'Avg')."""
    for s in stats:
        if name_hint.lower() in s.get("statName", "").lower():
            try:
                return float(s["statValue"].replace(",", ""))
            except (ValueError, AttributeError):
                pass
    # Fall back to first numeric value
    for s in stats:
        try:
            return float(s["statValue"].replace(",", ""))
        except (ValueError, AttributeError):
            pass
    return None


def _parse_birdie_ratio_row(r: dict) -> dict:
    """Extract total_birdies and total_bogeys from stat 02415."""
    out = {"total_birdies": None, "total_bogeys": None}
    for s in r.get("stats", []):
        name = s.get("statName", "")
        try:
            val = float(s["statValue"].replace(",", ""))
        except (ValueError, AttributeError):
            continue
        if "Total Birdies" in name:
            out["total_birdies"] = val
        elif "Total Bogeys" in name:
            out["total_bogeys"] = val
    return out


def fetch_all_stats(year: int = 2026) -> pd.DataFrame:
    """
    Pull all key stats and return a single DataFrame keyed by player_id + player_name.
    """
    frames = {}

    for stat_name, stat_id in STATS.items():
        rows = _gql(stat_id, year)
        if not rows:
            print(f"  {stat_name}: no data")
            continue

        records = []
        for r in rows:
            if stat_name == "birdie_ratio":
                extras = _parse_birdie_ratio_row(r)
                val = _parse_stat_value(r.get("stats", []), name_hint="Birdie to Bogey")
                records.append({
                    "player_id":      r.get("playerId", ""),
                    "player_name":    r.get("playerName", ""),
                    "birdie_ratio":   val,
                    "total_birdies":  extras["total_birdies"],
                    "total_bogeys_season": extras["total_bogeys"],
                })
            else:
                val = _parse_stat_value(r.get("stats", []))
                records.append({
                    "player_id":         r.get("playerId", ""),
                    "player_name":       r.get("playerName", ""),
                    f"{stat_name}_rank": r.get("rank"),
                    stat_name:           val,
                })
        frames[stat_name] = pd.DataFrame(records)
        print(f"  {stat_name}: {len(records)} players")

    if not frames:
        return pd.DataFrame()

    # Merge all stat frames on player_id
    base = list(frames.values())[0][["player_id", "player_name"]].copy()
    for stat_name, df in frames.items():
        merge_cols = [c for c in df.columns if c not in ("player_name",)]
        base = base.merge(df[merge_cols], on="player_id", how="outer")

    # Fill player names
    name_cols = [c for c in base.columns if "player_name" in c]
    if name_cols:
        base["player_name"] = base[name_cols].bfill(axis=1).iloc[:, 0]
        base = base.drop(columns=[c for c in name_cols if c != "player_name"])

    # Derive birdies_per_round from total_birdies / measured_rounds (from SG data)
    # sg_putting stat includes measured_rounds — extract it separately
    sg_rows = _gql("02564", year)
    rounds_map = {}
    for r in sg_rows:
        for s in r.get("stats", []):
            if "Measured Rounds" in s.get("statName", ""):
                try:
                    rounds_map[r["playerId"]] = float(s["statValue"])
                except (ValueError, KeyError):
                    pass

    base["measured_rounds"] = base["player_id"].map(rounds_map)
    base["birdie_avg"] = (
        base["total_birdies"] / base["measured_rounds"].clip(lower=1)
    ).where(base["measured_rounds"].notna())

    # Clean up duplicate/suffix columns from merges
    base = base.loc[:, ~base.columns.duplicated()]

    print(f"\nMerged stats: {len(base)} players")
    return base


def save_stats(df: pd.DataFrame, year: int = 2026) -> Path:
    out = PROCESSED_DIR / f"pga_stats_{year}.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved → {out}")
    return out


def load_stats(year: int = 2026) -> pd.DataFrame:
    path = PROCESSED_DIR / f"pga_stats_{year}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


if __name__ == "__main__":
    print(f"Fetching PGA Tour stats for 2026...\n")
    df = fetch_all_stats(2026)
    if not df.empty:
        save_stats(df)
        print("\nTop 10 by SG Total:")
        top = df.dropna(subset=["sg_total"]).sort_values("sg_total", ascending=False).head(10)
        for _, r in top.iterrows():
            print(f"  {r['player_name']:30s}  SG:T {r['sg_total']:+.3f}  App {r.get('sg_approach', float('nan')):+.3f}  Putt {r.get('sg_putting', float('nan')):+.3f}")
