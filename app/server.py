"""
Flask web app for Golf Model.
"""

import sys
import os
import json
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Ensure data dirs exist (Railway ephemeral filesystem)
for _d in ["data/raw", "data/processed", "data/odds_history"]:
    (Path(__file__).parent.parent / _d).mkdir(parents=True, exist_ok=True)

from flask import Flask, render_template, jsonify, request
from features import build_features
from simulator import run as sim_run
from betting_model import find_edges
from fantasy_model import build_lineup, load_usage, record_starts

app = Flask(__name__)

# In-memory cache so the page loads fast after first run
_cache = {
    "date":     None,
    "sim":      None,
    "edges":    None,
    "lineup":   None,
    "features": None,
}

CURRENT_TOURNAMENT = "Cadillac Championship"
CURRENT_FEDEX      = 700
CURRENT_SEGMENT    = 2


def _run_model(refresh: bool = False):
    today = str(date.today())
    if _cache["date"] == today and not refresh and _cache["sim"] is not None:
        return

    features = build_features(refresh=refresh)
    sim      = sim_run(refresh=False)
    edges    = find_edges(sim) if not sim.empty else {"value_bets": [], "matchup_bets": []}
    lineup   = build_lineup(features, fedex_pts=CURRENT_FEDEX, segment=CURRENT_SEGMENT)

    _cache["date"]     = today
    _cache["features"] = features
    _cache["sim"]      = sim
    _cache["edges"]    = edges
    _cache["lineup"]   = lineup


@app.route("/")
def index():
    _run_model()
    return render_template("index.html",
                           tournament=CURRENT_TOURNAMENT,
                           fedex=CURRENT_FEDEX,
                           segment=CURRENT_SEGMENT,
                           today=str(date.today()))


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    refresh = request.json.get("refresh_stats", False)
    try:
        _run_model(refresh=refresh)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/sim")
def api_sim():
    _run_model()
    if _cache["sim"] is None or _cache["sim"].empty:
        return jsonify([])
    df = _cache["sim"]
    return jsonify(df[["player_name", "p_win", "p_top5", "p_top10", "p_top20", "p_cut", "model_score"]]
                   .head(60).to_dict("records"))


@app.route("/api/bets")
def api_bets():
    _run_model()
    edges = _cache["edges"] or {"value_bets": [], "matchup_bets": []}
    return jsonify(edges)


@app.route("/api/fantasy")
def api_fantasy():
    _run_model()
    lineup = _cache["lineup"] or {}
    usage  = load_usage()
    return jsonify({"lineup": lineup, "usage": usage})


@app.route("/api/record", methods=["POST"])
def api_record():
    lineup = _cache["lineup"]
    if not lineup:
        return jsonify({"ok": False, "error": "No lineup loaded"}), 400
    names = [p["player_name"] for p in lineup.get("starters", [])]
    record_starts(names, CURRENT_SEGMENT)
    return jsonify({"ok": True, "recorded": names})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    debug = os.environ.get("FLASK_ENV") != "production"
    print(f"Golf Model — {CURRENT_TOURNAMENT} ({CURRENT_FEDEX} FedExCup pts)")
    print(f"Starting server at http://127.0.0.1:{port}")
    app.run(debug=debug, host="0.0.0.0", port=port)
