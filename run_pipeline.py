"""
run_pipeline.py — Daily orchestrator (run by GitHub Actions cron)
==================================================================
Ties Modules 1-3, 6 and the Neon layer together. Odds ingestion is stubbed
until you plug in your odds source (e.g. the-odds-api.com free tier).

  python run_pipeline.py --gw 1 --mode daily      # snapshot + xPts
  python run_pipeline.py --gw 1 --mode deadline   # + emergency check + solve
"""

from __future__ import annotations

import argparse
import os

from module1_data_ingestion import FPLClient
from module3_minutes_engine import MinutesEngine, StartsModel
from db import EngineDB

RIVAL_MANAGER_IDS = [int(x) for x in
                     os.environ.get("RIVAL_IDS", "").split(",") if x.strip()]


def daily(gw: int) -> None:
    client = FPLClient()
    db = EngineDB()                              # DATABASE_URL -> Neon

    players = client.players_frame()
    for p in players:                            # align key name for db layer
        p["player_id"] = p.pop("id")
    n = db.save_player_snapshot(gw, players)
    print(f"snapshot: {n} players saved for GW{gw}")

    for mid in RIVAL_MANAGER_IDS:
        try:
            picks = client.entry_picks(mid, max(gw - 1, 1))["picks"]
            db.save_rival_picks(gw, mid, picks)
            print(f"rival {mid}: {len(picks)} picks saved")
        except Exception as e:                   # rivals may not exist pre-GW1
            print(f"rival {mid}: skipped ({e})")

    # TODO: odds ingestion -> OddsCalibrator -> db.save_odds(gw, calibrated)
    # TODO: MinutesEngine per player (needs match history store) -> db.save_xmins
    # TODO: xpts_gameweek per player -> db.save_xpts


def deadline(gw: int) -> None:
    daily(gw)
    # TODO: EmergencyMonitor.check() against the last saved xmins snapshot,
    #       then SquadOptimizer(forced_out=...) re-solve and notify
    print("deadline mode: emergency check + solve (wire once xMins store is live)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw", type=int, required=True)
    ap.add_argument("--mode", choices=["daily", "deadline"], default="daily")
    a = ap.parse_args()
    (daily if a.mode == "daily" else deadline)(a.gw)
