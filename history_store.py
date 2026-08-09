"""
history_store.py — Per-Player Match History for the Minutes Model
==================================================================
MinutesEngine needs list[MatchRecord] per player. That data lives in
element-summary/{id}/ — ~700 calls, so we pull WEEKLY and cache in Neon,
then rebuild MatchRecord objects from the cache on every daily run.

  python history_store.py --refresh --gw 3     # weekly: hit the API, cache
  (daily runs just read the cache via load_history)

Python >= 3.10
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Column, DateTime, Integer, String, Table, delete, insert, select

from db import EngineDB, metadata
from module1_data_ingestion import FPLClient
from module3_minutes_engine import MatchRecord

match_history = Table(
    "match_history", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("captured_at", DateTime(timezone=True)),
    Column("player_id", Integer, nullable=False, index=True),
    Column("fixture_id", Integer),
    Column("gw", Integer),
    Column("kickoff", DateTime(timezone=True)),
    Column("minutes", Integer),
    Column("started", Integer),          # 1/0 — inferred, see below
    Column("competition", String(8), default="PL"),
)


def _infer_started(minutes: int, was_home: bool) -> bool:
    """FPL history has no explicit 'started' field. Minutes >= 60 is a very
    strong proxy for a start; 1-59 is ambiguous but usually a cameo."""
    return minutes >= 60


class HistoryStore:
    def __init__(self, db: Optional[EngineDB] = None,
                 client: Optional[FPLClient] = None):
        self.db = db or EngineDB()
        self.client = client or FPLClient()
        metadata.create_all(self.db.engine)

    # -- weekly refresh -------------------------------------------------------

    def refresh(self, player_ids: Optional[list[int]] = None,
                sleep: float = 0.15, min_minutes_season: int = 1) -> int:
        """Pull element-summary for each player and replace their cached rows."""
        if player_ids is None:
            players = self.client.players_frame()
            # skip players who have never featured — saves hundreds of calls
            player_ids = [p["id"] for p in players
                          if p["minutes"] >= min_minutes_season]
        now = datetime.now(timezone.utc)
        total = 0
        for n, pid in enumerate(player_ids, 1):
            try:
                hist = self.client.element_summary(pid).get("history", [])
            except Exception as e:
                print(f"  [skip] player {pid}: {e}")
                continue
            rows = []
            for h in hist:
                ko = h.get("kickoff_time")
                rows.append({
                    "captured_at": now, "player_id": pid,
                    "fixture_id": h.get("fixture"), "gw": h.get("round"),
                    "kickoff": datetime.fromisoformat(ko.replace("Z", "+00:00")) if ko else None,
                    "minutes": h.get("minutes", 0),
                    "started": int(_infer_started(h.get("minutes", 0), h.get("was_home", True))),
                    "competition": "PL",
                })
            with self.db.engine.begin() as cx:
                cx.execute(delete(match_history).where(match_history.c.player_id == pid))
                if rows:
                    cx.execute(insert(match_history), rows)
            total += len(rows)
            if n % 50 == 0:
                print(f"  refreshed {n}/{len(player_ids)} players")
            time.sleep(sleep)          # be polite to the FPL API
        print(f"[history] cached {total} match rows for {len(player_ids)} players")
        return total

    # -- daily read -----------------------------------------------------------

    def load_history(self, player_id: int,
                     as_of: Optional[datetime] = None) -> list[MatchRecord]:
        """Rebuild MatchRecord list (with days_ago / days_rest_before) from cache."""
        as_of = as_of or datetime.now(timezone.utc)
        with self.db.engine.connect() as cx:
            rows = list(cx.execute(
                select(match_history)
                .where(match_history.c.player_id == player_id)
                .order_by(match_history.c.kickoff)))
        out: list[MatchRecord] = []
        prev_ko = None
        for r in rows:
            if not r.kickoff:
                continue
            ko = r.kickoff if r.kickoff.tzinfo else r.kickoff.replace(tzinfo=timezone.utc)
            days_ago = max(0, (as_of - ko).days)
            rest = (ko - prev_ko).days if prev_ko else 7
            prev_ko = ko
            out.append(MatchRecord(
                days_ago=days_ago,
                in_squad=True,                       # appears in history => in squad
                started=bool(r.started),
                minutes=r.minutes or 0,
                days_rest_before=max(1, min(rest, 14)),
                competition=r.competition or "PL",
            ))
        return out

    def load_all(self, as_of: Optional[datetime] = None) -> dict[int, list[MatchRecord]]:
        with self.db.engine.connect() as cx:
            pids = [r[0] for r in cx.execute(
                select(match_history.c.player_id).distinct())]
        return {pid: self.load_history(pid, as_of) for pid in pids}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="hit the FPL API (weekly)")
    ap.add_argument("--limit", type=int, default=0, help="cap players (testing)")
    a = ap.parse_args()
    store = HistoryStore()
    if a.refresh:
        ids = None
        if a.limit:
            ids = [p["id"] for p in store.client.players_frame()][:a.limit]
        store.refresh(player_ids=ids)
    cache = store.load_all()
    print(f"cache holds {len(cache)} players; "
          f"sample sizes: {[len(v) for v in list(cache.values())[:5]]}")
