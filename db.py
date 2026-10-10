"""
db.py — Snapshot & Persistence Layer (Neon PostgreSQL)
=======================================================
The FPL API keeps no history. This layer snapshots everything the engine
needs so Module 8 (backtesting) has ground truth from day one.

Tables:
  player_snapshots   daily bootstrap-static pulls (price, flags, ownership, xG/xA)
  odds_snapshots     calibrated match odds (fair probs + implied lambdas)
  xmins_estimates    Module 3 outputs per player per GW (audit trail)
  xpts_predictions   Module 2 outputs per player per GW (for MAE/Brier later)
  rival_picks        mini-league rivals' squads per GW (EO layer, Module 7)
  gw_results         actual points post-GW (backtest target)

Usage:
  export DATABASE_URL="postgresql+psycopg2://user:pass@ep-xxx.neon.tech/fpl?sslmode=require"
  db = EngineDB()                 # reads DATABASE_URL
  db = EngineDB("sqlite:///fpl.db")   # local dev / tests

Python >= 3.10
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer, MetaData,
                        String, Table, create_engine, insert, select)

metadata = MetaData()

player_snapshots = Table(
    "player_snapshots", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("captured_at", DateTime(timezone=True), nullable=False),
    Column("gw", Integer, nullable=False),
    Column("player_id", Integer, nullable=False),
    Column("web_name", String(64)),
    Column("team", String(8)),
    Column("position", String(4)),
    Column("now_cost", Float),
    Column("selected_by_percent", Float),
    Column("status", String(2)),
    Column("chance_next", Integer),
    Column("minutes", Integer),
    Column("xg", Float),
    Column("xa", Float),
    Column("form", Float),
)

odds_snapshots = Table(
    "odds_snapshots", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("captured_at", DateTime(timezone=True), nullable=False),
    Column("gw", Integer, nullable=False),
    Column("home_team", String(8)),
    Column("away_team", String(8)),
    Column("p_home", Float), Column("p_draw", Float), Column("p_away", Float),
    Column("lambda_home", Float), Column("lambda_away", Float),
    Column("p_cs_home", Float), Column("p_cs_away", Float),
)

xmins_estimates = Table(
    "xmins_estimates", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("captured_at", DateTime(timezone=True), nullable=False),
    Column("gw", Integer, nullable=False),
    Column("player_id", Integer, nullable=False),
    Column("p_start", Float), Column("p60", Float), Column("p_any", Float),
    Column("xmins", Float),
    Column("flags", String(512)),          # joined audit log from Module 3
)

xpts_predictions = Table(
    "xpts_predictions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("captured_at", DateTime(timezone=True), nullable=False),
    Column("gw", Integer, nullable=False),
    Column("player_id", Integer, nullable=False),
    Column("xpts", Float, nullable=False),
    Column("n_fixtures", Integer, default=1),
    Column("model_version", String(32), default="v1"),
)

rival_picks = Table(
    "rival_picks", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("captured_at", DateTime(timezone=True), nullable=False),
    Column("gw", Integer, nullable=False),
    Column("manager_id", Integer, nullable=False),
    Column("player_id", Integer, nullable=False),
    Column("is_starting", Boolean),
    Column("is_captain", Boolean),
    Column("is_vice", Boolean),
)

gw_results = Table(
    "gw_results", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("gw", Integer, nullable=False),
    Column("player_id", Integer, nullable=False),
    Column("total_points", Integer),
    Column("minutes", Integer),
    Column("goals", Integer), Column("assists", Integer),
    Column("clean_sheet", Boolean), Column("bonus", Integer),
)


recommendations = Table(
    "recommendations", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("captured_at", DateTime(timezone=True), nullable=False),
    Column("gw", Integer, nullable=False),
    Column("mode", String(16)),              # daily | deadline
    Column("transfers_in", String(512)),
    Column("transfers_out", String(512)),
    Column("hits", Integer),
    Column("captain", String(64)),
    Column("vice", String(64)),
    Column("bench_order", String(512)),
    Column("projected_pts", Float),
    Column("summary", String(2048)),         # human-readable brief
)


def normalize_db_url(url: str) -> str:
    """Point any Postgres URL at the installed psycopg2 driver.

    Neon's dashboard hands out several variants (postgres://, postgresql://,
    postgresql+psycopg://). SQLAlchemy rejects postgres:// outright and loads
    psycopg v3 for +psycopg, which isn't installed — so pin the scheme."""
    url = url.strip()
    for prefix in ("postgres://", "postgresql://", "postgresql+psycopg://"):
        if url.startswith(prefix):
            return "postgresql+psycopg2://" + url[len(prefix):]
    return url


class EngineDB:
    def __init__(self, url: Optional[str] = None):
        url = url or os.environ.get("DATABASE_URL")
        if not url:
            raise ValueError("Set DATABASE_URL (Neon connection string) or pass url=")
        self.engine = create_engine(normalize_db_url(url), pool_pre_ping=True)
        metadata.create_all(self.engine)

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    # ---- writers -----------------------------------------------------------

    def save_player_snapshot(self, gw: int, players: list[dict]) -> int:
        rows = [{**{k: p.get(k) for k in
                    ("player_id", "web_name", "team", "position", "now_cost",
                     "selected_by_percent", "status", "chance_next",
                     "minutes", "xg", "xa", "form")},
                 "captured_at": self._now(), "gw": gw} for p in players]
        with self.engine.begin() as cx:
            cx.execute(insert(player_snapshots), rows)
        return len(rows)

    def save_odds(self, gw: int, calibrated: list[Any]) -> int:
        """calibrated: list of Module 1 CalibratedMatch objects."""
        rows = [{"captured_at": self._now(), "gw": gw,
                 "home_team": c.home_team, "away_team": c.away_team,
                 "p_home": c.p_home, "p_draw": c.p_draw, "p_away": c.p_away,
                 "lambda_home": c.lambda_home, "lambda_away": c.lambda_away,
                 "p_cs_home": c.p_cs_home, "p_cs_away": c.p_cs_away}
                for c in calibrated]
        with self.engine.begin() as cx:
            cx.execute(insert(odds_snapshots), rows)
        return len(rows)

    def save_xmins(self, gw: int, estimates: dict[int, Any]) -> int:
        """estimates: {player_id: Module 3 XMinsEstimate}"""
        rows = [{"captured_at": self._now(), "gw": gw, "player_id": pid,
                 "p_start": e.p_start, "p60": e.p60, "p_any": e.p_any,
                 "xmins": e.xmins, "flags": " | ".join(e.flags)[:512]}
                for pid, e in estimates.items()]
        with self.engine.begin() as cx:
            cx.execute(insert(xmins_estimates), rows)
        return len(rows)

    def save_xpts(self, gw: int, xpts: dict[int, float],
                  n_fixtures: Optional[dict[int, int]] = None,
                  model_version: str = "v1") -> int:
        rows = [{"captured_at": self._now(), "gw": gw, "player_id": pid,
                 "xpts": round(v, 3),
                 "n_fixtures": (n_fixtures or {}).get(pid, 1),
                 "model_version": model_version}
                for pid, v in xpts.items()]
        with self.engine.begin() as cx:
            cx.execute(insert(xpts_predictions), rows)
        return len(rows)

    def save_rival_picks(self, gw: int, manager_id: int, picks: list[dict]) -> int:
        """picks: FPL API entry/{id}/event/{gw}/picks/ 'picks' array."""
        rows = [{"captured_at": self._now(), "gw": gw, "manager_id": manager_id,
                 "player_id": p["element"],
                 "is_starting": p["position"] <= 11,
                 "is_captain": p["is_captain"], "is_vice": p["is_vice_captain"]}
                for p in picks]
        with self.engine.begin() as cx:
            cx.execute(insert(rival_picks), rows)
        return len(rows)

    def save_gw_results(self, gw: int, results: list[dict]) -> int:
        rows = [{**r, "gw": gw} for r in results]
        with self.engine.begin() as cx:
            cx.execute(insert(gw_results), rows)
        return len(rows)

    def save_recommendation(self, gw: int, mode: str, rec: dict) -> None:
        cols = {c.name for c in recommendations.columns}
        row = {k: v for k, v in rec.items() if k in cols}   # ignore brief-only keys
        row.update({"captured_at": self._now(), "gw": gw, "mode": mode})
        with self.engine.begin() as cx:
            cx.execute(insert(recommendations), [row])

    def latest_recommendation(self, gw: Optional[int] = None) -> Optional[dict]:
        q = select(recommendations).order_by(recommendations.c.captured_at.desc())
        if gw is not None:
            q = q.where(recommendations.c.gw == gw)
        with self.engine.connect() as cx:
            r = cx.execute(q.limit(1)).first()
        return dict(r._mapping) if r else None

    # ---- readers (backtest & EO) --------------------------------------------

    def prediction_vs_actual(self, gw: int) -> list[dict]:
        """Join xPts predictions to actuals for Module 8 MAE computation."""
        with self.engine.connect() as cx:
            preds = {r.player_id: r.xpts for r in
                     cx.execute(select(xpts_predictions).where(xpts_predictions.c.gw == gw))}
            out = []
            for r in cx.execute(select(gw_results).where(gw_results.c.gw == gw)):
                if r.player_id in preds:
                    out.append({"player_id": r.player_id, "xpts": preds[r.player_id],
                                "actual": r.total_points,
                                "abs_error": abs(preds[r.player_id] - r.total_points)})
        return out

    def rival_eo(self, gw: int) -> dict[int, float]:
        """Effective ownership among tracked rivals: start% + captain%."""
        with self.engine.connect() as cx:
            rows = list(cx.execute(select(rival_picks).where(rival_picks.c.gw == gw)))
        managers = {r.manager_id for r in rows}
        if not managers:
            return {}
        n = len(managers)
        eo: dict[int, float] = {}
        for r in rows:
            eo.setdefault(r.player_id, 0.0)
            if r.is_starting:
                eo[r.player_id] += 1.0 / n
            if r.is_captain:
                eo[r.player_id] += 1.0 / n
        return {pid: round(v * 100, 1) for pid, v in eo.items()}


if __name__ == "__main__":
    # Local smoke test on SQLite (swap DATABASE_URL for Neon in production)
    db = EngineDB("sqlite:///./fpl_test.db")
    db.save_player_snapshot(1, [{"player_id": 1, "web_name": "Saka", "team": "ARS",
                                 "position": "MID", "now_cost": 10.2,
                                 "selected_by_percent": 45.3, "status": "a",
                                 "chance_next": None, "minutes": 900,
                                 "xg": 4.1, "xa": 3.2, "form": 6.8}])
    db.save_xpts(1, {1: 8.25, 2: 5.49})
    db.save_gw_results(1, [{"player_id": 1, "total_points": 9, "minutes": 90,
                            "goals": 1, "assists": 0, "clean_sheet": True, "bonus": 2}])
    db.save_rival_picks(1, 555001, [
        {"element": 1, "position": 3, "is_captain": True, "is_vice_captain": False},
        {"element": 2, "position": 5, "is_captain": False, "is_vice_captain": True},
    ])
    db.save_rival_picks(1, 555002, [
        {"element": 1, "position": 2, "is_captain": False, "is_vice_captain": False},
    ])
    print("Backtest join:", db.prediction_vs_actual(1))
    print("Rival EO (%):", db.rival_eo(1))
