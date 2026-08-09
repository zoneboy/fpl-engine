"""
Module 1: Data Ingestion & Betting Market Calibration
======================================================
FPL API integration + bookmaker odds de-vigging + implied team xG extraction.

Author: FPL Quant Engine
Python >= 3.10
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests
from scipy.optimize import brentq
from scipy.stats import poisson

# ---------------------------------------------------------------------------
# 1.1 FPL API CLIENT
# ---------------------------------------------------------------------------

FPL_BASE = "https://fantasy.premierleague.com/api"


class FPLClient:
    """Thin, cached client for the official FPL API.

    Endpoints covered:
      - bootstrap-static/          (players, teams, events, chips, prices, flags)
      - fixtures/                  (full fixture list incl. DGW/BGW detection)
      - element-summary/{id}/      (per-player history & upcoming fixtures)
      - entry/{id}/event/{gw}/picks/ (any manager's squad -> mini-league EO)
    """

    def __init__(self, session: Optional[requests.Session] = None,
                 max_retries: int = 3, backoff: float = 1.5):
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "fpl-quant-engine/1.0"})
        self.max_retries = max_retries
        self.backoff = backoff
        self._cache: dict[str, Any] = {}

    def _get(self, path: str, use_cache: bool = True) -> dict | list:
        if use_cache and path in self._cache:
            return self._cache[path]
        url = f"{FPL_BASE}/{path.lstrip('/')}"
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = self.session.get(url, timeout=20)
                r.raise_for_status()
                data = r.json()
                self._cache[path] = data
                return data
            except (requests.RequestException, ValueError) as e:
                last_err = e
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status is not None and 400 <= status < 500 and status != 429:
                    break                     # 4xx won't heal on retry (except 429)
                time.sleep(self.backoff ** attempt)
        raise RuntimeError(f"FPL API failed after {self.max_retries} tries: {url}") from last_err

    # ---- public endpoints -------------------------------------------------

    def bootstrap(self) -> dict:
        """Master payload: elements (players), teams, events (gameweeks)."""
        return self._get("bootstrap-static/")

    def fixtures(self, gw: Optional[int] = None) -> list[dict]:
        path = f"fixtures/?event={gw}" if gw else "fixtures/"
        return self._get(path)

    def element_summary(self, player_id: int) -> dict:
        return self._get(f"element-summary/{player_id}/")

    def entry_picks(self, manager_id: int, gw: int) -> dict:
        return self._get(f"entry/{manager_id}/event/{gw}/picks/", use_cache=False)

    # ---- convenience views ------------------------------------------------

    def players_frame(self) -> list[dict]:
        """Normalized player rows with the fields downstream modules need."""
        bs = self.bootstrap()
        teams = {t["id"]: t["short_name"] for t in bs["teams"]}
        pos = {p["id"]: p["singular_name_short"] for p in bs["element_types"]}
        rows = []
        for e in bs["elements"]:
            rows.append({
                "id": e["id"],
                "web_name": e["web_name"],
                "team_id": e["team"],
                "team": teams[e["team"]],
                "position": pos[e["element_type"]],
                "now_cost": e["now_cost"] / 10.0,
                "selected_by_percent": float(e["selected_by_percent"]),
                "status": e["status"],                 # a/d/i/s/u
                "chance_next": e["chance_of_playing_next_round"],
                "minutes": e["minutes"],
                "xg": float(e.get("expected_goals", 0) or 0),
                "xa": float(e.get("expected_assists", 0) or 0),
                "form": float(e["form"]),
            })
        return rows

    def dgw_bgw_map(self) -> dict[int, dict[int, int]]:
        """{gw: {team_id: n_fixtures}} — n=0 flags a blank, n>=2 a double."""
        bs = self.bootstrap()
        team_ids = [t["id"] for t in bs["teams"]]
        gws = [ev["id"] for ev in bs["events"]]
        out: dict[int, dict[int, int]] = {gw: {t: 0 for t in team_ids} for gw in gws}
        for fx in self.fixtures():
            if fx["event"] is None:
                continue
            out[fx["event"]][fx["team_h"]] += 1
            out[fx["event"]][fx["team_a"]] += 1
        return out


# ---------------------------------------------------------------------------
# 1.2 BETTING MARKET CALIBRATION
# ---------------------------------------------------------------------------

@dataclass
class MatchOdds:
    """Raw decimal odds for one fixture."""
    home_team: str
    away_team: str
    # 1X2 market
    home_win: float
    draw: float
    away_win: float
    # Team total goals Over/Under lines: {line: (over_odds, under_odds)}
    home_totals: dict[float, tuple[float, float]] = field(default_factory=dict)
    away_totals: dict[float, tuple[float, float]] = field(default_factory=dict)
    # Clean sheet (yes, no) odds
    home_cs: Optional[tuple[float, float]] = None
    away_cs: Optional[tuple[float, float]] = None
    # Derived when only a match-total market is available (see odds_client)
    total_goals: Optional[float] = None
    home_share: float = 0.55


def strip_vig(odds: dict[str, float]) -> dict[str, float]:
    """Proportional (multiplicative) normalization of the overround.

    P_fair(E) = (1/O_E) / sum_k(1/O_k)
    """
    inv = {k: 1.0 / v for k, v in odds.items()}
    total = sum(inv.values())
    return {k: v / total for k, v in inv.items()}


def lambda_from_over_under(line: float, p_over_fair: float,
                           lo: float = 0.01, hi: float = 8.0) -> float:
    """Solve for Poisson lambda given a fair Over probability at `line`.

    P(X > k) = 1 - CDF(k; lambda), where k = floor(line) for X.5 lines.
    Solved with Brent's root-finding method.
    """
    k = math.floor(line)

    def f(lam: float) -> float:
        return (1.0 - poisson.cdf(k, lam)) - p_over_fair

    # Guard: ensure a sign change exists in the bracket
    if f(lo) > 0:
        return lo
    if f(hi) < 0:
        return hi
    return float(brentq(f, lo, hi, xtol=1e-6))


def lambda_from_clean_sheet(p_cs_fair: float) -> float:
    """Clean sheet prob for the *opponent* implies lambda: P(X=0)=e^-lam."""
    p = min(max(p_cs_fair, 1e-6), 1 - 1e-6)
    return -math.log(p)


@dataclass
class CalibratedMatch:
    home_team: str
    away_team: str
    p_home: float
    p_draw: float
    p_away: float
    lambda_home: float       # home team expected goals
    lambda_away: float
    p_cs_home: float         # P(home keeps clean sheet) = P(away scores 0)
    p_cs_away: float


class OddsCalibrator:
    """Convert raw bookmaker odds into fair probabilities and team xG."""

    def calibrate(self, m: MatchOdds) -> CalibratedMatch:
        fair_1x2 = strip_vig({"H": m.home_win, "D": m.draw, "A": m.away_win})

        lam_h = self._team_lambda(m.home_totals, m.away_cs)
        lam_a = self._team_lambda(m.away_totals, m.home_cs)

        return CalibratedMatch(
            home_team=m.home_team, away_team=m.away_team,
            p_home=fair_1x2["H"], p_draw=fair_1x2["D"], p_away=fair_1x2["A"],
            lambda_home=lam_h, lambda_away=lam_a,
            p_cs_home=math.exp(-lam_a),
            p_cs_away=math.exp(-lam_h),
        )

    def _team_lambda(self, totals: dict[float, tuple[float, float]],
                     opp_cs: Optional[tuple[float, float]]) -> float:
        """Average lambda implied by all O/U lines; blend with CS market if present."""
        lams: list[float] = []
        for line, (over, under) in totals.items():
            fair = strip_vig({"over": over, "under": under})
            lams.append(lambda_from_over_under(line, fair["over"]))
        if opp_cs:  # opponent CS market implies THIS team's lambda
            fair = strip_vig({"yes": opp_cs[0], "no": opp_cs[1]})
            lams.append(lambda_from_clean_sheet(fair["yes"]))
        if not lams:
            raise ValueError("No goal markets supplied for lambda extraction")
        return sum(lams) / len(lams)


# ---------------------------------------------------------------------------
# Example execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Live FPL pull (requires internet):
    #   client = FPLClient(); players = client.players_frame()

    # Odds calibration demo — Arsenal (H) vs Wolves (A)
    odds = MatchOdds(
        home_team="ARS", away_team="WOL",
        home_win=1.30, draw=5.75, away_win=11.00,
        home_totals={1.5: (1.36, 3.10), 2.5: (2.05, 1.78)},
        away_totals={0.5: (2.30, 1.62), 1.5: (5.50, 1.15)},
        home_cs=(2.10, 1.72),   # (yes, no)
        away_cs=(6.00, 1.12),
    )
    cal = OddsCalibrator().calibrate(odds)
    print(f"{cal.home_team} v {cal.away_team}")
    print(f"  Fair 1X2:  H={cal.p_home:.3f}  D={cal.p_draw:.3f}  A={cal.p_away:.3f}")
    print(f"  Team xG:   {cal.home_team}={cal.lambda_home:.2f}  {cal.away_team}={cal.lambda_away:.2f}")
    print(f"  CS prob:   {cal.home_team}={cal.p_cs_home:.3f}  {cal.away_team}={cal.p_cs_away:.3f}")
