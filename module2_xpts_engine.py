"""
Module 2: Predictive Expected Points (xPts) Engine
===================================================
Ensemble team-xG (betting market 60% / Dixon-Coles 40%), individual share
allocation, 2025/26 BPS rule adjustments, DGW/BGW handling.

Author: FPL Quant Engine
Python >= 3.10
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

# ---------------------------------------------------------------------------
# 2.1 DIXON-COLES POISSON MODEL (internal 40% leg of the ensemble)
# ---------------------------------------------------------------------------

class DixonColesModel:
    """Time-decayed Dixon-Coles attack/defence model.

    Fit on rolling historical results: (home, away, home_goals, away_goals, days_ago).
    Produces per-fixture team lambdas.
    """

    def __init__(self, xi: float = 0.0018):
        self.xi = xi                       # exponential time-decay rate
        self.teams: list[str] = []
        self.attack: dict[str, float] = {}
        self.defence: dict[str, float] = {}
        self.home_adv: float = 0.25
        self.rho: float = -0.05            # low-score correlation

    @staticmethod
    def _tau(x: int, y: int, lx: float, ly: float, rho: float) -> float:
        """Dixon-Coles low-score correction factor."""
        if x == 0 and y == 0:
            return 1 - lx * ly * rho
        if x == 0 and y == 1:
            return 1 + lx * rho
        if x == 1 and y == 0:
            return 1 + ly * rho
        if x == 1 and y == 1:
            return 1 - rho
        return 1.0

    def fit(self, matches: list[tuple[str, str, int, int, int]]) -> None:
        """matches: (home, away, home_goals, away_goals, days_ago)"""
        self.teams = sorted({m[0] for m in matches} | {m[1] for m in matches})
        n = len(self.teams)
        idx = {t: i for i, t in enumerate(self.teams)}

        def unpack(params: np.ndarray):
            atk = params[:n]
            dfc = params[n:2 * n]
            home_adv, rho = params[2 * n], params[2 * n + 1]
            return atk, dfc, home_adv, rho

        def nll(params: np.ndarray) -> float:
            atk, dfc, home_adv, rho = unpack(params)
            ll = 0.0
            for h, a, hg, ag, days in matches:
                w = math.exp(-self.xi * days)
                lh = math.exp(atk[idx[h]] + dfc[idx[a]] + home_adv)
                la = math.exp(atk[idx[a]] + dfc[idx[h]])
                tau = max(self._tau(hg, ag, lh, la, rho), 1e-10)
                ll += w * (math.log(tau)
                           + poisson.logpmf(hg, lh)
                           + poisson.logpmf(ag, la))
            # identifiability: sum of attack ratings = 0
            return -ll + 100.0 * (np.sum(atk)) ** 2

        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25, -0.05]])
        res = minimize(nll, x0, method="L-BFGS-B", options={"maxiter": 400})
        atk, dfc, self.home_adv, self.rho = unpack(res.x)
        self.attack = {t: float(atk[idx[t]]) for t in self.teams}
        self.defence = {t: float(dfc[idx[t]]) for t in self.teams}

    def predict_lambdas(self, home: str, away: str) -> tuple[float, float]:
        lh = math.exp(self.attack[home] + self.defence[away] + self.home_adv)
        la = math.exp(self.attack[away] + self.defence[home])
        return lh, la


def ensemble_team_xg(market_lambda: Optional[float], dc_lambda: Optional[float],
                     w_market: float = 0.6) -> float:
    """Blend market-implied xG (60%) with Dixon-Coles (40%). Falls back gracefully."""
    if market_lambda is not None and dc_lambda is not None:
        return w_market * market_lambda + (1 - w_market) * dc_lambda
    return market_lambda if market_lambda is not None else (dc_lambda or 0.0)


# ---------------------------------------------------------------------------
# 2.2 PLAYER INPUTS & xMINS RECALIBRATION
# ---------------------------------------------------------------------------

STATUS_XMINS_CAP = {  # Module 3 mapping, applied here at prediction time
    "100": None,   # no cap -> baseline
    "75": 60.0,
    "50": 30.0,
    "25": 0.0,
    "0": 0.0,
}

APPEARANCE_LONG = 2   # >= 60 mins
APPEARANCE_SHORT = 1  # 1-59 mins

GOAL_POINTS = {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4}
CS_POINTS = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}
ASSIST_POINTS = 3
GOALS_CONCEDED_PENALTY = {"GKP": True, "DEF": True, "MID": False, "FWD": False}  # -1 per 2


@dataclass
class PlayerProfile:
    """Rolling 10-match per-90 baselines + availability."""
    player_id: int
    name: str
    team: str
    position: str                 # GKP / DEF / MID / FWD
    baseline_xmins: float         # typical minutes when in squad
    xg90: float                   # rolling 10-match xG/90
    xa90: float                   # rolling 10-match xA/90
    team_xg90_share_context: float  # team's avg xG/90 over same window (for share calc)
    saves90: float = 0.0          # GK only
    cbi90: float = 0.0            # clearances+blocks+interceptions per 90
    tackles90: float = 0.0
    p_card90: float = 0.06        # prob of a card per 90
    status_flag: str = "100"      # "100"/"75"/"50"/"25"/"0"


def effective_xmins(p: PlayerProfile) -> float:
    cap = STATUS_XMINS_CAP.get(p.status_flag)
    if cap is None:
        return p.baseline_xmins
    return min(p.baseline_xmins, cap)


# ---------------------------------------------------------------------------
# 2.3 BPS / BONUS MODEL (with 2025/26-style rule adjustments)
# ---------------------------------------------------------------------------

def expected_bps(p: PlayerProfile, xg_share: float, xa_share: float,
                 team_lambda: float, mins_ratio: float) -> float:
    """Simplified expected-BPS proxy incorporating the specified rule tweaks:
       - No -1 penalty for being tackled (removed).
       - +1 BPS per 3 CBI actions.
       - +2 BPS per save (+1 extra assumed on ~30% of saves being in-box/big-chance).
    """
    exp_goals = team_lambda * xg_share
    exp_assists = team_lambda * xa_share
    bps = 0.0
    bps += exp_goals * 24                      # goal scored (position-avg proxy)
    bps += exp_assists * 9
    bps += (p.cbi90 * mins_ratio) / 3.0 * 1    # CBI: +1 per 3
    bps += (p.tackles90 * mins_ratio) * 2      # won tackles (no tackled penalty)
    bps += (p.saves90 * mins_ratio) * (2 + 0.3 * 1)  # saves: +2, ~30% get +1 extra
    bps += 4 * mins_ratio                      # baseline participation BPS
    return bps


def expected_bonus_from_bps(xbps: float, scale: float = 30.0) -> float:
    """Map expected BPS to expected bonus points (0-3) with a soft logistic curve."""
    return 3.0 / (1.0 + math.exp(-(xbps - scale) / 6.0))


# ---------------------------------------------------------------------------
# 2.4 CORE xPTS CALCULATION
# ---------------------------------------------------------------------------

@dataclass
class FixtureContext:
    """One fixture from the player's team's perspective."""
    team_lambda: float        # ensemble xG for the player's team
    opp_lambda: float         # ensemble xG for the opponent
    rotation_decay: float = 1.0   # DGW fixture-2 minutes decay, e.g. 0.85


def xpts_single_fixture(p: PlayerProfile, fx: FixtureContext,
                        xmins_est: "object | None" = None) -> float:
    """If `xmins_est` (a Module 3 XMinsEstimate) is supplied, its p_start/p60/
    xmins are used directly — this is the preferred path. Otherwise falls back
    to the internal flag-cap heuristics."""
    if xmins_est is not None:
        xmins = xmins_est.xmins * fx.rotation_decay
        if xmins <= 0:
            return 0.0
        mins_ratio = xmins / 90.0
        p_start = xmins_est.p_any        # appearance prob = P(any minutes)
        p_60 = xmins_est.p60 * fx.rotation_decay
    else:
        xmins = effective_xmins(p) * fx.rotation_decay
        if xmins <= 0:
            return 0.0
        mins_ratio = xmins / 90.0
        p_start = min(1.0, xmins / max(p.baseline_xmins, 1.0)) if p.baseline_xmins else 0.0
        p_60 = max(0.0, min(1.0, (xmins - 45.0) / 30.0))  # smooth proxy for P(>=60 mins)

    # --- appearance ---
    pts = p_60 * APPEARANCE_LONG + (1 - p_60) * APPEARANCE_SHORT

    # --- attacking returns: allocate team lambda by individual per-90 share ---
    team_ctx = max(p.team_xg90_share_context, 0.1)
    xg_share = (p.xg90 / team_ctx) * mins_ratio
    xa_share = (p.xa90 / team_ctx) * mins_ratio
    exp_goals = fx.team_lambda * xg_share
    exp_assists = fx.team_lambda * xa_share
    pts += exp_goals * GOAL_POINTS[p.position]
    pts += exp_assists * ASSIST_POINTS

    # --- defensive returns ---
    p_cs = math.exp(-fx.opp_lambda)
    pts += p_cs * CS_POINTS[p.position] * p_60   # CS requires >= 60 mins
    if GOALS_CONCEDED_PENALTY[p.position]:
        pts -= (fx.opp_lambda * mins_ratio) / 2.0  # -1 per 2 conceded (expected)
    if p.position == "GKP":
        pts += (p.saves90 * mins_ratio) / 3.0      # +1 per 3 saves

    # --- bonus ---
    xbps = expected_bps(p, xg_share, xa_share, fx.team_lambda, mins_ratio)
    pts += expected_bonus_from_bps(xbps)

    # --- cards ---
    pts -= (p.p_card90 * mins_ratio) * 1.0         # yellow = -1 (reds rare, folded in)

    return pts * p_start  # scale by P(involved in the match)


def xpts_gameweek(p: PlayerProfile, fixtures: list[FixtureContext],
                  xmins_est: "object | None" = None) -> float:
    """DGW: sum fixtures independently (fixture 2 carries rotation decay).
       BGW: empty list -> 0.0 (hardcoded)."""
    if not fixtures:
        return 0.0
    return sum(xpts_single_fixture(p, fx, xmins_est) for fx in fixtures)


# ---------------------------------------------------------------------------
# Example execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- fit Dixon-Coles on a toy rolling sample -----------------------------
    hist = [
        ("ARS", "CHE", 3, 1, 7), ("LIV", "ARS", 2, 2, 14), ("CHE", "LIV", 1, 2, 21),
        ("ARS", "LIV", 2, 0, 28), ("CHE", "ARS", 0, 2, 35), ("LIV", "CHE", 3, 1, 42),
        ("WOL", "ARS", 0, 3, 10), ("ARS", "WOL", 4, 0, 45), ("WOL", "LIV", 1, 2, 30),
        ("CHE", "WOL", 2, 0, 17), ("WOL", "CHE", 1, 1, 50), ("LIV", "WOL", 2, 0, 55),
    ]
    dc = DixonColesModel()
    dc.fit(hist)
    dc_home, dc_away = dc.predict_lambdas("ARS", "WOL")
    print(f"Dixon-Coles ARS v WOL:  lambda_H={dc_home:.2f}  lambda_A={dc_away:.2f}")

    # --- ensemble with market lambdas from Module 1 --------------------------
    market_h, market_a = 2.35, 0.75          # from OddsCalibrator
    lam_h = ensemble_team_xg(market_h, dc_home)
    lam_a = ensemble_team_xg(market_a, dc_away)
    print(f"Ensemble team xG:       ARS={lam_h:.2f}  WOL={lam_a:.2f}\n")

    # --- player xPts ----------------------------------------------------------
    saka = PlayerProfile(
        player_id=1, name="Saka", team="ARS", position="MID",
        baseline_xmins=84, xg90=0.45, xa90=0.35,
        team_xg90_share_context=1.9, tackles90=1.2, cbi90=0.8,
    )
    gabriel = PlayerProfile(
        player_id=2, name="Gabriel", team="ARS", position="DEF",
        baseline_xmins=90, xg90=0.12, xa90=0.03,
        team_xg90_share_context=1.9, cbi90=6.5, tackles90=1.0,
    )
    raya = PlayerProfile(
        player_id=3, name="Raya", team="ARS", position="GKP",
        baseline_xmins=90, xg90=0.0, xa90=0.0,
        team_xg90_share_context=1.9, saves90=2.6,
    )
    flagged = PlayerProfile(
        player_id=4, name="Havertz (75%)", team="ARS", position="FWD",
        baseline_xmins=85, xg90=0.55, xa90=0.20,
        team_xg90_share_context=1.9, status_flag="75",
    )

    single = [FixtureContext(team_lambda=lam_h, opp_lambda=lam_a)]
    dgw = [FixtureContext(lam_h, lam_a),
           FixtureContext(1.6, 1.2, rotation_decay=0.85)]

    print("Single GW xPts:")
    for pl in (saka, gabriel, raya, flagged):
        print(f"  {pl.name:<15} {xpts_gameweek(pl, single):5.2f}")

    print("\nDouble GW xPts (Saka):", f"{xpts_gameweek(saka, dgw):.2f}")
    print("Blank GW xPts (Saka): ", f"{xpts_gameweek(saka, []):.2f}")
