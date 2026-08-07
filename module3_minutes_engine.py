"""
Module 3 (UPGRADED): Minutes Model, Team News & Emergency Recalibration
========================================================================
Replaces flag-only xMins caps with a proper probabilistic minutes model:

  xMins = P(start) * E[mins | start]  +  P(cameo) * E[mins | cameo]

Components:
  3.1  Starts-probability model  (weighted starts history, rest days,
       congestion, injury-return ramp, manager rotation tendency)
  3.2  Minutes-given-start model (sub-off profile) + cameo minutes
  3.3  FPL status flags & press-conference news as Bayesian overrides
  3.4  DGW fixture-2 rotation decay derived from congestion (not hardcoded)
  3.5  Emergency solver trigger (starter -> 0 xMins within 3h of deadline)

Outputs an XMinsEstimate consumable by Module 2 (xmins, p_start, p60).

Author: FPL Quant Engine
Python >= 3.10
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# 3.0 DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class MatchRecord:
    """One past match from the player's perspective (most recent first not required)."""
    days_ago: int
    in_squad: bool
    started: bool
    minutes: int
    days_rest_before: int          # days since the team's previous match
    competition: str = "PL"        # PL / UCL / FA / EFL — cups reveal rotation


@dataclass
class NewsOverride:
    """Press-conference / journalist update. Overrides model output when fresh."""
    timestamp: datetime
    kind: str                      # "ruled_out" | "doubt_major" | "doubt_minor" | "confirmed_start" | "confirmed_bench"
    confidence: float = 0.9        # source reliability 0-1
    note: str = ""


@dataclass
class XMinsEstimate:
    """Rich output for Module 2 — richer than a single xMins scalar."""
    p_start: float
    p_cameo: float                 # P(appears off the bench | doesn't start)
    e_mins_start: float
    e_mins_cameo: float
    xmins: float
    p60: float                     # P(plays >= 60) — drives CS & appearance pts
    p_any: float                   # P(any minutes) — drives appearance floor
    flags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 3.1 STARTS-PROBABILITY MODEL
# ---------------------------------------------------------------------------

class StartsModel:
    """P(start) from exponentially-weighted history + contextual adjustments.

    Not a black box: every adjustment is a named, inspectable log-odds shift,
    which makes post-GW audits (Module 8) diagnosable.
    """

    def __init__(self, half_life_days: float = 30.0,
                 manager_rotation_index: float = 0.0):
        # manager_rotation_index: 0 = settled XI (e.g. classic Arteta),
        # 1 = heavy rotator (e.g. cup-congested Pep). Scales rest-day sensitivity.
        self.half_life = half_life_days
        self.rotation_idx = max(0.0, min(1.0, manager_rotation_index))

    # -- base rate ----------------------------------------------------------
    def base_start_rate(self, history: list[MatchRecord]) -> float:
        """Exponentially-weighted PL start rate over matches the player was fit for."""
        num, den = 0.0, 0.0
        for m in history:
            if not m.in_squad:
                continue                      # injured/suspended games don't count against
            w = 0.5 ** (m.days_ago / self.half_life)
            if m.competition != "PL":
                w *= 0.35                     # cup selection is weak evidence for PL XI
            num += w * (1.0 if m.started else 0.0)
            den += w
        if den == 0:
            return 0.30                       # no evidence -> conservative prior
        return num / den

    # -- contextual log-odds adjustments -------------------------------------
    @staticmethod
    def _logit(p: float) -> float:
        p = min(max(p, 1e-4), 1 - 1e-4)
        return math.log(p / (1 - p))

    @staticmethod
    def _sigmoid(z: float) -> float:
        return 1.0 / (1.0 + math.exp(-z))

    def p_start(self, history: list[MatchRecord], days_rest: int,
                matches_next_7_days: int = 1,
                games_since_return: Optional[int] = None,
                explain: Optional[list[str]] = None) -> float:
        z = self._logit(self.base_start_rate(history))
        log = explain if explain is not None else []

        # Rest-day effect, scaled by how rotation-happy the manager is
        if days_rest <= 3:
            shift = -(0.4 + 1.2 * self.rotation_idx)
            z += shift
            log.append(f"short rest ({days_rest}d): {shift:+.2f} log-odds")
        elif days_rest >= 6:
            shift = +0.25
            z += shift
            log.append(f"full week's rest: {shift:+.2f}")

        # Congestion ahead: managers pre-rotate before pileups
        if matches_next_7_days >= 3:
            shift = -(0.3 + 0.8 * self.rotation_idx)
            z += shift
            log.append(f"congestion ({matches_next_7_days} in 7d): {shift:+.2f}")

        # Injury-return ramp: bench -> 60 -> 90 pattern
        if games_since_return is not None and games_since_return <= 2:
            shift = -1.2 + 0.5 * games_since_return
            z += shift
            log.append(f"return ramp (game {games_since_return + 1} back): {shift:+.2f}")

        return self._sigmoid(z)


# ---------------------------------------------------------------------------
# 3.2 MINUTES GIVEN START / CAMEO
# ---------------------------------------------------------------------------

def e_mins_given_start(history: list[MatchRecord]) -> float:
    """Weighted average minutes in PL starts (captures sub-off profile)."""
    num, den = 0.0, 0.0
    for m in history:
        if m.started and m.competition == "PL":
            w = 0.5 ** (m.days_ago / 30.0)
            num += w * m.minutes
            den += w
    return (num / den) if den else 72.0


def cameo_profile(history: list[MatchRecord]) -> tuple[float, float]:
    """(P(cameo | not starting), E[mins | cameo]) from bench appearances."""
    bench_games = [m for m in history if m.in_squad and not m.started and m.competition == "PL"]
    if not bench_games:
        return 0.4, 15.0
    cameos = [m for m in bench_games if m.minutes > 0]
    p = len(cameos) / len(bench_games)
    e = sum(m.minutes for m in cameos) / len(cameos) if cameos else 15.0
    return p, e


def p60_from_components(p_start: float, e_mins_start: float) -> float:
    """P(>=60 mins) ~ P(start) x P(not subbed before 60 | start)."""
    # smooth mapping: a 90-min player ~0.97, a 70-min player ~0.75, a 62-min player ~0.5
    p_not_early_sub = 1.0 / (1.0 + math.exp(-(e_mins_start - 62.0) / 6.0))
    return p_start * p_not_early_sub


# ---------------------------------------------------------------------------
# 3.3 FLAGS & NEWS OVERRIDE LAYER (Bayesian caps, applied LAST)
# ---------------------------------------------------------------------------

FLAG_CAPS = {          # FPL official flag -> (p_start multiplier, xmins hard cap)
    "100": (1.00, None),
    "75":  (0.75, 65.0),
    "50":  (0.45, 35.0),
    "25":  (0.15, 20.0),
    "0":   (0.00, 0.0),
}

NEWS_EFFECTS = {       # fresher & higher-confidence news beats flags
    "ruled_out":       (0.00, 0.0),
    "doubt_major":     (0.30, 30.0),
    "doubt_minor":     (0.80, None),
    "confirmed_start": (0.97, None),
    "confirmed_bench": (0.08, None),
}


def apply_overrides(p_start: float, xmins_cap: Optional[float],
                    flag: str, news: Optional[NewsOverride],
                    now: datetime, flags_log: list[str]) -> tuple[float, Optional[float]]:
    mult, cap = FLAG_CAPS.get(flag, (1.0, None))
    p_start *= mult
    if cap is not None:
        xmins_cap = cap if xmins_cap is None else min(xmins_cap, cap)
    if flag != "100":
        flags_log.append(f"FPL flag {flag}%")

    if news and (now - news.timestamp) < timedelta(hours=48):
        n_mult, n_cap = NEWS_EFFECTS[news.kind]
        # blend by source confidence
        p_start = news.confidence * (n_mult if news.kind.startswith("confirmed") or news.kind == "ruled_out"
                                     else p_start * n_mult) \
                  + (1 - news.confidence) * p_start
        if n_cap is not None:
            xmins_cap = n_cap if xmins_cap is None else min(xmins_cap, n_cap)
        flags_log.append(f"news[{news.kind}] conf={news.confidence:.0%}: {news.note}")
    return p_start, xmins_cap


# ---------------------------------------------------------------------------
# 3.4 FULL ESTIMATOR + DGW ROTATION DECAY
# ---------------------------------------------------------------------------

class MinutesEngine:
    def __init__(self, starts_model: Optional[StartsModel] = None):
        self.starts = starts_model or StartsModel()

    def estimate(self, history: list[MatchRecord], days_rest: int,
                 matches_next_7_days: int = 1,
                 games_since_return: Optional[int] = None,
                 flag: str = "100", news: Optional[NewsOverride] = None,
                 now: Optional[datetime] = None) -> XMinsEstimate:
        now = now or datetime.now(timezone.utc)
        flags_log: list[str] = []

        ps = self.starts.p_start(history, days_rest, matches_next_7_days,
                                 games_since_return, explain=flags_log)
        ems = e_mins_given_start(history)
        pc, emc = cameo_profile(history)

        ps, cap = apply_overrides(ps, None, flag, news, now, flags_log)
        if cap is not None:
            ems = min(ems, cap)
            emc = min(emc, cap)

        xmins = ps * ems + (1 - ps) * pc * emc
        p60 = p60_from_components(ps, ems)
        p_any = ps + (1 - ps) * pc

        return XMinsEstimate(p_start=round(ps, 3), p_cameo=round(pc, 3),
                             e_mins_start=round(ems, 1), e_mins_cameo=round(emc, 1),
                             xmins=round(xmins, 1), p60=round(p60, 3),
                             p_any=round(p_any, 3), flags=flags_log)

    @staticmethod
    def dgw_fixture2_decay(days_between_fixtures: int,
                           manager_rotation_index: float = 0.3) -> float:
        """Derived (not hardcoded) fixture-2 minutes decay for DGWs."""
        base = 1.0 / (1.0 + math.exp(-(days_between_fixtures - 2.5)))  # 2d apart ~0.38, 4d ~0.82
        return round(min(1.0, max(0.35, base * (1 - 0.25 * manager_rotation_index) + 0.15)), 3)


# ---------------------------------------------------------------------------
# 3.5 EMERGENCY SOLVER TRIGGER
# ---------------------------------------------------------------------------

@dataclass
class EmergencyEvent:
    player_id: int
    name: str
    old_xmins: float
    new_xmins: float
    hours_to_deadline: float
    action: str = "RE_SOLVE_WITH_FORCED_REPLACEMENT"


class EmergencyMonitor:
    """Detects a nailed starter collapsing to ~0 xMins inside the danger window
    and fires a callback that should re-run the Module 6 MILP with the player
    excluded (forced replacement constraint)."""

    STARTER_THRESHOLD = 70.0     # was projected a near-full game
    COLLAPSE_THRESHOLD = 10.0    # now projected essentially out
    WINDOW_HOURS = 3.0

    def __init__(self, deadline: datetime,
                 on_trigger: Callable[[EmergencyEvent], None]):
        self.deadline = deadline
        self.on_trigger = on_trigger
        self.snapshot: dict[int, float] = {}

    def record_snapshot(self, xmins_by_player: dict[int, float]) -> None:
        self.snapshot = dict(xmins_by_player)

    def check(self, latest: dict[int, tuple[str, float]],
              now: Optional[datetime] = None) -> list[EmergencyEvent]:
        """latest: {player_id: (name, new_xmins)}"""
        now = now or datetime.now(timezone.utc)
        hrs = (self.deadline - now).total_seconds() / 3600.0
        events: list[EmergencyEvent] = []
        if not (0 < hrs <= self.WINDOW_HOURS):
            return events
        for pid, (name, new_xm) in latest.items():
            old = self.snapshot.get(pid, 0.0)
            if old >= self.STARTER_THRESHOLD and new_xm <= self.COLLAPSE_THRESHOLD:
                ev = EmergencyEvent(pid, name, old, new_xm, round(hrs, 2))
                events.append(ev)
                self.on_trigger(ev)
        return events


# ---------------------------------------------------------------------------
# Example execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    engine = MinutesEngine(StartsModel(manager_rotation_index=0.2))

    # Nailed starter: started 9 of last 10 PL games, plays ~88 mins
    nailed = [MatchRecord(days_ago=7 * i, in_squad=True, started=(i != 4),
                          minutes=88 if i != 4 else 12, days_rest_before=7)
              for i in range(10)]

    # Rotation risk: 5 starts, 5 cameos, subbed around 64 mins when starting
    rotation = [MatchRecord(days_ago=7 * i, in_squad=True, started=(i % 2 == 0),
                            minutes=64 if i % 2 == 0 else 18, days_rest_before=7)
                for i in range(10)]

    print("== Nailed starter, full rest ==")
    est = engine.estimate(nailed, days_rest=7)
    print(f"  p_start={est.p_start}  xMins={est.xmins}  p60={est.p60}")

    print("\n== Same player, 3 days rest + congestion + 75% flag ==")
    est = engine.estimate(nailed, days_rest=3, matches_next_7_days=3, flag="75")
    print(f"  p_start={est.p_start}  xMins={est.xmins}  p60={est.p60}")
    for f in est.flags:
        print(f"   - {f}")

    print("\n== Rotation risk, game 2 back from injury ==")
    est = engine.estimate(rotation, days_rest=7, games_since_return=1)
    print(f"  p_start={est.p_start}  xMins={est.xmins}  p60={est.p60}  p_any={est.p_any}")

    print("\n== Press conf: confirmed bench 1h ago (beats clean flag) ==")
    news = NewsOverride(datetime.now(timezone.utc) - timedelta(hours=1),
                        "confirmed_bench", 0.95, "manager presser")
    est = engine.estimate(nailed, days_rest=7, news=news)
    print(f"  p_start={est.p_start}  xMins={est.xmins}")

    print("\n== DGW fixture-2 decay ==")
    for d in (2, 3, 4, 6):
        print(f"  {d} days between fixtures -> decay {MinutesEngine.dgw_fixture2_decay(d)}")

    print("\n== Emergency trigger demo ==")
    deadline = datetime.now(timezone.utc) + timedelta(hours=2)
    fired: list[EmergencyEvent] = []
    mon = EmergencyMonitor(deadline, on_trigger=lambda ev: fired.append(ev))
    mon.record_snapshot({101: 88.0, 102: 25.0})
    mon.check({101: ("Saka", 0.0), 102: ("SubKeeper", 0.0)})
    for ev in fired:
        print(f"  TRIGGERED: {ev.name} {ev.old_xmins}->{ev.new_xmins} xMins, "
              f"{ev.hours_to_deadline}h to deadline -> {ev.action}")
