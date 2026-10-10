"""
odds_client.py — Betting Odds Ingestion (the-odds-api.com)
===========================================================
Fetches EPL odds, maps bookmaker payloads into Module 1 MatchOdds objects,
then calibrates to fair probabilities + implied team lambdas.

Free tier: 500 requests/month. One daily pull = ~30/month. Set:
    ODDS_API_KEY  (GitHub secret)

Markets requested:
  h2h            -> 1X2
  totals         -> match total goals (fallback when team totals unavailable)
  team_totals    -> per-team Over/Under (best source for team lambda)

Python >= 3.10
"""

from __future__ import annotations

import math
import os
from typing import Optional

import requests

from module1_data_ingestion import CalibratedMatch, MatchOdds, OddsCalibrator

ODDS_BASE = "https://api.the-odds-api.com/v4"
SPORT = "soccer_epl"

# the-odds-api uses full club names; FPL uses short codes.
TEAM_MAP = {
    "Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU",
    "Brentford": "BRE", "Brighton and Hove Albion": "BHA", "Brighton": "BHA",
    "Chelsea": "CHE", "Coventry City": "COV", "Crystal Palace": "CRY",
    "Everton": "EVE", "Fulham": "FUL", "Hull City": "HUL",
    "Ipswich Town": "IPS", "Leeds United": "LEE", "Liverpool": "LIV",
    "Manchester City": "MCI", "Manchester United": "MUN",
    "Newcastle United": "NEW", "Nottingham Forest": "NFO",
    "Sunderland": "SUN", "Tottenham Hotspur": "TOT", "West Ham United": "WHU",
    "Wolverhampton Wanderers": "WOL",
}


def short_name(full: str) -> str:
    """Map a bookmaker club name to an FPL short code (falls back to input)."""
    return TEAM_MAP.get(full, full)


class OddsClient:
    def __init__(self, api_key: Optional[str] = None, region: str = "uk",
                 bookmaker: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY")
        if not self.api_key:
            raise ValueError("Set ODDS_API_KEY (the-odds-api.com)")
        self.region = region
        self.bookmaker = bookmaker      # e.g. "pinnacle" — sharpest if available

    def fetch_raw(self, markets: str = "h2h,totals") -> list[dict]:
        params = {
            "apiKey": self.api_key,
            "regions": self.region,
            "markets": markets,
            "oddsFormat": "decimal",
        }
        r = requests.get(f"{ODDS_BASE}/sports/{SPORT}/odds", params=params, timeout=25)
        r.raise_for_status()
        remaining = r.headers.get("x-requests-remaining")
        if remaining:
            print(f"[odds] API requests remaining this month: {remaining}")
        return r.json()

    # -- payload mapping -----------------------------------------------------

    def _pick_book(self, event: dict) -> Optional[dict]:
        books = event.get("bookmakers", [])
        if not books:
            return None
        if self.bookmaker:
            for b in books:
                if b["key"] == self.bookmaker:
                    return b
        # Many UK books quote h2h only; without a goals market the fixture
        # can't be calibrated and gets dropped. Prefer a book with both.
        for b in books:
            keys = {m["key"] for m in b.get("markets", [])}
            if {"h2h", "totals"} <= keys:
                return b
        return books[0]

    def to_match_odds(self, event: dict) -> Optional[MatchOdds]:
        book = self._pick_book(event)
        if not book:
            return None
        home_full, away_full = event["home_team"], event["away_team"]
        home, away = short_name(home_full), short_name(away_full)

        h2h = {m["key"]: m for m in book["markets"]}.get("h2h")
        if not h2h:
            return None
        prices = {o["name"]: o["price"] for o in h2h["outcomes"]}
        try:
            hw, aw = prices[home_full], prices[away_full]
            dr = prices["Draw"]
        except KeyError:
            return None

        mo = MatchOdds(home_team=home, away_team=away,
                       home_win=hw, draw=dr, away_win=aw)

        # Team totals (preferred) — outcomes carry a `description` = team name
        tt = {m["key"]: m for m in book["markets"]}.get("team_totals")
        if tt:
            for o in tt["outcomes"]:
                team = short_name(o.get("description", ""))
                line, side, price = o["point"], o["name"].lower(), o["price"]
                target = mo.home_totals if team == home else mo.away_totals
                cur = target.get(line, (None, None))
                target[line] = (price, cur[1]) if side == "over" else (cur[0], price)
            for d in (mo.home_totals, mo.away_totals):   # drop incomplete pairs
                for k in [k for k, v in d.items() if None in v]:
                    del d[k]

        # Derive team totals from the match total, split by 1X2 supremacy.
        if not mo.home_totals and not mo.away_totals:
            tot = {m["key"]: m for m in book["markets"]}.get("totals")
            if tot:
                mo.total_goals = self._total_from_market(tot)
                mo.home_share = self._supremacy_share(hw, dr, aw)
        return mo

    @staticmethod
    def _total_from_market(tot: dict) -> float:
        """Implied total match goals (lambda_home + lambda_away) from O/U."""
        from module1_data_ingestion import lambda_from_over_under, strip_vig
        lams = []
        by_line: dict[float, dict[str, float]] = {}
        for o in tot["outcomes"]:
            by_line.setdefault(o["point"], {})[o["name"].lower()] = o["price"]
        for line, sides in by_line.items():
            if "over" in sides and "under" in sides:
                fair = strip_vig({"over": sides["over"], "under": sides["under"]})
                lams.append(lambda_from_over_under(line, fair["over"]))
        return sum(lams) / len(lams) if lams else 2.7

    @staticmethod
    def _supremacy_share(hw: float, dr: float, aw: float) -> float:
        """Home team's share of total goals, inferred from win probabilities.

        A 50/50 match splits 50/50; a heavy favourite takes proportionally more.
        Uses sqrt of the win-odds ratio to damp extremes, then clamps."""
        from module1_data_ingestion import strip_vig
        f = strip_vig({"H": hw, "D": dr, "A": aw})
        ratio = (f["H"] + 0.5 * f["D"]) / max(f["A"] + 0.5 * f["D"], 1e-6)
        share = math.sqrt(ratio) / (1 + math.sqrt(ratio))
        return min(max(share, 0.25), 0.78)

    def fetch_calibrated(self) -> list[CalibratedMatch]:
        """Full path: API -> MatchOdds -> CalibratedMatch (fair probs + lambdas).

        NOTE: `team_totals` is not served on the /odds endpoint (it's an
        event-level market and generally paid), and requesting it returns 422.
        We request h2h + totals and derive team lambdas from the match total
        split by 1X2 supremacy — see `to_match_odds`.
        """
        cal = OddsCalibrator()
        out: list[CalibratedMatch] = []
        for ev in self.fetch_raw(markets="h2h,totals"):
            mo = self.to_match_odds(ev)
            if not mo:
                continue
            try:
                if getattr(mo, "total_goals", None):
                    out.append(self._calibrate_from_total(mo))
                else:
                    out.append(cal.calibrate(mo))
            except ValueError:
                continue          # no usable goal markets for this fixture
        return out

    @staticmethod
    def _calibrate_from_total(mo) -> CalibratedMatch:
        """Build CalibratedMatch when only the match total + 1X2 are available."""
        from module1_data_ingestion import strip_vig
        f = strip_vig({"H": mo.home_win, "D": mo.draw, "A": mo.away_win})
        lam_h = mo.total_goals * mo.home_share
        lam_a = mo.total_goals * (1 - mo.home_share)
        return CalibratedMatch(
            home_team=mo.home_team, away_team=mo.away_team,
            p_home=f["H"], p_draw=f["D"], p_away=f["A"],
            lambda_home=lam_h, lambda_away=lam_a,
            p_cs_home=math.exp(-lam_a), p_cs_away=math.exp(-lam_h))


if __name__ == "__main__":
    # Offline test with a mocked payload (no API key needed)
    fake_event = {
        "home_team": "Arsenal", "away_team": "Coventry City",
        "bookmakers": [{
            "key": "demo",
            "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": "Arsenal", "price": 1.30},
                    {"name": "Coventry City", "price": 11.0},
                    {"name": "Draw", "price": 5.75}]},
                {"key": "team_totals", "outcomes": [
                    {"name": "Over", "point": 1.5, "price": 1.36, "description": "Arsenal"},
                    {"name": "Under", "point": 1.5, "price": 3.10, "description": "Arsenal"},
                    {"name": "Over", "point": 0.5, "price": 2.30, "description": "Coventry City"},
                    {"name": "Under", "point": 0.5, "price": 1.62, "description": "Coventry City"}]},
            ]}]
    }
    client = OddsClient.__new__(OddsClient)      # bypass key check for the demo
    client.bookmaker = None
    mo = client.to_match_odds(fake_event)
    cm = OddsCalibrator().calibrate(mo)
    print(f"{cm.home_team} v {cm.away_team}")
    print(f"  fair 1X2: {cm.p_home:.3f} / {cm.p_draw:.3f} / {cm.p_away:.3f}")
    print(f"  team xG:  {cm.lambda_home:.2f} vs {cm.lambda_away:.2f}")
    print(f"  CS prob:  {cm.p_cs_home:.3f} / {cm.p_cs_away:.3f}")
