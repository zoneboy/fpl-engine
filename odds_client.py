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

        # Fallback: split the match total ~55/45 by 1X2 strength
        if not mo.home_totals and not mo.away_totals:
            tot = {m["key"]: m for m in book["markets"]}.get("totals")
            if tot:
                for o in tot["outcomes"]:
                    line, side, price = o["point"], o["name"].lower(), o["price"]
                    # attribute the same line to both sides at a shifted level
                    for target in (mo.home_totals, mo.away_totals):
                        cur = target.get(line / 2, (None, None))
                        target[line / 2] = (price, cur[1]) if side == "over" else (cur[0], price)
                for d in (mo.home_totals, mo.away_totals):
                    for k in [k for k, v in d.items() if None in v]:
                        del d[k]
        return mo

    def fetch_calibrated(self) -> list[CalibratedMatch]:
        """Full path: API -> MatchOdds -> CalibratedMatch (fair probs + lambdas)."""
        cal = OddsCalibrator()
        out: list[CalibratedMatch] = []
        for ev in self.fetch_raw(markets="h2h,totals,team_totals"):
            mo = self.to_match_odds(ev)
            if not mo:
                continue
            try:
                out.append(cal.calibrate(mo))
            except ValueError:
                continue          # no usable goal markets for this fixture
        return out


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
