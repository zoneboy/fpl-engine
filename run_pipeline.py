"""
run_pipeline.py — Full Orchestrator
====================================
Chain: FPL snapshot -> odds -> xMins -> xPts -> MILP optimizer -> Neon -> Telegram

Modes:
  daily     snapshot + odds + xPts + solve + save + notify
  weekly    the above, preceded by a match-history refresh (~700 API calls)
  deadline  the above + emergency check against the last xMins snapshot

  python run_pipeline.py --gw 1 --mode daily

Env / GitHub secrets:
  DATABASE_URL       Neon connection string        (required)
  ODDS_API_KEY       the-odds-api.com key          (optional; xPts degrades to DC only)
  RIVAL_IDS          comma-separated manager ids   (optional)
  MY_ENTRY_ID        your FPL manager id           (optional; enables real squad + budget)
  TELEGRAM_TOKEN     bot token                     (optional)
  TELEGRAM_CHAT_ID   your chat id                  (optional)
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import notify
from db import EngineDB
from history_store import HistoryStore
from module1_data_ingestion import FPLClient
from module2_xpts_engine import (FixtureContext, PlayerProfile,
                                 ensemble_team_xg, xpts_gameweek)
from module3_minutes_engine import MinutesEngine, StartsModel, XMinsEstimate
from module6_optimizer import OptPlayer, SquadOptimizer

RIVAL_IDS = [int(x) for x in os.environ.get("RIVAL_IDS", "").split(",") if x.strip()]
MY_ENTRY_ID = os.environ.get("MY_ENTRY_ID")
HORIZON = 5

FLAG_FROM_CHANCE = {None: "100", 100: "100", 75: "75", 50: "50", 25: "25", 0: "0"}

# --- pre-season priors -------------------------------------------------------
# Before any 26/27 minutes exist, FPL price is the best public summary of a
# player's expected role and output (it encodes last season's returns). These
# priors stop the optimizer treating every player as identical.

PRICE_FLOOR = {"GKP": 4.0, "DEF": 4.0, "MID": 4.5, "FWD": 4.5}
# xG90 + xA90 at the top of each position's price range
PRIOR_ATT_TOP = {"GKP": 0.00, "DEF": 0.18, "MID": 0.70, "FWD": 0.85}
PRIOR_ATT_BASE = {"GKP": 0.00, "DEF": 0.04, "MID": 0.10, "FWD": 0.15}


def preseason_priors(p: dict) -> tuple[float, float, float]:
    """(xg90, xa90, p_start_prior) from price + ownership when no data exists."""
    pos = p["position"]
    floor = PRICE_FLOOR[pos]
    # normalised price position within the position's range (0 = cheap, 1 = premium)
    q = min(max((p["now_cost"] - floor) / 8.0, 0.0), 1.0)
    att = PRIOR_ATT_BASE[pos] + (PRIOR_ATT_TOP[pos] - PRIOR_ATT_BASE[pos]) * (q ** 0.75)
    # attackers skew to goals, midfielders split, defenders mostly assists
    goal_share = {"GKP": 0.0, "DEF": 0.35, "MID": 0.55, "FWD": 0.75}[pos]
    xg90, xa90 = att * goal_share, att * (1 - goal_share)
    # nailed-ness: price is the strongest signal, ownership a mild confirmation
    p_start = 0.25 + 0.60 * q + 0.15 * min(p["selected_by_percent"] / 30.0, 1.0)
    return xg90, xa90, min(p_start, 0.95)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def fetch_market_lambdas(gw_teams: set[str]) -> dict[str, float]:
    """{team_short: lambda} for the upcoming round. Empty dict if no API key."""
    if not os.environ.get("ODDS_API_KEY"):
        print("[odds] ODDS_API_KEY not set — skipping market leg")
        return {}
    try:
        from odds_client import OddsClient
        cal = OddsClient().fetch_calibrated()
    except Exception as e:
        print(f"[odds] fetch failed ({e}) — continuing without market leg")
        return {}
    out: dict[str, float] = {}
    for c in cal:
        out[c.home_team] = c.lambda_home
        out[c.away_team] = c.lambda_away
    print(f"[odds] lambdas for {len(out)} teams")
    return out


def build_fixture_contexts(client: FPLClient, gw: int,
                           market: dict[str, float]) -> dict[int, list[FixtureContext]]:
    """{team_id: [FixtureContext, ...]} — handles DGW (2+) and BGW (empty)."""
    bs = client.bootstrap()
    short = {t["id"]: t["short_name"] for t in bs["teams"]}
    strength = {t["id"]: (t["strength_attack_home"] + t["strength_attack_away"]) / 2.0
                for t in bs["teams"]}
    avg_strength = sum(strength.values()) / max(len(strength), 1)

    def fallback_lambda(team_id: int) -> float:
        # crude prior until Dixon-Coles is fitted on real results
        return 1.35 * (strength[team_id] / avg_strength if avg_strength else 1.0)

    out: dict[int, list[FixtureContext]] = {t["id"]: [] for t in bs["teams"]}
    for fx in client.fixtures(gw=gw):
        h, a = fx["team_h"], fx["team_a"]
        lam_h = ensemble_team_xg(market.get(short[h]), fallback_lambda(h))
        lam_a = ensemble_team_xg(market.get(short[a]), fallback_lambda(a))
        n_h, n_a = len(out[h]), len(out[a])
        out[h].append(FixtureContext(lam_h, lam_a,
                                     rotation_decay=1.0 if n_h == 0 else 0.85))
        out[a].append(FixtureContext(lam_a, lam_h,
                                     rotation_decay=1.0 if n_a == 0 else 0.85))
    return out


def compute_xpts(client: FPLClient, db: EngineDB, gw: int,
                 market: dict[str, float]) -> tuple[dict[int, float], dict[int, object]]:
    """Returns ({player_id: xPts}, {player_id: XMinsEstimate})."""
    players = client.players_frame()
    contexts = build_fixture_contexts(client, gw, market)
    store = HistoryStore(db=db, client=client)
    cache = store.load_all()
    engine = MinutesEngine(StartsModel(manager_rotation_index=0.25))

    bs = client.bootstrap()
    team_xg_ctx = {t["id"]: 1.5 for t in bs["teams"]}   # avg team xG/90 prior

    xpts: dict[int, float] = {}
    xmins: dict[int, object] = {}
    for p in players:
        hist = cache.get(p["id"], [])
        flag = FLAG_FROM_CHANCE.get(p["chance_next"], "100")
        if p["status"] in ("i", "s", "u"):     # injured / suspended / unavailable
            flag = "0"
        est = engine.estimate(hist, days_rest=7, flag=flag)

        # Pre-season / no-data fallback: derive priors from price + ownership,
        # otherwise every player looks identical and the optimizer buys junk.
        if not hist or p["minutes"] < 90:
            xg90, xa90, ps_prior = preseason_priors(p)
            if flag != "0":
                est = XMinsEstimate(
                    p_start=round(ps_prior, 3), p_cameo=0.5,
                    e_mins_start=78.0, e_mins_cameo=18.0,
                    xmins=round(ps_prior * 78.0 + (1 - ps_prior) * 0.5 * 18.0, 1),
                    p60=round(ps_prior * 0.88, 3),
                    p_any=round(ps_prior + (1 - ps_prior) * 0.5, 3),
                    flags=["preseason prior (price/ownership)"])
        else:
            mins90 = max(p["minutes"], 90)
            xg90 = p["xg"] / mins90 * 90
            xa90 = p["xa"] / mins90 * 90

        xmins[p["id"]] = est
        prof = PlayerProfile(
            player_id=p["id"], name=p["web_name"], team=p["team"],
            position=p["position"], baseline_xmins=est.e_mins_start,
            xg90=xg90, xa90=xa90,
            team_xg90_share_context=team_xg_ctx.get(p["team_id"], 1.5),
        )
        fixtures = contexts.get(p["team_id"], [])
        xpts[p["id"]] = round(xpts_gameweek(prof, fixtures, est), 3)

    print(f"[xpts] GW{gw}: computed for {len(xpts)} players")
    return xpts, xmins


def my_squad(client: FPLClient, gw: int) -> tuple[set[int], float, int]:
    """(squad ids, bank, free transfers) — falls back to empty/blank draft."""
    if not MY_ENTRY_ID:
        print("[squad] MY_ENTRY_ID not set — solving from scratch")
        return set(), 100.0, 1
    try:
        data = client.entry_picks(int(MY_ENTRY_ID), max(gw - 1, 1))
        ids = {p["element"] for p in data["picks"]}
        bank = data.get("entry_history", {}).get("bank", 0) / 10.0
        return ids, bank, 1
    except Exception as e:
        print(f"[squad] could not load entry ({e}) — solving from scratch")
        return set(), 100.0, 1


def solve(client: FPLClient, gw: int, xpts_by_gw: dict[int, dict[int, float]],
          forced_out: set[int] | None = None
          ) -> tuple[object, dict[int, str], dict[int, str]]:
    players = client.players_frame()
    names = {p["id"]: f"{p['web_name']} ({p['team']})" for p in players}
    positions = {p["id"]: p["position"] for p in players}
    squad, bank, ft = my_squad(client, gw)
    gws = sorted(xpts_by_gw)

    pool = [OptPlayer(id=p["id"], name=p["web_name"], team=p["team"],
                      position=p["position"], buy_price=p["now_cost"],
                      sell_price=p["now_cost"] if p["id"] in squad else None,
                      xpts={g: xpts_by_gw[g].get(p["id"], 0.0) for g in gws})
            for p in players if p["status"] != "u"]

    opt = SquadOptimizer(pool, gws, current_squad=squad, bank=bank,
                         free_transfers=ft, forced_out=forced_out)
    return opt.solve(time_limit=120), names, positions


def package(res, names: dict[int, str], positions: dict[int, str],
            gw: int) -> dict:
    nm = lambda i: names.get(i, str(i))
    xi = res.starting.get(gw, [])
    by_pos = {pos: [i for i in xi if positions.get(i) == pos]
              for pos in ("GKP", "DEF", "MID", "FWD")}
    formation = (f"{len(by_pos['DEF'])}-{len(by_pos['MID'])}"
                 f"-{len(by_pos['FWD'])}")
    lineup = "\n".join(f"{pos}: " + ", ".join(map(nm, by_pos[pos]))
                       for pos in ("GKP", "DEF", "MID", "FWD") if by_pos[pos])
    return {
        "transfers_in": ", ".join(map(nm, res.transfers_in.get(gw, []))) or "",
        "transfers_out": ", ".join(map(nm, res.transfers_out.get(gw, []))) or "",
        "hits": res.hits.get(gw, 0),
        "captain": nm(res.captain.get(gw, -1)),
        "vice": nm(res.vice.get(gw, -1)),
        "bench_order": ", ".join(map(nm, res.bench_order.get(gw, []))),
        "lineup": lineup,
        "formation": formation,
        "projected_pts": res.objective,
        "summary": f"status={res.status} | XI ({formation}): "
                   + "; ".join(map(nm, xi)),
    }


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------

def run(gw: int, mode: str) -> None:
    client, db = FPLClient(), EngineDB()

    if mode == "weekly":
        print("[history] weekly refresh starting (this takes a few minutes)")
        HistoryStore(db=db, client=client).refresh()

    # 1. snapshot
    players = client.players_frame()
    for p in players:
        p["player_id"] = p.pop("id")
    print(f"[snapshot] {db.save_player_snapshot(gw, players)} players")
    for mid in RIVAL_IDS:
        try:
            db.save_rival_picks(gw, mid, client.entry_picks(mid, max(gw - 1, 1))["picks"])
        except Exception as e:
            print(f"[rival {mid}] skipped ({e})")

    # 2. odds
    market = fetch_market_lambdas(set())
    if market:
        try:
            from odds_client import OddsClient
            db.save_odds(gw, OddsClient().fetch_calibrated())
        except Exception as e:
            print(f"[odds] save skipped ({e})")

    # 3. xMins + xPts for the horizon
    xpts_by_gw: dict[int, dict[int, float]] = {}
    xmins_now = {}
    for g in range(gw, gw + HORIZON):
        xp, xm = compute_xpts(client, db, g, market)
        xpts_by_gw[g] = xp
        if g == gw:
            xmins_now = xm
            db.save_xmins(gw, xm)
            db.save_xpts(gw, xp)

    # 4. emergency check (deadline mode only)
    forced_out: set[int] = set()
    if mode == "deadline":
        for pid, est in xmins_now.items():
            if getattr(est, "xmins", 0) <= 10:
                forced_out.add(pid)
        print(f"[emergency] {len(forced_out)} players excluded (xMins <= 10)")

    # 5. solve
    res, names, positions = solve(client, gw, xpts_by_gw, forced_out)
    rec = package(res, names, positions, gw)
    db.save_recommendation(gw, mode, rec)

    # 6. report
    brief = notify.format_brief(gw, rec, urgent=(mode == "deadline" and bool(forced_out)))
    print("\n" + "=" * 46 + f"\n{brief}\n" + "=" * 46)
    notify.send(brief)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw", type=int, required=True)
    ap.add_argument("--mode", choices=["daily", "weekly", "deadline"], default="daily")
    a = ap.parse_args()
    run(a.gw, a.mode)
