# FPL Decision-Support Engine

Quantitative Fantasy Premier League engine. Computes expected points for every
player, solves for the optimal squad path over a rolling 5-gameweek horizon, and
pushes transfer/captain/bench recommendations to Telegram. Goal: win a
competitive classic mini-league over the 2026/27 season.

**Season starts Friday 21 August 2026** (Arsenal v Coventry). GW1 deadline is
roughly 18:30 BST that day.

---

## Stack & deployment

- Python 3.12, no framework
- **Neon** (PostgreSQL) — all persistence, via SQLAlchemy Core
- **GitHub Actions** — the runtime. No server. Cron schedules in
  `.github/workflows/pipeline.yml`
- Telegram bot for notifications
- Netlify + React dashboard planned (not built) — will read from Neon

Local dev: `python -m venv .venv && pip install -r requirements.txt`, with a
`.env` holding `DATABASE_URL`. SQLite works for tests:
`EngineDB("sqlite:///fpl_test.db")`.

---

## Architecture

```
FPL API ──┐
Odds API ─┼─> xMins (M3) ─> xPts (M2) ─> MILP optimizer (M6) ─> Neon ─> Telegram
Neon cache┘
```

| File | Role | Status |
|---|---|---|
| `module1_data_ingestion.py` | FPL API client, vig-stripping, Brent lambda extraction | done |
| `module2_xpts_engine.py` | Dixon-Coles, 60/40 ensemble, xPts per player | done, needs real DC fit |
| `module3_minutes_engine.py` | Starts probability, xMins, news overrides, emergency trigger | done |
| `module6_optimizer.py` | Multi-period MILP (PuLP/CBC) | done |
| `db.py` | Neon schema + readers/writers | done |
| `odds_client.py` | the-odds-api.com → MatchOdds → CalibratedMatch | done, **untested against live API** |
| `history_store.py` | Weekly element-summary pull → MatchRecord cache | done, **untested against live API** |
| `notify.py` | Telegram push | done |
| `gameweek.py` | GW/deadline resolution + Actions deadline gate (stdlib only) | done |
| `run_pipeline.py` | Orchestrator (daily/weekly/deadline modes) | done, **untested end-to-end** |

**Not built:** Module 4 (price-change prediction), Module 5 (chip strategy),
Module 7 (mini-league game theory), Module 8 (backtesting/recalibration).
Hooks exist for all four — see "Next work" below.

---

## Data flow conventions

- Player identity is the **FPL element id** everywhere. Never join on names.
- Team identity: FPL `short_name` ("ARS"). `odds_client.TEAM_MAP` converts
  bookmaker full names. **Verify this map each season** — promoted clubs change.
- Money is in millions as float (4.5), not FPL's tenths-of-millions int (45).
  `players_frame()` already divides by 10.
- `xpts_gameweek(profile, fixtures, xmins_est)` — always pass the Module 3
  `XMinsEstimate` as the third arg. The no-arg path is a legacy fallback.
- DGW = multiple `FixtureContext` in the list; BGW = empty list → 0.0.
- Everything written to Neon carries `captured_at` so history is reconstructable.
- **Gameweek is auto-detected.** `run_pipeline.py` without `--gw` plans for the
  FPL `is_next` event (upcoming deadline), falling back to `is_current` at
  season end — see `gameweek.resolve_gameweek`. Scheduled Actions never pass
  `--gw`; a manual dispatch passes it only if a number is typed. There is no
  `CURRENT_GW` repo variable to maintain.
- **Deadline runs follow the real deadline.** The deadline-watch cron fires
  every 30 min (`:07`/`:37`); `gameweek.py` gates each slot on the runner's
  bare `python3` before any install. Full runs happen hourly from 6h out,
  every slot in the last 3h, never inside 15 min (a run takes ~15 min).
  Simulated with GitHub cron delays up to 25 min: always >= 1 run 30-90 min
  before the deadline, ~8.5 full runs (= odds calls) per GW. Tune via
  `WATCH_HOURS` / `DENSE_HOURS` / `MIN_LEAD_HOURS`. `gameweek.py` must stay
  stdlib-only.

---

## Environment / secrets

| Name | Required | Purpose |
|---|---|---|
| `DATABASE_URL` | yes | Neon connection string |
| `ODDS_API_KEY` | no* | the-odds-api.com; without it xPts loses the 60% market leg |
| `MY_ENTRY_ID` | no* | your FPL manager id; without it the optimizer solves from scratch |
| `RIVAL_IDS` | no | comma-separated manager ids for the EO layer |
| `TELEGRAM_TOKEN` / `TELEGRAM_CHAT_ID` | no | phone notifications |

\* strongly recommended before GW1.

---

## Testing

Every module has a runnable `__main__` demo with synthetic data.
**Always run the affected demo after editing** — that's the regression check:

```bash
python module2_xpts_engine.py     # xPts sanity (Saka ~8 pts vs weak opponent)
python module3_minutes_engine.py  # xMins + emergency trigger
python module6_optimizer.py       # MILP solves to Optimal, 1 FT/week, 0 hits
python odds_client.py             # offline mocked payload
python notify.py                  # message formatting
python gameweek.py --now 2026-10-17T08:07Z   # deadline gate dry-run (live FPL API)
```

Do not commit real API keys, entry ids, or the `.env` file.

---

## Next work (priority order, before 21 August)

1. **End-to-end live test.** `run_pipeline.py --gw 1 --mode weekly` has never
   run against the real APIs. Expect breakage in `odds_client.to_match_odds`
   (payload shape) and `history_store.refresh` (rate limits).
2. **Odds API validation.** Confirm the-odds-api returns `team_totals` for EPL
   on the free tier. If not, the match-total fallback in `to_match_odds` is
   crude — replace with a proper supremacy/total split.
3. **Fit Dixon-Coles on real data.** `build_fixture_contexts` currently uses a
   team-strength prior as the DC leg. Load 2025/26 results and fit properly.
4. **Post-GW results ingestion** → `db.save_gw_results()`. Nothing writes this
   table yet, so Module 8 has no ground truth.
5. **Module 8** — MAE/Brier per GW, then adaptive ensemble reweighting.
6. **Module 7** — EO layer. `db.rival_eo()` and `OptPlayer.eo_tilt` already
   exist; needs the leading/chasing logic and wiring into `solve()`.
7. **Module 5** — chip strategy (FH/BB/TC/WC) as binary vars in the MILP.
8. **Module 4** — price-change prediction. Note `sell_price` is currently set
   equal to `buy_price` in `run_pipeline.solve()`; implement the 50%-retention
   rule properly once purchase prices are tracked.
9. **Netlify dashboard** reading `recommendations` + `xpts_predictions`.

## Known issues

- `history_store._infer_started` uses minutes >= 60 as a proxy; FPL history has
  no explicit starter flag. Consider cross-referencing fixture lineup data.
- `compute_xpts` uses a flat 1.5 team-xG/90 context prior for share allocation.
  Should be per-team rolling actuals.
- The optimizer treats all owned players' sell price as current price (no
  profit tracking yet).
- Early season (GW1-4) predictions will be weak — thin current-season data.
  The betting-odds leg carries most of the signal until then.
