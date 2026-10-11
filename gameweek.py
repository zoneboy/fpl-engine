"""
gameweek.py — Gameweek resolution & deadline gate (stdlib only)
================================================================
Which GW are we planning for, when is its deadline, and should a scheduled
deadline-watch run do the full pipeline right now?

Kept free of third-party imports so the GitHub Actions gate can run on the
runner's bare python3 *before* installing requirements — skipped slots then
cost seconds, not a full pip install + 15-minute pipeline.

  python gameweek.py                          # gate: prints decision, sets $GITHUB_OUTPUT run=true|false
  python gameweek.py --now 2026-10-10T08:07Z  # dry-run the gate at a given time
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from datetime import datetime, timezone
from typing import Optional

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

# Deadline-watch policy (the cron fires every 30 min, at :07 and :37):
WATCH_HOURS = 6.0       # start watching this long before the deadline
DENSE_HOURS = 3.0       # every slot inside this (= EmergencyMonitor.WINDOW_HOURS)
MIN_LEAD_HOURS = 0.25   # a full run takes ~15 min; later than this can't land in time


def _deadline(ev: Optional[dict]) -> Optional[datetime]:
    d = (ev or {}).get("deadline_time")
    return datetime.fromisoformat(d.replace("Z", "+00:00")) if d else None


def resolve_gameweek(events: list[dict],
                     gw: Optional[int] = None) -> tuple[int, Optional[datetime]]:
    """(gameweek, its deadline) from bootstrap-static `events`.

    Explicit gw wins. Otherwise the upcoming deadline (`is_next`), then the
    live one (`is_current`, end of season), then the first unfinished event."""
    if gw is not None:
        return gw, _deadline(next((e for e in events if e["id"] == gw), None))
    for key in ("is_next", "is_current"):
        ev = next((e for e in events if e.get(key)), None)
        if ev:
            return ev["id"], _deadline(ev)
    ev = next((e for e in events if not e.get("finished")), events[-1])
    return ev["id"], _deadline(ev)


def should_run_deadline(deadline: Optional[datetime],
                        now: datetime) -> tuple[bool, str]:
    """Gate for scheduled deadline-watch slots: (run?, reason).

    Hourly from WATCH_HOURS out, every slot inside DENSE_HOURS, nothing once
    a run could no longer finish before the deadline. With half-hourly slots
    this guarantees >= 2 runs between 30 and 90 minutes before any deadline,
    even if GitHub delays a scheduled job."""
    if deadline is None:
        return False, "no deadline known"
    hours = (deadline - now).total_seconds() / 3600.0
    if hours <= 0:
        return False, "deadline already passed"
    if hours < MIN_LEAD_HOURS:
        return False, f"{hours * 60:.0f} min left, too late for a ~15 min run"
    if hours > WATCH_HOURS:
        return False, f"{hours:.1f}h to deadline, outside the {WATCH_HOURS:g}h watch"
    if hours > DENSE_HOURS and now.minute >= 30:
        return False, f"{hours:.1f}h to deadline, hourly phase: skipping half-hour slot"
    return True, f"{hours:.2f}h to deadline"


def fetch_events() -> list[dict]:
    req = urllib.request.Request(BOOTSTRAP_URL,
                                 headers={"User-Agent": "fpl-quant-engine/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["events"]


def _gate(now: datetime) -> bool:
    gw, deadline = resolve_gameweek(fetch_events())
    run, why = should_run_deadline(deadline, now)
    when = deadline.strftime("%a %d %b %H:%M UTC") if deadline else "unknown"
    print(f"[gate] GW{gw} deadline {when}: {'RUN' if run else 'skip'} ({why})")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"run={'true' if run else 'false'}\n")
    return run


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--now", help="pretend it is this UTC time (ISO 8601)")
    a = ap.parse_args()
    now = (datetime.fromisoformat(a.now.replace("Z", "+00:00"))
           if a.now else datetime.now(timezone.utc))
    _gate(now)
