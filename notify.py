"""
notify.py — Push recommendations to your phone (Telegram)
==========================================================
Setup (one time, ~3 minutes):
  1. In Telegram, message @BotFather -> /newbot -> follow prompts -> copy the token
  2. Message your own new bot once (say "hi") so it can reply to you
  3. Visit https://api.telegram.org/bot<TOKEN>/getUpdates -> copy "chat":{"id": ...}
  4. Add GitHub secrets: TELEGRAM_TOKEN and TELEGRAM_CHAT_ID

If the secrets are absent, notifications silently no-op so the pipeline
still runs (useful for local testing).
"""

from __future__ import annotations

import os
from typing import Optional

import requests


def send(text: str, token: Optional[str] = None,
         chat_id: Optional[str] = None) -> bool:
    token = token or os.environ.get("TELEGRAM_TOKEN")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[notify] no Telegram credentials — skipping push")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=15)
        r.raise_for_status()
        print("[notify] sent to Telegram")
        return True
    except requests.RequestException as e:
        print(f"[notify] failed: {e}")
        return False


def format_brief(gw: int, rec: dict, urgent: bool = False) -> str:
    head = f"🚨 *URGENT — GW{gw}*" if urgent else f"⚽ *FPL Engine — GW{gw}*"
    lines = [head, ""]
    if rec.get("transfers_in"):
        lines.append(f"*Transfer:* {rec['transfers_out']} ➡️ {rec['transfers_in']}")
        if rec.get("hits"):
            lines.append(f"_Cost: -{rec['hits'] * 4} pts hit_")
    else:
        lines.append("*Transfer:* none — roll the FT")
    lines += [
        f"*Captain:* {rec.get('captain', '?')}",
        f"*Vice:* {rec.get('vice', '?')}",
    ]
    if rec.get("lineup"):
        formation = f" ({rec['formation']})" if rec.get("formation") else ""
        lines += ["", f"*XI{formation}:*", rec["lineup"]]
    lines += [
        f"*Bench order:* {rec.get('bench_order', '?')}",
        "",
        f"Projected (5 GW): {rec.get('projected_pts', 0):.1f} pts",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    demo = {"transfers_in": "Isak", "transfers_out": "Havertz", "hits": 0,
            "captain": "Salah", "vice": "Saka",
            "bench_order": "Mitchell, Wissa, Rogers",
            "lineup": ("GKP: Raya\nDEF: Gabriel, Timber, Calafiori\n"
                       "MID: Saka, Palmer, Ødegaard, Rice\n"
                       "FWD: Haaland, Isak, João Pedro"),
            "formation": "3-4-3",
            "projected_pts": 312.4}
    print(format_brief(12, demo))
