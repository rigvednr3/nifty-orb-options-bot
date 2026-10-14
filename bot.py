"""
Live / paper runner. Start it any time before 09:45 on a trading day:

    python bot.py                # uses config.yaml (mode: paper by default)

It sleeps until the opening range is complete, applies the day filters, watches for a breakout
on each completed confirm-timeframe candle, runs the Greeks gate, and manages exits until square-off.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import yaml

from greeks import implied_vol, years_to_expiry
from strategy import (Position, build_opening_range, day_filters, detect_breakout, evaluate_entry,
                      hhmm, iv_rank, round_trip_costs)

BASE = Path(__file__).parent
cfg = yaml.safe_load(open(BASE / "config.yaml"))
F = {k: BASE / v for k, v in cfg["files"].items()}
(BASE / "data").mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(F["log"]), logging.StreamHandler(sys.stdout)])
log = logging.getLogger("bot")


def notify(msg: str):
    """Optional Telegram alert: set TG_BOT_TOKEN and TG_CHAT_ID env vars."""
    log.info(msg)
    tok, chat = os.environ.get("TG_BOT_TOKEN"), os.environ.get("TG_CHAT_ID")
    if tok and chat:
        try:
            data = urllib.parse.urlencode({"chat_id": chat, "text": msg}).encode()
            urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/sendMessage", data, timeout=5)
        except Exception as e:  # alerts must never crash the bot
            log.warning(f"telegram failed: {e}")


def event_days() -> set[str]:
    p = F["event_days"]
    return {l.strip() for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")} if p.exists() else set()


def load_iv_history() -> list[float]:
    p = F["iv_history"]
    if not p.exists():
        return []
    return [float(r["atm_iv"]) for r in csv.DictReader(open(p))]


def save_iv(day: date, iv: float):
    p = F["iv_history"]
    rows = list(csv.DictReader(open(p))) if p.exists() else []
    if any(r["date"] == str(day) for r in rows):
        return
    with open(p, "a", newline="") as f:
        w = csv.writer(f)
        if not rows:
            w.writerow(["date", "atm_iv"])
        w.writerow([day, round(iv, 2)])


def journal(row: dict):
    p = F["trade_journal"]
    new = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def load_state(today: date) -> dict:
    p = F["state"]
    if p.exists():
        s = json.loads(p.read_text())
        if s.get("date") == str(today):
            return s
    return {"date": str(today), "trades": 0, "pnl": 0.0, "directions": [], "open": None}


def save_state(s: dict):
    F["state"].write_text(json.dumps(s, default=str, indent=2))


def sleep_until(t):
    while datetime.now().time() < t:
        time.sleep(5)


def main():
    from broker import KiteExecutor, KiteFeed, PaperExecutor, kite_client

    today = date.today()
    if today.weekday() >= 5:
        log.info("Weekend — nothing to do.")
        return
    kite = kite_client()
    feed = KiteFeed(kite, cfg)
    ex = KiteExecutor(kite, feed, cfg) if cfg["mode"] == "live" else PaperExecutor(feed, cfg)
    state = load_state(today)
    notify(f"ORB bot started ({cfg['mode'].upper()}) {today}")
    if state["open"]:
        notify("⚠ An open position from earlier today is recorded in state.json — check your broker "
               "and close it manually; the bot will not manage it after a restart.")

    s = cfg["session"]
    tf = cfg["orb"]["confirm_tf_minutes"]
    sleep_until((datetime.combine(today, hhmm(s["orb_end"])) + timedelta(seconds=20)).time())

    # ── Opening range + day filters ──
    orb = build_opening_range(feed.candles(tf, today), cfg)
    if orb is None:
        notify("No opening-range candles — holiday or data issue. Exiting.")
        return
    spot, vix = feed.spot(), feed.vix()
    expiry = feed.expiry(today)
    lot = feed.lot_size(expiry)

    atm, px = feed.atm_iv_inputs(expiry, spot)
    T = years_to_expiry(datetime.now(), expiry)
    g = cfg["greeks"]
    ivs = [implied_vol(p, spot, atm, T, g["risk_free_rate"], t, g["dividend_yield"]) for t, p in px.items()]
    ivs = [v for v in ivs if v]
    atm_iv = sum(ivs) / len(ivs) * 100 if ivs else None
    hist = load_iv_history()
    rank = iv_rank(atm_iv, hist, cfg) if atm_iv else None
    if atm_iv:
        save_iv(today, atm_iv)

    ok, why = day_filters(orb, feed.prev_close(), vix, str(today) in event_days(), rank, cfg)
    notify(f"ORB {orb.low:.0f}–{orb.high:.0f} ({orb.size:.0f} pts) | VIX {vix:.1f} | ATM IV "
           f"{atm_iv or 0:.1f} rank {('%.0f' % rank) if rank is not None else 'n/a'} | expiry {expiry} lot {lot}"
           f"\nDay filter: {'PASS' if ok else 'SKIP'} — {why}")
    if not ok:
        return

    pos: Position | None = None
    last_checked = None
    no_entry_after, sq_off = hhmm(s["no_new_entries_after"]), hhmm(s["square_off"])
    o = cfg["orb"]

    while True:
        now = datetime.now()
        try:
            # ── Manage open position ──
            if pos:
                spot, opt = feed.spot(), feed.option_ltp(pos.symbol)
                reason = pos.update(spot, opt, now, cfg)
                if reason:
                    exit_px = ex.sell(pos.symbol, pos.qty)
                    costs = round_trip_costs(pos.entry_px, exit_px, pos.qty, cfg)
                    pnl = (exit_px - pos.entry_px) * pos.qty - costs
                    state["pnl"] += pnl
                    state["open"] = None
                    save_state(state)
                    journal({"date": today, "symbol": pos.symbol, "dir": pos.direction, "qty": pos.qty,
                             "entry_time": pos.entry_time.strftime("%H:%M"), "entry_px": pos.entry_px,
                             "exit_time": now.strftime("%H:%M"), "exit_px": exit_px, "entry_spot": pos.entry_spot,
                             "exit_spot": spot, "reason": reason, "costs": round(costs, 2), "pnl": round(pnl, 2),
                             "mode": cfg["mode"]})
                    notify(f"EXIT {pos.symbol} @ {exit_px} ({reason}) | P&L ₹{pnl:,.0f} | day ₹{state['pnl']:,.0f}")
                    pos = None

            # ── Look for entry on each newly completed candle ──
            can_enter = (pos is None and now.time() < no_entry_after
                         and state["trades"] < o["max_trades_per_day"]
                         and state["pnl"] > -cfg["risk"]["daily_loss_limit"])
            if can_enter:
                c = feed.candles(tf, today)
                if not c.empty and c.index[-1] != last_checked:
                    last_checked = c.index[-1]
                    direction = detect_breakout(float(c["close"].iloc[-1]), orb, cfg)
                    allowed = not state["directions"] or (o["allow_reversal"] and direction
                                                         and direction not in state["directions"])
                    if direction and allowed:
                        spot, vix = feed.spot(), feed.vix()
                        chain = feed.chain(expiry, spot, direction)
                        d = evaluate_entry(direction, spot, now, expiry, chain, vix, orb, lot, cfg)
                        for line in d.log:
                            log.info("  " + line)
                        notify(f"Breakout {direction} @ {spot:.0f}: {'TAKE' if d.take else 'SKIP'} — {d.reason}")
                        if d.take:
                            c0 = d.candidate
                            ref = c0.ltp if not chain else next((r.get("ask") or r["ltp"] for r in chain
                                                                 if r["symbol"] == c0.symbol), c0.ltp)
                            fill = ex.buy(c0.symbol, d.lots * lot, ref)
                            if fill:
                                pos = Position(direction, c0.symbol, c0.strike, expiry, d.lots * lot, fill, spot,
                                               d.stop_spot, d.target_spot, d.risk_pts,
                                               round(fill * (1 - cfg["risk"]["premium_stop_pct"] / 100), 2), now)
                                state["trades"] += 1
                                state["directions"].append(direction)
                                state["open"] = {"symbol": c0.symbol, "qty": pos.qty, "entry": fill}
                                save_state(state)
                                notify(f"ENTRY {c0.symbol} ×{pos.qty} @ {fill} | index stop {pos.stop_spot:.0f} "
                                       f"target {pos.target_spot:.0f} | premium stop {pos.premium_stop}")
                        # A rejected breakout does not use up the day's trade; the next candle is checked again.

            done_for_day = (now.time() >= no_entry_after or now.time() >= sq_off
                            or state["trades"] >= o["max_trades_per_day"]
                            or state["pnl"] <= -cfg["risk"]["daily_loss_limit"])
            if pos is None and done_for_day:
                break
        except Exception as e:
            log.exception(f"loop error: {e}")
            if pos:
                notify(f"⚠ Error while holding {pos.symbol}: {e}. Retrying — watch the position.")
        time.sleep(cfg["execution"]["poll_seconds"])

    notify(f"Day done. Trades {state['trades']} | P&L ₹{state['pnl']:,.0f}")


if __name__ == "__main__":
    main()
