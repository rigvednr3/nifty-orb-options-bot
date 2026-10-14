"""
Backtest the exact same ORB + Greeks logic the live bot uses.

    python backtest.py --spot data/nifty_5min.csv --vix data/vix_daily.csv

spot CSV : datetime,open,high,low,close        (5-minute NIFTY 50 candles, IST, candle START time)
vix  CSV : date,close                           (daily INDIA VIX close)

IMPORTANT LIMITATION: historical option quotes are not used. Option premiums are MODELLED with
Black-Scholes using the previous day's India VIX as IV (+ a small premium). Real premiums have skew,
spreads and intraday IV moves (incl. IV crush), so treat results as a first filter, not proof.
Validate with paper trading on real quotes before going live.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

from greeks import bs_price, years_to_expiry
from strategy import (Position, build_opening_range, day_filters, detect_breakout, evaluate_entry, hhmm,
                      iv_rank, nearest_expiry_on_weekday, round_trip_costs, strike_grid)

BASE = Path(__file__).parent


def load(spot_path, vix_path):
    s = pd.read_csv(spot_path, parse_dates=["datetime"]).set_index("datetime").sort_index()
    s.index = s.index.tz_localize(None) if s.index.tz else s.index
    v = pd.read_csv(vix_path, parse_dates=["date"]).set_index("date")["close"].sort_index()
    return s, v


def run(spot: pd.DataFrame, vix: pd.Series, cfg: dict, iv_premium_pts: float = 0.5, events: set = frozenset()):
    g, o, u, sess = cfg["greeks"], cfg["orb"], cfg["underlying"], cfg["session"]
    r, q = g["risk_free_rate"], g["dividend_yield"]
    tf = o["confirm_tf_minutes"]
    lot = u["lot_size"]
    slip = cfg["costs"]["slippage_pct"] / 100
    trades, skips = [], Counter()
    vix_hist: list[float] = []
    prev_close = None

    for day, d5 in spot.groupby(spot.index.date):
        vprev = vix[vix.index.date < day]
        if vprev.empty:
            prev_close = float(d5["close"].iloc[-1])
            continue
        vx = float(vprev.iloc[-1])                     # yesterday's VIX close: no look-ahead
        rank = iv_rank(vx, vix_hist, cfg)
        vix_hist.append(vx)

        orb = build_opening_range(d5, cfg)
        if orb is None:
            continue
        ok, why = day_filters(orb, prev_close, vx, str(day) in events, rank, cfg)
        prev_close = float(d5["close"].iloc[-1])
        if not ok:
            skips["day-" + (" ".join(why.split()[:2]) if why.startswith("IV") else why.split()[0])] += 1
            continue

        conf = d5.resample(f"{tf}min", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        expiry = nearest_expiry_on_weekday(day, u["expiry_weekday"], g["min_dte_days"])
        iv = (vx + iv_premium_pts) / 100
        taken, dirs, day_rejects = 0, [], set()
        after = conf[conf.index.time >= hhmm(sess["orb_end"])]

        for ts, c in after.iterrows():
            decide_at = ts + timedelta(minutes=tf)                 # candle has CLOSED
            if decide_at.time() >= hhmm(sess["no_new_entries_after"]) or taken >= o["max_trades_per_day"]:
                break
            if trades and trades[-1]["exit_dt"] > decide_at:       # still in a trade
                continue
            day_pnl = sum(t["pnl"] for t in trades if t["date"] == day)
            if day_pnl <= -cfg["risk"]["daily_loss_limit"]:
                break
            direction = detect_breakout(float(c["close"]), orb, cfg)
            if not direction or (dirs and not (o["allow_reversal"] and direction not in dirs)):
                continue
            S = float(c["close"])
            T = years_to_expiry(decide_at, expiry)
            chain = [{"symbol": f"NIFTY{expiry:%y%m%d}{int(K)}{direction}", "strike": K, "type": direction,
                      "ltp": round(bs_price(S, K, T, r, iv, direction, q), 2)}
                     for K in strike_grid(S, u["strike_step"], g["strikes_each_side"])]
            d = evaluate_entry(direction, S, decide_at, expiry, chain, vx, orb, lot, cfg)
            if not d.take:
                day_rejects.add("rejected-" + (d.reason.split(":")[1].split()[0] if d.reason.startswith("Greeks")
                                                else d.reason.split(":")[0]))
                continue

            k = d.candidate
            entry_px = k.ltp * (1 + slip)
            qty = d.lots * lot
            pos = Position(direction, k.symbol, k.strike, expiry, qty, entry_px, S, d.stop_spot, d.target_spot,
                           d.risk_pts, entry_px * (1 - cfg["risk"]["premium_stop_pct"] / 100), decide_at)

            def opt_at(spot_px, when):
                return bs_price(spot_px, k.strike, years_to_expiry(when, expiry), r, iv, direction, q)

            exit_spot = exit_dt = reason = None
            for t5, b in d5[d5.index >= decide_at].iterrows():
                if t5.time() >= hhmm(sess["square_off"]):
                    exit_spot, exit_dt, reason = float(b["open"]), t5, "TIME_EXIT"
                    break
                adverse = b["low"] if direction == "CE" else b["high"]
                favour = b["high"] if direction == "CE" else b["low"]
                for px in (adverse, favour, b["close"]):              # pessimistic intrabar order
                    stop_before = pos.stop_spot
                    rs = pos.update(float(px), opt_at(float(px), t5), t5, cfg)
                    if rs:
                        reason, exit_dt = rs, t5 + timedelta(minutes=5)
                        exit_spot = {"INDEX_STOP": stop_before, "TRAIL_STOP": stop_before,
                                     "TARGET": pos.target_spot}.get(rs, float(px))
                        break
                if reason:
                    break
            if reason is None:
                last_t = d5.index[-1]
                exit_spot, exit_dt, reason = float(d5["close"].iloc[-1]), last_t, "TIME_EXIT"

            raw_exit = opt_at(exit_spot, exit_dt)
            if reason == "PREMIUM_STOP":
                raw_exit = min(raw_exit, pos.premium_stop)
            exit_px = raw_exit * (1 - slip)
            costs = round_trip_costs(entry_px, exit_px, qty, cfg)
            pnl = (exit_px - entry_px) * qty - costs
            trades.append({"date": day, "dir": direction, "strike": k.strike, "expiry": expiry, "lots": d.lots,
                           "entry_dt": decide_at, "exit_dt": exit_dt, "entry_spot": round(S, 1),
                           "exit_spot": round(exit_spot, 1), "entry_px": round(entry_px, 2),
                           "exit_px": round(exit_px, 2), "delta": round(k.g.delta, 2), "iv": round(iv * 100, 1),
                           "theta_pct": round(k.theta_pct, 1), "edge": round(k.edge, 2), "reason": reason,
                           "costs": round(costs, 2), "pnl": round(pnl, 2)})
            taken += 1
            dirs.append(direction)
        if not taken:
            for why in day_rejects:
                skips[why] += 1

    return pd.DataFrame(trades), skips


def summarize(t: pd.DataFrame, skips: Counter, days: int, capital: float) -> str:
    out = [f"Trading days: {days}"]
    if t.empty:
        return "\n".join(out + ["No trades.", f"Skips: {dict(skips)}"])
    wins, losses = t[t.pnl > 0], t[t.pnl <= 0]
    eq = t.pnl.cumsum()
    dd = (eq - eq.cummax()).min()
    pf = wins.pnl.sum() / abs(losses.pnl.sum()) if len(losses) and losses.pnl.sum() else float("inf")
    out += [
        f"Trades: {len(t)}   Win rate: {len(wins)/len(t)*100:.1f}%",
        f"Avg win: ₹{wins.pnl.mean() if len(wins) else 0:,.0f}   Avg loss: ₹{losses.pnl.mean() if len(losses) else 0:,.0f}",
        f"Profit factor: {pf:.2f}   Expectancy/trade: ₹{t.pnl.mean():,.0f}",
        f"Net P&L (after costs): ₹{t.pnl.sum():,.0f}  ({t.pnl.sum()/capital*100:.1f}% of capital)",
        f"Total costs paid: ₹{t.costs.sum():,.0f}",
        f"Max drawdown: ₹{dd:,.0f}  ({dd/capital*100:.1f}% of capital)",
        "Exits: " + ", ".join(f"{k} {v}" for k, v in t.reason.value_counts().items()),
        "Days/breakouts skipped: " + ", ".join(f"{k} {v}" for k, v in skips.most_common()),
    ]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spot", default=str(BASE / "data/nifty_5min.csv"))
    ap.add_argument("--vix", default=str(BASE / "data/vix_daily.csv"))
    ap.add_argument("--iv-premium", type=float, default=0.5, help="vol pts added to VIX to model option IV")
    ap.add_argument("--out", default=str(BASE / "data/backtest_trades.csv"))
    a = ap.parse_args()

    cfg = yaml.safe_load(open(BASE / "config.yaml"))
    ev_path = BASE / cfg["files"]["event_days"]
    events = {l.strip() for l in ev_path.read_text().splitlines()
              if l.strip() and not l.startswith("#")} if ev_path.exists() else set()
    spot, vix = load(a.spot, a.vix)
    trades, skips = run(spot, vix, cfg, a.iv_premium, events)
    report = summarize(trades, skips, len(set(spot.index.date)), cfg["risk"]["capital"])
    print(report)
    trades.to_csv(a.out, index=False)
    print(f"\nTrades → {a.out}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not trades.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(pd.to_datetime(trades.date), trades.pnl.cumsum(), lw=1.6)
            ax.axhline(0, color="grey", lw=0.8)
            ax.set_title("Equity curve (net of costs)")
            ax.set_ylabel("₹")
            fig.tight_layout()
            png = Path(a.out).with_suffix(".png")
            fig.savefig(png, dpi=120)
            print(f"Equity curve → {png}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
