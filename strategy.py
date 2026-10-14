"""
Core decision logic — shared by the live bot and the backtester so both trade identically.

Flow each day:
  1. build_opening_range()   → high/low of 09:15–09:45
  2. day_filters()           → gap, range size, VIX band, event day, IV rank
  3. detect_breakout()       → confirm-timeframe candle CLOSES beyond the range  → CE or PE
  4. evaluate_entry()        → Greeks gate picks the strike (or rejects the trade) and sizes it
  5. Position.update()       → index stop / breakeven+trail / target / premium stop / time exit
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, time

import pandas as pd

from greeks import Greeks, bs_price, greeks, implied_vol, session_hours_to_years, years_to_expiry


def hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


# ───────────────────────── 1. Opening range ─────────────────────────
@dataclass
class OpeningRange:
    high: float
    low: float
    open: float

    @property
    def size(self) -> float:
        return self.high - self.low

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2

    def size_pct(self) -> float:
        return self.size / self.open * 100


def build_opening_range(candles: pd.DataFrame, cfg: dict) -> OpeningRange | None:
    """candles: intraday OHLC with a DatetimeIndex (candle START times), any timeframe ≤ 30 min."""
    start, end = hhmm(cfg["session"]["market_open"]), hhmm(cfg["session"]["orb_end"])
    w = candles[(candles.index.time >= start) & (candles.index.time < end)]
    if w.empty:
        return None
    return OpeningRange(high=float(w["high"].max()), low=float(w["low"].min()), open=float(w["open"].iloc[0]))


# ───────────────────────── 2. Day filters ─────────────────────────
def day_filters(orb: OpeningRange, prev_close: float | None, vix: float, is_event_day: bool,
                atm_iv_rank: float | None, cfg: dict) -> tuple[bool, str]:
    o, g = cfg["orb"], cfg["greeks"]
    if is_event_day:
        return False, "event day"
    if prev_close:
        gap = abs(orb.open - prev_close) / prev_close * 100
        if gap > o["max_gap_pct"]:
            return False, f"gap {gap:.2f}% > {o['max_gap_pct']}%"
    rp = orb.size_pct()
    if rp < o["min_range_pct"]:
        return False, f"range {rp:.2f}% too narrow"
    if rp > o["max_range_pct"]:
        return False, f"range {rp:.2f}% too wide"
    if not (g["vix_min"] <= vix <= g["vix_max"]):
        return False, f"VIX {vix:.1f} outside {g['vix_min']}–{g['vix_max']}"
    if atm_iv_rank is not None and atm_iv_rank > g["max_iv_rank"]:
        return False, f"IV rank {atm_iv_rank:.0f} > {g['max_iv_rank']} (IV-crush risk)"
    return True, f"ok (range {orb.size:.0f} pts / {rp:.2f}%)"


def iv_rank(current: float, history: list[float], cfg: dict) -> float | None:
    g = cfg["greeks"]
    h = history[-g["iv_rank_lookback"]:]
    if len(h) < g["iv_rank_min_history"]:
        return None
    lo, hi = min(h), max(h)
    return 50.0 if hi - lo < 1e-9 else (current - lo) / (hi - lo) * 100


# ───────────────────────── 3. Breakout ─────────────────────────
def detect_breakout(candle_close: float, orb: OpeningRange, cfg: dict) -> str | None:
    b = cfg["orb"]["breakout_buffer_pts"]
    if candle_close > orb.high + b:
        return "CE"
    if candle_close < orb.low - b:
        return "PE"
    return None


def index_stop_target(direction: str, entry_spot: float, orb: OpeningRange, cfg: dict):
    o = cfg["orb"]
    sign = 1 if direction == "CE" else -1
    level = orb.high if direction == "CE" else orb.low
    if o["stop_mode"] == "opposite":
        stop = orb.low if direction == "CE" else orb.high
    else:
        stop = orb.mid
    # Target is projected from the breakout LEVEL, not the entry, so a late (chased) entry gets
    # less reward and more risk — the edge gate then naturally rejects it.
    target = level + sign * o["target_r"] * abs(level - stop)
    risk = abs(entry_spot - stop)
    return stop, target, risk


# ───────────────────────── 4. Greeks gate ─────────────────────────
@dataclass
class Candidate:
    symbol: str
    strike: float
    opt: str
    ltp: float
    g: Greeks
    theta_pct: float = 0.0      # theta cost over the expected hold, % of premium
    gain_at_target: float = 0.0
    loss_at_stop: float = 0.0
    edge: float = 0.0
    breakdown: dict = field(default_factory=dict)
    fail: str = ""


@dataclass
class Decision:
    take: bool
    reason: str
    direction: str = ""
    candidate: Candidate | None = None
    lots: int = 0
    stop_spot: float = 0.0
    target_spot: float = 0.0
    risk_pts: float = 0.0
    premium_stop: float = 0.0
    log: list[str] = field(default_factory=list)


def evaluate_entry(direction: str, spot: float, now: datetime, expiry: date, chain: list[dict],
                   vix: float, orb: OpeningRange, lot_size: int, cfg: dict) -> Decision:
    """
    chain: [{"symbol", "strike", "type": "CE"/"PE", "ltp", optional "bid","ask"}]
    Returns a Decision. Every rejection carries a human-readable reason.
    """
    g, o, rk = cfg["greeks"], cfg["orb"], cfg["risk"]
    r, q = g["risk_free_rate"], g["dividend_yield"]
    level = orb.high if direction == "CE" else orb.low

    ext = abs(spot - level) / spot * 100
    if ext > o["max_entry_extension_pct"]:
        return Decision(False, f"chasing: spot {ext:.2f}% beyond level", direction)

    stop_spot, target_spot, risk_pts = index_stop_target(direction, spot, orb, cfg)
    T = years_to_expiry(now, expiry)
    T_win = max(T - session_hours_to_years(g["expected_hold_hours"]), 1e-6)   # winners take longer
    T_loss = max(T - session_hours_to_years(g["stop_hold_hours"]), 1e-6)      # stops usually hit sooner
    sign = 1 if direction == "CE" else -1
    move = target_spot - spot
    hold_days = g["expected_hold_hours"] / 6.25          # fraction of one trading session

    def iv_at(iv0: float, s_exit: float) -> float:
        # Nifty IV falls when the index rises and rises when it falls (spot-vol beta),
        # plus a small baseline drift down while held. Hurts CE buyers, helps PE buyers.
        chg = -g["spot_vol_beta"] * (s_exit / spot - 1) * 100 - g["iv_drift_pts"]
        return max(iv0 + chg / 100, 0.01)

    cands: list[Candidate] = []
    for row in chain:
        if row["type"] != direction or row["ltp"] <= 0:
            continue
        iv = implied_vol(row["ltp"], spot, row["strike"], T, r, direction, q)
        if iv is None:
            continue
        gk = greeks(spot, row["strike"], T, r, iv, direction, q)
        c = Candidate(row["symbol"], row["strike"], direction, row["ltp"], gk)
        c.theta_pct = abs(gk.theta_day) * hold_days / row["ltp"] * 100

        # Full revaluation at target and at stop, after the expected holding time,
        # with IV assumed to drop slightly (buyers should assume vol works against them).
        px_target = bs_price(target_spot, row["strike"], T_win, r, iv_at(iv, target_spot), direction, q)
        px_stop = bs_price(stop_spot, row["strike"], T_loss, r, iv_at(iv, stop_spot), direction, q)
        c.gain_at_target = px_target - row["ltp"]
        loss = row["ltp"] - px_stop
        c.loss_at_stop = max(min(loss, row["ltp"] * rk["premium_stop_pct"] / 100), 0.05)
        c.edge = c.gain_at_target / c.loss_at_stop if c.gain_at_target > 0 else 0.0
        c.breakdown = {  # Taylor view of where the P&L comes from (for the log)
            "delta": round(gk.delta * move, 1),
            "gamma": round(0.5 * gk.gamma * move * move, 1),
            "theta": round(gk.theta_day * hold_days, 1),
            "vega": round(gk.vega_pt * (iv_at(iv, target_spot) - iv) * 100, 1),
        }

        d = abs(gk.delta)
        if not (g["delta_min"] <= d <= g["delta_max"]):
            c.fail = f"delta {d:.2f} outside band"
        elif iv * 100 - vix > g["max_iv_over_vix"]:
            c.fail = f"IV {iv*100:.1f} > VIX {vix:.1f}+{g['max_iv_over_vix']}"
        elif c.theta_pct > g["max_hold_theta_pct"]:
            c.fail = f"theta {c.theta_pct:.1f}% of premium over hold, too high"
        elif c.edge < g["min_edge_ratio"]:
            c.fail = f"edge {c.edge:.2f} < {g['min_edge_ratio']}"
        elif "bid" in row and "ask" in row and row["bid"] and row["ask"]:
            mid = (row["bid"] + row["ask"]) / 2
            spread = (row["ask"] - row["bid"]) / mid * 100
            if spread > g["max_spread_pct"]:
                c.fail = f"spread {spread:.1f}% too wide"
        cands.append(c)

    shown = [c for c in cands if g["delta_min"] - 0.1 <= abs(c.g.delta) <= g["delta_max"] + 0.1]
    log = [f"{c.symbol} K={c.strike:.0f} ltp={c.ltp:.1f} Δ={c.g.delta:+.2f} Γ={c.g.gamma:.4f} "
           f"Θ={c.g.theta_day:.1f}/day ({c.theta_pct:.1f}% over hold) V={c.g.vega_pt:.1f} IV={c.g.iv*100:.1f} "
           f"edge={c.edge:.2f} {'✗ ' + c.fail if c.fail else '✓'}" for c in shown]
    ok = [c for c in cands if not c.fail]
    if not ok:
        best = min(cands, key=lambda c: abs(abs(c.g.delta) - g["delta_target"]), default=None)
        why = best.fail if best else "no priceable strikes"
        return Decision(False, f"Greeks gate: {why}", direction, best, log=log)

    # Prefer the best edge; tie-break toward delta_target.
    best = max(ok, key=lambda c: (round(c.edge, 1), -abs(abs(c.g.delta) - g["delta_target"])))

    per_lot_risk = best.loss_at_stop * lot_size * (1 + cfg["costs"]["slippage_pct"] / 100)
    lots = int(rk["risk_per_trade"] // per_lot_risk)
    lots = min(lots, rk["max_lots"], int(rk["capital"] // (best.ltp * lot_size)))
    if lots < 1:
        return Decision(False, f"size: 1 lot risks ₹{per_lot_risk:.0f} > risk_per_trade", direction, best, log=log)

    return Decision(True, f"BUY {best.symbol} ×{lots} lots (edge {best.edge:.2f})", direction, best, lots,
                    stop_spot, target_spot, risk_pts,
                    round(best.ltp * (1 - rk["premium_stop_pct"] / 100), 2), log)


# ───────────────────────── 5. Position / exits ─────────────────────────
@dataclass
class Position:
    direction: str
    symbol: str
    strike: float
    expiry: date
    qty: int
    entry_px: float
    entry_spot: float
    stop_spot: float
    target_spot: float
    risk_pts: float
    premium_stop: float
    entry_time: datetime
    best_fav: float = 0.0
    trailing: bool = False

    @property
    def sign(self) -> int:
        return 1 if self.direction == "CE" else -1

    def update(self, spot: float, opt_ltp: float, now: datetime, cfg: dict) -> str | None:
        """Call on every price update. Returns an exit reason, or None to hold."""
        if now.time() >= hhmm(cfg["session"]["square_off"]):
            return "TIME_EXIT"
        if opt_ltp <= self.premium_stop:
            return "PREMIUM_STOP"
        if (spot - self.stop_spot) * self.sign <= 0:
            return "TRAIL_STOP" if self.trailing else "INDEX_STOP"
        if (spot - self.target_spot) * self.sign >= 0:
            return "TARGET"
        fav = (spot - self.entry_spot) * self.sign
        self.best_fav = max(self.best_fav, fav)
        if self.best_fav >= cfg["orb"]["breakeven_after_r"] * self.risk_pts:
            self.trailing = True
            new_stop = self.entry_spot + self.sign * (self.best_fav - self.risk_pts)
            new_stop = self.entry_spot + self.sign * max((new_stop - self.entry_spot) * self.sign, 0)
            if (new_stop - self.stop_spot) * self.sign > 0:
                self.stop_spot = new_stop
        return None


# ───────────────────────── Costs ─────────────────────────
def round_trip_costs(buy_px: float, sell_px: float, qty: int, cfg: dict) -> float:
    c = cfg["costs"]
    buy_val, sell_val = buy_px * qty, sell_px * qty
    turnover = buy_val + sell_val
    brokerage = 2 * c["brokerage_per_order"]
    stt = sell_val * c["stt_sell_pct"] / 100
    exch = turnover * c["exchange_txn_pct"] / 100
    sebi = turnover * c["sebi_per_crore"] / 1e7
    gst = (brokerage + exch + sebi) * c["gst_pct"] / 100
    stamp = buy_val * c["stamp_buy_pct"] / 100
    return brokerage + stt + exch + sebi + gst + stamp


def nearest_expiry_on_weekday(d: date, weekday: int, min_dte: int) -> date:
    """Backtest helper: next weekly expiry on `weekday` with at least min_dte calendar days left."""
    from datetime import timedelta
    e = d + timedelta(days=(weekday - d.weekday()) % 7)
    while (e - d).days < min_dte:
        e += timedelta(days=7)
    return e


def strike_grid(spot: float, step: int, n: int) -> list[float]:
    atm = round(spot / step) * step
    return [atm + i * step for i in range(-n, n + 1)]


def ceil_tick(x: float, tick: float = 0.05) -> float:
    return round(math.ceil(x / tick) * tick, 2)


def floor_tick(x: float, tick: float = 0.05) -> float:
    return round(max(math.floor(x / tick) * tick, tick), 2)
