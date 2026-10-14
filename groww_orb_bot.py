"""
Nifty 30-min Opening Range Breakout — Greeks-gated OPTION BUYING bot for GROWW
=============================================================================
Buys a NIFTY CE on an upside breakout / PE on a downside breakout of the 09:15–09:45 candle,
only when the option's Greeks (delta, IV vs VIX, IV rank, theta, modelled edge, spread) justify it.

  pip install growwapi pyotp pandas numpy pyyaml
  export GROWW_API_KEY=...  GROWW_TOTP_SECRET=...      (or GROWW_API_SECRET for key+secret flow)

  python groww_orb_bot.py --check              # test login, symbols, chain, Greeks gate (any time)
  python groww_orb_bot.py --fetch-history 365  # data for backtest.py
  python groww_orb_bot.py                      # run today (start before 09:45 IST)

Defaults to mode: paper (real Groww quotes, simulated fills). To change settings, create
groww_config.yaml next to this file with only the keys you want to override, e.g.
    mode: live
    risk: {max_lots: 1}
All times are IST regardless of the server's timezone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, time

import pandas as pd


# ════════ GREEKS (Black-Scholes) ════════


SECONDS_PER_YEAR = 365.0 * 24 * 3600
EXPIRY_CLOSE = time(15, 30)


def _cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


SESSION_OPEN, SESSION_HOURS = time(9, 15), 6.25


def years_to_expiry(now: datetime, expiry: date) -> float:
    """
    'Trading-clock' time to expiry in years. Today's remaining session counts as a fraction of one day,
    every later calendar day counts as one day (so weekends still decay, like VIX's 365-day convention).
    This makes intraday theta realistic: an option loses roughly one day of theta across the 6.25-hour
    session, not 2/24ths of it over two hours.
    """
    open_dt = datetime.combine(now.date(), SESSION_OPEN)
    close_dt = datetime.combine(now.date(), EXPIRY_CLOSE)
    if now <= open_dt:
        frac = 1.0
    elif now >= close_dt:
        frac = 0.0
    else:
        frac = (close_dt - now).total_seconds() / (SESSION_HOURS * 3600)
    days = frac + max((expiry - now.date()).days, 0)
    return max(days / 365.0, 1 / (365.0 * 24 * 60))


def session_hours_to_years(hours: float) -> float:
    return hours / SESSION_HOURS / 365.0


def _d1d2(S, K, T, r, sigma, q):
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sq
    return d1, d1 - sq


def bs_price(S: float, K: float, T: float, r: float, sigma: float, opt: str, q: float = 0.0) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if opt == "CE" else max(K - S, 0.0)
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    if opt == "CE":
        return S * math.exp(-q * T) * _cdf(d1) - K * math.exp(-r * T) * _cdf(d2)
    return K * math.exp(-r * T) * _cdf(-d2) - S * math.exp(-q * T) * _cdf(-d1)


@dataclass
class Greeks:
    price: float
    delta: float
    gamma: float       # per 1 index point
    theta_day: float   # ₹ per unit per calendar day (negative for long options)
    vega_pt: float     # ₹ per unit per 1 vol point (1%)
    iv: float          # decimal, e.g. 0.13


def greeks(S: float, K: float, T: float, r: float, sigma: float, opt: str, q: float = 0.0) -> Greeks:
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    sqT = math.sqrt(T)
    eq, er = math.exp(-q * T), math.exp(-r * T)
    gamma = eq * _pdf(d1) / (S * sigma * sqT)
    vega = S * eq * _pdf(d1) * sqT / 100.0
    decay = -S * eq * _pdf(d1) * sigma / (2 * sqT)
    if opt == "CE":
        delta = eq * _cdf(d1)
        theta = decay - r * K * er * _cdf(d2) + q * S * eq * _cdf(d1)
    else:
        delta = eq * (_cdf(d1) - 1.0)
        theta = decay + r * K * er * _cdf(-d2) - q * S * eq * _cdf(-d1)
    return Greeks(bs_price(S, K, T, r, sigma, opt, q), delta, gamma, theta / 365.0, vega, sigma)


def implied_vol(price: float, S: float, K: float, T: float, r: float, opt: str, q: float = 0.0,
                lo: float = 0.01, hi: float = 2.5, tol: float = 1e-5) -> float | None:
    """Bisection IV solver. Returns None if the price is outside the no-arbitrage band."""
    if price <= 0 or T <= 0:
        return None
    if price < bs_price(S, K, T, r, lo, opt, q) - 1e-6 or price > bs_price(S, K, T, r, hi, opt, q):
        return None
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, T, r, mid, opt, q) > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)

# ════════ STRATEGY (identical to strategy.py used by backtest.py) ════════





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

# ════════════════════════════════════════════════════════════════════
#  GROWW ADAPTER  (growwapi SDK)
# ════════════════════════════════════════════════════════════════════
import json
import logging
import os
import sys
import time as _time
import urllib.parse
import urllib.request
from datetime import timedelta, timezone
from pathlib import Path

import yaml

IST = timezone(timedelta(hours=5, minutes=30))
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)
log = logging.getLogger("bot")


def now_ist() -> datetime:
    """Always Indian time, even on a UTC cloud server."""
    return datetime.now(IST).replace(tzinfo=None)


DEFAULT_CONFIG = """
mode: paper            # paper = real Groww quotes, simulated fills | live = real orders on Groww
underlying: {strike_step: 50, lot_size: 65, expiry_weekday: 1}   # lot_size is a fallback; live reads it from Groww
groww: {vix_symbol: INDIAVIX}                                    # auto-detected from instrument list if possible
session: {market_open: "09:15", orb_end: "09:45", no_new_entries_after: "13:30", square_off: "15:15"}
orb:
  confirm_tf_minutes: 5
  breakout_buffer_pts: 5
  min_range_pct: 0.25
  max_range_pct: 0.90
  max_gap_pct: 1.0
  max_entry_extension_pct: 0.30
  stop_mode: mid
  target_r: 2.0
  breakeven_after_r: 1.0
  max_trades_per_day: 1
  allow_reversal: false
greeks:
  risk_free_rate: 0.065
  dividend_yield: 0.012
  min_dte_days: 2
  strikes_each_side: 6
  delta_min: 0.45
  delta_max: 0.72
  delta_target: 0.58
  max_hold_theta_pct: 8.0
  vix_min: 10.0
  vix_max: 24.0
  max_iv_over_vix: 3.0
  max_iv_rank: 80
  iv_rank_lookback: 60
  iv_rank_min_history: 20
  expected_hold_hours: 2.0
  stop_hold_hours: 1.0
  spot_vol_beta: 0.8
  iv_drift_pts: 0.2
  min_edge_ratio: 1.0
  max_spread_pct: 1.5
risk: {capital: 200000, risk_per_trade: 3000, max_lots: 4, premium_stop_pct: 35, daily_loss_limit: 4000}
execution: {limit_buffer_pct: 0.5, fill_timeout_sec: 8, poll_seconds: 5}
costs: {brokerage_per_order: 20, stt_sell_pct: 0.15, exchange_txn_pct: 0.03503, gst_pct: 18,
        stamp_buy_pct: 0.003, sebi_per_crore: 10, slippage_pct: 0.5}
files: {event_days: event_days.txt, iv_history: data/iv_history.csv, trade_journal: data/trades.csv,
        log: data/bot.log, state: data/state.json}
"""


def load_config() -> dict:
    cfg = yaml.safe_load(DEFAULT_CONFIG)
    user = BASE / "groww_config.yaml"
    if user.exists():                                   # optional overrides, same keys
        for k, v in (yaml.safe_load(user.read_text()) or {}).items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def groww_client():
    """
    Auth, set as environment variables (never hard-code them):
      TOTP flow (no daily approval):  GROWW_API_KEY=<TOTP token>  GROWW_TOTP_SECRET=<TOTP secret>
      Key+secret flow (approve daily on Groww Cloud API Keys page): GROWW_API_KEY  GROWW_API_SECRET
    """
    from growwapi import GrowwAPI
    key = os.environ["GROWW_API_KEY"]
    if os.environ.get("GROWW_TOTP_SECRET"):
        import pyotp
        token = GrowwAPI.get_access_token(api_key=key, totp=pyotp.TOTP(os.environ["GROWW_TOTP_SECRET"]).now())
    else:
        token = GrowwAPI.get_access_token(api_key=key, secret=os.environ["GROWW_API_SECRET"])
    return GrowwAPI(token)


def _candles_df(resp) -> pd.DataFrame:
    rows = resp.get("candles", []) if isinstance(resp, dict) else []
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    df = pd.DataFrame([r[:5] for r in rows], columns=["ts", "open", "high", "low", "close"])
    if pd.api.types.is_numeric_dtype(df["ts"]):        # epoch seconds → IST
        df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    else:
        df["ts"] = pd.to_datetime(df["ts"])
        if df["ts"].dt.tz is not None:
            df["ts"] = df["ts"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return df.set_index("ts").astype(float).sort_index()


class GrowwFeed:
    def __init__(self, g, cfg: dict):
        self.g, self.cfg = g, cfg
        self.inst = None
        try:
            self.inst = g.get_all_instruments()
        except Exception as e:
            log.warning(f"instrument list unavailable ({e}); using config fallbacks")
        self.vix_symbol = self._find_vix_symbol()
        self._vix_warned = False
        self.INTERVAL = {1: g.CANDLE_INTERVAL_MIN_1, 5: g.CANDLE_INTERVAL_MIN_5,
                         15: g.CANDLE_INTERVAL_MIN_15, 30: g.CANDLE_INTERVAL_MIN_30}

    def _find_vix_symbol(self) -> str:
        sym = self.cfg["groww"]["vix_symbol"]
        if self.inst is not None:
            m = self.inst[(self.inst["instrument_type"] == "IDX")
                          & self.inst["trading_symbol"].astype(str).str.contains("VIX", case=False, na=False)]
            if not m.empty:
                sym = str(m["trading_symbol"].iloc[0])
        return sym

    # ── prices ──
    def _ltp(self, segment, key: str) -> float:
        return float(self.g.get_ltp(segment=segment, exchange_trading_symbols=key)[key])

    def spot(self) -> float:
        return self._ltp(self.g.SEGMENT_CASH, "NSE_NIFTY")

    def vix(self) -> float:
        try:
            return self._ltp(self.g.SEGMENT_CASH, "NSE_" + self.vix_symbol)
        except Exception as e:
            if not self._vix_warned:
                log.warning(f"India VIX not available via Groww ({e}); using ATM option IV as proxy")
                self._vix_warned = True
            spot = self.spot()
            exp = self.expiry(now_ist().date())
            _, px = self.atm_iv_inputs(exp, spot)
            atm = round(spot / self.cfg["underlying"]["strike_step"]) * self.cfg["underlying"]["strike_step"]
            T = years_to_expiry(now_ist(), exp)
            gk = self.cfg["greeks"]
            ivs = [implied_vol(p, spot, atm, T, gk["risk_free_rate"], t, gk["dividend_yield"]) for t, p in px.items()]
            ivs = [v for v in ivs if v]
            return sum(ivs) / len(ivs) * 100 if ivs else 15.0

    def option_ltp(self, symbol: str) -> float:
        return self._ltp(self.g.SEGMENT_FNO, "NSE_" + symbol)

    # ── candles ──
    def history(self, groww_symbol: str, start: datetime, end: datetime, interval) -> pd.DataFrame:
        r = self.g.get_historical_candles(exchange=self.g.EXCHANGE_NSE, segment=self.g.SEGMENT_CASH,
                                          groww_symbol=groww_symbol,
                                          start_time=start.strftime("%Y-%m-%d %H:%M:%S"),
                                          end_time=end.strftime("%Y-%m-%d %H:%M:%S"), candle_interval=interval)
        return _candles_df(r)

    def candles(self, tf_minutes: int, day: date) -> pd.DataFrame:
        """Today's COMPLETED candles (any still-forming candle is dropped)."""
        now = now_ist()
        df = self.history("NSE-NIFTY", datetime.combine(day, time(9, 15)), now, self.INTERVAL[tf_minutes])
        return df[df.index + timedelta(minutes=tf_minutes) <= now]

    def prev_close(self) -> float | None:
        today = now_ist().date()
        try:
            d = self.history("NSE-NIFTY", datetime.combine(today - timedelta(days=10), time(9, 0)),
                             now_ist(), self.g.CANDLE_INTERVAL_DAY)
            d = d[d.index.date < today]
            return float(d["close"].iloc[-1]) if not d.empty else None
        except Exception as e:
            log.warning(f"prev close unavailable ({e}); gap filter off today")
            return None

    # ── contracts ──
    def expiry(self, today: date) -> date:
        exps = set()
        for k in range(3):                                   # this month + next two (handles rollover)
            y, m = today.year + (today.month - 1 + k) // 12, (today.month - 1 + k) % 12 + 1
            r = self.g.get_expiries(exchange=self.g.EXCHANGE_NSE, underlying_symbol="NIFTY", year=y, month=m)
            exps |= {date.fromisoformat(e[:10]) for e in r.get("expiries", [])}
        ok = sorted(e for e in exps if (e - today).days >= self.cfg["greeks"]["min_dte_days"])
        return ok[0]

    def lot_size(self, expiry: date) -> int:
        if self.inst is not None:
            i = self.inst
            m = i[(i["underlying_symbol"] == "NIFTY") & (i["segment"] == "FNO")
                  & (i["instrument_type"].isin(["CE", "PE"]))]
            m = m[pd.to_datetime(m["expiry_date"], errors="coerce").dt.date == expiry]
            if not m.empty:
                return int(m["lot_size"].iloc[0])
        log.warning("lot size not found in instrument list — using config fallback")
        return int(self.cfg["underlying"]["lot_size"])

    def _option_chain(self, expiry: date) -> dict:
        r = self.g.get_option_chain(exchange=self.g.EXCHANGE_NSE, underlying="NIFTY", expiry_date=str(expiry))
        return {float(k): v for k, v in r.get("strikes", {}).items()}

    def chain(self, expiry: date, spot: float, opt_type: str) -> list[dict]:
        u = self.cfg["underlying"]
        grid = strike_grid(spot, u["strike_step"], self.cfg["greeks"]["strikes_each_side"])
        oc = self._option_chain(expiry)
        atm = round(spot / u["strike_step"]) * u["strike_step"]
        out = []
        for K in grid:
            leg = (oc.get(K) or {}).get(opt_type)
            if not leg or not leg.get("ltp"):
                continue
            row = {"symbol": leg["trading_symbol"], "strike": K, "type": opt_type, "ltp": float(leg["ltp"])}
            if abs(K - atm) <= 3 * u["strike_step"]:          # bid/ask only near ATM (rate limits)
                try:
                    q = self.g.get_quote(exchange=self.g.EXCHANGE_NSE, segment=self.g.SEGMENT_FNO,
                                         trading_symbol=leg["trading_symbol"])
                    row["bid"], row["ask"] = q.get("bid_price") or 0, q.get("offer_price") or 0
                except Exception:
                    pass
            out.append(row)
        return out

    def atm_iv_inputs(self, expiry: date, spot: float):
        step = self.cfg["underlying"]["strike_step"]
        atm = round(spot / step) * step
        leg = self._option_chain(expiry).get(float(atm), {})
        return atm, {t: float(leg[t]["ltp"]) for t in ("CE", "PE") if leg.get(t, {}).get("ltp")}


class PaperExecutor:
    def __init__(self, feed, cfg):
        self.feed, self.slip = feed, cfg["costs"]["slippage_pct"] / 100

    def buy(self, symbol, qty, ref_px):
        px = ceil_tick(ref_px * (1 + self.slip))
        log.info(f"[PAPER] BUY {symbol} qty={qty} @ {px}")
        return px

    def sell(self, symbol, qty):
        px = floor_tick(self.feed.option_ltp(symbol) * (1 - self.slip))
        log.info(f"[PAPER] SELL {symbol} qty={qty} @ {px}")
        return px


class GrowwExecutor:
    """Real Groww orders: NSE · FNO · MIS · LIMIT at LTP ± buffer."""
    FILLED, DEAD = {"EXECUTED", "COMPLETED"}, {"REJECTED", "FAILED", "CANCELLED"}

    def __init__(self, g, feed, cfg):
        self.g, self.feed, self.ex = g, feed, cfg["execution"]

    def _place(self, symbol, qty, side, price) -> str:
        g = self.g
        r = g.place_order(trading_symbol=symbol, quantity=qty, validity=g.VALIDITY_DAY, exchange=g.EXCHANGE_NSE,
                          segment=g.SEGMENT_FNO, product=g.PRODUCT_MIS, order_type=g.ORDER_TYPE_LIMIT,
                          transaction_type=side, price=price)
        return r["groww_order_id"]

    def _detail(self, oid) -> dict:
        return self.g.get_order_detail(groww_order_id=oid, segment=self.g.SEGMENT_FNO)

    def _wait(self, oid) -> dict:
        t0 = _time.time()
        while _time.time() - t0 < self.ex["fill_timeout_sec"]:
            d = self._detail(oid)
            if d.get("order_status") in self.FILLED | self.DEAD:
                return d
            _time.sleep(1)
        try:
            self.g.cancel_order(segment=self.g.SEGMENT_FNO, groww_order_id=oid)
        except Exception as e:
            log.warning(f"cancel failed: {e}")
        _time.sleep(1.5)
        return self._detail(oid)

    def buy(self, symbol, qty, ref_px):
        price = ceil_tick(ref_px * (1 + self.ex["limit_buffer_pct"] / 100))
        d = self._wait(self._place(symbol, qty, self.g.TRANSACTION_TYPE_BUY, price))
        filled = int(d.get("filled_quantity") or 0)
        if filled == qty:
            avg = float(d.get("average_fill_price") or price)
            log.info(f"[LIVE] BUY {symbol} qty={qty} filled @ {avg}")
            return avg
        if filled:
            notify(f"⚠ PARTIAL BUY {filled}/{qty} {symbol} — bot is NOT managing it. Exit manually on Groww.")
        log.warning(f"[LIVE] BUY {symbol} not filled ({d.get('order_status')}); trade skipped")
        return None

    def sell(self, symbol, qty):
        remaining, value = qty, 0.0
        for attempt in range(6):
            ltp = self.feed.option_ltp(symbol)
            price = floor_tick(ltp * (1 - self.ex["limit_buffer_pct"] / 100 * (attempt + 1)))
            d = self._wait(self._place(symbol, remaining, self.g.TRANSACTION_TYPE_SELL, price))
            f = int(d.get("filled_quantity") or 0)
            value += f * float(d.get("average_fill_price") or price)
            remaining -= f
            if remaining <= 0:
                avg = value / qty
                log.info(f"[LIVE] SELL {symbol} qty={qty} filled @ {avg:.2f}")
                return avg
        raise RuntimeError(f"Could not fully exit {symbol} ({remaining} left) — CLOSE MANUALLY ON GROWW NOW")


# ════════════════════════════════════════════════════════════════════
#  RUNNER
# ════════════════════════════════════════════════════════════════════
def notify(msg: str):
    log.info(msg)
    tok, chat = os.environ.get("TG_BOT_TOKEN"), os.environ.get("TG_CHAT_ID")
    if tok and chat:
        try:
            data = urllib.parse.urlencode({"chat_id": chat, "text": msg}).encode()
            urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/sendMessage", data, timeout=5)
        except Exception as e:
            log.warning(f"telegram failed: {e}")


def _csv_rows(p: Path) -> list[dict]:
    import csv
    return list(csv.DictReader(open(p))) if p.exists() else []


def _append_csv(p: Path, row: dict):
    import csv
    new = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def run(cfg: dict):
    F = {k: BASE / v for k, v in cfg["files"].items()}
    today = now_ist().date()
    if today.weekday() >= 5:
        log.info("Weekend — nothing to do.")
        return
    g = groww_client()
    feed = GrowwFeed(g, cfg)
    ex = GrowwExecutor(g, feed, cfg) if cfg["mode"] == "live" else PaperExecutor(feed, cfg)

    st_path = F["state"]
    state = json.loads(st_path.read_text()) if st_path.exists() else {}
    if state.get("date") != str(today):
        state = {"date": str(today), "trades": 0, "pnl": 0.0, "directions": [], "open": None}
    save = lambda: st_path.write_text(json.dumps(state, default=str, indent=2))
    notify(f"Groww ORB bot started ({cfg['mode'].upper()}) {today}")
    if state["open"]:
        notify("⚠ state.json shows an open position from earlier today — check Groww and close it manually.")

    s, o, gk = cfg["session"], cfg["orb"], cfg["greeks"]
    tf = o["confirm_tf_minutes"]
    start_at = datetime.combine(today, hhmm(s["orb_end"])) + timedelta(seconds=20)
    while now_ist() < start_at:
        _time.sleep(5)

    orb = build_opening_range(feed.candles(tf, today), cfg)
    if orb is None:
        notify("No opening-range candles — holiday or data issue. Exiting.")
        return
    spot, vix = feed.spot(), feed.vix()
    expiry = feed.expiry(today)
    lot = feed.lot_size(expiry)

    atm, px = feed.atm_iv_inputs(expiry, spot)
    T = years_to_expiry(now_ist(), expiry)
    ivs = [implied_vol(p, spot, atm, T, gk["risk_free_rate"], t, gk["dividend_yield"]) for t, p in px.items()]
    ivs = [v for v in ivs if v]
    atm_iv = sum(ivs) / len(ivs) * 100 if ivs else None
    hist = [float(r["atm_iv"]) for r in _csv_rows(F["iv_history"])]
    rank = iv_rank(atm_iv, hist, cfg) if atm_iv else None
    if atm_iv and not any(r["date"] == str(today) for r in _csv_rows(F["iv_history"])):
        _append_csv(F["iv_history"], {"date": today, "atm_iv": round(atm_iv, 2)})

    ev = F["event_days"]
    events = {l.strip() for l in ev.read_text().splitlines() if l.strip() and not l.startswith("#")} if ev.exists() else set()
    ok, why = day_filters(orb, feed.prev_close(), vix, str(today) in events, rank, cfg)
    notify(f"ORB {orb.low:.0f}–{orb.high:.0f} ({orb.size:.0f} pts) | VIX {vix:.1f} | ATM IV {atm_iv or 0:.1f} "
           f"rank {('%.0f' % rank) if rank is not None else 'n/a'} | expiry {expiry} lot {lot}\n"
           f"Day filter: {'PASS' if ok else 'SKIP'} — {why}")
    if not ok:
        return

    pos: Position | None = None
    last_checked = None
    no_entry_after, sq_off = hhmm(s["no_new_entries_after"]), hhmm(s["square_off"])

    while True:
        now = now_ist()
        try:
            if pos:
                spot, opt = feed.spot(), feed.option_ltp(pos.symbol)
                reason = pos.update(spot, opt, now, cfg)
                if reason:
                    exit_px = ex.sell(pos.symbol, pos.qty)
                    costs = round_trip_costs(pos.entry_px, exit_px, pos.qty, cfg)
                    pnl = (exit_px - pos.entry_px) * pos.qty - costs
                    state["pnl"] += pnl
                    state["open"] = None
                    save()
                    _append_csv(F["trade_journal"], {
                        "date": today, "symbol": pos.symbol, "dir": pos.direction, "qty": pos.qty,
                        "entry_time": pos.entry_time.strftime("%H:%M"), "entry_px": pos.entry_px,
                        "exit_time": now.strftime("%H:%M"), "exit_px": round(exit_px, 2),
                        "entry_spot": round(pos.entry_spot, 1), "exit_spot": round(spot, 1), "reason": reason,
                        "costs": round(costs, 2), "pnl": round(pnl, 2), "mode": cfg["mode"]})
                    notify(f"EXIT {pos.symbol} @ {exit_px:.2f} ({reason}) | P&L ₹{pnl:,.0f} | day ₹{state['pnl']:,.0f}")
                    pos = None

            can_enter = (pos is None and now.time() < no_entry_after and state["trades"] < o["max_trades_per_day"]
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
                            k = d.candidate
                            ref = next((r.get("ask") or r["ltp"] for r in chain if r["symbol"] == k.symbol), k.ltp)
                            fill = ex.buy(k.symbol, d.lots * lot, ref)
                            if fill:
                                pos = Position(direction, k.symbol, k.strike, expiry, d.lots * lot, fill, spot,
                                               d.stop_spot, d.target_spot, d.risk_pts,
                                               round(fill * (1 - cfg["risk"]["premium_stop_pct"] / 100), 2), now)
                                state["trades"] += 1
                                state["directions"].append(direction)
                                state["open"] = {"symbol": k.symbol, "qty": pos.qty, "entry": fill}
                                save()
                                notify(f"ENTRY {k.symbol} ×{pos.qty} @ {fill:.2f} | index stop {pos.stop_spot:.0f} "
                                       f"target {pos.target_spot:.0f} | premium stop {pos.premium_stop}")

            done = (now.time() >= no_entry_after or now.time() >= sq_off
                    or state["trades"] >= o["max_trades_per_day"] or state["pnl"] <= -cfg["risk"]["daily_loss_limit"])
            if pos is None and done:
                break
        except Exception as e:
            log.exception(f"loop error: {e}")
            if pos:
                notify(f"⚠ Error while holding {pos.symbol}: {e}. Retrying — watch the position on Groww.")
        _time.sleep(cfg["execution"]["poll_seconds"])

    notify(f"Day done. Trades {state['trades']} | P&L ₹{state['pnl']:,.0f}")


def check(cfg: dict):
    """Connectivity + data check you can run any time (prices are last-traded outside market hours)."""
    g = groww_client()
    feed = GrowwFeed(g, cfg)
    today = now_ist().date()
    spot = feed.spot()
    print(f"NIFTY spot      : {spot}")
    print(f"VIX symbol      : {feed.vix_symbol}  → {feed.vix():.2f}")
    print(f"Prev close      : {feed.prev_close()}")
    exp = feed.expiry(today)
    print(f"Expiry (≥{cfg['greeks']['min_dte_days']} DTE): {exp}   lot size: {feed.lot_size(exp)}")
    c = feed.candles(cfg["orb"]["confirm_tf_minutes"], today)
    print(f"Today's candles : {len(c)}")
    orb = OpeningRange(spot + 40, spot - 40, spot)                   # dummy range just to exercise the gate
    for t in ("CE", "PE"):
        ch = feed.chain(exp, spot, t)
        test_spot = orb.high + 6 if t == "CE" else orb.low - 6
        d = evaluate_entry(t, test_spot, now_ist(), exp, ch, feed.vix(), orb, feed.lot_size(exp), cfg)
        print(f"\n{t} chain: {len(ch)} strikes. Sample Greeks gate (dummy breakout): {d.reason}")
        for line in d.log:
            print("  " + line)


def fetch_history(cfg: dict, days: int):
    """Download NIFTY 5-min candles + VIX daily into data/ for backtest.py."""
    g = groww_client()
    feed = GrowwFeed(g, cfg)
    end = now_ist()
    start = end - timedelta(days=days)
    frames, s = [], start
    while s < end:
        e = min(s + timedelta(days=29), end)
        frames.append(feed.history("NSE-NIFTY", s, e, g.CANDLE_INTERVAL_MIN_5))
        s = e + timedelta(minutes=1)
        _time.sleep(0.3)
    n = pd.concat(frames)
    n = n[~n.index.duplicated()]
    n.index.name = "datetime"
    n[["open", "high", "low", "close"]].to_csv(DATA / "nifty_5min.csv")
    try:
        v_frames, s = [], start
        while s < end:
            e = min(s + timedelta(days=179), end)
            v_frames.append(feed.history("NSE-" + feed.vix_symbol, s, e, g.CANDLE_INTERVAL_DAY))
            s = e + timedelta(days=1)
        v = pd.concat(v_frames)
        v = v[~v.index.duplicated()]
        pd.DataFrame({"date": v.index.date, "close": v["close"].values}).to_csv(DATA / "vix_daily.csv", index=False)
        print(f"Saved {len(n)} Nifty 5-min candles and {len(v)} VIX days to {DATA}")
    except Exception as e:
        print(f"Saved {len(n)} Nifty candles. VIX history failed ({e}) — download INDIA VIX daily from NSE "
              f"and save as data/vix_daily.csv with columns date,close")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Nifty 30-min ORB option-buying bot for Groww")
    ap.add_argument("--check", action="store_true", help="test connection, symbols and the Greeks gate")
    ap.add_argument("--fetch-history", type=int, metavar="DAYS", help="download data for backtest.py")
    a = ap.parse_args()
    cfg = load_config()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(BASE / cfg["files"]["log"]), logging.StreamHandler(sys.stdout)])
    logging.Formatter.converter = lambda *a: now_ist().timetuple()          # log in IST
    if a.check:
        check(cfg)
    elif a.fetch_history:
        fetch_history(cfg, a.fetch_history)
    else:
        run(cfg)


if __name__ == "__main__":
    main()
