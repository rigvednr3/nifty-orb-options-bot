"""
Market data + execution adapters for Zerodha Kite Connect.

  KiteFeed       – spot, VIX, intraday candles, option chain, expiries, lot size
  PaperExecutor  – simulated fills on real quotes (mode: paper)
  KiteExecutor   – real LIMIT orders, MIS product (mode: live)

Kite Connect needs a paid API subscription with historical data access (for intraday candles)
and a fresh access token every day (run kite_login.py each morning).
To use another broker (Dhan, Upstox, Angel), write a class with the same methods.
"""
from __future__ import annotations

import logging
import os
import time as _time
from datetime import date, datetime, timedelta

import pandas as pd

from strategy import ceil_tick, floor_tick, strike_grid

log = logging.getLogger("bot")

NIFTY_KEY, VIX_KEY = "NSE:NIFTY 50", "NSE:INDIA VIX"
NIFTY_TOKEN = 256265


def kite_client():
    from kiteconnect import KiteConnect
    api_key = os.environ["KITE_API_KEY"]
    token_file = os.environ.get("KITE_TOKEN_FILE", ".kite_token")
    with open(token_file) as f:
        access_token = f.read().strip()
    k = KiteConnect(api_key=api_key)
    k.set_access_token(access_token)
    return k


class KiteFeed:
    def __init__(self, kite, cfg: dict):
        self.k, self.cfg = kite, cfg
        inst = pd.DataFrame(kite.instruments("NFO"))
        self.opts = inst[(inst["name"] == "NIFTY") & (inst["segment"] == "NFO-OPT")].copy()
        self.opts["expiry"] = pd.to_datetime(self.opts["expiry"]).dt.date

    def spot(self) -> float:
        return float(self.k.ltp(NIFTY_KEY)[NIFTY_KEY]["last_price"])

    def vix(self) -> float:
        return float(self.k.ltp(VIX_KEY)[VIX_KEY]["last_price"])

    def prev_close(self) -> float:
        return float(self.k.ohlc(NIFTY_KEY)[NIFTY_KEY]["ohlc"]["close"])

    def candles(self, tf_minutes: int, day: date) -> pd.DataFrame:
        """Today's COMPLETED candles only (Kite returns the forming candle too — we drop it)."""
        interval = "minute" if tf_minutes == 1 else f"{tf_minutes}minute"
        rows = self.k.historical_data(NIFTY_TOKEN, datetime.combine(day, datetime.min.time()),
                                      datetime.now(), interval)
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.set_index("date")
        return df[df.index + timedelta(minutes=tf_minutes) <= datetime.now()]

    def expiry(self, today: date) -> date:
        min_dte = self.cfg["greeks"]["min_dte_days"]
        exps = sorted(e for e in self.opts["expiry"].unique() if (e - today).days >= min_dte)
        return exps[0]

    def lot_size(self, expiry: date) -> int:
        return int(self.opts[self.opts["expiry"] == expiry]["lot_size"].iloc[0])

    def chain(self, expiry: date, spot: float, opt_type: str) -> list[dict]:
        u = self.cfg["underlying"]
        strikes = strike_grid(spot, u["strike_step"], self.cfg["greeks"]["strikes_each_side"])
        sub = self.opts[(self.opts["expiry"] == expiry) & (self.opts["strike"].isin(strikes))
                        & (self.opts["instrument_type"] == opt_type)]
        keys = ["NFO:" + s for s in sub["tradingsymbol"]]
        quotes = self.k.quote(keys) if keys else {}
        out = []
        for _, r in sub.iterrows():
            qd = quotes.get("NFO:" + r["tradingsymbol"])
            if not qd:
                continue
            depth = qd.get("depth", {})
            bid = depth.get("buy", [{}])[0].get("price") or 0
            ask = depth.get("sell", [{}])[0].get("price") or 0
            out.append({"symbol": r["tradingsymbol"], "strike": float(r["strike"]), "type": opt_type,
                        "ltp": float(qd["last_price"]), "bid": bid, "ask": ask})
        return out

    def option_ltp(self, symbol: str) -> float:
        key = "NFO:" + symbol
        return float(self.k.ltp(key)[key]["last_price"])

    def atm_iv_inputs(self, expiry: date, spot: float):
        """ATM CE and PE LTPs, used to log daily ATM IV for the IV-rank gate."""
        atm = round(spot / self.cfg["underlying"]["strike_step"]) * self.cfg["underlying"]["strike_step"]
        res = {}
        for t in ("CE", "PE"):
            row = self.opts[(self.opts["expiry"] == expiry) & (self.opts["strike"] == atm)
                            & (self.opts["instrument_type"] == t)]
            if not row.empty:
                res[t] = self.option_ltp(row["tradingsymbol"].iloc[0])
        return atm, res


class PaperExecutor:
    """Fills at the ask (buy) / bid (sell) if known, else LTP ± slippage."""

    def __init__(self, feed: KiteFeed, cfg: dict):
        self.feed, self.slip = feed, cfg["costs"]["slippage_pct"] / 100

    def buy(self, symbol: str, qty: int, ref_px: float) -> float | None:
        px = ceil_tick(ref_px * (1 + self.slip))
        log.info(f"[PAPER] BUY {symbol} qty={qty} @ {px}")
        return px

    def sell(self, symbol: str, qty: int) -> float:
        px = floor_tick(self.feed.option_ltp(symbol) * (1 - self.slip))
        log.info(f"[PAPER] SELL {symbol} qty={qty} @ {px}")
        return px


class KiteExecutor:
    """Real orders. LIMIT at LTP ± buffer so you never get a wild market fill on an illiquid strike."""

    def __init__(self, kite, feed: KiteFeed, cfg: dict):
        self.k, self.feed, self.ex = kite, feed, cfg["execution"]

    def _place(self, symbol: str, qty: int, side: str, price: float) -> str:
        k = self.k
        return k.place_order(variety=k.VARIETY_REGULAR, exchange=k.EXCHANGE_NFO, tradingsymbol=symbol,
                             transaction_type=side, quantity=qty, product=k.PRODUCT_MIS,
                             order_type=k.ORDER_TYPE_LIMIT, price=price, validity=k.VALIDITY_DAY)

    def _wait(self, order_id: str) -> tuple[str, float]:
        t0 = _time.time()
        while _time.time() - t0 < self.ex["fill_timeout_sec"]:
            last = self.k.order_history(order_id)[-1]
            if last["status"] in ("COMPLETE", "REJECTED", "CANCELLED"):
                return last["status"], float(last.get("average_price") or 0)
            _time.sleep(1)
        return "OPEN", 0.0

    def buy(self, symbol: str, qty: int, ref_px: float) -> float | None:
        price = ceil_tick(ref_px * (1 + self.ex["limit_buffer_pct"] / 100))
        oid = self._place(symbol, qty, self.k.TRANSACTION_TYPE_BUY, price)
        status, avg = self._wait(oid)
        if status == "COMPLETE":
            log.info(f"[LIVE] BUY {symbol} qty={qty} filled @ {avg}")
            return avg
        if status == "OPEN":
            self.k.cancel_order(self.k.VARIETY_REGULAR, oid)
            # A partial fill can exist after cancel — check and keep only what filled.
            filled = int(self.k.order_history(oid)[-1].get("filled_quantity") or 0)
            if filled:
                log.warning(f"[LIVE] partial fill {filled}/{qty} — manage manually")
        log.warning(f"[LIVE] BUY {symbol} not filled ({status}); skipping trade")
        return None

    def sell(self, symbol: str, qty: int) -> float:
        """Exits must happen: re-price wider on each retry."""
        remaining, value = qty, 0.0
        for attempt in range(6):
            ltp = self.feed.option_ltp(symbol)
            price = floor_tick(ltp * (1 - self.ex["limit_buffer_pct"] / 100 * (attempt + 1)))
            oid = self._place(symbol, remaining, self.k.TRANSACTION_TYPE_SELL, price)
            status, avg = self._wait(oid)
            if status == "OPEN":
                self.k.cancel_order(self.k.VARIETY_REGULAR, oid)
                _time.sleep(1)
            last = self.k.order_history(oid)[-1]
            filled = int(last.get("filled_quantity") or 0)
            value += filled * float(last.get("average_price") or 0)
            remaining -= filled
            if remaining <= 0:
                avg_px = value / qty
                log.info(f"[LIVE] SELL {symbol} qty={qty} filled @ {avg_px:.2f}")
                return avg_px
        raise RuntimeError(f"Could not fully exit {symbol} ({remaining} left) — CLOSE MANUALLY NOW")
