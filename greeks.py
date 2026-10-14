"""Black-Scholes pricing, Greeks and implied volatility for European index options (Nifty)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time

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
