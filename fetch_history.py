"""
Download real history from Kite Connect for the backtester.

    python fetch_history.py --days 730

Writes data/nifty_5min.csv and data/vix_daily.csv. Kite limits 5-minute requests to ~100 days
per call, so this fetches in chunks.
"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from broker import kite_client

NIFTY_TOKEN, VIX_TOKEN = 256265, 264969
out = Path(__file__).parent / "data"
out.mkdir(exist_ok=True)


def fetch(k, token, interval, start, end, chunk_days):
    frames, s = [], start
    while s < end:
        e = min(s + timedelta(days=chunk_days), end)
        frames.append(pd.DataFrame(k.historical_data(token, s, e, interval)))
        s = e + timedelta(days=1)
    df = pd.concat(frames).drop_duplicates("date")
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=730)
    a = ap.parse_args()
    k = kite_client()
    end = datetime.now()
    start = end - timedelta(days=a.days)

    n = fetch(k, NIFTY_TOKEN, "5minute", start, end, 95)
    n.rename(columns={"date": "datetime"})[["datetime", "open", "high", "low", "close"]] \
        .to_csv(out / "nifty_5min.csv", index=False)
    v = fetch(k, VIX_TOKEN, "day", start, end, 1900)
    v["date"] = v["date"].dt.date
    v[["date", "close"]].to_csv(out / "vix_daily.csv", index=False)
    print(f"Saved {len(n)} Nifty 5-min candles and {len(v)} VIX days to {out}")


if __name__ == "__main__":
    main()
