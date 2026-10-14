"""
Generates SYNTHETIC Nifty 5-min candles + VIX so you can check the code runs end to end.
The numbers it produces say nothing about real-market profitability. Use fetch_history.py for real data.
"""
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
out = Path(__file__).parent / "data"
out.mkdir(exist_ok=True)

rows, vix_rows = [], []
px, vix = 24500.0, 14.0
d = date(2025, 10, 1)
while len(vix_rows) < 250:
    if d.weekday() < 5:
        vix = float(np.clip(vix + rng.normal(0, 0.6), 10, 26))
        sigma5 = vix / 100 / np.sqrt(252 * 75)              # 75 five-min bars per day
        trend = rng.choice([0, 1, -1], p=[0.6, 0.2, 0.2]) * rng.uniform(0.4, 1.0) / 100 / 75
        px *= 1 + rng.normal(0, 0.003)                      # overnight gap
        t = datetime.combine(d, time(9, 15))
        for _ in range(75):
            o = px
            path = o * np.cumprod(1 + rng.normal(trend / 5, sigma5 / np.sqrt(5), 5))
            px = float(path[-1])
            rows.append([t, o, max(o, path.max()), min(o, path.min()), px])
            t += timedelta(minutes=5)
        vix_rows.append([d, round(vix, 2)])
    d += timedelta(days=1)

pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close"]).round(2).to_csv(out / "nifty_5min.csv", index=False)
pd.DataFrame(vix_rows, columns=["date", "close"]).to_csv(out / "vix_daily.csv", index=False)
print(f"Synthetic data written to {out} ({len(vix_rows)} days)")
