# Nifty ORB Options Bot

An automated intraday trading system for India's Nifty 50 index options. It trades the classic
**30-minute Opening Range Breakout**, but decides *whether* to take each breakout, and *which* option
to buy, using a Black-Scholes model of the option's Greeks instead of a fixed rule.

Built by **Rigved Nimkhedkar** · Python · runs on Groww or Zerodha Kite Connect

## The idea

Most ORB bots buy the at-the-money option the moment price breaks the range. Option buyers lose
money on many of those trades even when the direction is right: time decay and falling implied
volatility eat the premium before the index move pays off.

This bot prices that cost before entering. For every strike near the money it:

1. solves the option's **implied volatility** from its live premium
2. computes **delta, gamma, theta and vega**
3. **re-prices the option** at the index target and at the index stop, after the time each usually
   takes and the IV shift Nifty typically shows (IV falls in rallies and rises in selloffs)
4. buys only if the modelled premium gain at target is at least the modelled loss at stop

So a breakout the index chart likes can still be rejected, because the option itself is a bad bet.
Every decision is logged with the full Greeks table, so each trade or skip can be explained.

## Status

- Strategy, Greeks engine, risk sizing, broker adapters and backtester are complete.
- Tested end-to-end against simulated market days and a 250-day synthetic backtest. The order paths
  for both brokers were exercised against mocked APIs.
- **Not yet validated on real market history or with real money.** The next steps are a backtest
  on 2 years of real Nifty data, then 4–6 weeks of paper trading on live quotes.
- No performance claims are made here. The backtester is built to be pessimistic (worst-case
  intrabar fills, slippage and full Indian transaction costs) so it doesn't flatter the strategy.

## Engineering highlights

- **One strategy module shared by the live bot and the backtester**, so a backtest trades the exact
  same rules that would run live
- Self-written Black-Scholes pricer, Greeks and implied-vol solver on a trading-session clock, so
  intraday theta is realistic
- Day filters for event days, gap size, range size, India VIX band and IV rank
- Risk-based position sizing, index and premium stops, breakeven-then-trailing exits, daily loss cap
- LIMIT-only execution with partial-fill handling and escalating exit retries
- Broker adapters for **Groww** (single-file deployable) and **Zerodha Kite Connect**
- Paper mode by default, IST-safe scheduling on cloud servers, optional Telegram alerts

## Quick start (Groww)

`groww_orb_bot.py` is one self-contained file (strategy + Greeks + Groww API).
```bash
pip install growwapi pyotp pandas numpy pyyaml
export GROWW_API_KEY=<TOTP token>  GROWW_TOTP_SECRET=<TOTP secret>   # from Groww Cloud → API keys
python groww_orb_bot.py --check               # verify login, symbols, option chain, Greeks gate
python groww_orb_bot.py --fetch-history 365   # real Nifty data for backtest.py
python groww_orb_bot.py                       # run today in paper mode (start before 09:45 IST)
```
Overrides go in an optional `groww_config.yaml` next to it (e.g. `mode: live`). Times are always IST,
so cron on a UTC server: `10 4 * * 1-5 cd ~/bot && python3 groww_orb_bot.py`  (= 09:40 IST).
The other files are the modular Zerodha Kite version; `backtest.py` works with data from either.

## Strategy rules

Buys a **Nifty CE on an upside breakout** or a **PE on a downside breakout** of the first 30-minute
candle (09:15–09:45), only when the option's Greeks say the trade is worth paying for.

### How a trade is decided

```
09:45  Opening range = high/low of 09:15–09:45
       │
       ├─ Day filters ── skip the whole day if:
       │     event day (event_days.txt) · gap > 1% · range < 0.25% or > 0.9% of spot
       │     India VIX outside 10–24 · ATM IV rank > 80 (top of its 60-day range)
       │
every 5-min candle close (until 13:30)
       ├─ Breakout?  close > ORB high + 5  → CE      close < ORB low − 5 → PE
       ├─ Chasing?   skip if spot already > 0.30% beyond the level
       │
       ├─ Greeks gate — for each strike ±6 around ATM, from its live price:
       │     IV (solved from premium) → Δ, Γ, Θ, Vega
       │     Δ  0.45–0.72        ATM to moderately ITM: moves with the index, less time value to bleed
       │     IV ≤ VIX + 3 pts    don't buy an overpriced strike
       │     Θ over expected 2h hold ≤ 8% of premium
       │     EDGE ≥ 1.0          option gain if index hits target ÷ option loss if index hits stop,
       │                         after decay (2h for winners, 1h for losers) and the IV change Nifty usually
       │                         shows (IV falls on rallies, rises on selloffs) — full Black-Scholes revaluation
       │     bid-ask spread ≤ 1.5% (live only)
       │   → picks the passing strike with the best edge
       │
       └─ Size: lots = risk_per_trade ÷ (modelled loss at stop × lot size), capped by max_lots & capital

Exits (first one hit):
   index stop (ORB midpoint) · premium stop −35% · index target = breakout level + 1 full range
   after +1R: stop → breakeven, then trails 1R behind the best price · 15:15 square-off
Daily: 1 trade, ₹4,000 daily loss limit. Never buys an expiry within 2 days (rolls to next weekly).
```

Every decision is logged with the full Greeks table, so you can see *why* a breakout was taken or skipped.

## Files

| file | what it does |
|---|---|
| `groww_orb_bot.py` | the complete bot for Groww in one file |
| `config.yaml` | every parameter (start in `mode: paper`) |
| `strategy.py` | ORB, filters, Greeks gate, position/exit logic — shared by bot and backtester |
| `greeks.py` | Black-Scholes price, Greeks, implied-vol solver, trading-clock time to expiry |
| `broker.py` | Zerodha Kite data feed + paper and live executors |
| `bot.py` | the live/paper runner |
| `backtest.py` | backtests the same logic on historical 5-min data |
| `fetch_history.py` | downloads Nifty 5-min + VIX history from Kite |
| `make_sample_data.py` | synthetic data, only to check the code runs |
| `kite_login.py` | daily Kite login → access token |
| `event_days.txt` | dates to sit out (RBI, Budget, Fed, results…) |

## Setup (Zerodha Kite)

```bash
pip install -r requirements.txt
export KITE_API_KEY=... KITE_API_SECRET=...
python kite_login.py          # every trading morning
```
Kite Connect needs the paid API plan (intraday candles come from the historical-data API).
Optional phone alerts: `export TG_BOT_TOKEN=... TG_CHAT_ID=...` (Telegram).

## Order of operations — don't skip steps

1. **Backtest on real data**
   `python fetch_history.py --days 730` then `python backtest.py`
   Look at profit factor, max drawdown, and how many trades survive costs. Try `confirm_tf_minutes: 15`,
   `stop_mode: opposite`, `min_edge_ratio` 1.3 vs 1.8.
2. **Paper trade for at least 4–6 weeks**: `mode: paper`, run `python bot.py` before 09:45
   (cron at 09:40 IST). Compare paper fills to what the backtest predicted.
3. **Live with 1 lot** (`max_lots: 1`) only if paper results hold up. Set `mode: live`.

## Known limitations — read these

- **Backtest premiums are modelled**, not real: Black-Scholes with yesterday's VIX as IV. Real options have
  skew, wider spreads at 09:45 and intraday IV crush. The backtest is a filter for bad ideas, not proof
  of a good one. Paper trading on real quotes is the real test.
- **IV rank needs history.** The bot logs ATM IV daily to `data/iv_history.csv`; the IV-rank gate
  switches on after 20 days. Until then only the VIX band protects you.
- **Restarts:** if the bot crashes while holding, it alerts you but does not resume managing that
  position. Keep the Kite app open on trading days and know how to exit manually.
- **Verify** lot size, STT rate and expiry day in `config.yaml` against your broker — they change.
- Live mode uses LIMIT orders; an entry that doesn't fill in 8s is cancelled and the trade skipped.

## Disclaimer

Research and engineering project, not investment advice. Most individual F&O traders in India lose
money (SEBI study). Use paper mode and your own judgement.
