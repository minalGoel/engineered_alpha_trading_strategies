# Indian Futures Backtesting — Rules & Cost Model

> Reference document for strategy builder context. All figures current as of April 2026.
> Instrument scope: NSE index futures (NIFTY, Bank NIFTY) and stock futures.

---

## 1. Execution model

### TradingView Pine Script

- `process_orders_on_close=true` fills orders at the close of the signal bar. On intraday timeframes (1m–15m), the close of bar N and open of bar N+1 are near-identical in continuous trading — the look-ahead bias is negligible (~1–3 pts of execution latency, largely subsumed by slippage).
- `process_orders_on_close=false` fills at the next bar's open. This is the correct setting for any strategy where the signal depends on the bar's close price. The cost is small on intraday charts but enforces causal correctness.
- **Rule: always use `process_orders_on_close=false` for strategies that derive signals from `close`.** The performance difference may be small, but it eliminates a class of inflation that's hard to quantify.

### Signal and fill separation

- If a signal fires based on a condition evaluated at bar close (e.g., `close > level`), the fill must occur at the next bar's open or later. Filling at the same close is using information you couldn't have acted on in time.
- On sub-5-minute charts, this distinction shrinks toward zero. On 15-minute and above, it matters.
- For confirmation-based signals (N consecutive closes beyond a level), the fill occurs on the bar where the Nth close is confirmed — at that bar's close (with `process_orders_on_close=true`) or the next bar's open (with `false`). The latter is correct.

### Continuous futures (NIFTY1!, BANKNIFTY1!)

- TradingView back-adjusts continuous futures by default, removing rollover price gaps from the historical series.
- On back-adjusted data, `close[1]` on the first bar of a new session and today's `open` are on a consistent price scale. Gap calculations between sessions reflect actual overnight sentiment, not contract splices.
- If using non-adjusted continuous contracts, rollover days create phantom gaps of 50–100+ points that will corrupt any gap-based signal.
- **Rule: always use back-adjusted continuous contracts for strategies that reference prior session close or compute overnight gaps.** Document the adjustment method used.

---

## 2. Cost model — NSE futures (effective April 1, 2026)

### STT (Securities Transaction Tax)

| Instrument | Side | Rate | Notes |
|---|---|---|---|
| Equity futures | Sell only | **0.05%** of contract value | Increased from 0.02% in Budget 2026 |
| Equity options (premium) | Sell only | 0.15% of premium value | Increased from 0.10% |
| Equity options (exercise) | Exercise | 0.15% of intrinsic value | Increased from 0.125% |

STT is the dominant cost for futures trading. At NIFTY ~23,500 and lot size 65, STT per lot on sell = ₹764. This alone exceeds total brokerage + exchange fees combined.

### Full cost schedule per order (Zerodha-equivalent discount broker)

| Component | Buy side | Sell side | Scales with |
|---|---|---|---|
| Brokerage | ₹20 flat | ₹20 flat | Nothing (flat per order) |
| STT | — | 0.05% of turnover | Contract value × lots |
| Exchange txn | 0.00173% | 0.00173% | Contract value × lots |
| SEBI charges | 0.0001% | 0.0001% | Contract value × lots |
| Stamp duty | 0.002% | — | Contract value × lots (buy side only) |
| GST | 18% of (brokerage + exchange) | 18% of (brokerage + exchange) | Derived |

### Reference costs at common position sizes (NIFTY @ 23,500, lot = 65)

| Lots | Contract value | Round-trip cost | Breakeven (pts) | STT % of total |
|---|---|---|---|---|
| 1 | ₹15.28L | ~₹907 | ~14.0 | ~84% |
| 2 | ₹30.55L | ~₹1,720 | ~13.2 | ~87% |
| 5 | ₹76.38L | ~₹4,150 | ~12.8 | ~89% |
| 10 | ₹152.75L | ~₹8,210 | ~12.6 | ~90% |

Key insight: breakeven decreases slightly with more lots because flat brokerage (₹20) is amortised. But STT dominates at all sizes.

### TradingView commission settings

Use percent-based commission to correctly model STT scaling:

```
commission_type = strategy.commission.percent
commission_value = 0.028   // ~0.028% per side, ~0.056% round-trip
```

This captures STT + exchange + SEBI + stamp + GST + brokerage as a blended per-side percentage. Accurate within ±5% across 1–10 lot sizes. Slightly overcharges brokerage at large sizes (flat ₹20 modelled as percent), but the error is <1% of total cost.

**Do not use `cash_per_order` or `cash_per_contract`** — these fail to model STT's turnover-based scaling correctly.

### Cost model for stock futures

Stock futures have different lot sizes and contract values but identical fee structure. Recalculate using:

```
round_trip_cost ≈ contract_value × lots × 0.00056 + ₹40 (brokerage) + GST
```

The 0.056% round-trip approximation holds for any NSE futures contract.

---

## 3. Slippage model

### Index futures (NIFTY, Bank NIFTY)

NIFTY futures are among the top 5 most liquid index futures globally. Daily volume: 200,000+ contracts. Bid-ask spread: typically 1 tick (0.05 pts).

| Position size | Realistic slippage | TradingView setting |
|---|---|---|
| 1–5 lots | 0.05–0.10 pts | `slippage=1` to `slippage=2` |
| 5–20 lots | 0.10–0.25 pts | `slippage=2` to `slippage=5` |
| 20–50 lots | 0.25–0.50 pts | `slippage=5` to `slippage=10` |
| 50+ lots | Model separately | Use market impact function |

TradingView's `slippage` parameter is in **ticks** (minimum price movement), not points. NIFTY tick size = 0.05 pts. So `slippage=2` = 0.10 points.

**Do not conflate slippage with execution latency.** If using `process_orders_on_close=false`, the bar-to-bar price difference already captures execution timing. Additional slippage models only the bid-ask spread crossing and any market impact at your order size.

### Stock futures

Less liquid than index futures. Typical bid-ask: 1–5 ticks depending on the stock. Use `slippage=3` to `slippage=10` for liquid F&O stocks (Reliance, HDFC Bank, etc.). For less liquid names, model impact separately or add `slippage=10` to `slippage=20`.

### When slippage matters more than usual

- Breakout entries: order flow is directional, book thins. Add 1–2 extra ticks.
- Expiry day exits: massive institutional squaring-off between 14:00–15:30. Add 3–5 extra ticks for time exits near expiry.
- Pre-open/post-open (09:15–09:18): spread can be 2–5x normal. Avoid entries in this window or model wider slippage.

---

## 4. Position sizing and margin

### NSE lot sizes (as of January 2026)

| Index | Lot size | Approx contract value (at current levels) |
|---|---|---|
| NIFTY 50 | 65 | ₹15–16L |
| Bank NIFTY | 30 | ₹15–16L |
| NIFTY Next 50 | 25 | Varies |
| SENSEX (BSE) | 20 | ₹15–16L |

SEBI mandates contract values in the ₹15–20L band. Lot sizes are revised periodically to maintain this range.

### Margin types

| Type | Typical % | Use case |
|---|---|---|
| MIS (intraday) | 5–8% | Auto-squared off by 15:15–15:20 |
| NRML (overnight) | 12–15% | Carried overnight, full SPAN + exposure margin |
| CO (cover order) | 3–5% | Reduced margin with mandatory stop-loss |

For intraday strategies, use MIS margin for position sizing calculations. At ₹5L capital and 6.5% MIS margin, max NIFTY lots ≈ 5.

### Compounding and position sizing

- Compounding (scaling lots with equity growth) inflates late-period returns. **Always report both compounded and fixed-lot results.** The compounded equity curve shows capital efficiency; the fixed-lot curve isolates signal quality.
- Chan's principle: *separate signal performance from position sizing performance.* Report Sharpe, win rate, and per-trade expectancy on fixed 1-lot sizing. Then layer compounding as a separate analysis.
- With auto-lot sizing based on margin, a 20% equity gain increases lots from 5 to 6 — a discrete 20% leverage jump. This creates convex return profiles that look better in backtest than they perform live.

---

## 5. Backtest validation checklist

### Before trusting any backtest result

1. **Execution model**: Is `process_orders_on_close=false`? Does the fill price represent something achievable in live trading?
2. **Cost model**: Is commission modelled as percent of turnover (not flat per order)? Does it include STT at the current 0.05% rate?
3. **Slippage**: Is it calibrated to the instrument's liquidity and the strategy's typical order flow context?
4. **Compounding**: Have you reported fixed-lot results separately from compounded results?
5. **Data**: Are you using back-adjusted continuous futures? Have you verified that rollover days don't generate false signals?
6. **Out-of-sample**: Was the test period used for parameter selection, or was it held out? If parameters were chosen by observing backtest output, the entire result is in-sample.

### Parameter count and statistical significance

Every tunable parameter (range bounds, thresholds, time windows, multipliers, filter toggles) reduces degrees of freedom. Per Tulchinsky (Finding Alphas, Table 9.1), the minimum backtest days required for statistical significance at Sharpe 1.0 scales roughly as:

| Free parameters | Min backtest days |
|---|---|
| 3–5 | ~250 (1 year) |
| 5–10 | ~500 (2 years) |
| 10–15 | ~1,000 (4 years) |
| 15+ | ~2,500+ (10 years) |

If the strategy has 12 parameters optimised over 2 years of data, the result is likely overfit.

### Walk-forward validation

The only valid test: fix parameters on a training period, then run unchanged on a subsequent out-of-sample period with no re-optimisation. A 70/30 train/test split is standard. Report Sharpe, return, and max drawdown on the test period only.

De Prado's rule: *a backtest that uses the same data for parameter selection and performance measurement is not a backtest — it is an in-sample fit.*

### Deflated Sharpe Ratio

The standard Sharpe ratio does not account for the number of strategies/parameter combinations tested to arrive at this one, non-normality of returns (breakout strategies have fat tails), or the length of the backtest. Apply the Deflated Sharpe Ratio (de Prado, AFML Ch14) to estimate the probability that the observed Sharpe is a false positive.

Conservative rule of thumb: divide in-sample Sharpe by 2 as an out-of-sample estimate when parameters have been optimised (Chan).

### Sensitivity testing

Remove one filter or parameter at a time and re-run. Per Tulchinsky: *we trust the signal more if each variable contributes.* If removing a filter has negligible impact, it's decorative. If removing it destroys the strategy, the strategy depends on that exact parameterisation — a sign of overfitting.

---

## 6. Common backtest inflation sources (futures-specific)

| Source | Direction | Typical magnitude | How to detect |
|---|---|---|---|
| Same-bar signal + fill | Inflates | 1–5% on intraday | Toggle `process_orders_on_close` |
| Understated slippage | Inflates | 1–3% for liquid futures | Test with 2x and 5x slippage |
| Flat commission (not STT-scaled) | Inflates | 5–20% depending on trade count | Compare flat vs percent commission |
| Compounding inflating late returns | Inflates | 10–40% of total return | Compare fixed-lot vs compounded |
| Continuous contract rollover gaps | Either | 0–5% | Check signals on rollover days |
| Expiry-day time exits at clean close | Inflates | 2–5% | Flag exits within 3 days of expiry |
| Dead zone / filter overfitting | Inflates | 10–50% of Sharpe | Sensitivity test each filter |
| Insufficient out-of-sample testing | Inflates | Unknown until tested | Walk-forward split |

### The final test

After addressing all of the above, re-run the strategy with:
- `process_orders_on_close=false`
- Percent-based commission at 0.028% per side
- Slippage appropriate to position size (2–5 ticks for index futures)
- Compounding OFF (fixed 1-lot)
- Walk-forward split (train on first 70% of data, test on last 30%)

The out-of-sample Sharpe on fixed lots with realistic costs is the number that represents actual edge. Everything above that number is backtest inflation.

---

## 7. Quick reference — TradingView strategy() template

```pine
// Indian futures — realistic cost model (post April 2026 STT)
strategy("Strategy Name", 
  overlay=true, 
  initial_capital=500000, 
  default_qty_type=strategy.fixed, 
  default_qty_value=1,
  commission_type=strategy.commission.percent,
  commission_value=0.028,          // ~0.028% per side ≈ STT + exchange + GST + stamp + brokerage
  slippage=2,                      // 2 ticks = 0.10 pts (NIFTY futures, ≤5 lots)
  process_orders_on_close=false,   // next-bar-open fill
  calc_on_every_tick=false)
```

Adjust `slippage` upward for less liquid stock futures or larger position sizes. Adjust `commission_value` only if fee structure changes (STT revision, broker change).

---

## 8. Instrument-specific notes

### NIFTY 50 futures
- Lot size: 65 (from Jan 2026)
- Tick size: 0.05 pts
- Daily volume: 200,000+ contracts
- Expiry: weekly (Thursday) + monthly (last Thursday)
- Bid-ask: 1 tick in normal hours, 2–5 ticks in pre-open and near expiry close

### Bank NIFTY futures
- Lot size: 30 (from Jan 2026)
- Tick size: 0.05 pts
- Daily volume: 100,000+ contracts
- Expiry: monthly only (last Wednesday) — no weekly futures since Nov 2024 SEBI rules
- Higher intraday volatility than NIFTY; same cost model applies

### Stock futures
- Lot sizes vary by stock (SEBI ₹15–20L contract value mandate)
- Liquidity varies widely — top 20 F&O stocks are liquid, rest may have 5–20 tick spreads
- Same STT/fee structure as index futures
- Additional risk: corporate actions, bans on F&O trading when open interest exceeds MWPL limits
