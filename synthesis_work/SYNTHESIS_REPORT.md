# ALPHA SYNTHESIS REPORT
## 258 Strategies Read. 291 Backtested. 2 Composite Strategies Produced.

---

# PART 1 — WHAT THE 291 STRATEGIES TOLD US

## 1.1 Signal Family Performance Summary

| Family | N | Worked | Win% | Total PnL | CV Passed | Best Strategy |
|---|---|---|---|---|---|---|
| **VWAP Reversion** | 25 | 15 | **60%** | **+1,130,807** | 2 | vwap_mean_reversion_v19 (+360k) |
| **VWAP Momentum** | 8 | 4 | **50%** | -92,332 | 0 | vwap_cross_momentum_v1 (+182k) |
| **Structural Levels** | 15 | 7 | **47%** | -233,318 | 0 | fibonacci_breakout_v1 (+315k) |
| **ORB** | 22 | 10 | **45%** | -473,752 | 0 | orb_closing_range_v1 (+186k) |
| **Gap** | 22 | 9 | **41%** | -51,298 | 0 | overnight_gap_reversion_v1 (+109k) |
| **Volume/Microstructure** | 18 | 7 | 39% | -924,255 | 0 | microstructure_reversion_v1 (+273k) |
| **Trend Following** | 11 | 4 | 36% | -769,317 | **2** | adx_strong_trend_pullback (+368k) |
| **ATR/Breakout** | 12 | 4 | 33% | -1,420,089 | 1 | donchian_channel_breakout_v1 (+309k) |
| **Momentum Composite** | 12 | 4 | 33% | -1,269,680 | 0 | acceleration_momentum_v1 (+325k) |
| **Time/Session** | 10 | 3 | 30% | -885,460 | 0 | european_open (+79k) |
| **Multi-Signal Composite** | 14 | 4 | 29% | -1,833,806 | 0 | multi_factor_composite_v1 (+114k) |
| **Bollinger/Keltner** | 11 | 3 | 27% | +33,776 | 0 | bollinger_mean_reversion_v1 (+656k) |
| **EMA Crossover** | 11 | 3 | 27% | -646,796 | 1 | multi_ema_ribbon_squeeze (+143k) |
| **RSI/Oscillator** | 15 | 4 | 27% | -2,016,318 | 0 | rsi_reversal_v1 (+208k) |
| **VIX/Vol Regime** | 25 | 6 | 24% | -1,391,069 | 0 | low_vol_compression (+237k) |
| **Advanced Statistical** | 19 | 4 | 21% | -2,006,129 | 0 | fractal_dimension_v1 (+97k) |

**Dominant finding:** VWAP Reversion is the only family with aggregate positive PnL across 25 strategies. Every other family loses money in aggregate. Trend Following is the only other family with multiple CV passes (parabolic_sar, hull_ma, intraday_trend_following all passed), but its aggregate PnL is negative because the failures far outnumber the successes.

## 1.2 Failure Cause Distribution

| Cause | Count | Description |
|---|---|---|
| **D — Regime-dependent** | 119 | Works in some conditions, fails in others; no condition filter present |
| **E — Overfit** | 46 | Too few trades, too few positive folds, or driven by outlier days |
| **C — Backwards signal** | 43 | Consistently negative; FLIP CANDIDATES |
| **B — No signal** | 30 | Near-zero edge, random direction |
| **A — Cost killed** | 14 | Right direction but edge < transaction cost |
| **PASSED** | 6 | CV passed |

The single largest cause of failure is **regime-dependence** (119 of 252 failures). These strategies had signal content but no routing logic to match signal to market condition. This is exactly the problem composite strategies with if/else routing are designed to solve.

## 1.3 Flip Candidates Identified (Type C — Backwards Signal)

43 strategies consistently lost money. The flip test reveals two dominant patterns:

### Pattern 1: MOMENTUM→FADE (27 of 43 flips)

Momentum/breakout signals at 5-second resolution **reliably bought the top and sold the bottom.** They fired AFTER the move was already complete. The flip: when these signals fire, the move is EXHAUSTED, not beginning.

**Strongest examples:**
- `tick_direction_momentum_v1` (PnL -881k): 62%+ uptick dominance → actually marks exhaustion
- `order_flow_imbalance_v1` (PnL -363k): High bar delta z-score → marks end of institutional burst
- `momentum_ignition_v1` (PnL -335k): Range breakout + volume surge → marks false breakout peak
- `inside_bar_breakout_v177` (PnL -339k): Compression breakout → marks failed breakout
- `narrow_range_breakout_v1` (PnL -278k): NR7 breakout → marks noise, not signal
- `adx_trend_strength_v1` (PnL -404k): ADX >25 + DI spread → by the time ADX confirms, trend is ending
- `closing_auction_anticipation_v1` (PnL -523k): Late-day momentum continuation → should fade instead

**Market logic for the flip:** At 5-second resolution, classic momentum confirmation (EMA cross, ADX rising, tick dominance, volume surge with breakout) arrives at the TAIL END of institutional order execution. The TWAP/VWAP algo is finishing, not starting. The "breakout" is the last burst before the flow reverses.

### Pattern 2: REVERSION→CONTINUE (16 of 43 flips)

Mean-reversion signals (RSI extreme, z-score, divergence) at 5-second resolution **reliably caught falling knives.** What they identified as "exhaustion" was actually mid-move.

**Strongest examples:**
- `rsi_connors_pullback_v49` (PnL -517k, 0/12 folds): RSI dip in uptrend → the dip continues
- `rsi_divergence_vwap_v1` (PnL -379k): RSI divergence → divergence resolves by price continuing
- `connors_rsi_extreme_v1` (PnL -143k): Composite RSI extreme → momentum continues through it
- `deviation_from_ema_v1` (PnL -392k): Z-score peak from EMA → price continues away
- `support_resistance_reversal_v1` (PnL -100k, 0/12): PDH/PDL rejection → price breaks through

**Market logic for the flip:** At 5-second resolution, short-timeframe RSI and z-score "extremes" are NOT true exhaustion — they are the EARLY PHASE of institutional directional flow. What Connors RSI calls "oversold at 30-second scale" is a 30-second pause in a multi-minute TWAP order. The flow resumes after the pause.

## 1.4 The 6 Signal Atoms We Trust

These signal components appeared in **multiple independent strategy designs across different families** and consistently showed positive edge:

| # | Signal Atom | Strategies | Families | Evidence |
|---|---|---|---|---|
| **1** | **VWAP z-score deviation** (1.5-2.7σ) | 9 | 2 | All 9 VWAP z-score strategies with positive PnL. Strongest: v19 (+360k), v12 (+297k), v3 (+246k). The signal is VWAP z-score crossing a threshold; the atom is the deviation itself. |
| **2** | **Volume ratio surge** (1.3-2.0x baseline) | 5 | 5 | Appeared as confirmation in microstructure_vol_reversion, multi_timeframe_breakout, vwap_rsi_volume_hybrid, atr_breakout_volatility, orb_momentum_v29. The only signal that worked across ALL 5 families it appeared in. |
| **3** | **RSI(2-3min) at extreme** (<30 or >70) | 7 | 4 | rsi_reversal (+208k), intraday_trend_following (+193k), multi_factor_composite (+114k), vwap_mean_reversion_basic (+96k). Mid-timeframe RSI — not the ultra-short Connors RSI that failed. |
| **4** | **VWAP directional filter** (price vs VWAP side) | 34 | 14 | Present in 54% of all positive strategies. Functions as a directional regime gate, not a signal. Aligns entries with the intraday dominant side. |
| **5** | **EMA pullback in confirmed trend** (EMA(24) bounce) | 3 | 1 | Three CV-passed strategies: parabolic_sar_momentum, hull_ma_momentum, intraday_trend_following. Entry on pullback TO the EMA, not breakout THROUGH it. |
| **6** | **Bar range dynamics** (compression → expansion) | 3 | 3 | low_vol_compression (+237k), spread_dynamics (+173k), atr_breakout (+64k). Compression below baseline → expansion bar in VWAP-aligned direction. |

## 1.5 TAP Coverage Map

### Overcrowded Cells (wasted effort):

| Cell | Strategies | Win% | Aggregate PnL |
|---|---|---|---|
| Momentum × Ultra-short × All-day | 47 | 38% | -3,058k |
| Reversion × Ultra-short × All-day | 47 | 30% | -2,591k |
| Momentum × Short × All-day | 21 | 33% | -2,512k |
| Hybrid × Ultra-short × All-day | 15 | 27% | -1,138k |

**109 of 258 strategies (42%)** were "all-day" strategies with no condition filter at Ultra-short horizons. These 109 strategies collectively lost **9.3 million PnL.** This is where the 291 wasted most of their effort — applying a single rule uniformly across all market conditions.

### Winning Cells:

| Cell | Strategies | Win% | Aggregate PnL | Best |
|---|---|---|---|---|
| **Reversion × Short × All-day** | 22 | **45%** | **+813k** | vwap_mean_reversion_v19 |
| **Structural × Ultra-short × All-day** | 7 | **71%** | **+334k** | fibonacci_breakout_v1 |
| **Momentum × Short × Open** | 3 | **100%** | **+310k** | first_pullback_after_trend |
| **Momentum × Ultra-short × Open** | 2 | **100%** | **+265k** | open_high_low_v1 |
| **Momentum × Medium × Open** | 1 | **100%** | **+186k** | orb_closing_range_v1 |

### Empty Cells of Interest (unexplored territory):

- **Reversion × Short × Open**: Zero strategies tried VWAP reversion specifically in the opening period
- **Momentum × Short × High-vol**: Zero strategies tried trend following in high-vol specifically
- **Structural × any × Close**: Zero strategies used structural levels for closing trades
- **All Medium horizon cells**: Almost entirely empty — longer holds barely explored

### What the TAP Map Tells Us:

1. **Reversion at Short horizons (30-300s) is the proven edge.** Not ultra-short, not all-day — SHORT.
2. **Momentum works ONLY in the opening period** — morning momentum after structural level breaks.
3. **Condition-specific routing would have saved most of the 109 all-day failures** — the same signal that loses all-day wins in the morning.
4. **Structural levels have the best hit rate** (71%) when used as reference points, not as entry signals.

---

# PART 2 — STRATEGY 1: VWAP EXHAUSTION SNAP-BACK

## 2-Sentence Story:

When institutional order flow pushes NIFTY more than 2 standard deviations from session VWAP, the large side of the order book is temporarily exhausted — resting counter-orders and VWAP-benchmarked rebalancing algos snap price back toward fair value. We enter the reversion when volume confirms the exhaustion is complete, not while the flush is still in progress.

## Signal Atoms Used (traceable to the 291):

- **Primary — VWAP z-score deviation:** vwap_mean_reversion_v19 (+360k), v12 (+297k), v3 (+246k), v13 (+236k), vwap_zscore_bounce (+182k), v5 (+118k), v14 (+88k), v11 (+87k), v1 (+70k). Nine independent strategy designs, all positive. The most robust signal in the entire 291.
- **Confirmation — Volume ratio exhaustion:** vol_ratio indicator appeared in 5 strategies across 5 families, all positive. Entry requires the volume surge to have PEAKED (current bar volume declining from the prior bar's spike) — entering the exhaust, not the flush.
- **Regime filter — RSI(2-3min) at extreme:** 7 strategies across 4 families. RSI confirms the move has reached a reversible extreme, not just a deviation that can keep going.
- **Flip component — tick_direction_momentum / order_flow_imbalance INVERTED:** These momentum signals (PnL -881k, -363k) reliably identified the END of institutional bursts. When order_flow_imbalance z-score exceeds 1.5σ AND tick_direction dominance exceeds 60%, this is the EXHAUSTION CONFIRMATION. In the original strategies, this was an entry to follow. Flipped, it is the entry to FADE.

## TAP Cells Covered:

| Branch | Condition | Signal Type | Horizon |
|---|---|---|---|
| A | Trending day — price extended from VWAP with high volume | Reversion | Short (30-120s) |
| B | Gap day — opening gap creates initial VWAP dislocation | Reversion | Short (30-120s) |
| C | Range day — price oscillating around VWAP, fade at extremes | Reversion | Ultra-short (15-60s) |

## Entry Logic (plain English, bar-timing explicit):

```
CLASSIFY REGIME at bar close:
  gap_up   = open > prev_close * 1.003
  gap_down = open < prev_close * 0.997
  flat     = neither

BRANCH A — TRENDING DEVIATION (default):
  IF vwap_zscore < -zscore_threshold (e.g. -2.0):
    IF rsi_24 < rsi_low_threshold (e.g. 30):
      IF volume_ratio > vol_threshold (e.g. 1.5x) AND volume declining from prior bar:
        IF bar_close > bar_open (first bullish bar = exhaustion confirmed):
          IF at least confirmation_bars (2) consecutive bars with zscore < -threshold:
            → ENTER LONG (buy CE) at OPEN of next bar

  IF vwap_zscore > +zscore_threshold:
    [Mirror logic for short / buy PE]

BRANCH B — GAP DEVIATION (gap days only):
  IF gap_up AND price > vwap AND vwap_zscore > +1.5:
    IF rsi_24 > rsi_high_threshold:
      IF volume exhaustion confirmed (same pattern as Branch A):
        → ENTER SHORT (buy PE) at OPEN of next bar — fade the gap extension

  IF gap_down AND price < vwap AND vwap_zscore < -1.5:
    [Mirror for long / buy CE — fade the gap extension]

BRANCH C — RANGE DAY (low VWAP slope, price oscillating):
  IF abs(vwap_slope_60) < 0.001 (flat VWAP over 5 minutes):
    IF vwap_zscore crosses below -1.5 OR above +1.5:
      IF volume_ratio > 1.3x:
        → ENTER opposite direction at OPEN of next bar
        [Tighter target (5 pts), faster exit — range-day moves are smaller]

ELSE:
  → No trade (undefined regime — sit out)

DEAD ZONE EXCLUSION:
  IF abs(vwap_zscore) between 1.0 and 1.5 → NO TRADE
  This is the ambiguous middle — deviation is real enough to look like a signal
  but not extreme enough to be a true exhaustion. The 291 showed that moderate
  deviations (1.0-1.5σ) were unpredictable; only extreme deviations (>2.0σ)
  reliably reverted.
```

## Exit Logic:

- **Profit target:** 7 option points (Branch A/B), 5 option points (Branch C range day)
- **Hard stop:** 5 option points
- **Time stop:** 24 bars (120 seconds) — if no resolution in 2 minutes, the reversion thesis is wrong
- **Signal reversal exit:** If vwap_zscore moves FURTHER from zero after entry (i.e., the deviation deepens beyond entry level + 0.5σ) → exit immediately regardless of P&L. The exhaustion thesis has been invalidated.
- **EOD:** Flatten at 15:20 IST unconditionally

## Instrument:

NIFTY options (CE for long, PE for short) — ATM. At 7-point spot targets, ATM delta (~0.5) is required. A 1-strike OTM (delta ~0.25-0.30) would need 20+ point spot moves to hit 7 option points. ATM is correct for both strategies.

## Parameters (list all, max 5):

| # | Parameter | Value | What it controls |
|---|---|---|---|
| 1 | zscore_threshold | 2.0 | How far price must deviate from VWAP (in σ) to trigger |
| 2 | vol_ratio_threshold | 1.5 | Minimum relative volume to confirm institutional activity |
| 3 | rsi_extreme_level | 35 / 65 | RSI(36) level confirming exhaustion — verified from 7 successful strategies using RSI(24-36) at 30-38/62-70 |
| 4 | confirmation_bars | 2 | Consecutive bars at extreme before entry (anti-whipsaw) |
| 5 | target_points | 7 | Profit target in option points |

## The Flip Component Explained:

**Failed strategies:** `tick_direction_momentum_v1` (PnL -881k, 2/12 folds) and `order_flow_imbalance_v1` (PnL -363k, 3/12 folds).

**What they were doing:** Both detected strong one-sided order flow — tick_direction counted the percentage of up-closing bars (>62%), order_flow_imbalance measured the z-score of volume-weighted bar delta. Both entered IN THE DIRECTION of the detected flow.

**What the flip means:** These signals reliably identified the PEAK of institutional order flow, not the start. When 62%+ of bars are closing up, the buying is nearly FINISHED. When the bar delta z-score exceeds 1.5σ, the TWAP order is in its final burst. The flip: treat these signals as EXHAUSTION CONFIRMATION, not direction signals.

**Why the flip has market logic:** Institutional TWAP/VWAP algorithms execute large orders by splitting them across time. The early bars have light participation (building). The later bars have heavy participation (completing). Tick-direction dominance and order-flow z-scores PEAK at the end of execution, not the beginning. By the time momentum signals "confirm" the direction, the institutional order is done — and counter-flow (VWAP rebalancing, profit-taking, hedging) begins immediately.

**How it is used in Strategy 1:** The tick-direction and order-flow signals are incorporated NOT as entry triggers but as EXHAUSTION CONFIRMATION within the VWAP z-score entry logic. Specifically: a VWAP z-score at -2.0σ is necessary but not sufficient. The additional requirement that tick-direction dominance has peaked (>60% and now declining) or order-flow z-score has peaked (>1.5σ and now declining) confirms that the flush is ending. This replaces the raw volume confirmation with a higher-quality confirmation that has been validated by the flip test.

## Where This Strategy Must NOT Trade:

- **VIX > 28:** Extreme VIX means the standard deviation used in z-score calculation is stale — realized vol is exploding and "2σ" is no longer extreme
- **First 10 minutes (09:15-09:25):** Session VWAP is not yet anchored; z-score is meaningless with <120 bars
- **Last 10 minutes (15:10-15:20):** Closing auction volatility creates false z-score extremes
- **Expiry day (Thursday):** Gamma effects distort VWAP anchoring
- **Gap-open > 1.0%:** Large gaps (>1%) create a VWAP that is too far from any tradeable mean; reversion target is unclear

## What Makes This Different From the Failed Versions:

The closest failed strategies are `vwap_mean_reversion_v10` (PnL -51k, moderate z-score at -2.3σ with no volume confirmation) and `vwap_rsi_confluence_v1` (PnL -41k, VWAP + RSI but no volume exhaustion check).

**Structural differences:**
1. **Exhaustion confirmation, not deviation entry:** Failed versions entered when z-score crossed the threshold. Strategy 1 enters when the z-score has been AT the threshold for 2+ bars AND volume is DECLINING from the peak — confirming the flush is over, not still in progress.
2. **Flip-based confirmation layer:** The order-flow exhaustion signal is structurally absent from all 25 VWAP reversion strategies in the 291 — none of them used volume-flow direction as a confirmation. Strategy 1 adds it because the flip test proved it detects the end of institutional flow.
3. **Regime routing:** Failed versions applied one set of parameters all day. Strategy 1 routes to different thresholds for trending days (deep 2σ), gap days (moderate 1.5σ + gap filter), and range days (shallow 1.5σ + tight target).
4. **Dead zone:** Failed versions had no exclusion zone between 1.0-1.5σ. Strategy 1 explicitly sits out this ambiguous range.

## Kill Condition:

If 30-day rolling win rate drops below 38% OR net edge drops below -3 bps per trade over 30 trades → pause immediately. Do not optimize. Investigate whether VWAP anchoring assumptions have changed (e.g., new market microstructure, session time change, or structural vol regime shift).

## TradingView Implementation Notes:

- Signal bar: computed at close of bar[0]
- Entry bar: open of bar[1] — use `strategy.entry()` default behavior (`process_orders_on_close=false`)
- VWAP must use `ta.vwap` anchored to session start (reset on `ta.change(time("D"))`)
- VWAP standard deviation: session-expanding window — `ta.stdev(close - vwap, math.max(240, bar_index - session_start_bar))`. Floor at 240 bars (20 min); grows through the session. By 11:00 IST the window includes 840+ bars of session context. Do NOT cap or shrink — the wider the window, the more meaningful the 2σ threshold becomes.
- Volume ratio: `volume / ta.sma(volume, 120)` (120 bars = 10 minutes)
- RSI: `ta.rsi(close, 36)` (36 bars = 3 minutes at 5s). Verified from 7 successful strategies: RSI(24-36) with thresholds 30-38/62-70 worked. Ultra-short RSI(3-6 bars) failed consistently.
- Confirmation: count consecutive bars where zscore < -threshold using a `var int` counter
- Dead zone: explicit `if abs(zscore) > 1.0 and abs(zscore) < 1.5` → skip

---

# PART 3 — STRATEGY 2: TREND PULLBACK TO STRUCTURAL ANCHOR

## 2-Sentence Story:

On days when NIFTY establishes a clear directional trend confirmed by VWAP and gap direction, institutional TWAP algos create predictable pullback-and-resume patterns — the pullback to a structural anchor (opening range level, session EMA, prior-day high/low) is where the next wave of algorithmic buying or selling re-engages. We enter on the first confirmed bounce off the structural anchor, not on the initial breakout which is where the 291's momentum strategies consistently lost money.

## Signal Atoms Used (traceable to the 291):

- **Primary — EMA pullback in confirmed trend:** parabolic_sar_momentum_v1 (PASSED, +249k), intraday_trend_following_v1 (PASSED, +193k), hull_ma_momentum_v1 (PASSED, -166k but 9/12 folds). Three of the six CV-passed strategies used this exact atom. Entry on pullback TO the short EMA when the longer EMA confirms trend.
- **Structural anchor — ORB level / PDH-PDL / Fibonacci:** fibonacci_breakout_v1 (+315k), orb_failure_trade_v1 (+94k), orb_retest_v1 (+45k), pdh_pdl_fakeout_v141 (+143k). Structural levels provide the WHERE for the pullback target.
- **Confirmation — Bar range compression→expansion:** low_vol_compression_long_v1 (+237k), spread_dynamics_v1 (+173k). Compression during the pullback, expansion on the resume bar.
- **Flip component — momentum breakout signals INVERTED:** inside_bar_breakout_v177 (PnL -339k), narrow_range_breakout_v1 (PnL -278k), momentum_ignition_v1 (PnL -335k). These breakout signals consistently bought AFTER the move peaked. Flipped: when a compression breakout fires WITHOUT a preceding pullback to structure, it is a FALSE BREAKOUT and should be FADED, not followed.

## TAP Cells Covered:

| Branch | Condition | Signal Type | Horizon |
|---|---|---|---|
| A | Morning trend day (gap-aligned, VWAP directional) | Momentum | Short (90-300s) |
| B | Mid-session structural level retest | Structural | Ultra-short (30-90s) |

## Entry Logic (plain English, bar-timing explicit):

```
CLASSIFY REGIME at bar close:
  trending_up   = ema_120 slope > 0 AND close > vwap AND close > ema_120
  trending_down = ema_120 slope < 0 AND close < vwap AND close < ema_120
  gap_aligned   = (gap_up AND trending_up) OR (gap_down AND trending_down)
  ranging       = NOT trending_up AND NOT trending_down

BRANCH A — TREND PULLBACK (morning bias, 09:25-13:00):
  IF trending_up:
    IF close pulls back to ema_36 (within 0.08% of ema_36 from above):
      IF ema_36 is near a structural anchor (within 15 pts of ORB high, PDH, or Fib 38.2%):
        [Higher confidence — structural confluence. Two levels within 20 pts =
         one zone. Pullback target is the nearer edge; invalidation is the far edge.]
      IF bar_range contracts below 60% of 10-bar average range (pullback compression):
        IF next bar closes above ema_36 AND bar_range expands > 1.2x prior bar:
          IF confirmation: 2 consecutive bars closing above ema_36:
            → ENTER LONG (buy CE) at OPEN of next bar

  IF trending_down:
    [Mirror for short / buy PE — pullback up to ema_36 then continuation down]

  ANTI-WHIPSAW: If ema_36 has crossed ema_120 more than 3 times in the prior
  180 bars (15 minutes) → NO TRADE. The trend is not established.

  NOTE ON EMA LOOKBACKS (verified from CV-passed strategies):
    - ema_36 (3 min) = pullback anchor. intraday_trend_following used EMA(24)=2min,
      first_pullback_after_trend used EMA(36)=3min, quality_momentum used EMA(36)=3min.
      Using 36 (the midpoint) as it gives slightly more smoothing.
    - ema_120 (10 min) = trend confirmation. Both intraday_trend_following (PASSED)
      and quality_momentum (+119k) used EMA(120)=10min as the slow/trend EMA.
      The original spec's EMA(72)=6min was too short — corrected here.

BRANCH B — STRUCTURAL LEVEL BOUNCE (09:45-15:00):
  IF price touches a structural anchor (ORB high/low, PDH, PDL, yesterday VWAP):
    IF price is within 0.15% of the level:
      IF volume on the touch bar > 1.3x 10-min average:
        IF rejection: bar close is on the opposite side of the bar range from the touch
          (e.g., touched ORB low from above → close in top 40% of bar range):
          IF 2 consecutive bars confirm the rejection (both close above the level):
            → ENTER in direction of rejection at OPEN of next bar

  GAP FILTER: On gap-up days, only take LONG structural bounces.
              On gap-down days, only take SHORT structural bounces.
              On flat days, take both directions.
  [This is the gap-as-regime-filter pattern from the reference implementation]

DEAD ZONE EXCLUSION:
  If the structural level is an ORB boundary AND the ORB width < 30 NIFTY spot
  points → NO TRADE on that ORB level. Sub-30pt ORs are noise — not a meaningful
  structural boundary to trade against. No upper-bound exclusion: for a pullback-
  to-level strategy (not breakout), wide ORs are valid structural references.
  [The reference implementation's 55-85pt dead zone was calibrated for breakout
  failure — that logic does not apply to Strategy 2's pullback-to-level design.]

ELSE:
  → No trade (ranging regime without structural anchor — sit out)
```

## Exit Logic:

- **Profit target:** 8 option points (Branch A trend), 6 option points (Branch B structural bounce)
- **Hard stop:** 5 option points
- **Time stop:** 36 bars (180 seconds) for Branch A, 18 bars (90 seconds) for Branch B
- **Signal reversal exit (INVALIDATION):** If after a long entry the price closes below the structural anchor level that triggered entry → exit immediately. The anchor has broken and the pullback thesis is invalidated. This is the structural equivalent of the reference implementation's "close below OR low" invalidation.
- **EOD:** Flatten at 15:20 IST unconditionally

## Instrument:

NIFTY options (CE for long, PE for short) — ATM. Same reasoning as Strategy 1: 6-8 point spot targets require delta ~0.5 to capture efficiently.

## Parameters (list all, max 5):

| # | Parameter | Value | What it controls |
|---|---|---|---|
| 1 | ema_short | 36 (3 min) | Pullback anchor — verified from first_pullback_after_trend (+152k) and quality_momentum (+119k), both used EMA(36) |
| 2 | ema_long | 120 (10 min) | Trend confirmation — verified from intraday_trend_following (PASSED) and quality_momentum (+119k), both used EMA(120) |
| 3 | pullback_proximity | 0.08% | How close price must get to ema_short to qualify as a pullback |
| 4 | gap_threshold | 0.3% | Minimum gap size to activate gap directional filter |
| 5 | stop_points | 5 | Hard stop in option points |

## The Flip Component Explained:

**Failed strategies:** `inside_bar_breakout_v177` (PnL -339k, 3/12 folds), `narrow_range_breakout_v1` (PnL -278k, 2/12 folds), `momentum_ignition_v1` (PnL -335k, 3/12 folds).

**What they were doing:** All three detected compression (inside bars, narrow ranges, or tight ranges) and entered when price broke out. They assumed compression leads to expansion in the breakout direction.

**What the flip means:** At 5-second resolution, compression breakouts are NOISE, not signal. The "breakout" from a narrow range or inside bar at this timeframe is almost always immediately reversed. The flip: when a compression breakout fires WITHOUT the price being near a structural anchor or in a confirmed trend, it is a FALSE BREAKOUT — fade it.

**Why the flip has market logic:** At 5-second resolution, what appears as "compression" is often just a natural pause between two algorithmic order slices (TWAP time-slicing creates regular pauses). The "breakout" from this compression is the START of the next order slice — which may go in EITHER direction. There is no directional information in 5-second compression. HOWEVER, when this same compression pattern occurs AT a structural level (ORB, PDH/PDL) and IN a confirmed trend, the compression represents genuine supply-demand equilibrium at a meaningful level, and the breakout IS directional.

**How it is used in Strategy 2:** The compression→expansion pattern is used as a CONFIRMATION CONDITION in Branch A (pullback must show compression at the EMA then expansion on the resume bar), NOT as a standalone entry. Standalone compression breakouts — the exact pattern that lost money in the 291 — are explicitly excluded. The compression is only meaningful when it occurs at a structural confluence point.

## Where This Strategy Must NOT Trade:

- **VIX > 25:** High VIX destroys structural level reliability — levels get blown through
- **First 10 minutes (09:15-09:25):** ORB not yet established, EMA not yet anchored
- **Last 20 minutes (15:00-15:20):** Closing auction distorts trend structure
- **Expiry day (Thursday):** Gamma effects create false structural levels
- **Gap-open > 1.0%:** Large gaps invalidate prior-day structural levels
- **>3 EMA crosses in prior 15 minutes:** Choppy regime — anti-whipsaw gate

## What Makes This Different From the Failed Versions:

The closest failed strategies are `adx_trend_strength_v1` (PnL -404k, 3/12 — entered on ADX confirmation, not pullback), `supertrend_momentum_v1` (PnL -418k, 5/12 — entered on flip, not pullback), and `orb_momentum_v22` (PnL -183k, 2/12 — entered on ORB breakout without trend context).

**Structural differences:**
1. **Pullback entry, not breakout entry:** The single biggest difference. Every failed momentum strategy entered on the initial signal (EMA cross, ADX threshold, SuperTrend flip). Strategy 2 waits for the FIRST PULLBACK after the trend is established. This is what distinguished the 3 CV-passed trend strategies from the 8 that failed.
2. **Structural anchor requirement:** Failed ORB strategies (orb_momentum_v22, orb_vwap_confirmation) entered on ORB breakout alone. Strategy 2 requires the pullback to coincide with a structural level — the ORB level becomes a support/resistance point, not a breakout trigger.
3. **Anti-whipsaw gate:** Failed EMA crossover strategies (ema_crossover_momentum, intraday_momentum_ema_cross) had no filter for choppy conditions. Strategy 2 counts EMA crosses and refuses to trade if >3 crosses in 15 minutes.
4. **Gap as directional filter, not signal:** Failed gap strategies (gap_continuation_momentum, gap_momentum_continuation) used the gap as a signal. Strategy 2 uses it as a filter — gap direction limits which side of the trade is allowed.
5. **Dead zone for ORB width:** The reference implementation identified that ORB widths of 55-85 points are the "uncanny valley" — too wide for a tight breakout, too narrow for conviction. Strategy 2 excludes this range.

## Kill Condition:

If 30-day rolling win rate drops below 35% OR 3 consecutive losing days → pause immediately. Trend-following strategies are inherently streaky — a 3-day losing streak may indicate a regime shift from trending to range-bound market. Investigate whether NIFTY's intraday trend character has changed before resuming.

## TradingView Implementation Notes:

- Signal bar: computed at close of bar[0]
- Entry bar: open of bar[1] — use `strategy.entry()` default behavior (`process_orders_on_close=false`)
- Session reset: `ta.change(time("D"))` — reset ORB tracking, EMA counters, gap classification
- ORB detection: track high/low during 09:15-09:30 using hour/minute checks (not bar counts)
- EMA: `ta.ema(close, 36)` (3-min pullback anchor) and `ta.ema(close, 120)` (10-min trend). Verified against CV-passed strategies.
- Anti-whipsaw: `var int ema_cross_count` — increment on `ta.cross(ema36, ema120)`, reset on new session
- Gap classification: compare `open` of first bar to previous session's close
- Structural level proximity: `math.abs(close - orb_high) < 15` (in NIFTY spot points)
- Confirmation counter: `var int confirm_bars` — count consecutive bars meeting condition
- Invalidation: `if strategy.position_size > 0 and close < structural_level` → `strategy.close_all()`

---

# PART 4 — WHAT'S NEEDED BEFORE BACKTESTING

## 4.1 Data Fields Required

| Field | Source | Why |
|---|---|---|
| NIFTY spot OHLCV (5-second bars) | TBT data via Dhan/Upstox | Primary signal computation |
| Session VWAP | Computed from typical_price × volume | Strategy 1 primary signal |
| Previous day close | Derived from session boundary | Gap classification |
| Previous day high, low | Derived from session boundary | Strategy 2 structural levels |
| India VIX (5-second or 1-min) | NSE or broker feed | Strategy 1 & 2 regime gate |
| ATM option premiums (CE/PE) | Live chain or historical | Execution vehicle pricing |

## 4.2 Design Decisions Resolved

1. **VWAP standard deviation window:** Session-expanding from 240-bar floor. `stdev = ta.stdev(close - vwap, math.max(240, bar_index - session_start_bar))`. Grows with session. Do not cap.

2. **Structural level confluence:** Two levels within 20 pts = one zone. Pullback target is the nearer edge; invalidation is the far edge. Confluence is a stronger signal, not an ambiguous one.

3. **Option strike:** ATM for both strategies. At 7-point spot targets, ATM delta (~0.5) is required. OTM only makes sense for targets >15 option points.

4. **ORB dead zone:** Strategy 1: N/A (ORB not used as signal). Strategy 2: exclude ORB < 30 pts only. No upper-bound exclusion — wide ORs are valid pullback reference levels. The 55-85pt dead zone from the reference strategy was for breakout failure, not applicable to pullback-to-level design.

5. **EMA lookbacks (corrected from initial spec):** Pullback anchor = EMA(36) = 3 min. Trend confirmation = EMA(120) = 10 min. Verified from CV-passed strategies: intraday_trend_following used EMA(24)/EMA(120), quality_momentum used EMA(36)/EMA(120).

6. **RSI thresholds (corrected from initial spec):** RSI(36) at thresholds 35/65. Verified from 7 successful strategies: RSI(24-36) at 30-38/62-70 worked. Ultra-short RSI(3-6 bars) at any threshold failed. The distinction is period (2-3 min RSI = real exhaustion) not threshold depth.

## 4.3 TradingView Verification Notes

- **Bar timing verification:** After Pine implementation, verify on a chart that entries appear at the OPEN of the bar AFTER the signal bar. Plot signal conditions as shapes on signal bars; entries should appear one bar later.
- **VWAP anchoring:** Verify VWAP resets on session boundary. On gap-open days, VWAP should start from the first bar of the new session, not carry over.
- **Session boundary:** The strategies assume IST timezone. Set chart timezone explicitly to IST. Do not rely on exchange default.
- **Volume bars at 5s:** Confirm the data feed provides actual volume at 5-second resolution, not interpolated values. Zero-volume bars should be handled (skip signal computation on zero-volume bars).
- **Structural level persistence:** ORB levels should persist throughout the session after being established at 09:30. Plot them as horizontal lines to verify they remain static.

---

# STEP 8 — CROSS-CHECK THE PAIR

## Diversity check:
- [x] Different primary signal atoms? **YES** — VWAP z-score deviation (S1) vs EMA pullback in trend (S2)
- [x] Different dominant time horizons or conditions? **YES** — S1 is reversion at all times; S2 is morning-biased trend following
- [x] If both fired simultaneously, would they trade the same direction? **NO** — S1 fires when price is extended FROM VWAP (z-score extreme); S2 fires when price is pulling BACK TO an anchor in a trend. These are opposite market states. If VWAP z-score is at -2σ, the trend structure is broken and S2 would not fire. If S2 is entering on a pullback in an orderly trend, the z-score is near zero and S1 would not fire.
- [x] Different instrument preference? **BOTH** use NIFTY options but in different conditions

## Coverage check:
- [x] Between the two, do they cover both momentum AND reversion? **YES** — S1 is pure reversion; S2 is trend-following momentum
- [x] Between the two, do they cover both short-horizon AND ultra-short conditions? **YES** — S1 covers Short (30-120s) primary + Ultra-short (15-60s) on range days; S2 covers Short (90-300s) + Ultra-short (30-90s structural bounce)
- [x] Does at least one have a structural anchor? **YES** — S2 uses ORB, PDH/PDL, Fibonacci levels as primary anchors; S1 uses VWAP as the anchor

## Flip check:
- [x] Does each strategy contain a Chan-flipped component? **YES** — S1 flips tick_direction_momentum + order_flow_imbalance (momentum exhaustion confirmation); S2 flips inside_bar_breakout + narrow_range_breakout (compression breakout → false breakout filter)
- [x] Can you explain in market logic why each flip makes sense? **YES** — see flip sections above

## Story check:
- [x] S1 story: "Institutional flow exhausts at VWAP extremes; the counter-flow snaps price back." — **market logic, no statistics**
- [x] S2 story: "TWAP algos create pullback-and-resume patterns at structural levels during trend days; enter the pullback." — **market logic, no statistics**
