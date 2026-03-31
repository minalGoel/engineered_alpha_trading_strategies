# ALPHA SYNTHESIS WORKFLOW
## Based on: Finding Alphas (Tulchinsky) + Algorithmic Trading (Chan)
## Universe: NIFTY only | Instrument: Options OR Futures (no restriction)
## Output: Exactly 2 composite strategies

---

## YOUR MANDATE

Read every strategy file. Analyse what worked, what didn't, and why.
Then synthesise exactly **2 composite strategies** — no more, no less.

You are not writing code. You are not running backtests.
You are doing what a senior quant researcher does before any code is written:
**reading the evidence and thinking**.

Every strategy file is a data point. A failed strategy is not waste —
it tells you something the market doesn't reward. A passed strategy tells
you something it does. Your job is to extract that signal from 291 data points
and compress it into 2 strategies that a human can understand, backtest, and trade.

---

## THE THREE PRINCIPLES THAT GOVERN EVERYTHING YOU DO

### Principle 1 — The UnRule (Tulchinsky, Ch3)
> *"No rule is perfect. A combination of all rules may come as close to perfection as possible."*

No single strategy in the 291 is the answer. The answer is the combination.
Each strategy — including failed ones — describes a fragment of market reality.
Your job is to assemble those fragments into 2 wholes that are more complete
than any individual part.

### Principle 2 — Flip the Loser (Chan, Algorithmic Trading)
> *"If a strategy loses money consistently, simply reverse it — it may work."*

A strategy that reliably loses money pre-cost has a signal. The signal is just pointing
the wrong direction, or the entry/exit logic is inverted. Before discarding any
failed strategy, ask: **what if I flipped it?** If the losers cluster in one pattern,
the flip of that pattern is a candidate for inclusion in your composite strategy.

### Principle 3 — Seas of Alphas, Not Single Alphas (Tulchinsky, Ch15)
> *"Find as many local optima over as large an area as possible. Do not find the global optimum."*

The 2 composite strategies must each cover multiple signal types and time conditions
using if/else routing — not because complexity is good, but because no single rule
works in all market conditions. The if/else branches ARE the portfolio diversification,
baked into a single deployable strategy.

---

## BAR TIMING RULE — NON-NEGOTIABLE

**Signal computed on bar close. Entry executes at NEXT bar's OPEN.**

This is the only rule that makes Python backtests and TradingView agree.
TradingView's `strategy.entry()` fills at next bar open by default.
Every previous strategy that passed Python CV but failed TradingView violated this rule.

Write every entry condition as:
> "When [condition] is TRUE at bar close → ENTER at open of next bar."

No exceptions. Not "at close." Not "at fill." At **next bar open**.

---

## WORKFLOW — FOLLOW THESE STEPS IN ORDER

### STEP 1: READ EVERYTHING

Open every strategy file in the strategies folder.
For each file, extract and note:

**What was the core signal?**
Strip away all parameters, thresholds, and filters. What is the one-line description
of what this strategy was trying to detect? Examples:
- "Sustained buy-side volume pressure over 1 minute"
- "NIFTY price extended above VWAP with volume exhaustion"
- "Opening range breakout with volume confirmation"

**Did it work?** Define "worked" as ANY of:
- Positive expected_bps (edge)
- Win rate > 50%
- Profit factor > 1.0

If ANY of these are positive — even weakly — the strategy "partially worked."
If ALL are negative or zero — it failed.

**Why did it fail?** Assign one cause:
- **A — Cost killed it:** Signal in the right direction but transaction cost ate the edge
- **B — No signal:** Random entry, random exits, no predictive content
- **C — Backwards signal:** Consistently lost → FLIP CANDIDATE
- **D — Regime-dependent:** Works in specific conditions not filtered for
- **E — Overfit:** Too many parameters, too few trades to be statistically meaningful

Note the cause next to each strategy. Do not skip this.

---

### STEP 2: GROUP BY SIGNAL FAMILY

After reading all strategies, group them by the **underlying market mechanism**
they were trying to exploit — not by name or parameter values.

Expected families (adjust to what you actually find):

| Family | What It Detects | Direction |
|---|---|---|
| **Order Flow Imbalance** | Buy/sell volume asymmetry, bar delta, cumulative delta | Momentum |
| **VWAP Deviation** | Price extended from session anchor | Reversion |
| **Bar Imbalance Proxy** | CLV / (close−low)/(high−low) sustained signal | Momentum |
| **Opening Range** | First N-bar high/low as structural level | Breakout or Reversion |
| **Volume Exhaustion** | Buy or sell pressure peaking then reversing | Reversion |
| **Session Time Patterns** | Time-of-day effects, open/close behaviour | Structural |
| **Volatility Regime** | High vs low vol environment effects | Filter |

For each family record:
- How many strategies?
- What fraction worked?
- What fraction are Flip Candidates (type C)?
- Strongest single result from this family?

---

### STEP 3: APPLY THE FLIP TEST

For every family where > 50% of strategies failed:

**Ask: if I reversed the entry direction, would the signal have worked?**

- Type C failure (consistently negative edge) reversed = **Flip Candidate** ✓
- Type A failure (good signal, eaten by cost) reversed = useless (flipping a cost problem
  doesn't fix the cost) ✗
- Type B failure (no signal) reversed = still useless ✗

For each confirmed Flip Candidate, write:
> "Family X signal was systematically pointing the wrong direction.
> The flip — [describe what the reversed trade looks like] — is a candidate for inclusion."

Also ask the structural question Chan poses: if a strategy reliably bought tops and sold
bottoms, the flip reliably sells tops and buys bottoms. Is that a known pattern?
Does it have a market logic explanation? If yes, the flip is valid. If you can only
explain it statistically and not logically, treat it with caution.

---

### STEP 4: FIND WHAT CONSISTENTLY WORKED ACROSS CONDITIONS

From all partial and full successes, find signal components that appeared in
**multiple independent strategy designs** — different approaches, different parameters,
different instruments — and still showed positive edge.

Per Tulchinsky: *"We trust signals more when they are less sensitive to input changes."*

A signal in 5 different strategy designs that consistently earns is real.
A signal in 1 strategy with a specific parameter that makes it work is likely overfit.

Write down explicitly: **"These signal atoms I trust. They appeared consistently."**

This list — typically 3 to 6 atoms — becomes the raw material for synthesis.

---

### STEP 5: MAP THE TAP GRID

Triple-Axis Plan (Tulchinsky, Ch11): map what you have vs what is missing.

**Your three axes for NIFTY intraday:**
- **Axis 1 — Signal Type:** Momentum | Reversion | Structural | Hybrid
- **Axis 2 — Time Horizon:** Ultra-short (<90s) | Short (90s–15min) | Medium (15–90min)
- **Axis 3 — Market Condition:** Trending day | Range day | High-vol | Low-vol | Open | Close

For each strategy, assign a TAP cell: e.g. Momentum × Ultra-short × Trending.

Then identify:
- **Overcrowded cells:** Many strategies clustered here = high internal correlation,
  all variants of the same idea, diminishing marginal value. This is where the 291 wasted effort.
- **Empty cells:** Unexplored territory — or the opposite of what was tried.

Your 2 composite strategies must together cover MORE cells than any single strategy did.
The if/else routing inside each composite strategy is how you cover multiple cells
in one deployable unit.

---

### STEP 6: DESIGN THE 2 COMPOSITE STRATEGIES

Now synthesise. You have:
- Signal families + which worked + which are flip candidates
- The trusted signal atoms (appeared in multiple independent strategies)
- The TAP coverage map (overcrowded and empty cells)
- Chan's flip insights

Design **exactly 2 strategies**. Each is a composite with if/else routing.
Think of each as a decision tree that routes to different signal logic depending
on market conditions — not a single rule applied uniformly.

#### Hard Design Constraints:

1. **≤ 5 tunable parameters per strategy.** Every parameter beyond 5 fits noise,
   not signal. (Tulchinsky Ch9 — overfitting risk grows exponentially with parameters
   and shrinks dataset)

2. **Each strategy must have a 2-sentence story.** Market logic. Not statistics.
   If you cannot explain why it works in 2 sentences using words a trader understands
   — not IC, not z-score, not Sharpe — it is not a strategy. Redesign it.

3. **Each strategy must cover ≥ 2 TAP cells through if/else routing.**
   A composite that only works in one condition is not composite.

4. **Entry at next bar open. Always.** (See Bar Timing Rule above.)

5. **At least one branch in each strategy must use a Flip Candidate.**
   Chan's principle must be present. A formerly-failing signal, inverted,
   included deliberately because the flip has a logical explanation.

6. **Strategy 1 and Strategy 2 must be structurally different.**
   Different primary signal family. If both are predominantly momentum-based,
   one is redundant. If they would trade the same direction >60% of the time,
   redesign one of them.

---

### STEP 7: WRITE THE FINAL STRATEGY SPECIFICATIONS

For each strategy, produce a full specification in this format:

---

#### STRATEGY [N]: [NAME]

**2-Sentence Story:**
[Why does this work? What market mechanism does it exploit?
No statistics. Pure market logic. A trader who has never heard of z-scores
should be able to nod and say "yes, that makes sense."]

**Signal Atoms Used (traceable to the 291):**
- Primary: [atom name — which strategies showed this working, and how]
- Confirmation: [atom name — same]
- Regime filter: [atom name — same]
- Flip component: [which failed strategy was inverted, what the flip represents]

**TAP Cells Covered:**
| Branch | Condition | Signal Type | Horizon |
|---|---|---|---|
| A | [e.g. Trending, VWAP-directional] | [e.g. Momentum] | [e.g. Ultra-short] |
| B | [e.g. Range-bound, near VWAP] | [e.g. Reversion] | [e.g. Short] |

**Entry Logic (plain English, bar-timing explicit):**

```
IF [regime condition — e.g. price trending above VWAP]:
    IF [momentum signal fires at bar close]:
        → ENTER LONG/SHORT at OPEN of next bar
    ELSE:
        → No trade

ELSE IF [range condition — e.g. price oscillating around VWAP]:
    IF [reversion setup at bar close — e.g. exhaustion signal]:
        → ENTER LONG/SHORT at OPEN of next bar
    ELSE:
        → No trade

ELSE:
    → No trade (undefined regime — sit out)
```

**Exit Logic:**
- Profit target: [N option points OR N NIFTY spot points OR specific level]
- Hard stop: [N option points]
- Time stop: [N bars × 5 seconds = N seconds]
- Signal reversal exit: [if signal flips direction before target → exit immediately]
- EOD: Flatten at 15:20 IST unconditionally

**Instrument:**
NIFTY [options (CE/PE) | near-month futures | either depending on branch]

**Parameters (list all, max 5):**
| # | Parameter | Value | What it controls |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |

**The Flip Component Explained:**
[Name the failed strategy from the 291. Describe what it was doing.
Explain what the flip means in market terms — not in backtest terms.
Why does the inverted version have a logical reason to work?]

**Where This Strategy Must NOT Trade:**
- [Specific VIX level]
- [Specific time window — e.g. first 10 minutes, last 10 minutes]
- [Specific events — e.g. expiry day, RBI day, budget day]
- [Specific market conditions — e.g. gap-open days > 0.5%]

**What Makes This Different From the Failed Versions:**
[Name the closest failed strategy or family from the 291.
Describe specifically — not "better parameters" but structural differences:
different signal combination, different regime routing, different exit logic,
flipped component, different instrument choice.]

**Kill Condition:**
[Specific, pre-defined. E.g.: "If 30-day rolling win rate drops below 40% OR
net edge drops below −5 bps per trade over 20 trades → pause immediately.
Do not optimize. Investigate first."]

**TradingView Implementation Notes:**
- Signal bar: computed at close of bar[0]
- Entry bar: open of bar[1] — use `strategy.entry()` default behaviour
- [Any Pine-specific notes: e.g. VWAP must use `ta.vwap` anchored to session,
  volume must be confirmed non-zero before signal, etc.]

---

### STEP 8: CROSS-CHECK THE PAIR

Before finalising, verify both strategies together satisfy:

**Diversity check:**
- [ ] Different primary signal atoms?
- [ ] Different dominant time horizons or conditions?
- [ ] If both fired simultaneously, would they trade the same direction? (If YES → problem)
- [ ] Different instrument preference (options vs futures vs either)?

**Coverage check:**
- [ ] Between the two, do they cover both momentum AND reversion?
- [ ] Between the two, do they cover both short-horizon AND medium-horizon conditions?
- [ ] Does at least one have a structural anchor (VWAP, opening range, session level)?

**Flip check:**
- [ ] Does each strategy contain a Chan-flipped component?
- [ ] Can you explain in market logic — not statistics — why each flip makes sense?

**Story check:**
- [ ] Read both 2-sentence stories aloud. Do they sound like market logic?
- [ ] "When institutions complete large orders, the final burst of volume exhausts
  near-side liquidity and price snaps back" = market logic ✓
- [ ] "When the 12-bar EMA z-score of CLV exceeds 1.5 sigma" = statistics, not a story ✗

If any check fails → revise before finalising.

---

### STEP 9: PRESENT TO USER IN THIS ORDER

**Part 1 — What You Found (show your reasoning)**

Present your findings from the 291 strategies BEFORE presenting the strategies.
The user must see the reasoning before the output. A strategy without visible
reasoning is not trustworthy.

Structure this as:
- Signal families found and their performance summary
- Which families consistently worked (partially or fully)
- Which families consistently failed, and the failure cause distribution
- The Flip Candidates identified and what their inversions suggest
- The 3–6 signal atoms you trust (appeared in multiple independent strategies)
- The TAP coverage map: what was overcrowded, what was empty, what was never tried

**Part 2 — Strategy 1** (full spec per Step 7 format)

**Part 3 — Strategy 2** (full spec per Step 7 format)

**Part 4 — What's Needed Before Backtesting**
- Data fields required (timeframe, session handling, specific fields)
- Any open questions needing user input before backtest design is finalised
- TradingView-specific notes for the user's verification pass

---

## WHAT YOU ARE EXPLICITLY NOT DOING

- Not writing code
- Not running backtests
- Not producing more than 2 strategies (if you feel compelled to produce 3, compress into 2)
- Not producing a "shortlist" or "candidates" — exactly 2 final strategies
- Not producing parameter variations or sensitivity sweeps
- Not preserving the original 291 strategy structures — you are synthesising, not selecting
- Not treating 12-fold CV results as validated — many are overfit noise

---

## ON COMPLEXITY VS PARAMETERS

The if/else routing inside each composite strategy is NOT additional tunable parameters.
Branching logic is structural. "IF trending THEN use momentum signal ELSE use reversion"
has zero tunable parameters — it is a conditional route.

Parameters are thresholds: the z-score cutoff, the VWAP deviation percentage,
the stop size in points. These are the things that can be overfit.
Keep them to 5 or fewer, total, across the entire strategy including all branches.

Branching = robustness (covers more conditions).
Parameters = fragility risk (fits historical noise).

The goal is high branching complexity with low parameter count.
This is exactly what "a combination of rules" means in Tulchinsky's framework.

---

## STARTING INSTRUCTION

Begin by reading all strategy files. Read every single one.
Do not write any strategy specification until you have read everything.

When you have finished reading, begin with Part 1 of Step 9 —
tell the user what you found in the 291 before presenting anything else.

The evidence comes first. The synthesis comes second.


---

## APPENDIX: REFERENCE IMPLEMENTATION (STRUCTURAL ONLY)

One strategy from the existing portfolio has been validated end-to-end in TradingView
and produced real results. It is included here NOT as a signal to replicate — the
ORB signal is already represented in the 291 — but as a structural reference for
what a correctly-implemented Pine strategy looks like.

Study this for the following implementation patterns ONLY:

**1. Session handling**
New session detection via `ta.change(time("D"))`. All session variables reset on
new session. IST timezone explicitly set on the chart, not in code.

**2. OR period definition**
Hard-coded to 09:15–09:30 using hour/minute checks — not bar counts, not offsets.
OR is complete at exactly 09:30. OR high/low tracked with running max/min.

**3. Confirmation bar requirement**
Rather than acting on first close beyond OR, the strategy counts consecutive closes
beyond OR (`barsAboveOR >= 2`). This is the filter that prevented same-bar execution
from corrupting the signal. The 2 composite strategies should use an equivalent
confirmation pattern — the exact mechanism can differ, but some form of confirmation
that prevents acting on the first bar of a move is required.

**4. Invalidation logic**
A long trade is invalidated if price closes back below OR low after entry. This is a
structural exit that is NOT the stop-loss — it is a signal-reversal exit. Both new
strategies must have an equivalent: if the signal that triggered entry subsequently
inverts, exit regardless of whether stop or target has been hit.

**5. Gap filter as regime filter — not as signal**
The gap type (GAP_UP / GAP_DOWN / FLAT) is used to filter DIRECTION, not as a trading
signal itself. A gap up aligns with longs, a gap down with shorts, flat allows both.
This is the correct use of gap data as a regime filter. Neither new strategy should
treat gap direction as a signal — only as a filter that confirms or blocks a signal
from another source.

**6. Dead zone exclusion**
OR ranges between 55–85 pts are excluded. These ranges historically produce breakouts
that reverse — the OR is "real enough" to trigger breakout algos but "narrow enough"
that it lacks conviction. This concept — excluding the ambiguous middle of a
distribution — should be considered in the new strategies. What is the equivalent
"dead zone" for the signal components being combined?

**Bar timing note:**
This strategy uses `process_orders_on_close=true`. The 2-bar confirmation filter
compensates by ensuring no entry fires on the first bar of a move. The 2 new composite
strategies must use `process_orders_on_close=false` (default) with entry at next bar
open — do not replicate the `process_orders_on_close=true` setting unless the
confirmation filter is structurally equivalent to a 1-bar delay.

**Do not replicate the ORB signal itself.** The ORB family is already represented
in the 291 strategies. What the synthesis must add is the COMBINATION of ORB-type
structural levels with the other signal families (order flow, VWAP deviation,
volume exhaustion) that the 291 explored but never combined correctly.

