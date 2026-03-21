# Evidence Base — Strategy Evidence Ranked

## Last Updated: 2026-03-21
## Sources: Claude Deep Research, Gemini Deep Research, Project History

---

## Evidence Strength Rating

| Strategy | Rating | India-Specific Evidence | Key Sources | Post-Publication Decay |
|---|---|---|---|---|
| **Overnight gap/premium** | **STRONG** | 25yr Zerodha dataset, structural explanation (pre-open auction 2010) | Zerodha Substack 2000-2025; Haghani | Unknown — not widely exploited in India |
| **Cross-sectional momentum** | **STRONG** | IIM-A 26yr dataset, live PMS track records | Agarwalla-Jacob-Varma 1994-2025; S&P DJI 2005-2022; Capitalmind PMS 5yr live | -31% drawdown Sep24-Mar25 is live decay evidence |
| **Quality factor (QMJ)** | **STRONG** | 26yr IIM-A dataset, >10% four-factor alpha | Jacob-Pradeep-Varma 2022 (IIM-A WP); S&P DJI | Lower vol than momentum, shorter drawdowns |
| **ORB (30-min)** | **MEDIUM-HIGH** | Multiple backtests, Zerodha study | Zerodha Substack Jan22-Feb26; Wang & Gangwar 2025 SSRN; SaiMohanReddy (BankNifty failed) | BankNifty ORB killed by costs — NIFTY via option selling survives |
| **Variance risk premium** | **MEDIUM-HIGH** | VIX > realized vol structural, 3-5pp spread | India VIX history; Post-Oct-2024 regime change data | Severely compressed post-Oct-2024 (reforms + STT hike) |
| **VWAP mean reversion** | **MEDIUM** | Thin India evidence, strong global evidence | IJCRT 2025 (BankNifty, 65% acc, 53 trades); OneTradeJournal 2024-25 | Concept robust but India sample sizes small |
| **Expiry day effects** | **MEDIUM** | Pattern real, post-Oct-2024 regime is new | Max-pain convergence documented; Zerodha expiry data | Single weekly (Tue) expiry since Nov 2024 — short sample |
| **Pairs / stat arb** | **MEDIUM** | Academic evidence with caveats | Sen 2022 (arXiv); Sen et al. 2023; Bhattacharya et al. | Costs often excluded from studies; some sectors negative |
| **Earnings drift** | **LOW-MEDIUM** | Limited India-specific academic work | General evidence from US (Ball & Brown 1968+); sparse Indian replication | Unknown for India |
| **Factor: Value** | **LOW** | Decade-long droughts documented | S&P DJI India 2005-2022: worst-performing factor | Avoid |
| **Factor: Size (SMB)** | **LOW** | ~0% premium in India | IIM-A data; S&P DJI | No edge |

---

## Key Academic Sources (Indian Market)

### IIM Ahmedabad Factor Data
- **Agarwalla, Jacob & Varma (2013, updated through 2025)** — Indian Fama-French-Momentum four-factor model
- Data: 1994-2025, monthly, BSE-listed equities
- URL: faculty.iima.ac.in/iffm
- Finding: Momentum (WML) 21.9% annualized, negative skew, crash-prone. Quality (QMJ) >10% alpha, lower vol.

### S&P DJI Factor Study India (June 2022)
- Universe: BSE LargeMidCap, 2005-2022
- Ranking: Momentum > Low Volatility > Quality >> Value
- Size premium: ~0%

### Zerodha / Nithin Kamath Analysis
- NIFTY overnight returns 2000-2025: 92% of up-moves occur overnight (post-2011)
- ORB option selling study Jan 2022-Feb 2026: all years profitable, max DD ~6%
- Published on Zerodha Substack / Z-Connect

### Sen (2022, arXiv / IEEE INDICON)
- Cointegration-based pairs, 5 NSE sectors, 2018-2021
- Auto and Realty: highest returns. Banking: cointegrated but mixed returns.

### IJCRT 2025
- VWAP + OI filters on BankNifty. 65% accuracy, 1:2 R:R, 53 trades.

### Wang & Gangwar (2025, SSRN)
- Block-based intraday breakout optimization on NIFTY.

---

## Live Track Records (Verified)

| Fund/Operator | Strategy | AUM | CAGR (net) | Period | Sharpe | Max DD | Fee |
|---|---|---|---|---|---|---|---|
| Capitalmind Adaptive Momentum PMS | Quant momentum, weekly rebal | ₹950+ Cr | **30.2%** | 5yr (Mar 2019-) | ~1.5 est | ~25% est | 1% fixed, 0% perf |
| Capitalmind Surge India PMS | Fundamental | ₹575+ Cr | 24.2% | 5yr | — | — | 1% fixed |
| Marcellus CCP PMS | Quality/forensic | ₹5,659 Cr | ~15-18% | Since Dec 2018 | — | — | 0% + 20% above 8% |
| Nippon India Quant Fund (MF) | Multi-factor quant | — | 14.11% | Since Jan 2013 | — | — | MF expense ratio |
| Wright Research PMS | Multi-factor AI/ML | ₹322 Cr | Claims 90%+ outperformance | Since Aug 2023 | — | — | 1.5% or 15% perf |
| Nifty200 Momentum 30 Index | Momentum (backtested) | Index | 19.06% | 2005-2025 | — | -31.25% | N/A |

**Benchmark for us:** Capitalmind at 30.2% is the target. They run single-strategy (momentum). We run 5-7 strategies. If each strategy delivers even 15-20% individually, the uncorrelated portfolio should compound to 25-35%.

---

## What Evidence Does NOT Exist For (Gaps)

1. **India-specific mean reversion at intraday frequency** — global evidence transfers imperfectly, India microstructure is different (wider spreads, lower depth)
2. **Post-Oct-2024 option selling returns** — regime is only 5 months old, sample size too small for statistical confidence
3. **Tuesday expiry effects** — only 4 months of Tuesday weekly expiry data exists
4. **Fyers TBT data quality** — claimed <10ms but no independent verification
5. **Multi-strategy portfolio Sharpe in India** — no Indian fund publishes strategy-level decomposition; all portfolio Sharpe estimates are theoretical

## Contradictions Between Sources

| Topic | Claude Finding | Gemini Finding | Resolution |
|---|---|---|---|
| NIFTY lot size | 65 (correct) | 75 (wrong — this is FINNIFTY) | **65** confirmed |
| Min viable hold period | 60-120 seconds (compass), 5-30 min (playbook) | 5-30 min | **5-30 min** for strategies; 60-120s is theoretical minimum for latency |
| ORB on BankNifty | SaiMohanReddy — failed after costs | Not mentioned | **Avoid BankNifty ORB**, focus NIFTY |
| Option selling vs buying | Playbook: "monthly ATM options preferred over futures for <30 min" | Gemini: "futures more efficient for 2-5 day holds" | **Both correct** — options for intraday, futures for multi-day/overnight |
| Spread=0 validity | Cost model: valid for candle-to-candle backtest only | Compass: spread=0 "reverses sign of returns" | **Both correct in context** — spread=0 for backtest modeling, ₹0.30/side for live projections |
