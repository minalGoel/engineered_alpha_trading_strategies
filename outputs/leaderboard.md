# Strategy Leaderboard

**Total strategies:** 358 | **Validated:** 10 | **Significance threshold Sharpe:** 0.91

## Validated Strategies (ranked by Test Sharpe × √passing_stocks)

| Rank | Strategy | Family | Score | Test Sharpe | Train Sharpe | Stocks | Trades | PF | PnL | Sig? |
|------|----------|--------|-------|-------------|--------------|--------|--------|----|-----|------|
| 1 | vwap_mean_reversion_v12 | vwap mean_reversion intraday | 130.22 | 13.22 | 20.65 | 97 | 214846 | 1.16 | ₹2,817,772 | ✓ |
| 2 | vwap_mean_reversion_v19 | vwap mean_reversion intraday | 115.74 | 11.29 | 16.91 | 105 | 199614 | 1.15 | ₹2,884,553 | ✓ |
| 3 | stochastic_reversion_v1 | None | 52.14 | 5.44 | 9.49 | 92 | 5800 | 1.24 | ₹72,873 | ✓ |
| 4 | closing_auction_anticipation_v1 | time_of_day microstructure closing_auction | 41.53 | 3.33 | 5.97 | 156 | 29489 | 1.23 | ₹681,445 | ✓ |
| 5 | rsi_connors_pullback_v49 | mean_reversion pullback oversold | 22.33 | 4.08 | 6.97 | 30 | 71403 | 1.10 | ₹1,696,093 | ✓ |
| 6 | rsi_connors_pullback_v42 | mean_reversion pullback oversold | 18.56 | 4.64 | 3.85 | 16 | 30973 | 1.15 | ₹1,506,250 | ✓ |
| 7 | rsi_connors_pullback_v46 | mean_reversion pullback oversold | 14.05 | 1.83 | 2.90 | 59 | 33804 | 1.11 | ₹1,688,790 | ✓ |
| 8 | rsi_connors_pullback_v54 | mean_reversion pullback oversold | 13.41 | 3.72 | 4.23 | 13 | 30185 | 1.11 | ₹892,341 | ✓ |
| 9 | rsi_connors_pullback_v50 | mean_reversion pullback oversold | 11.85 | 1.61 | 2.88 | 54 | 27993 | 1.10 | ₹1,297,951 | ✓ |
| 10 | rsi_connors_pullback_v41 | mean_reversion pullback oversold | 11.31 | 1.55 | 2.91 | 53 | 30179 | 1.09 | ₹1,223,791 | ✓ |

> **Note on Sharpe values:** Absolute Sharpe ratios use per-trade capital (₹100K) as denominator, not total deployed capital (~50 stocks × ₹100K). This inflates absolute values but relative comparisons between strategies are valid.

## Failure Breakdown

- **FAILED_PARSE_ERROR:** 277
- **FAILED_NO_TRADES:** 48
- **FAILED_OVERFIT:** 20
- **FAILED_OOS:** 2
- **FAILED_TIMEOUT:** 1
