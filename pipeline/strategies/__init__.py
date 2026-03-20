"""Hand-implemented strategy signal generators for 5-second index option trading.

Each strategy file exports a `Strategy` class that inherits from `BaseStrategy`
and implements the `compute()` method to produce OptionSignals from spot, option,
and VIX DataFrames.

Qualified strategies (post-triage):
- implied_vol_dislocation_v1 — IV-RV spread mean reversion
- implied_vs_realized_vol_v1 — Variance risk premium
- intraday_vol_pattern_v1 — U-shaped vol exploitation
- vol_surface_signal_v1 — Put-call skew signals
- nifty_banknifty_spread_v1 — Index spread mean reversion
- expiry_day_pattern_v1 — Gamma hedging / pin risk
"""
