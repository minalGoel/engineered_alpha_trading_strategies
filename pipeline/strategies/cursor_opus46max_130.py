"""Circuit Breaker Proximity — cursor_opus46max_130

Thesis: When a stock approaches its circuit limit, exhaustion + reversion or
magnet effect occurs. Fade the move when velocity slows and volume declines.
Proxied by: large intraday move + declining velocity + declining volume.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_atr(high, low, close, period):
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    if n < period:
        return atr
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr


class Strategy(BaseStrategy):
    name = "cursor_opus46max_130"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 915     # 15:15
    max_trades_per_day = 4
    assumptions = [
        "Circuit limit bands not available; proxied by large intraday return > 4%",
        "Exhaustion detected via declining velocity + volume",
    ]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("circuit_proxy_pct", default=4.0, low=2.5, high=6.0),
            TunableParam("velocity_lookback", default=5.0, low=3.0, high=10.0),
            TunableParam("vix_max", default=25.0, low=18.0, high=30.0),
            TunableParam("stop_bps", default=30.0, low=15.0, high=50.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        circuit_pct = params.get("circuit_proxy_pct", 4.0)
        vel_lb = int(params.get("velocity_lookback", 5.0))
        vix_max = params.get("vix_max", 25.0)
        stop_bps = params.get("stop_bps", 30.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        day_id = df["day_id"].to_numpy()

        atr = _compute_atr(high, low, close, 14)

        # Daily open (prev close proxy)
        prev_close = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i-1]:
                prev_close[i] = close[i-1] if i > 0 else open_[i]
            else:
                prev_close[i] = prev_close[i-1]

        safe_pc = np.clip(prev_close, 1e-10, None)
        cum_ret_pct = (close - prev_close) / safe_pc * 100.0

        # Approach velocity (rate of change over vel_lb bars)
        velocity = np.zeros(n, dtype=np.float64)
        for i in range(vel_lb, n):
            velocity[i] = abs(cum_ret_pct[i]) - abs(cum_ret_pct[i - vel_lb])

        # Volume decline
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ratio = volume / avg_vol

        vol_declining = np.zeros(n, dtype=np.bool_)
        for i in range(3, n):
            vol_declining[i] = vol_ratio[i] < vol_ratio[i-3]

        # Velocity declining
        vel_declining = np.zeros(n, dtype=np.bool_)
        for i in range(3, n):
            vel_declining[i] = velocity[i] < velocity[i-3]

        exhaustion = vel_declining & vol_declining

        vix_ok = vix < vix_max
        # Skip last 30 min
        time_ok = (time_mins >= 560) & (time_mins <= 885)

        # Two consecutive up bars for long confirmation near lower circuit
        up2 = np.zeros(n, dtype=np.bool_)
        dn2 = np.zeros(n, dtype=np.bool_)
        for i in range(2, n):
            up2[i] = close[i] > close[i-1] and close[i-1] > close[i-2]
            dn2[i] = close[i] < close[i-1] and close[i-1] < close[i-2]

        # Long: near lower circuit, exhaustion, bounce
        long_entry = (
            (cum_ret_pct < -circuit_pct) &
            exhaustion &
            up2 &
            vix_ok &
            time_ok
        )

        # Short: near upper circuit, exhaustion, pullback
        short_entry = (
            (cum_ret_pct > circuit_pct) &
            exhaustion &
            dn2 &
            vix_ok &
            time_ok
        )

        # Signal exit: velocity re-accelerates
        sig_exit_long = velocity > 0.5
        sig_exit_short = velocity > 0.5

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_bps / 10000.0,
            target_pct=0.005,
            trailing_stop_pct=0.001,
            trailing_activate_pct=0.002,
            time_stop_bars=20,
        )
