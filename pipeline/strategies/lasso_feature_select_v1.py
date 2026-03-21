"""lasso_feature_select_v1 — Adaptive LASSO feature selection for NIFTY direction prediction.

Refits a LASSO regression model every 5 minutes on the trailing 60 minutes of 5-second
NIFTY bars, selecting which subset of 9 technical features is currently predictive of
the next 60-second index return. Trades ATM CE/PE when the sparse model (≥2 active
features) predicts a return exceeding the threshold.

Original: LASSO on 10 equity features, 1-min bars, 5-day training window, 20-bar target.
Conversion: LASSO on 9 index features, 5-second bars, 60-min rolling window, 12-bar target.
"""
from __future__ import annotations
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl

try:
    from sklearn.linear_model import Lasso as _Lasso
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ─── indicator helpers ────────────────────────────────────────────────────────

def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.empty(len(arr), dtype=float)
    out[0] = arr[0]
    alpha = 2.0 / (period + 1)
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    out = np.full(n, 50.0)
    if n <= period:
        return out
    delta = np.diff(close, prepend=close[0])
    gains  = np.where(delta > 0,  delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    ag = np.zeros(n)
    al = np.zeros(n)
    ag[period] = gains[1:period + 1].mean()
    al[period] = losses[1:period + 1].mean()
    for i in range(period + 1, n):
        ag[i] = (ag[i - 1] * (period - 1) + gains[i])  / period
        al[i] = (al[i - 1] * (period - 1) + losses[i]) / period
    for i in range(period, n):
        rs = ag[i] / al[i] if al[i] > 1e-12 else 1e12
        out[i] = 100.0 - 100.0 / (1.0 + rs)
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low,
         np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = np.zeros(n)
    if n < period:
        return atr
    atr[period - 1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _bb_position(close: np.ndarray, period: int = 60, nstd: float = 2.0) -> np.ndarray:
    n = len(close)
    out = np.full(n, 0.5)
    for i in range(period, n):
        w  = close[i - period:i]
        mu = w.mean()
        sd = w.std()
        if sd > 1e-10:
            lower = mu - nstd * sd
            band_width = 4.0 * sd  # upper - lower = 2*nstd*sd = 4*sd
            out[i] = (close[i] - lower) / band_width
    return np.clip(out, 0.0, 1.0)


def _obv_slope(close: np.ndarray, volume: np.ndarray, period: int = 12) -> np.ndarray:
    n = len(close)
    obv = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]
    slopes = np.zeros(n)
    x = np.arange(period, dtype=float) - (period - 1) / 2.0
    ss_x = (x * x).sum()
    if ss_x < 1e-12:
        return slopes
    for i in range(period, n):
        y = obv[i - period:i]
        slopes[i] = (x * (y - y.mean())).sum() / ss_x
    return slopes


# ─── strategy ─────────────────────────────────────────────────────────────────

class Strategy(BaseStrategy):
    name = "lasso_feature_select_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — need 30-min warmup before first LASSO fit
    session_end_minutes   = 920   # 15:20 IST
    max_lookback          = 720   # 60-minute warmup (720 × 5s)
    max_trades_per_day    = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("signal_threshold",    0.0003, 0.0001, 0.001),
            TunableParam("active_features_min", 2.0,    1.0,    5.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close    = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(float)
        high     = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(float)
        low      = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(float)
        vol      = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id   = spot_df["day_id"].to_numpy()

        threshold  = params.get("signal_threshold",    0.0003)
        active_min = int(params.get("active_features_min", 2.0))

        # ── VIX feature ──────────────────────────────────────────────────────
        vix_arr = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vc")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_arr = vix_joined["vc"].fill_null(15.0).to_numpy()

        # ── Session VWAP (resets per day) ────────────────────────────────────
        cum_pv = np.zeros(n)
        cum_v  = np.zeros(n)
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i - 1]:
                cum_pv[i] = close[i] * vol[i]
                cum_v[i]  = max(vol[i], 1.0)
            else:
                cum_pv[i] = cum_pv[i - 1] + close[i] * vol[i]
                cum_v[i]  = cum_v[i - 1] + max(vol[i], 1.0)
        vwap = cum_pv / cum_v

        # ── Compute 9 features ───────────────────────────────────────────────
        safe_close = np.where(close > 0, close, 1.0)

        # F1: VWAP distance (fractional)
        vwap_dist = (close - vwap) / safe_close

        # F2: RSI(36) normalised to [-1, 1]
        rsi_arr  = _rsi(close, 36)
        rsi_feat = (rsi_arr - 50.0) / 50.0

        # F3: MACD histogram (12/26/9 bars) normalised by price
        macd_line = _ema(close, 12) - _ema(close, 26)
        macd_hist = macd_line - _ema(macd_line, 9)
        macd_norm = macd_hist / safe_close

        # F4: Bollinger position centred at 0
        bb_feat = _bb_position(close, period=60, nstd=2.0) - 0.5

        # F5: Volume ratio excess (clipped)
        rol_vol = np.zeros(n)
        for i in range(60, n):
            rol_vol[i] = vol[i - 60:i].mean()
        if n > 60:
            rol_vol[:60] = max(rol_vol[60], 1.0)
        else:
            rol_vol[:] = 1.0
        vol_feat = np.clip(vol / np.where(rol_vol > 0, rol_vol, 1.0) - 1.0, -2.0, 2.0)

        # F6: ATR(24) normalised by price
        atr_arr  = _atr(high, low, close, 24)
        atr_norm = atr_arr / safe_close

        # F7: OBV slope(12) normalised to [-1, 1]
        obv_sl   = _obv_slope(close, vol, 12)
        obv_scale = np.mean(np.abs(vol)) * 12.0
        obv_norm = np.clip(obv_sl / max(obv_scale, 1.0), -1.0, 1.0)

        # F8: 1-minute return (ROC over 12 bars)
        roc_12 = np.zeros(n)
        roc_12[12:] = (close[12:] - close[:-12]) / np.where(close[:-12] > 0, close[:-12], 1.0)

        # F9: VIX normalised (15 → 0, 25 → 1)
        vix_feat = (vix_arr - 15.0) / 10.0

        # Stack: (n, 9)
        features = np.column_stack([
            vwap_dist, rsi_feat, macd_norm, bb_feat,
            vol_feat,  atr_norm, obv_norm,  roc_12, vix_feat,
        ])
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Forward return target for LASSO training (12 bars = 60 seconds) ──
        fwd_ret = np.zeros(n)
        fwd_ret[:-12] = (close[12:] - close[:-12]) / np.where(close[:-12] > 0, close[:-12], 1.0)
        fwd_ret = np.nan_to_num(fwd_ret, nan=0.0)

        # ── LASSO: rolling refit every 5 min on trailing 60-min window ───────
        TRAIN_WINDOW = 720   # 60-min rolling training window
        REFIT_EVERY  = 60    # refit every 5 minutes
        MIN_TRAIN    = 180   # minimum 180 bars (15 min) before first fit

        lasso_pred   = np.zeros(n)
        active_count = np.zeros(n, dtype=int)

        if _HAS_SKLEARN:
            model = _Lasso(alpha=1e-4, fit_intercept=True, max_iter=2000, warm_start=True)
            last_coef:      np.ndarray | None = None
            last_intercept: float             = 0.0
            last_fit_bar:   int               = -(REFIT_EVERY + 1)  # force fit at first eligible bar

            for i in range(MIN_TRAIN, n):
                # Refit if enough time has elapsed since last fit
                if (i - last_fit_bar) >= REFIT_EVERY:
                    start = max(0, i - TRAIN_WINDOW)
                    end   = i - 12   # exclude last 12 bars — fwd_ret needs future close
                    if end - start >= MIN_TRAIN:
                        X_tr = features[start:end]
                        y_tr = fwd_ret[start:end]
                        if np.std(y_tr) > 1e-12:
                            try:
                                model.fit(X_tr, y_tr)
                                last_coef      = model.coef_.copy()
                                last_intercept = float(model.intercept_)
                                last_fit_bar   = i
                            except Exception:
                                pass

                if last_coef is not None:
                    lasso_pred[i]   = float(np.dot(features[i], last_coef)) + last_intercept
                    active_count[i] = int(np.sum(np.abs(last_coef) > 1e-10))

        # ── Entry signals ────────────────────────────────────────────────────
        in_session     = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        enough_signals = active_count >= active_min

        buy_ce = in_session & enough_signals & (lasso_pred >  threshold)
        buy_pe = in_session & enough_signals & (lasso_pred < -threshold)

        return OptionSignals(
            buy_ce        = buy_ce,
            buy_pe        = buy_pe,
            sell_ce       = np.zeros(n, dtype=bool),
            sell_pe       = np.zeros(n, dtype=bool),
            stop_points   = np.full(n, 4.0),
            target_points = np.full(n, 7.0),
            strike_offset = np.zeros(n, dtype=np.int32),
            time_stop_bars     = 12,
            max_trades_per_day = self.max_trades_per_day,
        )
