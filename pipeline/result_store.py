"""Structured result storage for backtest runs.

Storage layout:
    result_store/backtest.db          — SQLite: runs, parameters, features,
                                        splits, results, daily_returns tables
    result_store/trades/<strategy_id>/<run_id>.parquet  — trade records

All writes are atomic:
    - SQLite: single transaction per save_run() call
    - Parquet: write to *.tmp then os.rename()

Schema version: "1.0"
Cost model version: "post_apr_2025" (STT 0.15% from 1 Apr 2025)
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from pipeline.config import (
    ROOT,
    BASE_SHARPE_THRESHOLD,
    CAPITAL_PER_ENTRY,
    NIFTY_LOT,
)
from pipeline.cost_model import compute_trade_costs
from pipeline.dsr import compute_dsr, compute_effective_n

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
SCHEMA_VERSION = "1.0"
COST_MODEL_VERSION = "post_apr_2025"
BACKTEST_ENGINE = "event_driven"
STORE_DIR = ROOT / "result_store"
DB_PATH = STORE_DIR / "backtest.db"
KILL_SHARPE_THRESHOLD = BASE_SHARPE_THRESHOLD  # 0.5


class ResultValidationError(Exception):
    """Raised when a required field is null at save time."""


# ── Strategy family extraction ────────────────────────────────────────────────

def extract_strategy_family(strategy_id: str) -> str:
    """Strip trailing version suffix (_v1, _v22, _002, etc.) to get family name."""
    return re.sub(r'_v?\d+[a-z]*$', '', strategy_id)


# ── Git helpers ───────────────────────────────────────────────────────────────

def _git_head_sha() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(ROOT), timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


# ── BPS helpers ───────────────────────────────────────────────────────────────

def _compute_bps_breakdown(trades_df: pl.DataFrame, lot_size: int) -> dict:
    """Compute average cost/edge breakdown in basis points across all trades."""
    if trades_df is None or trades_df.is_empty():
        return {
            "gross_edge_bps": 0.0,
            "spread_cost_bps": 0.0,
            "slippage_cost_bps": 0.0,
            "impact_cost_bps": 0.0,
            "fees_cost_bps": 0.0,
            "total_cost_bps": 0.0,
            "net_edge_bps": 0.0,
        }

    gross_bps_list = []
    fees_bps_list = []

    for row in trades_df.iter_rows(named=True):
        ep = row.get("entry_premium", 0.0) or 0.0
        xp = row.get("exit_premium", 0.0) or 0.0
        if ep <= 0:
            continue

        cost_info = compute_trade_costs(ep, xp, lot_size, capital=CAPITAL_PER_ENTRY)
        qty = cost_info["qty"]
        if qty <= 0:
            continue

        gross_pnl = cost_info["gross_pnl"]
        total_cost = cost_info["total_cost"]
        deployed = ep * qty

        gross_bps = (gross_pnl / deployed) * 10000 if deployed > 0 else 0.0
        fees_bps = (total_cost / deployed) * 10000 if deployed > 0 else 0.0

        gross_bps_list.append(gross_bps)
        fees_bps_list.append(fees_bps)

    if not gross_bps_list:
        return {
            "gross_edge_bps": 0.0,
            "spread_cost_bps": 0.0,
            "slippage_cost_bps": 0.0,
            "impact_cost_bps": 0.0,
            "fees_cost_bps": 0.0,
            "total_cost_bps": 0.0,
            "net_edge_bps": 0.0,
        }

    avg_gross = float(np.mean(gross_bps_list))
    avg_fees = float(np.mean(fees_bps_list))
    # No spread/slippage/impact in this model (limit orders at close)
    return {
        "gross_edge_bps": round(avg_gross, 2),
        "spread_cost_bps": 0.0,
        "slippage_cost_bps": 0.0,
        "impact_cost_bps": 0.0,
        "fees_cost_bps": round(avg_fees, 2),
        "total_cost_bps": round(avg_fees, 2),
        "net_edge_bps": round(avg_gross - avg_fees, 2),
    }


def _compute_daily_returns(
    trades_df: pl.DataFrame,
    lot_size: int,
    capital: float = CAPITAL_PER_ENTRY,
) -> np.ndarray:
    """Compute daily net return series from trades_df."""
    if trades_df is None or trades_df.is_empty():
        return np.array([])

    from pipeline.cost_model import net_pnl_quick

    if "entry_premium" not in trades_df.columns or "exit_premium" not in trades_df.columns:
        return np.array([])

    ep = trades_df["entry_premium"].to_numpy()
    xp = trades_df["exit_premium"].to_numpy()
    net_arr = np.array([
        net_pnl_quick(e, x, lot_size, capital=capital) for e, x in zip(ep, xp)
    ])

    trades_with_net = trades_df.with_columns(pl.Series("_net_pnl", net_arr))
    trades_with_date = trades_with_net.with_columns(
        pl.col("exit_time").cast(pl.Date).alias("trade_date")
    )
    daily = trades_with_date.group_by("trade_date").agg(
        pl.col("_net_pnl").sum().alias("daily_pnl")
    ).sort("trade_date")

    return (daily["daily_pnl"].to_numpy() / max(capital, 1)).astype(np.float64)


# ── ResultStore ───────────────────────────────────────────────────────────────

class ResultStore:
    """Persistent store for backtest run artifacts.

    All writes are atomic. Schema versioned at "1.0".
    """

    def __init__(self, store_dir: Optional[Path] = None):
        self._store_dir = Path(store_dir) if store_dir else STORE_DIR
        self._db_path = self._store_dir / "backtest.db"
        self._trades_dir = self._store_dir / "trades"
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._trades_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _ensure_schema(self):
        with self._connect() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            INSERT OR IGNORE INTO schema_meta (key, value)
                VALUES ('schema_version', '1.0'),
                       ('created_at', datetime('now'));

            CREATE TABLE IF NOT EXISTS runs (
                run_id              TEXT PRIMARY KEY,
                strategy_id         TEXT NOT NULL,
                strategy_family     TEXT NOT NULL,
                created_at          TEXT NOT NULL,
                data_source         TEXT,
                data_start          TEXT,
                data_end            TEXT,
                backtest_engine     TEXT NOT NULL DEFAULT 'event_driven',
                cost_model_version  TEXT NOT NULL DEFAULT 'post_apr_2025',
                git_commit          TEXT,
                optuna_study_id     TEXT,
                optuna_trial_number INTEGER,
                is_sensitivity_run  INTEGER NOT NULL DEFAULT 0,
                notes               TEXT,
                schema_version      TEXT NOT NULL DEFAULT '1.0'
            );

            CREATE TABLE IF NOT EXISTS parameters (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id       TEXT NOT NULL,
                strategy_id  TEXT NOT NULL,
                param_name   TEXT NOT NULL,
                param_value  REAL NOT NULL,
                param_type   TEXT NOT NULL DEFAULT 'other',
                is_optimized INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS features (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id              TEXT NOT NULL,
                strategy_id         TEXT NOT NULL,
                feature_name        TEXT NOT NULL,
                feature_description TEXT,
                computation_window  INTEGER,
                data_source         TEXT NOT NULL DEFAULT '5s',
                importance_score    REAL
            );

            CREATE TABLE IF NOT EXISTS splits (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id         TEXT NOT NULL,
                strategy_id    TEXT NOT NULL,
                fold_number    INTEGER NOT NULL DEFAULT 0,
                split_type     TEXT NOT NULL DEFAULT 'leave_one_out',
                train_start    TEXT,
                train_end      TEXT,
                test_start     TEXT,
                test_end       TEXT,
                n_train_bars   INTEGER,
                n_test_bars    INTEGER
            );

            CREATE TABLE IF NOT EXISTS results (
                id                       INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id                   TEXT NOT NULL,
                strategy_id              TEXT NOT NULL,
                fold_number              INTEGER NOT NULL DEFAULT 0,
                split                    TEXT NOT NULL DEFAULT 'full',
                sharpe_raw               REAL,
                sharpe_deflated          REAL,
                cagr                     REAL,
                max_drawdown             REAL,
                win_rate                 REAL,
                avg_hold_seconds         REAL,
                total_trades             INTEGER,
                trades_per_day           REAL,
                gross_edge_bps           REAL,
                spread_cost_bps          REAL,
                slippage_cost_bps        REAL,
                impact_cost_bps          REAL,
                fees_cost_bps            REAL,
                total_cost_bps           REAL,
                net_edge_bps             REAL,
                fill_rate                REAL,
                kill_condition_triggered INTEGER NOT NULL DEFAULT 0,
                backtest_engine          TEXT,
                cost_model_version       TEXT
            );

            CREATE TABLE IF NOT EXISTS daily_returns (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                fold_number INTEGER NOT NULL DEFAULT 0,
                trade_date  TEXT NOT NULL,
                return_val  REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_runs_strategy ON runs (strategy_id);
            CREATE INDEX IF NOT EXISTS idx_parameters_run ON parameters (run_id);
            CREATE INDEX IF NOT EXISTS idx_results_run ON results (run_id);
            CREATE INDEX IF NOT EXISTS idx_results_strategy ON results (strategy_id);
            CREATE INDEX IF NOT EXISTS idx_daily_returns_run ON daily_returns (run_id);
            """)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ── Public interface ──────────────────────────────────────────────────────

    def save_run(
        self,
        run_record: dict,
        parameters: list[dict],
        features: list[dict],
        splits: list[dict],
        results: list[dict],
        trades: Optional[pl.DataFrame],
        lot_size: int = NIFTY_LOT,
        capital: float = CAPITAL_PER_ENTRY,
        signal_arrays: Optional[dict] = None,
    ) -> str:
        """Save all artifacts for one backtest run.

        Args:
            run_record:  dict matching runs table schema. run_id generated here
                         if not provided.
            parameters:  list of dicts (one per parameter).
            features:    list of dicts (one per feature).
            splits:      list of dicts (one per fold).
            results:     list of dicts (one per fold × split). Bps fields computed
                         here from trades_df if not provided.
            trades:      Polars DataFrame of trade records (may be None).
            lot_size:    Contract lot size (for bps computation).
            capital:     Capital per entry (for bps computation).
            signal_arrays: Optional dict of {signal_name: np.ndarray} used to
                           populate entry_signal_values on trades.

        Returns:
            run_id string.

        Raises:
            ResultValidationError if a required field is null.
        """
        # ── Generate run_id ──
        run_id = run_record.get("run_id") or str(uuid.uuid4())
        strategy_id = run_record.get("strategy_id")
        if not strategy_id:
            raise ResultValidationError("run_record.strategy_id is required")

        # ── Validate required fields ──
        required_run_fields = ["strategy_id", "backtest_engine", "cost_model_version"]
        for f in required_run_fields:
            if not run_record.get(f):
                raise ResultValidationError(f"run_record.{f} is required")

        # ── Fill in defaults ──
        now_iso = datetime.now(timezone.utc).isoformat()
        run_record = {
            "run_id": run_id,
            "strategy_id": strategy_id,
            "strategy_family": extract_strategy_family(strategy_id),
            "created_at": run_record.get("created_at", now_iso),
            "data_source": run_record.get("data_source", "5second_data"),
            "data_start": run_record.get("data_start"),
            "data_end": run_record.get("data_end"),
            "backtest_engine": run_record.get("backtest_engine", BACKTEST_ENGINE),
            "cost_model_version": run_record.get("cost_model_version", COST_MODEL_VERSION),
            "git_commit": run_record.get("git_commit") or _git_head_sha(),
            "optuna_study_id": run_record.get("optuna_study_id"),
            "optuna_trial_number": run_record.get("optuna_trial_number"),
            "is_sensitivity_run": int(bool(run_record.get("is_sensitivity_run", False))),
            "notes": run_record.get("notes"),
            "schema_version": SCHEMA_VERSION,
        }

        # ── Enrich trades with entry_signal_values ──
        if trades is not None and not trades.is_empty() and signal_arrays:
            trades = _attach_signal_values(trades, signal_arrays)

        # ── Compute bps breakdown for each result record ──
        bps = _compute_bps_breakdown(trades, lot_size) if trades is not None else {}
        daily_returns = _compute_daily_returns(trades, lot_size, capital) if trades is not None else np.array([])

        # ── Compute DSR ──
        n_current = self.count_total_runs()  # query before inserting this run
        n_total = max(n_current + 1, 1)
        dsr_result = compute_dsr(daily_returns, n_total)

        # ── Enrich result records ──
        enriched_results = []
        for res in results:
            r = dict(res)
            r["run_id"] = run_id
            r["strategy_id"] = strategy_id
            r.setdefault("backtest_engine", run_record["backtest_engine"])
            r.setdefault("cost_model_version", run_record["cost_model_version"])
            # Fill bps from computed breakdown if not already set
            for k in ("gross_edge_bps", "spread_cost_bps", "slippage_cost_bps",
                      "impact_cost_bps", "fees_cost_bps", "total_cost_bps", "net_edge_bps"):
                r.setdefault(k, bps.get(k, 0.0))
            # Fill DSR
            r.setdefault("sharpe_deflated", dsr_result["sharpe_deflated"])
            r.setdefault("sharpe_raw", dsr_result["sharpe_raw"])
            # Kill condition: after-cost sharpe < 0.5
            sharpe = r.get("sharpe_raw") or 0.0
            kill = sharpe < KILL_SHARPE_THRESHOLD
            r.setdefault("kill_condition_triggered", int(kill))
            enriched_results.append(r)

        # ── Write to SQLite (single transaction = atomic) ──
        with self._connect() as conn:
            conn.execute("BEGIN")
            try:
                self._insert_run(conn, run_record)
                self._insert_rows(conn, "parameters", parameters, run_id, strategy_id)
                self._insert_rows(conn, "features", features, run_id, strategy_id)
                self._insert_rows(conn, "splits", splits, run_id, strategy_id)
                self._insert_results(conn, enriched_results)
                self._insert_daily_returns(conn, run_id, strategy_id, daily_returns)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

        # ── Write trades to Parquet (temp→rename for atomicity) ──
        if trades is not None and not trades.is_empty():
            self._save_trades_parquet(trades, strategy_id, run_id)

        log.info("Saved run %s for strategy %s (%d trades)",
                 run_id, strategy_id, len(trades) if trades is not None else 0)
        return run_id

    def load_run(self, run_id: str) -> dict:
        """Load all artifacts for a run_id."""
        with self._connect() as conn:
            run = dict(conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone() or {})
            if not run:
                return {}

            run["parameters"] = [dict(r) for r in conn.execute(
                "SELECT * FROM parameters WHERE run_id = ?", (run_id,)
            ).fetchall()]
            run["features"] = [dict(r) for r in conn.execute(
                "SELECT * FROM features WHERE run_id = ?", (run_id,)
            ).fetchall()]
            run["splits"] = [dict(r) for r in conn.execute(
                "SELECT * FROM splits WHERE run_id = ?", (run_id,)
            ).fetchall()]
            run["results"] = [dict(r) for r in conn.execute(
                "SELECT * FROM results WHERE run_id = ?", (run_id,)
            ).fetchall()]

        # Attach trades if present
        trades_path = self._trade_path(run["strategy_id"], run_id)
        if trades_path.exists():
            run["trades"] = pl.read_parquet(trades_path)
        return run

    def load_strategy_results(
        self,
        strategy_id: str,
        split: str = "test",
    ) -> pl.DataFrame:
        """Load result records for a strategy, optionally filtered by split."""
        with self._connect() as conn:
            if split:
                rows = conn.execute(
                    "SELECT r.*, rn.created_at, rn.backtest_engine as engine, "
                    "rn.cost_model_version as cost_ver, rn.is_sensitivity_run "
                    "FROM results r JOIN runs rn ON r.run_id = rn.run_id "
                    "WHERE r.strategy_id = ? AND r.split = ? "
                    "ORDER BY rn.created_at DESC",
                    (strategy_id, split),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT r.*, rn.created_at, rn.backtest_engine as engine, "
                    "rn.cost_model_version as cost_ver, rn.is_sensitivity_run "
                    "FROM results r JOIN runs rn ON r.run_id = rn.run_id "
                    "WHERE r.strategy_id = ? "
                    "ORDER BY rn.created_at DESC",
                    (strategy_id,),
                ).fetchall()
        if not rows:
            return pl.DataFrame()
        return pl.from_dicts([dict(r) for r in rows])

    def compare_runs(self, run_ids: list[str]) -> pl.DataFrame:
        """Side-by-side comparison of multiple runs."""
        if not run_ids:
            return pl.DataFrame()
        placeholders = ",".join("?" * len(run_ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT r.*, rn.created_at, rn.data_source, rn.backtest_engine, "
                f"rn.cost_model_version, rn.git_commit, rn.notes "
                f"FROM results r JOIN runs rn ON r.run_id = rn.run_id "
                f"WHERE r.run_id IN ({placeholders}) "
                f"ORDER BY r.run_id, r.fold_number, r.split",
                run_ids,
            ).fetchall()
        if not rows:
            return pl.DataFrame()
        return pl.from_dicts([dict(r) for r in rows])

    def count_total_runs(self) -> int:
        """Total number of run records in the store (for DSR N computation)."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM runs").fetchone()
            return int(row[0]) if row else 0

    def get_all_return_series(self) -> dict[str, np.ndarray]:
        """Return {run_id: daily_return_array} for all stored runs.

        Used for effective N computation in DSR.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT run_id, trade_date, return_val FROM daily_returns "
                "ORDER BY run_id, trade_date"
            ).fetchall()
        if not rows:
            return {}

        series: dict[str, list] = {}
        for row in rows:
            series.setdefault(row["run_id"], []).append(row["return_val"])
        return {k: np.array(v, dtype=np.float64) for k, v in series.items()}

    def get_strategy_list(self) -> list[dict]:
        """Return list of all strategies with their best test result."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT r.strategy_id, r.strategy_family,
                       MAX(res.sharpe_raw) as best_sharpe_raw,
                       MAX(res.sharpe_deflated) as best_sharpe_deflated,
                       MAX(res.net_edge_bps) as best_net_edge_bps,
                       MAX(res.kill_condition_triggered) as any_kill,
                       MAX(rn.backtest_engine = 'vectorized') as has_vectorized,
                       COUNT(DISTINCT r.run_id) as n_runs
                FROM runs r
                LEFT JOIN results res ON r.run_id = res.run_id AND res.split IN ('test', 'full')
                LEFT JOIN runs rn ON rn.run_id = r.run_id
                GROUP BY r.strategy_id
                ORDER BY best_sharpe_deflated DESC
            """).fetchall()
        return [dict(r) for r in rows]

    def get_all_runs(self) -> list[dict]:
        """Return all run records."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def get_run_parameters(self, run_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM parameters WHERE run_id = ?", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_run_features(self, run_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM features WHERE run_id = ?", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def load_trades(self, strategy_id: str, run_id: str) -> Optional[pl.DataFrame]:
        path = self._trade_path(strategy_id, run_id)
        if path.exists():
            return pl.read_parquet(path)
        return None

    def audit(self) -> list[str]:
        """Check result store for data quality issues.

        Returns list of warning strings. Does not raise — never blocks reads.
        """
        warnings = []

        with self._connect() as conn:
            # ── Missing required fields ──
            null_strategy = conn.execute(
                "SELECT COUNT(*) FROM runs WHERE strategy_id IS NULL OR strategy_id = ''"
            ).fetchone()[0]
            if null_strategy:
                warnings.append(f"WARN: {null_strategy} run(s) have null/empty strategy_id")

            null_engine = conn.execute(
                "SELECT COUNT(*) FROM runs WHERE backtest_engine IS NULL OR backtest_engine = ''"
            ).fetchone()[0]
            if null_engine:
                warnings.append(f"WARN: {null_engine} run(s) have null backtest_engine")

            # ── Vectorized backtests ──
            n_vec = conn.execute(
                "SELECT COUNT(*) FROM runs WHERE backtest_engine = 'vectorized'"
            ).fetchone()[0]
            if n_vec:
                warnings.append(
                    f"WARN: {n_vec} run(s) use backtest_engine='vectorized' — "
                    "SUSPECT, do not use for promotion decisions"
                )

            # ── DSR sanity: sharpe_deflated > sharpe_raw ──
            n_dsr_err = conn.execute(
                "SELECT COUNT(*) FROM results "
                "WHERE sharpe_deflated IS NOT NULL AND sharpe_raw IS NOT NULL "
                "  AND sharpe_deflated > sharpe_raw"
            ).fetchone()[0]
            if n_dsr_err:
                warnings.append(
                    f"WARN: {n_dsr_err} result(s) have sharpe_deflated > sharpe_raw "
                    "(computation error — DSR is a probability 0-1, raw Sharpe is unbounded)"
                )

            # ── Loss-making strategies ──
            n_loss = conn.execute(
                "SELECT COUNT(DISTINCT strategy_id) FROM results "
                "WHERE net_edge_bps < 0 AND split IN ('test', 'full')"
            ).fetchone()[0]
            if n_loss:
                warnings.append(
                    f"WARN: {n_loss} strategy(ies) have net_edge_bps < 0 (loss-making after costs)"
                )

            # ── Kill condition triggered ──
            killed = conn.execute(
                "SELECT strategy_id FROM results "
                "WHERE kill_condition_triggered = 1 AND split IN ('test', 'full') "
                "GROUP BY strategy_id"
            ).fetchall()
            for row in killed:
                warnings.append(
                    f"WARN: kill_condition_triggered for strategy '{row[0]}' "
                    f"(after-cost Sharpe < {KILL_SHARPE_THRESHOLD})"
                )

            # ── Runs with no result records ──
            n_no_results = conn.execute(
                "SELECT COUNT(*) FROM runs r "
                "WHERE NOT EXISTS (SELECT 1 FROM results res WHERE res.run_id = r.run_id)"
            ).fetchone()[0]
            if n_no_results:
                warnings.append(
                    f"WARN: {n_no_results} run(s) have no result records"
                )

            # ── Runs with no trade records ──
            all_run_ids = [r["run_id"] for r in conn.execute(
                "SELECT run_id, strategy_id FROM runs"
            ).fetchall()]

        run_rows = []
        with self._connect() as conn:
            run_rows = conn.execute("SELECT run_id, strategy_id FROM runs").fetchall()

        n_no_trades = 0
        for row in run_rows:
            path = self._trade_path(row["strategy_id"], row["run_id"])
            if not path.exists():
                n_no_trades += 1
        if n_no_trades:
            warnings.append(
                f"INFO: {n_no_trades} run(s) have no trade parquet file "
                "(may have zero trades or pre-dates this storage layer)"
            )

        # ── Sensitivity runs not saved to store ──
        with self._connect() as conn:
            n_runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            n_sens = conn.execute(
                "SELECT COUNT(*) FROM runs WHERE is_sensitivity_run = 1"
            ).fetchone()[0]

        if n_runs == 0:
            warnings.append(
                "INFO: Result store is empty. Run backtests with --save flag to populate."
            )

        if not warnings:
            warnings.append("OK: No issues found in result store.")

        return warnings

    # ── Private helpers ───────────────────────────────────────────────────────

    def _insert_run(self, conn, rec: dict):
        conn.execute("""
            INSERT OR REPLACE INTO runs
            (run_id, strategy_id, strategy_family, created_at, data_source,
             data_start, data_end, backtest_engine, cost_model_version,
             git_commit, optuna_study_id, optuna_trial_number,
             is_sensitivity_run, notes, schema_version)
            VALUES
            (:run_id, :strategy_id, :strategy_family, :created_at, :data_source,
             :data_start, :data_end, :backtest_engine, :cost_model_version,
             :git_commit, :optuna_study_id, :optuna_trial_number,
             :is_sensitivity_run, :notes, :schema_version)
        """, rec)

    def _insert_rows(self, conn, table: str, rows: list[dict], run_id: str, strategy_id: str):
        for row in rows:
            row = dict(row)
            row["run_id"] = run_id
            row["strategy_id"] = strategy_id
            if table == "parameters":
                conn.execute("""
                    INSERT INTO parameters
                    (run_id, strategy_id, param_name, param_value, param_type, is_optimized)
                    VALUES (:run_id, :strategy_id, :param_name, :param_value,
                            :param_type, :is_optimized)
                """, row)
            elif table == "features":
                conn.execute("""
                    INSERT INTO features
                    (run_id, strategy_id, feature_name, feature_description,
                     computation_window, data_source, importance_score)
                    VALUES (:run_id, :strategy_id, :feature_name, :feature_description,
                            :computation_window, :data_source, :importance_score)
                """, row)
            elif table == "splits":
                conn.execute("""
                    INSERT INTO splits
                    (run_id, strategy_id, fold_number, split_type,
                     train_start, train_end, test_start, test_end,
                     n_train_bars, n_test_bars)
                    VALUES (:run_id, :strategy_id, :fold_number, :split_type,
                            :train_start, :train_end, :test_start, :test_end,
                            :n_train_bars, :n_test_bars)
                """, row)

    def _insert_results(self, conn, rows: list[dict]):
        for row in rows:
            conn.execute("""
                INSERT INTO results
                (run_id, strategy_id, fold_number, split,
                 sharpe_raw, sharpe_deflated, cagr, max_drawdown, win_rate,
                 avg_hold_seconds, total_trades, trades_per_day,
                 gross_edge_bps, spread_cost_bps, slippage_cost_bps,
                 impact_cost_bps, fees_cost_bps, total_cost_bps, net_edge_bps,
                 fill_rate, kill_condition_triggered, backtest_engine, cost_model_version)
                VALUES
                (:run_id, :strategy_id, :fold_number, :split,
                 :sharpe_raw, :sharpe_deflated, :cagr, :max_drawdown, :win_rate,
                 :avg_hold_seconds, :total_trades, :trades_per_day,
                 :gross_edge_bps, :spread_cost_bps, :slippage_cost_bps,
                 :impact_cost_bps, :fees_cost_bps, :total_cost_bps, :net_edge_bps,
                 :fill_rate, :kill_condition_triggered, :backtest_engine, :cost_model_version)
            """, {
                "run_id": row.get("run_id"),
                "strategy_id": row.get("strategy_id"),
                "fold_number": row.get("fold_number", 0),
                "split": row.get("split", "full"),
                "sharpe_raw": row.get("sharpe_raw"),
                "sharpe_deflated": row.get("sharpe_deflated"),
                "cagr": row.get("cagr"),
                "max_drawdown": row.get("max_drawdown"),
                "win_rate": row.get("win_rate"),
                "avg_hold_seconds": row.get("avg_hold_seconds"),
                "total_trades": row.get("total_trades"),
                "trades_per_day": row.get("trades_per_day"),
                "gross_edge_bps": row.get("gross_edge_bps", 0.0),
                "spread_cost_bps": row.get("spread_cost_bps", 0.0),
                "slippage_cost_bps": row.get("slippage_cost_bps", 0.0),
                "impact_cost_bps": row.get("impact_cost_bps", 0.0),
                "fees_cost_bps": row.get("fees_cost_bps", 0.0),
                "total_cost_bps": row.get("total_cost_bps", 0.0),
                "net_edge_bps": row.get("net_edge_bps", 0.0),
                "fill_rate": row.get("fill_rate"),
                "kill_condition_triggered": int(bool(row.get("kill_condition_triggered", False))),
                "backtest_engine": row.get("backtest_engine", BACKTEST_ENGINE),
                "cost_model_version": row.get("cost_model_version", COST_MODEL_VERSION),
            })

    def _insert_daily_returns(
        self,
        conn,
        run_id: str,
        strategy_id: str,
        daily_returns: np.ndarray,
        fold_number: int = 0,
    ):
        for i, r in enumerate(daily_returns):
            conn.execute(
                "INSERT INTO daily_returns (run_id, strategy_id, fold_number, "
                "trade_date, return_val) VALUES (?, ?, ?, ?, ?)",
                (run_id, strategy_id, fold_number, str(i), float(r)),
            )

    def _save_trades_parquet(
        self,
        trades: pl.DataFrame,
        strategy_id: str,
        run_id: str,
    ):
        dest_dir = self._trades_dir / strategy_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{run_id}.parquet"
        tmp = dest.with_suffix(".tmp")
        trades.write_parquet(tmp)
        os.replace(tmp, dest)  # atomic rename

    def _trade_path(self, strategy_id: str, run_id: str) -> Path:
        return self._trades_dir / strategy_id / f"{run_id}.parquet"


# ── Trade record builder ──────────────────────────────────────────────────────

def _attach_signal_values(
    trades_df: pl.DataFrame,
    signal_arrays: dict[str, np.ndarray],
) -> pl.DataFrame:
    """Add entry_signal_values JSON column to trades_df using signal arrays.

    signal_arrays: dict of {signal_name: array of length n_bars}
    trades_df must have 'entry_bar' column with bar indices.
    """
    if "entry_bar" not in trades_df.columns:
        trades_df = trades_df.with_columns(
            pl.lit(None).cast(pl.Utf8).alias("entry_signal_values")
        )
        return trades_df

    entry_bars = trades_df["entry_bar"].to_numpy().astype(int)
    snapshots = []
    for bar_idx in entry_bars:
        snap = {}
        for name, arr in signal_arrays.items():
            try:
                val = arr[bar_idx]
                if isinstance(val, (np.bool_,)):
                    snap[name] = bool(val)
                elif isinstance(val, (np.integer,)):
                    snap[name] = int(val)
                elif isinstance(val, (np.floating,)):
                    v = float(val)
                    snap[name] = None if (math.isnan(v) or math.isinf(v)) else v
                else:
                    snap[name] = val
            except (IndexError, TypeError):
                snap[name] = None
        snapshots.append(json.dumps(snap))

    return trades_df.with_columns(
        pl.Series("entry_signal_values", snapshots, dtype=pl.Utf8)
    )


def build_feature_records(strategy) -> list[dict]:
    """Infer feature records from strategy's TunableParam and signal field names."""
    features = []
    for tp in strategy.tunable_params():
        features.append({
            "feature_name": tp.name,
            "feature_description": f"Tunable threshold: {tp.name} (default={tp.default}, "
                                   f"range=[{tp.low}, {tp.high}])",
            "computation_window": None,
            "data_source": "5s",
            "importance_score": None,
        })
    # Standard OptionSignals fields as derived features
    for sig_name, desc in [
        ("buy_ce", "Entry signal: buy call (bullish)"),
        ("buy_pe", "Entry signal: buy put (bearish)"),
        ("stop_points", "Risk parameter: stop loss in premium points"),
        ("target_points", "Risk parameter: target in premium points"),
        ("strike_offset", "Strike selection: offset from ATM"),
    ]:
        features.append({
            "feature_name": sig_name,
            "feature_description": desc,
            "computation_window": getattr(strategy, "max_lookback", None),
            "data_source": "5s",
            "importance_score": None,
        })
    return features


def build_run_record(
    strategy_id: str,
    spot_df,
    backtest_engine: str = BACKTEST_ENGINE,
    cost_model_version: str = COST_MODEL_VERSION,
    optuna_study_id: Optional[str] = None,
    optuna_trial_number: Optional[int] = None,
    is_sensitivity_run: bool = False,
    notes: Optional[str] = None,
) -> dict:
    """Build a run record dict from context."""
    data_start = None
    data_end = None
    if spot_df is not None and "datetime" in spot_df.columns and not spot_df.is_empty():
        data_start = str(spot_df["datetime"].min())
        data_end = str(spot_df["datetime"].max())

    return {
        "strategy_id": strategy_id,
        "backtest_engine": backtest_engine,
        "cost_model_version": cost_model_version,
        "data_start": data_start,
        "data_end": data_end,
        "optuna_study_id": optuna_study_id,
        "optuna_trial_number": optuna_trial_number,
        "is_sensitivity_run": is_sensitivity_run,
        "notes": notes,
    }


def build_result_record(
    metrics: dict,
    split: str = "full",
    fold_number: int = 0,
    total_trading_days: int = 1,
) -> dict:
    """Build a result record dict from a metrics dict."""
    sharpe = metrics.get("sharpe_annualized", 0.0) or 0.0
    total_trades = metrics.get("total_trades", 0) or 0
    trades_per_day = (total_trades / total_trading_days) if total_trading_days > 0 else 0.0

    # CAGR from total_pnl / capital annualized
    total_pnl = metrics.get("total_pnl", 0.0) or 0.0
    total_return = total_pnl / CAPITAL_PER_ENTRY
    n_days = max(total_trading_days, 1)
    cagr = (1 + total_return) ** (252.0 / n_days) - 1.0 if total_return > -1 else -1.0

    return {
        "fold_number": fold_number,
        "split": split,
        "sharpe_raw": sharpe,
        "sharpe_deflated": None,  # computed in save_run()
        "cagr": round(cagr, 4),
        "max_drawdown": metrics.get("max_drawdown"),
        "win_rate": metrics.get("win_rate"),
        "avg_hold_seconds": metrics.get("avg_holding_seconds"),
        "total_trades": total_trades,
        "trades_per_day": round(trades_per_day, 2),
        "fill_rate": None,  # not modelled
        "kill_condition_triggered": int(sharpe < KILL_SHARPE_THRESHOLD),
    }


def build_split_record(
    spot_df,
    fold_number: int = 0,
    split_type: str = "leave_one_out",
    test_day_id: Optional[int] = None,
) -> dict:
    """Build a split record from spot_df context."""
    if spot_df is None or spot_df.is_empty():
        return {
            "fold_number": fold_number,
            "split_type": split_type,
            "train_start": None, "train_end": None,
            "test_start": None, "test_end": None,
            "n_train_bars": None, "n_test_bars": None,
        }

    all_start = str(spot_df["datetime"].min())
    all_end = str(spot_df["datetime"].max())

    if test_day_id is not None and "day_id" in spot_df.columns:
        test_mask = spot_df["day_id"] == test_day_id
        test_df = spot_df.filter(test_mask)
        train_df = spot_df.filter(~test_mask)
        return {
            "fold_number": fold_number,
            "split_type": split_type,
            "train_start": str(train_df["datetime"].min()) if not train_df.is_empty() else None,
            "train_end": str(train_df["datetime"].max()) if not train_df.is_empty() else None,
            "test_start": str(test_df["datetime"].min()) if not test_df.is_empty() else None,
            "test_end": str(test_df["datetime"].max()) if not test_df.is_empty() else None,
            "n_train_bars": len(train_df),
            "n_test_bars": len(test_df),
        }

    return {
        "fold_number": fold_number,
        "split_type": "fixed",
        "train_start": all_start,
        "train_end": all_end,
        "test_start": None,
        "test_end": None,
        "n_train_bars": len(spot_df),
        "n_test_bars": None,
    }


# ── Singleton accessor ────────────────────────────────────────────────────────

_default_store: Optional[ResultStore] = None


def get_store() -> ResultStore:
    """Return the default ResultStore singleton (lazy-initialised)."""
    global _default_store
    if _default_store is None:
        _default_store = ResultStore()
    return _default_store
