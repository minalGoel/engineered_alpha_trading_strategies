"""Strategy dashboard backend — FastAPI server.

Reads exclusively from ResultStore. Never reads raw parquet/json directly.

Usage:
    python pipeline/run_all.py --dashboard
    python pipeline/dashboard_server.py            # standalone
    python pipeline/dashboard_server.py --port 8080
"""
from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ── Lazy FastAPI import so this module doesn't break if fastapi is absent ─────
try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    _FASTAPI_OK = True
except ImportError:
    _FASTAPI_OK = False
    log.warning("FastAPI/uvicorn not installed. Run: pip install fastapi uvicorn")

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "dashboard" / "static"


def _make_app():
    if not _FASTAPI_OK:
        raise RuntimeError("FastAPI not installed. Run: pip install fastapi uvicorn")

    from pipeline.result_store import get_store, extract_strategy_family
    from pipeline.dsr import compute_dsr, compute_effective_n, compute_expected_max_sr

    app = FastAPI(title="Strategy Dashboard", version="1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    # Serve static files (dashboard HTML + chart.min.js)
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ── Helper: recompute DSR with current N ──────────────────────────────────
    def _live_dsr(daily_returns_list: list, n_trials: int) -> dict:
        import numpy as np
        if not daily_returns_list:
            return {"sharpe_deflated": 0.0, "sr_benchmark": 0.0, "passes_dsr": False}
        arr = np.array(daily_returns_list, dtype=float)
        return compute_dsr(arr, n_trials)

    # ── /api/leaderboard ──────────────────────────────────────────────────────
    @app.get("/api/leaderboard")
    def get_leaderboard():
        store = get_store()
        strategies = store.get_strategy_list()
        n_total = store.count_total_runs()

        # Group by family
        families: dict[str, list] = {}
        for s in strategies:
            fam = s.get("strategy_family") or extract_strategy_family(s["strategy_id"])
            families.setdefault(fam, []).append(s)

        # Compute per-family stats
        result = []
        for fam_name, members in sorted(families.items()):
            # Intra-family correlation (if return series available)
            avg_corr = None
            member_ids = [m["strategy_id"] for m in members]
            all_series = store.get_all_return_series()
            # Filter to this family's run_ids
            fam_run_ids = set()
            for run in store.get_all_runs():
                if run["strategy_id"] in member_ids:
                    fam_run_ids.add(run["run_id"])
            fam_series = {k: v for k, v in all_series.items() if k in fam_run_ids}
            if len(fam_series) >= 2:
                import numpy as np
                import itertools
                corrs = []
                keys = list(fam_series.keys())
                for a, b in itertools.combinations(keys, 2):
                    sa = fam_series[a]
                    sb = fam_series[b]
                    n = min(len(sa), len(sb))
                    if n >= 5:
                        try:
                            c = float(np.corrcoef(sa[:n], sb[:n])[0, 1])
                            if math.isfinite(c):
                                corrs.append(c)
                        except Exception:
                            pass
                if corrs:
                    avg_corr = round(float(sum(corrs) / len(corrs)), 3)

            best_dsr = max((m.get("best_sharpe_deflated") or 0.0 for m in members), default=0.0)
            best_edge = max((m.get("best_net_edge_bps") or 0.0 for m in members), default=0.0)
            n_passing = sum(
                1 for m in members
                if not (m.get("any_kill") or 0)
                and (m.get("best_net_edge_bps") or 0) > 0
            )
            n_vectorized = sum(1 for m in members if m.get("has_vectorized"))

            result.append({
                "family": fam_name,
                "n_strategies": len(members),
                "best_dsr": round(best_dsr, 4),
                "best_net_edge_bps": round(best_edge, 2),
                "avg_intra_family_corr": avg_corr,
                "n_passing_kill": n_passing,
                "n_flagged_vectorized": n_vectorized,
                "strategies": [_strategy_row(m, n_total, store) for m in members],
            })

        result.sort(key=lambda x: x["best_dsr"], reverse=True)
        return {"families": result, "n_total_runs": n_total}

    def _strategy_row(m: dict, n_total: int, store) -> dict:
        sid = m["strategy_id"]
        sharpe = m.get("best_sharpe_raw") or 0.0
        dsr = m.get("best_sharpe_deflated") or 0.0
        net_edge = m.get("best_net_edge_bps") or 0.0
        kill = bool(m.get("any_kill"))
        engine = "event_driven"
        if m.get("has_vectorized"):
            engine = "vectorized"
        # Colour code
        if net_edge > 10 and not kill and engine == "event_driven":
            color = "green"
        elif 5 <= net_edge <= 10 or engine == "vectorized":
            color = "yellow"
        else:
            color = "red"
        return {
            "strategy_id": sid,
            "family": m.get("strategy_family", extract_strategy_family(sid)),
            "n_runs": m.get("n_runs", 0),
            "sharpe_raw": round(sharpe, 4),
            "sharpe_deflated": round(dsr, 4),
            "net_edge_bps": round(net_edge, 2),
            "kill_triggered": kill,
            "engine": engine,
            "color": color,
        }

    # ── /api/strategy/{strategy_id} ───────────────────────────────────────────
    @app.get("/api/strategy/{strategy_id}")
    def get_strategy(strategy_id: str):
        store = get_store()
        # Load all runs for this strategy
        runs = [r for r in store.get_all_runs() if r["strategy_id"] == strategy_id]
        if not runs:
            raise HTTPException(404, f"No runs found for strategy '{strategy_id}'")

        n_total = store.count_total_runs()

        # Load best result
        best_results = store.load_strategy_results(strategy_id, split="full")
        if best_results.is_empty():
            best_results = store.load_strategy_results(strategy_id, split=None)

        # Load most recent run's features and parameters
        latest_run_id = runs[0]["run_id"]
        features = store.get_run_features(latest_run_id)
        params = store.get_run_parameters(latest_run_id)

        # DSR with current N
        all_series = store.get_all_return_series()
        strategy_run_ids = {r["run_id"] for r in runs}
        strategy_series = {k: v for k, v in all_series.items() if k in strategy_run_ids}
        raw_n, eff_n = compute_effective_n(strategy_series) if strategy_series else (n_total, n_total)
        sr_benchmark = compute_expected_max_sr(max(eff_n, 1), 252) * math.sqrt(252)

        # Strategy spec from BaseStrategy fields
        strategy_spec = {}
        try:
            import importlib
            from pipeline.config import STRATEGY_DIR
            mod = importlib.import_module(f"pipeline.strategies.{strategy_id}")
            s = mod.Strategy()
            strategy_spec = {
                "name": s.name,
                "underlying": s.underlying,
                "timeframe": s.timeframe,
                "session_start_minutes": s.session_start_minutes,
                "session_end_minutes": s.session_end_minutes,
                "max_trades_per_day": s.max_trades_per_day,
                "max_lookback": s.max_lookback,
                "assumptions": s.assumptions or [],
            }
        except Exception:
            strategy_spec = {"name": strategy_id}

        return {
            "strategy_id": strategy_id,
            "family": extract_strategy_family(strategy_id),
            "strategy_spec": strategy_spec,
            "n_runs": len(runs),
            "n_total_runs": n_total,
            "effective_n": eff_n,
            "sr_benchmark_annualized": round(sr_benchmark, 4),
            "latest_run_id": latest_run_id,
            "features": features,
            "parameters": params,
            "cost_model_version": runs[0].get("cost_model_version", "post_apr_2025"),
        }

    # ── /api/strategy/{strategy_id}/runs ──────────────────────────────────────
    @app.get("/api/strategy/{strategy_id}/runs")
    def get_strategy_runs(strategy_id: str):
        store = get_store()
        runs = [r for r in store.get_all_runs() if r["strategy_id"] == strategy_id]
        # Attach result summary to each run
        enriched = []
        for run in runs:
            rid = run["run_id"]
            results = store.compare_runs([rid])
            summary = {}
            if not results.is_empty():
                row = results.to_dicts()[0]
                summary = {
                    "sharpe_raw": row.get("sharpe_raw"),
                    "sharpe_deflated": row.get("sharpe_deflated"),
                    "net_edge_bps": row.get("net_edge_bps"),
                    "total_trades": row.get("total_trades"),
                    "win_rate": row.get("win_rate"),
                    "kill_triggered": bool(row.get("kill_condition_triggered")),
                }
            params = store.get_run_parameters(rid)
            param_summary = {p["param_name"]: p["param_value"] for p in params}
            enriched.append({**run, "result_summary": summary, "param_summary": param_summary})
        return {"runs": enriched}

    # ── /api/strategy/{strategy_id}/results ───────────────────────────────────
    @app.get("/api/strategy/{strategy_id}/results")
    def get_strategy_results(strategy_id: str, split: Optional[str] = None):
        store = get_store()
        df = store.load_strategy_results(strategy_id, split=split or "")
        if df.is_empty():
            return {"results": []}
        return {"results": df.to_dicts()}

    # ── /api/strategy/{strategy_id}/trades/{run_id} ───────────────────────────
    @app.get("/api/strategy/{strategy_id}/trades/{run_id}")
    def get_trades(
        strategy_id: str,
        run_id: str,
        page: int = Query(1, ge=1),
        page_size: int = Query(100, ge=1, le=1000),
        direction: Optional[str] = None,
        exit_reason: Optional[str] = None,
        fold_number: Optional[int] = None,
    ):
        store = get_store()
        df = store.load_trades(strategy_id, run_id)
        if df is None or df.is_empty():
            return {"trades": [], "total": 0, "page": page, "page_size": page_size}

        # Apply filters
        if direction and "side" in df.columns:
            side_val = 1 if direction.upper() == "CE" else 3
            df = df.filter(df["side"] == side_val)
        if exit_reason and "exit_reason" in df.columns:
            df = df.filter(df["exit_reason"] == exit_reason)

        total = len(df)
        start = (page - 1) * page_size
        page_df = df.slice(start, page_size)

        # Convert timestamps to strings for JSON
        rows = []
        for row in page_df.iter_rows(named=True):
            r = {}
            for k, v in row.items():
                if hasattr(v, 'isoformat'):
                    r[k] = str(v)
                elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    r[k] = None
                else:
                    r[k] = v
            rows.append(r)

        return {"trades": rows, "total": total, "page": page, "page_size": page_size}

    # ── /api/run/{run_id} ─────────────────────────────────────────────────────
    @app.get("/api/run/{run_id}")
    def get_run(run_id: str):
        store = get_store()
        run = store.load_run(run_id)
        if not run:
            raise HTTPException(404, f"Run '{run_id}' not found")
        # Remove trades df from response (too large)
        run.pop("trades", None)
        return run

    # ── /api/compare ─────────────────────────────────────────────────────────
    @app.post("/api/compare")
    def compare_runs(body: dict):
        run_ids = body.get("run_ids", [])
        if not run_ids:
            return {"comparison": []}
        store = get_store()
        df = store.compare_runs(run_ids)
        if df.is_empty():
            return {"comparison": []}

        # Parameter diffs
        param_diffs = []
        if len(run_ids) >= 2:
            all_params = {}
            for rid in run_ids:
                params = store.get_run_parameters(rid)
                all_params[rid] = {p["param_name"]: p["param_value"] for p in params}
            all_keys = set()
            for pd_ in all_params.values():
                all_keys.update(pd_.keys())
            for key in sorted(all_keys):
                vals = {rid: all_params[rid].get(key) for rid in run_ids}
                unique_vals = set(v for v in vals.values() if v is not None)
                if len(unique_vals) > 1:
                    param_diffs.append({"param_name": key, "values": vals})

        return {
            "comparison": df.to_dicts(),
            "param_diffs": param_diffs,
        }

    # ── /api/portfolio ────────────────────────────────────────────────────────
    @app.get("/api/portfolio")
    def get_portfolio():
        import numpy as np
        store = get_store()
        strategies = store.get_strategy_list()
        n_total = store.count_total_runs()

        # Correlation heatmap: all strategies with stored return series
        all_series = store.get_all_return_series()
        strategy_ids = [s["strategy_id"] for s in strategies]

        # Map strategy -> latest run's return series
        all_runs = store.get_all_runs()
        strategy_latest_run = {}
        for run in all_runs:
            sid = run["strategy_id"]
            if sid not in strategy_latest_run:
                strategy_latest_run[sid] = run["run_id"]

        corr_series = {}
        for sid in strategy_ids:
            rid = strategy_latest_run.get(sid)
            if rid and rid in all_series:
                corr_series[sid] = all_series[rid]

        corr_matrix = None
        corr_labels = list(corr_series.keys())
        if len(corr_labels) >= 2:
            min_len = min(len(v) for v in corr_series.values())
            if min_len >= 3:
                mat = np.vstack([np.array(v)[:min_len] for v in corr_series.values()])
                c = np.corrcoef(mat)
                c = np.nan_to_num(c, nan=0.0).round(3)
                corr_matrix = c.tolist()

        # DSR summary table
        dsr_summary = []
        raw_n = len(all_series)
        _, eff_n = compute_effective_n(all_series) if all_series else (0, 0)
        sr_bench = compute_expected_max_sr(max(eff_n, 1), 252) * math.sqrt(252)

        for s in strategies:
            sid = s["strategy_id"]
            rid = strategy_latest_run.get(sid)
            series_data = all_series.get(rid, []) if rid else []
            if len(series_data) >= 5:
                import numpy as _np
                dsr_info = compute_dsr(_np.array(series_data), n_total)
            else:
                dsr_info = {"sharpe_raw": 0.0, "sharpe_deflated": 0.0, "passes_dsr": False}

            dsr_summary.append({
                "strategy_id": sid,
                "raw_n": raw_n,
                "effective_n": eff_n,
                "sr_benchmark": round(sr_bench, 4),
                "sharpe_raw": dsr_info.get("sharpe_raw", 0.0),
                "sharpe_deflated": dsr_info.get("sharpe_deflated", 0.0),
                "passes_dsr": dsr_info.get("passes_dsr", False),
                "kill_triggered": bool(s.get("any_kill")),
            })

        dsr_summary.sort(key=lambda x: x["sharpe_deflated"], reverse=True)

        # Kill condition summary
        kill_list = [
            {"strategy_id": s["strategy_id"], "kill_triggered": bool(s.get("any_kill"))}
            for s in strategies
        ]

        # Cost summary
        cost_summary = []
        for s in strategies:
            sid = s["strategy_id"]
            rid = strategy_latest_run.get(sid)
            if rid:
                results_df = store.compare_runs([rid])
                if not results_df.is_empty():
                    row = results_df.to_dicts()[0]
                    cost_summary.append({
                        "strategy_id": sid,
                        "total_cost_bps": row.get("total_cost_bps", 0),
                        "net_edge_bps": row.get("net_edge_bps", 0),
                    })
        cost_summary.sort(key=lambda x: x["total_cost_bps"], reverse=True)

        return {
            "corr_labels": corr_labels,
            "corr_matrix": corr_matrix,
            "dsr_summary": dsr_summary,
            "kill_summary": kill_list,
            "cost_summary": cost_summary,
            "n_total_runs": n_total,
            "effective_n": eff_n,
            "sr_benchmark": round(sr_bench, 4),
        }

    # ── /api/audit ────────────────────────────────────────────────────────────
    @app.get("/api/audit")
    def run_audit():
        store = get_store()
        warnings = store.audit()
        return {"warnings": warnings, "n_warnings": len(warnings)}

    # ── /api/dsr ──────────────────────────────────────────────────────────────
    @app.get("/api/dsr")
    def get_dsr_summary():
        import numpy as np
        store = get_store()
        n_total = store.count_total_runs()
        all_series = store.get_all_return_series()
        raw_n, eff_n = compute_effective_n(all_series) if all_series else (0, 0)
        sr_bench = compute_expected_max_sr(max(eff_n, 1), 252) * math.sqrt(252)

        all_runs = store.get_all_runs()
        strategy_latest = {}
        for run in all_runs:
            sid = run["strategy_id"]
            if sid not in strategy_latest:
                strategy_latest[sid] = run["run_id"]

        rows = []
        for sid, rid in strategy_latest.items():
            series = all_series.get(rid, [])
            if len(series) >= 5:
                dsr = compute_dsr(np.array(series), n_total)
            else:
                dsr = {"sharpe_raw": 0.0, "sharpe_deflated": 0.0, "passes_dsr": False}
            rows.append({
                "strategy_id": sid,
                "sharpe_raw": dsr.get("sharpe_raw", 0.0),
                "sharpe_deflated": dsr.get("sharpe_deflated", 0.0),
                "sr_benchmark": round(sr_bench, 4),
                "n_obs": dsr.get("n_obs", 0),
                "passes_dsr": dsr.get("passes_dsr", False),
            })

        rows.sort(key=lambda x: x["sharpe_deflated"], reverse=True)
        return {
            "raw_n": raw_n,
            "effective_n": eff_n,
            "sr_benchmark_annualized": round(sr_bench, 4),
            "strategies": rows,
        }

    # ── Root redirect to dashboard ────────────────────────────────────────────
    @app.get("/")
    def root():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/static/index.html")

    return app


# ── Startup ───────────────────────────────────────────────────────────────────

def start_server(host: str = "127.0.0.1", port: int = 8765):
    if not _FASTAPI_OK:
        print("ERROR: FastAPI not installed. Run: pip install fastapi uvicorn")
        sys.exit(1)
    app = _make_app()
    print(f"\nDashboard running at http://{host}:{port}")
    print(f"Open http://{host}:{port} in your browser\n")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Strategy Dashboard Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    start_server(host=args.host, port=args.port)
