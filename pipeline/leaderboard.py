"""Leaderboard generation (Phase 8).

Rank VALIDATED strategies by: Test Sharpe × sqrt(passing_stock_count).
"""
from __future__ import annotations
import math
import orjson
from pathlib import Path
from typing import Optional
import polars as pl
import logging

from pipeline.config import RESULTS_DIR, OUTPUTS_DIR, required_sharpe

log = logging.getLogger(__name__)


def build_leaderboard(
    all_strategy_results: list[dict],
) -> tuple[dict, str, dict, dict]:
    """Build leaderboard from all strategy results.

    Args:
        all_strategy_results: List of dicts with keys:
            name, verdict, family, tags, train_metrics, test_metrics,
            stock_filter, optimization

    Returns:
        leaderboard_json, leaderboard_md, winning_strategies_json, family_summary_json
    """
    # ── Classify strategies ─────────────────────────────────────────────
    validated = []
    failed_by_reason = {}

    for sr in all_strategy_results:
        verdict = sr.get("verdict", "FAILED_ERROR")
        if verdict == "VALIDATED":
            test_m = sr.get("test_metrics", {})
            sf = sr.get("stock_filter", {})
            sharpe = test_m.get("sharpe_annualized", 0.0)
            passing = sf.get("passing_stocks", 0)
            score = sharpe * math.sqrt(max(passing, 1))
            sr["score"] = round(score, 4)
            sr["test_sharpe"] = round(sharpe, 4)
            sr["passing_stocks"] = passing
            validated.append(sr)
        else:
            failed_by_reason.setdefault(verdict, []).append(sr["name"])

    # Sort validated by score descending
    validated.sort(key=lambda x: x["score"], reverse=True)

    # ── Significance threshold ──────────────────────────────────────────
    n_validated = len(validated)
    sig_threshold = required_sharpe(n_validated) if n_validated > 0 else 0.5

    # ── Build leaderboard JSON ──────────────────────────────────────────
    leaderboard_entries = []
    for rank, sr in enumerate(validated, 1):
        leaderboard_entries.append({
            "rank": rank,
            "name": sr["name"],
            "family": sr.get("family", ""),
            "score": sr["score"],
            "test_sharpe": sr["test_sharpe"],
            "train_sharpe": sr.get("optimization", {}).get("optimized_sharpe", 0),
            "passing_stocks": sr["passing_stocks"],
            "total_trades": sr.get("test_metrics", {}).get("total_trades", 0),
            "profit_factor": sr.get("test_metrics", {}).get("profit_factor", 0),
            "win_rate": sr.get("test_metrics", {}).get("win_rate", 0),
            "total_pnl": sr.get("test_metrics", {}).get("total_pnl", 0),
            "clears_significance": sr["test_sharpe"] >= sig_threshold,
        })

    failure_summary = {reason: len(names) for reason, names in failed_by_reason.items()}

    leaderboard_json = {
        "validated_count": n_validated,
        "total_strategies": len(all_strategy_results),
        "significance_threshold": round(sig_threshold, 4),
        "failure_summary": failure_summary,
        "validated_strategies": leaderboard_entries,
        "failed_strategies": {reason: sorted(names)
                              for reason, names in failed_by_reason.items()},
    }

    # ── Build leaderboard Markdown ──────────────────────────────────────
    md_lines = []
    md_lines.append("# Strategy Leaderboard")
    md_lines.append("")
    md_lines.append(f"**Total strategies:** {len(all_strategy_results)} | "
                    f"**Validated:** {n_validated} | "
                    f"**Significance threshold Sharpe:** {sig_threshold:.2f}")
    md_lines.append("")

    if validated:
        md_lines.append("## Validated Strategies (ranked by Test Sharpe × √passing_stocks)")
        md_lines.append("")
        md_lines.append("| Rank | Strategy | Family | Score | Test Sharpe | Train Sharpe | Stocks | Trades | PF | PnL | Sig? |")
        md_lines.append("|------|----------|--------|-------|-------------|--------------|--------|--------|----|-----|------|")
        for e in leaderboard_entries:
            sig = "✓" if e["clears_significance"] else "✗"
            md_lines.append(
                f"| {e['rank']} | {e['name']} | {e['family']} | {e['score']:.2f} | "
                f"{e['test_sharpe']:.2f} | {e['train_sharpe']:.2f} | {e['passing_stocks']} | "
                f"{e['total_trades']} | {e['profit_factor']:.2f} | ₹{e['total_pnl']:,.0f} | {sig} |"
            )
        md_lines.append("")

    md_lines.append("## Failure Breakdown")
    md_lines.append("")
    for reason, count in sorted(failure_summary.items(), key=lambda x: -x[1]):
        md_lines.append(f"- **{reason}:** {count}")
    md_lines.append("")

    leaderboard_md = "\n".join(md_lines)

    # ── Winning strategies JSON ─────────────────────────────────────────
    winning_json = {
        "count": n_validated,
        "significance_threshold": round(sig_threshold, 4),
        "strategies": leaderboard_entries,
    }

    # ── Family summary ──────────────────────────────────────────────────
    family_stats = {}
    for sr in all_strategy_results:
        family = sr.get("family", "unknown")
        if family not in family_stats:
            family_stats[family] = {
                "total": 0, "validated": 0, "failed": 0,
                "verdicts": {},
                "validated_names": [],
            }
        family_stats[family]["total"] += 1
        verdict = sr.get("verdict", "FAILED_ERROR")
        family_stats[family]["verdicts"][verdict] = family_stats[family]["verdicts"].get(verdict, 0) + 1
        if verdict == "VALIDATED":
            family_stats[family]["validated"] += 1
            family_stats[family]["validated_names"].append(sr["name"])
        else:
            family_stats[family]["failed"] += 1

    family_summary = {
        "total_families": len(family_stats),
        "families": family_stats,
    }

    return leaderboard_json, leaderboard_md, winning_json, family_summary


def build_master_signals(
    validated_strategies: list[dict],
) -> Optional[pl.DataFrame]:
    """Combine all test signals from validated strategies into master_signals.parquet."""
    all_signals = []

    for sr in validated_strategies:
        signals_path = RESULTS_DIR / sr["name"] / "test" / "signals.parquet"
        if signals_path.exists():
            try:
                df = pl.read_parquet(signals_path)
                all_signals.append(df)
            except Exception as e:
                log.warning("Failed to read signals for %s: %s", sr["name"], e)

    if not all_signals:
        return None

    return pl.concat(all_signals).sort("timestamp")
