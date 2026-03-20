"""Leaderboard generation for option trading strategies.

Rank by test Sharpe. Data-snooping correction with adjusted N.
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
) -> tuple[dict, str, dict]:
    """Build leaderboard from all strategy results.

    Returns:
        leaderboard_json, leaderboard_md, winning_strategies_json
    """
    validated = []
    failed_by_reason = {}

    for sr in all_strategy_results:
        verdict = sr.get("verdict", "FAILED_ERROR")
        if verdict == "VALIDATED":
            test_m = sr.get("test_metrics", {})
            sharpe = test_m.get("sharpe_annualized", 0.0)
            sr["test_sharpe"] = round(sharpe, 4)
            sr["score"] = round(sharpe, 4)  # rank purely by Sharpe
            validated.append(sr)
        else:
            failed_by_reason.setdefault(verdict, []).append(sr["name"])

    validated.sort(key=lambda x: x["score"], reverse=True)

    n_validated = len(validated)
    sig_threshold = required_sharpe(n_validated) if n_validated > 0 else 0.5

    # ── Build leaderboard JSON ──
    leaderboard_entries = []
    for rank, sr in enumerate(validated, 1):
        test_m = sr.get("test_metrics", {})
        leaderboard_entries.append({
            "rank": rank,
            "name": sr["name"],
            "underlying": sr.get("underlying", "BOTH"),
            "score": sr["score"],
            "test_sharpe": sr["test_sharpe"],
            "total_trades": test_m.get("total_trades", 0),
            "profit_factor": test_m.get("profit_factor", 0),
            "win_rate": test_m.get("win_rate", 0),
            "total_pnl": test_m.get("total_pnl", 0),
            "avg_points": test_m.get("avg_points_per_trade", 0),
            "clears_significance": sr["test_sharpe"] >= sig_threshold,
        })

    failure_summary = {reason: len(names) for reason, names in failed_by_reason.items()}

    leaderboard_json = {
        "validated_count": n_validated,
        "total_strategies": len(all_strategy_results),
        "significance_threshold": round(sig_threshold, 4),
        "data_limitation": "12 trading days only - results are preliminary",
        "failure_summary": failure_summary,
        "validated_strategies": leaderboard_entries,
        "failed_strategies": {reason: sorted(names)
                              for reason, names in failed_by_reason.items()},
    }

    # ── Build leaderboard Markdown ──
    md_lines = []
    md_lines.append("# Option Strategy Leaderboard")
    md_lines.append("")
    md_lines.append(f"**Total strategies:** {len(all_strategy_results)} | "
                    f"**Validated:** {n_validated} | "
                    f"**Significance threshold Sharpe:** {sig_threshold:.2f}")
    md_lines.append("")
    md_lines.append("> **WARNING:** Only 12 trading days of data. "
                    "These results are preliminary and must be validated with more data.")
    md_lines.append("")

    if validated:
        md_lines.append("## Validated Strategies (ranked by Test Sharpe)")
        md_lines.append("")
        md_lines.append("| Rank | Strategy | Underlying | Sharpe | Trades | PF | Avg Pts | PnL | Sig? |")
        md_lines.append("|------|----------|------------|--------|--------|----|---------|-----|------|")
        for e in leaderboard_entries:
            sig = "Y" if e["clears_significance"] else "N"
            md_lines.append(
                f"| {e['rank']} | {e['name']} | {e['underlying']} | "
                f"{e['test_sharpe']:.2f} | {e['total_trades']} | "
                f"{e['profit_factor']:.2f} | {e['avg_points']:.1f} | "
                f"INR {e['total_pnl']:,.0f} | {sig} |"
            )
        md_lines.append("")

    md_lines.append("## Failure Breakdown")
    md_lines.append("")
    for reason, count in sorted(failure_summary.items(), key=lambda x: -x[1]):
        md_lines.append(f"- **{reason}:** {count}")
    md_lines.append("")

    leaderboard_md = "\n".join(md_lines)

    # ── Winning strategies JSON ──
    winning_json = {
        "count": n_validated,
        "significance_threshold": round(sig_threshold, 4),
        "strategies": leaderboard_entries,
    }

    return leaderboard_json, leaderboard_md, winning_json
