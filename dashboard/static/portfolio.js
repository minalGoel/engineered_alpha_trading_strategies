/**
 * portfolio.js — Portfolio tab (DSR summary, cost, kills, correlation heatmap).
 * Depends on: core.js (state, api, tip, fmt*, colorNum)
 */

// ── PORTFOLIO TAB ────────────────────────────────────────────────────────
async function loadPortfolio() {
  const el = document.getElementById('tab-portfolio');
  el.innerHTML = '<div class="loading">Loading portfolio data\u2026</div>';
  try {
    const data = await api('/api/portfolio');
    state.portfolio = data;
    renderPortfolio(el, data);
  } catch(e) {
    el.innerHTML = '<div class="err">Error loading portfolio: ' + e.message + '</div>';
  }
}

function renderPortfolio(el, data) {
  const { corr_labels, corr_matrix, dsr_summary, kill_summary, cost_summary, effective_n, sr_benchmark, n_total_runs } = data;
  const killCount = kill_summary.filter(k=>k.kill_triggered).length;

  let html = '<div class="grid4" style="margin-bottom:16px">'
    + '<div class="stat"><div class="stat-label"' + tip('total_n') + '>Total Stored Runs (N)</div><div class="stat-value">' + n_total_runs + '</div></div>'
    + '<div class="stat"><div class="stat-label"' + tip('effective_n') + '>Effective N (rank-based)</div><div class="stat-value">' + effective_n + '</div></div>'
    + '<div class="stat"><div class="stat-label"' + tip('sr_benchmark') + '>E[max SR] Benchmark</div><div class="stat-value" style="color:var(--amber)">' + fmt4(sr_benchmark) + '</div></div>'
    + '<div class="stat"><div class="stat-label"' + tip('kill') + '>Kill Triggered</div>'
    + '<div class="stat-value neg">' + killCount + ' / ' + kill_summary.length + '</div></div></div>';

  // DSR Summary Table
  html += '<div class="card" style="margin-bottom:16px">'
    + '<div class="card-title">DSR Summary <span style="font-weight:400;color:var(--text2)">\u2014 primary "is there real signal?" view</span></div>'
    + '<div class="note" style="margin-bottom:10px">'
    + 'Benchmark: E[max SR | N_eff=' + effective_n + ', T] = ' + fmt4(sr_benchmark) + ' (annualized). Strategies must beat this to pass DSR.</div>'
    + '<div class="tbl-wrap"><table><thead><tr>'
    + '<th' + tip('strategy_id') + '>Strategy</th>'
    + '<th' + tip('total_n') + '>Raw N</th>'
    + '<th' + tip('effective_n') + '>Eff N</th>'
    + '<th' + tip('sr_benchmark') + '>E[max SR]</th>'
    + '<th' + tip('sharpe_raw') + '>Sharpe (raw)</th>'
    + '<th' + tip('dsr') + '>DSR</th>'
    + '<th' + tip('passes_dsr') + '>Passes DSR</th>'
    + '</tr></thead><tbody>';

  for (const s of dsr_summary) {
    html += '<tr>'
      + '<td style="cursor:pointer;font-weight:600" onclick="openStrategyOverlay(\'' + s.strategy_id + '\')">' + s.strategy_id + '</td>'
      + '<td>' + s.raw_n + '</td><td>' + s.effective_n + '</td>'
      + '<td style="color:var(--amber)">' + fmt4(s.sr_benchmark) + '</td>'
      + '<td>' + fmt4(s.sharpe_raw) + '</td>'
      + '<td style="' + colorNum(s.sharpe_deflated-0.5) + ';font-weight:600">' + fmt4(s.sharpe_deflated) + '</td>'
      + '<td>' + (s.passes_dsr ? '<span class="badge badge-green">PASS</span>' : '<span class="badge badge-red">FAIL</span>') + '</td></tr>';
  }
  html += '</tbody></table></div></div>';

  // Cost + Kill side-by-side
  html += '<div class="grid2" style="margin-bottom:16px">'
    + '<div class="card"><div class="card-title">Cost Summary <span style="font-weight:400;color:var(--text2)">(sorted by total cost)</span></div>'
    + '<div class="tbl-wrap"><table><thead><tr>'
    + '<th' + tip('strategy_id') + '>Strategy</th>'
    + '<th' + tip('total_cost') + '>Total Cost</th>'
    + '<th' + tip('net_edge_bps') + '>Net Edge</th>'
    + '</tr></thead><tbody>';
  for (const s of cost_summary.slice(0,20)) {
    html += '<tr><td>' + s.strategy_id + '</td>'
      + '<td style="color:var(--red)">' + fmtBps(s.total_cost_bps) + '</td>'
      + '<td style="' + colorNum(s.net_edge_bps) + '">' + fmtBps(s.net_edge_bps) + '</td></tr>';
  }
  html += '</tbody></table></div></div>'
    + '<div class="card"><div class="card-title">Kill Condition Status</div>'
    + '<div style="display:flex;flex-wrap:wrap;gap:8px">';
  for (const s of kill_summary) {
    const cls = s.kill_triggered ? 'badge-red' : 'badge-green';
    html += '<span class="badge ' + cls + '" style="cursor:pointer;padding:4px 10px" onclick="openStrategyOverlay(\'' + s.strategy_id + '\')"'
      + ' title="' + (s.kill_triggered ? 'Kill triggered' : 'OK') + '">' + s.strategy_id + '</span>';
  }
  html += '</div></div></div>';

  // Correlation Heatmap
  html += '<div class="card"><div class="card-title">Strategy Return Correlation Heatmap</div>';
  if (!corr_matrix) {
    html += '<div class="note">Not enough data to compute correlation matrix. Run backtests with --save to populate.</div>';
  } else {
    html += '<div class="heatmap-wrap"><table class="heatmap" id="heatmap-table"></table></div>';
  }
  html += '</div>';

  el.innerHTML = html;
  if (corr_matrix && corr_labels) _renderHeatmap(corr_labels, corr_matrix);
}

function _renderHeatmap(labels, matrix) {
  const tbl = document.getElementById('heatmap-table');
  if (!tbl) return;
  let html = '<tr><th></th>' + labels.map(l => '<th title="' + l + '">' + l.slice(0,8) + '</th>').join('') + '</tr>';
  for (let i=0; i<matrix.length; i++) {
    html += '<tr><th style="text-align:right;padding-right:6px" title="' + labels[i] + '">' + labels[i].slice(0,10) + '</th>';
    for (let j=0; j<matrix[i].length; j++) {
      const v = matrix[i][j];
      html += '<td style="background:' + _corrColor(v) + '" title="' + labels[i] + ' \u00d7 ' + labels[j] + ' = ' + v.toFixed(2) + '">' + v.toFixed(2) + '</td>';
    }
    html += '</tr>';
  }
  tbl.innerHTML = html;
}

function _corrColor(v) {
  if (v >= 0.8) return 'rgba(239,68,68,.6)';
  if (v >= 0.6) return 'rgba(245,158,11,.5)';
  if (v >= 0.4) return 'rgba(245,158,11,.3)';
  if (v >= 0.2) return 'rgba(59,130,246,.2)';
  if (v >= 0) return 'rgba(59,130,246,.08)';
  if (v >= -0.2) return 'rgba(34,197,94,.08)';
  return 'rgba(34,197,94,.25)';
}

// ── Init ─────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => { loadLeaderboard(); });
