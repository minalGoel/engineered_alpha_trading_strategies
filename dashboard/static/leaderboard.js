/**
 * leaderboard.js — Leaderboard tab rendering with 5-tier classification.
 * Depends on: core.js (state, api, tip, fmt*, colorNum, fmtPct, fmtBps)
 */

// ── LEADERBOARD TAB ──────────────────────────────────────────────────────
async function loadLeaderboard() {
  const el = document.getElementById('tab-leaderboard');
  el.innerHTML = '<div class="loading">Loading leaderboard\u2026</div>';
  try {
    const data = await api('/api/leaderboard');
    state.leaderboard = data;
    document.getElementById('run-count').textContent = data.n_total_runs + ' total runs';
    renderLeaderboard(el, data);
  } catch(e) {
    el.innerHTML = '<div class="err">Error loading leaderboard: ' + e.message + '</div>';
  }
}

// ── Tier Classification ──────────────────────────────────────────────────
function _classifyStrategies(cvData) {
  var tiers = { deploy: [], promising: [], watchlist: [], needsWork: [], noEdge: [] };
  var entries = Object.entries(cvData);
  for (var i = 0; i < entries.length; i++) {
    var name = entries[i][0], v = entries[i][1];
    var cvPass = v.cv_passed;
    var sens = v.sensitivity_verdict;
    var optSharpe = v.optimized_sharpe || v.default_sharpe || 0;

    if (cvPass && sens === 'ROBUST') {
      tiers.deploy.push([name, v]);
    } else if (cvPass) {
      tiers.promising.push([name, v]);
    } else if (sens === 'ROBUST' && optSharpe > 0) {
      tiers.watchlist.push([name, v]);
    } else if (optSharpe > 0) {
      tiers.needsWork.push([name, v]);
    } else {
      tiers.noEdge.push([name, v]);
    }
  }
  // Sort each tier by optimized Sharpe descending
  var byS = function(a, b) { return ((b[1].optimized_sharpe||0) - (a[1].optimized_sharpe||0)); };
  tiers.deploy.sort(byS);
  tiers.promising.sort(byS);
  tiers.watchlist.sort(byS);
  tiers.needsWork.sort(byS);
  tiers.noEdge.sort(byS);
  return tiers;
}

// ── Strategy Card (detailed — for top tiers) ─────────────────────────────
function _tierCard(name, v, borderColor) {
  var sharpe = v.optimized_sharpe || v.default_sharpe || 0;
  var edge = v.optimized_net_edge || 0;
  var wr = v.optimized_win_rate;
  var dd = v.optimized_max_dd || 0;
  var tpd = v.optimized_trades_per_day || 0;
  var days = v.cv_profitable_days || 0;
  var sensBadge = v.sensitivity_verdict === 'FRAGILE'
    ? '<span style="background:rgba(245,158,11,.15);color:var(--amber);border-radius:3px;padding:1px 5px;font-size:10px;font-weight:700">FRAGILE</span>'
    : v.sensitivity_verdict === 'ROBUST'
    ? '<span style="background:rgba(34,197,94,.15);color:var(--green);border-radius:3px;padding:1px 5px;font-size:10px;font-weight:700">ROBUST</span>'
    : '';

  return '<div style="background:var(--bg2);border:1px solid ' + borderColor + ';border-radius:8px;padding:10px 14px;cursor:pointer;transition:transform .1s;min-width:200px" '
    + 'onmouseenter="this.style.transform=\'scale(1.02)\'" onmouseleave="this.style.transform=\'scale(1)\'" '
    + 'onclick="openStrategyOverlay(\'' + name + '\')">'
    + '<div style="font-weight:700;font-size:12px;margin-bottom:6px;display:flex;align-items:center;gap:6px">' + name + ' ' + sensBadge + '</div>'
    + '<div style="display:grid;grid-template-columns:1fr 1fr;gap:2px 12px;font-size:11px">'
    + '<div style="color:var(--text2)">Sharpe: <span style="color:' + (sharpe > 0 ? 'var(--green)' : 'var(--red)') + ';font-weight:600">' + fmt2(sharpe) + '</span></div>'
    + '<div style="color:var(--text2)">Net Edge: <span style="' + colorNum(edge) + ';font-weight:600">' + fmtBps(edge) + '</span></div>'
    + '<div style="color:var(--text2)">Win Rate: ' + (wr != null ? fmtPct(wr) : '\u2014') + '</div>'
    + '<div style="color:var(--text2)">Max DD: <span style="color:var(--red)">\u20b9' + Math.round(dd).toLocaleString('en-IN') + '</span></div>'
    + '<div style="color:var(--text2)">OOS Days: <span style="color:' + (days >= 9 ? 'var(--green)' : 'var(--amber)') + '">' + days + '/12</span></div>'
    + '<div style="color:var(--text2)">Trades/day: ' + fmt2(tpd) + '</div>'
    + '</div></div>';
}

// ── Compact chip (for lower tiers) ───────────────────────────────────────
function _tierChip(name, v, borderColor) {
  var sharpe = v.optimized_sharpe || v.default_sharpe || 0;
  var edge = v.optimized_net_edge || 0;
  return '<span style="background:var(--bg2);border:1px solid ' + borderColor + ';border-radius:6px;padding:4px 10px;cursor:pointer;font-size:11px;transition:all .1s;display:inline-flex;align-items:center;gap:6px" '
    + 'onmouseenter="this.style.borderColor=\'var(--text)\'" onmouseleave="this.style.borderColor=\'' + borderColor + '\'" '
    + 'onclick="openStrategyOverlay(\'' + name + '\')">'
    + '<b>' + name + '</b>'
    + ' <span style="' + colorNum(sharpe) + '">' + fmt2(sharpe) + '</span>'
    + ' <span style="color:var(--text2);font-size:10px">' + fmtBps(edge) + '</span>'
    + '</span>';
}

// ── Main Render ──────────────────────────────────────────────────────────
function renderLeaderboard(el, data) {
  var cvData = data.cv_data || {};
  var tiers = _classifyStrategies(cvData);
  var dp = data.data_period || {};
  var nStrats = data.n_strategies_evaluated || 0;
  var dataStart = (dp.data_start || '').slice(0, 10);
  var dataEnd = (dp.data_end || '').slice(0, 10);

  // ── Global Banner ──────────────────────────────────────────────────────
  var html = '<div style="background:var(--bg2);border:1px solid var(--border);border-radius:8px;padding:12px 16px;margin-bottom:16px">'
    + '<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">'
    + '<div style="font-size:12px">'
    + '<span style="color:var(--text2)">Data:</span> <strong>' + dataStart + ' \u2192 ' + dataEnd + '</strong>'
    + ' &nbsp;\u00b7&nbsp; <span style="color:var(--text2)">Strategies evaluated:</span> <strong>' + nStrats + '</strong>'
    + ' &nbsp;\u00b7&nbsp; <span style="color:var(--text2)">Capital:</span> <strong>\u20b95L per entry</strong>'
    + '</div>'
    + '<div style="font-size:11px;color:var(--red);font-weight:700">'
    + '\u26a0 12 TRADING DAYS \u2014 all metrics are preliminary, not deployment-grade'
    + '</div></div></div>';

  // ── Tier 1: Ready to Paper Trade ───────────────────────────────────────
  if (tiers.deploy.length > 0) {
    html += _tierSection(
      '\ud83d\udfe2', 'READY TO PAPER TRADE', tiers.deploy.length + ' strategies',
      'rgba(34,197,94,.4)', 'rgba(34,197,94,.07)',
      'CV validated + parameter-ROBUST. Best candidates for live testing.',
      tiers.deploy.map(function(e) { return _tierCard(e[0], e[1], 'rgba(34,197,94,.5)'); }).join('')
    );
  } else {
    html += '<div style="background:rgba(34,197,94,.04);border:1px solid rgba(34,197,94,.15);border-radius:10px;padding:14px 20px;margin-bottom:12px">'
      + '<div style="font-size:12px;font-weight:700;color:var(--green);display:flex;align-items:center;gap:8px">'
      + '\ud83d\udfe2 READY TO PAPER TRADE <span style="font-weight:400;color:var(--text2);font-size:11px">\u2014 0 strategies</span></div>'
      + '<div style="font-size:11px;color:var(--text2);margin-top:4px">'
      + 'No strategies meet the full bar (CV validated + parameter-ROBUST). '
      + 'The 6 CV-passed strategies below are all sensitivity-FRAGILE.'
      + '</div></div>';
  }

  // ── Tier 2: Promising ──────────────────────────────────────────────────
  if (tiers.promising.length > 0) {
    html += _tierSection(
      '\ud83d\udfe1', 'PROMISING \u2014 VALIDATE FURTHER', tiers.promising.length + ' strategies',
      'rgba(245,158,11,.4)', 'rgba(245,158,11,.05)',
      'Edge confirmed out-of-sample (9/12+ profitable days) but parameters are sensitive to \u00b120% perturbation. Paper-trade with conservative sizing.',
      '<div style="display:flex;flex-wrap:wrap;gap:8px">'
      + tiers.promising.map(function(e) { return _tierCard(e[0], e[1], 'rgba(245,158,11,.4)'); }).join('')
      + '</div>'
    );
  }

  // ── Tier 3: Watch List ─────────────────────────────────────────────────
  if (tiers.watchlist.length > 0) {
    var showN = Math.min(tiers.watchlist.length, 15);
    html += _tierSection(
      '\ud83d\udd35', 'WATCH LIST', tiers.watchlist.length + ' strategies',
      'rgba(59,130,246,.3)', 'rgba(59,130,246,.04)',
      'Positive optimized edge + parameter-ROBUST, but failed out-of-sample CV. Signal exists but unproven on held-out data. Showing top ' + showN + '.',
      '<div style="display:flex;flex-wrap:wrap;gap:6px">'
      + tiers.watchlist.slice(0, showN).map(function(e) { return _tierChip(e[0], e[1], 'rgba(59,130,246,.3)'); }).join('')
      + '</div>'
    );
  }

  // ── Tier 4: Needs Work ─────────────────────────────────────────────────
  if (tiers.needsWork.length > 0) {
    var showM = Math.min(tiers.needsWork.length, 10);
    html += _tierSection(
      '\u26ab', 'NEEDS WORK', tiers.needsWork.length + ' strategies',
      'rgba(139,144,167,.2)', 'rgba(139,144,167,.03)',
      'Positive in-sample edge but both OOS validation and parameter stability fail. Needs more data, different approach, or regularization. Showing top ' + showM + '.',
      '<div style="display:flex;flex-wrap:wrap;gap:6px">'
      + tiers.needsWork.slice(0, showM).map(function(e) { return _tierChip(e[0], e[1], 'rgba(139,144,167,.2)'); }).join('')
      + '</div>'
    );
  }

  // ── Tier 5: No Edge ────────────────────────────────────────────────────
  if (tiers.noEdge.length > 0) {
    html += '<div style="background:rgba(239,68,68,.03);border:1px solid rgba(239,68,68,.15);border-radius:10px;padding:12px 20px;margin-bottom:12px">'
      + '<div style="font-size:12px;font-weight:700;color:var(--red);display:flex;align-items:center;gap:8px">'
      + '\ud83d\udd34 NO EDGE <span style="font-weight:400;color:var(--text2);font-size:11px">\u2014 '
      + tiers.noEdge.length + ' strategies</span></div>'
      + '<div style="font-size:11px;color:var(--text2);margin-top:4px">'
      + 'Negative or zero net edge even after optimization. No tradeable signal found \u2014 skip.'
      + '</div></div>';
  }

  // ── Family Table (unchanged, below tiers) ──────────────────────────────
  html += '<div class="card" style="margin-bottom:16px">'
    + '<div class="card-title">Strategy Families</div>'
    + '<div class="tbl-wrap"><table id="lb-table"><thead><tr>'
    + '<th' + tip('family') + '>Family</th>'
    + '<th' + tip('n_strategies') + ' onclick="sortLB(\'n_strategies\')"># Strategies</th>'
    + '<th' + tip('best_dsr') + ' onclick="sortLB(\'best_dsr\')">Best DSR \u25bc</th>'
    + '<th' + tip('best_net_edge') + ' onclick="sortLB(\'best_net_edge_bps\')">Best Net Edge</th>'
    + '<th' + tip('avg_corr') + ' onclick="sortLB(\'avg_intra_family_corr\')">Avg Corr</th>'
    + '<th' + tip('n_passing_kill') + ' onclick="sortLB(\'n_passing_kill\')"># Pass Kill</th>'
    + '<th' + tip('cv_pass') + ' onclick="sortLB(\'n_cv_passed\')">CV Pass</th>'
    + '<th' + tip('n_vectorized') + ' onclick="sortLB(\'n_flagged_vectorized\')"># Vec</th>'
    + '</tr></thead><tbody id="lb-body">';

  for (var fi = 0; fi < data.families.length; fi++) {
    var fam = data.families[fi];
    var famId = 'fam-' + fam.family.replace(/[^a-z0-9]/gi,'_');
    var cvN = fam.n_cv_passed || 0;
    var cvCell = cvN > 0 ? '<span class="badge badge-green">' + cvN + '</span>' : '\u2014';
    var vecCell = fam.n_flagged_vectorized > 0 ? '<span class="badge badge-yellow">' + fam.n_flagged_vectorized + '</span>' : '\u2014';
    var corrCell = fam.avg_intra_family_corr != null ? fmt2(fam.avg_intra_family_corr) : '\u2014';
    html += '<tr class="family-row" onclick="toggleFamily(\'' + famId + '\',\'' + fam.family + '\',this)">'
      + '<td><b>' + fam.family + '</b></td>'
      + '<td>' + fam.n_strategies + '</td>'
      + '<td style="' + colorNum(fam.best_dsr-0.5) + ';font-weight:600">' + fmt4(fam.best_dsr) + '</td>'
      + '<td style="' + colorNum(fam.best_net_edge_bps) + '">' + fmtBps(fam.best_net_edge_bps) + '</td>'
      + '<td>' + corrCell + '</td>'
      + '<td>' + fam.n_passing_kill + ' / ' + fam.n_strategies + '</td>'
      + '<td>' + cvCell + '</td>'
      + '<td>' + vecCell + '</td>'
      + '</tr>'
      + '<tr id="' + famId + '" style="display:none"><td colspan="8" style="padding:0">'
      + '<table style="width:100%;background:rgba(15,17,23,.6)"><thead><tr>'
      + '<th style="width:32px"></th>'
      + '<th' + tip('strategy_id') + '>Strategy</th>'
      + '<th' + tip('sharpe_default') + '>Sharpe (default)</th>'
      + '<th' + tip('net_edge_bps') + '>Net Edge</th>'
      + '<th' + tip('cv') + '>CV</th>'
      + '<th' + tip('prof_days') + '>Prof Days</th>'
      + '<th' + tip('sensitivity') + '>Sensitivity</th>'
      + '<th' + tip('kill') + '>Kill</th>'
      + '<th' + tip('engine') + '>Engine</th>'
      + '</tr></thead><tbody>';

    for (var si = 0; si < (fam.strategies||[]).length; si++) {
      var s = fam.strategies[si];
      var colorCls = 'row-' + (s.color||'red');
      var killBadge = s.kill_triggered ? '<span class="badge badge-red">KILL</span>' : '<span class="badge badge-green">OK</span>';
      var engineBadge = s.engine === 'vectorized' ? '<span class="badge badge-yellow">VEC</span>' : '<span class="badge badge-cyan">EVENT</span>';
      var cvBadge2 = s.cv_passed ? '<span class="badge badge-green">PASS</span>' : '<span class="badge badge-red">FAIL</span>';
      var sensBadge2 = s.sensitivity_verdict === 'ROBUST' ? '<span class="badge badge-green">ROBUST</span>'
        : s.sensitivity_verdict === 'FRAGILE' ? '<span class="badge badge-red">FRAGILE</span>'
        : '<span style="color:var(--text2)">\u2014</span>';
      var profCell = '\u2014';
      if (s.cv_profitable_days > 0) {
        var c = s.cv_profitable_days >= 9 ? 'var(--green)' : s.cv_profitable_days >= 7 ? 'var(--amber)' : 'var(--red)';
        profCell = '<span style="color:' + c + ';font-weight:600">' + s.cv_profitable_days + '/12</span>';
      }
      html += '<tr class="' + colorCls + '" style="cursor:pointer" onclick="openStrategyOverlay(\'' + s.strategy_id + '\')">'
        + '<td><input type="checkbox" class="compare-check" onclick="toggleCompare(event,\'' + s.strategy_id + '\')" title="Add to comparison"></td>'
        + '<td><b style="color:var(--text)">' + s.strategy_id + '</b></td>'
        + '<td style="' + colorNum(s.default_sharpe) + '">' + fmt2(s.default_sharpe) + '</td>'
        + '<td style="' + colorNum(s.net_edge_bps) + '">' + fmtBps(s.net_edge_bps) + '</td>'
        + '<td>' + cvBadge2 + '</td>'
        + '<td>' + profCell + '</td>'
        + '<td>' + sensBadge2 + '</td>'
        + '<td>' + killBadge + '</td>'
        + '<td>' + engineBadge + '</td>'
        + '</tr>';
    }
    html += '</tbody></table></td></tr>';
  }

  html += '</tbody></table></div></div>';
  el.innerHTML = html;
}

// ── Tier section helper ──────────────────────────────────────────────────
function _tierSection(icon, title, count, borderColor, bgColor, description, content) {
  return '<div style="background:' + bgColor + ';border:1px solid ' + borderColor + ';border-radius:10px;padding:14px 20px;margin-bottom:12px">'
    + '<div style="font-size:12px;font-weight:700;margin-bottom:6px;display:flex;align-items:center;gap:8px">'
    + icon + ' ' + title
    + ' <span style="font-weight:400;color:var(--text2);font-size:11px">\u2014 ' + count + '</span></div>'
    + '<div style="font-size:11px;color:var(--text2);margin-bottom:10px">' + description + '</div>'
    + content + '</div>';
}

function toggleFamily(famId, famName, tr) {
  var row = document.getElementById(famId);
  var open = row.style.display === '';
  row.style.display = open ? 'none' : '';
  tr.classList.toggle('open', !open);
}

function sortLB(col) {
  if (!state.leaderboard) return;
  var fams = state.leaderboard.families;
  if (state.sortCol === col) state.sortDir *= -1; else { state.sortCol = col; state.sortDir = -1; }
  fams.sort(function(a,b) { return state.sortDir * ((a[col]||0) - (b[col]||0)); });
  renderLeaderboard(document.getElementById('tab-leaderboard'), state.leaderboard);
}
