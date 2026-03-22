/**
 * leaderboard.js — Leaderboard tab rendering + family expand/collapse.
 * Depends on: core.js (state, api, tip, fmt*, colorNum)
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

function _cvCard(name, v) {
  // BUG 2: show sensitivity status explicitly — all 6 happen to be FRAGILE
  const sensColor = v.sensitivity_verdict === 'ROBUST' ? 'var(--green)' : 'var(--amber)';
  const sensBadge = v.sensitivity_verdict === 'FRAGILE'
    ? '<span style="background:rgba(245,158,11,.15);color:var(--amber);border-radius:3px;padding:1px 5px;font-size:10px;font-weight:700">FRAGILE</span>'
    : v.sensitivity_verdict === 'ROBUST'
    ? '<span style="background:rgba(34,197,94,.15);color:var(--green);border-radius:3px;padding:1px 5px;font-size:10px;font-weight:700">ROBUST</span>'
    : '';
  return '<div style="background:var(--bg2);border:1px solid rgba(34,197,94,.4);border-radius:8px;padding:10px 14px;cursor:pointer;transition:transform .1s" onmouseenter="this.style.transform=\'scale(1.02)\'" onmouseleave="this.style.transform=\'scale(1)\'" onclick="openStrategyOverlay(\'' + name + '\')">'
    + '<div style="font-weight:700;font-size:12px;margin-bottom:4px;display:flex;align-items:center;gap:6px">' + name + sensBadge + '</div>'
    // BUG 1: clarify this is the DEFAULT (unoptimized) Sharpe, not the CV-fold Sharpe
    + '<div style="font-size:11px;color:var(--text2)">Default Sharpe: <span style="' + ((v.default_sharpe||0) < 0 ? 'color:var(--red)' : '') + '">' + fmt2(v.default_sharpe) + '</span>'
    + ' \u00b7 OOS: ' + v.cv_profitable_days + '/12 profitable days</div>'
    + '</div>';
}

function _robustChip(name, v) {
  return '<span style="background:var(--bg3);border:1px solid rgba(245,158,11,.3);border-radius:6px;padding:4px 10px;cursor:pointer;font-size:11px;transition:all .1s" onmouseenter="this.style.borderColor=\'var(--amber)\'" onmouseleave="this.style.borderColor=\'rgba(245,158,11,.3)\'" onclick="openStrategyOverlay(\'' + name + '\')">'
    + name + ' <span style="color:var(--text2)">' + fmt2(v.default_sharpe) + '</span></span>';
}

function renderLeaderboard(el, data) {
  const cvData = data.cv_data || {};
  const cvPassed = Object.entries(cvData).filter(function(e) { return e[1].cv_passed; })
    .sort(function(a,b) { return (b[1].default_sharpe||0) - (a[1].default_sharpe||0); });
  const robustOnly = Object.entries(cvData)
    .filter(function(e) { return !e[1].cv_passed && e[1].sensitivity_verdict === 'ROBUST'; })
    .sort(function(a,b) { return (b[1].default_sharpe||0) - (a[1].default_sharpe||0); });

  let banner = '';
  if (cvPassed.length > 0) {
    // BUG 2: note if all passed strategies are FRAGILE
    const allFragile = cvPassed.every(function(e) { return e[1].sensitivity_verdict === 'FRAGILE'; });
    const fragileNote = allFragile
      ? '<div style="margin-top:10px;padding:6px 10px;background:rgba(245,158,11,.1);border-radius:6px;font-size:11px;color:var(--amber)">'
        + '\u26a0 All ' + cvPassed.length + ' strategies are sensitivity-FRAGILE — Sharpe degrades under \u00b120% parameter perturbation. '
        + 'CV pass does not imply robustness. Trade with caution and use conservative position sizing.</div>'
      : '';
    banner = '<div style="background:rgba(34,197,94,.07);border:1px solid rgba(34,197,94,.25);border-radius:10px;padding:16px 20px;margin-bottom:16px">'
      + '<div style="font-size:12px;font-weight:700;color:var(--green);margin-bottom:10px;display:flex;align-items:center;gap:8px">'
      + '<span style="font-size:16px">\u2713</span> ' + cvPassed.length + ' STRATEGIES PASSED NESTED LOO-CV'
      + '<span style="font-weight:400;color:var(--text2);font-size:11px">(9/12+ profitable out-of-sample days, tested with OPTIMIZED params per fold)</span></div>'
      + '<div style="display:flex;flex-wrap:wrap;gap:8px">'
      + cvPassed.map(function(e) { return _cvCard(e[0], e[1]); }).join('')
      + '</div>' + fragileNote + '</div>';
  }
  if (robustOnly.length > 0) {
    // BUG 3: rename section to make clear these are NOT out-of-sample validated
    banner += '<div style="background:rgba(139,144,167,.05);border:1px solid rgba(139,144,167,.2);border-radius:10px;padding:14px 20px;margin-bottom:16px">'
      + '<div style="font-size:12px;font-weight:700;color:var(--text2);margin-bottom:4px;display:flex;align-items:center;gap:8px">'
      + '\u25a6 ' + robustOnly.length + ' IN-SAMPLE-ONLY strategies'
      + '<span style="font-weight:400;font-size:11px">(sensitivity-stable, but CV-FAILED \u2014 not out-of-sample validated)</span></div>'
      + '<div style="font-size:11px;color:var(--red);margin-bottom:8px">'
      + '\u26a0 These did NOT pass out-of-sample cross-validation. Do not trade. Showing top 10 of ' + robustOnly.length + '.</div>'
      + '<div style="display:flex;flex-wrap:wrap;gap:6px">'
      + robustOnly.slice(0,10).map(function(e) { return _robustChip(e[0], e[1]); }).join('')
      + '</div></div>';
  }

  let html = banner
    + '<div class="card" style="margin-bottom:16px">'
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

  for (const fam of data.families) {
    const famId = 'fam-' + fam.family.replace(/[^a-z0-9]/gi,'_');
    const cvN = fam.n_cv_passed || 0;
    const cvCell = cvN > 0 ? '<span class="badge badge-green">' + cvN + '</span>' : '\u2014';
    const vecCell = fam.n_flagged_vectorized > 0 ? '<span class="badge badge-yellow">' + fam.n_flagged_vectorized + '</span>' : '\u2014';
    const corrCell = fam.avg_intra_family_corr != null ? fmt2(fam.avg_intra_family_corr) : '\u2014';
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

    for (const s of (fam.strategies||[])) {
      const colorCls = 'row-' + (s.color||'red');
      const killBadge = s.kill_triggered ? '<span class="badge badge-red">KILL</span>' : '<span class="badge badge-green">OK</span>';
      const engineBadge = s.engine === 'vectorized' ? '<span class="badge badge-yellow">VEC</span>' : '<span class="badge badge-cyan">EVENT</span>';
      const cvBadge = s.cv_passed ? '<span class="badge badge-green">PASS</span>' : '<span class="badge badge-red">FAIL</span>';
      const sensBadge = s.sensitivity_verdict === 'ROBUST' ? '<span class="badge badge-green">ROBUST</span>'
        : s.sensitivity_verdict === 'FRAGILE' ? '<span class="badge badge-red">FRAGILE</span>'
        : '<span style="color:var(--text2)">\u2014</span>';
      let profCell = '\u2014';
      if (s.cv_profitable_days > 0) {
        const c = s.cv_profitable_days >= 9 ? 'var(--green)' : s.cv_profitable_days >= 7 ? 'var(--amber)' : 'var(--red)';
        profCell = '<span style="color:' + c + ';font-weight:600">' + s.cv_profitable_days + '/12</span>';
      }
      html += '<tr class="' + colorCls + '" style="cursor:pointer" onclick="openStrategyOverlay(\'' + s.strategy_id + '\')">'
        + '<td><input type="checkbox" class="compare-check" onclick="toggleCompare(event,\'' + s.strategy_id + '\')" title="Add to comparison"></td>'
        + '<td><b style="color:var(--text)">' + s.strategy_id + '</b></td>'
        + '<td style="' + colorNum(s.default_sharpe) + '">' + fmt2(s.default_sharpe) + '</td>'
        + '<td style="' + colorNum(s.net_edge_bps) + '">' + fmtBps(s.net_edge_bps) + '</td>'
        + '<td>' + cvBadge + '</td>'
        + '<td>' + profCell + '</td>'
        + '<td>' + sensBadge + '</td>'
        + '<td>' + killBadge + '</td>'
        + '<td>' + engineBadge + '</td>'
        + '</tr>';
    }
    html += '</tbody></table></td></tr>';
  }

  html += '</tbody></table></div></div>';
  el.innerHTML = html;
}

function toggleFamily(famId, famName, tr) {
  const row = document.getElementById(famId);
  const open = row.style.display === '';
  row.style.display = open ? 'none' : '';
  tr.classList.toggle('open', !open);
}

function sortLB(col) {
  if (!state.leaderboard) return;
  const fams = state.leaderboard.families;
  if (state.sortCol === col) state.sortDir *= -1; else { state.sortCol = col; state.sortDir = -1; }
  fams.sort((a,b) => state.sortDir * ((a[col]||0) - (b[col]||0)));
  renderLeaderboard(document.getElementById('tab-leaderboard'), state.leaderboard);
}
