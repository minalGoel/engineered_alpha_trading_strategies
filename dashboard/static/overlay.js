/**
 * overlay.js — Strategy detail overlay modal + all detail tabs
 * (Overview, CV Analysis, Features, Backtest, Trade Log).
 * Depends on: core.js, Chart.js
 */

// ── Strategy Overlay ─────────────────────────────────────────────────────
async function openStrategyOverlay(strategyId) {
  state.selectedStrategy = strategyId;
  state.detailTab = 'overview';
  const overlay = document.getElementById('strategy-overlay');
  overlay.innerHTML = '<div class="overlay-backdrop" onclick="closeOverlayOnBackdrop(event)">'
    + '<div class="overlay-panel">'
    + '<div class="overlay-header"><h2>' + strategyId + '</h2><div class="loading" style="padding:8px">Loading\u2026</div></div>'
    + '</div></div>';

  try {
    const [detail, runs] = await Promise.all([
      api('/api/strategy/' + strategyId),
      api('/api/strategy/' + strategyId + '/runs'),
    ]);
    state.strategyDetail = detail;
    state.strategyRuns = runs;
    renderOverlayContent(detail, runs);
  } catch(e) {
    overlay.innerHTML = '<div class="overlay-backdrop" onclick="closeOverlayOnBackdrop(event)">'
      + '<div class="overlay-panel"><div class="overlay-header"><h2>' + strategyId + '</h2>'
      + '<button class="overlay-close" onclick="closeOverlay()">\u00d7</button></div>'
      + '<div class="overlay-body"><div class="err">' + e.message + '</div></div></div></div>';
  }
}

function renderOverlayContent(detail, runs) {
  const sid = detail.strategy_id;
  const spec = detail.strategy_spec || {};
  const runList = (runs.runs || []);
  const overlay = document.getElementById('strategy-overlay');

  let headerBadges = '<span class="badge badge-blue">' + (detail.family||'') + '</span>';
  if (spec.underlying) headerBadges += ' <span class="badge badge-purple">' + spec.underlying + '</span>';
  if (spec.timeframe) headerBadges += ' <span class="badge badge-cyan">' + spec.timeframe + '</span>';
  headerBadges += ' <span style="font-size:11px;color:var(--text2)">' + runList.length + ' run(s)</span>';

  overlay.innerHTML = '<div class="overlay-backdrop" onclick="closeOverlayOnBackdrop(event)">'
    + '<div class="overlay-panel" onclick="event.stopPropagation()">'
    + '<div class="overlay-header">'
    + '<h2>' + sid + '</h2>' + headerBadges
    + '<button class="overlay-close" onclick="closeOverlay()" title="Close (Esc)">\u00d7</button>'
    + '</div>'
    + '<div class="overlay-tabs">'
    + '<button class="overlay-tab active" onclick="switchOverlayTab(\'overview\',\'' + sid + '\')">Overview</button>'
    + '<button class="overlay-tab" onclick="switchOverlayTab(\'cv\',\'' + sid + '\')">CV Analysis</button>'
    + '<button class="overlay-tab" onclick="switchOverlayTab(\'features\',\'' + sid + '\')">Features & Params</button>'
    + '<button class="overlay-tab" onclick="switchOverlayTab(\'backtest\',\'' + sid + '\')">Backtest Results</button>'
    + '<button class="overlay-tab" onclick="switchOverlayTab(\'trades\',\'' + sid + '\')">Trade Log</button>'
    + '</div>'
    + '<div class="overlay-body" id="overlay-content"></div>'
    + '</div></div>';

  renderOverlayTab('overview', detail, runs);
}

function closeOverlay() {
  document.getElementById('strategy-overlay').innerHTML = '';
  Object.keys(state.charts).forEach(k => { if (state.charts[k]) { state.charts[k].destroy(); delete state.charts[k]; } });
}

function closeOverlayOnBackdrop(event) {
  if (event.target.classList.contains('overlay-backdrop')) closeOverlay();
}

document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') closeOverlay();
});

function switchOverlayTab(tab, sid) {
  state.detailTab = tab;
  document.querySelectorAll('.overlay-tab').forEach(function(b) {
    const match = { overview:'overview', cv:'cv', features:'features', backtest:'backtest', trades:'trade' };
    b.classList.toggle('active', b.textContent.trim().toLowerCase().startsWith(match[tab] || tab));
  });
  renderOverlayTab(tab, state.strategyDetail, state.strategyRuns);
}

async function renderOverlayTab(tab, detail, runs) {
  const el = document.getElementById('overlay-content');
  if (!el) return;
  if (tab === 'overview') renderOverviewTab(el, detail, runs);
  else if (tab === 'cv') await renderCVAnalysisTab(el, detail);
  else if (tab === 'features') renderFeaturesTab(el, detail, runs);
  else if (tab === 'backtest') await renderBacktestTab(el, detail, runs);
  else if (tab === 'trades') await renderTradesTab(el, detail, runs);
}

// Legacy alias
function loadStrategyDetail(strategyId) { openStrategyOverlay(strategyId); }

// ── Tab: Overview ────────────────────────────────────────────────────────
function renderOverviewTab(el, detail, runs) {
  const spec = detail.strategy_spec || {};
  const runList = runs.runs || [];
  const latest = runList[0] || {};
  const res = latest.result_summary || {};

  const sharpe = res.sharpe_raw || 0;
  const dsr = res.sharpe_deflated || 0;
  const netEdge = res.net_edge_bps || 0;
  const isVec = (latest.backtest_engine || '') === 'vectorized';
  const costVer = latest.cost_model_version || detail.cost_model_version || '?';
  const isCurrent = costVer.includes('apr_2025') || costVer.includes('post');

  let vecBanner = isVec ? '<div class="vec-banner">!! SUSPECT \u2014 vectorized backtest, no fill model. Do not use for promotion decisions.</div>' : '';
  let descBlock = '';
  if (spec.description) {
    descBlock = '<div class="strategy-desc">' + _escHtml(spec.description) + '</div>';
  }

  const gross = res.gross_edge_bps || 0;
  const fees = res.fees_cost_bps || 0;
  const net = res.net_edge_bps || 0;
  const total = Math.max(Math.abs(gross) + Math.abs(fees), 1);
  const grossPct = (Math.abs(gross)/total*100).toFixed(1);
  const feesPct = (Math.abs(fees)/total*100).toFixed(1);
  const netPct = (Math.abs(net)/total*100).toFixed(1);
  const netCls = net >= 0 ? 'cost-bar-net-pos' : 'cost-bar-net-neg';

  let specRows = '';
  const specFields = [
    ['Underlying', spec.underlying],
    ['Timeframe', spec.timeframe],
    ['Session', (spec.session_start_minutes ? _minsToTime(spec.session_start_minutes) + ' \u2013 ' + _minsToTime(spec.session_end_minutes) : null)],
    ['Max Trades/Day', spec.max_trades_per_day],
    ['Max Lookback', spec.max_lookback ? spec.max_lookback + ' bars (' + (spec.max_lookback*5/60).toFixed(1) + ' min)' : null],
    ['Assumptions', Array.isArray(spec.assumptions) && spec.assumptions.length > 0 ? spec.assumptions.join('; ') : null],
  ];
  for (const [label, val] of specFields) {
    if (val != null) specRows += '<tr><td>' + label + '</td><td>' + val + '</td></tr>';
  }

  let tunableHtml = '';
  if (spec.tunable_params && spec.tunable_params.length > 0) {
    tunableHtml = '<div style="margin-top:12px"><div class="card-title">Tunable Parameters</div>'
      + '<table class="spec-table"><thead><tr><th>Name</th><th>Default</th><th>Range</th></tr></thead><tbody>';
    for (const tp of spec.tunable_params) {
      tunableHtml += '<tr><td><code>' + tp.name + '</code></td><td>' + fmt4(tp.default) + '</td>'
        + '<td style="color:var(--text2)">[' + fmt4(tp.low) + ', ' + fmt4(tp.high) + ']</td></tr>';
    }
    tunableHtml += '</tbody></table></div>';
  }

  el.innerHTML = vecBanner + descBlock
    + '<div class="grid4" style="margin-bottom:16px">'
    + '<div class="stat"><div class="stat-label"' + tip('dsr') + '>DSR (deflated)</div>'
    + '<div class="stat-value ' + (dsr>0.5?'pos':dsr<0.5?'neg':'') + '">' + fmt4(dsr) + '</div></div>'
    + '<div class="stat"><div class="stat-label"' + tip('sharpe_raw') + '>Sharpe (raw)</div>'
    + '<div class="stat-value ' + (sharpe>0.5?'pos':'') + '">' + fmt4(sharpe) + '</div></div>'
    + '<div class="stat"><div class="stat-label"' + tip('net_edge_bps') + '>Net Edge</div>'
    + '<div class="stat-value ' + (netEdge>0?'pos':'neg') + '">' + fmtBps(netEdge) + '</div></div>'
    + '<div class="stat"><div class="stat-label"' + tip('win_rate') + '>Win Rate</div>'
    + '<div class="stat-value">' + fmtPct(res.win_rate) + '</div></div>'
    + '</div>'
    + '<div class="card" style="margin-bottom:16px">'
    + '<div class="card-title">Strategy Specification</div>'
    + '<table class="spec-table">' + specRows + '</table>'
    + tunableHtml + '</div>'
    + '<div class="card" style="margin-bottom:16px">'
    + '<div class="card-title">Cost Breakdown <span style="font-weight:400;color:var(--text2)">(avg per trade in bps)</span></div>'
    + '<div style="margin-bottom:8px"><div class="cost-bar">'
    + '<div class="cost-bar-seg cost-bar-gross" style="width:' + grossPct + '%">Gross ' + fmt2(gross) + '</div>'
    + '<div class="cost-bar-seg cost-bar-fees" style="width:' + feesPct + '%">Fees ' + fmt2(fees) + '</div>'
    + '<div class="cost-bar-seg ' + netCls + '" style="width:' + netPct + '%">Net ' + fmt2(net) + '</div>'
    + '</div></div>'
    + '<div style="font-size:11px;color:var(--text2)">'
    + 'Spread cost: 0 bps (limit orders at close) &nbsp;|&nbsp; Slippage: 0 bps (no slippage model) &nbsp;|&nbsp; '
    + 'Cost model: <span style="' + (isCurrent?'color:var(--green)':'color:var(--yellow)') + '">' + costVer + '</span>'
    + (isCurrent ? '' : ' <span class="badge badge-yellow">NOT CURRENT</span>')
    + '</div></div>';
}

// ── Tab: CV Analysis ─────────────────────────────────────────────────────
async function renderCVAnalysisTab(el, detail) {
  el.innerHTML = '<div class="loading">Loading CV data\u2026</div>';
  const sid = detail.strategy_id;
  let foldData, sensData;
  try {
    [foldData, sensData] = await Promise.all([
      api('/api/strategy/' + sid + '/cv-folds'),
      api('/api/strategy/' + sid + '/sensitivity-details'),
    ]);
  } catch(e) { el.innerHTML = '<div class="err">Error loading CV data: ' + e.message + '</div>'; return; }

  const folds = foldData.folds || [];
  const cvPassed = foldData.cv_passed;
  const profDays = foldData.cv_profitable_days || foldData.n_profitable_folds || 0;
  const sensVerdict = foldData.sensitivity_verdict || sensData.sensitivity_verdict;
  const ds = foldData.default_stats || {};
  const sensRuns = sensData.sensitivity_runs || [];
  const baseline = sensData.baseline || {};

  const verdictColor = cvPassed ? 'var(--green)' : 'var(--red)';
  const sensColor = sensVerdict === 'ROBUST' ? 'var(--green)' : sensVerdict === 'FRAGILE' ? 'var(--red)' : 'var(--text2)';
  const profDayColor = profDays >= 9 ? 'var(--green)' : profDays >= 7 ? 'var(--amber)' : 'var(--red)';
  const sharpeClass = (ds.sharpe || 0) > 0 ? 'pos' : 'neg';

  const foldRows = folds.map(function(f) {
    const rowStyle = (f.net_edge_bps||0) > 0 ? 'background:rgba(34,197,94,.04)' : '';
    const hold = f.avg_hold_seconds ? (f.avg_hold_seconds/60).toFixed(1)+'min' : '\u2014';
    const killCell = f.kill_triggered ? '<span class="badge badge-red">KILL</span>' : '<span class="badge badge-green">OK</span>';
    return '<tr style="' + rowStyle + '">'
      + '<td>#' + f.fold + '</td>'
      + '<td style="font-size:11px">' + (f.test_date||'\u2014') + '</td>'
      + '<td>' + (f.trades||0) + '</td>'
      + '<td style="' + colorNum(f.net_edge_bps) + ';font-weight:600">' + fmtBps(f.net_edge_bps) + '</td>'
      + '<td style="' + colorNum(f.gross_edge_bps) + '">' + fmtBps(f.gross_edge_bps) + '</td>'
      + '<td>' + fmtPct(f.win_rate) + '</td>'
      + '<td style="' + colorNum(f.sharpe) + '">' + fmt2(f.sharpe) + '</td>'
      + '<td>' + hold + '</td>'
      + '<td>' + killCell + '</td>'
      + '</tr>';
  }).join('');

  const sensRows = sensRuns.map(function(r) {
    const delta = r.sharpe_delta || 0;
    return '<tr>'
      + '<td><code>' + r.param + '</code></td>'
      + '<td><span class="badge badge-blue">' + r.direction + '</span></td>'
      + '<td style="' + colorNum(r.sharpe) + '">' + fmt2(r.sharpe) + '</td>'
      + '<td style="' + colorNum(delta) + ';font-weight:600">' + (delta >= 0 ? '+' : '') + fmt2(delta) + '</td>'
      + '<td style="' + colorNum(r.net_edge_bps) + '">' + fmtBps(r.net_edge_bps) + '</td>'
      + '<td>' + (r.total_trades||0) + '</td>'
      + '<td>' + fmtPct(r.win_rate) + '</td>'
      + '</tr>';
  }).join('');

  const sensBody = sensRuns.length === 0
    ? '<div class="note">No sensitivity runs stored for this strategy.</div>'
    : '<div style="margin-bottom:10px;font-size:11px;color:var(--text2)">'
      + 'Baseline Sharpe: <b>' + fmt2(baseline.sharpe) + '</b> \u00b7 Net Edge: <b>' + fmtBps(baseline.net_edge_bps) + '</b><br>'
      + 'Each bar = Sharpe change when one parameter is perturbed \u00b120%.</div>'
      + '<div class="chart-wrap"><canvas id="c-sens-chart"></canvas></div>'
      + '<div class="tbl-wrap" style="margin-top:12px"><table><thead><tr>'
      + '<th' + tip('sens_param') + '>Parameter</th><th' + tip('sens_dir') + '>Direction</th>'
      + '<th' + tip('sharpe_raw') + '>Sharpe</th><th' + tip('sharpe_delta') + '>Sharpe \u0394</th>'
      + '<th' + tip('net_edge_bps') + '>Net Edge</th><th' + tip('trades') + '>Trades</th>'
      + '<th' + tip('win_rate') + '>Win Rate</th></tr></thead>'
      + '<tbody>' + sensRows + '</tbody></table></div>';

  el.innerHTML = '<div class="grid4" style="margin-bottom:16px">'
    + '<div class="stat"><div class="stat-label"' + tip('cv') + '>CV Verdict</div>'
    + '<div class="stat-value" style="color:' + verdictColor + '">' + (cvPassed ? '\u2713 PASS' : '\u2717 FAIL') + '</div></div>'
    + '<div class="stat"><div class="stat-label"' + tip('prof_days') + '>Profitable Days (OOS)</div>'
    + '<div class="stat-value" style="color:' + profDayColor + '">' + profDays + ' / 12</div></div>'
    + '<div class="stat"><div class="stat-label"' + tip('sensitivity') + '>Sensitivity</div>'
    + '<div class="stat-value" style="color:' + sensColor + '">' + (sensVerdict || '\u2014') + '</div></div>'
    + '<div class="stat"><div class="stat-label"' + tip('sharpe_default') + '>Default Sharpe (in-sample)</div>'
    + '<div class="stat-value ' + sharpeClass + '">' + fmt2(ds.sharpe) + '</div></div></div>'
    + '<div class="note" style="margin-bottom:16px"><b>Nested LOO-CV:</b> Each fold optimizes on 11 training days, tests on 1 held-out day. '
    + 'Pass = \u22659/12 profitable out-of-sample days (net_edge_bps &gt; 0).</div>'
    + '<div class="card" style="margin-bottom:16px"><div class="card-title">Out-of-Sample Net Edge per Fold (bps)</div>'
    + '<div class="chart-wrap"><canvas id="c-fold-pnl"></canvas></div>'
    + '<div class="tbl-wrap" style="margin-top:12px"><table><thead><tr>'
    + '<th' + tip('fold') + '>Fold</th><th' + tip('test_date') + '>Test Date</th>'
    + '<th' + tip('trades') + '>Trades</th><th' + tip('net_edge_bps') + '>Net Edge</th>'
    + '<th' + tip('gross_edge') + '>Gross Edge</th><th' + tip('win_rate') + '>Win Rate</th>'
    + '<th' + tip('sharpe_raw') + '>Sharpe</th><th' + tip('avg_hold') + '>Avg Hold</th>'
    + '<th' + tip('kill') + '>Kill</th></tr></thead>'
    + '<tbody>' + foldRows + '</tbody></table></div></div>'
    + '<div class="card"><div class="card-title">Sensitivity Analysis \u2014 Sharpe \u0394 vs default params'
    + '<span style="margin-left:8px;font-weight:400;color:' + sensColor + '">' + (sensVerdict ? '[' + sensVerdict + ']' : '') + '</span></div>'
    + sensBody + '</div>';

  _renderFoldChart(folds);
  if (sensRuns.length > 0) _renderSensChart(sensRuns);
}

// ── Tab: Features & Parameters ───────────────────────────────────────────
function renderFeaturesTab(el, detail, runs) {
  const runList = runs.runs || [];
  const latest = runList[0] || {};
  const n_total = detail.n_total_runs || 1;
  const eff_n = detail.effective_n || n_total;
  const sr_bench = detail.sr_benchmark_annualized || 0;
  const features = detail.features || [];
  const params = detail.parameters || [];
  const latestRes = latest.result_summary || {};
  const dsrClass = (latestRes.sharpe_deflated||0) > 0.5 ? 'pos' : 'neg';

  let featuresHtml = features.length === 0
    ? '<div class="note">No feature records stored.</div>'
    : '<table><thead><tr><th' + tip('feat_name') + '>Name</th><th' + tip('feat_desc') + '>Description</th>'
      + '<th' + tip('feat_window') + '>Window</th><th' + tip('feat_source') + '>Source</th>'
      + '<th' + tip('feat_importance') + '>Importance</th></tr></thead><tbody>'
      + features.map(function(f) {
          return '<tr><td><code>' + f.feature_name + '</code></td>'
            + '<td style="max-width:200px">' + (f.feature_description||'\u2014') + '</td>'
            + '<td>' + (f.computation_window||'\u2014') + '</td>'
            + '<td>' + (f.data_source||'\u2014') + '</td>'
            + '<td>' + (f.importance_score != null ? fmt2(f.importance_score) : '\u2014') + '</td></tr>';
        }).join('') + '</tbody></table>';

  let paramsHtml = params.length === 0
    ? '<div class="note">No parameter records stored.</div>'
    : (function() {
        const hasOpt = params.some(function(p) { return p.is_optimized; });
        return '<table><thead><tr><th' + tip('param_name') + '>Name</th><th' + tip('param_value') + '>Value</th>'
          + '<th' + tip('param_type') + '>Type</th><th' + tip('param_optimized') + '>Optimized?</th></tr></thead><tbody>'
          + params.map(function(p) {
              const rowStyle = p.is_optimized ? ' style="background:rgba(245,158,11,.06)"' : '';
              const optCell = p.is_optimized ? '<span class="badge badge-amber">OPTIMIZED !!</span>' : 'No';
              return '<tr' + rowStyle + '><td><code>' + p.param_name + '</code></td>'
                + '<td>' + fmt4(p.param_value) + '</td>'
                + '<td><span class="badge badge-blue">' + (p.param_type||'other') + '</span></td>'
                + '<td>' + optCell + '</td></tr>';
            }).join('') + '</tbody></table>'
          + (hasOpt ? '<div class="note" style="margin-top:8px">!! Amber rows are optimized parameters \u2014 highest overfitting risk</div>' : '');
      })();

  el.innerHTML = '<div class="grid2" style="margin-bottom:16px">'
    + '<div class="card"><div class="card-title">Features</div>' + featuresHtml + '</div>'
    + '<div class="card"><div class="card-title">Parameters</div>' + paramsHtml + '</div></div>'
    + '<div class="card"><div class="card-title">DSR Analysis <span style="font-weight:400;color:var(--text2)">(multiple-testing corrected)</span></div>'
    + '<div class="dsr-panel"><div class="grid4" style="margin-bottom:12px">'
    + '<div class="stat"><div class="stat-label"' + tip('total_n') + '>Total Runs (N)</div><div class="stat-value">' + n_total + '</div></div>'
    + '<div class="stat"><div class="stat-label"' + tip('effective_n') + '>Effective N</div><div class="stat-value">' + eff_n + '</div></div>'
    + '<div class="stat"><div class="stat-label"' + tip('sr_benchmark') + '>E[max SR] Benchmark</div><div class="stat-value" style="color:var(--amber)">' + fmt4(sr_bench) + '</div></div>'
    + '<div class="stat"><div class="stat-label"' + tip('dsr') + '>DSR (must beat benchmark)</div><div class="stat-value ' + dsrClass + '">' + fmt4(latestRes.sharpe_deflated) + '</div></div></div>'
    + '<div class="note">DSR is P(true SR &gt; E[max SR | N_eff]). Passes if DSR &gt; 0.5. '
    + 'Benchmark is NOT zero \u2014 it is the expected max Sharpe from ' + eff_n + ' independent trials.</div></div></div>';
}

// ── Tab: Backtest Results ────────────────────────────────────────────────
async function renderBacktestTab(el, detail, runs) {
  el.innerHTML = '<div class="loading">Loading backtest data\u2026</div>';
  const runList = runs.runs || [];
  if (runList.length === 0) { el.innerHTML = '<div class="empty-state">No runs stored yet. Run with --save flag.</div>'; return; }
  const sid = detail.strategy_id;
  const latestRun = runList[0];
  const rid = latestRun.run_id;
  const isVec = (latestRun.backtest_engine || '') === 'vectorized';
  let resultsData;
  try { resultsData = await api('/api/strategy/' + sid + '/results'); }
  catch(e) { el.innerHTML = '<div class="err">Failed to load results: ' + e.message + '</div>'; return; }

  const results = resultsData.results || [];
  const fullResults = results.filter(r => r.run_id === rid);
  const latestRes = latestRun.result_summary || {};
  const sharpeRaw = latestRes.sharpe_raw || 0;
  const sharpeDefl = latestRes.sharpe_deflated || 0;
  const maxDD = latestRes.max_drawdown || 0;
  const winRate = latestRes.win_rate || 0;
  let vecBanner = isVec ? '<div class="vec-banner">!! SUSPECT \u2014 vectorized backtest. No fill model.</div>' : '';

  const optionRows = runList.map(function(r) {
    const sel = r.run_id === rid ? ' selected' : '';
    return '<option value="' + r.run_id + '"' + sel + '>' + r.run_id.slice(0,8) + '\u2026 \u2014 '
      + (r.created_at||'').slice(0,16) + ' \u2014 Sharpe: ' + fmt2(r.result_summary && r.result_summary.sharpe_raw) + '</option>';
  }).join('');
  const checkboxes = runList.slice(0,5).map(function(r) {
    return '<label style="font-size:11px"><input type="checkbox" onchange="runCompare()" class="cmp-run" data-rid="' + r.run_id + '"> ' + r.run_id.slice(0,6) + '\u2026</label>';
  }).join(' ');

  const foldRows = fullResults.map(function(r) {
    const killBadge = r.kill_condition_triggered ? '<span class="badge badge-red">KILL</span>' : '<span class="badge badge-green">OK</span>';
    return '<tr><td>' + r.fold_number + '</td><td><span class="badge badge-blue">' + r.split + '</span></td>'
      + '<td style="' + colorNum(r.sharpe_raw-0.5) + '">' + fmt4(r.sharpe_raw) + '</td>'
      + '<td style="' + colorNum(r.sharpe_deflated-0.5) + '">' + fmt4(r.sharpe_deflated) + '</td>'
      + '<td style="' + colorNum(r.net_edge_bps) + '">' + fmtBps(r.net_edge_bps) + '</td>'
      + '<td>' + fmtPct(r.win_rate) + '</td><td>' + (r.total_trades||0) + '</td><td>' + killBadge + '</td></tr>';
  }).join('');
  const foldTable = fullResults.length === 0 ? '<div class="note">No fold results stored.</div>'
    : '<table><thead><tr><th' + tip('fold') + '>Fold</th><th>Split</th><th' + tip('sharpe_raw') + '>Sharpe (raw)</th>'
      + '<th' + tip('dsr') + '>DSR</th><th' + tip('net_edge_bps') + '>Net Edge</th>'
      + '<th' + tip('win_rate') + '>Win Rate</th><th' + tip('trades') + '>Trades</th>'
      + '<th' + tip('kill') + '>Kill</th></tr></thead><tbody>' + foldRows + '</tbody></table>';
  const ddStr = maxDD.toLocaleString('en-IN', {maximumFractionDigits:0});

  el.innerHTML = vecBanner
    + '<div style="margin-bottom:14px;display:flex;gap:10px;align-items:center;flex-wrap:wrap">'
    + '<span style="color:var(--text2);font-size:12px;font-weight:600">Run:</span>'
    + '<select class="run-select" id="run-selector" onchange="loadBacktestRun(this.value,\'' + sid + '\')">' + optionRows + '</select>'
    + '<span style="font-size:11px;color:var(--text2)">Select 2+ for comparison:</span>' + checkboxes + '</div>'
    + '<div class="grid4" style="margin-bottom:16px">'
    + '<div class="stat"><div class="stat-label"' + tip('sharpe_raw') + '>Sharpe (raw)</div><div class="stat-value ' + (sharpeRaw > 0.5 ? 'pos' : '') + '">' + fmt4(sharpeRaw) + '</div></div>'
    + '<div class="stat"><div class="stat-label"' + tip('dsr') + '>DSR</div><div class="stat-value ' + (sharpeDefl > 0.5 ? 'pos' : 'neg') + '">' + fmt4(sharpeDefl) + '</div></div>'
    + '<div class="stat"><div class="stat-label"' + tip('max_drawdown') + '>Max Drawdown</div><div class="stat-value neg">\u20b9' + ddStr + '</div></div>'
    + '<div class="stat"><div class="stat-label"' + tip('win_rate') + '>Win Rate</div><div class="stat-value">' + fmtPct(winRate) + '</div></div></div>'
    + '<div class="card" style="margin-bottom:16px"><div class="card-title">Metrics by Fold</div>' + foldTable + '</div>'
    + '<div id="compare-section"></div>';
}

async function runCompare() {
  const boxes = document.querySelectorAll('.cmp-run:checked');
  const rids = Array.from(boxes).map(b => b.dataset.rid);
  if (rids.length < 2) { document.getElementById('compare-section').innerHTML = ''; return; }
  try {
    const cmp = await post('/api/compare', { run_ids: rids });
    renderComparison(document.getElementById('compare-section'), cmp, rids);
  } catch(e) { document.getElementById('compare-section').innerHTML = '<div class="err">' + e.message + '</div>'; }
}

function renderComparison(el, cmp, rids) {
  const rows = cmp.comparison || [];
  const diffs = cmp.param_diffs || [];
  const thCols = rids.map(function(r) { return '<th>' + r.slice(0,8) + '\u2026</th>'; }).join('');
  const metricKeys = ['sharpe_raw','sharpe_deflated','net_edge_bps','win_rate','max_drawdown','total_trades','cagr','gross_edge_bps','fees_cost_bps'];
  let tbody = '';
  for (const key of metricKeys) {
    const vals = rids.map(function(rid) { const r = rows.find(function(r){return r.run_id===rid;}); return r ? r[key] : null; });
    const allSame = vals.every(function(v){return v == vals[0];});
    tbody += '<tr><td style="color:var(--text2);font-weight:600">' + key + '</td>';
    for (const v of vals) { tbody += '<td style="' + (!allSame ? 'background:rgba(99,102,241,.1)' : '') + '">' + fmt4(v) + '</td>'; }
    tbody += '</tr>';
  }
  let tableHtml = rows.length === 0 ? '<div class="note">No data</div>'
    : '<div class="tbl-wrap"><table><thead><tr><th>Metric</th>' + thCols + '</tr></thead><tbody>' + tbody + '</tbody></table></div>';
  let diffHtml = '';
  if (diffs.length > 0) {
    const diffThCols = rids.map(function(r) { return '<th>' + r.slice(0,8) + '\u2026</th>'; }).join('');
    let diffRows = '';
    for (const d of diffs) {
      diffRows += '<tr><td><code>' + d.param_name + '</code></td>';
      for (const rid of rids) { diffRows += '<td style="background:rgba(245,158,11,.08)">' + fmt4(d.values[rid]) + '</td>'; }
      diffRows += '</tr>';
    }
    diffHtml = '<div class="card-title" style="margin-top:14px">Parameter Differences</div>'
      + '<table><thead><tr><th>Parameter</th>' + diffThCols + '</tr></thead><tbody>' + diffRows + '</tbody></table>';
  }
  el.innerHTML = '<div class="card" style="margin-bottom:16px"><div class="card-title">Run Comparison (' + rids.length + ' runs)</div>' + tableHtml + diffHtml + '</div>';
}

// ── Tab: Trade Log ───────────────────────────────────────────────────────
async function renderTradesTab(el, detail, runs) {
  el.innerHTML = '<div class="loading">Loading trades\u2026</div>';
  const sid = detail.strategy_id;
  const runList = runs.runs || [];
  if (runList.length === 0) { el.innerHTML = '<div class="empty-state">No runs stored yet.</div>'; return; }
  const rid = runList[0].run_id;
  state.tradeFilters.page = 1;
  state.tradeFilters._sid = sid;
  state.tradeFilters._rid = rid;
  await _loadAndRenderTrades(el, sid, rid);
}

async function _loadAndRenderTrades(el, sid, rid) {
  const f = state.tradeFilters;
  let url = '/api/strategy/' + sid + '/trades/' + rid + '?page=' + f.page + '&page_size=' + f.page_size;
  if (f.direction) url += '&direction=' + f.direction;
  if (f.exit_reason) url += '&exit_reason=' + f.exit_reason;
  try { const data = await api(url); state.trades = data; renderTradeTable(el, data, sid, rid); }
  catch(e) { el.innerHTML = '<div class="err">Failed to load trades: ' + e.message + '</div>'; }
}

function renderTradeTable(el, data, sid, rid) {
  const trades = data.trades || [];
  const total = data.total || 0;
  const page = data.page || 1;
  const pageSize = data.page_size || 100;
  const totalPages = Math.ceil(total / pageSize);
  const features = state.strategyDetail?.features || [];
  const featureNames = features.map(f => f.feature_name);
  const hasSignals = trades.some(t => t.entry_signal_values);

  let exitReasonOptions = '<option value="">All</option>';
  ['STOP','TARGET','SIGNAL','TIME','EOD'].forEach(r => {
    exitReasonOptions += '<option value="' + r + '"' + (state.tradeFilters.exit_reason === r ? ' selected' : '') + '>' + r + '</option>';
  });

  let tradeRows = '';
  trades.forEach(t => {
    const pnl = t.net_pnl || t.pnl || 0;
    const side = t.side === 1 ? '<span class="badge badge-blue">CE</span>' : '<span class="badge badge-purple">PE</span>';
    const hold = t.holding_bars != null ? (t.holding_bars * 5 / 60).toFixed(1) : '\u2014';
    tradeRows += '<tr>'
      + '<td style="font-size:11px">' + (t.entry_time||'').slice(0,19) + '</td>'
      + '<td style="font-size:11px">' + (t.exit_time||'').slice(0,19) + '</td>'
      + '<td>' + hold + '</td><td>' + side + '</td>'
      + '<td>' + fmt2(t.entry_premium) + '</td><td>' + fmt2(t.exit_premium) + '</td>'
      + '<td style="' + colorNum(t.pnl||0) + '">\u20b9' + ((t.pnl||0)).toFixed(0) + '</td>'
      + '<td style="' + colorNum(pnl) + ';font-weight:600">\u20b9' + pnl.toFixed(0) + '</td>'
      + '<td style="' + colorNum(t.net_pnl_bps||0) + '">' + fmtBps(t.net_pnl_bps) + '</td>'
      + '<td><span class="badge badge-blue">' + (t.exit_reason||'\u2014') + '</span></td>'
      + '<td>' + (t.fill_type||'limit') + '</td>'
      + '<td>' + fmtBps(t.slippage_actual_bps) + '</td></tr>';
  });

  let scatterSelect = '';
  if (hasSignals && featureNames.length > 0) {
    scatterSelect = '<select onchange="renderSignalScatter(this.value)" style="font-size:11px;margin-bottom:8px;background:var(--bg3);border:1px solid var(--border);color:var(--text);border-radius:4px;padding:4px 8px">';
    featureNames.forEach(f => { scatterSelect += '<option value="' + f + '">' + f + '</option>'; });
    scatterSelect += '</select>';
  }

  el.innerHTML = '<div class="filter-row">'
    + '<label style="color:var(--text2);font-weight:600;font-size:11px">DIRECTION:</label>'
    + '<select onchange="applyTradeFilter(\'direction\',this.value,\'' + sid + '\',\'' + rid + '\')">'
    + '<option value="">All</option>'
    + '<option value="CE"' + (state.tradeFilters.direction==='CE'?' selected':'') + '>CE (Call)</option>'
    + '<option value="PE"' + (state.tradeFilters.direction==='PE'?' selected':'') + '>PE (Put)</option></select>'
    + '<label style="color:var(--text2);font-weight:600;font-size:11px">EXIT REASON:</label>'
    + '<select onchange="applyTradeFilter(\'exit_reason\',this.value,\'' + sid + '\',\'' + rid + '\')">' + exitReasonOptions + '</select>'
    + '<span style="margin-left:auto;font-size:11px;color:var(--text2);font-weight:500">Total: ' + total + ' trades</span></div>'
    + '<div class="tbl-wrap"><table><thead><tr>'
    + '<th' + tip('entry_time') + '>Entry</th><th' + tip('exit_time') + '>Exit</th>'
    + '<th' + tip('hold_min') + '>Hold (min)</th><th' + tip('direction') + '>Dir</th>'
    + '<th' + tip('entry_prem') + '>Entry Prem</th><th' + tip('exit_prem') + '>Exit Prem</th>'
    + '<th' + tip('gross_pnl') + '>Gross PnL</th><th' + tip('net_pnl') + '>Net PnL</th>'
    + '<th' + tip('net_bps') + '>Net (bps)</th><th' + tip('exit_reason') + '>Exit</th>'
    + '<th' + tip('fill_type') + '>Fill</th><th' + tip('slippage') + '>Slip (bps)</th>'
    + '</tr></thead><tbody>' + tradeRows + '</tbody></table></div>'
    + '<div class="pagination">'
    + '<button onclick="tradePageNav(-1,\'' + sid + '\',\'' + rid + '\')"' + (page<=1?' disabled':'') + '>\u2190 Prev</button>'
    + '<span>Page ' + page + ' / ' + totalPages + '</span>'
    + '<button onclick="tradePageNav(1,\'' + sid + '\',\'' + rid + '\')"' + (page>=totalPages?' disabled':'') + '>Next \u2192</button></div>'
    + (!hasSignals ? '<div class="note" style="margin-top:10px">entry_signal_values not available for this run</div>' : '')
    + '<div class="grid3" style="margin-top:16px">'
    + '<div class="card"><div class="card-title">Net PnL Distribution</div><div class="chart-wrap-sm"><canvas id="c-pnl-hist"></canvas></div></div>'
    + '<div class="card"><div class="card-title">Hold Time Distribution</div><div class="chart-wrap-sm"><canvas id="c-hold-hist"></canvas></div></div>'
    + '<div class="card"><div class="card-title">Signal Scatter</div>' + scatterSelect
    + '<div class="chart-wrap-sm"><canvas id="c-signal-scatter"></canvas></div>'
    + (!hasSignals ? '<div class="note" style="margin-top:8px">No signal snapshots for this run</div>' : '') + '</div></div>';
  _renderTradeCharts(trades, featureNames[0], hasSignals);
}

function applyTradeFilter(key, val, sid, rid) {
  state.tradeFilters[key] = val;
  state.tradeFilters.page = 1;
  _loadAndRenderTrades(document.getElementById('overlay-content'), sid, rid);
}
function tradePageNav(delta, sid, rid) {
  const data = state.trades;
  const total = data?.total || 0;
  const ps = state.tradeFilters.page_size;
  const maxPage = Math.ceil(total / ps);
  state.tradeFilters.page = Math.max(1, Math.min(maxPage, state.tradeFilters.page + delta));
  _loadAndRenderTrades(document.getElementById('overlay-content'), sid, rid);
}

// ── Charts ───────────────────────────────────────────────────────────────
function _renderFoldChart(folds) {
  const canvas = document.getElementById('c-fold-pnl');
  if (!canvas || folds.length === 0) return;
  if (state.charts['c-fold-pnl']) state.charts['c-fold-pnl'].destroy();
  state.charts['c-fold-pnl'] = new Chart(canvas, {
    type: 'bar',
    data: { labels: folds.map(f => 'F' + f.fold + '\n' + (f.test_date||'')),
      datasets: [{ label: 'Net Edge (bps)', data: folds.map(f => f.net_edge_bps || 0),
        backgroundColor: folds.map(f => f.kill_triggered ? 'rgba(139,144,167,.5)' : (f.net_edge_bps||0) > 0 ? 'rgba(34,197,94,.7)' : 'rgba(239,68,68,.7)'),
        borderWidth: 0, borderRadius: 3 }] },
    options: { responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => {
        const f = folds[ctx.dataIndex];
        return ['Net: ' + (f.net_edge_bps||0).toFixed(1) + ' bps', 'Trades: ' + f.trades, 'Win: ' + ((f.win_rate||0)*100).toFixed(1) + '%', f.kill_triggered ? '!! Kill triggered' : ''].filter(Boolean);
      }}}},
      scales: { x: { ticks: { font: { size: 9 }, color: '#8b90a7', maxRotation: 0 }, grid: { color: 'rgba(46,51,71,.3)' } },
                y: { ticks: { font: { size: 9 }, color: '#8b90a7' }, title: { display: true, text: 'bps', color: '#8b90a7', font: { size: 10 } }, grid: { color: 'rgba(46,51,71,.3)' } } } }
  });
}

function _renderSensChart(sensRuns) {
  const canvas = document.getElementById('c-sens-chart');
  if (!canvas || sensRuns.length === 0) return;
  if (state.charts['c-sens-chart']) state.charts['c-sens-chart'].destroy();
  const values = sensRuns.map(r => r.sharpe_delta || 0);
  state.charts['c-sens-chart'] = new Chart(canvas, {
    type: 'bar',
    data: { labels: sensRuns.map(r => r.label || (r.param + ' ' + r.direction)),
      datasets: [{ label: 'Sharpe \u0394 vs baseline', data: values,
        backgroundColor: values.map(v => v >= 0 ? 'rgba(34,197,94,.7)' : 'rgba(239,68,68,.7)'), borderWidth: 0, borderRadius: 3 }] },
    options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { x: { ticks: { font: { size: 9 }, color: '#8b90a7' }, title: { display: true, text: 'Sharpe \u0394', color: '#8b90a7', font: { size: 10 } }, grid: { color: 'rgba(46,51,71,.3)' } },
                y: { ticks: { font: { size: 9 }, color: '#8b90a7' }, grid: { color: 'rgba(46,51,71,.3)' } } } }
  });
}

function _renderTradeCharts(trades, defaultFeature, hasSignals) {
  _renderHistogram('c-pnl-hist', trades.map(t => t.net_pnl || t.pnl || 0).filter(v => v != null), 'Net PnL (INR)', 'rgba(59,130,246,.7)');
  _renderHistogram('c-hold-hist', trades.map(t => (t.holding_bars || 0) * 5).filter(v => v > 0), 'Hold (seconds)', 'rgba(167,139,250,.7)');
  if (hasSignals && defaultFeature) renderSignalScatter(defaultFeature);
}

function _renderHistogram(canvasId, vals, label, color) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || vals.length === 0) return;
  if (state.charts[canvasId]) state.charts[canvasId].destroy();
  const n = Math.min(20, Math.ceil(Math.sqrt(vals.length)));
  const min = Math.min(...vals), max = Math.max(...vals);
  const step = (max - min) / n || 1;
  const bins = Array.from({length:n}, (_,i) => min + i*step);
  const counts = new Array(n).fill(0);
  for (const v of vals) { const i = Math.min(n-1, Math.floor((v - min) / step)); if (i >= 0) counts[i]++; }
  state.charts[canvasId] = new Chart(canvas, {
    type: 'bar',
    data: { labels: bins.map(b=>b.toFixed(0)), datasets: [{ data: counts, backgroundColor: color, borderWidth: 0, borderRadius: 2 }] },
    options: { responsive:true, maintainAspectRatio:false, plugins:{legend:{display:false}},
      scales:{x:{ticks:{font:{size:9},color:'#8b90a7'},grid:{color:'rgba(46,51,71,.3)'}},y:{ticks:{font:{size:9},color:'#8b90a7'},grid:{color:'rgba(46,51,71,.3)'}}} }
  });
}

function renderSignalScatter(featureName) {
  const canvas = document.getElementById('c-signal-scatter');
  if (!canvas) return;
  if (state.charts['c-signal-scatter']) state.charts['c-signal-scatter'].destroy();
  const trades = state.trades?.trades || [];
  const points = [];
  for (const t of trades) {
    if (!t.entry_signal_values) continue;
    try {
      const sv = typeof t.entry_signal_values === 'string' ? JSON.parse(t.entry_signal_values) : t.entry_signal_values;
      const x = sv[featureName]; const y = t.net_pnl || t.pnl || 0;
      if (x != null) points.push({ x: Number(x), y });
    } catch(e) {}
  }
  if (points.length === 0) return;
  state.charts['c-signal-scatter'] = new Chart(canvas, {
    type: 'scatter',
    data: { datasets: [{ label: featureName, data: points,
      backgroundColor: points.map(p => p.y >= 0 ? 'rgba(34,197,94,.6)' : 'rgba(239,68,68,.6)'), pointRadius: 3 }] },
    options: { responsive:true, maintainAspectRatio:false,
      plugins: { legend:{display:false}, tooltip:{callbacks:{label: ctx => 'x=' + ctx.parsed.x.toFixed(3) + ', y=' + ctx.parsed.y.toFixed(0)}} },
      scales: { x:{title:{display:true,text:featureName,font:{size:10},color:'#8b90a7'},ticks:{font:{size:9},color:'#8b90a7'},grid:{color:'rgba(46,51,71,.3)'}},
                y:{title:{display:true,text:'Net PnL',font:{size:10},color:'#8b90a7'},ticks:{font:{size:9},color:'#8b90a7'},grid:{color:'rgba(46,51,71,.3)'}} } }
  });
}
