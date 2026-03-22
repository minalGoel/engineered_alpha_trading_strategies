/**
 * core.js — Shared state, API helpers, formatters, tooltip definitions.
 * Loaded first by index.html before other dashboard scripts.
 */

const API = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? ''  // same-origin when running locally
  : (window.__API_URL__ || '');  // set via tunnel URL for Pages deployment

// ── Tooltip definitions ──────────────────────────────────────────────────
const TIPS = {
  // Leaderboard family table
  family: 'Signal logic family grouping related strategies',
  n_strategies: 'Number of distinct strategy variants in this family',
  best_dsr: 'Best Deflated Sharpe Ratio across all family members. DSR corrects for multiple testing bias.',
  best_net_edge: 'Best net edge (after all transaction costs) in basis points',
  avg_corr: 'Average pairwise return correlation between strategies in this family. Low = more diversified.',
  n_passing_kill: 'Strategies that did not trigger any kill condition and have positive net edge',
  cv_pass: 'Strategies that passed nested leave-one-out cross-validation (9+/12 profitable OOS days)',
  n_vectorized: 'Strategies using vectorized backtest engine (no fill model \u2014 treat with caution)',
  // Strategy sub-table
  strategy_id: 'Unique strategy identifier',
  sharpe_default: 'Sharpe ratio computed with default (non-optimized) parameters on full in-sample data',
  net_edge_bps: 'Net profit per trade in basis points, after all NSE transaction costs',
  cv: 'Cross-validation verdict: PASS if 9+/12 out-of-sample days are profitable',
  prof_days: 'Number of profitable out-of-sample days out of 12 total in nested LOO-CV',
  sensitivity: 'Parameter sensitivity: ROBUST = Sharpe survives \u00b120% param perturbation, FRAGILE = it collapses',
  kill: 'Kill condition: flags strategies with pathological behavior (e.g. extreme drawdown, too few trades)',
  engine: 'Backtest engine: event-driven (realistic fills) vs vectorized (no fill model)',
  // Overview stats
  dsr: 'Deflated Sharpe Ratio \u2014 probability that the true Sharpe exceeds the expected maximum from N trials',
  sharpe_raw: 'Annualized Sharpe ratio before any multiple-testing correction',
  win_rate: 'Fraction of trades that were profitable (net of costs)',
  max_drawdown: 'Largest peak-to-trough equity decline in INR',
  // DSR analysis
  total_n: 'Total number of backtest runs stored across all strategies',
  effective_n: 'Rank-based effective number of independent trials (accounts for correlated strategies)',
  sr_benchmark: 'Expected maximum Sharpe from N independent trials \u2014 the bar a strategy must beat',
  // Features/params
  feat_name: 'Feature/indicator name used in signal generation',
  feat_desc: 'Description of what the feature captures',
  feat_window: 'Lookback window or computation period',
  feat_source: 'Data source: spot price, option chain, VIX, etc.',
  feat_importance: 'Feature importance score (higher = more predictive)',
  param_name: 'Parameter name',
  param_value: 'Current parameter value used in backtest',
  param_type: 'Category: threshold, window, structural, etc.',
  param_optimized: 'Whether this parameter was tuned by Optuna \u2014 optimized params have highest overfitting risk',
  // CV folds
  fold: 'Cross-validation fold number',
  test_date: 'The held-out test date for this fold',
  trades: 'Number of trades executed',
  gross_edge: 'Gross edge before transaction costs (bps)',
  avg_hold: 'Average trade holding duration',
  // Sensitivity
  sens_param: 'Parameter being perturbed',
  sens_dir: 'Perturbation direction: +20% or -20% from default',
  sharpe_delta: 'Change in Sharpe ratio vs baseline default parameters',
  // Trade log
  entry_time: 'Trade entry timestamp',
  exit_time: 'Trade exit timestamp',
  hold_min: 'Trade holding duration in minutes',
  direction: 'CE = Call (bullish), PE = Put (bearish)',
  entry_prem: 'Option premium at entry',
  exit_prem: 'Option premium at exit',
  gross_pnl: 'Gross profit/loss before transaction costs (INR)',
  net_pnl: 'Net profit/loss after all transaction costs (INR)',
  net_bps: 'Net PnL in basis points relative to premium',
  exit_reason: 'Why the trade was closed: STOP, TARGET, SIGNAL, TIME, or EOD',
  fill_type: 'Order fill type: limit or market',
  slippage: 'Actual slippage observed (bps)',
  // Portfolio
  total_cost: 'Total transaction cost per trade in basis points (includes all NSE fees)',
  corr_val: 'Pairwise return correlation coefficient',
  passes_dsr: 'Whether the strategy Deflated Sharpe > 0.5 (beats expected max SR benchmark)',
};

function tip(key) { return TIPS[key] ? ' data-tip="' + TIPS[key].replace(/"/g, '&quot;') + '"' : ''; }
function tipAttr(key) { return TIPS[key] ? ' title="' + TIPS[key].replace(/"/g, '&quot;') + '"' : ''; }

// ── State ────────────────────────────────────────────────────────────────
let state = {
  leaderboard: null,
  portfolio: null,
  selectedStrategy: null,
  strategyDetail: null,
  strategyRuns: null,
  strategyResults: null,
  trades: null,
  tradeFilters: { page:1, page_size:100, direction:'', exit_reason:'', fold_number:'' },
  detailTab: 'overview',
  sortCol: null, sortDir: 1,
  selectedRunIds: [],
  comparisonData: null,
  charts: {},
};

// ── Fetch helpers ────────────────────────────────────────────────────────
async function api(path) {
  const r = await fetch(API + path);
  if (!r.ok) throw new Error('HTTP ' + r.status + ': ' + path);
  return r.json();
}
async function post(path, body) {
  const r = await fetch(API + path, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body) });
  if (!r.ok) throw new Error('HTTP ' + r.status + ': ' + path);
  return r.json();
}

// ── Tab switching ────────────────────────────────────────────────────────
function showTab(tab) {
  document.querySelectorAll('.tab-btn').forEach((b,i) => b.classList.toggle('active', i === (tab==='leaderboard'?0:1)));
  document.getElementById('tab-leaderboard').style.display = tab==='leaderboard'?'':'none';
  document.getElementById('tab-portfolio').style.display = tab==='portfolio'?'':'none';
  if (tab==='portfolio' && !state.portfolio) loadPortfolio();
}

// ── Number format helpers ────────────────────────────────────────────────
const fmt2 = v => (v==null||isNaN(v)) ? '\u2014' : Number(v).toFixed(2);
const fmt4 = v => (v==null||isNaN(v)) ? '\u2014' : Number(v).toFixed(4);
const fmtPct = v => (v==null||isNaN(v)) ? '\u2014' : (Number(v)*100).toFixed(1)+'%';
const fmtBps = v => (v==null||isNaN(v)) ? '\u2014' : Number(v).toFixed(1)+' bps';
const colorNum = (v, threshold=0) => v == null ? '' : (v > threshold ? 'color:var(--green)' : v < threshold ? 'color:var(--red)' : '');

// ── Utility helpers ──────────────────────────────────────────────────────
function _minsToTime(m) {
  const h = Math.floor(m/60);
  const mm = m % 60;
  return String(h).padStart(2,'0') + ':' + String(mm).padStart(2,'0');
}
function _escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function toggleCompare(evt, sid) { evt.stopPropagation(); }
