"""
Cost model for NSE index option trades (NIFTY / BANKNIFTY).
Based on Upstox brokerage charges: https://upstox.com/brokerage-charges/

All values in INR unless noted. Premium values are in option points (₹1/point/unit).
Turnover = premium × lot_size × lots (in INR).

Charges for EQUITY OPTIONS on NSE (rates effective from 1 Oct 2024):
─────────────────────────────────────────────────────────────────────
  Brokerage:           ₹20 flat per executed order (not per lot)
  STT:                 0.15% on sell-side premium turnover (from 1 Apr 2025)
  Exchange Txn (NSE):  0.03553% on premium turnover (both sides)
  SEBI Fee:            ₹10/crore on turnover (both sides)
  Stamp Duty:          0.003% on buy-side turnover
  IPFT:                ₹0.50/lakh on premium turnover (both sides)
  GST:                 18% on (brokerage + exchange txn + IPFT)
─────────────────────────────────────────────────────────────────────

Execution model: limit orders at candle close prices. No bid-ask spread.
"""

import math

# ── Lot sizes ────────────────────────────────────────────────────────────
NIFTY_LOT = 65
BANKNIFTY_LOT = 30

LOT_SIZES = {
    "NIFTY": NIFTY_LOT,
    "BANKNIFTY": BANKNIFTY_LOT,
    "NSE:NIFTY50-INDEX": NIFTY_LOT,
    "NSE:NIFTYBANK-INDEX": BANKNIFTY_LOT,
}

# ── Upstox charges (equity options, from 1 Oct 2024) ─────────────────────
BROKERAGE_PER_ORDER = 20.0       # ₹20 flat per executed order
STT_RATE = 0.0015                # 0.15% on sell-side premium turnover (from 1 Apr 2025)
EXCHANGE_TXN_RATE = 0.0003553    # 0.03553% on premium turnover (both sides)
SEBI_FEE_RATE = 0.000001         # ₹10/crore = 0.0001% on turnover (both sides)
STAMP_DUTY_RATE = 0.00003        # 0.003% on buy-side turnover
IPFT_RATE = 0.000005             # ₹0.50/lakh on premium turnover (both sides)
GST_RATE = 0.18                  # 18% on (brokerage + exchange txn + IPFT)

# ── Capital deployment ───────────────────────────────────────────────────
CAPITAL_PER_ENTRY = 100_000      # ₹1 lakh per strategy entry (modifiable at backtest time)


def get_lot_size(underlying: str) -> int:
    """Return lot size for an underlying. Raises KeyError if unknown."""
    key = underlying.upper()
    for k, v in LOT_SIZES.items():
        if key in k.upper() or k.upper() in key:
            return v
    raise KeyError(f"Unknown underlying: {underlying}")


def compute_lots(entry_premium: float, lot_size: int, capital: float = CAPITAL_PER_ENTRY) -> int:
    """Compute number of lots affordable with given capital.

    lots = floor(capital / (entry_premium × lot_size))
    Always returns at least 1 (minimum trade size).
    """
    if entry_premium <= 0 or lot_size <= 0:
        return 1
    lots = int(capital / (entry_premium * lot_size))
    return max(lots, 1)


def compute_trade_costs(
    entry_premium: float,
    exit_premium: float,
    lot_size: int,
    lots: int = 0,
    capital: float = CAPITAL_PER_ENTRY,
) -> dict:
    """
    Compute full cost breakdown for an option buy-then-sell round trip.

    Parameters
    ----------
    entry_premium  : option premium at entry (points, ₹1 per point per unit)
    exit_premium   : option premium at exit (points)
    lot_size       : contract lot size (65 for NIFTY, 30 for BANKNIFTY)
    lots           : number of lots (0 = auto-compute from capital)
    capital        : capital budget per entry in INR (default ₹1,00,000)

    Returns
    -------
    dict with full cost breakdown, all in INR
    """
    if lots <= 0:
        lots = compute_lots(entry_premium, lot_size, capital)
    qty = lot_size * lots
    buy_turnover = entry_premium * qty
    sell_turnover = exit_premium * qty
    total_turnover = buy_turnover + sell_turnover

    gross_pnl = (exit_premium - entry_premium) * qty

    # 1. STT: 0.15% on sell-side premium turnover only
    stt = sell_turnover * STT_RATE

    # 2. Exchange transaction charges: 0.03553% on premium turnover (both sides)
    exchange_txn = total_turnover * EXCHANGE_TXN_RATE

    # 3. SEBI fee: ₹10/crore on turnover (both sides)
    sebi_fee = total_turnover * SEBI_FEE_RATE

    # 4. Stamp duty: 0.003% on buy-side turnover only
    stamp_duty = buy_turnover * STAMP_DUTY_RATE

    # 5. IPFT: ₹0.50/lakh on premium turnover (both sides)
    ipft = total_turnover * IPFT_RATE

    # 6. Brokerage: ₹20 per order × 2 (entry + exit), flat regardless of lots
    brokerage = BROKERAGE_PER_ORDER * 2

    # 7. GST: 18% on (brokerage + exchange txn + IPFT)
    gst = (brokerage + exchange_txn + ipft) * GST_RATE

    regulatory = exchange_txn + sebi_fee + stamp_duty + ipft
    total_cost = stt + brokerage + regulatory + gst
    net_pnl = gross_pnl - total_cost

    return {
        "lots": lots,
        "qty": qty,
        "gross_pnl": gross_pnl,
        "stt": stt,
        "brokerage": brokerage,
        "exchange_txn": exchange_txn,
        "sebi_fee": sebi_fee,
        "stamp_duty": stamp_duty,
        "ipft": ipft,
        "gst": gst,
        "regulatory": regulatory,
        "total_cost": total_cost,
        "net_pnl": net_pnl,
    }


def net_pnl_quick(
    entry_premium: float,
    exit_premium: float,
    lot_size: int,
    lots: int = 0,
    capital: float = CAPITAL_PER_ENTRY,
) -> float:
    """Return just the net PnL (INR) for a round trip. Fast path.

    If lots=0, auto-computes from capital / (entry_premium × lot_size).
    """
    if lots <= 0:
        lots = compute_lots(entry_premium, lot_size, capital)
    qty = lot_size * lots
    buy_to = entry_premium * qty
    sell_to = exit_premium * qty
    total_to = buy_to + sell_to

    gross = (exit_premium - entry_premium) * qty

    stt = sell_to * STT_RATE
    exchange_txn = total_to * EXCHANGE_TXN_RATE
    sebi = total_to * SEBI_FEE_RATE
    stamp = buy_to * STAMP_DUTY_RATE
    ipft = total_to * IPFT_RATE
    brokerage = BROKERAGE_PER_ORDER * 2
    gst = (brokerage + exchange_txn + ipft) * GST_RATE

    costs = stt + brokerage + exchange_txn + sebi + stamp + ipft + gst
    return gross - costs


def min_points_to_breakeven(
    lot_size: int,
    entry_premium: float = 100.0,
    lots: int = 0,
    capital: float = CAPITAL_PER_ENTRY,
) -> float:
    """Minimum premium move (points) needed for a round-trip to break even."""
    if lots <= 0:
        lots = compute_lots(entry_premium, lot_size, capital)
    qty = lot_size * lots
    turnover_approx = entry_premium * qty * 2  # both sides ≈ same premium

    fixed = BROKERAGE_PER_ORDER * 2
    stt = entry_premium * qty * STT_RATE
    exchange_txn = turnover_approx * EXCHANGE_TXN_RATE
    sebi = turnover_approx * SEBI_FEE_RATE
    stamp = entry_premium * qty * STAMP_DUTY_RATE
    ipft = turnover_approx * IPFT_RATE
    gst = (fixed + exchange_txn + ipft) * GST_RATE

    total = fixed + stt + exchange_txn + sebi + stamp + ipft + gst
    return total / qty


def print_cost_breakdown(entry_premium, exit_premium, lot_size, lots=0, capital=CAPITAL_PER_ENTRY):
    """Print a human-readable cost breakdown."""
    if lots <= 0:
        lots = compute_lots(entry_premium, lot_size, capital)
    c = compute_trade_costs(entry_premium, exit_premium, lot_size, lots, capital)
    qty = lot_size * lots
    print(f"  Capital: ₹{capital:,.0f} → {lots} lot{'s' if lots>1 else ''} × {lot_size} = {qty} units")
    print(f"  Entry: ₹{entry_premium:.2f} × {qty} units = ₹{entry_premium * qty:,.2f}")
    print(f"  Exit:  ₹{exit_premium:.2f} × {qty} units = ₹{exit_premium * qty:,.2f}")
    print(f"  Gross PnL:       ₹{c['gross_pnl']:>10.2f}")
    print(f"  ─── Costs ───")
    print(f"  STT (0.15%×sell): ₹{c['stt']:>10.2f}")
    print(f"  Brokerage (₹20×2): ₹{c['brokerage']:>10.2f}")
    print(f"  Exchange txn:      ₹{c['exchange_txn']:>10.2f}")
    print(f"  SEBI fee:          ₹{c['sebi_fee']:>10.2f}")
    print(f"  Stamp duty:        ₹{c['stamp_duty']:>10.2f}")
    print(f"  IPFT:              ₹{c['ipft']:>10.2f}")
    print(f"  GST (18%):         ₹{c['gst']:>10.2f}")
    print(f"  ─── Total cost ─── ₹{c['total_cost']:>10.2f}")
    print(f"  Net PnL:           ₹{c['net_pnl']:>+10.2f}")
