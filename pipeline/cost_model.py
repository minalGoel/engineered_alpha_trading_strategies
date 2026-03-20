"""
Cost model for NSE index option trades (NIFTY / BANKNIFTY).
Based on Upstox brokerage charges (https://upstox.com/brokerage-charges/).

All values in INR unless noted. Premium values are in option points.
Turnover = premium × lot_size × lots (in INR, since 1 option point = ₹1).

Charges for INTRADAY OPTIONS on NSE:
─────────────────────────────────────────────────────────────────────
  Brokerage:           ₹20 flat per executed order (not per lot)
  STT:                 0.025% on sell-side turnover
  Exchange Txn:        0.03553% on premium turnover (both sides) [Mar 2026]
  SEBI Fee:            ₹10/crore on turnover (both sides)
  Stamp Duty:          0.003% on buy-side turnover
  IPFT:                ₹0.50/lakh on premium turnover (both sides)
  GST:                 18% on (brokerage + exchange txn + IPFT)
─────────────────────────────────────────────────────────────────────
"""

# Lot sizes
NIFTY_LOT = 65
BANKNIFTY_LOT = 30

LOT_SIZES = {
    "NIFTY": NIFTY_LOT,
    "BANKNIFTY": BANKNIFTY_LOT,
    "NSE:NIFTY50-INDEX": NIFTY_LOT,
    "NSE:NIFTYBANK-INDEX": BANKNIFTY_LOT,
}

# ── Upstox actual charges ──────────────────────────────────────────
BROKERAGE_PER_ORDER = 20.0       # ₹20 flat per order (1 lot or 100 lots = same)
STT_RATE = 0.00025               # 0.025% on sell-side turnover
EXCHANGE_TXN_RATE = 0.0003553    # 0.03553% on premium turnover (both sides)
SEBI_FEE_RATE = 0.000001         # ₹10/crore = 0.0001%
STAMP_DUTY_RATE = 0.00003        # 0.003% on buy-side turnover
IPFT_RATE = 0.000005             # ₹0.50/lakh = 0.0005%
GST_RATE = 0.18                  # 18% on (brokerage + exchange + IPFT)

# Bid-ask spread assumption (configurable per analysis)
SPREAD_POINTS = 1.5              # default: ₹1.50 per side in option premium points


def get_lot_size(underlying: str) -> int:
    """Return lot size for an underlying. Raises KeyError if unknown."""
    key = underlying.upper()
    for k, v in LOT_SIZES.items():
        if key in k.upper() or k.upper() in key:
            return v
    raise KeyError(f"Unknown underlying: {underlying}")


def compute_trade_costs(
    entry_premium: float,
    exit_premium: float,
    lot_size: int,
    lots: int = 1,
    spread_per_side: float = SPREAD_POINTS,
) -> dict:
    """
    Compute full cost breakdown for an option buy-then-sell round trip.

    Parameters
    ----------
    entry_premium  : option premium at entry (points, ₹1 per point per unit)
    exit_premium   : option premium at exit (points)
    lot_size       : contract lot size (75 for NIFTY, 15 for BANKNIFTY)
    lots           : number of lots in the order
    spread_per_side: bid-ask spread assumption in points per side

    Returns
    -------
    dict with full cost breakdown, all in INR
    """
    qty = lot_size * lots
    buy_turnover = entry_premium * qty     # INR
    sell_turnover = exit_premium * qty     # INR
    total_turnover = buy_turnover + sell_turnover

    gross_pnl = (exit_premium - entry_premium) * qty

    # 1. Bid-ask spread: lost on both entry and exit
    spread_cost = spread_per_side * 2 * qty

    # 2. STT: 0.025% on sell-side turnover only
    stt = sell_turnover * STT_RATE

    # 3. Exchange transaction charges: 0.03553% on both sides
    exchange_txn = total_turnover * EXCHANGE_TXN_RATE

    # 4. SEBI fee: ₹10/crore on both sides
    sebi_fee = total_turnover * SEBI_FEE_RATE

    # 5. Stamp duty: 0.003% on buy-side turnover only
    stamp_duty = buy_turnover * STAMP_DUTY_RATE

    # 6. IPFT: ₹0.50/lakh on both sides
    ipft = total_turnover * IPFT_RATE

    # 7. GST: 18% on (brokerage + exchange txn + IPFT)
    brokerage = BROKERAGE_PER_ORDER * 2  # entry order + exit order (flat, not per lot)
    gst = (brokerage + exchange_txn + ipft) * GST_RATE

    # Regulatory total = exchange_txn + sebi + stamp + ipft
    regulatory = exchange_txn + sebi_fee + stamp_duty + ipft

    total_cost = spread_cost + stt + brokerage + regulatory + gst
    net_pnl = gross_pnl - total_cost

    return {
        "gross_pnl": gross_pnl,
        "spread_cost": spread_cost,
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
    lots: int = 1,
    spread_per_side: float = SPREAD_POINTS,
) -> float:
    """Return just the net PnL (INR) for a round trip. Fast path."""
    qty = lot_size * lots
    buy_to = entry_premium * qty
    sell_to = exit_premium * qty
    total_to = buy_to + sell_to

    gross = (exit_premium - entry_premium) * qty
    spread_cost = spread_per_side * 2 * qty
    stt = sell_to * STT_RATE
    exchange_txn = total_to * EXCHANGE_TXN_RATE
    sebi = total_to * SEBI_FEE_RATE
    stamp = buy_to * STAMP_DUTY_RATE
    ipft = total_to * IPFT_RATE
    brokerage = BROKERAGE_PER_ORDER * 2
    gst = (brokerage + exchange_txn + ipft) * GST_RATE

    costs = spread_cost + stt + brokerage + exchange_txn + sebi + stamp + ipft + gst
    return gross - costs


def min_points_to_breakeven(
    lot_size: int,
    entry_premium: float = 100.0,
    lots: int = 1,
    spread_per_side: float = SPREAD_POINTS,
) -> float:
    """Minimum premium move (points) needed for a round-trip to break even."""
    # Approximate: assume exit ≈ entry + delta, delta small vs entry
    qty = lot_size * lots
    turnover_approx = entry_premium * qty * 2  # both sides ≈ same premium

    fixed = BROKERAGE_PER_ORDER * 2
    spread = spread_per_side * 2 * qty
    stt = entry_premium * qty * STT_RATE
    exchange_txn = turnover_approx * EXCHANGE_TXN_RATE
    sebi = turnover_approx * SEBI_FEE_RATE
    stamp = entry_premium * qty * STAMP_DUTY_RATE
    ipft = turnover_approx * IPFT_RATE
    gst = (fixed + exchange_txn + ipft) * GST_RATE

    total = fixed + spread + stt + exchange_txn + sebi + stamp + ipft + gst
    return total / qty


def print_cost_breakdown(entry_premium, exit_premium, lot_size, lots, spread_per_side=SPREAD_POINTS):
    """Print a human-readable cost breakdown."""
    c = compute_trade_costs(entry_premium, exit_premium, lot_size, lots, spread_per_side)
    qty = lot_size * lots
    print(f"  Entry: ₹{entry_premium:.2f} × {qty} units ({lots} lot{'s' if lots>1 else ''} × {lot_size})")
    print(f"  Exit:  ₹{exit_premium:.2f} × {qty} units")
    print(f"  Gross PnL:       ₹{c['gross_pnl']:>10.2f}")
    print(f"  ─── Costs ───")
    print(f"  Spread ({spread_per_side}×2×{qty}):  ₹{c['spread_cost']:>10.2f}")
    print(f"  STT (0.025%×sell):  ₹{c['stt']:>10.2f}")
    print(f"  Brokerage (₹20×2): ₹{c['brokerage']:>10.2f}")
    print(f"  Exchange txn:      ₹{c['exchange_txn']:>10.2f}")
    print(f"  SEBI fee:          ₹{c['sebi_fee']:>10.2f}")
    print(f"  Stamp duty:        ₹{c['stamp_duty']:>10.2f}")
    print(f"  IPFT:              ₹{c['ipft']:>10.2f}")
    print(f"  GST (18%):         ₹{c['gst']:>10.2f}")
    print(f"  ─── Total cost ─── ₹{c['total_cost']:>10.2f}")
    print(f"  Net PnL:           ₹{c['net_pnl']:>+10.2f}")
