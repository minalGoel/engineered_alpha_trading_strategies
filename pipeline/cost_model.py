"""
Cost model for NSE index option trades (NIFTY / BANKNIFTY).

All values in INR unless noted. Premium values are in points.
"""

# Lot sizes
NIFTY_LOT = 75
BANKNIFTY_LOT = 15

LOT_SIZES = {
    "NIFTY": NIFTY_LOT,
    "BANKNIFTY": BANKNIFTY_LOT,
    "NSE:NIFTY50-INDEX": NIFTY_LOT,
    "NSE:NIFTYBANK-INDEX": BANKNIFTY_LOT,
}

# Cost constants
SPREAD_POINTS = 1.5       # bid-ask spread per side in option premium points
STT_RATE = 0.000625       # 0.0625% on sell-side premium * qty
BROKERAGE_PER_ORDER = 20  # INR flat per order
EXCHANGE_PER_LOT = 5      # INR per lot per side (approx exchange + SEBI + GST)


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
) -> dict:
    """
    Compute full cost breakdown for an option buy-then-sell round trip.

    Parameters
    ----------
    entry_premium : float  – option premium at entry (points)
    exit_premium  : float  – option premium at exit  (points)
    lot_size      : int    – contract lot size (75 for NIFTY, 15 for BANKNIFTY)
    lots          : int    – number of lots traded

    Returns
    -------
    dict with gross_pnl, spread_cost, stt, brokerage, exchange, total_cost, net_pnl
         all in INR
    """
    qty = lot_size * lots

    gross_pnl = (exit_premium - entry_premium) * qty

    # Bid-ask spread: lost on both entry and exit
    spread_cost = SPREAD_POINTS * 2 * qty

    # STT: 0.0625% of sell-side turnover (premium * qty)
    stt = exit_premium * STT_RATE * qty

    # Brokerage: flat per order, entry + exit
    brokerage = BROKERAGE_PER_ORDER * 2

    # Exchange txn charges + SEBI + stamp + GST (approx)
    exchange = EXCHANGE_PER_LOT * 2 * lots

    total_cost = spread_cost + stt + brokerage + exchange
    net_pnl = gross_pnl - total_cost

    return {
        "gross_pnl": gross_pnl,
        "spread_cost": spread_cost,
        "stt": stt,
        "brokerage": brokerage,
        "exchange": exchange,
        "total_cost": total_cost,
        "net_pnl": net_pnl,
    }


def net_pnl_quick(
    entry_premium: float,
    exit_premium: float,
    lot_size: int,
    lots: int = 1,
) -> float:
    """Return just the net PnL (INR) for a round trip. Fast path."""
    qty = lot_size * lots
    gross = (exit_premium - entry_premium) * qty
    costs = (
        SPREAD_POINTS * 2 * qty
        + exit_premium * STT_RATE * qty
        + BROKERAGE_PER_ORDER * 2
        + EXCHANGE_PER_LOT * 2 * lots
    )
    return gross - costs


def min_points_to_breakeven(lot_size: int, entry_premium: float = 100.0, lots: int = 1) -> float:
    """
    Minimum premium move (points) needed for a round-trip to break even.
    Useful for sanity-checking strategy viability.
    """
    qty = lot_size * lots
    # cost = spread*2*qty + exit_prem*STT*qty + brokerage*2 + exchange*2*lots
    # gross = delta_pts * qty
    # At breakeven: delta_pts * qty = spread*2*qty + (entry+delta)*STT*qty + 40 + 10*lots
    # Approximate (delta is small vs entry): delta ≈ spread*2 + entry*STT + (40+10*lots)/qty
    fixed = (BROKERAGE_PER_ORDER * 2 + EXCHANGE_PER_LOT * 2 * lots) / qty
    variable = SPREAD_POINTS * 2 + entry_premium * STT_RATE
    return variable + fixed
