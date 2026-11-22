"""Scenario and jump-to-default P&L for capital-structure positions.

Positions are held per security: bonds and converts in face (currency millions), equity and preferred
in shares (millions), CDS in notional (positive = bought protection), puts in contracts x 100 shares.
Prices are per 100 face for bonds/CDS upfront points, per share for equity/preferred/options.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Position:
    security: str            # claim id, "equity", "cds", or an option label
    kind: str                # bond | convert | equity | preferred | cds | put
    quantity: float          # see module docstring
    price: float             # current price (bond: clean per 100; cds: upfront points; equity: per share)
    strike: float | None = None


def jump_to_default(positions: list[Position], recoveries: dict, equity_after: float = 0.0,
                    cds_recovery: float | None = None) -> dict:
    """P&L if the issuer defaults now: bonds/converts/preferred go to their waterfall recovery (fraction
    of face or liquidation preference), equity to `equity_after` per share, CDS pays (1 - R) on notional,
    puts pay strike - equity_after."""
    rows = []
    for p in positions:
        if p.kind in ("bond", "convert"):
            R = recoveries[p.security]
            pnl = p.quantity * (R * 100 - p.price) / 100
        elif p.kind == "preferred":
            R = recoveries.get(p.security, 0.0)
            pnl = p.quantity * (R * p.strike - p.price) if p.strike else p.quantity * -p.price
        elif p.kind == "equity":
            pnl = p.quantity * (equity_after - p.price)
        elif p.kind == "cds":
            R = cds_recovery if cds_recovery is not None else recoveries.get("cds_reference", 0.4)
            pnl = p.quantity * ((1 - R) - p.price / 100)
        elif p.kind == "put":
            pnl = p.quantity * 100 * (max(p.strike - equity_after, 0.0) - p.price) / 1e6
        else:
            raise ValueError(p.kind)
        rows.append({"security": p.security, "kind": p.kind, "pnl": float(pnl)})
    return {"rows": rows, "total": float(sum(r["pnl"] for r in rows))}


def repricing_pnl(positions: list[Position], new_prices: dict) -> dict:
    rows = []
    for p in positions:
        new = new_prices[p.security]
        if p.kind in ("bond", "convert", "cds"):
            # bonds per 100 face; CDS upfront points rise with spreads, so a protection buyer gains
            pnl = p.quantity * (new - p.price) / 100
        elif p.kind == "put":
            pnl = p.quantity * 100 * (new - p.price) / 1e6
        else:
            pnl = p.quantity * (new - p.price)
        rows.append({"security": p.security, "pnl": float(pnl)})
    return {"rows": rows, "total": float(sum(r["pnl"] for r in rows))}


def recovery_uncertainty(positions: list[Position], recovery_sets: dict, equity_after: float = 0.0) -> dict:
    """Jump-to-default P&L under several recovery scenarios (e.g. waterfall low / central / high)."""
    return {name: jump_to_default(positions, recs, equity_after)["total"] for name, recs in recovery_sets.items()}


def hedge_ratio_jtd_neutral(bond_face: float, bond_price: float, bond_recovery: float, put_strike: float,
                            put_price: float, equity_after: float = 0.0) -> float:
    """Number of put contracts that offset a long bond's jump-to-default loss."""
    loss = bond_face * (bond_price - bond_recovery * 100) / 100 * 1e6
    gain_per_contract = 100 * (max(put_strike - equity_after, 0) - put_price)
    return float(loss / gain_per_contract) if gain_per_contract > 0 else np.inf
