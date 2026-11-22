"""Equity options as credit instruments.

* Black-Scholes prices and implied volatility.
* Model-free upper bound on the risk-neutral default probability from a put: if the stock is worth
  (nearly) zero after default, a put pays K at default, so P >= K e^{-rT} Q(default by T) and
  Q <= P e^{rT} / K. Valid for American puts too (American >= European). Liquidity: deep out-of-the-money
  puts trade at wide spreads; use the bid for a conservative bound.
* Carr-Wu (2011) unit recovery claim: if the stock stays above B before default and falls below A at
  default, an American put spread between strikes A < B, scaled by 1/(B - A), pays $1 at default.
  With a constant hazard lambda its value is U = lambda/(lambda + r) (1 - e^{-(lambda + r) T}).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

N = ndtr


def bs_price(S, K, T, r, q, sigma, kind: str = "call") -> float:
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
        return float(intrinsic)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return float(S * np.exp(-q * T) * N(d1) - K * np.exp(-r * T) * N(d2))
    return float(K * np.exp(-r * T) * N(-d2) - S * np.exp(-q * T) * N(-d1))


def bs_implied_vol(price, S, K, T, r, q, kind: str = "call") -> float:
    lo, hi = 1e-4, 8.0
    f = lambda v: bs_price(S, K, T, r, q, v, kind) - price
    if f(lo) > 0 or f(hi) < 0:
        return float("nan")
    return float(brentq(f, lo, hi, xtol=1e-10))


def put_pd_upper_bound(put_price: float, K: float, T: float, r: float) -> float:
    return float(min(put_price * np.exp(r * T) / K, 1.0))


def unit_recovery_claim(put_low: float, put_high: float, K_low: float, K_high: float) -> float:
    if K_high <= K_low:
        raise ValueError("K_high must exceed K_low")
    return (put_high - put_low) / (K_high - K_low)


def hazard_from_urc(U: float, T: float, r: float) -> float:
    """Constant hazard rate consistent with a unit-recovery-claim value U (Carr-Wu)."""
    g = lambda lam: lam / (lam + r) * (1 - np.exp(-(lam + r) * T)) - U
    if U <= 0:
        return 0.0
    if g(50.0) < 0:
        return float("inf")
    return float(brentq(g, 1e-12, 50.0, xtol=1e-12))


def urc_default_probability(put_low, put_high, K_low, K_high, T, r) -> dict:
    U = unit_recovery_claim(put_low, put_high, K_low, K_high)
    lam = hazard_from_urc(U, T, r)
    return {"urc": U, "hazard": lam, "pd": float(1 - np.exp(-lam * T)) if np.isfinite(lam) else 1.0}
