"""Merton (1974) structural model.

Firm assets V follow GBM with volatility s and payout rate q. Debt is a single zero-coupon claim of face
D due at T. Equity is a European call on V; default happens only at T, if V_T < D.

Calibration problem: V and s are not observable. The standard approach solves two equations,
E_obs = E(V, s) and sigma_E,obs * E_obs = N(d1) e^{-qT} s V, given equity value and equity volatility.
The iterative KMV / Bharath-Shumway procedure instead uses a history of equity values (see
`calibrate_iterative`). Both inherit the model's assumptions: lognormal assets, a static debt amount,
default only at T, frictionless markets and no strategic behaviour (docs/ASSUMPTIONS_AND_LIMITS.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr

N = ndtr


@dataclass
class MertonResult:
    V: float
    sigma_V: float
    E: float
    B: float               # market value of debt
    pd_q: float            # risk-neutral default probability to T
    pd_p: float | None     # physical default probability (if mu given)
    spread: float          # credit spread of the debt, continuous comp.
    dd: float              # distance to default (risk-neutral d2)
    recovery_q: float      # expected recovery given default (fraction of face, risk-neutral)
    sigma_E: float
    leverage: float        # D e^{-rT} / V


def d1d2(V, D, T, r, q, s):
    d1 = (np.log(V / D) + (r - q + 0.5 * s * s) * T) / (s * np.sqrt(T))
    return d1, d1 - s * np.sqrt(T)


def merton(V, s, D, T, r, q=0.0, mu=None) -> MertonResult:
    d1, d2 = d1d2(V, D, T, r, q, s)
    E = V * np.exp(-q * T) * N(d1) - D * np.exp(-r * T) * N(d2)
    B = V * np.exp(-q * T) - E
    pd_q = N(-d2)
    pd_p = None
    if mu is not None:
        d2p = (np.log(V / D) + (mu - q - 0.5 * s * s) * T) / (s * np.sqrt(T))
        pd_p = float(N(-d2p))
    spread = -np.log(B / (D * np.exp(-r * T))) / T
    rec = V * np.exp((r - q) * T) * N(-d1) / (D * N(-d2)) if N(-d2) > 1e-300 else 1.0
    sigma_E = V * np.exp(-q * T) * N(d1) * s / E if E > 0 else np.inf
    return MertonResult(float(V), float(s), float(E), float(B), float(pd_q), pd_p, float(spread), float(d2),
                        float(rec), float(sigma_E), float(D * np.exp(-r * T) / V))


def calibrate(E_obs: float, sigma_E_obs: float, D: float, T: float, r: float, q: float = 0.0,
              mu=None) -> MertonResult:
    """Solve for (V, s) from observed equity value and equity volatility (two-equation method)."""

    def resid(x):
        V, s = np.exp(x)
        d1, d2 = d1d2(V, D, T, r, q, s)
        E = V * np.exp(-q * T) * N(d1) - D * np.exp(-r * T) * N(d2)
        sE = V * np.exp(-q * T) * N(d1) * s
        return [(E - E_obs) / E_obs, (sE - sigma_E_obs * E_obs) / (sigma_E_obs * E_obs)]

    V0 = E_obs + D * np.exp(-r * T)
    s0 = max(sigma_E_obs * E_obs / V0, 1e-3)
    best = None
    for scale in (1.0, 0.5, 2.0):
        sol = least_squares(resid, np.log([V0, s0 * scale]), xtol=1e-14, ftol=1e-14, gtol=1e-14)
        if best is None or sol.cost < best.cost:
            best = sol
    V, s = np.exp(best.x)
    if best.cost > 1e-12:
        raise RuntimeError(f"Merton calibration did not converge (cost {best.cost:.2e})")
    return merton(V, s, D, T, r, q, mu)


def calibrate_iterative(equity_values: np.ndarray, D: np.ndarray | float, T: float, r: np.ndarray | float,
                        dt: float = 1 / 252, tol: float = 1e-6, max_iter: int = 200) -> dict:
    """KMV-style iteration (Crosbie-Bohn; Bharath-Shumway 2008 'iterative Merton').

    Start with s = equity vol x E/(E+D); invert each day's equity value for V_t; re-estimate s from the
    implied asset log returns; repeat until s converges. Returns the final asset path and vol.
    """
    E = np.asarray(equity_values, dtype=float)
    Dv = np.broadcast_to(np.asarray(D, dtype=float), E.shape)
    rv = np.broadcast_to(np.asarray(r, dtype=float), E.shape)
    sE = np.std(np.diff(np.log(E)), ddof=1) / np.sqrt(dt)
    s = sE * E[-1] / (E[-1] + Dv[-1])
    V = E + Dv * np.exp(-rv * T)
    for it in range(max_iter):
        Vn = np.empty_like(E)
        for i in range(len(E)):
            Vn[i] = _invert_equity(E[i], Dv[i], T, rv[i], s, guess=V[i])
        s_new = np.std(np.diff(np.log(Vn)), ddof=1) / np.sqrt(dt)
        V = Vn
        if abs(s_new - s) < tol:
            s = s_new
            break
        s = s_new
    mu = np.mean(np.diff(np.log(V))) / dt + 0.5 * s * s
    return {"V": V, "sigma_V": float(s), "mu": float(mu), "iterations": it + 1}


def _invert_equity(E, D, T, r, s, guess):
    from scipy.optimize import brentq

    f = lambda V: V * N(d1d2(V, D, T, r, 0.0, s)[0]) - D * np.exp(-r * T) * N(d1d2(V, D, T, r, 0.0, s)[1]) - E
    lo, hi = max(E, 1e-9), E + D * 2.0 + 1.0
    return brentq(f, lo, hi, xtol=1e-10)


def tranche_values(V, s, faces: list[float], T, r, q=0.0) -> dict:
    """Multi-tranche Merton: claims of decreasing seniority with faces F1, F2, ... all due at T.

    Tranche k pays min(max(V_T - sum_{j<k} F_j, 0), F_k); equity gets the residual. Value of tranche k =
    C(sum_{j<k} F_j) - C(sum_{j<=k} F_j) with C(K) a call on V (C(0) = V e^{-qT}).
    """
    from .lognormal import bs_call

    cum = np.concatenate([[0.0], np.cumsum(faces)])
    calls = bs_call(V, cum, T, r, q, s)
    vals = calls[:-1] - calls[1:]
    out = []
    for k, F in enumerate(faces):
        y = -np.log(vals[k] / F) / T if vals[k] > 0 else np.inf
        out.append({"face": F, "value": float(vals[k]), "yield": float(y), "spread": float(y - r)})
    return {"tranches": out, "equity": float(calls[-1])}
