"""Black-Cox (1976) first-passage model.

Default occurs the first time assets V touch a barrier H(t) = H0 e^{g t} (a safety covenant, or the
point where equity holders stop funding the firm), or at the debt maturity T if V_T < D. At a barrier
hit, creditors receive (1 - alpha) H(tau); at maturity they receive min(V_T, D), less a proportional
bankruptcy cost alpha if V_T < D. Equity is a down-and-out call on V.

Compared with Merton this produces a term structure of default probabilities and positive short-term
default risk, but it keeps the same lognormal-asset assumptions, and the barrier level is a modelling
choice that the market does not reveal (docs/ASSUMPTIONS_AND_LIMITS.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .lognormal import barrier_partial_moments, discounted_hit, first_passage_prob


@dataclass
class BlackCoxResult:
    V: float
    sigma_V: float
    E: float
    B: float
    pd_q: float
    pd_barrier: float
    spread: float
    H0: float
    sigma_E: float


def black_cox(V, s, D, T, r, q=0.0, H0=None, g=0.0, alpha=0.0) -> BlackCoxResult:
    """Value equity and a zero-coupon debt claim of face D under the Black-Cox model.

    H0 defaults to the PV of the debt face at the risk-free rate (barrier = D e^{-r(T-t)}, g = r), the
    classic choice under which the barrier ends at D.
    """
    if H0 is None:
        H0, g = D * np.exp(-r * T), r
    mu = r - q
    HT = H0 * np.exp(g * T)
    if V <= H0:
        return BlackCoxResult(V, s, 0.0, (1 - alpha) * V, 1.0, 1.0, np.inf, H0, np.nan)
    # equity: (V_T - D)^+ on no-hit paths
    p_up, ev_up = barrier_partial_moments(V, mu, s, T, H0, g, max(D, HT), np.inf)
    E = np.exp(-r * T) * (float(ev_up) - D * float(p_up))
    # debt at maturity
    p_mid, ev_mid = barrier_partial_moments(V, mu, s, T, H0, g, HT, max(D, HT))
    B_T = np.exp(-r * T) * (D * float(p_up) + (1 - alpha) * float(ev_mid))
    # debt at barrier hit: (1 - alpha) H0 e^{g tau}, discounted -> H0 E[exp(-(r-g) tau)]
    B_hit = (1 - alpha) * H0 * discounted_hit(V, mu, s, T, H0, r - g, g)
    B = B_T + B_hit
    p_hit = float(first_passage_prob(V, mu, s, T, H0, g))
    pd = p_hit + float(p_mid)
    spread = -np.log(B / (D * np.exp(-r * T))) / T if B > 0 else np.inf
    # equity vol from a numerical delta
    h = 1e-4 * V
    Eu = _equity(V + h, s, D, T, r, q, H0, g)
    Ed = _equity(V - h, s, D, T, r, q, H0, g)
    delta = (Eu - Ed) / (2 * h)
    sE = delta * V * s / E if E > 0 else np.inf
    return BlackCoxResult(float(V), float(s), float(E), float(B), float(pd), p_hit, float(spread), float(H0), float(sE))


def _equity(V, s, D, T, r, q, H0, g):
    mu = r - q
    HT = H0 * np.exp(g * T)
    if V <= H0:
        return 0.0
    p_up, ev_up = barrier_partial_moments(V, mu, s, T, H0, g, max(D, HT), np.inf)
    return float(np.exp(-r * T) * (ev_up - D * p_up))


def pd_term_structure(V, s, H0, g, r, q, times) -> np.ndarray:
    """Risk-neutral probability of a barrier hit by each time (default only via the barrier)."""
    return np.array([float(first_passage_prob(V, r - q, s, t, H0, g)) for t in np.atleast_1d(times)])


def calibrate(E_obs, sigma_E_obs, D, T, r, q=0.0, barrier_ratio=None, g=None, alpha=0.0) -> BlackCoxResult:
    """Solve (V, s) from equity value and volatility. Barrier H0 = barrier_ratio x D (default: PV of D)."""

    def params(V):
        if barrier_ratio is None:
            return D * np.exp(-r * T), r
        return barrier_ratio * D, (0.0 if g is None else g)

    def resid(x):
        V, s = np.exp(x)
        H0, gg = params(V)
        res = black_cox(V, s, D, T, r, q, H0, gg, alpha)
        return [(res.E - E_obs) / E_obs, (res.sigma_E - sigma_E_obs) / sigma_E_obs]

    V0 = E_obs + D * np.exp(-r * T)
    s0 = max(sigma_E_obs * E_obs / V0, 1e-3)
    best = None
    for scale in (1.0, 0.6, 1.5):
        try:
            sol = least_squares(resid, np.log([V0, s0 * scale]), xtol=1e-13, ftol=1e-13, gtol=1e-13)
        except (ValueError, FloatingPointError):
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None or best.cost > 1e-10:
        raise RuntimeError("Black-Cox calibration did not converge")
    V, s = np.exp(best.x)
    H0, gg = params(V)
    return black_cox(V, s, D, T, r, q, H0, gg, alpha)
