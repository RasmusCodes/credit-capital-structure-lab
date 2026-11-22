"""CreditGrades (Finger et al., RiskMetrics 2002): an industry first-passage model for equity-implied CDS.

Asset value per share V = S + Lbar D, asset vol sigma = sigma_S S / (S + Lbar D); the default barrier is
L D with an uncertain recovery L (lognormal, mean Lbar, log-sd lam). Barrier uncertainty gives a
non-zero default probability over very short horizons, which plain Black-Cox lacks.

Survival:  P(t) = N(-A_t/2 + ln d / A_t) - d N(-A_t/2 - ln d / A_t),
           d = V e^{lam^2} / (Lbar D),  A_t^2 = sigma^2 t + lam^2.
CDS spread (continuous premium, recovery R on the CDS):
           c = r (1-R) (1 - P(0) + H(t)) / (P(0) - P(t) e^{-rt} - H(t)),
           H(t) = e^{r xi} (G(t + xi) - G(xi)), xi = lam^2 / sigma^2,
           G(u) = d^{z+1/2} N(-ln d/(sigma sqrt u) - z sigma sqrt u) + d^{-z+1/2} N(-ln d/(sigma sqrt u) + z sigma sqrt u),
           z = sqrt(1/4 + 2 r / sigma^2).
The closed form is checked in tests against a numerical integration of the survival curve.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

N = ndtr


def _params(S, sigma_S, D, Lbar, lam):
    V = S + Lbar * D
    sigma = sigma_S * S / V
    d = V * np.exp(lam * lam) / (Lbar * D)
    return V, sigma, d


def survival(t, S, sigma_S, D, Lbar=0.5, lam=0.3):
    t = np.asarray(t, dtype=float)
    _, sigma, d = _params(S, sigma_S, D, Lbar, lam)
    A = np.sqrt(sigma * sigma * t + lam * lam)
    return N(-A / 2 + np.log(d) / A) - d * N(-A / 2 - np.log(d) / A)


def cds_spread(T, S, sigma_S, D, r, R=0.4, Lbar=0.5, lam=0.3):
    """Equity-implied par CDS spread (decimal, continuous premium) for maturity T."""
    _, sigma, d = _params(S, sigma_S, D, Lbar, lam)
    xi = lam * lam / (sigma * sigma)
    z = np.sqrt(0.25 + 2 * r / (sigma * sigma))

    def G(u):
        su = sigma * np.sqrt(u)
        ld = np.log(d)
        return d ** (z + 0.5) * N(-ld / su - z * su) + d ** (-z + 0.5) * N(-ld / su + z * su)

    P0 = float(survival(0.0, S, sigma_S, D, Lbar, lam))
    PT = float(survival(T, S, sigma_S, D, Lbar, lam))
    H = np.exp(r * xi) * (G(T + xi) - G(xi))
    return float(r * (1 - R) * (1 - P0 + H) / (P0 - PT * np.exp(-r * T) - H))


def cds_spread_numeric(T, S, sigma_S, D, r, R=0.4, Lbar=0.5, lam=0.3, n=20000):
    """Same spread by integrating the survival curve (continuous premium); used to verify the closed form."""
    t = np.linspace(0.0, T, n + 1)
    P = survival(t, S, sigma_S, D, Lbar, lam)
    disc = np.exp(-r * t)
    # protection: default at t=0 jump (1 - P(0)) plus continuous part
    prot = (1 - R) * ((1.0 - P[0]) + np.sum(0.5 * (disc[1:] + disc[:-1]) * (P[:-1] - P[1:])))
    prem = np.sum(0.5 * (disc[1:] * P[1:] + disc[:-1] * P[:-1]) * np.diff(t))
    return float(prot / prem)


def implied_vol_from_spread(spread, T, S, D, r, R=0.4, Lbar=0.5, lam=0.3):
    """Equity volatility that makes the CreditGrades spread equal a market spread."""
    from scipy.optimize import brentq

    f = lambda v: cds_spread(T, S, v, D, r, R, Lbar, lam) - spread
    lo, hi = 0.01, 5.0
    if f(lo) > 0:
        return lo
    if f(hi) < 0:
        return np.inf
    return float(brentq(f, lo, hi, xtol=1e-10))
