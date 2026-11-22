"""Closed-form building blocks for geometric Brownian motion, with and without a lower barrier.

Asset value V_t = V0 exp((mu - s^2/2) t + s W_t). Work in X_t = ln V_t - g t, so that a barrier growing
at rate g, H(t) = H0 exp(g t), becomes the constant level b = ln H0 for X. X has drift
nu = mu - s^2/2 - g and volatility s.

Everything here is exact (reflection principle); tests compare it with Monte Carlo.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

N = ndtr


def _z(x, m, sd):
    return (x - m) / sd


def partial_moments(V0, mu, s, T, a, b):
    """For lognormal V_T: P(a < V_T <= b) and E[V_T 1{a < V_T <= b}] (a, b may be 0 / inf)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    sd = s * np.sqrt(T)
    m = np.log(V0) + (mu - 0.5 * s * s) * T
    with np.errstate(divide="ignore"):
        la = np.where(a > 0, np.log(np.maximum(a, 1e-300)), -np.inf)
        lb = np.where(np.isfinite(b), np.log(np.maximum(b, 1e-300)), np.inf)
    p = N(_z(lb, m, sd)) - N(_z(la, m, sd))
    ev = V0 * np.exp(mu * T) * (N(_z(lb, m + sd * sd, sd)) - N(_z(la, m + sd * sd, sd)))
    return p, ev


def barrier_partial_moments(V0, mu, s, T, H0, g, a, b):
    """P(a < V_T <= b, no hit) and E[V_T 1{a < V_T <= b, no hit}] for barrier H(t) = H0 e^{g t} < V(t).

    Requires a >= H(T) (the no-hit event already forces V_T > H(T)); a is clipped up to H(T).
    """
    if H0 <= 0:
        return partial_moments(V0, mu, s, T, a, b)
    HT = H0 * np.exp(g * T)
    a = np.maximum(np.asarray(a, dtype=float), HT)
    b = np.asarray(b, dtype=float)
    x0 = np.log(V0)
    bb = np.log(H0)
    nu = mu - 0.5 * s * s - g
    sd = s * np.sqrt(T)
    # X_T thresholds: X = ln V - g T
    with np.errstate(divide="ignore"):
        xa = np.log(a) - g * T
        xb = np.where(np.isfinite(b), np.log(np.maximum(b, 1e-300)) - g * T, np.inf)
    refl = 2 * bb - x0
    k = np.exp(2 * nu * (bb - x0) / (s * s))

    def prob(start):
        m = start + nu * T
        return N(_z(xb, m, sd)) - N(_z(xa, m, sd))

    def expv(start):
        # E[exp(X_T + gT) 1{xa < X_T <= xb}] for X_T ~ N(start + nu T, sd^2)
        m = start + nu * T
        return np.exp(start + nu * T + 0.5 * sd * sd + g * T) * (
            N(_z(xb, m + sd * sd, sd)) - N(_z(xa, m + sd * sd, sd)))

    p = prob(x0) - k * prob(refl)
    ev = expv(x0) - k * expv(refl)
    return np.maximum(p, 0.0), np.maximum(ev, 0.0)


def first_passage_prob(V0, mu, s, T, H0, g=0.0):
    """P(V_t <= H0 e^{g t} for some t <= T), barrier below V0."""
    if H0 <= 0:
        return np.zeros_like(np.asarray(T, dtype=float))
    T = np.asarray(T, dtype=float)
    a = np.log(V0 / H0)  # distance to barrier, > 0
    nu = mu - 0.5 * s * s - g
    sd = s * np.sqrt(np.maximum(T, 1e-300))
    p = N((-a - nu * T) / sd) + np.exp(-2 * nu * a / (s * s)) * N((-a + nu * T) / sd)
    return np.where(T > 0, np.clip(p, 0.0, 1.0), 0.0)


def discounted_hit(V0, mu, s, T, H0, r, g=0.0):
    """E[exp(-r tau) 1{tau <= T}] for the first passage of V to the barrier H0 e^{g t}."""
    if H0 <= 0:
        return 0.0
    a = np.log(V0 / H0)
    nu = mu - 0.5 * s * s - g
    nt = np.sqrt(nu * nu + 2 * r * s * s)
    sd = s * np.sqrt(T)
    return float(np.exp(-a * (nu + nt) / (s * s)) * N((-a + nt * T) / sd)
                 + np.exp(-a * (nu - nt) / (s * s)) * N((-a - nt * T) / sd))


def hit_density(V0, mu, s, t, H0, g=0.0):
    """Density of the first-passage time at t (inverse Gaussian)."""
    t = np.asarray(t, dtype=float)
    a = np.log(V0 / H0)
    nu = mu - 0.5 * s * s - g
    return a / (s * np.sqrt(2 * np.pi * t ** 3)) * np.exp(-(a + nu * t) ** 2 / (2 * s * s * t))


def bs_call(V0, K, T, r, q, s):
    """Black-Scholes call on a dividend-paying asset (K may be an array; K=0 gives the forward PV)."""
    K = np.asarray(K, dtype=float)
    p, ev = partial_moments(V0, r - q, s, T, K, np.inf)
    return np.exp(-r * T) * (ev - K * p)
