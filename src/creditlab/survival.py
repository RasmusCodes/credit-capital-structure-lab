"""Survival analysis for default times.

* Kaplan-Meier survival with Greenwood standard errors, and the Nelson-Aalen cumulative hazard, for
  firm-level default/censoring data.
* Conversions between cumulative default rates (cohort / static-pool tables), marginal (conditional)
  default rates and average hazard rates.
* A discrete-time hazard (logit) model, the Shumway (2001) approach, fitted by IRLS without extra
  dependencies.
* Physical-versus-risk-neutral comparison: the ratio of market-implied to historical default
  probabilities, the quantity behind the "credit spread puzzle".
"""
from __future__ import annotations

import numpy as np


def kaplan_meier(durations, events):
    """durations: time to default or censoring; events: 1 = default, 0 = censored.

    Returns (times, survival, greenwood_se) at each distinct event time.
    """
    t = np.asarray(durations, dtype=float)
    e = np.asarray(events, dtype=int)
    order = np.argsort(t)
    t, e = t[order], e[order]
    times = np.unique(t[e == 1])
    S = 1.0
    var_sum = 0.0
    surv, se = [], []
    for ti in times:
        at_risk = np.sum(t >= ti)
        d = np.sum((t == ti) & (e == 1))
        S *= 1.0 - d / at_risk
        if at_risk > d:
            var_sum += d / (at_risk * (at_risk - d))
        surv.append(S)
        se.append(S * np.sqrt(var_sum))
    return times, np.array(surv), np.array(se)


def nelson_aalen(durations, events):
    t = np.asarray(durations, dtype=float)
    e = np.asarray(events, dtype=int)
    times = np.unique(t[e == 1])
    H, out = 0.0, []
    for ti in times:
        at_risk = np.sum(t >= ti)
        d = np.sum((t == ti) & (e == 1))
        H += d / at_risk
        out.append(H)
    return times, np.array(out)


def marginal_from_cumulative(cum_pd) -> np.ndarray:
    """Conditional default rate in year k given survival to k-1, from cumulative PDs (years 1..n)."""
    c = np.concatenate([[0.0], np.asarray(cum_pd, dtype=float)])
    surv = 1.0 - c
    return 1.0 - surv[1:] / surv[:-1]


def average_hazard(cum_pd, years) -> np.ndarray:
    """Average constant intensity consistent with cumulative PD by each horizon."""
    return -np.log(1.0 - np.asarray(cum_pd, dtype=float)) / np.asarray(years, dtype=float)


def fit_discrete_hazard(X, y, l2: float = 0.0, max_iter: int = 100, tol: float = 1e-10):
    """Logit of the probability of default in the next period given covariates X (panel rows).

    Returns coefficients (with intercept first) and standard errors from the inverse Hessian.
    """
    X = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)])
    y = np.asarray(y, dtype=float)
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        z = X @ beta
        p = 1.0 / (1.0 + np.exp(-z))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + l2 * np.eye(X.shape[1])
        g = X.T @ (y - p) - l2 * beta
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    cov = np.linalg.inv(H)
    return beta, np.sqrt(np.diag(cov))


def risk_premium_ratio(rn_pd, physical_pd) -> np.ndarray:
    """Ratio of risk-neutral (market-implied) to physical (historical) cumulative default probability."""
    return np.asarray(rn_pd, dtype=float) / np.maximum(np.asarray(physical_pd, dtype=float), 1e-12)
