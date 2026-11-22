"""Price every claim in a capital structure on one asset-value process.

The firm's assets follow GBM. At the horizon T the firm defaults if V_T is below the default point DP
(by default, total debt). Each claim then receives its waterfall payout from `waterfall.distribute`:
going-concern priority shares if solvent, or the default waterfall (administrative costs, collateral,
structural subordination, optional gift to equity) if not. Under the Black-Cox variant the firm also
defaults the first time V hits the barrier H(t) = H0 e^{g t}, and claims receive the default waterfall
of H(tau).

Expectations are exact up to quadrature: payouts are piecewise smooth in V with kinks where a claim
becomes paid in full, so the lognormal integral is split at those breakpoints (Gauss-Legendre in log V)
and the region above the last breakpoint, where every claim is paid in full and equity is linear in V,
is integrated in closed form.

This generalises multi-tranche Merton (see merton.tranche_values, reproduced by tests) to collateral,
several legal entities, guarantees, preferred stock and deviations from absolute priority.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import least_squares

from ..capital_structure import CapitalStructure
from ..hazard import HazardCurve
from ..waterfall import breakpoints, distribute
from .lognormal import barrier_partial_moments, discounted_hit, first_passage_prob, hit_density, partial_moments

_GL_X, _GL_W = leggauss(16)
_GL_T_X, _GL_T_W = leggauss(64)


@dataclass
class StructuralPricing:
    V0: float
    sigma: float
    T: float
    r: float
    model: str
    default_point: float
    H0: float
    g: float
    values: dict                 # claim id -> PV
    pd_q: float                  # risk-neutral default probability by T
    pd_barrier: float            # of which via the barrier (Black-Cox)
    cond_recovery: dict          # claim id -> E^Q[payout | default] / amount
    yields: dict                 # claim id -> continuously compounded yield of a zero with face = amount
    equity: float
    sigma_E: float = float("nan")
    extra: dict = field(default_factory=dict)


def _segments(edges: np.ndarray):
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a * (1 + 1e-12):
            ya, yb = np.log(a), np.log(b)
            y = 0.5 * (yb - ya) * _GL_X + 0.5 * (yb + ya)
            yield y, 0.5 * (yb - ya) * _GL_W


def _bp_cache(cs, ac, gift):
    bp_s = breakpoints(cs, in_default=False)
    bp_d = breakpoints(cs, in_default=True, admin_cost=ac, gift_to_equity=gift)
    return np.unique(np.concatenate([bp_s, bp_d]))


def price_structure(cs: CapitalStructure, V0: float, sigma: float, T: float, r: float, q: float = 0.0,
                    model: str = "merton", H0: float | None = None, g: float = 0.0,
                    default_point: float | None = None, admin_cost: float | None = None,
                    gift_to_equity: float = 0.0, bps: np.ndarray | None = None,
                    with_sigma_E: bool = True) -> StructuralPricing:
    ac = cs.admin_cost if admin_cost is None else admin_cost
    DP = cs.total_debt() if default_point is None else float(default_point)
    if bps is None:
        bps = _bp_cache(cs, ac, gift_to_equity)
    claims = cs.claims
    amounts = np.array([c.amount for c in claims])
    is_common = np.array([c.kind == "common" for c in claims])
    total_claims = float(amounts[~is_common].sum())
    mu = r - q
    sd = sigma * np.sqrt(T)
    m = np.log(V0) + (mu - 0.5 * sigma * sigma) * T
    barrier = model == "black_cox" and H0 is not None and H0 > 0
    if barrier and V0 <= H0:
        res = distribute(cs, np.array([V0]), admin_cost=ac, gift_to_equity=gift_to_equity, in_default=True)
        vals = {c.id: float(res.payouts[c.id][0]) for c in claims}
        rec = {c.id: float(res.recovery[c.id][0]) for c in claims}
        return StructuralPricing(V0, sigma, T, r, model, DP, H0, g, vals, 1.0, 1.0, rec,
                                 {c.id: np.inf for c in claims}, vals.get(_common_id(cs), 0.0))
    HT = H0 * np.exp(g * T) if barrier else 0.0

    lo = max(np.exp(m - 9 * sd), HT * (1 + 1e-12) if barrier else 0.0)
    upper = max(float(bps.max()) if len(bps) else 0.0, DP, total_claims) * (1 + 1e-9)
    edges = [lo] + [b for b in np.concatenate([bps, [DP]]) if lo < b < upper] + [upper]
    edges = np.unique(np.array(edges, dtype=float))

    if barrier:
        x0, bb = np.log(V0), np.log(H0)
        nu = mu - 0.5 * sigma * sigma - g
        kref = np.exp(2 * nu * (bb - x0) / (sigma * sigma))

        def dens(y):
            x = y - g * T
            phi = lambda mean: np.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
            return np.where(x > bb, phi(x0 + nu * T) - kref * phi(2 * bb - x0 + nu * T), 0.0)
    else:
        def dens(y):
            return np.exp(-0.5 * ((y - m) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))

    pv = np.zeros(len(claims))
    def_exp = np.zeros(len(claims))   # undiscounted E[payout 1{default at T}]
    p_def_T = 0.0
    if upper > lo:
        ys, ws = [], []
        for y, w in _segments(edges):
            ys.append(y)
            ws.append(w)
        y = np.concatenate(ys)
        w = np.concatenate(ws) * dens(y)
        Vt = np.exp(y)
        indef = Vt < DP
        res = distribute(cs, Vt, admin_cost=ac, gift_to_equity=gift_to_equity, in_default=indef)
        P = np.column_stack([res.payouts[c.id] for c in claims])
        pv += P.T @ w
        def_exp += P.T @ (w * indef)
        p_def_T = float(np.sum(w * indef))
    # closed-form tail above `upper`: all claims paid in full, equity = V - total claims
    if barrier:
        p_t, ev_t = barrier_partial_moments(V0, mu, sigma, T, H0, g, upper, np.inf)
    else:
        p_t, ev_t = partial_moments(V0, mu, sigma, T, upper, np.inf)
    p_t, ev_t = float(p_t), float(ev_t)
    tail = np.where(is_common, 0.0, amounts * p_t)
    n_common = max(int(is_common.sum()), 1)
    tail = np.where(is_common, (ev_t - total_claims * p_t) / n_common, tail)
    pv = (pv + tail) * np.exp(-r * T)

    # barrier hits
    p_hit = 0.0
    hit_exp = np.zeros(len(claims))
    if barrier:
        p_hit = float(first_passage_prob(V0, mu, sigma, T, H0, g))
        if abs(g) < 1e-14:
            resH = distribute(cs, np.array([H0]), admin_cost=ac, gift_to_equity=gift_to_equity, in_default=True)
            PH = np.array([resH.payouts[c.id][0] for c in claims])
            pv += PH * discounted_hit(V0, mu, sigma, T, H0, r, 0.0)
            hit_exp = PH * p_hit
        else:
            u = 0.5 * (_GL_T_X + 1.0)
            tau = T * u * u
            wt = 0.5 * _GL_T_W * 2 * T * u * hit_density(V0, mu, sigma, tau, H0, g)
            resH = distribute(cs, H0 * np.exp(g * tau), admin_cost=ac, gift_to_equity=gift_to_equity, in_default=True)
            PH = np.column_stack([resH.payouts[c.id] for c in claims])
            pv += PH.T @ (wt * np.exp(-r * tau))
            hit_exp = PH.T @ wt

    pd = p_hit + p_def_T
    cond = {}
    for i, c in enumerate(claims):
        if c.kind == "common" or c.amount <= 0 or pd <= 1e-14:
            cond[c.id] = float("nan")
        else:
            cond[c.id] = float((def_exp[i] + hit_exp[i]) / (pd * c.amount))
    yields = {}
    for i, c in enumerate(claims):
        if c.kind != "common" and c.amount > 0 and pv[i] > 0:
            yields[c.id] = float(-np.log(pv[i] / c.amount) / T)
        else:
            yields[c.id] = float("nan")
    values = {c.id: float(pv[i]) for i, c in enumerate(claims)}
    eq = float(sum(values[c.id] for c in claims if c.kind == "common"))
    out = StructuralPricing(V0, sigma, T, r, model, DP, H0 or 0.0, g, values, float(pd), p_hit, cond, yields, eq)
    if with_sigma_E and eq > 0:
        h = 1e-3 * V0
        kw = dict(T=T, r=r, q=q, model=model, H0=H0, g=g, default_point=default_point, admin_cost=ac,
                  gift_to_equity=gift_to_equity, bps=bps, with_sigma_E=False)
        up = price_structure(cs, V0 + h, sigma, **kw).equity
        dn = price_structure(cs, V0 - h, sigma, **kw).equity
        out.sigma_E = float((up - dn) / (2 * h) * V0 * sigma / eq)
    return out


def _common_id(cs):
    for c in cs.claims:
        if c.kind == "common":
            return c.id
    return None


def calibrate_structure(cs: CapitalStructure, E_obs: float, sigma_E_obs: float, T: float, r: float,
                        q: float = 0.0, model: str = "merton", barrier_ratio: float | None = None, g: float = 0.0,
                        default_point: float | None = None, admin_cost: float | None = None,
                        gift_to_equity: float = 0.0) -> StructuralPricing:
    """Find asset value and volatility so the model's common equity matches its market value and volatility.

    barrier_ratio: Black-Cox barrier H0 as a fraction of the default point (e.g. 0.6). Ignored for Merton.
    """
    ac = cs.admin_cost if admin_cost is None else admin_cost
    bps = _bp_cache(cs, ac, gift_to_equity)
    DP = cs.total_debt() if default_point is None else default_point

    def run(V, s, sig=True):
        H0 = barrier_ratio * DP if (model == "black_cox" and barrier_ratio) else None
        return price_structure(cs, V, s, T, r, q, model, H0, g, DP, ac, gift_to_equity, bps, with_sigma_E=sig)

    def resid(x):
        V, s = np.exp(x)
        p = run(V, s)
        return [(p.equity - E_obs) / E_obs, (p.sigma_E - sigma_E_obs) / sigma_E_obs]

    V0 = E_obs + cs.total_claims() * np.exp(-r * T)
    s0 = max(sigma_E_obs * E_obs / V0, 0.01)
    best = None
    for scale in (1.0, 0.7, 1.4):
        try:
            sol = least_squares(resid, np.log([V0, s0 * scale]), xtol=1e-12, ftol=1e-12, gtol=1e-12,
                                diff_step=1e-6)
        except (ValueError, FloatingPointError, ZeroDivisionError):
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None:
        raise RuntimeError("calibration failed")
    V, s = np.exp(best.x)
    out = run(V, s)
    out.extra["calibration_cost"] = float(best.cost)
    out.extra["converged"] = bool(best.cost < 1e-10)
    return out


def structural_hazard_curve(V0: float, sigma: float, H0: float, g: float, r: float, q: float = 0.0,
                            times=(0.5, 1, 2, 3, 4, 5, 7, 10)) -> HazardCurve:
    """Piecewise-constant hazard curve matching Black-Cox first-passage survival at `times`.

    Used to price coupon bonds of any maturity off the structural model (hybrid approach): survival from
    the asset process, recovery per claim from the waterfall at the barrier.
    """
    t = np.asarray(times, dtype=float)
    S = 1.0 - np.array([float(first_passage_prob(V0, r - q, sigma, ti, H0, g)) for ti in t])
    S = np.maximum(S, 1e-300)
    cum = -np.log(S)
    prev = np.concatenate([[0.0], cum[:-1]])
    dt = np.diff(np.concatenate([[0.0], t]))
    lam = np.maximum((cum - prev) / dt, 0.0)
    return HazardCurve(t, lam, "black-cox")


def barrier_recoveries(cs: CapitalStructure, H: float, admin_cost: float | None = None,
                       gift_to_equity: float = 0.0) -> dict:
    """Recovery fraction of each claim if the firm defaults with asset value H."""
    res = distribute(cs, np.array([H]), admin_cost=admin_cost, gift_to_equity=gift_to_equity, in_default=True)
    return {c.id: float(res.recovery[c.id][0]) for c in cs.claims}
