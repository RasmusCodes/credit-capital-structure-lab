"""Reduced-form (intensity) credit models.

A HazardCurve holds a piecewise-constant risk-neutral default intensity lambda(t). Survival
Q(t) = exp(-integral_0^t lambda). Default is a surprise event; the hazard is calibrated to market prices
of bonds or CDS *given* a recovery assumption: the same price implies very different default
probabilities at different recoveries (see spreads.recovery_adjusted_pd).

Recovery conventions for risky bonds
------------------------------------
RFV  recovery of face (par): at default the holder receives R x face. Matches bankruptcy practice
     (claims are par + accrued) and CDS settlement; the platform default.
RT   recovery of Treasury (Jarrow-Turnbull): R x value of an otherwise identical riskless bond.
RMV  recovery of market value (Duffie-Singleton): R x pre-default value; gives spread = (1-R) lambda
     exactly, convenient but not how bankruptcy claims work.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, least_squares

from .bonds import FixedBond
from .curves import ZeroCurve


@dataclass
class HazardCurve:
    knots: np.ndarray      # segment end times t_1 < ... < t_n (years)
    hazards: np.ndarray    # intensity on (t_{i-1}, t_i]; the last value is extended flat
    label: str = ""

    def __post_init__(self):
        self.knots = np.atleast_1d(np.asarray(self.knots, dtype=float))
        self.hazards = np.atleast_1d(np.asarray(self.hazards, dtype=float))
        if len(self.knots) != len(self.hazards):
            raise ValueError("knots and hazards must have equal length")
        if np.any(np.diff(self.knots) <= 0):
            raise ValueError("knots must increase")
        starts = np.concatenate([[0.0], self.knots[:-1]])
        self._cum_at_knots = np.concatenate([[0.0], np.cumsum(self.hazards * (self.knots - starts))])

    @classmethod
    def flat(cls, lam: float, label: str = "flat") -> "HazardCurve":
        return cls(np.array([100.0]), np.array([lam]), label)

    def hazard(self, t):
        t = np.asarray(t, dtype=float)
        idx = np.searchsorted(self.knots, t, side="left")
        idx = np.clip(idx, 0, len(self.hazards) - 1)
        return self.hazards[idx]

    def cum_hazard(self, t):
        t = np.asarray(t, dtype=float)
        idx = np.searchsorted(self.knots, t, side="left")
        idx_c = np.clip(idx, 0, len(self.hazards) - 1)
        start = np.where(idx_c == 0, 0.0, self.knots[np.maximum(idx_c - 1, 0)])
        base = self._cum_at_knots[idx_c]
        return base + self.hazards[idx_c] * (t - start)

    def survival(self, t):
        return np.exp(-self.cum_hazard(t))

    def pd(self, t):
        return 1.0 - self.survival(t)

    def marginal_pd(self, t1, t2):
        """P(default in (t1, t2] | survival to t1)."""
        return 1.0 - self.survival(t2) / self.survival(t1)

    def shifted(self, d_lambda: float) -> "HazardCurve":
        return HazardCurve(self.knots.copy(), np.maximum(self.hazards + d_lambda, 0.0), self.label)

    def scaled(self, k: float) -> "HazardCurve":
        return HazardCurve(self.knots.copy(), self.hazards * k, self.label)

    def average_hazard(self, t: float) -> float:
        return float(self.cum_hazard(t) / t)


# ---------------------------------------------------------------------- risky bond pricing
def _default_grid(T: float, per_year: int) -> np.ndarray:
    n = max(int(np.ceil(T * per_year)), 1)
    return np.linspace(0.0, T, n + 1)


def risky_dirty_price(bond: FixedBond, settle, zc: ZeroCurve, hc: HazardCurve, recovery: float,
                      convention: str = "RFV", grid_per_year: int = 52) -> float:
    """Dirty price per 100 face of a fixed-rate bond under a hazard curve and recovery convention."""
    t = bond.times(settle)
    _, cf = bond.cashflows(settle)
    D = zc.df(t)
    S = hc.survival(t)
    conv = convention.upper()
    if conv == "RMV":
        return float(np.sum(cf * D * S ** (1.0 - recovery)))
    if conv == "RT":
        return float(np.sum(cf * D * (S + recovery * (1.0 - S))))
    if conv != "RFV":
        raise ValueError(f"unknown recovery convention {convention}")
    pv_surv = float(np.sum(cf * D * S))
    g = _default_grid(float(t[-1]), grid_per_year)
    Sg = hc.survival(g)
    mid = 0.5 * (g[1:] + g[:-1])
    pv_def = recovery * bond.face * float(np.sum(zc.df(mid) * (Sg[:-1] - Sg[1:])))
    return pv_surv + pv_def


def riskless_dirty_price(bond: FixedBond, settle, zc: ZeroCurve) -> float:
    return bond.dirty_from_curve(zc, settle)


@dataclass
class HazardFit:
    curve: HazardCurve
    status: str               # "ok", "price_above_riskless", "price_below_recovery", ...
    detail: str = ""


def implied_flat_hazard(bond: FixedBond, clean: float, settle, zc: ZeroCurve, recovery: float,
                        convention: str = "RFV") -> HazardFit:
    """Flat hazard rate that reprices the bond. Flags prices inconsistent with the recovery assumption."""
    dirty = clean + bond.accrued(settle)
    riskless = riskless_dirty_price(bond, settle, zc)
    if dirty >= riskless:
        return HazardFit(HazardCurve.flat(0.0), "price_above_riskless",
                         f"dirty {dirty:.3f} >= riskless {riskless:.3f}: no default premium in price")
    f = lambda lam: risky_dirty_price(bond, settle, zc, HazardCurve.flat(lam), recovery, convention) - dirty
    hi = 60.0
    if f(hi) > 0:
        return HazardFit(HazardCurve.flat(np.inf), "price_below_recovery",
                         f"dirty {dirty:.3f} below the recovery floor implied by R={recovery:.0%}")
    lam = brentq(f, 0.0, hi, xtol=1e-12, maxiter=500)
    return HazardFit(HazardCurve.flat(lam), "ok")


def bootstrap_hazard(bonds: list[FixedBond], cleans: list[float], settle, zc: ZeroCurve,
                     recoveries, convention: str = "RFV") -> tuple[HazardCurve, list[dict]]:
    """Sequential piecewise-constant hazard with a knot at each bond's maturity (sorted by maturity).

    Negative segment hazards (an inverted curve the data cannot support) are floored at zero and the
    repricing error is reported rather than hidden.
    """
    recs = np.broadcast_to(np.asarray(recoveries, dtype=float), (len(bonds),))
    order = np.argsort([b.maturity for b in bonds])
    knots: list[float] = []
    lams: list[float] = []
    diag: list[dict] = []
    for i in order:
        b, px, R = bonds[i], cleans[i], float(recs[i])
        T = float(b.times(settle)[-1])
        if knots and T <= knots[-1] + 1e-6:
            diag.append({"bond": b.name, "status": "same_maturity_skipped"})
            continue
        dirty = px + b.accrued(settle)

        def f(lam):
            hc = HazardCurve(np.array(knots + [T]), np.array(lams + [lam]))
            return risky_dirty_price(b, settle, zc, hc, R, convention) - dirty

        status = "ok"
        if f(0.0) < 0:
            lam = 0.0
            status = "floored_at_zero"
        elif f(60.0) > 0:
            lam = 60.0
            status = "price_below_recovery"
        else:
            lam = brentq(f, 0.0, 60.0, xtol=1e-12)
        err = f(lam)
        knots.append(T)
        lams.append(lam)
        diag.append({"bond": b.name, "T": T, "hazard": lam, "status": status, "error": err})
    return HazardCurve(np.array(knots), np.array(lams), "bootstrap"), diag


def fit_hazard_curve(bonds: list[FixedBond], cleans: list[float], settle, zc: ZeroCurve, recoveries,
                     knots, convention: str = "RFV", smooth: float = 0.0) -> tuple[HazardCurve, list[dict]]:
    """Least-squares hazard curve through more bonds than knots (residuals = relative value).

    Errors are weighted by 1/CS01 so that residuals are approximately in spread (bp) space.
    `smooth` penalises differences between adjacent log-hazards.
    """
    recs = np.broadcast_to(np.asarray(recoveries, dtype=float), (len(bonds),))
    knots = np.asarray(knots, dtype=float)
    dirty = np.array([c + b.accrued(settle) for b, c in zip(bonds, cleans)])
    w = []
    for b, c in zip(bonds, cleans):
        try:
            w.append(1.0 / max(b.spread_risk(c, zc, settle)["cs01"], 1e-6))
        except ValueError:
            w.append(1.0)
    w = np.array(w)

    def resid(x):
        hc = HazardCurve(knots, np.exp(x))
        model = np.array([risky_dirty_price(b, settle, zc, hc, float(R), convention) for b, R in zip(bonds, recs)])
        r = (model - dirty) * w
        if smooth > 0 and len(x) > 1:
            r = np.concatenate([r, smooth * np.diff(x)])
        return r

    flat_fit = []
    for b, c, R in zip(bonds, cleans, recs):
        fit = implied_flat_hazard(b, c, settle, zc, float(R), convention)
        if fit.status == "ok":
            flat_fit.append(float(fit.curve.hazards[0]))
    x0 = np.full(len(knots), np.log(max(np.median(flat_fit) if flat_fit else 0.05, 1e-4)))
    sol = least_squares(resid, x0, method="trf", xtol=1e-12, ftol=1e-12)
    hc = HazardCurve(knots, np.exp(sol.x), "fitted")
    out = []
    for b, c, R, wi in zip(bonds, cleans, recs, w):
        model_clean = risky_dirty_price(b, settle, zc, hc, float(R), convention) - b.accrued(settle)
        out.append({"bond": b.name, "market": c, "model": model_clean, "price_resid": c - model_clean,
                    "spread_resid_bp": -(c - model_clean) * wi})  # + = bond cheap (wider than curve)
    return hc, out
