"""Single-name CDS under ISDA-style standard conventions.

* Quarterly coupons on the adjusted IMM dates (20 Mar/Jun/Sep/Dec, weekend -> following Monday),
  ACT/360 accrual, last accrual period inclusive of the maturity date.
* Protection from the step-in date (trade date + 1 calendar day) to the scheduled maturity.
* Accrued premium paid on default. Legs are integrated on a daily grid (exact to well under 0.1bp
  versus the closed-form ISDA integrals for piecewise-flat curves).
* Upfront is quoted clean (excluding accrued premium) and forward-valued to cash settlement (T+3
  business days). The "quoted" (conventional) spread converts to upfront through a flat hazard curve,
  as in the ISDA converter.

Differences to the official ISDA CDS Standard Model, documented rather than hidden: no holiday
calendar, and the discount curve is whatever the caller supplies (Treasury here; ISDA uses the
SOFR swap curve). Both move upfronts by hundredths of a point.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .curves import ZeroCurve
from .daycount import Date, add_business_days, adjust_following, previous_imm, to_date
from .hazard import HazardCurve


@dataclass
class CDSContract:
    trade_date: Date
    maturity: Date          # scheduled termination date (unadjusted IMM date)
    coupon: float           # running coupon, decimal (0.01 = 100bp, 0.05 = 500bp)
    recovery: float = 0.40
    notional: float = 1.0

    def __post_init__(self):
        self.trade_date = to_date(self.trade_date)
        self.maturity = to_date(self.maturity)

    @property
    def step_in(self) -> Date:
        return self.trade_date + dt.timedelta(days=1)

    @property
    def cash_settle(self) -> Date:
        return add_business_days(self.trade_date, 3)

    @property
    def accrual_start(self) -> Date:
        d = previous_imm(self.step_in)
        adj = adjust_following(d)
        if adj > self.step_in:
            d = previous_imm(d - dt.timedelta(days=1))
            adj = adjust_following(d)
        return adj

    def periods(self) -> list[tuple[Date, Date, Date]]:
        """[(accrual start, accrual end (exclusive), payment date)]."""
        start = self.accrual_start
        pays: list[Date] = []
        y, m = start.year, start.month
        while True:
            m += 3
            if m > 12:
                m -= 12
                y += 1
            d = dt.date(y, m, 20)
            if d >= self.maturity:
                break
            pays.append(adjust_following(d))
        out = []
        s = start
        for p in pays:
            out.append((s, p, p))
            s = p
        out.append((s, self.maturity + dt.timedelta(days=1), adjust_following(self.maturity)))
        return out

    def accrued_days(self) -> int:
        return (self.step_in - self.accrual_start).days


def _t(trade: Date, d: Date) -> float:
    return (d - trade).days / 365.0


def cds_legs(c: CDSContract, zc: ZeroCurve, hc: HazardCurve) -> dict:
    """Protection leg PV and risky annuities (per unit notional, valued at the trade date)."""
    trade = c.trade_date
    days = np.arange((c.step_in - trade).days, (c.maturity - trade).days + 2)  # day offsets, incl. maturity day
    t = days / 365.0
    Q = hc.survival(t)
    dQ = Q[:-1] - Q[1:]
    tm = 0.5 * (t[1:] + t[:-1])
    Dm = zc.df(tm)
    protection = (1.0 - c.recovery) * float(np.sum(Dm * dQ))
    # premium leg
    rpv01 = 0.0
    aod = 0.0
    day_mid = 0.5 * (days[1:] + days[:-1])
    for s, e, p in c.periods():
        frac = (e - s).days / 360.0
        rpv01 += frac * float(zc.df(_t(trade, p))) * float(hc.survival(_t(trade, e - dt.timedelta(days=1))))
        s_off = (s - trade).days
        e_off = (e - trade).days
        mask = (day_mid >= max(s_off, (c.step_in - trade).days)) & (day_mid < e_off)
        if mask.any():
            acc = (day_mid[mask] - s_off) / 360.0
            aod += float(np.sum(acc * Dm[mask] * dQ[mask]))
    rpv01_dirty = rpv01 + aod
    accrued = c.accrued_days() / 360.0
    return {"protection": protection, "rpv01_dirty": rpv01_dirty, "rpv01_clean": rpv01_dirty - accrued,
            "accrued": accrued, "df_cash_settle": float(zc.df(_t(trade, c.cash_settle)))}


def par_spread(c: CDSContract, zc: ZeroCurve, hc: HazardCurve) -> float:
    L = cds_legs(c, zc, hc)
    return L["protection"] / L["rpv01_clean"]


def upfront(c: CDSContract, zc: ZeroCurve, hc: HazardCurve) -> float:
    """Clean upfront paid by the protection buyer, fraction of notional (x100 = points)."""
    L = cds_legs(c, zc, hc)
    dirty = (L["protection"] - c.coupon * L["rpv01_dirty"]) / L["df_cash_settle"]
    return dirty + c.coupon * L["accrued"]


def _solve_hazard(f, lo: float = 0.0) -> float:
    """Root of f(lambda) = 0 with an expanding bracket.

    The clean risky annuity turns negative at absurd hazards (the front stub's accrued exceeds the
    expected coupons), which flips par-spread signs; expanding from small values avoids that region.
    """
    flo = f(lo)
    if flo >= 0:
        return lo
    hi = 0.25
    while hi <= 64.0:
        fhi = f(hi)
        if np.isfinite(fhi) and fhi > 0:
            return float(brentq(f, lo, hi, xtol=1e-14))
        lo = hi
        hi *= 2.0
    raise ValueError("no hazard rate reproduces the target")


def flat_hazard_from_spread(spread: float, trade_date, maturity, zc: ZeroCurve, recovery: float = 0.40) -> float:
    """Flat hazard whose par spread equals `spread` (decimal)."""
    c = CDSContract(trade_date, maturity, spread, recovery)
    return _solve_hazard(lambda lam: par_spread(c, zc, HazardCurve.flat(lam)) - spread, 1e-10)


def upfront_from_quoted_spread(quoted: float, coupon: float, trade_date, maturity, zc: ZeroCurve,
                               recovery: float = 0.40) -> float:
    lam = flat_hazard_from_spread(quoted, trade_date, maturity, zc, recovery)
    return upfront(CDSContract(trade_date, maturity, coupon, recovery), zc, HazardCurve.flat(lam))


def quoted_spread_from_upfront(upf: float, coupon: float, trade_date, maturity, zc: ZeroCurve,
                               recovery: float = 0.40) -> float:
    c = CDSContract(trade_date, maturity, coupon, recovery)
    lam = _solve_hazard(lambda x: upfront(c, zc, HazardCurve.flat(x)) - upf, 1e-10)
    return par_spread(c, zc, HazardCurve.flat(lam))


def bootstrap_cds(trade_date, maturities, par_spreads, zc: ZeroCurve, recovery: float = 0.40) -> HazardCurve:
    """Piecewise-flat hazard curve repricing a term structure of par spreads (decimal)."""
    trade_date = to_date(trade_date)
    mats = [to_date(m) for m in maturities]
    order = np.argsort(mats)
    knots: list[float] = []
    lams: list[float] = []
    for i in order:
        T = _t(trade_date, mats[i]) + 1.0 / 365.0
        c = CDSContract(trade_date, mats[i], par_spreads[i], recovery)

        def f(lam):
            hc = HazardCurve(np.array(knots + [T]), np.array(lams + [lam]))
            return par_spread(c, zc, hc) - par_spreads[i]

        lam = _solve_hazard(f, 0.0)
        knots.append(T)
        lams.append(lam)
    return HazardCurve(np.array(knots), np.array(lams), "cds bootstrap")


def cds_risk(c: CDSContract, zc: ZeroCurve, quoted: float) -> dict:
    """Protection-buyer risk for a contract quoted at `quoted` (flat-hazard convention).

    cs01: change in clean upfront (fraction of notional) for +1bp in the quoted spread.
    jtd: payoff to the protection buyer on immediate default net of the upfront paid.
    """
    u0 = upfront_from_quoted_spread(quoted, c.coupon, c.trade_date, c.maturity, zc, c.recovery)
    u1 = upfront_from_quoted_spread(quoted + 1e-4, c.coupon, c.trade_date, c.maturity, zc, c.recovery)
    lam = flat_hazard_from_spread(quoted, c.trade_date, c.maturity, zc, c.recovery)
    legs = cds_legs(c, zc, HazardCurve.flat(lam))
    return {"upfront": u0, "cs01": u1 - u0, "rpv01_clean": legs["rpv01_clean"], "hazard": lam,
            "jtd": (1.0 - c.recovery) - u0 - c.coupon * legs["accrued"]}


def standard_maturity(trade_date, years: int) -> Date:
    """Standard CDS maturity: semiannual roll on 20 Mar / 20 Sep (post-2015 convention)."""
    d = to_date(trade_date)
    if (d.month, d.day) >= (9, 20):
        base = dt.date(d.year, 12, 20)
    elif (d.month, d.day) >= (3, 20):
        base = dt.date(d.year, 6, 20)
    else:
        base = dt.date(d.year - 1, 12, 20)
    return dt.date(base.year + years, base.month, 20)
