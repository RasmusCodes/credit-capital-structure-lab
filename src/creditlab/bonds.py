"""Fixed-rate bond analytics.

Conventions
-----------
* Prices are per 100 face. `clean` excludes accrued interest; `dirty` includes it.
* Street-convention yield: PV = sum_k CF_k / (1 + y/f)^(k-1+w), w = remaining fraction of the current
  coupon period on the bond's day count (30/360 for US corporates).
* Curve pricing discounts each cash flow at the risk-free zero curve (time in ACT/365F from settlement);
  the Z-spread is the constant continuously compounded spread added to the zero curve that reprices the
  dirty price.
* Spread DV01 / CS01 = price change for a 1bp parallel increase in the Z-spread (per 100 face), sign
  reported as a positive number for a long position's loss.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from .curves import ZeroCurve
from .daycount import Date, coupon_dates, thirty360, to_date, year_frac


@dataclass
class FixedBond:
    name: str
    coupon: float                  # annual coupon rate, decimal (0.0925 = 9.25%)
    maturity: Date
    freq: int = 2
    face: float = 100.0
    day_count: str = "30/360"
    call_schedule: tuple = ()      # ((date, call price per 100), ...)
    issuer: str = ""
    seniority: str = ""
    currency: str = "USD"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.maturity = to_date(self.maturity)
        self.call_schedule = tuple((to_date(d), float(p)) for d, p in self.call_schedule)

    # ------------------------------------------------------------------ schedule
    def accrual_fraction(self, settle: Date) -> tuple[float, Date, list[Date]]:
        """Fraction of the current coupon period already accrued, plus the schedule."""
        settle = to_date(settle)
        prev, future = coupon_dates(self.maturity, self.freq, settle)
        nxt = future[0]
        if self.day_count.upper().startswith("30"):
            accrued = thirty360(prev, settle)
            period = 1.0 / self.freq
            frac = accrued / period
        else:
            frac = (settle - prev).days / max((nxt - prev).days, 1)
        return min(max(frac, 0.0), 1.0), prev, future

    def accrued(self, settle) -> float:
        frac, _, _ = self.accrual_fraction(settle)
        return self.coupon / self.freq * self.face * frac

    def cashflows(self, settle) -> tuple[list[Date], np.ndarray]:
        _, _, future = self.accrual_fraction(settle)
        amounts = np.full(len(future), self.coupon / self.freq * self.face)
        amounts[-1] += self.face
        return future, amounts

    def times(self, settle) -> np.ndarray:
        settle = to_date(settle)
        dates, _ = self.cashflows(settle)
        return np.array([year_frac(settle, d) for d in dates])

    # ------------------------------------------------------------------ yield convention
    def dirty_from_yield(self, y: float, settle) -> float:
        frac, _, future = self.accrual_fraction(settle)
        w = 1.0 - frac
        _, cf = self.cashflows(settle)
        k = np.arange(len(cf))
        return float(np.sum(cf / (1.0 + y / self.freq) ** (k + w)))

    def clean_from_yield(self, y: float, settle) -> float:
        return self.dirty_from_yield(y, settle) - self.accrued(settle)

    def yield_from_clean(self, clean: float, settle) -> float:
        target = clean + self.accrued(settle)
        f = lambda y: self.dirty_from_yield(y, settle) - target
        return float(brentq(f, -0.9 * self.freq, 10.0, xtol=1e-12, maxiter=500))

    def risk_from_yield(self, y: float, settle) -> dict:
        frac, _, _ = self.accrual_fraction(settle)
        w = 1.0 - frac
        _, cf = self.cashflows(settle)
        k = np.arange(len(cf))
        tk = (k + w) / self.freq
        disc = (1.0 + y / self.freq) ** (k + w)
        pv = cf / disc
        p = pv.sum()
        mac = float((tk * pv).sum() / p)
        mod = mac / (1.0 + y / self.freq)
        conv = float((tk * (tk + 1.0 / self.freq) * pv).sum() / (p * (1.0 + y / self.freq) ** 2))
        return {"dirty": float(p), "macaulay": mac, "modified": mod, "convexity": conv,
                "dv01": mod * p / 1e4}

    # ------------------------------------------------------------------ yield to call / worst
    def yield_to_date(self, clean: float, settle, redemption_date: Date, redemption_price: float) -> float:
        """Yield assuming redemption at `redemption_price` on `redemption_date` (coupon dates only approx.)."""
        alt = FixedBond(self.name, self.coupon, redemption_date, self.freq, self.face, self.day_count)
        target = clean + self.accrued(settle)

        def dirty(y):
            frac, _, fut = alt.accrual_fraction(settle)
            w = 1.0 - frac
            cf = np.full(len(fut), self.coupon / self.freq * self.face)
            cf[-1] += redemption_price * self.face / 100.0
            k = np.arange(len(cf))
            return float(np.sum(cf / (1.0 + y / self.freq) ** (k + w)))

        return float(brentq(lambda y: dirty(y) - target, -0.9 * self.freq, 10.0, xtol=1e-12))

    def yield_to_worst(self, clean: float, settle) -> tuple[float, str]:
        settle = to_date(settle)
        best = (self.yield_from_clean(clean, settle), "maturity")
        for d, px in self.call_schedule:
            if d > settle + dt.timedelta(days=30) and d < self.maturity:
                y = self.yield_to_date(clean, settle, d, px)
                if y < best[0]:
                    best = (y, f"call {d.isoformat()} @ {px:g}")
        return best

    # ------------------------------------------------------------------ curve pricing
    def dirty_from_curve(self, curve: ZeroCurve, settle, zspread: float = 0.0) -> float:
        t = self.times(settle)
        _, cf = self.cashflows(settle)
        return float(np.sum(cf * curve.df(t) * np.exp(-zspread * t)))

    def zspread(self, clean: float, curve: ZeroCurve, settle) -> float:
        target = clean + self.accrued(settle)
        f = lambda z: self.dirty_from_curve(curve, settle, z) - target
        return float(brentq(f, -0.5, 5.0, xtol=1e-12, maxiter=500))

    def spread_risk(self, clean: float, curve: ZeroCurve, settle) -> dict:
        z = self.zspread(clean, curve, settle)
        p0 = self.dirty_from_curve(curve, settle, z)
        up = self.dirty_from_curve(curve, settle, z + 1e-4)
        dn = self.dirty_from_curve(curve, settle, z - 1e-4)
        cs01 = (dn - up) / 2.0
        spread_conv = (up + dn - 2 * p0) / (p0 * 1e-8)
        return {"zspread": z, "cs01": cs01, "spread_duration": cs01 / p0 * 1e4, "spread_convexity": spread_conv}

    def gspread(self, clean: float, curve: ZeroCurve, settle) -> float:
        """Yield minus the interpolated par Treasury yield at the bond's maturity (semiannual, decimal)."""
        from .curves import par_yield
        y = self.yield_from_clean(clean, settle)
        T = year_frac(to_date(settle), self.maturity)
        return y - par_yield(curve, max(T, 0.5))


def carry_rolldown(bond: FixedBond, clean: float, settle, horizon_years: float, curve: ZeroCurve,
                   spread_at_maturity=None, funding_rate: float | None = None) -> dict:
    """Horizon return decomposition under static curves.

    spread_at_maturity: optional callable T -> Z-spread (decimal) describing the issuer's spread curve
    across remaining maturity. If None, the bond's own Z-spread is held constant (no spread roll-down).
    funding_rate: repo / funding rate (decimal); defaults to the risk-free zero rate at the horizon.
    Returns income (coupon accrual), roll (clean price change from ageing), funding cost and total, all
    per 100 face over the horizon and annualised as a percent of the dirty price.
    """
    settle = to_date(settle)
    h_days = int(round(horizon_years * 365))
    hor = settle + dt.timedelta(days=h_days)
    if hor >= bond.maturity:
        raise ValueError("horizon beyond maturity")
    z0 = bond.zspread(clean, curve, settle)
    dirty0 = clean + bond.accrued(settle)
    T_left = year_frac(hor, bond.maturity)
    z_h = z0 if spread_at_maturity is None else float(spread_at_maturity(T_left))
    if spread_at_maturity is not None:
        z_now_curve = float(spread_at_maturity(year_frac(settle, bond.maturity)))
        z_h = z_h + (z0 - z_now_curve)  # keep the bond's own residual vs its curve
    # static curve in maturity space: price at horizon on the same zero curve
    dirty_h = bond.dirty_from_curve(curve, hor, z_h)
    clean_h = dirty_h - bond.accrued(hor)
    dates, cfs = bond.cashflows(settle)
    coupons = float(sum(c for d, c in zip(dates, cfs) if d <= hor))
    income = coupons + bond.accrued(hor) - bond.accrued(settle)
    roll = clean_h - clean
    fr = float(curve.zero(max(horizon_years, 1e-3))) if funding_rate is None else funding_rate
    funding = dirty0 * fr * horizon_years
    total = income + roll - funding
    ann = lambda x: x / dirty0 / horizon_years * 100.0
    return {"z0": z0, "z_h": z_h, "income": income, "roll": roll, "funding": funding, "total": total,
            "income_pct_pa": ann(income), "roll_pct_pa": ann(roll), "funding_pct_pa": ann(funding),
            "total_pct_pa": ann(total), "clean_h": clean_h}
