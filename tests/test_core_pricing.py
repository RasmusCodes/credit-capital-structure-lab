"""Curves, bonds, hazard-rate pricing and CDS."""
import datetime as dt
import csv
from pathlib import Path

import numpy as np
import pytest

from creditlab import cds
from creditlab.bonds import FixedBond, carry_rolldown
from creditlab.curves import TREASURY_TENORS, ZeroCurve, par_yield
from creditlab.hazard import (HazardCurve, bootstrap_hazard, fit_hazard_curve, implied_flat_hazard,
                              risky_dirty_price)

ROOT = Path(__file__).resolve().parents[1]


def ust_curve(date: str) -> ZeroCurve:
    rows = list(csv.DictReader(open(ROOT / "data" / "snapshots" / "ust_par_curve_2026.csv")))
    row = [r for r in rows if r["Date"] == date][0]
    return ZeroCurve.from_treasury_par({TREASURY_TENORS[k]: float(v) for k, v in row.items()
                                        if k in TREASURY_TENORS and v})


# ------------------------------------------------------------------ curves
def test_flat_curve():
    zc = ZeroCurve.flat(0.05)
    assert zc.df(2.0) == pytest.approx(np.exp(-0.1))
    assert zc.forward(1.0, 3.0) == pytest.approx(0.05)


def test_treasury_bootstrap_reprices_par_yields():
    tenors = {0.25: 4.0, 0.5: 4.05, 1.0: 4.1, 2.0: 4.0, 3.0: 3.95, 5.0: 4.0, 7.0: 4.15, 10.0: 4.3, 30.0: 4.8}
    zc = ZeroCurve.from_treasury_par(tenors)
    for T in (2.0, 3.0, 5.0, 7.0, 10.0, 30.0):
        assert par_yield(zc, T) == pytest.approx(tenors[T] / 100, abs=1e-9)


# ------------------------------------------------------------------ bonds
def test_par_bond_at_coupon_date():
    b = FixedBond("10% 5y", 0.10, dt.date(2031, 6, 15))
    settle = dt.date(2026, 6, 15)
    assert b.accrued(settle) == pytest.approx(0.0)
    assert b.clean_from_yield(0.10, settle) == pytest.approx(100.0, abs=1e-10)
    assert b.yield_from_clean(100.0, settle) == pytest.approx(0.10, abs=1e-12)


def test_accrued_30_360():
    b = FixedBond("6%", 0.06, dt.date(2030, 3, 1))
    # 2 months after 1 Sep 2026 coupon: 60/360 * 6 = 1.0
    assert b.accrued(dt.date(2026, 11, 1)) == pytest.approx(1.0)


def test_duration_convexity_match_finite_differences():
    b = FixedBond("9.25% 2030", 0.0925, dt.date(2030, 6, 1))
    settle = dt.date(2026, 11, 24)
    y = 0.13
    r = b.risk_from_yield(y, settle)
    h = 1e-5
    pu, pd_, p0 = b.dirty_from_yield(y + h, settle), b.dirty_from_yield(y - h, settle), b.dirty_from_yield(y, settle)
    assert r["modified"] == pytest.approx(-(pu - pd_) / (2 * h) / p0, rel=1e-6)
    assert r["convexity"] == pytest.approx((pu + pd_ - 2 * p0) / (h * h) / p0, rel=1e-4)


def test_zspread_round_trip():
    zc = ZeroCurve.flat(0.04)
    b = FixedBond("8% 2031", 0.08, dt.date(2031, 2, 1))
    settle = dt.date(2026, 11, 24)
    dirty = b.dirty_from_curve(zc, settle, 0.0612)
    assert b.zspread(dirty - b.accrued(settle), zc, settle) == pytest.approx(0.0612, abs=1e-10)


def test_carry_rolldown_flat_world():
    zc = ZeroCurve.flat(0.04)
    b = FixedBond("4% 2036", 0.04, dt.date(2036, 11, 24))
    settle = dt.date(2026, 11, 24)
    clean = b.dirty_from_curve(zc, settle) - b.accrued(settle)
    out = carry_rolldown(b, clean, settle, 0.5, zc, funding_rate=0.0)
    # on a flat curve at zero spread the total return is the risk-free rate (income + pull to par)
    assert out["total_pct_pa"] == pytest.approx(100 * (np.exp(0.04 * 0.5) - 1) / 0.5, rel=0.03)


# ------------------------------------------------------------------ hazard
def test_piecewise_hazard_integral():
    hc = HazardCurve(np.array([1.0, 3.0]), np.array([0.1, 0.3]))
    assert hc.cum_hazard(0.5) == pytest.approx(0.05)
    assert hc.cum_hazard(2.0) == pytest.approx(0.1 + 0.3)
    assert hc.cum_hazard(5.0) == pytest.approx(0.1 + 0.6 + 0.6)  # flat extension
    assert hc.marginal_pd(1.0, 2.0) == pytest.approx(1 - np.exp(-0.3))


def test_rmv_zero_coupon_spread_is_loss_rate():
    zc = ZeroCurve.flat(0.03)
    b = FixedBond("zero", 0.0, dt.date(2031, 11, 24), freq=1)
    settle = dt.date(2026, 11, 24)
    lam, R = 0.08, 0.35
    p = risky_dirty_price(b, settle, zc, HazardCurve.flat(lam), R, "RMV")
    T = b.times(settle)[-1]
    assert -np.log(p / 100) / T - 0.03 == pytest.approx(lam * (1 - R), abs=1e-12)


def test_rfv_zero_recovery_equals_survival_discounting():
    zc = ZeroCurve.flat(0.03)
    b = FixedBond("7%", 0.07, dt.date(2030, 5, 1))
    settle = dt.date(2026, 11, 24)
    hc = HazardCurve.flat(0.1)
    t = b.times(settle)
    _, cf = b.cashflows(settle)
    expected = float(np.sum(cf * zc.df(t) * hc.survival(t)))
    assert risky_dirty_price(b, settle, zc, hc, 0.0) == pytest.approx(expected, rel=1e-12)


def test_implied_hazard_round_trip_and_flags():
    zc = ZeroCurve.flat(0.04)
    b = FixedBond("9% 2031", 0.09, dt.date(2031, 2, 1))
    settle = dt.date(2026, 11, 24)
    for R in (0.0, 0.3, 0.6):
        dirty = risky_dirty_price(b, settle, zc, HazardCurve.flat(0.12), R)
        fit = implied_flat_hazard(b, dirty - b.accrued(settle), settle, zc, R)
        assert fit.status == "ok" and fit.curve.hazards[0] == pytest.approx(0.12, abs=1e-9)
    assert implied_flat_hazard(b, 125.0, settle, zc, 0.4).status == "price_above_riskless"
    assert implied_flat_hazard(b, 30.0, settle, zc, 0.4).status == "price_below_recovery"


def test_bootstrap_and_fit_recover_piecewise_curve():
    zc = ZeroCurve.flat(0.04)
    settle = dt.date(2026, 11, 24)
    bonds = [FixedBond("A", 0.08, dt.date(2028, 6, 1)), FixedBond("B", 0.09, dt.date(2030, 6, 1)),
             FixedBond("C", 0.095, dt.date(2032, 6, 1))]
    T = [b.times(settle)[-1] for b in bonds]
    true = HazardCurve(np.array(T), np.array([0.20, 0.12, 0.08]))
    cleans = [risky_dirty_price(b, settle, zc, true, 0.35) - b.accrued(settle) for b in bonds]
    hc, diag = bootstrap_hazard(bonds, cleans, settle, zc, 0.35)
    assert np.allclose(hc.hazards, [0.20, 0.12, 0.08], atol=1e-8)
    fitted, resid = fit_hazard_curve(bonds, cleans, settle, zc, 0.35, knots=T)
    assert max(abs(r["price_resid"]) for r in resid) < 1e-5


# ------------------------------------------------------------------ CDS
def test_cds_reproduces_markit_coreweave_upfront():
    """Markit 5Y official close 22-Oct-2026 (LSEG CDSOV): conventional spread 1,167.011bp, coupon 500bp,
    R 40%, upfront 19.834%. Our converter on the Treasury curve must agree within 0.05 points."""
    zc = ust_curve("2026-10-22")
    mat = cds.standard_maturity("2026-10-22", 5)
    assert mat == dt.date(2031, 12, 20)
    u = cds.upfront_from_quoted_spread(0.1167011, 0.05, "2026-10-22", mat, zc, 0.40)
    assert u * 100 == pytest.approx(19.834, abs=0.05)
    q = cds.quoted_spread_from_upfront(0.19834, 0.05, "2026-10-22", mat, zc, 0.40)
    assert q * 1e4 == pytest.approx(1167.011, abs=3.0)


def test_cds_par_spread_near_credit_triangle_and_bootstrap():
    zc = ZeroCurve.flat(0.04)
    trade = dt.date(2026, 11, 20)
    mat = cds.standard_maturity(trade, 5)
    lam, R = 0.05, 0.4
    s = cds.par_spread(cds.CDSContract(trade, mat, 0.01, R), zc, HazardCurve.flat(lam))
    assert s == pytest.approx(lam * (1 - R), rel=0.01)
    mats = [cds.standard_maturity(trade, y) for y in (1, 3, 5)]
    spreads = [0.030, 0.025, 0.022]
    hc = cds.bootstrap_cds(trade, mats, spreads, zc, R)
    for m, sp in zip(mats, spreads):
        assert cds.par_spread(cds.CDSContract(trade, m, sp, R), zc, hc) == pytest.approx(sp, abs=1e-10)
