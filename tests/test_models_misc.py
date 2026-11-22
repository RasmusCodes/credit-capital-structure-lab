"""Rating transitions, survival analysis, options, convertibles, spreads and scenarios."""
import datetime as dt

import numpy as np
import pytest

from creditlab import options, scenarios, spreads, survival
from creditlab.bonds import FixedBond
from creditlab.convertible import Convertible, implied_hazard, implied_vol, price_convertible
from creditlab.curves import ZeroCurve
from creditlab.hazard import HazardCurve, risky_dirty_price
from creditlab.transition import TransitionMatrix, mobility_index


def toy_matrix():
    P = np.array([[0.90, 0.08, 0.015, 0.005],
                  [0.05, 0.85, 0.07, 0.03],
                  [0.01, 0.09, 0.70, 0.20],
                  [0.0, 0.0, 0.0, 1.0]])
    return TransitionMatrix(P, ["BB", "B", "CCC", "D"])


def test_transition_matrix_properties():
    tm = toy_matrix()
    cum = tm.cumulative_pd(10)
    assert np.all(np.diff(cum, axis=1) >= -1e-12)          # cumulative PDs never fall
    assert np.all(cum[:, -1] > cum[:, 0])
    assert cum[2, 0] == pytest.approx(0.20)
    Q = tm.generator()
    assert np.allclose(Q.sum(axis=1), 0.0, atol=1e-12)
    off = Q[~np.eye(4, dtype=bool)]
    assert np.all(off >= 0)
    assert np.max(np.abs(tm.at(1.0) - tm.P)) < 0.02      # IRW correction is a small perturbation
    assert tm.implied_rating(0.20, 1) == "CCC"
    assert 0 < mobility_index(tm.P) < 1


def test_from_published_removes_nr_and_scaling_round_trip():
    rows = [[85.0, 10.0, 1.0, 0.0, 4.0], [5.0, 80.0, 8.0, 2.0, 5.0], [1.0, 9.0, 60.0, 20.0, 10.0]]
    tm = TransitionMatrix.from_published(rows, ["BB", "B", "CCC", "D"], nr_col=True)
    assert np.allclose(tm.P.sum(axis=1), 1.0)
    assert tm.P[0, 0] == pytest.approx(85 / 96)
    l = tm.risk_neutral_scaling("B", 0.25, 3.0)
    from scipy.linalg import expm
    Q = tm.generator()
    Q[1, :] *= l
    assert expm(Q * 3.0)[1, -1] == pytest.approx(0.25, abs=1e-8)


def test_kaplan_meier_matches_exponential_and_greenwood():
    rng = np.random.default_rng(1)
    lam = 0.1
    t = rng.exponential(1 / lam, 5000)
    c = rng.uniform(0, 30, 5000)
    d = np.minimum(t, c)
    e = (t <= c).astype(int)
    times, S, se = survival.kaplan_meier(d, e)
    k = np.searchsorted(times, 5.0)
    assert S[k] == pytest.approx(np.exp(-lam * 5.0), abs=0.02)
    assert np.all(se >= 0)
    tt, H = survival.nelson_aalen(d, e)
    k = np.searchsorted(tt, 5.0)
    assert H[k] == pytest.approx(lam * 5.0, abs=0.03)


def test_marginal_rates_and_discrete_hazard_fit():
    cum = np.array([0.02, 0.05, 0.09])
    m = survival.marginal_from_cumulative(cum)
    assert np.prod(1 - m) == pytest.approx(1 - 0.09)
    rng = np.random.default_rng(4)
    X = rng.normal(size=(20000, 2))
    beta = np.array([-3.0, 0.8, -0.5])
    p = 1 / (1 + np.exp(-(beta[0] + X @ beta[1:])))
    y = (rng.random(20000) < p).astype(float)
    b, se = survival.fit_discrete_hazard(X, y)
    assert np.allclose(b, beta, atol=4 * se.max())


def test_options_parity_iv_and_put_bounds():
    S, K, T, r, q, v = 40.0, 35.0, 1.5, 0.04, 0.0, 0.9
    c = options.bs_price(S, K, T, r, q, v, "call")
    p = options.bs_price(S, K, T, r, q, v, "put")
    assert c - p == pytest.approx(S - K * np.exp(-r * T))
    assert options.bs_implied_vol(c, S, K, T, r, q, "call") == pytest.approx(v, abs=1e-8)
    lam, T2 = 0.15, 2.0
    U = lam / (lam + r) * (1 - np.exp(-(lam + r) * T2))
    assert options.hazard_from_urc(U, T2, r) == pytest.approx(lam, abs=1e-10)
    assert options.put_pd_upper_bound(0.5, 5.0, 2.0, 0.04) == pytest.approx(0.5 * np.exp(0.08) / 5.0)


def test_convertible_limits():
    zc = ZeroCurve.flat(0.04)
    cb = Convertible("cv", 0.02, 5.0, conversion_price=200.0, recovery=0.4)
    # deep out of the money and low vol: worth its straight-bond floor, which matches hazard pricing
    lam = 0.08
    floor_tree = price_convertible(cb, 20.0, 0.2, 0.04, 0.0, lam, steps=1500, convert=False)["price"]
    settle = dt.date(2026, 11, 23)
    b = FixedBond("same", 0.02, dt.date(2031, 11, 23))
    hz = risky_dirty_price(b, settle, zc, HazardCurve.flat(lam), 0.4, grid_per_year=365)
    assert floor_tree == pytest.approx(hz, abs=0.25)
    full = price_convertible(cb, 20.0, 0.2, 0.04, 0.0, lam, steps=600)["price"]
    floor_same = price_convertible(cb, 20.0, 0.2, 0.04, 0.0, lam, steps=600, convert=False)["price"]
    assert full >= floor_same - 1e-12
    # deep in the money: at least parity
    itm = price_convertible(cb, 400.0, 0.4, 0.04, 0.0, 0.02, steps=600)
    assert itm["price"] >= itm["parity"] - 1e-9
    # more volatility, more value; more hazard, less value (no call)
    p1 = price_convertible(cb, 150.0, 0.3, 0.04, 0.0, 0.05, steps=400)["price"]
    p2 = price_convertible(cb, 150.0, 0.6, 0.04, 0.0, 0.05, steps=400)["price"]
    p3 = price_convertible(cb, 150.0, 0.3, 0.04, 0.0, 0.15, steps=400)["price"]
    assert p2 > p1 > p3
    # a holder put at par raises the value
    cbp = Convertible("cvp", 0.0, 5.0, 200.0, puts=[(2.0, 100.0)], recovery=0.4)
    assert price_convertible(cbp, 60.0, 0.5, 0.04, 0.0, 0.1, steps=400)["price"] > price_convertible(
        Convertible("cv0", 0.0, 5.0, 200.0, recovery=0.4), 60.0, 0.5, 0.04, 0.0, 0.1, steps=400)["price"]


def test_convertible_inversions_round_trip():
    cb = Convertible("cv", 0.01, 4.0, conversion_price=60.0, recovery=0.35)
    px = price_convertible(cb, 40.0, 0.7, 0.04, 0.0, 0.12, steps=300)["price"]
    assert implied_vol(cb, px, 40.0, 0.04, 0.0, 0.12, steps=300) == pytest.approx(0.7, abs=1e-3)
    assert implied_hazard(cb, px, 40.0, 0.7, 0.04, 0.0, steps=300) == pytest.approx(0.12, abs=1e-3)


def test_recovery_adjusted_pd_and_breakeven():
    tab = spreads.pd_recovery_table(0.10, 5.0)
    pds = [r["pd"] for r in tab]
    assert all(np.diff(pds) > 0)                      # same spread, higher recovery -> higher PD
    zc = ZeroCurve.flat(0.04)
    settle = dt.date(2026, 11, 23)
    b = FixedBond("9%", 0.09, dt.date(2031, 11, 23))
    clean = risky_dirty_price(b, settle, zc, HazardCurve.flat(0.15), 0.3) - b.accrued(settle)
    target = 1 - np.exp(-0.15 * 2.0)
    R = spreads.breakeven_recovery(b, clean, settle, zc, target, 2.0)
    assert R == pytest.approx(0.3, abs=1e-5)
    d = spreads.decompose_spread(0.11, 0.10, 0.03, 0.4)
    assert d["expected_loss_bp"] == pytest.approx(180.0)
    assert d["default_risk_premium_bp"] == pytest.approx(820.0)
    assert d["non_default_bp"] == pytest.approx(100.0)


def test_jump_to_default_and_hedge_ratio():
    pos = [scenarios.Position("n", "bond", 10.0, 80.0), scenarios.Position("equity", "equity", -0.1, 40.0),
           scenarios.Position("put", "put", 2000, 1.0, strike=10.0)]
    j = scenarios.jump_to_default(pos, {"n": 0.35})
    assert j["rows"][0]["pnl"] == pytest.approx(10.0 * (35 - 80) / 100)
    assert j["rows"][1]["pnl"] == pytest.approx(0.1 * 40.0)
    assert j["rows"][2]["pnl"] == pytest.approx(2000 * 100 * 9.0 / 1e6)
    n = scenarios.hedge_ratio_jtd_neutral(1.0, 80.0, 0.35, 10.0, 1.0)
    assert n * 100 * 9.0 == pytest.approx(450000.0)
