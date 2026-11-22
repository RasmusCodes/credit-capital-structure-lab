"""Merton, Black-Cox, CreditGrades and the waterfall pricer."""
import numpy as np
import pytest

from creditlab.capital_structure import CapitalStructure, Claim, Entity
from creditlab.structural import black_cox, creditgrades, merton
from creditlab.structural.lognormal import (barrier_partial_moments, discounted_hit, first_passage_prob,
                                            partial_moments)
from creditlab.structural.waterfall_pricing import calibrate_structure, price_structure


def test_merton_hull_example():
    """Hull, Options, Futures and Other Derivatives: E=3, sigma_E=80%, D=10, r=5%, T=1 gives
    V0=12.40, sigma_V=21.23%, N(-d2)=12.7%, market value of debt 9.40."""
    res = merton.calibrate(3.0, 0.80, 10.0, 1.0, 0.05)
    assert res.V == pytest.approx(12.40, abs=0.01)
    assert res.sigma_V == pytest.approx(0.2123, abs=0.0005)
    assert res.pd_q == pytest.approx(0.127, abs=0.001)
    assert res.B == pytest.approx(9.40, abs=0.01)


def test_merton_identities_and_tranches():
    V, s, D, T, r, q = 100.0, 0.3, 70.0, 3.0, 0.04, 0.01
    res = merton.merton(V, s, D, T, r, q)
    assert res.E + res.B == pytest.approx(V * np.exp(-q * T))
    tr = merton.tranche_values(V, s, [40.0, 30.0], T, r, q)
    total = sum(t["value"] for t in tr["tranches"]) + tr["equity"]
    assert total == pytest.approx(V * np.exp(-q * T), rel=1e-12)
    assert tr["equity"] == pytest.approx(res.E, rel=1e-12)
    assert tr["tranches"][0]["spread"] < tr["tranches"][1]["spread"]


def _mc_paths(V0, mu, s, T, n_paths=60000, n_steps=400, seed=7):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    x = np.full(n_paths, np.log(V0))
    xs = [x.copy()]
    for _ in range(n_steps):
        x = x + (mu - 0.5 * s * s) * dt + s * np.sqrt(dt) * rng.standard_normal(n_paths)
        xs.append(x.copy())
    return np.array(xs), dt


def test_first_passage_and_barrier_moments_vs_monte_carlo():
    V0, mu, s, T, H0, g = 100.0, 0.03, 0.35, 2.0, 70.0, 0.01
    xs, dt = _mc_paths(V0, mu, s, T)
    t = np.arange(xs.shape[0]) * dt
    logH = np.log(H0) + g * t
    # Brownian-bridge correction for crossings between steps
    rng = np.random.default_rng(11)
    a = xs[:-1] - logH[:-1, None]
    b = xs[1:] - logH[1:, None]
    p_cross = np.exp(-2 * np.maximum(a, 0) * np.maximum(b, 0) / (s * s * dt))
    hit = (a <= 0) | (b <= 0) | (rng.random(a.shape) < p_cross)
    hit_any = hit.any(axis=0)
    p_mc = hit_any.mean()
    p_cf = float(first_passage_prob(V0, mu, s, T, H0, g))
    assert p_cf == pytest.approx(p_mc, abs=4 * np.sqrt(p_mc * (1 - p_mc) / len(hit_any)) + 0.003)
    VT = np.exp(xs[-1])
    lo, hi = 90.0, 150.0
    mask = (~hit_any) & (VT > lo) & (VT <= hi)
    p, ev = barrier_partial_moments(V0, mu, s, T, H0, g, lo, hi)
    assert float(p) == pytest.approx(mask.mean(), abs=0.01)
    assert float(ev) == pytest.approx((VT * mask).mean(), rel=0.03)


def test_discounted_hit_vs_quadrature():
    from creditlab.structural.lognormal import hit_density
    V0, mu, s, T, H0, r = 100.0, 0.02, 0.4, 3.0, 60.0, 0.05
    t = np.linspace(1e-6, T, 200001)
    f = hit_density(V0, mu, s, t, H0) * np.exp(-r * t)
    val = np.trapezoid(f, t)
    assert discounted_hit(V0, mu, s, T, H0, r) == pytest.approx(val, rel=1e-5)


def test_black_cox_reduces_to_merton_without_barrier():
    V, s, D, T, r = 100.0, 0.3, 80.0, 4.0, 0.04
    bc = black_cox.black_cox(V, s, D, T, r, H0=1e-9, g=0.0)
    m = merton.merton(V, s, D, T, r)
    assert bc.E == pytest.approx(m.E, rel=1e-8)
    assert bc.B == pytest.approx(m.B, rel=1e-8)


def test_black_cox_value_conservation():
    """With no bankruptcy cost, equity + debt = asset value (q = 0)."""
    V, s, D, T, r = 100.0, 0.35, 75.0, 5.0, 0.04
    bc = black_cox.black_cox(V, s, D, T, r, H0=55.0, g=0.0, alpha=0.0)
    assert bc.E + bc.B == pytest.approx(V, rel=1e-9)
    assert bc.pd_q > merton.merton(V, s, D, T, r).pd_q


def test_creditgrades_closed_form_matches_numeric():
    args = dict(S=20.0, sigma_S=0.7, D=30.0, r=0.04, R=0.4)
    for T in (1.0, 5.0):
        cf = creditgrades.cds_spread(T, **args)
        nu = creditgrades.cds_spread_numeric(T, **args)
        assert cf == pytest.approx(nu, abs=0.5e-4)


def _two_tranche(senior=40.0, junior=30.0):
    return CapitalStructure(
        "toy", [Entity("top", "Top", None, 1.0)], [],
        [Claim("sr", "Senior", "senior_unsecured", "top", senior, rank=10),
         Claim("jr", "Junior", "subordinated", "top", junior, rank=20),
         Claim("eq", "Common", "common", "top")])


def test_waterfall_pricer_reproduces_multi_tranche_merton():
    cs = _two_tranche()
    V, s, T, r = 100.0, 0.45, 3.0, 0.04
    sp = price_structure(cs, V, s, T, r)
    tr = merton.tranche_values(V, s, [40.0, 30.0], T, r)
    assert sp.values["sr"] == pytest.approx(tr["tranches"][0]["value"], rel=1e-7)
    assert sp.values["jr"] == pytest.approx(tr["tranches"][1]["value"], rel=1e-7)
    assert sp.values["eq"] == pytest.approx(tr["equity"], rel=1e-7)
    assert sp.pd_q == pytest.approx(merton.merton(V, s, 70.0, T, r).pd_q, abs=1e-8)


def test_waterfall_pricer_black_cox_matches_closed_form():
    cs = CapitalStructure("one", [Entity("top", "Top", None, 1.0)], [],
                          [Claim("d", "Debt", "senior_unsecured", "top", 75.0), Claim("eq", "Common", "common", "top")])
    V, s, T, r, H0 = 100.0, 0.35, 5.0, 0.04, 55.0
    sp = price_structure(cs, V, s, T, r, model="black_cox", H0=H0, g=0.0)
    bc = black_cox.black_cox(V, s, 75.0, T, r, H0=H0, g=0.0)
    assert sp.values["d"] == pytest.approx(bc.B, rel=1e-6)
    assert sp.values["eq"] == pytest.approx(bc.E, rel=1e-6)
    assert sp.pd_q == pytest.approx(bc.pd_q, abs=1e-7)


def test_calibrate_structure_matches_merton_calibration():
    cs = CapitalStructure("hull", [Entity("top", "Top", None, 1.0)], [],
                          [Claim("d", "Debt", "senior_unsecured", "top", 10.0), Claim("eq", "Common", "common", "top")])
    sp = calibrate_structure(cs, 3.0, 0.80, 1.0, 0.05)
    assert sp.V0 == pytest.approx(12.40, abs=0.01)
    assert sp.sigma == pytest.approx(0.2123, abs=0.001)
    assert sp.extra["converged"]


def test_partial_moments_sum():
    p, ev = partial_moments(100.0, 0.05, 0.3, 2.0, 0.0, np.inf)
    assert float(p) == pytest.approx(1.0)
    assert float(ev) == pytest.approx(100 * np.exp(0.1))
