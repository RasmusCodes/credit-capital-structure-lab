"""Relative-value analytics across one issuer's securities (and across issuers).

Every comparison here is put in a common currency - default probability or spread at an explicit
recovery - and comes with the reasons it may NOT be an arbitrage:

* bonds vs CDS: the basis (CDS minus the bond's par-equivalent CDS spread) is a funding and liquidity
  premium as much as a mispricing; negative-basis trades need repo financing that can vanish in stress;
* equity vs credit: structural-model spreads depend on the asset-volatility and barrier assumptions, and
  equity can stay "rich" to credit for long periods (Yu 2006, Duarte-Longstaff-Yu 2007);
* convertibles vs equity/options: the convert's implied volatility depends on the hazard rate assumed,
  converts trade with less liquidity, and call/put/make-whole terms change the option;
* maturities within an issuer: curve residuals can reflect coupon/call differences or index eligibility;
* senior vs subordinated: spreads should scale with loss given default; the waterfall says by how much,
  but plan negotiations rarely follow absolute priority exactly.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np

from . import cds as cdsmod
from .bonds import FixedBond, carry_rolldown
from .curves import ZeroCurve
from .daycount import to_date, year_frac
from .hazard import HazardCurve, fit_hazard_curve, implied_flat_hazard, risky_dirty_price


@dataclass
class BondQuote:
    claim_id: str
    bond: FixedBond
    clean: float
    trade_date: dt.date
    source: str
    recovery: float                  # recovery used to read default risk from this bond
    alt_prices: list = field(default_factory=list)   # other quotes (e.g. Reg S line) for dispersion
    entity: str = ""
    seniority: str = ""
    stock_move_since_trade: float | None = None


def bond_analytics(q: BondQuote, settle, zc: ZeroCurve, recovery_grid=(0.0, 0.2, 0.4, 0.6)) -> dict:
    b, px = q.bond, q.clean
    settle = to_date(settle)
    T = year_frac(settle, b.maturity)
    out = {"claim_id": q.claim_id, "name": b.name, "clean": px, "trade_date": q.trade_date.isoformat(),
           "maturity": b.maturity.isoformat(), "T": T, "coupon": b.coupon, "source": q.source,
           "recovery_used": q.recovery}
    out["ytm"] = b.yield_from_clean(px, settle)
    ytw, how = b.yield_to_worst(px, settle)
    out["ytw"], out["ytw_basis"] = ytw, how
    risk = b.risk_from_yield(out["ytm"], settle)
    out.update({"mod_duration": risk["modified"], "convexity": risk["convexity"], "dv01": risk["dv01"]})
    sr = b.spread_risk(px, zc, settle)
    out.update({"zspread": sr["zspread"], "cs01": sr["cs01"], "spread_duration": sr["spread_duration"]})
    fit = implied_flat_hazard(b, px, settle, zc, q.recovery)
    lam = float(fit.curve.hazards[0])
    out["hazard"] = lam
    out["hazard_status"] = fit.status
    for h in (1, 3, 5):
        out[f"pd_{h}y"] = float(1 - np.exp(-lam * h)) if np.isfinite(lam) else 1.0
    out["pd_to_maturity"] = float(1 - np.exp(-lam * T)) if np.isfinite(lam) else 1.0
    out["pd_by_recovery"] = {}
    for R in recovery_grid:
        f = implied_flat_hazard(b, px, settle, zc, R)
        l = float(f.curve.hazards[0])
        out["pd_by_recovery"][f"{R:.1f}"] = float(1 - np.exp(-l * min(T, 5.0))) if np.isfinite(l) else 1.0
    # par-equivalent CDS spread: CDS par spread to the bond's maturity under the bond-implied hazard
    if fit.status == "ok":
        c = cdsmod.CDSContract(settle - dt.timedelta(days=1), b.maturity, 0.01, q.recovery)
        out["pecs"] = cdsmod.par_spread(c, zc, HazardCurve.flat(lam))
    else:
        out["pecs"] = None
    if q.alt_prices:
        allp = [px] + list(q.alt_prices)
        out["price_dispersion"] = float(max(allp) - min(allp))
    out["stale_days"] = max((settle - q.trade_date).days - 3, 0)   # settlement is T+1 business day after the market date
    out["stock_move_since_trade"] = q.stock_move_since_trade
    return out


def issuer_curve(quotes: list[BondQuote], settle, zc: ZeroCurve, smooth: float = 0.0):
    """Common hazard curve through all bonds of one issuer (recoveries per bond); residuals in bp.

    Knots: up to three, at maturities spread across the curve, so that residuals are not forced to zero.
    """
    settle = to_date(settle)
    bonds = [q.bond for q in quotes]
    prices = [q.clean for q in quotes]
    recs = [q.recovery for q in quotes]
    T = sorted(year_frac(settle, b.maturity) for b in bonds)
    if len(T) <= 2:
        knots = [T[-1]]
    else:
        knots = sorted({T[len(T) // 3], T[-1]})
    hc, resid = fit_hazard_curve(bonds, prices, settle, zc, recs, knots, smooth=smooth)
    return hc, resid


def cds_basis(analytics: dict, cds_spread: float, cds_date, cds_maturity=None) -> dict:
    """CDS-bond basis in bp. Negative basis: the bond trades cheap to CDS (positive non-default premium)."""
    pecs = analytics.get("pecs")
    out = {"cds_bp": cds_spread * 1e4, "pecs_bp": None if pecs is None else pecs * 1e4,
           "z_bp": analytics["zspread"] * 1e4, "cds_date": str(cds_date)}
    if pecs is not None:
        out["basis_bp"] = (cds_spread - pecs) * 1e4
    out["z_basis_bp"] = (cds_spread - analytics["zspread"]) * 1e4
    return out


def seniority_check(senior: dict, junior: dict, rec_senior: float, rec_junior: float) -> dict:
    """Compare two bonds of one issuer with different seniority.

    If both carry the same default event, spread_j / spread_s should be close to (1-R_j)/(1-R_s). A ratio
    far above (below) suggests the junior is cheap (rich) relative to the waterfall's recoveries -- or that
    the market disagrees with the waterfall.
    """
    zs, zj = senior["zspread"], junior["zspread"]
    implied = zj / zs if zs > 0 else np.nan
    model = (1 - rec_junior) / (1 - rec_senior) if rec_senior < 1 else np.inf
    return {"spread_ratio_market": implied, "lgd_ratio_waterfall": model,
            "hazard_senior": senior["hazard"], "hazard_junior": junior["hazard"],
            "hazard_gap": junior["hazard"] - senior["hazard"]}


def equity_vs_credit(model_spread: float, market_spread: float) -> dict:
    """Structural-model (equity-implied) spread vs market spread at the same maturity and recovery."""
    return {"model_bp": model_spread * 1e4, "market_bp": market_spread * 1e4,
            "gap_bp": (market_spread - model_spread) * 1e4,
            "ratio": market_spread / model_spread if model_spread > 0 else np.inf}


def carry_table(quotes: list[BondQuote], settle, zc: ZeroCurve, hc_issuer: HazardCurve | None = None,
                horizon: float = 1.0, funding_spread: float = 0.0) -> list[dict]:
    """1-year carry and roll-down per bond, with expected default loss under the issuer hazard curve.

    Roll-down uses the issuer's fitted curve shape (z-spread as a function of maturity) if given.
    expected_loss_pct: probability of default over the horizon x (price - recovery) / price, i.e. the
    loss a buy-and-hold position takes if the market's (risk-neutral) default probability is right.
    """
    settle = to_date(settle)
    out = []
    spread_fn = None
    if hc_issuer is not None:
        Ts = np.linspace(0.25, 12.0, 48)

        def z_at(T):
            fake = FixedBond("x", 0.08, settle + dt.timedelta(days=int(T * 365)))
            dirty = risky_dirty_price(fake, settle, zc, hc_issuer, 0.35)
            return fake.zspread(dirty - fake.accrued(settle), zc, settle)

        zs = np.array([z_at(T) for T in Ts])
        spread_fn = lambda T: float(np.interp(T, Ts, zs))
    for q in quotes:
        try:
            cr = carry_rolldown(q.bond, q.clean, settle, horizon, zc, spread_fn,
                                funding_rate=float(zc.zero(horizon)) + funding_spread)
        except ValueError:
            continue
        lam = implied_flat_hazard(q.bond, q.clean, settle, zc, q.recovery).curve.hazards[0]
        pd_h = float(1 - np.exp(-lam * horizon)) if np.isfinite(lam) else 1.0
        dirty = q.clean + q.bond.accrued(settle)
        el = pd_h * max(dirty - q.recovery * q.bond.face, 0.0) / dirty * 100
        cr.update({"claim_id": q.claim_id, "name": q.bond.name, "pd_horizon": pd_h, "expected_loss_pct": el,
                   "carry_minus_el_pct": cr["total_pct_pa"] - el / horizon})
        out.append(cr)
    return out


def cross_issuer_table(rows: list[dict]) -> list[dict]:
    """Rows of {issuer, leverage, model_pd_5y, market_pd_5y, spread_bp}. Adds the market/model PD ratio
    and a rank, a first screen for issuers whose markets price risk out of line with fundamentals."""
    out = []
    for r in rows:
        rr = dict(r)
        mp, kp = r.get("market_pd_5y"), r.get("model_pd_5y")
        rr["pd_ratio"] = mp / kp if mp is not None and kp not in (None, 0) else None
        out.append(rr)
    ranked = sorted([r for r in out if r["pd_ratio"] is not None], key=lambda r: r["pd_ratio"])
    for i, r in enumerate(ranked):
        r["rank_market_vs_model"] = i + 1
    return out
