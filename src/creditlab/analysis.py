"""Case-study analysis: run every model on one case and collect comparable outputs.

Pipeline (each step is a function so tests and the web exporter can reuse it):
  structure_summary      claims by class and entity, maturities, leverage
  recovery_curves        waterfall recoveries over a grid of enterprise values + the case's scenarios
  bond_section           market-implied hazards, PDs, curve residuals, carry/roll for straight bonds
  structural_section     Merton (KMV default point), Merton (all claims), Black-Cox, CreditGrades and the
                         waterfall-consistent structural model calibrated to the equity
  convert_section        convertible prices from the jump-to-default tree vs market
  option_section         put-implied default-probability bounds vs bond-implied PDs
All probabilities are risk-neutral unless labelled otherwise.
"""
from __future__ import annotations

import numpy as np

from . import cds as cdsmod
from .case import Case
from .convertible import implied_hazard, implied_vol, price_convertible
from .hazard import HazardCurve, implied_flat_hazard, risky_dirty_price
from .relative_value import bond_analytics, carry_table, issuer_curve
from .structural import creditgrades, merton
from .structural.waterfall_pricing import (barrier_recoveries, calibrate_structure, price_structure,
                                           structural_hazard_curve)
from .waterfall import distribute, fulcrum


def _equity_vol(case: Case) -> tuple[float, str]:
    o = case.equity.get("options") or {}
    if o.get("iv_1y"):
        return float(o["iv_1y"]), "ATM implied volatility, 1-year (listed options)"
    return float(case.equity["vol_1y"]), "1-year realised volatility"


def debt_wam(case: Case) -> float:
    d = [(c.amount, c.maturity) for c in case.structure.debt() if c.maturity is not None and c.amount > 0]
    if not d:
        return 5.0
    w = np.array([a for a, _ in d])
    m = np.array([t for _, t in d])
    return float(np.sum(w * m) / np.sum(w))


# --------------------------------------------------------------------------------------------- structure
def structure_summary(case: Case) -> dict:
    cs = case.structure
    by_kind: dict = {}
    by_entity: dict = {}
    for c in cs.claims:
        if c.kind == "common":
            continue
        by_kind[c.kind] = by_kind.get(c.kind, 0.0) + c.amount
        by_entity[c.entity] = by_entity.get(c.entity, 0.0) + c.amount
    mc = case.market_cap
    debt = cs.total_debt()
    return {"total_debt_claims": debt, "total_claims": cs.total_claims(), "market_cap": mc,
            "debt_to_market_cap": debt / mc if mc else None, "by_kind": by_kind, "by_entity": by_entity,
            "maturity_profile": cs.maturity_profile(), "wam_years": debt_wam(case),
            "default_point_1y": cs.default_point(1.0)}


# --------------------------------------------------------------------------------------------- waterfall
def recovery_curves(case: Case, n: int = 121) -> dict:
    cs = case.structure
    top = cs.total_claims() * 1.25
    V = np.linspace(0.0, top, n)
    res = distribute(cs, V)
    curves = {c.id: [None if np.isnan(x) else float(x) for x in res.recovery[c.id]] for c in cs.claims if c.kind != "common"}
    curves["equity_value"] = [float(x) for x in res.payouts[[c.id for c in cs.claims if c.kind == "common"][0]]]
    out = {"V": V.tolist(), "recovery": curves, "scenarios": {}}
    scen = case.data.get("default_value_scenarios")
    if scen:
        for k in ("low", "central", "high"):
            r = distribute(cs, np.array([float(scen[k])]))
            out["scenarios"][k] = {"V": scen[k], "recovery": {c.id: float(r.recovery[c.id][0]) for c in cs.claims
                                                             if c.kind != "common"},
                                   "fulcrum": fulcrum(cs, float(scen[k]))}
        out["scenario_basis"] = scen.get("basis")
    return out


def central_recovery(case: Case, rc: dict | None = None, fallback: float = 0.4) -> dict:
    rc = rc or recovery_curves(case)
    if "central" in rc["scenarios"]:
        return rc["scenarios"]["central"]["recovery"]
    return {c.id: fallback for c in case.structure.claims}


# --------------------------------------------------------------------------------------------- bonds
def bond_section(case: Case, recoveries: dict) -> dict:
    if not case.bonds:
        return {"bonds": [], "curve": None, "carry": []}
    for q in case.bonds:
        q.recovery = float(np.clip(recoveries.get(q.claim_id, q.recovery), 0.0, 0.95))
    rows = [bond_analytics(q, case.settle, case.zc) for q in case.bonds]
    for r in rows:
        # with recovery near 100% the spread is mostly non-default (calls, liquidity) and 1/(1-R) explodes:
        # the implied hazard says nothing about default and must not be read as a probability
        r["pd_identified"] = r["recovery_used"] < 0.85 and r["hazard_status"] == "ok"
    curve = None
    if len(case.bonds) >= 3:
        hc, resid = issuer_curve(case.bonds, case.settle, case.zc)
        curve = {"knots": hc.knots.tolist(), "hazards": hc.hazards.tolist(), "residuals": resid}
        carry = carry_table(case.bonds, case.settle, case.zc, hc)
    else:
        carry = carry_table(case.bonds, case.settle, case.zc, None)
    return {"bonds": rows, "curve": curve, "carry": carry}


# --------------------------------------------------------------------------------------------- structural
def structural_section(case: Case, recoveries: dict | None = None, horizon: float | None = None,
                       barrier_fraction: float = 1.0) -> dict:
    """Calibrate the structural models to the equity and compare their outputs with the markets."""
    cs = case.structure
    E = case.market_cap
    sE, sE_src = _equity_vol(case)
    r1 = float(case.zc.zero(1.0))
    T = horizon or min(max(debt_wam(case), 1.0), 10.0)
    rT = float(case.zc.zero(T))
    DP1 = cs.default_point(1.0)
    D_all = cs.total_debt()
    out = {"equity_value": E, "equity_vol": sE, "equity_vol_source": sE_src, "horizon": T,
           "default_point_1y": DP1, "total_debt": D_all}
    # Merton, KMV-style one-year default point
    try:
        m1 = merton.calibrate(E, sE, DP1, 1.0, r1)
        out["merton_kmv"] = {"V": m1.V, "sigma_V": m1.sigma_V, "pd_1y": m1.pd_q, "dd": m1.dd, "spread": m1.spread}
    except RuntimeError as exc:
        out["merton_kmv"] = {"error": str(exc)}
    # Merton with all debt due at the weighted-average maturity
    try:
        mT = merton.calibrate(E, sE, D_all, T, rT)
        out["merton_wam"] = {"V": mT.V, "sigma_V": mT.sigma_V, "pd_T": mT.pd_q, "spread": mT.spread,
                             "recovery_q": mT.recovery_q}
    except RuntimeError as exc:
        out["merton_wam"] = {"error": str(exc)}
    # CreditGrades (per share)
    sh = case.equity["shares_m"]
    S = case.equity["price"]
    D_ps = D_all / sh
    cg = {"D_per_share": D_ps, "Lbar": 0.5, "lam": 0.3, "R": 0.4}
    cg["survival"] = {str(t): float(creditgrades.survival(t, S, sE, D_ps)) for t in (1, 3, 5)}
    cg["pd_5y"] = 1 - cg["survival"]["5"]
    cg["cds_5y"] = creditgrades.cds_spread(5.0, S, sE, D_ps, float(case.zc.zero(5.0)))
    out["creditgrades"] = cg
    # waterfall-consistent structural model (Black-Cox with barrier at the KMV default point)
    try:
        bc = calibrate_structure(cs, E, sE, T, rT, model="black_cox", barrier_ratio=barrier_fraction * DP1 / D_all,
                                 default_point=D_all)
        H0 = bc.H0
        hc = structural_hazard_curve(bc.V0, bc.sigma, H0, 0.0, rT)
        rec_H = barrier_recoveries(cs, H0)
        model_bonds = []
        for q in case.bonds:
            R = float(np.clip(rec_H.get(q.claim_id, 0.0), 0.0, 0.95))
            dirty = risky_dirty_price(q.bond, case.settle, case.zc, hc, R)
            model_clean = dirty - q.bond.accrued(case.settle)
            model_bonds.append({"claim_id": q.claim_id, "market": q.clean, "model": model_clean,
                                "gap_points": q.clean - model_clean, "recovery_at_barrier": R})
        out["black_cox_waterfall"] = {
            "V": bc.V0, "sigma_V": bc.sigma, "H0": H0, "pd_T": bc.pd_q, "pd_barrier": bc.pd_barrier,
            "converged": bc.extra.get("converged"), "equity_model": bc.equity,
            "pd_1y": float(hc.pd(1.0)), "pd_5y": float(hc.pd(5.0)),
            "hazard_knots": hc.knots.tolist(), "hazards": hc.hazards.tolist(),
            "recovery_at_barrier": rec_H, "cond_recovery": bc.cond_recovery, "claim_values": bc.values,
            "bonds": model_bonds}
    except RuntimeError as exc:
        out["black_cox_waterfall"] = {"error": str(exc)}
    # Merton-waterfall: all claims valued at horizon T
    try:
        mw = calibrate_structure(cs, E, sE, T, rT, model="merton", default_point=D_all)
        out["merton_waterfall"] = {"V": mw.V0, "sigma_V": mw.sigma, "pd_T": mw.pd_q, "converged": mw.extra.get("converged"),
                                   "claim_values": mw.values, "yields": mw.yields, "cond_recovery": mw.cond_recovery}
    except RuntimeError as exc:
        out["merton_waterfall"] = {"error": str(exc)}
    return out


# --------------------------------------------------------------------------------------------- converts
def convert_section(case: Case, recoveries: dict, hazard_curve: HazardCurve | None, steps: int = 300) -> dict:
    """Price each convertible with the jump-to-default tree.

    Volatility: listed ATM implied vol at the convert's maturity (flat beyond the longest listed expiry).
    Hazard: average of the supplied issuer hazard curve to the convert's maturity (bond-implied if the
    issuer has straight bonds, otherwise structural), held constant.
    """
    o = case.equity.get("options") or {}
    term = o.get("atm_iv_term") or []
    S = case.equity["price"]
    rows = []
    for cv in case.converts:
        cb = cv["cb"]
        cb.recovery = float(np.clip(recoveries.get(cv["claim_id"], cb.recovery), 0.0, 0.95))
        T = cb.maturity
        if term:
            iv = float(np.interp(T, [x["T"] for x in term], [x["iv"] for x in term]))
        else:
            iv = float(case.equity["vol_1y"])
        r = float(case.zc.zero(T))
        lam = hazard_curve.average_hazard(T) if hazard_curve is not None else 0.0
        row = {"claim_id": cv["claim_id"], "name": cb.name, "market": cv["price"], "trade_date": cv["trade_date"],
               "T": T, "conversion_price": cb.conversion_price, "parity": cb.ratio * S, "vol_used": iv,
               "hazard_used": lam, "recovery": cb.recovery, "stock_move_since_trade": cv.get("stock_move_since_trade")}
        model = price_convertible(cb, S, iv, r, 0.0, lam, steps=steps)
        row.update({"model": model["price"], "delta_pct_parity": model.get("delta_pct_parity")})
        row["bond_floor"] = price_convertible(cb, S, iv, r, 0.0, lam, steps=steps, convert=False)["price"]
        if cv["price"] is not None:
            row["gap_points"] = cv["price"] - model["price"]
            row["market_option_value"] = cv["price"] - row["bond_floor"]
            row["model_option_value"] = model["price"] - row["bond_floor"]
            row["implied_vol_at_hazard"] = implied_vol(cb, cv["price"], S, r, 0.0, lam, steps=200)
            row["implied_hazard_at_vol"] = implied_hazard(cb, cv["price"], S, iv, r, 0.0, steps=200)
            row["implied_hazard_straight"] = implied_hazard(cb, cv["price"], S, iv, r, 0.0, steps=200, convert=False)
            hs = row["implied_hazard_straight"]
            row["implied_pd_5y_straight"] = float(1 - np.exp(-hs * min(T, 5.0))) if np.isfinite(hs) else 1.0
            row["moneyness"] = row["parity"] / 100.0
        rows.append(row)
    return {"converts": rows}


# --------------------------------------------------------------------------------------------- options
def option_section(case: Case, bond_rows: list) -> dict:
    o = case.equity.get("options") or {}
    bounds = o.get("deep_put_pd_bounds") or []
    if not bounds:
        return {"bounds": []}
    best = min(bounds, key=lambda b: b["pd_upper_bound"] / max(b["T"], 1e-9))
    out = {"bounds": bounds, "tightest": best}
    comp = []
    for b in bond_rows:
        if not b.get("pd_identified", True):
            continue
        lam = b["hazard"]
        pd_bond = float(1 - np.exp(-lam * best["T"])) if np.isfinite(lam) else 1.0
        comp.append({"claim_id": b["claim_id"], "pd_bond_at_put_horizon": pd_bond, "put_upper_bound": best["pd_upper_bound"],
                     "bond_exceeds_put_bound": pd_bond > best["pd_upper_bound"]})
    out["bond_vs_put"] = comp
    return out


# --------------------------------------------------------------------------------------------- CDS
def cds_section(case: Case, recoveries: dict) -> dict | None:
    if not case.data.get("cds"):
        return None
    from .daycount import to_date
    from .bonds import FixedBond
    from .case import treasury_curve
    q = case.data["cds"][0]
    marks = case.data.get("dated_bond_marks")
    zc = treasury_curve(q["date"])
    trade = to_date(q["date"])
    mat = to_date(q["maturity"])
    lam = cdsmod.flat_hazard_from_spread(q["par_spread_bp"] / 1e4, trade, mat, zc, q["recovery"])
    upf = cdsmod.upfront_from_quoted_spread(q["conventional_spread_bp"] / 1e4, q["coupon_bp"] / 1e4, trade, mat, zc,
                                            q["recovery"])
    out = {"quote": q, "flat_hazard_at_par": lam, "upfront_check_pct": upf * 100,
           "upfront_reported_pct": q["upfront_pct"], "basis": []}
    if marks:
        settle = to_date("2026-10-26")
        for cid, px in marks["marks"].items():
            c = case.structure.claim(cid)
            b = FixedBond(c.name, c.coupon, to_date(case.data["claims"][[x["id"] for x in case.data["claims"]].index(cid)]["maturity"]))
            z = b.zspread(px, zc, settle)
            fit = implied_flat_hazard(b, px, settle, zc, q["recovery"])
            l_b = float(fit.curve.hazards[0])
            cc = cdsmod.CDSContract(trade, b.maturity, 0.01, q["recovery"])
            pecs = cdsmod.par_spread(cc, zc, HazardCurve.flat(l_b)) if fit.status == "ok" else None
            cds_at_T = cdsmod.par_spread(cc, zc, HazardCurve.flat(lam))
            out["basis"].append({"claim_id": cid, "bond_price": px, "z_bp": z * 1e4,
                                 "pecs_bp": None if pecs is None else pecs * 1e4, "cds_at_bond_maturity_bp": cds_at_T * 1e4,
                                 "basis_bp": None if pecs is None else (cds_at_T - pecs) * 1e4,
                                 "note": "CDS curve assumed flat at the 5Y par spread (only the 5Y point is documented)"})
    return out


def observable_asset_section(case: Case, horizons=(1.0, 5.0, 10.0)) -> dict | None:
    """For a company whose assets are marked to market (Strategy: bitcoin + cash), skip the calibration step:
    price every claim at the observed asset value and volatility, then compare with market prices.

    Perpetual preferreds are compared through annualised loss rates: market = current yield minus the
    30-year Treasury par yield; model = the claim's zero-coupon spread at the horizon. Both are crude, and
    neither captures dividend suspension (STRD is non-cumulative) or the issuer's discretion over STRC's rate.
    """
    a = case.data.get("assets")
    if not a:
        return None
    from .curves import par_yield
    from .structural import black_cox as bcm
    cs = case.structure
    V0, sig = a["total_observable_assets"], a["asset_vol"]
    # cash paid out of the asset pool every year: preferred dividends + coupons, as a payout rate q
    payout = 0.0
    for c in case.data["claims"]:
        if c["kind"] == "preferred":
            payout += c["instrument"].get("dividend_rate", 0.0) * c["amount"]
        elif c.get("coupon"):
            payout += c["coupon"] * c["principal"]
    q = payout / V0
    out = {"V0": V0, "sigma_V": sig, "btc_price": a["btc_price"], "btc_held": a["btc_held"], "annual_payout": payout,
           "payout_rate": q, "by_horizon": {}}
    # how far equity-based calibration lands from the observed asset value and volatility
    E, sE = case.market_cap, _equity_vol(case)[0]
    r1 = float(case.zc.zero(1.0))
    chk = {}
    for name, fn in (("merton_kmv_1y", lambda: merton.calibrate(E, sE, cs.default_point(1.0), 1.0, r1)),
                     ("merton_all_debt_5y", lambda: merton.calibrate(E, sE, cs.total_debt(), 5.0, float(case.zc.zero(5.0)))),
                     ("merton_all_claims_5y", lambda: merton.calibrate(E, sE, cs.total_claims(), 5.0, float(case.zc.zero(5.0))))):
        try:
            m = fn()
            chk[name] = {"V": m.V, "sigma_V": m.sigma_V, "V_error": m.V / V0 - 1, "sigma_ratio": m.sigma_V / sig}
        except RuntimeError as exc:
            chk[name] = {"error": str(exc)}
    out["calibration_check"] = chk
    out["equity_vol_used"] = sE
    y30 = par_yield(case.zc, 30.0)
    for T in horizons:
        rT = float(case.zc.zero(T))
        sp = price_structure(cs, V0, sig, T, rT, q=q, model="merton", default_point=cs.total_debt())
        claims = {}
        for c in cs.claims:
            if c.kind == "common":
                continue
            v = sp.values[c.id]
            claims[c.id] = {"value": v, "pct_of_claim": v / c.amount if c.amount else None,
                            "spread": sp.yields[c.id] - rT if np.isfinite(sp.yields[c.id]) else None,
                            "pd_impaired": None}
        # probability each tranche is impaired at T: V_T below the claims ranked ahead of it + itself
        from .structural.lognormal import partial_moments
        cum = 0.0
        for c in sorted([c for c in cs.claims if c.kind != "common"], key=lambda c: (c.rank, c.id)):
            cum += c.amount
            p, _ = partial_moments(V0, rT - q, sig, T, 0.0, cum / (1 - cs.admin_cost))
            claims[c.id]["pd_impaired"] = float(p)
            # first-passage version: probability assets ever fall to this tranche's detachment point by T
            claims[c.id]["pd_touch"] = float(bcm.first_passage_prob(V0, rT - q, sig, T, cum / (1 - cs.admin_cost)))
        out["by_horizon"][str(T)] = {"equity_model": sp.equity, "equity_market": case.market_cap,
                                     "pd_debt_default": sp.pd_q, "claims": claims,
                                     "cond_recovery": {k: v for k, v in sp.cond_recovery.items() if np.isfinite(v)}}
    prefs = []
    for p in case.preferreds:
        c = cs.claim(p["claim_id"])
        cy = p["dividend_rate"] * p["stated"] / p["price"]
        model10 = out["by_horizon"]["10.0"]["claims"][c.id]
        prefs.append({"ticker": p["ticker"], "price": p["price"], "dividend_rate": p["dividend_rate"],
                      "cumulative": p["cumulative"], "current_yield": cy, "ust_30y": y30, "market_spread": cy - y30,
                      "model_spread_10y": model10["spread"], "pd_impaired_10y": model10["pd_impaired"],
                      "pd_touch_10y": model10["pd_touch"], "attach_musd": None})
    cum = 0.0
    for c in sorted([c for c in cs.claims if c.kind != "common"], key=lambda c: (c.rank, c.id)):
        for p in prefs:
            if p["ticker"].lower() == c.id:
                p["attach_musd"] = cum
                p["detach_musd"] = cum + c.amount
                p["asset_drop_to_impair"] = 1 - cum / (1 - cs.admin_cost) / V0
        cum += c.amount
    out["preferreds"] = prefs
    return out


def historical_section(case: Case) -> dict | None:
    """Wolfspeed 2025: reproduce the court-filed liquidation analysis, then compare plan recoveries with
    absolute priority at the plan's valuation range."""
    h = case.data.get("historical")
    if not h:
        return None
    cs = case.structure
    out = {"liquidation": {}, "plan_vs_apr": {}}
    for k in ("low", "high"):
        V = h["liquidation_proceeds"][k]
        r = distribute(cs, np.array([V]), admin_cost=0.0)
        out["liquidation"][k] = {"proceeds": V, "recovery": {c.id: float(r.recovery[c.id][0]) for c in cs.claims if c.kind != "common"}}
    out["liquidation_reported"] = h["liquidation_recoveries_reported"]
    for k in ("low", "mid", "high"):
        V = h["plan_ev"][k]
        r = distribute(cs, np.array([V]), admin_cost=0.0)
        out["plan_vs_apr"][k] = {"value": V, "apr_recovery": {c.id: float(r.recovery[c.id][0]) for c in cs.claims if c.kind != "common"}}
    pc = h["plan_consideration"]
    eqv = h["plan_equity_value"]
    claims = {c.id: c.amount for c in cs.claims}
    conv_claim = claims["cv_2026"] + claims["cv_2028"] + claims["cv_2029"]
    derived = {}
    for k in ("low", "mid", "high"):
        e = eqv[k]
        derived[k] = {
            "ssn": (pc["ssn"]["new_senior_secured_notes"] + pc["ssn"]["cash_payment"]) / claims["ssn"],
            "convertible": (pc["convertible"]["new_equity_pct"] / 100 * e + pc["convertible"]["takeback_2l_notes"]) / conv_claim,
            "renesas": (pc["renesas"]["new_equity_pct"] / 100 * e + pc["renesas"]["renesas_2l_converts"]) / claims["renesas"],
            "old_equity_value": pc["old_equity"]["new_equity_pct"] / 100 * e}
    out["plan_recoveries_derived"] = derived
    out["plan_recoveries_reported"] = h["plan_recoveries_reported"]
    pm = h["post_emergence_market"]
    out["post_emergence"] = {"market_cap": pm["shares_m"] * pm["price"], "date": pm["date"],
                             "plan_equity_value": eqv}
    return out


def historical_hazard(rating: str, horizon: str = "3y") -> dict | None:
    """Average annual default intensity for a rating from S&P's Form NRSRO cohort table (withdrawn issuers counted
    as non-defaults, as published). The 3-year cohort is the default: one year is too noisy, ten years too old."""
    from pathlib import Path
    import json as _json
    p = Path(__file__).resolve().parents[2] / "data" / "reference" / "sp_corporate_transitions_2025.json"
    if not p.exists():
        return None
    ref = _json.loads(p.read_text())
    row = ref["tables"][horizon]["rows"].get(rating)
    if not row or not row["n"]:
        return None
    years = {"1y": 1.0, "3y": 3.0, "10y": 10.0}[horizon]
    d = row["pct"].get("Default", 0.0) / 100
    return {"rating": rating, "horizon": horizon, "cohort": ref["tables"][horizon]["period"], "n": row["n"],
            "default_rate": d, "hazard": float(-np.log(1 - d) / years)}


def decomposition_section(case: Case, out: dict) -> dict | None:
    """Split each identified bond's spread into expected loss, default-risk premium and a non-default part.

    Spreads are compared in CDS-equivalent terms (the bond's par-equivalent CDS spread, PECS). Expected loss uses
    S&P's historical default intensity for the notes' rating and the waterfall's central recovery. The non-default
    part (PECS minus CDS, i.e. minus the basis) is identified only on the date with a documented CDS quote.
    """
    ratings = case.data.get("ratings") or {}
    rt = (ratings.get("notes") or {}).get("sp")
    hist = historical_hazard(rt) if rt else None
    if not hist or not out["bonds"]["bonds"]:
        return None
    from .spreads import decompose_spread
    res = {"physical": hist, "rating_source": ratings["notes"]["source"], "current": [], "dated": []}
    for b in out["bonds"]["bonds"]:
        if not b.get("pd_identified") or b.get("pecs") is None:
            continue
        d = decompose_spread(b["pecs"], None, hist["hazard"], b["recovery_used"])
        d.update({"claim_id": b["claim_id"], "name": b["name"], "date": b["trade_date"], "z_bp": b["zspread"] * 1e4,
                  "pecs_bp": b["pecs"] * 1e4})
        res["current"].append(d)
    cdsa = out.get("cds")
    if cdsa and cdsa.get("basis"):
        rec = out["recovery_used"]
        for x in cdsa["basis"]:
            if x.get("pecs_bp") is None:
                continue
            d = decompose_spread(x["pecs_bp"] / 1e4, x["cds_at_bond_maturity_bp"] / 1e4, hist["hazard"], rec.get(x["claim_id"], 0.4))
            d.update({"claim_id": x["claim_id"], "date": cdsa["quote"]["date"], "pecs_bp": x["pecs_bp"],
                      "cds_bp": x["cds_at_bond_maturity_bp"]})
            res["dated"].append(d)
    return res


def example_trade_section(case: Case, out: dict, face_musd: float = 10.0) -> dict | None:
    """A hedged position and its scenario P&L: long a note, long deep puts sized so that the position neither gains
    nor loses on an immediate default at the central waterfall recovery (stock assumed worthless at default).

    Scenarios: default now at the low/central/high recoveries; equity -50% / +50% without default (assets moved so the
    equity-calibrated Black-Cox model reproduces the new equity value; the note repriced on the hybrid structural
    curve, puts with Black-Scholes at the listed implied volatility); one year of carry with nothing else changing.
    Model-to-model changes are used throughout so the market-vs-model level gap does not leak into the P&L.
    """
    from .options import bs_implied_vol, bs_price
    from .scenarios import Position, hedge_ratio_jtd_neutral, jump_to_default
    bonds = {b["claim_id"]: b for b in out["bonds"]["bonds"] if b.get("pd_identified")}
    t = (out.get("options") or {}).get("tightest")
    bc = out["structural"].get("black_cox_waterfall", {})
    if not bonds or not t or "V" not in bc:
        return None
    cid = "notes_2031a" if "notes_2031a" in bonds else next(iter(bonds))
    b = bonds[cid]
    q = next(x for x in case.bonds if x.claim_id == cid)
    sc = out["recovery_curves"].get("scenarios") or {}
    R = {k: v["recovery"][cid] for k, v in sc.items()} or {"central": b["recovery_used"]}
    Rc = R.get("central", b["recovery_used"])
    S = case.equity["price"]
    r_put = float(case.zc.zero(t["T"]))
    # deep out-of-the-money puts trade far above at-the-money volatility (skew): reprice at the put's own implied
    # volatility from its mid quote, held fixed for the strike (sticky strike); fall back to ATM vol if it fails
    mid = 0.5 * (t["bid"] + t["ask"])
    iv = bs_implied_vol(mid, S, t["strike"], t["T"], r_put, 0.0, "put")
    iv_source = "implied volatility of the put's mid quote (sticky strike)"
    if not np.isfinite(iv):
        term = case.equity["options"]["atm_iv_term"]
        iv = float(np.interp(t["T"], [x["T"] for x in term], [x["iv"] for x in term]))
        iv_source = "at-the-money implied volatility (the put's own was not recoverable)"
    put_now = bs_price(S, t["strike"], t["T"], r_put, 0.0, iv, "put")
    n_puts = hedge_ratio_jtd_neutral(face_musd, q.clean, Rc, t["strike"], t["ask"])
    premium = n_puts * 100 * t["ask"] / 1e6
    pos = [Position(cid, "bond", face_musd, q.clean), Position("put", "put", n_puts, t["ask"], strike=t["strike"])]
    rows = []
    for k in ("low", "central", "high"):
        if k in R:
            j = jump_to_default(pos, {cid: R[k]})
            rows.append({"scenario": f"Default now, recovery {R[k]:.0%} ({k})", "bond": j["rows"][0]["pnl"],
                         "puts": j["rows"][1]["pnl"], "total": j["total"]})
    # equity moves through the structural model
    from scipy.optimize import brentq
    cs, T = case.structure, out["structural"]["horizon"]
    rT = float(case.zc.zero(T))
    H0, sig = bc["H0"], bc["sigma_V"]

    def model_note(V):
        hc = structural_hazard_curve(V, sig, H0, 0.0, rT)
        Rb = float(np.clip(barrier_recoveries(cs, H0).get(cid, 0.0), 0.0, 0.95))
        return risky_dirty_price(q.bond, case.settle, case.zc, hc, Rb) - q.bond.accrued(case.settle)

    def model_equity(V):
        return price_structure(cs, V, sig, T, rT, model="black_cox", H0=H0, default_point=cs.total_debt(),
                               with_sigma_E=False).equity

    base_note = model_note(bc["V"])
    for mult in (0.5, 1.5):
        target = case.market_cap * mult
        try:
            V1 = brentq(lambda V: model_equity(V) - target, H0 * 1.0001, bc["V"] * 4)
        except ValueError:
            continue
        note = model_note(V1)
        put1 = bs_price(S * mult, t["strike"], t["T"], r_put, 0.0, iv, "put")
        bond_pnl = face_musd * (note - base_note) / 100
        put_pnl = n_puts * 100 * (put1 - put_now) / 1e6
        rows.append({"scenario": f"Equity {mult - 1:+.0%}, no default", "bond": bond_pnl, "puts": put_pnl, "total": bond_pnl + put_pnl,
                     "note_price_change": note - base_note})
    put_1y = bs_price(S, t["strike"], max(t["T"] - 1, 0.01), float(case.zc.zero(max(t["T"] - 1, 0.01))), 0.0, iv, "put")
    carry_bond = face_musd * q.bond.coupon   # coupon income only; pull to par and roll-down left out
    rows.append({"scenario": "One year, nothing changes (coupon only)", "bond": carry_bond,
                 "puts": n_puts * 100 * (put_1y - put_now) / 1e6, "total": carry_bond + n_puts * 100 * (put_1y - put_now) / 1e6})
    return {"note": cid, "note_name": b["name"], "face_musd": face_musd, "note_price": q.clean, "recovery_central": Rc,
            "put": {"expiry": t["expiry"], "strike": t["strike"], "bid": t["bid"], "ask": t["ask"], "iv": iv,
                    "iv_source": iv_source, "model": put_now},
            "contracts": n_puts, "premium_musd": premium, "rows": rows,
            "caveats": ["puts bought at the ask, Black-Scholes (European) for the no-default repricing",
                        "stock assumed worthless at default (any recovery to equity weakens the hedge)",
                        "bond financed at no cost (repo and borrow not modelled)", "last-trade bond price, not an executable quote"]}


def run_case(case: Case) -> dict:
    out = {"id": case.id, "company": case.data["company"], "market_date": case.data["market_date"],
           "settlement_date": case.data["settlement_date"]}
    if case.data.get("historical"):
        out["structure"] = {"total_claims": case.structure.total_claims()}
        out["recovery_curves"] = recovery_curves(case)
        out["historical"] = historical_section(case)
        return out
    out["structure"] = structure_summary(case)
    rc = recovery_curves(case)
    out["recovery_curves"] = rc
    rec = central_recovery(case, rc)
    out["observable_assets"] = observable_asset_section(case)
    if out["observable_assets"]:
        # assets are observed: use the model's own expected recovery given a debt default (5-year horizon)
        rec = dict(rec)
        rec.update(out["observable_assets"]["by_horizon"]["5.0"]["cond_recovery"])
        out["recovery_source"] = "observable-asset model, expected recovery given default within 5 years"
    else:
        out["recovery_source"] = "waterfall at the central default-value scenario"
    out["recovery_used"] = rec
    out["bonds"] = bond_section(case, rec)
    out["structural"] = structural_section(case, rec)
    hc = None
    identified = [b for b in out["bonds"]["bonds"] if b.get("pd_identified")]
    if out["bonds"]["curve"] and len(identified) == len(out["bonds"]["bonds"]):
        hc = HazardCurve(np.array(out["bonds"]["curve"]["knots"]), np.array(out["bonds"]["curve"]["hazards"]))
        out["convert_hazard_source"] = "issuer hazard curve fitted to the straight bonds"
    elif identified:
        hc = HazardCurve.flat(identified[0]["hazard"])
        out["convert_hazard_source"] = "flat hazard from the issuer's straight bond"
    elif "hazards" in out["structural"].get("black_cox_waterfall", {}):
        b = out["structural"]["black_cox_waterfall"]
        hc = HazardCurve(np.array(b["hazard_knots"]), np.array(b["hazards"]))
        out["convert_hazard_source"] = "structural (Black-Cox) first-passage hazard"
    out["converts"] = convert_section(case, rec, hc) if case.converts else {"converts": []}
    out["options"] = option_section(case, out["bonds"]["bonds"])
    out["cds"] = cds_section(case, rec)
    out["decomposition"] = decomposition_section(case, out)
    out["example_trade"] = example_trade_section(case, out)
    return out
