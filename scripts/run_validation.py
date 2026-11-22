"""Validation experiments: benchmarks against published answers, sensitivity, restructuring scenarios,
historical default experience and cross-model comparison. Writes reports/validation/results.json.

Usage: python scripts/run_validation.py
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from creditlab import cds
from creditlab.capital_structure import CapitalStructure, Claim, Entity, substantively_consolidated, with_overrides
from creditlab.case import load_case, treasury_curve
from creditlab.hazard import HazardCurve, implied_flat_hazard
from creditlab.structural import black_cox, creditgrades, merton
from creditlab.structural.lognormal import first_passage_prob
from creditlab.structural.waterfall_pricing import calibrate_structure, price_structure
from creditlab.survival import marginal_from_cumulative
from creditlab.transition import TransitionMatrix
from creditlab.waterfall import distribute

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "validation"
OUT.mkdir(parents=True, exist_ok=True)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        x = float(o)
        return x if np.isfinite(x) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ---------------------------------------------------------------------------------------------- A. benchmarks
def benchmarks() -> list[dict]:
    rows = []
    m = merton.calibrate(3.0, 0.80, 10.0, 1.0, 0.05)
    for k, mine, ref in (("V0", m.V, 12.40), ("sigma_V", m.sigma_V, 0.2123), ("PD", m.pd_q, 0.127), ("debt value", m.B, 9.40)):
        rows.append({"test": "Merton two-equation calibration", "quantity": k, "model": mine, "reference": ref,
                     "abs_error": abs(mine - ref), "reference_source": "Hull, Options, Futures and Other Derivatives (Merton example: E=3, sigma_E=80%, D=10, r=5%, T=1)"})
    zc = treasury_curve("2026-10-22")
    mat = cds.standard_maturity("2026-10-22", 5)
    u = cds.upfront_from_quoted_spread(0.1167011, 0.05, "2026-10-22", mat, zc, 0.4)
    rows.append({"test": "CDS quoted spread to upfront (ISDA-style)", "quantity": "upfront, points", "model": u * 100,
                 "reference": 19.834, "abs_error": abs(u * 100 - 19.834),
                 "reference_source": "Markit 5Y official close 22-Oct-2026, CoreWeave SNRFOR XR, via LSEG (conventional spread 1,167.011bp)"})
    q = cds.quoted_spread_from_upfront(0.19834, 0.05, "2026-10-22", mat, zc, 0.4)
    rows.append({"test": "CDS upfront to quoted spread", "quantity": "spread, bp", "model": q * 1e4, "reference": 1167.011,
                 "abs_error": abs(q * 1e4 - 1167.011), "reference_source": "same Markit close"})
    w = load_case("wolfspeed_2025")
    h = w.data["historical"]
    for k in ("low", "high"):
        r = distribute(w.structure, np.array([h["liquidation_proceeds"][k]]), admin_cost=0.0)
        rep = h["liquidation_recoveries_reported"]
        idx = 0 if k == "low" else 1
        for cid, ref in (("ssn", rep["ssn"][idx]), ("admin", rep["admin"][idx]), ("cv_2029", rep["unsecured"][idx])):
            mine = float(r.recovery[cid][0])
            rows.append({"test": f"Wolfspeed Chapter 7 liquidation analysis ({k} case)", "quantity": f"recovery {cid}",
                         "model": mine, "reference": ref, "abs_error": abs(round(mine, 3) - ref),
                         "reference_source": "Wolfspeed disclosure statement Exhibit D (8-K 0001193125-25-153145)"})
    cs = CapitalStructure("t", [Entity("t", "T", None, 1.0)], [], [
        Claim("s", "S", "senior_unsecured", "t", 40.0), Claim("j", "J", "subordinated", "t", 30.0, rank=20),
        Claim("e", "E", "common", "t")])
    sp = price_structure(cs, 100.0, 0.45, 3.0, 0.04)
    tr = merton.tranche_values(100.0, 0.45, [40.0, 30.0], 3.0, 0.04)
    rows.append({"test": "Waterfall pricer vs multi-tranche Merton (closed form)", "quantity": "junior tranche value",
                 "model": sp.values["j"], "reference": tr["tranches"][1]["value"],
                 "abs_error": abs(sp.values["j"] - tr["tranches"][1]["value"]), "reference_source": "closed form (call spreads)"})
    S, sig, D, r = 20.0, 0.7, 30.0, 0.04
    cf, nu = creditgrades.cds_spread(5.0, S, sig, D, r), creditgrades.cds_spread_numeric(5.0, S, sig, D, r)
    rows.append({"test": "CreditGrades closed-form CDS spread", "quantity": "5y spread, bp", "model": cf * 1e4,
                 "reference": nu * 1e4, "abs_error": abs(cf - nu) * 1e4, "reference_source": "numerical integration of the survival curve"})
    rng = np.random.default_rng(3)
    V0, mu, s, T, H = 100.0, 0.03, 0.35, 2.0, 70.0
    n, steps = 40000, 500
    dt = T / steps
    x = np.full(n, np.log(V0))
    hit = np.zeros(n, bool)
    lh = np.log(H)
    for _ in range(steps):
        xn = x + (mu - 0.5 * s * s) * dt + s * np.sqrt(dt) * rng.standard_normal(n)
        a, b = x - lh, xn - lh
        pc = np.exp(-2 * np.maximum(a, 0) * np.maximum(b, 0) / (s * s * dt))
        hit |= (b <= 0) | (rng.random(n) < pc)
        x = xn
    p_mc = hit.mean()
    rows.append({"test": "Black-Cox first-passage probability", "quantity": "P(hit by 2y)", "model": float(first_passage_prob(V0, mu, s, T, H)),
                 "reference": float(p_mc), "abs_error": abs(float(first_passage_prob(V0, mu, s, T, H)) - p_mc),
                 "reference_source": f"Monte Carlo, {n:,} paths x {steps} steps with Brownian-bridge correction (s.e. {np.sqrt(p_mc*(1-p_mc)/n):.4f})"})
    return rows


# ---------------------------------------------------------------------------------------------- B. observable assets
def observable_assets() -> dict:
    a = json.loads((ROOT / "reports" / "cases" / "strategy" / "analysis.json").read_text())
    return a["observable_assets"]


# ---------------------------------------------------------------------------------------------- C. sensitivities
def sensitivities() -> dict:
    case = load_case("coreweave")
    cs = case.structure
    out = {}
    q = [b for b in case.bonds if b.claim_id == "notes_2031b"][0]
    pdr = []
    for R in np.arange(0.0, 0.75, 0.05):
        fit = implied_flat_hazard(q.bond, q.clean, case.settle, case.zc, float(R))
        lam = float(fit.curve.hazards[0])
        pdr.append({"recovery": float(R), "hazard": lam, "pd_5y": float(1 - np.exp(-5 * lam)) if np.isfinite(lam) else 1.0,
                    "status": fit.status})
    out["bond_pd_vs_recovery"] = {"bond": q.bond.name, "price": q.clean, "rows": pdr}
    E = case.market_cap
    o = case.equity["options"]
    vols = {"ATM IV 1y": o["iv_1y"], "ATM IV 2y": o["iv_2y"], "realised 3m": case.equity["vol_3m"],
            "realised 1y": case.equity["vol_1y"]}
    T = 4.56
    rT = float(case.zc.zero(T))
    DP1, D = cs.default_point(1.0), cs.total_debt()
    grid = []
    for vname, v in vols.items():
        for bf in (0.5, 0.75, 1.0, 1.25):
            try:
                bc = calibrate_structure(cs, E, v, T, rT, model="black_cox", barrier_ratio=bf * DP1 / D, default_point=D)
                pd5 = float(first_passage_prob(bc.V0, rT, bc.sigma, 5.0, bc.H0))
                grid.append({"vol_source": vname, "equity_vol": v, "barrier_fraction": bf, "V": bc.V0, "sigma_V": bc.sigma,
                             "pd_5y_barrier": pd5, "pd_T": bc.pd_q})
            except RuntimeError:
                grid.append({"vol_source": vname, "equity_vol": v, "barrier_fraction": bf, "error": "no fit"})
    out["structural_grid"] = grid
    central = case.data["default_value_scenarios"]["central"]
    base = float(distribute(cs, np.array([central])).recovery["notes_2031b"][0])
    torn = []

    def rec(cs2, V=central, **kw):
        return float(distribute(cs2, np.array([V]), **kw).recovery["notes_2031b"][0])

    sc = case.data["default_value_scenarios"]
    torn.append({"driver": "Value at default (35% / 75% of PP&E)", "low": rec(cs, sc["low"]), "high": rec(cs, sc["high"])})
    torn.append({"driver": "Administrative costs (10% / 2%)", "low": rec(cs, admin_cost=0.10), "high": rec(cs, admin_cost=0.02)})
    torn.append({"driver": "Lease rejections (100% / 0% of leases)",
                 "low": rec(with_overrides(cs, {"lease_rej": 2.0})), "high": rec(with_overrides(cs, {"lease_rej": 0.0}))})
    torn.append({"driver": "Trade claims (+50% / -50%)", "low": rec(with_overrides(cs, {"trade": 1.5})),
                 "high": rec(with_overrides(cs, {"trade": 0.5}))})
    cs_spv_hi = copy.deepcopy(cs)
    for e in cs_spv_hi.entities:
        if e.id in ("spv_r", "spv_nr"):
            e.value_share *= 1.25
    cs_spv_hi.entity("opco").value_share = 1 - sum(e.value_share for e in cs_spv_hi.entities if e.id != "opco")
    cs_spv_lo = copy.deepcopy(cs)
    for e in cs_spv_lo.entities:
        if e.id in ("spv_r", "spv_nr"):
            e.value_share *= 0.75
    cs_spv_lo.entity("opco").value_share = 1 - sum(e.value_share for e in cs_spv_lo.entities if e.id != "opco")
    torn.append({"driver": "Value held in SPVs (+25% / -25%)", "low": rec(cs_spv_hi), "high": rec(cs_spv_lo)})
    torn.append({"driver": "Revolver drawn (100% / 48% = current)", "low": rec(with_overrides(cs, {"rcf": 1 / 0.85})),
                 "high": rec(with_overrides(cs, {"rcf": 1200 / 2125}))})
    torn.append({"driver": "Substantive consolidation (all entities pooled)", "low": rec(substantively_consolidated(cs)),
                 "high": rec(substantively_consolidated(cs))})
    out["recovery_tornado"] = {"claim": "notes_2031b", "base": base, "rows": torn,
                               "base_value": central}
    return out


# ---------------------------------------------------------------------------------------------- D. restructuring
def restructuring_scenarios() -> dict:
    case = load_case("coreweave")
    cs = case.structure
    V = case.data["default_value_scenarios"]["central"]
    groups = {"SPV term loans (recourse)": ["ddtl_1_0", "ddtl_2_0", "ddtl_2_1", "ddtl_3_0", "ddtl_5_0", "ddtl_5_5"],
              "DDTL 4.0 (non-recourse)": ["ddtl_4_0"], "OEM financing": ["oem_r", "oem_nr"], "Revolver": ["rcf"],
              "Senior notes": ["notes_2030", "notes_2031a", "notes_2031b", "notes_2032", "notes_eur_2032"],
              "Convertibles": ["cv_2031", "cv_2032", "cv_2033"], "Trade and leases": ["trade", "lease_rej"],
              "Magnetar": ["magnetar"]}

    def table(cs2, **kw):
        r = distribute(cs2, np.array([V]), **kw)
        g = {}
        for name, ids in groups.items():
            ids = [i for i in ids if i in r.payouts]
            amt = sum(cs2.claim(i).amount for i in ids)
            g[name] = float(sum(r.payouts[i][0] for i in ids) / amt) if amt else None
        g["Equity value ($m)"] = float(r.residual[0])
        return g

    sc = {"Absolute priority (base)": table(cs),
          "5% gift to old equity": table(cs, gift_to_equity=0.05),
          "Substantive consolidation": table(substantively_consolidated(cs)),
          "All leases rejected": table(with_overrides(cs, {"lease_rej": 2.0})),
          "Recourse DDTLs lose parent guarantee": table(with_overrides(cs, non_recourse=tuple(groups["SPV term loans (recourse)"]))),
          "Secured claims allowed +10% (make-whole style)": table(with_overrides(cs, {k: 1.1 for k in
                                                          ["delayed_draw_term_loan", "equipment_financing", "revolver"]}))}
    return {"value": V, "scenarios": sc}


# ---------------------------------------------------------------------------------------------- E. historical
RATINGS_USED = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-", "B+", "B", "B-",
                "CCC+", "CCC", "CCC-", "CC"]


def historical() -> dict:
    ref = json.loads((ROOT / "data" / "reference" / "sp_corporate_transitions_2025.json").read_text())
    t1 = ref["tables"]["1y"]["rows"]
    labels = RATINGS_USED + ["D"]
    P = np.zeros((len(labels), len(labels)))
    for i, r in enumerate(RATINGS_USED):
        row = t1[r]["pct"]
        for j, c in enumerate(RATINGS_USED):
            P[i, j] = row.get(c, 0.0) + (row.get("C", 0.0) if c == "CC" else 0.0)
        P[i, -1] = row.get("Default", 0.0)
        s = P[i].sum()
        P[i] = P[i] / s if s > 0 else np.eye(len(labels))[i]
    P[-1, -1] = 1.0
    tm = TransitionMatrix(P, labels, 1.0, ref["source"] + " (Table 7, paid-off and withdrawn removed)")
    cum = tm.cumulative_pd(10)
    out = {"source": ref["source"], "default_rates": [], "markov_vs_cohort": []}
    for k in ("1y", "3y", "10y"):
        rows = ref["tables"][k]["rows"]
        for r in RATINGS_USED:
            row = rows.get(r)
            if not row or not row["n"]:
                continue
            d = row["pct"].get("Default", 0.0)
            gone = row["pct"].get("Paid Off", 0.0) + row["pct"].get("Withdrawn (Other)", 0.0)
            out["default_rates"].append({"horizon": k, "rating": r, "n": row["n"], "default_pct": d,
                                         "default_pct_ex_withdrawn": d / (1 - gone / 100) if gone < 100 else None})
    for r in ("BB", "B+", "B", "B-", "CCC+", "CCC"):
        i = labels.index(r)
        for k, yrs in (("3y", 3), ("10y", 10)):
            row = ref["tables"][k]["rows"][r]
            out["markov_vs_cohort"].append({"rating": r, "horizon": k, "markov_from_1y_matrix_pct": float(cum[i, yrs - 1] * 100),
                                            "cohort_reported_pct": row["pct"].get("Default", 0.0), "cohort_n": row["n"]})
    out["generator_check"] = {"max_abs_diff_expm_vs_P": float(np.max(np.abs(tm.at(1.0) - P)))}
    # market-implied vs historical for the case issuers
    mk = []
    for cid, rating in (("coreweave", "B"), ("coreweave", "B+")):
        a = json.loads((ROOT / "reports" / "cases" / cid / "analysis.json").read_text())
        b = [x for x in a["bonds"]["bonds"] if x.get("pd_identified")]
        pd1 = float(np.mean([x["pd_1y"] for x in b]))
        pd3 = float(np.mean([x["pd_3y"] for x in b]))
        r3 = ref["tables"]["3y"]["rows"][rating]["pct"].get("Default", 0.0) / 100
        r1 = ref["tables"]["1y"]["rows"][rating]["pct"].get("Default", 0.0) / 100
        mk.append({"issuer": a["company"], "agency_rating_compared": rating, "market_pd_1y": pd1, "market_pd_3y": pd3,
                   "historical_1y": r1, "historical_3y": r3, "ratio_3y": pd3 / r3 if r3 else None,
                   "implied_rating_1y": tm.implied_rating(pd1, 1), "implied_rating_3y": tm.implied_rating(pd3, 3)})
    out["market_vs_historical"] = mk
    out["matrix"] = {"labels": labels, "P": P.tolist()}
    out["marginal_B_from_markov"] = marginal_from_cumulative(cum[labels.index("B")]).tolist()
    return out


# ---------------------------------------------------------------------------------------------- F. cross-model
def cross_model() -> list[dict]:
    rows = []
    for cid in ("coreweave", "nebius", "iren", "strategy", "wolfspeed"):
        a = json.loads((ROOT / "reports" / "cases" / cid / "analysis.json").read_text())
        s = a["structural"]
        b = [x for x in a["bonds"]["bonds"] if x.get("pd_identified")]
        cvs = [c for c in a["converts"]["converts"] if c.get("implied_pd_5y_straight") is not None and c["parity"] < 60]
        row = {"issuer": a["company"], "merton_kmv_pd_1y": s["merton_kmv"].get("pd_1y"),
               "black_cox_pd_1y": s["black_cox_waterfall"].get("pd_1y"), "black_cox_pd_5y": s["black_cox_waterfall"].get("pd_5y"),
               "creditgrades_pd_5y": s["creditgrades"].get("pd_5y"), "creditgrades_cds_5y_bp": s["creditgrades"].get("cds_5y", 0) * 1e4,
               "bond_pd_5y": float(np.mean([x["pd_5y"] for x in b])) if b else None,
               "convert_pd_5y_upper": float(np.median([c["implied_pd_5y_straight"] for c in cvs])) if cvs else None,
               "put_pd_bound": (a["options"].get("tightest") or {}).get("pd_upper_bound"),
               "put_bound_horizon": (a["options"].get("tightest") or {}).get("T")}
        if a.get("observable_assets"):
            row["observable_asset_pd_5y"] = a["observable_assets"]["by_horizon"]["5.0"]["pd_debt_default"]
        rows.append(row)
    return rows


def main():
    res = {"benchmarks": benchmarks(), "observable_assets": observable_assets(), "sensitivities": sensitivities(),
           "restructuring": restructuring_scenarios(), "historical": historical(), "cross_model": cross_model()}
    (OUT / "results.json").write_text(json.dumps(clean(res), indent=1), encoding="utf-8")
    print("benchmarks:")
    for r in res["benchmarks"]:
        print(f"  {r['test'][:52]:<52} {r['quantity'][:22]:<22} model {r['model']:.4f} ref {r['reference']:.4f} err {r['abs_error']:.4g}")
    print("markov vs cohort:", [(x["rating"], x["horizon"], round(x["markov_from_1y_matrix_pct"], 1), x["cohort_reported_pct"]) for x in res["historical"]["markov_vs_cohort"]])
    print("market vs hist:", res["historical"]["market_vs_historical"])
    print("tornado:", [(r["driver"][:30], round(r["low"], 3), round(r["high"], 3)) for r in res["sensitivities"]["recovery_tornado"]["rows"]])
    for k, v in res["restructuring"]["scenarios"].items():
        print("  ", k, {kk: (round(vv, 3) if isinstance(vv, float) and vv < 10 else (round(vv) if vv else vv)) for kk, vv in v.items()})
    for r in res["cross_model"]:
        print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
