"""Export the data the interactive demo needs (web/data/cases.json) and JS parity vectors (web/test/vectors.json).

The demo re-implements the waterfall and structural formulas in JavaScript (web/lab-engine.js). The parity
vectors are computed here with the Python engine; web/test/index.html recomputes them in the browser and
fails loudly on any mismatch.
Usage: python scripts/export_web.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from creditlab.capital_structure import CapitalStructure, Claim, CollateralPool, Entity
from creditlab.case import load_case
from creditlab.structural.black_cox import pd_term_structure
from creditlab.structural.lognormal import barrier_partial_moments, discounted_hit, first_passage_prob, partial_moments
from creditlab.structural.waterfall_pricing import price_structure
from creditlab.waterfall import breakpoints, distribute

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
GROUP = {"revolver": "Secured", "term_loan": "Secured", "delayed_draw_term_loan": "Secured", "secured_notes": "Secured",
         "second_lien_notes": "Secured", "equipment_financing": "Secured", "admin": "Priority", "senior_unsecured": "Senior unsecured",
         "convertible": "Senior unsecured", "trade": "Senior unsecured", "lease": "Senior unsecured", "other_debt": "Senior unsecured",
         "subordinated": "Subordinated", "preferred": "Preferred", "common": "Equity"}
REPO = "https://github.com/RasmusCodes/credit-capital-structure-lab"


def fnum(x, d=6):
    if x is None:
        return None
    x = float(x)
    return round(x, d) if np.isfinite(x) else None


def struct_json(cs: CapitalStructure) -> dict:
    return {"entities": [{"id": e.id, "name": e.name, "parent": e.parent, "share": e.value_share} for e in cs.entities],
            "pools": [{"id": p.id, "entity": p.entity, "share": p.share, "description": p.description} for p in cs.pools],
            "admin_cost": cs.admin_cost,
            "claims": [{"id": c.id, "name": c.name, "kind": c.kind, "group": GROUP.get(c.kind, "Other"), "entity": c.entity,
                        "amount": fnum(c.amount, 3), "rank": c.rank, "secured_by": [list(x) for x in c.secured_by],
                        "guarantors": list(c.guarantors), "guarantee_rank": c.guarantee_rank,
                        "turnover_to": list(c.turnover_to), "maturity": fnum(c.maturity, 4),
                        "coupon": fnum(c.coupon, 6)} for c in cs.claims]}


def case_json(cid: str) -> dict:
    case = load_case(cid)
    d = case.data
    a = json.loads((ROOT / "reports" / "cases" / cid / "analysis.json").read_text())
    out = {"id": cid, "name": d["company"], "ticker": d["ticker"], "kind": "case", "balance_sheet_date": d["balance_sheet_date"],
           "market_date": d["market_date"], "summary": d.get("summary", ""), "structure": struct_json(case.structure),
           "report": f"{REPO}/blob/main/reports/cases/{cid}/README.md", "case_file": f"{REPO}/blob/main/cases/{cid}/case.json"}
    mkt = {}
    for c in d["claims"]:
        m = c.get("market") or {}
        px = m.get("clean", m.get("price"))
        if px is not None:
            mkt[c["id"]] = {"price": px, "date": m.get("date"), "type": (c.get("instrument") or {}).get("type", "bond"),
                            "source": m.get("source", "")}
    out["market"] = mkt
    if "historical" in a:
        h = d["historical"]
        out["kind"] = "historical"
        out["historical"] = {"liquidation": h["liquidation_proceeds"], "liquidation_reported": h["liquidation_recoveries_reported"],
                             "plan_ev": h["plan_ev"], "plan_recoveries_reported": h["plan_recoveries_reported"],
                             "plan_recoveries_derived": a["historical"]["plan_recoveries_derived"]}
        out["scenarios"] = {"low": h["liquidation_proceeds"]["low"], "central": h["plan_ev"]["mid"], "high": h["liquidation_proceeds"]["high"]}
        out["scenario_labels"] = {"low": "Ch. 7 low", "central": "Plan EV mid", "high": "Ch. 7 high"}
        out["calibration"] = None
        return out
    e = d["equity"]
    s = a["structural"]
    bc = s["black_cox_waterfall"]
    out["equity"] = {"price": e["price"], "shares_m": e["shares_m"], "market_cap": case.market_cap, "vol": s["equity_vol"],
                     "vol_source": s["equity_vol_source"]}
    rT = float(case.zc.zero(s["horizon"]))
    if a.get("observable_assets"):
        oa = a["observable_assets"]
        out["calibration"] = {"model": "observed", "V0": oa["V0"], "sigma": oa["sigma_V"], "q": oa["payout_rate"],
                              "H0": None, "T": 5.0, "r": float(case.zc.zero(5.0)), "default_point": case.structure.total_debt(),
                              "note": "Assets observed directly (bitcoin + cash); no calibration."}
    else:
        out["calibration"] = {"model": "black_cox", "V0": bc["V"], "sigma": bc["sigma_V"], "q": 0.0, "H0": bc["H0"],
                              "T": s["horizon"], "r": rT, "default_point": s["total_debt"],
                              "note": "Black-Cox calibrated to equity value and volatility; barrier at the KMV default point."}
    sc = d.get("default_value_scenarios")
    if sc:
        out["scenarios"] = {k: sc[k] for k in ("low", "central", "high")}
        out["scenario_basis"] = sc.get("basis")
    pds = []
    ident = [b for b in a["bonds"]["bonds"] if b.get("pd_identified")]
    if ident:
        pds.append({"label": "Bonds (waterfall recovery)", "pd_5y": float(np.mean([b["pd_5y"] for b in ident])),
                    "pd_1y": float(np.mean([b["pd_1y"] for b in ident]))})
    cvs = [c for c in a["converts"]["converts"] if c.get("implied_pd_5y_straight") is not None and c["parity"] < 60]
    if cvs:
        pds.append({"label": "Converts, option ignored (upper bound)", "pd_5y": float(np.median([c["implied_pd_5y_straight"] for c in cvs]))})
    pds.append({"label": "Black-Cox (equity-calibrated)", "pd_5y": bc.get("pd_5y"), "pd_1y": bc.get("pd_1y")})
    pds.append({"label": "CreditGrades", "pd_5y": s["creditgrades"]["pd_5y"], "pd_1y": 1 - s["creditgrades"]["survival"]["1"]})
    if a.get("observable_assets"):
        pds.append({"label": "Observed assets", "pd_5y": a["observable_assets"]["by_horizon"]["5.0"]["pd_debt_default"]})
    t = (a["options"] or {}).get("tightest")
    if t:
        pds.append({"label": f"Deep puts, upper bound to {t['expiry']}", "pd_T": t["pd_upper_bound"], "T": t["T"]})
    out["pd_sources"] = pds
    out["bond_pd_by_recovery"] = {b["claim_id"]: b["pd_by_recovery"] for b in a["bonds"]["bonds"]}
    sig = json.loads((ROOT / "reports" / "scanner.json").read_text())
    out["signals"] = [x for x in sig if x["issuer"] == d["company"]][:6]
    return out


def examples() -> list[dict]:
    """Two hypothetical structures to explore the mechanics (clearly labelled as such)."""
    lbo = CapitalStructure("Hypothetical LBO", [Entity("co", "Company", None, 1.0)], [CollateralPool("all", "co", 0.85)], [
        Claim("rcf", "Revolver (1L)", "revolver", "co", 150.0, secured_by=(("all", 1),)),
        Claim("tlb", "Term loan B (1L)", "term_loan", "co", 900.0, secured_by=(("all", 1),)),
        Claim("sec2", "Second-lien notes", "second_lien_notes", "co", 300.0, secured_by=(("all", 2),)),
        Claim("snr", "Senior unsecured notes", "senior_unsecured", "co", 400.0),
        Claim("trade", "Trade payables", "trade", "co", 120.0),
        Claim("sub", "Senior subordinated notes", "subordinated", "co", 250.0, rank=10, turnover_to=("snr",)),
        Claim("pref", "Preferred stock", "preferred", "co", 100.0, rank=30),
        Claim("eq", "Common equity", "common", "co")], admin_cost=0.05)
    for c, m in zip(lbo.claims, (4.0, 6.0, 7.0, 8.0, 0.5, 9.0, None, None)):
        c.maturity = m
    hold = CapitalStructure("Hypothetical holdco / opco", [Entity("hold", "HoldCo", None, 0.0), Entity("op", "OpCo", "hold", 0.75),
                                                          Entity("spv", "Asset SPV", "op", 0.25)],
                            [CollateralPool("spv_all", "spv", 1.0), CollateralPool("abl", "op", 0.3)], [
        Claim("spv_loan", "SPV asset loan (non-recourse)", "term_loan", "spv", 400.0, secured_by=(("spv_all", 1),)),
        Claim("abl", "OpCo ABL", "revolver", "op", 250.0, secured_by=(("abl", 1),)),
        Claim("opco_notes", "OpCo senior notes", "senior_unsecured", "op", 600.0),
        Claim("trade", "OpCo trade payables", "trade", "op", 150.0),
        Claim("hold_notes", "HoldCo notes (no guarantees)", "senior_unsecured", "hold", 500.0),
        Claim("hold_conv", "HoldCo convertible", "convertible", "hold", 300.0),
        Claim("eq", "Common equity", "common", "hold")], admin_cost=0.05)
    for c, m in zip(hold.claims, (5.0, 3.0, 6.0, 0.5, 4.0, 5.0, None)):
        c.maturity = m
    out = []
    for cs, eid, V0, sig, blurb in ((lbo, "example_lbo", 2600.0, 0.30, "A leveraged buyout: first-lien revolver and term loan, second-lien notes, unsecured and subordinated notes (with turnover to the senior notes), preferred and common."),
                                    (hold, "example_holdco", 2600.0, 0.35, "Structural subordination: HoldCo creditors only reach OpCo's value after OpCo's own creditors; a non-recourse SPV keeps its assets for its lender.")):
        out.append({"id": eid, "name": cs.name, "ticker": "", "kind": "example", "summary": blurb + " Hypothetical numbers.",
                    "structure": struct_json(cs), "market": {}, "scenarios": {"low": round(cs.total_debt() * 0.45), "central": round(cs.total_debt() * 0.7),
                                                                              "high": round(cs.total_debt() * 0.95)},
                    "scenario_basis": "share of total debt", "equity": None,
                    "calibration": {"model": "black_cox", "V0": V0, "sigma": sig, "q": 0.0, "H0": cs.default_point(1.0) * 0.9,
                                    "T": 5.0, "r": 0.04, "default_point": cs.total_debt(), "note": "Hypothetical inputs."},
                    "pd_sources": [], "bond_pd_by_recovery": {}, "signals": []})
    return out


def vectors() -> dict:
    """Reference values for the JavaScript parity test."""
    from scipy.special import ndtr
    xs = [-6.0, -3.2, -1.0, -0.1, 0.0, 0.3, 1.7, 4.5]
    vec = {"normcdf": {"x": xs, "y": [float(ndtr(x)) for x in xs]}}
    case = load_case("coreweave")
    cs = case.structure
    Vs = [5000.0, 19872.0, 31228.0, 42584.0, 60000.0, 90000.0]
    res = distribute(cs, np.array(Vs))
    vec["waterfall_coreweave"] = {"structure": struct_json(cs), "V": Vs,
                                  "payouts": {c.id: [float(x) for x in res.payouts[c.id]] for c in cs.claims}}
    ex = examples()
    lbo = ex[0]["structure"]
    from creditlab.case import Case  # noqa: F401
    cs_lbo = CapitalStructure("lbo", [Entity(e["id"], e["name"], e["parent"], e["share"]) for e in lbo["entities"]],
                              [CollateralPool(p["id"], p["entity"], p["share"]) for p in lbo["pools"]],
                              [Claim(c["id"], c["name"], c["kind"], c["entity"], c["amount"], c["rank"],
                                     tuple(tuple(x) for x in c["secured_by"]), tuple(c["guarantors"]), c["guarantee_rank"],
                                     tuple(c["turnover_to"]), c["maturity"]) for c in lbo["claims"]], admin_cost=lbo["admin_cost"])
    res2 = distribute(cs_lbo, np.array([800.0, 1500.0, 2100.0, 2600.0]), gift_to_equity=0.03)
    vec["waterfall_lbo_gift"] = {"structure": lbo, "V": [800.0, 1500.0, 2100.0, 2600.0], "gift": 0.03,
                                 "payouts": {c.id: [float(x) for x in res2.payouts[c.id]] for c in cs_lbo.claims}}
    vec["breakpoints_lbo"] = {"in_default": [float(x) for x in breakpoints(cs_lbo, in_default=True)]}
    for model, H in (("merton", None), ("black_cox", 1300.0)):
        sp = price_structure(cs_lbo, 2600.0, 0.32, 5.0, 0.04, model=model, H0=H, default_point=cs_lbo.total_debt(), with_sigma_E=False)
        vec[f"pricing_lbo_{model}"] = {"V0": 2600.0, "sigma": 0.32, "T": 5.0, "r": 0.04, "H0": H, "default_point": cs_lbo.total_debt(),
                                       "values": sp.values, "pd": sp.pd_q, "pd_barrier": sp.pd_barrier}
    vec["first_passage"] = {"args": [100.0, 0.03, 0.35, 2.0, 70.0], "p": float(first_passage_prob(100.0, 0.03, 0.35, 2.0, 70.0))}
    vec["discounted_hit"] = {"args": [100.0, 0.02, 0.4, 3.0, 60.0, 0.05], "v": float(discounted_hit(100.0, 0.02, 0.4, 3.0, 60.0, 0.05))}
    p, ev = barrier_partial_moments(100.0, 0.03, 0.35, 2.0, 70.0, 0.0, 90.0, 150.0)
    vec["barrier_moments"] = {"args": [100.0, 0.03, 0.35, 2.0, 70.0, 0.0, 90.0, 150.0], "p": float(p), "ev": float(ev)}
    p, ev = partial_moments(100.0, 0.05, 0.3, 2.0, 80.0, 120.0)
    vec["partial_moments"] = {"args": [100.0, 0.05, 0.3, 2.0, 80.0, 120.0], "p": float(p), "ev": float(ev)}
    vec["pd_term"] = {"args": [100.0, 0.35, 70.0, 0.0, 0.04, 0.0], "times": [0.5, 1, 2, 5, 10],
                      "pd": pd_term_structure(100.0, 0.35, 70.0, 0.0, 0.04, 0.0, [0.5, 1, 2, 5, 10]).tolist()}
    return vec


def main():
    (WEB / "data").mkdir(parents=True, exist_ok=True)
    (WEB / "test").mkdir(parents=True, exist_ok=True)
    cases = [case_json(c) for c in ("coreweave", "strategy", "nebius", "iren", "wolfspeed", "wolfspeed_2025")] + examples()
    meta = {"generated_from": REPO, "market_date": "2026-11-20", "units": "USD millions", "cases": cases}
    (WEB / "data" / "cases.json").write_text(json.dumps(meta, separators=(",", ":")), encoding="utf-8")
    (WEB / "test" / "vectors.json").write_text(json.dumps(vectors(), indent=1), encoding="utf-8")
    print("cases.json", (WEB / "data" / "cases.json").stat().st_size, "bytes; vectors.json written")


if __name__ == "__main__":
    main()
