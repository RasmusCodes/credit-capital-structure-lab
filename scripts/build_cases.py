"""Build the documented case files cases/<id>/case.json from filing extracts and market snapshots.

Inputs (see docs/DATA.md):
  data/filings/<T>_capstructure.json          facts extracted from 10-Q / 6-K / 8-K filings, each with a quote
  data/raw/trace/2026-11-21/finra_last_trades.json   FINRA TRACE last sales (git-ignored; prices used are copied
                                                into the case files with CUSIP, date and source)
  data/snapshots/equity_stats_2026-11-21.json last closes and realised volatilities
  data/raw/options/2026-11-21/summary.json    listed option quotes (git-ignored; derived IVs go into case files)
  data/raw/yahoo/2026-11-21/BTC-USD.json      bitcoin closes (Strategy case)

Every modelling judgment is written into the case's `assumptions` list with its basis and status
("convention", "estimate", "derived"). Run: python scripts/build_cases.py
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

from creditlab.capital_structure import lease_rejection_claim
from creditlab.case import treasury_curve
from creditlab.options import bs_implied_vol, put_pd_upper_bound

ROOT = Path(__file__).resolve().parents[1]
MARKET_DATE = "2026-11-20"
SETTLE = "2026-11-23"
SNAP = "2026-11-21"
TRACE_SRC = "FINRA TRACE last sale (finra.org Corporate and Agency Bonds)"
EQ_SRC = "Yahoo Finance daily close"
SP_CONV = "S&P recovery-rating convention (as used in the October 2026 CoreWeave note)"


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


FILINGS = {t: jload(ROOT / "data" / "filings" / f"{t}_capstructure.json") for t in ("CRWV", "NBIS", "IREN", "MSTR", "WOLF")}
TRACE_RAW = jload(ROOT / "data" / "raw" / "trace" / SNAP / "finra_last_trades.json")
TRACE = {}
for issuer, rows in TRACE_RAW["data"].items():
    for r in rows:
        rec = dict(zip(TRACE_RAW["columns"], r))
        TRACE[rec["cusip"]] = rec
EQ = jload(ROOT / "data" / "snapshots" / f"equity_stats_{SNAP}.json")
OPT = jload(ROOT / "data" / "raw" / "options" / SNAP / "summary.json")
ZC = treasury_curve(MARKET_DATE)


def inst(t, prefix):
    for i in FILINGS[t]["instruments"]:
        if i["name"].startswith(prefix):
            return i
    raise KeyError(f"{t}: {prefix}")


def src(t, item=None, section=None, quote=None):
    f = FILINGS[t]
    if item is not None:
        s = item.get("source", {})
        return {"doc": s.get("doc"), "accession": s.get("accession"), "section": s.get("section"),
                "quote": s.get("quote")}
    return {"doc": f"{f['filing']['form']} for period ended {f['balance_sheet_date']}",
            "accession": f["filing"]["accession"], "section": section, "quote": quote}


ISSUER_TICKER = {"COREWEAVE": "CRWV", "NEBIUS": "NBIS", "IREN": "IREN", "MICROSTRATEGY": "MSTR", "WOLFSPEED": "WOLF"}
_CLOSES: dict = {}


def stock_close(ticker, date):
    """Close of `ticker` on `date` from the cached Yahoo chart (to judge how stale a bond print is)."""
    if ticker not in _CLOSES:
        j = jload(ROOT / "data" / "raw" / "yahoo" / SNAP / f"{ticker}.json")["chart"]["result"][0]
        _CLOSES[ticker] = {dt.datetime.fromtimestamp(ts, dt.UTC).date().isoformat(): c
                           for ts, c in zip(j["timestamp"], j["indicators"]["quote"][0]["close"]) if c}
    return _CLOSES[ticker].get(date)


def trace(cusip, alts=()):
    r = TRACE[cusip]
    out = {"cusip": cusip, "clean": r["lastSalePrice"], "date": r["lastTradeDate"], "yield_reported": r["lastSaleYield"],
           "source": TRACE_SRC}
    issuer = next((k for k in ISSUER_TICKER if r["issuerName"].startswith(k)), None)
    if issuer:
        tk = ISSUER_TICKER[issuer]
        s_trade, s_now = stock_close(tk, r["lastTradeDate"]), stock_close(tk, MARKET_DATE)
        if s_trade and s_now:
            out["stock_on_trade_date"] = round(s_trade, 4)
            out["stock_move_since_trade"] = round(s_now / s_trade - 1, 4)
    if alts:
        out["alt"] = [{"cusip": a, "clean": TRACE[a]["lastSalePrice"], "date": TRACE[a]["lastTradeDate"]} for a in alts]
    return out


def claim(cid, name, kind, entity, principal, coupon=None, months=6.0, rank=10, secured_by=(), guarantors=(),
          maturity=None, instrument=None, market=None, source=None, status="filed", notes="", amount=None, **kw):
    if amount is None:
        amount = principal * (1 + (coupon or 0.0) * months / 12.0)
    c = {"id": cid, "name": name, "kind": kind, "entity": entity, "principal": round(principal, 3),
         "amount": round(amount, 3), "rank": rank, "secured_by": [list(x) for x in secured_by],
         "guarantors": list(guarantors), "maturity": maturity, "coupon": coupon, "instrument": instrument or {},
         "market": market or {}, "source": source or {}, "status": status, "notes": notes}
    c.update(kw)
    return c


def equity_block(sym, shares_m, shares_asof, shares_src, extra_shares=None):
    e = EQ[sym]
    out = {"ticker": sym, "price": e["last_close"], "date": e["last_date"], "price_source": EQ_SRC,
           "shares_m": shares_m, "shares_as_of": shares_asof, "shares_source": shares_src,
           "vol_1y": e["vol_1y"], "vol_6m": e["vol_6m"], "vol_3m": e["vol_3m"], "vol_ewma94": e["vol_ewma94"],
           "vol_source": "realised, daily log returns (Yahoo Finance closes)",
           "return_1m": e["return_1m"], "high_52w": e["high_52w"], "low_52w": e["low_52w"]}
    if extra_shares:
        out["extra_shares"] = extra_shares
    if sym in OPT:
        out["options"] = options_block(sym)
    return out


def options_block(sym):
    o = OPT[sym]
    S = o["spot"]
    term = []
    for e in o["expiries"]:
        a = e.get("atm")
        if not a or e["T"] < 20 / 365:
            continue
        r = float(ZC.zero(e["T"]))
        iv = bs_implied_vol(a["call_mid"], S, a["strike"], e["T"], r, 0.0, "call")
        if np.isfinite(iv):
            term.append({"expiry": e["expiry"], "T": round(e["T"], 4), "strike": a["strike"], "iv": round(iv, 4)})
    Ts = np.array([x["T"] for x in term])
    ivs = np.array([x["iv"] for x in term])
    iv1 = float(np.interp(1.0, Ts, ivs)) if len(term) else None
    iv2 = float(np.interp(2.0, Ts, ivs)) if len(term) else None
    bounds = []
    for e in o["expiries"]:
        if e["T"] < 0.9:
            continue
        r = float(ZC.zero(e["T"]))
        for p in e["deep_puts"][:2]:
            if p["bid"] and p["bid"] > 0:
                bounds.append({"expiry": e["expiry"], "T": round(e["T"], 3), "strike": p["strike"], "bid": p["bid"],
                               "ask": p["ask"], "open_interest": p["oi"],
                               "pd_upper_bound": round(put_pd_upper_bound(p["bid"], p["strike"], e["T"], r), 4)})
    return {"source": "Yahoo Finance listed options, two-sided quotes at the 2026-11-20 close; IV from the ATM call mid "
                      "(Black-Scholes, no dividends, Treasury zero rate)",
            "atm_iv_term": term, "iv_1y": iv1, "iv_2y": iv2, "deep_put_pd_bounds": bounds}


def assumption(aid, value, basis, status="estimate", rng=None):
    a = {"id": aid, "value": value, "basis": basis, "status": status}
    if rng is not None:
        a["range"] = rng
    return a


def write(case):
    # rounded value shares must still sum to exactly one: put the rounding residue on the largest entity
    ents = case["entities"]
    big = max(ents, key=lambda e: e["value_share"])
    big["value_share"] = round(1.0 - sum(e["value_share"] for e in ents if e is not big), 9)
    out = ROOT / "cases" / case["id"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "case.json").write_text(json.dumps(case, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote cases/{case['id']}/case.json: {len(case['claims'])} claims")


def base(cid, company, ticker, t):
    f = FILINGS[t]
    return {"id": cid, "company": company, "ticker": ticker, "currency": "USD", "units": "USD millions",
            "balance_sheet_date": f["balance_sheet_date"], "market_date": MARKET_DATE, "settlement_date": SETTLE,
            "filing": f["filing"], "sources": f.get("sources", []), "missing": list(f.get("missing", []))}


# --------------------------------------------------------------------------------------------- CoreWeave
def coreweave():
    t = "CRWV"
    f = FILINGS[t]
    c = base("coreweave", "CoreWeave, Inc.", "CRWV", t)
    ppe = f["other_liabilities"]["ppe_net_musd"]          # 56,778
    spv_nc, nr_nc = 22200.0, 5200.0                          # 10-Q Note 10
    oem_nr = inst(t, "OEM and Software License Financing (no")["principal_outstanding_musd"]
    shares = {"spv_nr": nr_nc / ppe, "spv_r": (spv_nc - nr_nc) / ppe, "oem_nr": oem_nr / ppe}
    shares["opco"] = 1.0 - sum(shares.values())
    opco_book = shares["opco"] * ppe
    oem_r = inst(t, "OEM and Software License Financing (re")
    rcf = inst(t, "Revolving")
    rcf_claim = 0.85 * rcf["commitment_musd"]
    c["summary"] = ("AI cloud provider financed with SPV delayed-draw term loans secured on GPU fleets and customer "
                    "contracts, OEM equipment financing, a secured revolver, and parent-level senior notes and "
                    "convertibles guaranteed by the operating subsidiaries.")
    c["entities"] = [
        {"id": "parent", "name": "CoreWeave, Inc. (issuer of notes, converts, revolver borrower)", "parent": None, "value_share": 0.0},
        {"id": "opco", "name": "Guarantor and other operating subsidiaries", "parent": "parent", "value_share": round(shares["opco"], 6)},
        {"id": "spv_r", "name": "SPV borrowers of recourse DDTLs (1.0, 2.0, 2.1, 3.0, 5.0, 5.5)", "parent": "opco", "value_share": round(shares["spv_r"], 6)},
        {"id": "spv_nr", "name": "CCAC VIII (DDTL 4.0 borrower, non-recourse)", "parent": "opco", "value_share": round(shares["spv_nr"], 6)},
        {"id": "oem_nr", "name": "Subsidiaries with non-recourse OEM financing", "parent": "opco", "value_share": round(shares["oem_nr"], 6)},
    ]
    c["pools"] = [
        {"id": "p_spv_r", "entity": "spv_r", "share": 1.0, "description": "substantially all assets of the SPV borrowers + equity pledges"},
        {"id": "p_spv_nr", "entity": "spv_nr", "share": 1.0, "description": "substantially all assets of CCAC VIII"},
        {"id": "p_oem_nr", "entity": "oem_nr", "share": 1.0, "description": "financed equipment"},
        {"id": "p_oem", "entity": "opco", "share": round(oem_r["principal_outstanding_musd"] / opco_book, 6), "description": "equipment financed by recourse OEM / software financing (book value assumed equal to balance)"},
        {"id": "p_rcf", "entity": "opco", "share": round(rcf_claim / opco_book, 6), "description": "assets pledged to the revolver (size not disclosed; set equal to its assumed default claim at book)"},
    ]
    claims = []
    ddtls = [("ddtl_1_0", "DDTL 1.0 Facility"), ("ddtl_2_0", "DDTL 2.0 Facility"), ("ddtl_2_1", "DDTL 2.1 Facility"),
             ("ddtl_3_0", "DDTL 3.0 Facility"), ("ddtl_5_0", "DDTL 5.0 Facility"), ("ddtl_5_5", "DDTL 5.5 Facility")]
    for cid, nm in ddtls:
        i = inst(t, nm)
        claims.append(claim(cid, nm + " (SPV, parent-guaranteed)", "delayed_draw_term_loan", "spv_r",
                            i["principal_outstanding_musd"], i["effective_rate_pct"] / 100, secured_by=[("p_spv_r", 1)],
                            guarantors=["parent"], maturity=i["maturity"], source=src(t, i), status=i["status"],
                            notes="Each SPV has its own collateral; pooling the recourse SPVs is a simplification."))
    i = inst(t, "DDTL 4.0")
    claims.append(claim("ddtl_4_0", "DDTL 4.0 Facility (CCAC VIII, non-recourse)", "delayed_draw_term_loan", "spv_nr",
                        i["principal_outstanding_musd"], i["effective_rate_pct"] / 100, secured_by=[("p_spv_nr", 1)],
                        maturity=i["maturity"], source=src(t, i), status=i["status"],
                        notes="$3.9bn undrawn commitment not included (assumed undrawn at default)."))
    i = inst(t, "OEM and Software License Financing (no")
    claims.append(claim("oem_nr", "OEM financing (non-recourse)", "equipment_financing", "oem_nr", i["principal_outstanding_musd"],
                        i["effective_rate_pct"] / 100, secured_by=[("p_oem_nr", 1)], maturity="2028-08-15", source=src(t, i)))
    claims.append(claim("oem_r", "OEM and software financing (recourse)", "equipment_financing", "opco",
                        oem_r["principal_outstanding_musd"], oem_r["effective_rate_pct"] / 100, secured_by=[("p_oem", 1)],
                        guarantors=["parent"], maturity="2030-07-15", source=src(t, oem_r)))
    claims.append(claim("rcf", "Revolving credit facility (85% of $2.5bn commitment drawn at default)", "revolver", "parent",
                        rcf_claim, rcf["effective_rate_pct"] / 100, secured_by=[("p_rcf", 1)], guarantors=["opco"],
                        maturity="2029-11-15", source=src(t, rcf), status="derived",
                        notes=f"Drawn $1,200m + $546m letters of credit at 9/30; default draw per {SP_CONV}."))
    notes = [("notes_2030", "9.25% Senior Notes due 2030", "21873SAB4", ["U2069EAA8"]),
             ("notes_2031a", "9.00% Senior Notes due 2031", "21873SAC2", ["U2069EAB6"]),
             ("notes_2031b", "9.75% Senior Notes due 2031", "21873SAG3", ["U2069EAC4", "U2069EAD2"]),
             ("notes_2032", "9.625% Senior Notes due 2032", "21873SAK4", ["U2069EAE0"])]
    for cid, nm, cusip, alts in notes:
        i = inst(t, nm)
        claims.append(claim(cid, nm, "senior_unsecured", "parent", i["principal_outstanding_musd"], i["coupon_pct"] / 100,
                            guarantors=["opco"], maturity=i["maturity"], source=src(t, i),
                            instrument={"type": "bond", "freq": 2, "isin": i.get("isin")}, market=trace(cusip, alts),
                            notes="Callable; call schedule not modelled (prices far below call prices)."))
    i = inst(t, "8.50% EUR")
    claims.append(claim("notes_eur_2032", "8.50% EUR Senior Notes due 2032 (EUR 2.0bn)", "senior_unsecured", "parent",
                        i["principal_outstanding_musd"], 0.085, guarantors=["opco"], maturity="2032-07-15", source=src(t, i),
                        status="derived", notes="USD carrying amount at 9/30 FX; no USD price on TRACE."))
    cvs = [("cv_2031", "1.75% Convertible Senior Notes due 2031", "21873SAD0", 107.80, None),
           ("cv_2032", "1.75% Convertible Senior Notes due 2032", "21873SAH1", 119.60, "2029-10-08"),
           ("cv_2033", "2.875% Convertible Senior Notes due 20", "21873SAL2", 97.85, "2030-04-01")]
    for cid, nm, cusip, cp, call in cvs:
        i = inst(t, nm)
        claims.append(claim(cid, i["name"], "convertible", "parent", i["principal_outstanding_musd"], i["coupon_pct"] / 100,
                            guarantors=["opco"], maturity=i["maturity"], source=src(t, i),
                            instrument={"type": "convertible", "conversion_price": cp, "first_call": call},
                            market=trace(cusip),
                            notes=("Subsidiary guarantees confirmed in the 8-K indenture description for the 2031 and 2033 "
                                   "series; assumed identical for the 2032 series (8-K not checked).")))
    i = inst(t, "Magnetar")
    claims.append(claim("magnetar", "Magnetar loan", "other_debt", "parent", i["principal_outstanding_musd"], 0.0, source=src(t, i)))
    ol = f["other_liabilities"]
    claims.append(claim("trade", "Accounts payable and accrued liabilities", "trade", "opco",
                        ol["accounts_payable_musd"] + ol["accrued_liabilities_musd"], 0.0,
                        source=src(t, section="Balance sheet", quote="accounts payable 2,928; accrued liabilities 5,641"),
                        notes="Treated as general unsecured claims at the operating subsidiaries."))
    rej = lease_rejection_claim(0.5 * 2591.0, 32954.0 / 2591.0)
    claims.append(claim("lease_rej", "Lease-rejection claims (50% of operating leases rejected)", "lease", "opco", rej, 0.0,
                        status="derived", source=src(t, section="Note 8 lease maturity table",
                                                     quote="Total undiscounted lease payments | 32,954; 2028 | 2,591"),
                        notes="11 U.S.C. 502(b)(6) cap on half of the operating leases: rent $2,591m/yr (2028), remaining rent $32,954m."))
    claims.append(claim("equity", "Class A + B common stock", "common", "parent", 0.0, amount=0.0))
    c["claims"] = claims
    c["admin_cost"] = 0.05
    c["equity"] = equity_block("CRWV", f["shares_outstanding"]["total"], f["shares_outstanding"]["as_of"], "10-Q cover page")
    c["cds"] = [{"tenor": "5Y", "maturity": "2031-12-20", "par_spread_bp": 1140.598, "conventional_spread_bp": 1167.011,
                 "upfront_pct": 19.834, "coupon_bp": 500, "recovery": 0.40, "date": "2026-10-22", "tier": "SNRFOR", "doc": "XR",
                 "source": "Markit 5Y official close via LSEG Workspace (CDSOV page), recorded in the October 2026 CoreWeave note"}]
    c["dated_bond_marks"] = {"date": "2026-10-22", "source": "BondTerminal closing marks recorded in the October 2026 CoreWeave note",
                             "marks": {"notes_2030": 84.35, "notes_2031a": 80.57, "notes_2031b": 80.36, "notes_2032": 79.25}}
    c["ratings"] = {"notes": {"sp": TRACE["21873SAC2"]["sp"], "moodys": TRACE["21873SAC2"]["moodys"],
                              "source": "FINRA TRACE security data for the senior notes, queried 2026-11-21"},
                    "issuer": {"sp": "B+", "fitch": "BB-", "source": "LSEG CDSOV page recorded 2026-10-23 (ratings dated 19-May-2025)"}}
    ppe_val = ppe
    c["default_value_scenarios"] = {"basis": "enterprise value at default as a share of net PP&E ($56.8bn)",
                                    "low": round(0.35 * ppe_val), "central": round(0.55 * ppe_val), "high": round(0.75 * ppe_val)}
    c["assumptions"] = [
        assumption("entity_value_shares", shares, "Value at default spread like book PP&E: SPV debt secured by $22.2bn of "
                   "non-current assets, of which CCAC VIII $5.2bn (10-Q Note 10); total net PP&E $56.8bn; non-recourse OEM "
                   "subsidiaries at their financing balance.", "derived"),
        assumption("admin_cost", 0.05, f"Administrative and professional costs, 5% of value ({SP_CONV}).", "convention"),
        assumption("prepetition_interest_months", 6, f"Claims = principal + 6 months' interest ({SP_CONV}).", "convention"),
        assumption("revolver_draw_at_default", 0.85, f"Revolver 85% drawn at default ({SP_CONV}).", "convention"),
        assumption("lease_rejection_share", 0.5, "Half of operating leases rejected; claims capped by 502(b)(6).", "estimate", [0.0, 1.0]),
        assumption("default_value_share_of_ppe", [0.35, 0.55, 0.75], "Low/central/high value at default. Wolfspeed's "
                   "court-filed liquidation analysis recovered 13.7-44.8% of PP&E book value (fabs); a going concern with "
                   "assumed customer contracts can be worth more.", "estimate"),
        assumption("customer_prepayments", "assumed", "Customer contracts ($10.0bn deferred revenue, $132bn RPO) assumed "
                   "in a restructuring, so prepayments create no claims.", "estimate"),
        assumption("undrawn_commitments", 0.0, "DDTL 4.0 ($3.9bn) and 5.5 ($1.0bn) undrawn commitments not drawn before default.", "estimate"),
    ]
    c["missing"] += ["Collateral behind the revolver", "Individual SPV asset values beyond the two aggregates in Note 10",
                     "Call schedules of the senior notes", "Repo / borrow cost for shorting the notes"]
    write(c)


# --------------------------------------------------------------------------------------------- Strategy
def strategy():
    t = "MSTR"
    f = FILINGS[t]
    c = base("strategy", "Strategy Inc", "MSTR", t)
    btc = f["bitcoin"]
    btc_snap = jload(ROOT / "data" / "snapshots" / f"btc_usd_{MARKET_DATE}.json")
    btc_px = btc_snap["price"]
    btc_val = btc["btc_held"] * btc_px / 1e6
    cash = f["cash"]["cash_and_equivalents_musd"] + f["cash"]["other_reserves_musd"]["short_term_investments"]
    c["summary"] = ("Bitcoin treasury company: convertible notes, five perpetual preferred series and common equity "
                    "backed mainly by 850,849 BTC. Unusual for credit work because the asset value is observable.")
    c["entities"] = [{"id": "parent", "name": "Strategy Inc", "parent": None, "value_share": 1.0}]
    c["pools"] = [{"id": "p_nonbtc", "entity": "parent", "share": 0.001, "description": "non-bitcoin assets securing two small loans"}]
    claims = []
    i = inst(t, "Other long-term secured")
    claims.append(claim("secured_loans", "Secured term loans (non-bitcoin collateral)", "term_loan", "parent",
                        i["principal_outstanding_musd"], 0.052, secured_by=[("p_nonbtc", 1)], maturity="2027-06-15", source=src(t, i)))
    cvs = [("cv_2028", "0.625% Convertible Senior Notes due 2028", "594972AQ4"),
           ("cv_2029", "0% Convertible Senior Notes due 2029", "594972AS0"),
           ("cv_2030a", "0.625% Convertible Senior Notes due 2030", "594972AJ0"),
           ("cv_2030b", "0% Convertible Senior Notes due 2030", "594972AU5"),
           ("cv_2031", "0.875% Convertible Senior Notes due 2031", "594972AL5"),
           ("cv_2032", "2.25% Convertible Senior Notes due 2032", "594972AN1")]
    for cid, nm, cusip in cvs:
        i = inst(t, nm)
        assert abs(TRACE[cusip]["coupon"] - i["coupon_pct"]) < 1e-9, (cid, cusip)
        claims.append(claim(cid, i["name"], "convertible", "parent", i["principal_outstanding_musd"], i["coupon_pct"] / 100,
                            maturity=i["maturity"], source=src(t, i),
                            instrument={"type": "convertible", "conversion_price": i["conversion_price_usd"],
                                        "put_dates": [i["holder_put_dates"]], "first_call": i["first_call_date"]},
                            market=trace(cusip)))
    pref_px = {"STRF": EQ["STRF"]["last_close"], "STRC": EQ["STRC"]["last_close"], "STRK": EQ["STRK"]["last_close"],
               "STRD": EQ["STRD"]["last_close"]}
    for k, (tk, nm) in enumerate([("STRF", "10.00% Series A Perpetual Strife"), ("STRC", "Variable Rate Series A Perpetual Stretch"),
                                  ("STRE", "10.00% Series A Perpetual Stream"), ("STRK", "8.00% Series A Perpetual Strike"),
                                  ("STRD", "10.00% Series A Perpetual Stride")]):
        i = inst(t, nm)
        mkt = {"price": pref_px[tk], "date": MARKET_DATE, "source": EQ_SRC} if tk in pref_px else {}
        claims.append(claim(tk.lower(), i["name"], "preferred", "parent", i["principal_outstanding_musd"], 0.0, rank=30 + k,
                            source=src(t, i), market=mkt,
                            instrument={"type": "preferred", "ticker": tk, "dividend_rate": i["coupon_pct"] / 100,
                                        "stated_amount": 100.0, "liquidation_pref": i["liquidation_pref_per_share"],
                                        "cumulative": i["dividend_type"] == "cumulative",
                                        "shares_m": (i["shares_outstanding"] or 0) / 1e6, "currency": i["currency"]},
                            notes=i.get("notes", "")))
    claims.append(claim("equity", "Class A + B common stock", "common", "parent", 0.0, amount=0.0))
    c["claims"] = claims
    c["admin_cost"] = 0.02
    sh = f["shares_outstanding"]
    c["equity"] = equity_block("MSTR", sh["total"] / 1e6, sh["as_of"], "10-Q cover page")
    c["assets"] = {"btc_held": btc["btc_held"], "btc_as_of": btc["as_of"], "btc_source": btc["source"],
                   "btc_price": btc_px, "btc_price_date": btc_snap["time_utc"], "btc_price_source": btc_snap["source"],
                   "btc_value": btc_val, "btc_vol_1y": EQ["BTC-USD"]["vol_1y"], "btc_vol_3m": EQ["BTC-USD"]["vol_3m"],
                   "cash_and_st_investments": cash, "cash_as_of": f["balance_sheet_date"],
                   "usd_reserve_11_15": f["cash"]["other_reserves_musd"]["usd_reserve_11_15"],
                   "total_observable_assets": btc_val + cash,
                   "asset_vol": EQ["BTC-USD"]["vol_1y"] * btc_val / (btc_val + cash)}
    c["assumptions"] = [
        assumption("asset_value", round(btc_val + cash, 1), "Bitcoin held (8-K, 15 Nov) x BTC close + cash and short-term "
                   "investments at 30 Sep. Ignores the software business, taxes and post-quarter cash use.", "derived"),
        assumption("asset_vol", round(EQ["BTC-USD"]["vol_1y"] * btc_val / (btc_val + cash), 4),
                   "1-year realised BTC volatility scaled by bitcoin's share of assets (cash has no volatility).", "derived"),
        assumption("admin_cost", 0.02, "Liquidating bitcoin is cheap relative to an operating business; 2% assumed.", "estimate", [0.0, 0.05]),
        assumption("strc_shares", "9/30 count", "STRC shares are the 30 Sep count (90.96m); 8-K repurchases imply about "
                   "77.65m by 15 Nov (derived in the filing extract).", "derived"),
    ]
    c["missing"] += ["STRE (EUR, Luxembourg-listed) price", "Whether any bitcoin is pledged"]
    write(c)


# --------------------------------------------------------------------------------------------- Wolfspeed (current)
def wolfspeed():
    t = "WOLF"
    f = FILINGS[t]
    c = base("wolfspeed", "Wolfspeed, Inc.", "WOLF", t)
    c["summary"] = ("Silicon-carbide maker that emerged from a prepackaged Chapter 11 in September 2025 with first-lien "
                    "notes, 1.5-lien and second-lien convertibles and second-lien PIK toggle notes.")
    c["entities"] = [{"id": "parent", "name": "Wolfspeed, Inc. and guarantor Wolfspeed Texas LLC", "parent": None, "value_share": 1.0}]
    c["pools"] = [{"id": "p_all", "entity": "parent", "share": 1.0, "description": "substantially all assets (shared by 1L, 1.5L, 2L)"}]
    claims = []
    i = inst(t, "New Senior Secured Notes")
    claims.append(claim("ssn_2030", "15.875% Senior Secured Notes due 2030 (first lien)", "secured_notes", "parent",
                        i["principal_outstanding_musd"], 0.15875, secured_by=[("p_all", 1)], maturity="2030-06-23",
                        source=src(t, i), instrument={"type": "bond", "freq": 2}, market=trace("977852AK8", ["U97162AC2"]),
                        notes="Coupon stepped to 15.875% cash on 2026-06-23 (10-Q). Callable; schedule not modelled."))
    i = inst(t, "3.5% Convertible 1.5")
    claims.append(claim("cv_15l_2031", "3.5% Convertible 1.5-Lien Notes due 2031", "secured_notes", "parent",
                        i["principal_outstanding_musd"], 0.035, secured_by=[("p_all", 2)], maturity="2031-03-15", source=src(t, i),
                        instrument={"type": "convertible", "conversion_price": 20.14, "first_call": "2028-03-20"},
                        market=trace("97785WAC0")))
    i = inst(t, "7.00% / 12.00%")
    claims.append(claim("pik_2l_2031", "7%/12% Second-Lien PIK Toggle Notes due 2031", "second_lien_notes", "parent",
                        i["principal_outstanding_musd"], 0.12, secured_by=[("p_all", 3)], maturity="2031-06-15", source=src(t, i),
                        market=trace("977852AS1"), notes="Last TRACE print 2026-10-08 (stale)."))
    i = inst(t, "2.5% Convertible Second-Lien Senior Secured Notes due 2031 (New 2L Non")
    claims.append(claim("cv_2l_2031", "2.5% Second-Lien Convertible Notes due 2031 (non-Renesas)", "second_lien_notes", "parent",
                        i["principal_outstanding_musd"], 0.025, secured_by=[("p_all", 3)], maturity="2031-06-15", source=src(t, i),
                        instrument={"type": "convertible", "conversion_price": 12.23, "first_call": "2028-09-29"},
                        market=trace("977852AP7"), notes="Deep in the money: trades on equity, not credit."))
    i = inst(t, "2.5% Convertible Second-Lien Senior Secured Notes due 2031 (New 2L Ren")
    claims.append(claim("cv_2l_renesas", "2.5% Second-Lien Convertible Notes due 2031 (Renesas)", "second_lien_notes", "parent",
                        i["principal_outstanding_musd"], 0.025, secured_by=[("p_all", 3)], maturity="2031-06-15", source=src(t, i),
                        instrument={"type": "convertible", "conversion_price": 18.35}, notes="Held by Renesas; not traded."))
    other = 2071.5 - (930.3 + 750.3)
    claims.append(claim("other_liab", "Other liabilities (payables, accruals, leases)", "trade", "parent", other, 0.0,
                        status="derived", source=src(t, section="Balance sheet 9/27/26",
                                                     quote="total liabilities $2,071.5M; long-term debt $930.3M; convertible notes net $750.3M")))
    claims.append(claim("equity", "Common stock", "common", "parent", 0.0, amount=0.0))
    c["claims"] = claims
    c["admin_cost"] = 0.05
    sh = f["shares_outstanding"]
    c["equity"] = equity_block("WOLF", sh["total"] / 1e6, sh["as_of"], "10-Q cover page")
    noncash = 2875.5 - 995.1
    c["default_value_scenarios"] = {"basis": "share of non-cash assets ($1,880m)", "low": round(0.35 * noncash),
                                    "central": round(0.55 * noncash), "high": round(0.75 * noncash)}
    c["assumptions"] = [
        assumption("admin_cost", 0.05, "5% administrative costs; the 2025 liquidation analysis had Chapter 7 costs of 5.2-5.4% of proceeds.", "convention"),
        assumption("lien_ranks", "1L > 1.5L > 2L", "Intercreditor ranking per instrument names in the 10-Q.", "filed"),
        assumption("default_value_share", [0.35, 0.55, 0.75], "Share of non-cash assets realised at default.", "estimate"),
    ]
    write(c)


# --------------------------------------------------------------------------------------------- Wolfspeed 2025 (historical)
def wolfspeed_2025():
    acc = "0001193125-25-153145"
    ds = {"doc": "Disclosure Statement, Exhibit D Liquidation Analysis (Ex. 99.1 to 8-K filed 2025-07-01)", "accession": acc}
    c = {"id": "wolfspeed_2025", "company": "Wolfspeed, Inc. (pre-petition, June 2025)", "ticker": "WOLF", "currency": "USD",
         "units": "USD millions", "balance_sheet_date": "2025-06-30", "market_date": "2025-06-30", "settlement_date": "2025-07-01",
         "filing": {"form": "8-K Ex. 99.1 (Disclosure Statement)", "accession": acc, "filed": "2025-07-01"},
         "summary": ("Historical test case: claims and recoveries in Wolfspeed's 2025 prepackaged Chapter 11, from the "
                     "court-filed disclosure statement."),
         "entities": [{"id": "parent", "name": "Wolfspeed debtors", "parent": None, "value_share": 1.0}],
         "pools": [{"id": "p_all", "entity": "parent", "share": 1.0, "description": "collateral of the senior secured notes"}],
         "admin_cost": 0.0}
    L = lambda cid, nm, kind, amt, rank=10, sec=(): claim(cid, nm, kind, "parent", amt, 0.0, rank=rank, secured_by=sec,
                                                          amount=amt, source=dict(ds, section="Liquidation Analysis claims table"))
    c["claims"] = [
        L("ssn", "Senior Secured Notes (allowed incl. Make-Whole Amount)", "secured_notes", 1811.576, sec=[("p_all", 1)]),
        L("admin", "Administrative claims", "admin", 56.980, rank=0),
        L("other_secured", "Other secured claims (listed with admin/priority)", "admin", 119.081, rank=0),
        L("priority_tax", "Priority tax claims", "admin", 15.333, rank=0),
        L("grants", "Grants and government subsidies", "admin", 549.500, rank=0),
        L("cv_2026", "Convertible notes due 2026", "convertible", 576.677),
        L("cv_2028", "Convertible notes due 2028", "convertible", 750.693),
        L("cv_2029", "Convertible notes due 2029", "convertible", 1769.141),
        L("renesas", "Unsecured customer refundable deposit (Renesas)", "senior_unsecured", 2127.009),
        L("guc", "General unsecured claims", "trade", 302.646),
        claim("old_equity", "Existing equity", "common", "parent", 0.0, amount=0.0),
    ]
    c["equity"] = {"ticker": "WOLF", "price": None, "shares_m": 156.47939, "shares_as_of": "2025-09-29",
                   "shares_source": "8-K 0001193125-25-224251 (old shares cancelled)"}
    c["historical"] = {
        "liquidation_proceeds": {"low": 1431.121, "high": 2760.730, "source": dict(ds, section="Exhibit D line 14")},
        "liquidation_recoveries_reported": {"ssn": [0.790, 1.000], "admin": [0.0, 1.0], "unsecured": [0.0, 0.038],
                                            "source": dict(ds, section="Exhibit D lines 15-28")},
        "plan_ev": {"low": 2350, "mid": 2600, "high": 2850, "source": "Disclosure Statement Exhibit F valuation (PWP, 12 Jun 2025)"},
        "plan_equity_value": {"low": 850, "mid": 1100, "high": 1350},
        "plan_recoveries_reported": {"ssn": "86%", "convertible": "25-30%", "renesas": "27-29%", "guc": "100% (unimpaired)",
                                     "old_equity": "5.0% of new common (3.0% if Renesas contingency triggered)",
                                     "source": "Disclosure Statement Section I.B summary table"},
        "plan_consideration": {
            "ssn": {"new_senior_secured_notes": 1259.2, "cash_payment": 304.9, "note": "cash = $277.5m redeemed at 109.875% (emergence 8-K), derived"},
            "convertible": {"new_equity_pct": 56.3, "takeback_2l_notes": 296.4, "rights_offering_2l_converts": 301.13},
            "renesas": {"new_equity_pct": 38.7, "renesas_2l_converts": 203.6, "warrants_shares": 4943555},
            "old_equity": {"new_equity_pct": 5.0}},
        "post_emergence_market": {"date": MARKET_DATE, "shares_m": FILINGS["WOLF"]["shares_outstanding"]["total"] / 1e6,
                                  "price": EQ["WOLF"]["last_close"], "source": "10-Q cover share count (30 Oct 2026); Yahoo close"},
    }
    c["assumptions"] = [assumption("claims", "as filed", "Claim amounts exactly as in the liquidation analysis table.", "filed")]
    c["missing"] = ["Unimpaired-class claim amounts under the plan", "Confirmed-plan (not solicitation) recovery estimates"]
    write(c)


# --------------------------------------------------------------------------------------------- Nebius
def nebius():
    t = "NBIS"
    f = FILINGS[t]
    c = base("nebius", "Nebius Group N.V.", "NBIS", t)
    ppe = f["other_liabilities"]["ppe_net_musd"]
    tl = inst(t, "Term Loan Facility")
    c2 = tl["principal_outstanding_musd"] / 0.7
    c["summary"] = ("AI cloud provider funded mainly by parent-level convertible notes, with a $3.5bn secured bridge "
                    "(signed after the quarter) on the operating subsidiaries and an SPV term loan.")
    c["entities"] = [
        {"id": "parent", "name": "Nebius Group N.V. (convert issuer)", "parent": None, "value_share": 0.0},
        {"id": "ops", "name": "Nebius Inc., Nebius B.V. and operating subsidiaries", "parent": "parent", "value_share": round(1 - c2 / ppe, 6)},
        {"id": "compute2", "name": "Nebius Compute II borrowers", "parent": "ops", "value_share": round(c2 / ppe, 6)},
    ]
    c["pools"] = [{"id": "p_ops", "entity": "ops", "share": 1.0, "description": "Citi bridge collateral: substantially all assets of borrowers incl. shares of Nebius Inc./B.V."},
                  {"id": "p_c2", "entity": "compute2", "share": 1.0, "description": "assets of the Compute II borrowers"}]
    br = inst(t, "Citi Bridge")
    claims = [claim("citi_bridge", "Citi senior secured bridge ($3.5bn, assumed fully drawn at default)", "term_loan", "parent", 3500.0,
                    0.07, secured_by=[("p_ops", 1)], guarantors=["ops"], maturity="2027-10-16", source=src(t, br), status="estimate",
                    notes="Signed 2026-10-17, after the balance sheet date; drawn amount not disclosed."),
              claim("tl_compute2", "Nebius Compute II term loan", "term_loan", "compute2", tl["principal_outstanding_musd"], 0.07,
                    secured_by=[("p_c2", 1)], maturity=tl["maturity"], source=src(t, tl), notes="Non-recourse except bad-acts guaranty.")]
    cv_px = {"63954QAB2": None, "63954QAD8": None}
    for i in f["instruments"]:
        if i["type"] != "convertible_notes":
            continue
        cusip = i["cusip"]
        cid = "cv_" + i["maturity"][:7].replace("-", "_")
        claims.append(claim(cid, i["name"], "convertible", "parent", i["principal_outstanding_musd"], i["coupon_pct"] / 100,
                            maturity=i["maturity"], source=src(t, i),
                            instrument={"type": "convertible", "conversion_price": i["conversion_price_usd"]},
                            market=trace(cusip) if cusip in TRACE else {}))
    claims.append(claim("trade", "Accounts payable, accrued and other liabilities", "trade", "ops", 779.0, 0.0,
                        source=src(t, section="Balance sheet", quote="accounts payable, accrued and other liabilities 779.0")))
    claims.append(claim("equity", "Class A + B common (+ NVIDIA pre-funded warrants)", "common", "parent", 0.0, amount=0.0))
    c["claims"] = claims
    c["admin_cost"] = 0.05
    sh = f["shares_outstanding"]
    eq = equity_block("NBIS", sh["total"] + 21.065936, sh["as_of"], "Interim balance sheet; includes 21.07m NVIDIA pre-funded warrants")
    c["equity"] = eq
    c["default_value_scenarios"] = {"basis": f"share of net PP&E (${ppe:,.0f}m)", "low": round(0.35 * ppe),
                                    "central": round(0.55 * ppe), "high": round(0.75 * ppe)}
    c["assumptions"] = [
        assumption("compute2_value", round(c2, 1), "Compute II asset value = term loan / 0.70 loan-to-value (not disclosed).", "estimate"),
        assumption("bridge_drawn_at_default", 3500.0, "Bridge treated as fully drawn before any default.", "estimate"),
        assumption("admin_cost", 0.05, f"{SP_CONV}.", "convention"),
        assumption("prepetition_interest_months", 6, f"{SP_CONV}.", "convention"),
        assumption("customer_prepayments", "assumed", "$9.4bn deferred revenue: contracts assumed, no refund claims.", "estimate"),
        assumption("leases", "omitted", "Lease schedule not extracted; rejection claims omitted (flatters unsecured recoveries).", "estimate"),
        assumption("default_value_share_of_ppe", [0.35, 0.55, 0.75], "As in the CoreWeave case.", "estimate"),
    ]
    write(c)


# --------------------------------------------------------------------------------------------- IREN
def iren():
    t = "IREN"
    f = FILINGS[t]
    c = base("iren", "IREN Limited", "IREN", t)
    ppe = f["other_liabilities"]["ppe_net_musd"]
    ddtl = inst(t, "DDTL Facility")
    uspp = inst(t, "5.96% Senior Secured Notes")
    msft = (ddtl["principal_outstanding_musd"] + uspp["principal_outstanding_musd"]) / 0.8
    ops_book = ppe - msft
    g2 = inst(t, "GPU Financing 2.0")
    g3 = inst(t, "GPU Financing 3.0")
    fl = inst(t, "GPU equipment finance leases")
    c["summary"] = ("Data-center and AI cloud operator: six parent-level convertible issues without subsidiary "
                    "guarantees, GPU equipment financings and a Microsoft-contract SPV financed by a DDTL and private notes.")
    c["entities"] = [
        {"id": "parent", "name": "IREN Limited (convert issuer)", "parent": None, "value_share": 0.0},
        {"id": "ops", "name": "Operating subsidiaries (data centers, GPU fleets)", "parent": "parent", "value_share": round(ops_book / ppe, 6)},
        {"id": "msft_spv", "name": "Microsoft-contract SPV (DDTL + USPP notes)", "parent": "ops", "value_share": round(msft / ppe, 6)},
    ]
    c["pools"] = [
        {"id": "p_gpu2", "entity": "ops", "share": round(g2["principal_outstanding_musd"] / 0.8 / ops_book, 6), "description": "GPUs financed by GPU Financing 2.0 (Mackenzie)"},
        {"id": "p_gpu3", "entity": "ops", "share": round(g3["principal_outstanding_musd"] / 0.8 / ops_book, 6), "description": "equipment financed by CA Leasing"},
        {"id": "p_fl", "entity": "ops", "share": round(fl["principal_outstanding_musd"] / ops_book, 6), "description": "GPUs under finance leases"},
        {"id": "p_msft", "entity": "msft_spv", "share": 1.0, "description": "SPV assets, Microsoft contract cash flows, equity pledge"},
    ]
    claims = [
        claim("gpu_fin_2", g2["name"], "term_loan", "ops", g2["principal_outstanding_musd"], 0.09, secured_by=[("p_gpu2", 1)],
              guarantors=["parent"], maturity=g2["maturity"], source=src(t, g2)),
        claim("gpu_fin_3", g3["name"], "equipment_financing", "ops", g3["principal_outstanding_musd"], 0.0951, secured_by=[("p_gpu3", 1)],
              guarantors=["parent"], maturity=g3["maturity"], source=src(t, g3)),
        claim("fin_leases", fl["name"], "equipment_financing", "ops", fl["principal_outstanding_musd"], 0.0, secured_by=[("p_fl", 1)],
              guarantors=["parent"], source=src(t, fl)),
        claim("msft_ddtl", ddtl["name"], "delayed_draw_term_loan", "msft_spv", ddtl["principal_outstanding_musd"], 0.06,
              secured_by=[("p_msft", 1)], maturity="2031-12-31", source=src(t, ddtl), notes="Shared security package with the USPP notes."),
        claim("uspp_2031", uspp["name"], "secured_notes", "msft_spv", uspp["principal_outstanding_musd"], 0.0596,
              secured_by=[("p_msft", 1)], maturity="2031-12-31", source=src(t, uspp), status=uspp["status"]),
        claim("other_debt", "Other recourse debt", "other_debt", "ops", 14.0, 0.0, source=src(t, inst(t, "Other debt"))),
    ]
    for i in f["instruments"]:
        if i["type"] != "convertible_notes":
            continue
        cusip = i["cusip"]
        cid = "cv_" + i["maturity"][:7].replace("-", "_")
        alts = ["46270CAF6"] if cusip == "46270CAE9" else []
        claims.append(claim(cid, i["name"], "convertible", "parent", i["principal_outstanding_musd"], i["coupon_pct"] / 100,
                            maturity=i["maturity"], source=src(t, i),
                            instrument={"type": "convertible", "conversion_price": i["conversion_price_usd"]},
                            market=trace(cusip, alts) if cusip in TRACE else {},
                            notes="Parent obligation without subsidiary guarantees: structurally subordinated."))
    claims.append(claim("trade", "Payables and accruals (incl. $2,493m PP&E payables)", "trade", "ops", 2583.0, 0.0,
                        source=src(t, section="Balance sheet", quote="AP and accrued combined 2,583 (incl. PP&E payables 2,493)")))
    claims.append(claim("equity", "Ordinary shares", "common", "parent", 0.0, amount=0.0))
    c["claims"] = claims
    c["admin_cost"] = 0.05
    sh = f["shares_outstanding"]
    c["equity"] = equity_block("IREN", sh["total"], sh["as_of"], "10-Q cover page")
    c["default_value_scenarios"] = {"basis": f"share of net PP&E (${ppe:,.0f}m)", "low": round(0.35 * ppe),
                                    "central": round(0.55 * ppe), "high": round(0.75 * ppe)}
    c["assumptions"] = [
        assumption("collateral_ltv", 0.8, "GPU financings and the Microsoft SPV debt at 80% of the book value of their collateral (not disclosed).", "estimate"),
        assumption("admin_cost", 0.05, f"{SP_CONV}.", "convention"),
        assumption("prepetition_interest_months", 6, f"{SP_CONV}.", "convention"),
        assumption("customer_prepayments", "assumed", "$2.5bn Microsoft prepayment: contract assumed, no refund claim.", "estimate"),
        assumption("default_value_share_of_ppe", [0.35, 0.55, 0.75], "As in the CoreWeave case.", "estimate"),
    ]
    write(c)


if __name__ == "__main__":
    coreweave()
    strategy()
    wolfspeed()
    wolfspeed_2025()
    nebius()
    iren()
