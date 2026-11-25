"""Write the human-readable reports: reports/cases/<id>/README.md (+ charts) and reports/VALIDATION.md.

Reads reports/cases/<id>/analysis.json, cases/<id>/case.json and reports/validation/results.json.
Usage: python scripts/make_reports.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
# validated categorical slots 1-4 (dataviz reference palette, light mode) + chart chrome
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "text.parse_math": False,
                     "legend.frameon": False})

REPRESENTATIVE = {
    "coreweave": [("ddtl_2_0", "SPV term loan (recourse)"), ("ddtl_4_0", "SPV term loan (non-recourse)"),
                  ("rcf", "Revolver"), ("notes_2031b", "Senior notes and converts")],
    "nebius": [("citi_bridge", "Citi secured bridge"), ("tl_compute2", "Compute II term loan"),
               ("cv_2030_02", "Convertible notes"), ("trade", "Trade claims")],
    "iren": [("gpu_fin_2", "GPU financing"), ("uspp_2031", "Microsoft SPV notes"), ("cv_2033_12", "Convertible notes"),
             ("trade", "Trade claims")],
    "strategy": [("cv_2030b", "Convertible notes"), ("strf", "STRF (senior preferred)"), ("strc", "STRC"),
                 ("strd", "STRD (junior preferred)")],
    "wolfspeed": [("ssn_2030", "First-lien notes"), ("cv_15l_2031", "1.5-lien converts"), ("pik_2l_2031", "Second-lien notes"),
                  ("other_liab", "Other liabilities")],
}


def pct(x, d=0):
    return "n/a" if x is None else f"{x * 100:.{d}f}%"


def num(x, d=1):
    return "n/a" if x is None else f"{x:,.{d}f}"


def recovery_chart(cid, a, case, path):
    rc = a["recovery_curves"]
    V = np.array(rc["V"]) / 1000
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ys = {}
    for (claim, label), col in zip(REPRESENTATIVE[cid], (S1, S2, S3, S4)):
        y = np.array([np.nan if v is None else v for v in rc["recovery"][claim]]) * 100
        ys[label] = y
        ax.plot(V, y, color=col, lw=2, label=label)
    # direct labels where each curve is farthest from the others (no collisions)
    used = []
    for label, y in ys.items():
        best, best_gap = None, -1.0
        for k in range(int(0.08 * len(V)), int(0.7 * len(V))):
            if not (8 < y[k] < 92):
                continue
            gap = min([abs(y[k] - o[k]) for l2, o in ys.items() if l2 != label] or [99])
            gap = min([gap] + [np.hypot((V[k] - ux) / V.max() * 100, y[k] - uy) for ux, uy in used])
            if gap > best_gap:
                best, best_gap = k, gap
        if best is not None:
            used.append((V[best], y[best]))
            ax.annotate(label, (V[best], y[best]), xytext=(5, -11), textcoords="offset points", color=INK2, fontsize=8.5)
    for k, ls in (("low", ":"), ("central", "--"), ("high", ":")):
        if k in rc["scenarios"]:
            x = rc["scenarios"][k]["V"] / 1000
            ax.axvline(x, color=MUTED, lw=1, ls=ls)
            ax.text(x, 103, k, ha="center", color=MUTED, fontsize=8)
    ax.set_ylim(0, 110)
    ax.set_xlim(0, V.max())
    ax.set_xlabel("Value available to creditors at default ($bn)")
    ax.set_ylabel("Recovery (% of claim)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"{case['company']}: recovery by claim as value at default changes", loc="left", color=INK, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def pd_chart(a, case, path):
    s = a["structural"]
    rows = []
    b = [x for x in a["bonds"]["bonds"] if x.get("pd_identified")]
    if b:
        rows.append(("Bonds, waterfall recovery", float(np.mean([x["pd_5y"] for x in b]))))
    cvs = [c for c in a["converts"]["converts"] if c.get("implied_pd_5y_straight") is not None and c["parity"] < 60]
    if cvs:
        rows.append(("Converts, option ignored (upper bound)", float(np.median([c["implied_pd_5y_straight"] for c in cvs]))))
    if s["black_cox_waterfall"].get("pd_5y") is not None:
        rows.append(("Black-Cox (equity-calibrated)", s["black_cox_waterfall"]["pd_5y"]))
    rows.append(("CreditGrades", s["creditgrades"]["pd_5y"]))
    if a.get("observable_assets"):
        rows.append(("Observed assets (bitcoin + cash)", a["observable_assets"]["by_horizon"]["5.0"]["pd_debt_default"]))
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(rows) + 1.1))
    ax.set_axisbelow(True)
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] * 100 for r in rows], color=S1, height=0.55)
    for yi, (lab, v) in zip(y, rows):
        ax.text(v * 100 + 1, yi, f"{v:.0%}", va="center", color=INK2, fontsize=9)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Risk-neutral probability of default within 5 years (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"{case['company']}: five-year default probability by source", loc="left", color=INK, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def case_report(cid):
    a = json.loads((REP / "cases" / cid / "analysis.json").read_text())
    case = json.loads((ROOT / "cases" / cid / "case.json").read_text(encoding="utf-8"))
    out = REP / "cases" / cid
    L = [f"# {case['company']} ({case['ticker']})", "",
         f"Balance sheet {case['balance_sheet_date']} ({case['filing']['form']}, accession {case['filing']['accession']}); "
         f"market data {case['market_date']} close. Amounts in $ millions. Generated by `scripts/make_reports.py`; "
         "inputs and their sources are in `cases/" + cid + "/case.json`.", "", case.get("summary", ""), ""]
    if "historical" in a:
        return wolfspeed_2025_report(a, case, L, out)
    s = a["structural"]
    st = a["structure"]
    L += ["## Capital structure", "",
          f"Debt and other claims ranking ahead of equity: **${st['total_claims']:,.0f}m** (allowed amounts at default: principal "
          f"plus six months' interest, revolver 85% drawn). Equity market value ${st['market_cap']:,.0f}m. Debt-weighted "
          f"maturity {st['wam_years']:.1f} years.", "",
          "| Claim | Obligor | Kind | Allowed claim | Rank | Secured by | Guarantors | Source |", "|---|---|---|---|---|---|---|---|"]
    for c in case["claims"]:
        if c["kind"] == "common":
            continue
        sec = ", ".join(f"{p} (lien {l})" for p, l in c["secured_by"]) or "-"
        src = (c.get("source") or {}).get("section") or (c.get("source") or {}).get("doc") or "-"
        L.append(f"| {c['name']} | {c['entity']} | {c['kind']} | {c['amount']:,.0f} | {c['rank']} | {sec} | "
                 f"{', '.join(c['guarantors']) or '-'} | {src} |")
    L += ["", "## Recoveries from the waterfall", "", f"![Recovery curves](recovery.png)", ""]
    rc = a["recovery_curves"]
    if rc["scenarios"]:
        L += [f"Scenario values ({rc.get('scenario_basis', '')}): " + ", ".join(
            f"{k} ${v['V']:,.0f}m" for k, v in rc["scenarios"].items()) + ".", "",
              "| Claim | Low | Central | High |", "|---|---|---|---|"]
        for c in case["claims"]:
            if c["kind"] == "common":
                continue
            L.append(f"| {c['name']} | " + " | ".join(pct(rc["scenarios"][k]["recovery"][c["id"]]) for k in ("low", "central", "high")) + " |")
        L += ["", f"Fulcrum (most senior impaired) claim: " + ", ".join(f"{k}: `{v['fulcrum']}`" for k, v in rc["scenarios"].items()), ""]
    L += [f"Recoveries used to read default risk from prices: {a.get('recovery_source')}.", ""]
    if a["bonds"]["bonds"]:
        L += ["## Bonds: market-implied default risk", "",
              "| Bond | Clean | Last trade | YTM | Z-spread | Recovery | Hazard | PD 1y | PD 5y | Par-equiv. CDS |", "|---|---|---|---|---|---|---|---|---|---|"]
        for b in a["bonds"]["bonds"]:
            ident = b.get("pd_identified")
            pecs = f"{b['pecs'] * 1e4:,.0f}bp" if (ident and b.get("pecs")) else "-"
            L.append(f"| {b['name']} | {b['clean']:.3f} | {b['trade_date']} | {pct(b['ytm'], 2)} | {b['zspread'] * 1e4:,.0f}bp | "
                     f"{pct(b['recovery_used'])} | {pct(b['hazard'], 1) if ident else 'not identified'} | "
                     f"{pct(b['pd_1y']) if ident else '-'} | {pct(b['pd_5y']) if ident else '-'} | {pecs} |")
        L.append("")
        if a["bonds"]["curve"]:
            L += ["Residuals against one hazard curve fitted to all the notes (positive = cheap):", "",
                  "| Bond | Market | Curve price | Residual (bp) |", "|---|---|---|---|"]
            for r in a["bonds"]["curve"]["residuals"]:
                L.append(f"| {r['bond']} | {r['market']:.3f} | {r['model']:.3f} | {r['spread_resid_bp']:+.0f} |")
            L.append("")
        if a["bonds"]["carry"]:
            L += ["One-year carry and roll-down at unchanged curves, minus the expected default loss at market-implied PDs:", "",
                  "| Bond | Income | Roll-down | Funding | Total | Expected loss | Total minus EL |", "|---|---|---|---|---|---|---|"]
            for c in a["bonds"]["carry"]:
                L.append(f"| {c['name']} | {c['income_pct_pa']:.1f}% | {c['roll_pct_pa']:.1f}% | {c['funding_pct_pa']:.1f}% | "
                         f"{c['total_pct_pa']:.1f}% | {c['expected_loss_pct']:.1f}% | {c['carry_minus_el_pct']:.1f}% |")
            L.append("")
    L += ["## Structural models (calibrated to the equity)", "",
          f"Equity ${s['equity_value']:,.0f}m, volatility {pct(s['equity_vol'])} ({s['equity_vol_source']}).", "",
          "| Model | Asset value | Asset vol | PD 1y | PD 5y / to T | Notes |", "|---|---|---|---|---|---|"]
    mk, mw, bc, cg = s["merton_kmv"], s["merton_wam"], s["black_cox_waterfall"], s["creditgrades"]
    L.append(f"| Merton, KMV default point (1y) | {num(mk.get('V'), 0)} | {pct(mk.get('sigma_V'))} | {pct(mk.get('pd_1y'), 1)} | - | default point ${s['default_point_1y']:,.0f}m |")
    L.append(f"| Merton, all debt due at {s['horizon']:.1f}y | {num(mw.get('V'), 0)} | {pct(mw.get('sigma_V'))} | - | {pct(mw.get('pd_T'), 1)} | expected recovery {pct(mw.get('recovery_q'))} |")
    L.append(f"| Black-Cox + waterfall | {num(bc.get('V'), 0)} | {pct(bc.get('sigma_V'))} | {pct(bc.get('pd_1y'), 1)} | {pct(bc.get('pd_5y'), 1)} | barrier ${bc.get('H0', 0):,.0f}m |")
    L.append(f"| CreditGrades | - | - | {pct(1 - cg['survival']['1'], 1)} | {pct(cg['pd_5y'], 1)} | 5y CDS-equivalent {cg['cds_5y'] * 1e4:,.0f}bp |")
    L += ["", "![PD by source](pd_sources.png)", ""]
    if bc.get("bonds"):
        L += ["Bond prices from the equity-calibrated Black-Cox model (survival from the asset process, recovery from the "
              "waterfall at the barrier):", "", "| Bond | Market | Model | Market minus model |", "|---|---|---|---|"]
        for mb in bc["bonds"]:
            L.append(f"| {mb['claim_id']} | {mb['market']:.2f} | {mb['model']:.2f} | {mb['gap_points']:+.2f} |")
        L.append("")
    if a["converts"]["converts"]:
        L += ["## Convertibles", "", f"Hazard input: {a.get('convert_hazard_source', 'n/a')}.", "",
              "| Convert | Market | Last trade | Parity | Bond floor | Model | Implied vol at that hazard | Hazard, option ignored |",
              "|---|---|---|---|---|---|---|---|"]
        for c in a["converts"]["converts"]:
            ivh = c.get("implied_vol_at_hazard")
            hs = c.get("implied_hazard_straight")
            L.append(f"| {c['name']} | {num(c['market'], 2)} | {c.get('trade_date') or '-'} | {c['parity']:.1f} | {c['bond_floor']:.1f} | "
                     f"{c['model']:.1f} | {pct(ivh) if ivh is not None else 'below zero-vol value'} | {pct(hs, 1) if hs is not None else '-'} |")
        L.append("")
    if a["options"].get("bounds"):
        t = a["options"]["tightest"]
        L += ["## Deep puts as default insurance", "",
              f"Tightest bound: {t['expiry']} ${t['strike']:g} put, bid {t['bid']} / ask {t['ask']} (open interest {t['open_interest']}): "
              f"risk-neutral default probability by expiry at most **{pct(t['pd_upper_bound'], 1)}** (stock worthless at default).", ""]
    if a.get("cds"):
        c = a["cds"]
        L += ["## CDS (22 October 2026 snapshot)", "",
              f"Markit 5Y par spread {c['quote']['par_spread_bp']:,.1f}bp; our converter turns the conventional spread into "
              f"{c['upfront_check_pct']:.2f} points upfront against Markit's {c['upfront_reported_pct']:.3f}.", "",
              "| Bond (BondTerminal 22 Oct) | Price | Par-equivalent CDS | CDS at bond maturity | Basis |", "|---|---|---|---|---|"]
        for b in c["basis"]:
            L.append(f"| {b['claim_id']} | {b['bond_price']:.2f} | {b['pecs_bp']:,.0f}bp | {b['cds_at_bond_maturity_bp']:,.0f}bp | {b['basis_bp']:+.0f}bp |")
        L += ["", "The CDS curve is a single 5Y point, so the CDS spread at each bond's maturity assumes a flat curve.", ""]
    dec = a.get("decomposition")
    if dec:
        ph = dec["physical"]
        L += ["## Spread decomposition", "",
              f"Expected loss uses S&P's historical default intensity for {ph['rating']}-rated issuers ({ph['default_rate']:.2%} "
              f"defaulted in the {ph['cohort']} cohort of {ph['n']}: {ph['hazard']:.2%} a year) and the waterfall's central "
              f"recovery. Spreads are the bonds' par-equivalent CDS spreads, so they compare directly with CDS. Notes' rating: "
              f"{dec['rating_source']}.", ""]
        if dec["dated"]:
            L += [f"On {dec['dated'][0]['date']} (the one date with a documented CDS quote, BondTerminal bond marks):", "",
                  "| Bond | Bond spread (CDS-equivalent) | CDS at bond maturity | Expected loss | Default-risk premium | Non-default component |",
                  "|---|---|---|---|---|---|"]
            for x in dec["dated"]:
                L.append(f"| {x['claim_id']} | {x['pecs_bp']:,.0f}bp | {x['cds_bp']:,.0f}bp | {x['expected_loss_bp']:,.0f}bp | "
                         f"{x['default_risk_premium_bp']:,.0f}bp | {x['non_default_bp']:+,.0f}bp |")
            L += ["", "Expected loss at historical default rates is about a tenth of the default component: the rest is a "
                  "risk premium, or the market sees this issuer as far riskier than the average issuer with its rating. A "
                  "negative non-default component means the bonds trade tighter than CDS (a positive basis).", ""]
        if dec["current"]:
            L += [f"On {dec['current'][0]['date']} (no CDS quote): the risk premium and the non-default part cannot be separated.", "",
                  "| Bond | Bond spread (CDS-equivalent) | Expected loss | Remainder |", "|---|---|---|---|"]
            for x in dec["current"]:
                L.append(f"| {x['name']} | {x['pecs_bp']:,.0f}bp | {x['expected_loss_bp']:,.0f}bp | {x['pecs_bp'] - x['expected_loss_bp']:,.0f}bp |")
            L.append("")
    tr = a.get("example_trade")
    if tr:
        p = tr["put"]
        L += ["## Example position and scenario P&L", "",
              f"Long ${tr['face_musd']:.0f}m face of {tr['note_name']} at {tr['note_price']:.3f}, plus {tr['contracts']:,.0f} "
              f"{p['expiry']} ${p['strike']:g} puts bought at the ask (${p['ask']:.2f}; premium ${tr['premium_musd']:.2f}m), sized "
              f"so an immediate default at the central recovery ({tr['recovery_central']:.0%}) costs nothing. Puts are repriced "
              f"at {p['iv']:.0%} volatility: the {p['iv_source']}.", "",
              "| Scenario | Notes ($m) | Puts ($m) | Total ($m) |", "|---|---|---|---|"]
        for r in tr["rows"]:
            L.append(f"| {r['scenario']} | {r['bond']:+.2f} | {r['puts']:+.2f} | {r['total']:+.2f} |")
        down = next((r for r in tr["rows"] if r["scenario"].startswith("Equity -")), None)
        road = ""
        if down and down["puts"] > 0:
            road = (f" If the equity halves without a default, the notes lose {-down['bond'] / down['puts']:.1f} times what the "
                    "puts gain: the hedge covers the jump to default, not the spread widening on the way there.")
        L += ["", "The puts cancel an immediate default at the recovery they were sized for; at other recoveries the position "
              "gains or loses the difference." + road, "", "Caveats: " + "; ".join(tr["caveats"]) + ".", ""]
    oa = a.get("observable_assets")
    if oa:
        L += ["## Observable assets", "",
              f"Assets ${oa['V0']:,.0f}m ({oa['btc_held']:,} BTC at ${oa['btc_price']:,.0f} plus cash), volatility {pct(oa['sigma_V'])}, "
              f"annual dividends and interest ${oa['annual_payout']:,.0f}m ({pct(oa['payout_rate'], 1)} of assets).", "",
              "| Horizon | Model equity | Market equity | Debt default probability |", "|---|---|---|---|"]
        for T, v in oa["by_horizon"].items():
            L.append(f"| {float(T):.0f}y | {v['equity_model']:,.0f} | {v['equity_market']:,.0f} | {pct(v['pd_debt_default'], 2)} |")
        L += ["", "Equity-based calibration against the observed assets:", "", "| Calibration | Implied assets | Error | Implied asset vol | vs observed |", "|---|---|---|---|---|"]
        for k, v in oa["calibration_check"].items():
            if "V" in v:
                L.append(f"| {k} | {v['V']:,.0f} | {v['V_error']:+.0%} | {pct(v['sigma_V'])} | {v['sigma_ratio']:.2f}x |")
        L += ["", "| Preferred | Price | Dividend | Current yield | Spread over 30y UST | Model spread (10y) | P(impaired by 10y) | Asset fall to impair |",
              "|---|---|---|---|---|---|---|---|"]
        for p in oa["preferreds"]:
            L.append(f"| {p['ticker']}{'' if p['cumulative'] else ' (non-cumulative)'} | {p['price']:.2f} | {pct(p['dividend_rate'])} | {pct(p['current_yield'], 1)} | "
                     f"{pct(p['market_spread'], 1)} | {pct(p['model_spread_10y'], 1)} | {pct(p['pd_impaired_10y'], 1)} | {pct(p.get('asset_drop_to_impair'))} |")
        L.append("")
    L += ["## Missing information", ""] + [f"- {m}" for m in case.get("missing", [])] + ["", "## Assumptions", ""]
    L += [f"- **{x['id']}** ({x['status']}): {x['basis']}" for x in case.get("assumptions", [])]
    recovery_chart(cid, a, case, out / "recovery.png")
    pd_chart(a, case, out / "pd_sources.png")
    (out / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", out / "README.md")


def wolfspeed_2025_report(a, case, L, out):
    h = a["historical"]
    L += ["## 1. The court-filed liquidation analysis, reproduced", "",
          "| Class | Claim ($m) | Engine, low | Filed, low | Engine, high | Filed, high |", "|---|---|---|---|---|---|"]
    rep = case["historical"]["liquidation_recoveries_reported"]
    for c in case["claims"]:
        if c["kind"] == "common":
            continue
        key = "ssn" if c["id"] == "ssn" else ("admin" if c["kind"] == "admin" else "unsecured")
        L.append(f"| {c['name']} | {c['amount']:,.1f} | {pct(h['liquidation']['low']['recovery'][c['id']], 1)} | {pct(rep[key][0], 1)} | "
                 f"{pct(h['liquidation']['high']['recovery'][c['id']], 1)} | {pct(rep[key][1], 1)} |")
    pv = h["plan_vs_apr"]
    der = h["plan_recoveries_derived"]
    L += ["", "## 2. Plan recoveries versus absolute priority at the plan's valuation", "",
          "| Class | Plan, as filed | Plan, recomputed from consideration (low / mid / high equity value) | Absolute priority at plan EV $2.35bn / $2.6bn / $2.85bn |",
          "|---|---|---|---|",
          f"| Senior secured notes | 86% | {pct(der['mid']['ssn'])} | " + " / ".join(pct(pv[k]['apr_recovery']['ssn']) for k in ('low', 'mid', 'high')) + " |",
          f"| Convertible notes | 25-30% | " + " / ".join(pct(der[k]['convertible']) for k in ('low', 'mid', 'high')) + " | " + " / ".join(pct(pv[k]['apr_recovery']['cv_2029'], 1) for k in ('low', 'mid', 'high')) + " |",
          f"| Renesas | 27-29% | " + " / ".join(pct(der[k]['renesas']) for k in ('low', 'mid', 'high')) + " | " + " / ".join(pct(pv[k]['apr_recovery']['renesas'], 1) for k in ('low', 'mid', 'high')) + " |",
          "| General unsecured | 100% (unimpaired) | 100% | " + " / ".join(pct(pv[k]['apr_recovery']['guc'], 1) for k in ('low', 'mid', 'high')) + " |",
          f"| Old equity | 5% of new equity | ${der['mid']['old_equity_value']:,.0f}m at mid value | 0 |", "",
          "Absolute priority at the plan's own enterprise value would have left the unsecured classes with almost nothing, "
          "because the secured claim was allowed including its make-whole and the priority claims (including $549.5m of "
          "government grants) rank ahead. The plan instead paid trade creditors in full, gave the notes and Renesas "
          "25-30%, and gave old shareholders 5% of the new equity. The difference is what negotiated restructurings add: "
          "value from outside enterprise value (cash on hand, a backstopped rights offering), concessions by the secured "
          "class (86% of an allowed claim that included the make-whole), claims left unimpaired to keep the business "
          "running, and a gift to equity in exchange for a fast, consensual case.", "",
          f"## 3. One year after emergence", "",
          f"Market value of the new equity on {h['post_emergence']['date']}: ${h['post_emergence']['market_cap']:,.0f}m "
          f"(shares from the latest 10-Q cover x closing price), against the plan's equity value range "
          f"${h['post_emergence']['plan_equity_value']['low']:,}-{h['post_emergence']['plan_equity_value']['high']:,}m.", ""]
    (out / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", out / "README.md")


def validation_report():
    v = json.loads((REP / "validation" / "results.json").read_text())
    L = ["# Validation", "", "Generated by `scripts/run_validation.py` and `scripts/make_reports.py`. Every number below is "
         "reproducible from the repository (market inputs are dated snapshots).", "",
         "## 1. Benchmarks against published or independent answers", "",
         "| Test | Quantity | This engine | Reference | Abs. error | Reference source |", "|---|---|---|---|---|---|"]
    for r in v["benchmarks"]:
        L.append(f"| {r['test']} | {r['quantity']} | {r['model']:.4f} | {r['reference']:.4f} | {r['abs_error']:.2g} | {r['reference_source']} |")
    oa = v["observable_assets"]
    L += ["", "## 2. Structural calibration where the asset value is observable (Strategy)", "",
          f"Strategy's assets are mostly bitcoin, so asset value (${oa['V0']:,.0f}m) and volatility ({oa['sigma_V']:.1%}) can be "
          "measured directly instead of being inferred from the equity. Calibrating the textbook Merton model to the equity "
          "instead gives:", "", "| Calibration | Implied assets | Error vs observed | Implied asset vol | Ratio to observed |", "|---|---|---|---|---|"]
    for k, c in oa["calibration_check"].items():
        if "V" in c:
            L.append(f"| {k} | {c['V']:,.0f} | {c['V_error']:+.0%} | {c['sigma_V']:.0%} | {c['sigma_ratio']:.2f}x |")
    L += ["", "The equity is more volatile than leverage alone explains (its premium or discount to the bitcoin it owns moves "
          "too), so equity-based calibration understates assets and roughly doubles asset volatility. For companies whose "
          "assets cannot be observed, this error is invisible.", ""]
    s = v["sensitivities"]
    L += ["## 3. Sensitivity", "", f"### Bond-implied default probability depends on the recovery assumed ({s['bond_pd_vs_recovery']['bond']}, price {s['bond_pd_vs_recovery']['price']})", "",
          "| Recovery | Hazard | 5-year PD |", "|---|---|---|"]
    for r in s["bond_pd_vs_recovery"]["rows"][::2]:
        L.append(f"| {r['recovery']:.0%} | {r['hazard']:.1%} | {r['pd_5y']:.0%} |")
    L += ["", "### Structural default probability depends on the volatility input and the barrier (CoreWeave)", "",
          "| Equity vol source | Vol | Barrier / KMV point | Asset vol | 5-year barrier PD |", "|---|---|---|---|---|"]
    for r in s["structural_grid"]:
        if "error" in r:
            L.append(f"| {r['vol_source']} | {r['equity_vol']:.0%} | {r['barrier_fraction']} | no fit | - |")
        else:
            L.append(f"| {r['vol_source']} | {r['equity_vol']:.0%} | {r['barrier_fraction']} | {r['sigma_V']:.0%} | {r['pd_5y_barrier']:.0%} |")
    t = s["recovery_tornado"]
    L += ["", f"### What moves the senior notes' recovery (CoreWeave, base {t['base']:.1%} at ${t['base_value']:,.0f}m)", "",
          "| Driver | Low case | High case |", "|---|---|---|"]
    for r in sorted(t["rows"], key=lambda r: -(abs(r["high"] - r["low"]))):
        L.append(f"| {r['driver']} | {r['low']:.1%} | {r['high']:.1%} |")
    rs = v["restructuring"]
    L += ["", f"## 4. Restructuring scenarios (CoreWeave, value ${rs['value']:,.0f}m)", ""]
    keys = list(next(iter(rs["scenarios"].values())).keys())
    L += ["| Scenario | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for name, row in rs["scenarios"].items():
        L.append(f"| {name} | " + " | ".join(
            (f"{row[k]:,.0f}" if k.startswith("Equity") else (pct(row[k]) if row[k] is not None else "-")) for k in keys) + " |")
    h = v["historical"]
    L += ["", "## 5. Historical default experience", "", f"Source: {h['source']}.", "",
          "### Rating-transition model vs what the cohorts did", "",
          "Multi-year default rates from powers of the one-year matrix (2024-25, paid-off and withdrawn removed) against "
          "the single cohorts S&P reports:", "", "| Rating | Horizon | Markov from 1y matrix | Cohort reported | Cohort size |", "|---|---|---|---|---|"]
    for r in h["markov_vs_cohort"]:
        L.append(f"| {r['rating']} | {r['horizon']} | {r['markov_from_1y_matrix_pct']:.1f}% | {r['cohort_reported_pct']:.2f}% | {r['cohort_n']} |")
    L += ["", "The two disagree by up to half in either direction: a single year's matrix is a noisy, regime-specific "
          "estimate, the cohorts are single starting years, and withdrawn issuers that later default are not counted. "
          "Rating-based PDs carry this uncertainty even before any market risk premium.", "",
          "### Market-implied vs historical (CoreWeave notes)", "", "| Rating compared | Market PD 1y | Historical 1y | Market PD 3y | Historical 3y | Ratio (3y) | Market-implied rating |", "|---|---|---|---|---|---|---|"]
    for r in h["market_vs_historical"]:
        L.append(f"| {r['agency_rating_compared']} | {r['market_pd_1y']:.1%} | {r['historical_1y']:.2%} | {r['market_pd_3y']:.1%} | "
                 f"{r['historical_3y']:.2%} | {r['ratio_3y']:.1f}x | {r['implied_rating_1y']} |")
    L += ["", "## 6. Every model, side by side", "",
          "| Issuer | Merton KMV PD 1y | Black-Cox PD 1y | Black-Cox PD 5y | CreditGrades PD 5y | Bonds PD 5y | Converts PD 5y (upper) | Put bound (horizon) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in v["cross_model"]:
        pb = "-" if r.get("put_pd_bound") is None else f"{r['put_pd_bound']:.0%} ({r['put_bound_horizon']:.1f}y)"
        L.append(f"| {r['issuer']} | {pct(r['merton_kmv_pd_1y'], 1)} | {pct(r['black_cox_pd_1y'], 1)} | {pct(r['black_cox_pd_5y'])} | "
                 f"{pct(r['creditgrades_pd_5y'])} | {pct(r['bond_pd_5y'])} | {pct(r['convert_pd_5y_upper'])} | {pb} |")
    (REP / "VALIDATION.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote reports/VALIDATION.md")


if __name__ == "__main__":
    for cid in ("coreweave", "strategy", "nebius", "iren", "wolfspeed", "wolfspeed_2025"):
        case_report(cid)
    validation_report()
