"""Relative-value scanner: turn case analyses into ranked candidate trades with explicit risk flags.

A signal is a statement of the form "security A looks cheap relative to security B by X, measured
this way". It is never labelled an arbitrage. Every signal lists the frictions that could make the gap
persist or reverse against a hedged position:

  stale_price          last trade more than 4 calendar days before the market date
  price_dispersion     the same bond prints >= 1.5 points apart on different lines (144A vs Reg S)
  recovery_sensitive   the implied default probability moves > 25 points across 0-60% recovery
  pd_not_identified    recovery near 100%: the spread is not informative about default
  short_bond_funding   the trade shorts a bond or convert: repo / borrow cost unobserved
  short_equity_borrow  the trade shorts equity: borrow cost and recall risk unobserved
  illiquid_option      option open interest < 100 or bid-ask wider than 50% of mid
  model_dependent      the sign of the signal depends on the model variant
  dated_inputs         inputs from different dates (e.g. CDS from an earlier close)
  structural_inputs    depends on asset volatility / barrier choices of a structural model
  restricted_access    144A / Reg S securities: generally institutional-only
  deep_itm_convert     convert trades on equity, not credit (parity > 120)
Confidence: high = no flags; medium = 1-2; low = 3 or more, or model_dependent.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Signal:
    issuer: str
    kind: str
    long: str
    short: str
    metric: str
    value: float
    unit: str
    read: str
    flags: list = field(default_factory=list)
    confidence: str = ""
    score: float = 0.0

    def finalize(self, strength: float):
        n = len(set(self.flags))
        self.confidence = "low" if (n >= 3 or "model_dependent" in self.flags) else ("medium" if n else "high")
        penalty = {"high": 1.0, "medium": 0.6, "low": 0.3}[self.confidence]
        self.score = float(strength * penalty)
        return self


def _bond_flags(b: dict) -> list:
    f = []
    if b.get("stale_days", 0) > 4:
        f.append("stale_price")
    if abs(b.get("stock_move_since_trade") or 0) > 0.05:
        f.append("stale_price")
    if (b.get("price_dispersion") or 0) >= 1.5:
        f.append("price_dispersion")
    pdr = b.get("pd_by_recovery") or {}
    if pdr and (max(pdr.values()) - min(pdr.values())) > 0.25:
        f.append("recovery_sensitive")
    if not b.get("pd_identified", True):
        f.append("pd_not_identified")
    return f


def scan_case(case_json: dict, analysis: dict) -> list[Signal]:
    issuer = case_json["company"]
    sigs: list[Signal] = []
    market_date = case_json["market_date"]
    bonds = analysis.get("bonds", {}).get("bonds", [])
    bmap = {b["claim_id"]: b for b in bonds}
    access = []
    if case_json["ticker"] in ("CRWV",):
        access = ["restricted_access"]
    # 1. curve residuals
    curve = analysis.get("bonds", {}).get("curve")
    if curve:
        res = curve["residuals"]
        cheap = max(res, key=lambda r: r["spread_resid_bp"])
        rich = min(res, key=lambda r: r["spread_resid_bp"])
        gap = cheap["spread_resid_bp"] - rich["spread_resid_bp"]
        if gap > 10:
            cid_c = [k for k, v in bmap.items() if v["name"] == cheap["bond"]][0]
            cid_r = [k for k, v in bmap.items() if v["name"] == rich["bond"]][0]
            flags = _bond_flags(bmap[cid_c]) + _bond_flags(bmap[cid_r]) + ["short_bond_funding"] + access
            sigs.append(Signal(issuer, "curve", cheap["bond"], rich["bond"], "spread residual gap vs fitted hazard curve",
                               gap, "bp", f"{cheap['bond']} trades {cheap['spread_resid_bp']:+.0f}bp off the issuer curve, "
                               f"{rich['bond']} {rich['spread_resid_bp']:+.0f}bp. Coupon and call differences are not modelled.",
                               sorted(set(flags))).finalize(gap / 50))
    # 2. CDS-bond basis (dated)
    cdsa = analysis.get("cds")
    if cdsa and cdsa.get("basis"):
        for b in cdsa["basis"]:
            if b.get("basis_bp") is None:
                continue
            v = b["basis_bp"]
            long_, short_ = (b["claim_id"], "CDS protection (buy)") if v > 0 else ("CDS protection (sell)", b["claim_id"])
            read = (f"CDS {b['cds_at_bond_maturity_bp']:.0f}bp vs the bond's par-equivalent spread {b['pecs_bp']:.0f}bp "
                    f"(basis {v:+.0f}bp) on {cdsa['quote']['date']}. A positive basis means protection is dear relative to the "
                    "bond: long bond + buy protection earns the basis if both are held to maturity or default, but the "
                    "CDS curve is a single 5Y point here.")
            sigs.append(Signal(issuer, "cds_basis", long_, short_, "CDS minus par-equivalent bond spread", v, "bp", read,
                               ["dated_inputs"] + access).finalize(abs(v) / 100))
    # 3. equity vs credit (structural bond prices)
    bc = analysis.get("structural", {}).get("black_cox_waterfall", {})
    for mb in bc.get("bonds", []):
        g = mb["gap_points"]
        b = bmap.get(mb["claim_id"], {})
        if abs(g) < 2 or not b.get("pd_identified", True):
            continue  # near-par recovery: price driven by calls and coupon, not default
        if g > 0:
            long_, short_, read = ("equity (or equity puts)", mb["claim_id"],
                                   f"Market {mb['market']:.2f} vs {mb['model']:.2f} from the equity-calibrated Black-Cox model: the bond "
                                   "prices less default than the equity implies (bond rich / equity cheap).")
            flags = ["structural_inputs", "short_bond_funding"] + _bond_flags(b)
        else:
            long_, short_, read = (mb["claim_id"], "equity", f"Market {mb['market']:.2f} vs model {mb['model']:.2f}: bond cheap vs equity.")
            flags = ["structural_inputs", "short_equity_borrow"] + _bond_flags(b)
        sigs.append(Signal(issuer, "equity_vs_credit", long_, short_, "market minus structural model price", g, "points",
                           read, sorted(set(flags + access))).finalize(abs(g) / 5))
    # 4. bonds vs deep puts
    opt = analysis.get("options", {})
    if opt.get("bond_vs_put"):
        t = opt["tightest"]
        for c in opt["bond_vs_put"]:
            if not c["bond_exceeds_put_bound"]:
                continue
            b = bmap.get(c["claim_id"], {})
            excess = c["pd_bond_at_put_horizon"] - c["put_upper_bound"]
            flags = _bond_flags(b)
            if (t.get("open_interest") or 0) < 100 or (t["ask"] - t["bid"]) > 0.5 * 0.5 * (t["ask"] + t["bid"]):
                flags.append("illiquid_option")
            read = (f"At the waterfall recovery the bond implies {c['pd_bond_at_put_horizon']:.0%} default by {t['expiry']}, but the "
                    f"{t['expiry']} ${t['strike']:g} put bid caps the risk-neutral default probability at {t['put_upper_bound'] if 'put_upper_bound' in t else c['put_upper_bound']:.0%} "
                    "(stock assumed worthless at default). Long bond + long puts is the classic structure; the gap closes if recovery is higher than assumed.")
            sigs.append(Signal(issuer, "bond_vs_puts", c["claim_id"], f"equity put {t['expiry']} {t['strike']:g}",
                               "bond-implied PD minus put upper bound", excess, "probability", read,
                               sorted(set(flags + access))).finalize(excess / 0.1))
    # 5. converts vs options and straight bonds
    for cv in analysis.get("converts", {}).get("converts", []):
        if cv.get("market") is None:
            continue
        flags = []
        if cv["parity"] > 120:
            flags.append("deep_itm_convert")
            continue
        gap = cv.get("gap_points", 0.0)
        hs = cv.get("implied_hazard_straight")
        h_used = cv["hazard_used"]
        straight_says_rich = hs is not None and np.isfinite(hs) and hs < h_used
        model_says_cheap = gap < -2
        if model_says_cheap and straight_says_rich:
            flags.append("model_dependent")
        trade_date = cv.get("trade_date")
        if trade_date and trade_date < market_date and (np.datetime64(market_date) - np.datetime64(trade_date)).astype(int) > 4:
            flags.append("stale_price")
        mv = cv.get("stock_move_since_trade")
        if mv is not None and abs(mv) > 0.05:
            # the print predates a large stock move: compare nothing, the gap is an artefact of timing
            continue
        if abs(gap) < 2:
            continue
        iv_h = cv.get("implied_vol_at_hazard")
        iv_txt = f"{iv_h:.0%}" if iv_h is not None and np.isfinite(iv_h) else "n/a (below the zero-volatility value)"
        read = (f"Market {cv['market']:.2f} vs {cv['model']:.2f} (jump-to-default tree at listed vol {cv['vol_used']:.0%}, hazard "
                f"{cv['hazard_used']:.1%}, recovery {cv['recovery']:.0%}); implied vol at that hazard {iv_txt}. Option-ignoring "
                f"bound: hazard <= {hs:.1%}." if hs is not None and np.isfinite(hs) else "")
        long_, short_ = (cv["claim_id"], "listed calls / equity delta") if gap < 0 else ("listed calls", cv["claim_id"])
        sigs.append(Signal(issuer, "convert_vs_options", long_, short_, "market minus model price", gap, "points", read,
                           sorted(set(flags + ["short_equity_borrow"] + access))).finalize(abs(gap) / 5))
    # 6. preferred stack (observable assets)
    oa = analysis.get("observable_assets")
    if oa:
        prefs = oa["preferreds"]
        for hi, lo in zip(prefs[:-1], prefs[1:]):
            mkt_gap = lo["market_spread"] - hi["market_spread"]
            mod_gap = (lo["model_spread_10y"] or 0) - (hi["model_spread_10y"] or 0)
            excess = mkt_gap - mod_gap
            read = (f"{lo['ticker']} yields {lo['market_spread']:.1%} over 30y Treasuries vs {hi['ticker']} {hi['market_spread']:.1%}; "
                    f"liquidation priority alone explains {mod_gap:.1%} of the {mkt_gap:.1%} gap at observed bitcoin value and "
                    f"volatility. Dividend features (cumulative or not), rate resets and liquidity explain the rest - or the "
                    f"junior series is cheap.")
            flags = ["structural_inputs"]
            if not lo["cumulative"]:
                flags.append("model_dependent")
            sigs.append(Signal(issuer, "preferred_stack", lo["ticker"], hi["ticker"], "market spread gap minus model gap",
                               excess * 1e4, "bp", read, flags).finalize(abs(excess) / 0.02))
    return sigs


def cross_issuer(rows: list[dict]) -> list[Signal]:
    """Compare market-implied and equity-implied 5-year default probabilities across issuers."""
    sigs = []
    for r in rows:
        if r.get("market_pd_5y") is None or r.get("model_pd_5y") in (None, 0):
            continue
        ratio = r["market_pd_5y"] / r["model_pd_5y"]
        read = (f"Debt markets imply {r['market_pd_5y']:.1%} 5-year default ({r['market_source']}); the structural model "
                f"({r.get('model_source', '')}) {r['model_pd_5y']:.1%}. Ratio {ratio:.2f}.")
        long_, short_ = ("debt", "equity") if ratio > 1 else ("equity", "debt")
        sigs.append(Signal(r["issuer"], "cross_market_pd", long_, short_, "market / model 5y PD", ratio, "x", read,
                           ["structural_inputs", "recovery_sensitive"]).finalize(abs(np.log(ratio)) / 0.3))
    return sigs


def run(root: Path) -> list[dict]:
    sigs: list[Signal] = []
    xrows = []
    for d in sorted((root / "reports" / "cases").iterdir()):
        a_path = d / "analysis.json"
        c_path = root / "cases" / d.name / "case.json"
        if not a_path.exists() or not c_path.exists():
            continue
        a = json.loads(a_path.read_text())
        c = json.loads(c_path.read_text(encoding="utf-8"))
        if "historical" in a:
            continue
        sigs += scan_case(c, a)
        bc = a.get("structural", {}).get("black_cox_waterfall", {})
        model_pd = bc.get("pd_5y")
        oa = a.get("observable_assets")
        if oa:  # assets observed directly: no calibration needed
            model_pd = oa["by_horizon"]["5.0"]["pd_debt_default"]
        mpd, src = None, None
        ident = [b for b in a["bonds"]["bonds"] if b.get("pd_identified")]
        if ident:
            mpd = float(np.mean([b["pd_5y"] for b in ident]))
            src = "straight bonds at waterfall recovery"
        else:
            cvs = [cv for cv in a["converts"]["converts"] if cv.get("implied_pd_5y_straight") is not None and cv["parity"] < 60]
            if cvs:
                mpd = float(np.median([cv["implied_pd_5y_straight"] for cv in cvs]))
                src = "out-of-the-money converts, option ignored (upper bound)"
        xrows.append({"issuer": c["company"], "market_pd_5y": mpd, "market_source": src, "model_pd_5y": model_pd,
                      "model_source": "observed assets (bitcoin + cash)" if oa else "equity-calibrated Black-Cox"})
    sigs += cross_issuer(xrows)
    sigs.sort(key=lambda s: -s.score)
    return [asdict(s) for s in sigs]
