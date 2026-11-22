"""Case files, the analysis pipeline, the historical benchmark and the scanner (uses committed case files only)."""
import json
from pathlib import Path

import numpy as np
import pytest

from creditlab.analysis import recovery_curves, run_case, structure_summary
from creditlab.case import list_cases, load_case
from creditlab.capital_structure import substantively_consolidated
from creditlab.scanner import Signal, scan_case
from creditlab.waterfall import distribute

ROOT = Path(__file__).resolve().parents[1]


def test_every_case_loads_and_validates():
    ids = list_cases()
    assert {"coreweave", "strategy", "nebius", "iren", "wolfspeed", "wolfspeed_2025"} <= set(ids)
    for cid in ids:
        case = load_case(cid)
        assert case.structure.validate() == []
        for c in case.data["claims"]:
            if c["kind"] != "common":
                assert c["source"], f"{cid}/{c['id']} has no source"
        for a in case.data.get("assumptions", []):
            assert a["basis"] and a["status"] in {"filed", "derived", "estimate", "convention"}


def test_coreweave_capital_structure_ties_to_the_10q():
    d = json.loads((ROOT / "cases" / "coreweave" / "case.json").read_text(encoding="utf-8"))
    funded = [c for c in d["claims"] if c["id"] not in ("trade", "lease_rej", "equity")]
    principal = sum(c["principal"] for c in funded) - 0.85 * 2500 + 1200   # revolver: drawn $1.2bn at 9/30
    assert principal == pytest.approx(47505, abs=2)    # 10-Q Note 10 total debt principal


def test_wolfspeed_liquidation_analysis_reproduced_exactly():
    case = load_case("wolfspeed_2025")
    h = case.data["historical"]
    lo = distribute(case.structure, np.array([h["liquidation_proceeds"]["low"]]), admin_cost=0.0)
    hi = distribute(case.structure, np.array([h["liquidation_proceeds"]["high"]]), admin_cost=0.0)
    assert lo.recovery["ssn"][0] == pytest.approx(0.790, abs=5e-4)
    assert lo.recovery["cv_2029"][0] == 0.0
    assert hi.recovery["ssn"][0] == pytest.approx(1.0)
    assert hi.recovery["grants"][0] == pytest.approx(1.0)
    assert hi.recovery["renesas"][0] == pytest.approx(0.038, abs=5e-4)
    assert hi.payouts["guc"][0] == pytest.approx(11.406, abs=0.01)   # filed: $11,406k


def test_substantive_consolidation_conserves_value():
    case = load_case("coreweave")
    cs2 = substantively_consolidated(case.structure)
    assert cs2.validate() == []
    V = 31228.0
    a = distribute(cs2, np.array([V]))
    total = sum(a.payouts[c.id][0] for c in cs2.claims)
    assert total == pytest.approx(V * (1 - cs2.admin_cost))


def test_analysis_runs_and_is_internally_consistent():
    case = load_case("coreweave")
    res = run_case(case)
    s = res["structural"]
    assert s["black_cox_waterfall"]["converged"]
    assert s["black_cox_waterfall"]["equity_model"] == pytest.approx(case.market_cap, rel=1e-4)
    for b in res["bonds"]["bonds"]:
        assert 0 < b["pd_5y"] < 1 and b["zspread"] > 0
    rc = recovery_curves(case)
    for cid, ys in rc["recovery"].items():
        if cid == "equity_value":
            continue
        y = np.array([np.nan if v is None else v for v in ys])
        assert np.all(np.diff(y[~np.isnan(y)]) >= -1e-9)
    summ = structure_summary(case)
    assert summ["total_claims"] > summ["market_cap"]


def test_scanner_flags_and_confidence():
    s = Signal("X", "curve", "a", "b", "m", 30.0, "bp", "r", ["stale_price", "short_bond_funding", "dated_inputs"]).finalize(1.0)
    assert s.confidence == "low" and s.score == pytest.approx(0.3)
    s2 = Signal("X", "curve", "a", "b", "m", 30.0, "bp", "r", []).finalize(1.0)
    assert s2.confidence == "high"
    case = json.loads((ROOT / "cases" / "coreweave" / "case.json").read_text(encoding="utf-8"))
    a = run_case(load_case("coreweave"))
    sigs = scan_case(case, a)
    assert sigs and all(sig.confidence in ("high", "medium", "low") for sig in sigs)
    assert all("arbitrage" not in sig.read.lower() for sig in sigs)
