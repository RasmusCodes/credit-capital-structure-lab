"""Bankruptcy waterfall invariants and legal-structure effects."""
import numpy as np
import pytest

from creditlab.capital_structure import (CapitalStructure, Claim, CollateralPool, Entity,
                                         lease_rejection_claim)
from creditlab.waterfall import breakpoints, distribute, full_payment_threshold, fulcrum


def holdco_opco(upstream_guarantee=False):
    ents = [Entity("hold", "HoldCo", None, 0.1), Entity("op", "OpCo", "hold", 0.6), Entity("spv", "SPV", "op", 0.3)]
    pools = [CollateralPool("gpu", "spv", 1.0, "GPUs and contracts"), CollateralPool("abl", "op", 0.4, "receivables")]
    claims = [
        Claim("dd", "SPV term loan", "delayed_draw_term_loan", "spv", 50.0, secured_by=(("gpu", 1),), guarantors=("hold",)),
        Claim("rcf", "Revolver", "revolver", "op", 20.0, secured_by=(("abl", 1),)),
        Claim("trade", "Trade payables", "trade", "op", 15.0),
        Claim("notes", "HoldCo notes", "senior_unsecured", "hold", 60.0,
              guarantors=("op",) if upstream_guarantee else ()),
        Claim("pref", "Preferred", "preferred", "hold", 10.0, rank=30),
        Claim("eq", "Common", "common", "hold"),
    ]
    return CapitalStructure("t", ents, pools, claims, admin_cost=0.05)


def test_conservation_and_monotonicity():
    cs = holdco_opco()
    assert cs.validate() == []
    V = np.linspace(0, 400, 801)
    res = distribute(cs, V)
    total = sum(res.payouts[c.id] for c in cs.claims)
    assert np.allclose(total, V * (1 - 0.05))
    for c in cs.claims:
        assert np.all(np.diff(res.payouts[c.id]) >= -1e-9), c.id


def test_absolute_priority_single_entity():
    cs = CapitalStructure("apr", [Entity("t", "T", None, 1.0)], [], [
        Claim("s", "Senior", "senior_unsecured", "t", 100.0, rank=10),
        Claim("j", "Sub", "subordinated", "t", 50.0, rank=20),
        Claim("e", "Common", "common", "t")])
    r = distribute(cs, [60.0, 120.0, 200.0])
    assert np.allclose(r.payouts["s"], [60, 100, 100])
    assert np.allclose(r.payouts["j"], [0, 20, 50])
    assert np.allclose(r.payouts["e"], [0, 0, 50])


def test_collateral_and_deficiency_claim():
    cs = CapitalStructure("sec", [Entity("t", "T", None, 1.0)], [CollateralPool("p", "t", 0.5)], [
        Claim("sec", "Secured", "term_loan", "t", 80.0, secured_by=(("p", 1),)),
        Claim("uns", "Unsecured", "senior_unsecured", "t", 60.0),
        Claim("e", "Common", "common", "t")])
    r = distribute(cs, 100.0)
    # pool 50 to secured; deficiency 30 shares the other 50 pro rata with 60 of unsecured
    assert r.payouts["sec"][0] == pytest.approx(50 + 50 * 30 / 90)
    assert r.payouts["uns"][0] == pytest.approx(50 * 60 / 90)


def test_structural_subordination_and_upstream_guarantee():
    plain = distribute(holdco_opco(False), 120.0)
    guaranteed = distribute(holdco_opco(True), 120.0)
    assert guaranteed.payouts["notes"][0] > plain.payouts["notes"][0]
    assert guaranteed.payouts["trade"][0] < plain.payouts["trade"][0]


def test_non_recourse_vs_recourse_spv_debt():
    cs = holdco_opco()
    low = 40.0  # SPV collateral worth 0.3*V*(1-ac) = 11.4 < 50 loan
    with_guar = distribute(cs, low).payouts["dd"][0]
    cs.claim("dd").guarantors = ()
    without = distribute(cs, low).payouts["dd"][0]
    assert with_guar > without
    assert without == pytest.approx(0.3 * low * 0.95)  # non-recourse lender keeps only its SPV


def test_turnover_to_senior_notes():
    base = [Claim("sn", "Senior notes", "senior_unsecured", "t", 100.0),
            Claim("tr", "Trade", "trade", "t", 100.0),
            Claim("sub", "Sub notes", "subordinated", "t", 100.0, rank=10, turnover_to=("sn",)),
            Claim("e", "Common", "common", "t")]
    cs = CapitalStructure("to", [Entity("t", "T", None, 1.0)], [], base)
    r = distribute(cs, 150.0)
    # pro rata 50 each; sub hands its 50 to senior notes (shortfall 50)
    assert r.payouts["tr"][0] == pytest.approx(50.0)
    assert r.payouts["sn"][0] == pytest.approx(100.0)
    assert r.payouts["sub"][0] == pytest.approx(0.0)


def test_gift_to_equity_only_when_impaired():
    cs = CapitalStructure("g", [Entity("t", "T", None, 1.0)], [], [
        Claim("d", "Debt", "senior_unsecured", "t", 100.0), Claim("e", "Common", "common", "t")])
    r = distribute(cs, [80.0, 300.0], gift_to_equity=0.05, admin_cost=0.0)
    assert r.payouts["e"][0] == pytest.approx(4.0)
    assert r.payouts["d"][0] == pytest.approx(76.0)
    assert r.payouts["e"][1] == pytest.approx(200.0)


def test_lease_rejection_cap():
    assert lease_rejection_claim(10.0, 10.0) == pytest.approx(15.0)
    assert lease_rejection_claim(10.0, 2.0) == pytest.approx(10.0)
    assert lease_rejection_claim(10.0, 30.0) == pytest.approx(30.0)
    assert lease_rejection_claim(10.0, 0.5) == pytest.approx(5.0)


def test_breakpoints_and_fulcrum():
    cs = CapitalStructure("b", [Entity("t", "T", None, 1.0)], [], [
        Claim("s", "Senior", "senior_unsecured", "t", 100.0),
        Claim("j", "Sub", "subordinated", "t", 50.0, rank=20),
        Claim("e", "Common", "common", "t")])
    bp = breakpoints(cs, admin_cost=0.0)
    assert np.allclose(bp, [100.0, 150.0], atol=1e-5)
    assert full_payment_threshold(cs, "j", admin_cost=0.0) == pytest.approx(150.0, abs=1e-4)
    assert fulcrum(cs, 120.0, admin_cost=0.0) == "j"
    assert fulcrum(cs, 80.0, admin_cost=0.0) == "s"
