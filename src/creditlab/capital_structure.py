"""Capital-structure data model: legal entities, collateral pools and claims.

The model is deliberately explicit about the things that decide recoveries in practice:

* which legal entity owes a claim (structural subordination: a parent's creditors only reach a
  subsidiary's value after that subsidiary's own creditors are paid);
* which assets secure it, and with which lien priority;
* who guarantees it (upstream or downstream guarantees give a claim against each guarantor);
* its rank among unsecured claims at each obligor (senior, subordinated, preferred, common);
* contractual subordination with turnover (subordinated notes share pro rata with trade creditors but
  must hand their recoveries to the designated senior debt until it is paid in full).

Amounts are in currency millions. A claim's `amount` is the allowed claim at default (principal plus
accrued interest if the user includes it; liquidation preference for preferred stock). Common equity
carries no amount: it is the residual.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

# default unsecured ranks (lower = more senior)
RANK_PRIORITY = 0        # administrative, DIP, priority tax and wage claims
RANK_SENIOR = 10         # senior unsecured, deficiency claims, trade, lease rejection, convertibles
RANK_SUBORDINATED = 20
RANK_PREFERRED = 30
RANK_COMMON = 40

DEBT_KINDS = {"admin", "revolver", "term_loan", "delayed_draw_term_loan", "secured_notes", "second_lien_notes",
              "senior_unsecured", "convertible", "equipment_financing", "lease", "trade", "subordinated", "other_debt"}
EQUITY_KINDS = {"preferred", "common"}


@dataclass
class Entity:
    id: str
    name: str
    parent: str | None = None     # owner of this entity's residual value (None = top company)
    value_share: float = 0.0      # share of total enterprise value produced by this entity's own assets
    notes: str = ""


@dataclass
class CollateralPool:
    id: str
    entity: str
    share: float                  # fraction of the entity's own value pledged in this pool
    description: str = ""


@dataclass
class Claim:
    id: str
    name: str
    kind: str
    entity: str                   # obligor
    amount: float = 0.0           # allowed claim at default (0 for common equity)
    rank: int = RANK_SENIOR       # unsecured rank at the obligor (and of any deficiency claim)
    secured_by: tuple = ()        # ((pool_id, lien_rank), ...); lien 1 = first lien
    guarantors: tuple = ()        # entity ids guaranteeing the claim
    guarantee_rank: int = RANK_SENIOR
    turnover_to: tuple = ()       # claim ids that receive this claim's recoveries until paid in full
    maturity: float | None = None  # years from the valuation date
    coupon: float | None = None   # decimal
    market: dict = field(default_factory=dict)   # price, date, source, identifiers
    source: dict = field(default_factory=dict)   # filing citation for the amount and terms
    notes: str = ""

    @property
    def is_debt(self) -> bool:
        return self.kind in DEBT_KINDS

    @property
    def is_secured(self) -> bool:
        return bool(self.secured_by)


@dataclass
class CapitalStructure:
    name: str
    entities: list[Entity]
    pools: list[CollateralPool] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    admin_cost: float = 0.0       # fraction of value lost to administration / professional fees in a default
    as_of: str = ""
    currency: str = "USD"
    shares_outstanding: float | None = None   # common shares, millions
    notes: str = ""
    sources: list = field(default_factory=list)

    # ------------------------------------------------------------------ lookups
    def entity(self, eid: str) -> Entity:
        for e in self.entities:
            if e.id == eid:
                return e
        raise KeyError(eid)

    def claim(self, cid: str) -> Claim:
        for c in self.claims:
            if c.id == cid:
                return c
        raise KeyError(cid)

    def pool(self, pid: str) -> CollateralPool:
        for p in self.pools:
            if p.id == pid:
                return p
        raise KeyError(pid)

    @property
    def top(self) -> Entity:
        tops = [e for e in self.entities if e.parent is None]
        if len(tops) != 1:
            raise ValueError("exactly one top entity required")
        return tops[0]

    def children_first(self) -> list[Entity]:
        """Entities ordered so that every subsidiary comes before its parent."""
        depth = {}

        def d(e):
            if e.id in depth:
                return depth[e.id]
            depth[e.id] = 0 if e.parent is None else d(self.entity(e.parent)) + 1
            return depth[e.id]

        for e in self.entities:
            d(e)
        return sorted(self.entities, key=lambda e: -depth[e.id])

    # ------------------------------------------------------------------ summaries
    def debt(self) -> list[Claim]:
        return [c for c in self.claims if c.is_debt]

    def total_debt(self) -> float:
        return float(sum(c.amount for c in self.debt()))

    def total_claims(self) -> float:
        return float(sum(c.amount for c in self.claims if c.kind != "common"))

    def maturity_profile(self, buckets=(1, 2, 3, 5, 7, 10, 30)) -> list[dict]:
        out = []
        lo = 0.0
        for hi in buckets:
            amt = sum(c.amount for c in self.debt() if c.maturity is not None and lo < c.maturity <= hi)
            out.append({"from": lo, "to": hi, "amount": amt})
            lo = hi
        undated = sum(c.amount for c in self.debt() if c.maturity is None)
        out.append({"from": None, "to": None, "amount": undated, "label": "no stated maturity"})
        return out

    def default_point(self, horizon: float, long_term_weight: float = 0.5) -> float:
        """KMV-style default point: debt due within the horizon + a fraction of longer-dated debt."""
        st = sum(c.amount for c in self.debt() if c.maturity is not None and c.maturity <= horizon)
        lt = sum(c.amount for c in self.debt() if c.maturity is None or c.maturity > horizon)
        return float(st + long_term_weight * lt)

    def validate(self) -> list[str]:
        problems = []
        ids = [e.id for e in self.entities]
        if len(set(ids)) != len(ids):
            problems.append("duplicate entity ids")
        share = sum(e.value_share for e in self.entities)
        if abs(share - 1.0) > 1e-6:
            problems.append(f"entity value shares sum to {share:.4f}, not 1")
        for e in self.entities:
            ps = sum(p.share for p in self.pools if p.entity == e.id)
            if ps > 1 + 1e-9:
                problems.append(f"pools of {e.id} pledge {ps:.2%} of its value")
            if e.parent is not None and e.parent not in ids:
                problems.append(f"{e.id}: unknown parent {e.parent}")
        pool_ids = {p.id for p in self.pools}
        claim_ids = {c.id for c in self.claims}
        for c in self.claims:
            if c.entity not in ids:
                problems.append(f"{c.id}: unknown obligor {c.entity}")
            for pid, _ in c.secured_by:
                if pid not in pool_ids:
                    problems.append(f"{c.id}: unknown pool {pid}")
            for g in c.guarantors:
                if g not in ids:
                    problems.append(f"{c.id}: unknown guarantor {g}")
            for t in c.turnover_to:
                if t not in claim_ids:
                    problems.append(f"{c.id}: unknown turnover beneficiary {t}")
            if c.amount < 0:
                problems.append(f"{c.id}: negative amount")
        try:
            self.top
        except ValueError as exc:
            problems.append(str(exc))
        return problems

    # ------------------------------------------------------------------ transformations (demo sliders)
    def scaled_debt(self, k: float) -> "CapitalStructure":
        cs = copy.deepcopy(self)
        for c in cs.claims:
            if c.is_debt and c.kind != "admin":
                c.amount *= k
        return cs

    def shifted_maturities(self, years: float) -> "CapitalStructure":
        cs = copy.deepcopy(self)
        for c in cs.claims:
            if c.maturity is not None:
                c.maturity = max(c.maturity + years, 0.01)
        return cs


def substantively_consolidated(cs: CapitalStructure) -> CapitalStructure:
    """All debtors pooled into one estate (a court remedy, or what an unwound SPV structure looks like):
    entity boundaries and guarantees disappear, liens on specific collateral survive."""
    out = copy.deepcopy(cs)
    share = {e.id: e.value_share for e in cs.entities}
    out.entities = [Entity("consolidated", f"{cs.name} (substantively consolidated)", None, 1.0)]
    for p in out.pools:
        p.share = p.share * share[p.entity]
        p.entity = "consolidated"
    for c in out.claims:
        c.entity = "consolidated"
        c.guarantors = ()
    return out


def with_overrides(cs: CapitalStructure, amount_scale: dict | None = None, drop: tuple = (),
                   non_recourse: tuple = ()) -> CapitalStructure:
    """Scenario helper: scale claim amounts by kind or id, drop claims, or remove guarantees."""
    out = copy.deepcopy(cs)
    amount_scale = amount_scale or {}
    out.claims = [c for c in out.claims if c.id not in drop]
    for c in out.claims:
        k = amount_scale.get(c.id, amount_scale.get(c.kind))
        if k is not None:
            c.amount *= k
        if c.id in non_recourse:
            c.guarantors = ()
    return out


def lease_rejection_claim(annual_rent: float, remaining_years: float, unpaid_prepetition: float = 0.0,
                          rent_based: bool = True) -> float:
    """Landlord claim for a rejected real-property lease under 11 U.S.C. 502(b)(6).

    Cap = rent for the greater of one year or 15% (not exceeding three years) of the remaining term, plus
    unpaid pre-petition rent. Courts split on whether "15%" refers to remaining rent (rent_based=True,
    the larger-claim reading when rents escalate) or remaining time; with level rent they coincide.
    Equipment and finance leases are not capped this way and are better modelled as secured financing.
    """
    remaining_rent = annual_rent * remaining_years
    fifteen = 0.15 * remaining_rent if rent_based else annual_rent * 0.15 * remaining_years
    cap = max(annual_rent, min(fifteen, 3.0 * annual_rent))
    return min(cap, remaining_rent) + unpaid_prepetition


def maturities_array(cs: CapitalStructure) -> np.ndarray:
    return np.array([np.nan if c.maturity is None else c.maturity for c in cs.claims])
