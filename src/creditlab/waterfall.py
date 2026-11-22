"""Absolute-priority waterfall for a restructuring or liquidation (vectorised over enterprise values).

Given a total distributable value V (an array of scenarios is allowed), `distribute` returns each
claim's recovery:

1. Administrative costs consume `admin_cost` x V (in default scenarios only).
2. Value is split across legal entities by their `value_share`.
3. Entities are processed children first. At each entity:
   a. each collateral pool pays its secured claims by lien priority (pro rata within a lien);
   b. the unencumbered value plus pool surpluses plus subsidiaries' residual equity is paid out by
      unsecured rank: priority claims, then senior unsecured claims (including deficiency claims of
      recourse secured debt and guarantee claims against this entity), subordinated, preferred;
   c. turnover: recoveries of contractually subordinated claims are handed to their senior
      beneficiaries until those are paid in full;
   d. the residual flows to the parent; at the top entity it goes to common equity.
4. An optional gift (a deviation from absolute priority) carves `gift_to_equity` x V out for old
   equity whenever creditors are impaired - the pattern of many negotiated plans.

Simplifications (see docs/ASSUMPTIONS_AND_LIMITS.md): a claim allowed against several obligors is
allowed in full at each but is capped at what is still unpaid when that obligor is processed (children
first); post-petition interest, plan negotiations, substantive consolidation, avoidance actions,
valuation fights and timing are not modelled; value shares are fixed fractions of V.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .capital_structure import RANK_COMMON, CapitalStructure


@dataclass
class WaterfallResult:
    V: np.ndarray
    payouts: dict          # claim id -> array of amounts recovered
    recovery: dict         # claim id -> array of recovery fractions (nan for common)
    residual: np.ndarray   # value to common equity
    entity_detail: dict    # per entity: own value, pledged value, residual passed up
    claim_ids: list

    def table(self, i: int = 0) -> list[dict]:
        out = []
        for cid in self.claim_ids:
            out.append({"claim": cid, "payout": float(self.payouts[cid][i]),
                        "recovery": float(self.recovery[cid][i])})
        return out


def _pro_rata(avail: np.ndarray, demands: list[np.ndarray]) -> list[np.ndarray]:
    """Split avail among claims pro rata to their (array) demands, never paying more than demanded."""
    if not demands:
        return []
    total = np.sum(demands, axis=0)
    pay_total = np.minimum(avail, total)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(total > 0, pay_total / total, 0.0)
    return [d * frac for d in demands]


def distribute(cs: CapitalStructure, V, admin_cost: float | None = None, gift_to_equity: float = 0.0,
               in_default=True) -> WaterfallResult:
    """Distribute value V (scalar or array) through the capital structure.

    in_default: bool or boolean array. Where False (a solvent firm), no admin cost and no gift apply.
    """
    V = np.atleast_1d(np.asarray(V, dtype=float))
    n = V.shape[0]
    in_def = np.broadcast_to(np.asarray(in_default, dtype=bool), (n,))
    ac = cs.admin_cost if admin_cost is None else admin_cost
    net = np.where(in_def, V * (1.0 - ac), V)
    gift = np.where(in_def, gift_to_equity * net, 0.0)
    creditor_value = net - gift

    claims = cs.claims
    cid = [c.id for c in claims]
    outstanding = {c.id: np.full(n, float(c.amount)) for c in claims if c.kind != "common"}
    paid = {c.id: np.zeros(n) for c in claims}
    residual_up = {e.id: np.zeros(n) for e in cs.entities}
    detail = {}
    top = cs.top.id
    equity_residual = np.zeros(n)

    for e in cs.children_first():
        own = creditor_value * e.value_share
        unenc = own.copy()
        pledged_total = np.zeros(n)
        for p in [p for p in cs.pools if p.entity == e.id]:
            pv = own * p.share
            unenc -= pv
            pledged_total += pv
            secured = [(c, lien) for c in claims for (pid, lien) in c.secured_by if pid == p.id]
            for lien in sorted({l for _, l in secured}):
                group = [c for c, l in secured if l == lien]
                pays = _pro_rata(pv, [outstanding[c.id] for c in group])
                for c, pay in zip(group, pays):
                    outstanding[c.id] = outstanding[c.id] - pay
                    paid[c.id] = paid[c.id] + pay
                    pv = pv - pay
            unenc += pv  # pool surplus returns to the estate
        avail = unenc + residual_up[e.id]

        # unsecured tiers at this entity: (claim, rank)
        tier_members: dict[int, list] = {}
        for c in claims:
            if c.kind == "common":
                continue
            if c.entity == e.id:
                tier_members.setdefault(c.rank, []).append(c)
            elif e.id in c.guarantors:
                tier_members.setdefault(c.guarantee_rank, []).append(c)
        for rank in sorted(tier_members):
            if rank >= RANK_COMMON:
                continue
            group = tier_members[rank]
            demands = [outstanding[c.id].copy() for c in group]
            pays = _pro_rata(avail, demands)
            alloc = {c.id: pay for c, pay in zip(group, pays)}
            # turnover: subordinated members hand recoveries to beneficiaries' remaining shortfall
            for c in group:
                if not c.turnover_to:
                    continue
                for b in c.turnover_to:
                    if b not in outstanding:
                        continue
                    already = alloc.get(b, np.zeros(n))
                    shortfall = np.maximum(outstanding[b] - already, 0.0)
                    move = np.minimum(alloc[c.id], shortfall)
                    alloc[c.id] = alloc[c.id] - move
                    alloc[b] = already + move
            for k, pay in alloc.items():
                outstanding[k] = outstanding[k] - pay
                paid[k] = paid[k] + pay
                avail = avail - pay
        avail = np.maximum(avail, 0.0)
        detail[e.id] = {"own_value": own, "pledged": pledged_total, "residual": avail.copy()}
        if e.parent is None:
            equity_residual = avail
        else:
            residual_up[e.parent] = residual_up[e.parent] + avail

    common = [c for c in claims if c.kind == "common"]
    to_equity = equity_residual + gift
    if common:
        for c in common:
            paid[c.id] = to_equity / len(common)
    recovery = {}
    for c in claims:
        if c.kind == "common" or c.amount <= 0:
            recovery[c.id] = np.full(n, np.nan)
        else:
            recovery[c.id] = paid[c.id] / c.amount
    return WaterfallResult(V, paid, recovery, to_equity, detail, cid)


def payout_matrix(cs: CapitalStructure, V, **kw) -> np.ndarray:
    """Array (len(V), n_claims) of payouts, columns in cs.claims order."""
    res = distribute(cs, V, **kw)
    return np.column_stack([res.payouts[c.id] for c in cs.claims])


def full_payment_threshold(cs: CapitalStructure, claim_id: str, hi: float | None = None,
                           tol: float = 1e-7, **kw) -> float:
    """Smallest V at which `claim_id` is paid in full (bisection; payouts are monotone in V)."""
    c = cs.claim(claim_id)
    if c.amount <= 0:
        return 0.0
    hi = hi or max(cs.total_claims() * 4.0, 1.0)
    f = lambda v: distribute(cs, v, **kw).payouts[claim_id][0] - c.amount * (1 - 1e-12)
    while f(hi) < 0:
        hi *= 2.0
        if hi > 1e15:
            return np.inf
    lo = 0.0
    while hi - lo > tol * max(1.0, hi):
        mid = 0.5 * (lo + hi)
        if f(mid) >= 0:
            hi = mid
        else:
            lo = mid
    return hi


def breakpoints(cs: CapitalStructure, **kw) -> np.ndarray:
    """Values of V at which some claim becomes paid in full (kinks of the payout functions)."""
    pts = [full_payment_threshold(cs, c.id, **kw) for c in cs.claims if c.kind != "common" and c.amount > 0]
    pts = [p for p in pts if np.isfinite(p) and p > 0]
    return np.unique(np.round(pts, 9))


def fulcrum(cs: CapitalStructure, V: float, **kw) -> str | None:
    """The most senior claim (by recovery ordering) that is impaired but receives value at V."""
    res = distribute(cs, V, **kw)
    cands = [(float(res.recovery[c.id][0]), c) for c in cs.claims
             if c.kind != "common" and c.amount > 0 and 0 < res.recovery[c.id][0] < 1 - 1e-9]
    if not cands:
        return None
    return max(cands, key=lambda t: t[0])[1].id
