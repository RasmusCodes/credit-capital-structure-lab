"""Case-study files: one JSON per company (cases/<id>/case.json) -> engine objects.

A case file is the documented record of a study: every claim carries its filing citation, every market
price its source and date, every modelling judgment an entry in `assumptions` with its basis. The loader
turns it into a CapitalStructure (claims at their allowed amounts in default), FixedBond / Convertible
objects with market quotes, and the equity inputs for structural calibration.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .bonds import FixedBond
from .capital_structure import CapitalStructure, Claim, CollateralPool, Entity
from .convertible import Convertible
from .curves import TREASURY_TENORS, ZeroCurve
from .daycount import to_date, year_frac
from .relative_value import BondQuote

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Case:
    id: str
    data: dict
    structure: CapitalStructure
    settle: object
    bonds: list = field(default_factory=list)        # BondQuote
    converts: list = field(default_factory=list)     # dicts: claim_id, cb, price, trade_date, source
    preferreds: list = field(default_factory=list)   # dicts
    zc: ZeroCurve | None = None

    @property
    def equity(self) -> dict:
        return self.data["equity"]

    @property
    def market_cap(self) -> float:
        e = self.data["equity"]
        return e["price"] * e["shares_m"]

    def assumption(self, key, default=None):
        for a in self.data.get("assumptions", []):
            if a["id"] == key:
                return a["value"]
        return default


def treasury_curve(date: str) -> ZeroCurve:
    year = date[:4]
    path = ROOT / "data" / "snapshots" / f"ust_par_curve_{year}.csv"
    rows = list(csv.DictReader(open(path)))
    rows = [r for r in rows if r["Date"] <= date]
    row = rows[-1]
    return ZeroCurve.from_treasury_par({TREASURY_TENORS[k]: float(v) for k, v in row.items()
                                        if k in TREASURY_TENORS and v}, label=f"UST {row['Date']}")


def _claim(c: dict, settle) -> Claim:
    mat = c.get("maturity")
    T = None
    if mat:
        T = max(year_frac(to_date(settle), to_date(mat)), 0.01)
    return Claim(id=c["id"], name=c["name"], kind=c["kind"], entity=c["entity"], amount=float(c.get("amount", 0.0)),
                 rank=int(c.get("rank", 10)), secured_by=tuple(tuple(x) for x in c.get("secured_by", [])),
                 guarantors=tuple(c.get("guarantors", [])), guarantee_rank=int(c.get("guarantee_rank", 10)),
                 turnover_to=tuple(c.get("turnover_to", [])), maturity=T, coupon=c.get("coupon"),
                 market=c.get("market", {}), source=c.get("source", {}), notes=c.get("notes", ""))


def load_case(case_id: str | Path) -> Case:
    path = Path(case_id)
    if not path.suffix:
        path = ROOT / "cases" / str(case_id) / "case.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    settle = to_date(d["settlement_date"])
    cs = CapitalStructure(
        name=d["company"],
        entities=[Entity(e["id"], e["name"], e.get("parent"), float(e["value_share"]), e.get("notes", ""))
                  for e in d["entities"]],
        pools=[CollateralPool(p["id"], p["entity"], float(p["share"]), p.get("description", "")) for p in d.get("pools", [])],
        claims=[_claim(c, settle) for c in d["claims"]],
        admin_cost=float(d.get("admin_cost", 0.0)),
        as_of=d.get("balance_sheet_date", ""),
        currency=d.get("currency", "USD"),
        shares_outstanding=d["equity"].get("shares_m"),
        notes=d.get("summary", ""),
        sources=d.get("sources", []),
    )
    problems = cs.validate()
    if problems:
        raise ValueError(f"{path}: {problems}")
    case = Case(d["id"], d, cs, settle, zc=treasury_curve(d["market_date"]))
    for c in d["claims"]:
        inst = c.get("instrument", {})
        mkt = c.get("market", {})
        if inst.get("type") == "bond" and mkt.get("clean") is not None:
            b = FixedBond(c["name"], float(c["coupon"]), to_date(c["maturity"]), int(inst.get("freq", 2)),
                          call_schedule=tuple(inst.get("call_schedule", [])), issuer=d["company"],
                          seniority=c["kind"])
            case.bonds.append(BondQuote(c["id"], b, float(mkt["clean"]), to_date(mkt["date"]), mkt.get("source", ""),
                                        float(c.get("recovery_for_pricing", inst.get("recovery", 0.4))),
                                        [a["clean"] for a in mkt.get("alt", [])], c["entity"], c["kind"],
                                        mkt.get("stock_move_since_trade")))
        elif inst.get("type") == "convertible":
            T = year_frac(settle, to_date(c["maturity"]))
            puts = [(year_frac(settle, to_date(p)), 100.0) for p in inst.get("put_dates", []) if to_date(p) > settle]
            call_start = None
            if inst.get("first_call"):
                call_start = max(year_frac(settle, to_date(inst["first_call"])), 0.0)
            cb = Convertible(c["name"], float(c["coupon"]), T, float(inst["conversion_price"]), puts=puts,
                             call_start=call_start, recovery=float(c.get("recovery_for_pricing", 0.4)))
            case.converts.append({"claim_id": c["id"], "cb": cb, "price": mkt.get("clean"), "trade_date": mkt.get("date"),
                                  "source": mkt.get("source", ""), "cusip": mkt.get("cusip"),
                                  "stock_move_since_trade": mkt.get("stock_move_since_trade")})
        elif c["kind"] == "preferred" and mkt.get("price") is not None:
            case.preferreds.append({"claim_id": c["id"], "ticker": inst.get("ticker"), "price": mkt["price"],
                                    "date": mkt.get("date"), "dividend_rate": inst.get("dividend_rate"),
                                    "stated": inst.get("stated_amount", 100.0), "liq_pref": inst.get("liquidation_pref"),
                                    "cumulative": inst.get("cumulative", True), "shares_m": inst.get("shares_m")})
    return case


def list_cases() -> list[str]:
    return sorted(p.parent.name for p in (ROOT / "cases").glob("*/case.json"))
