"""FINRA TRACE last-sale snapshot: query helper and parser.

FINRA's public Corporate and Agency Bonds grid (https://www.finra.org/finra-data/fixed-income/corp-and-agency)
is served by a JSON endpoint that answers only requests made from the finra.org page (XSRF cookie + an
explicit field list; plain scripted requests get 404). The snapshot used by the case studies was taken by
running `scripts/trace_query.js` in the browser console on that page and saving the result as
data/raw/trace/<date>/finra_last_trades.json (git-ignored; FINRA data terms apply).

This module turns that file into a lookup by CUSIP for the case builders.
Usage: python scripts/fetch_trace.py data/raw/trace/2026-11-21/finra_last_trades.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load_last_trades(path: str | Path) -> dict:
    raw = json.loads(Path(path).read_text())
    cols = raw["columns"]
    out = {}
    for issuer, rows in raw["data"].items():
        for r in rows:
            rec = dict(zip(cols, r))
            rec["issuer_query"] = issuer
            out[rec["cusip"]] = rec
    return out


if __name__ == "__main__":
    trades = load_last_trades(sys.argv[1])
    for cusip, r in sorted(trades.items(), key=lambda kv: (kv[1]["issuerName"], kv[1]["maturity"])):
        print(f"{r['issuerName']:<20} {cusip}  {r['coupon']:>7.3f}% {r['maturity']}  last {r['lastSalePrice']}  on {r['lastTradeDate']}")
