"""Listed equity options: implied volatility term structure and deep put quotes for case-study names.

Usage: python scripts/fetch_options.py CRWV NBIS IREN MSTR WOLF
Source: Yahoo Finance options endpoint (cookie + crumb handshake). Raw JSON cached in
data/raw/options/<date>/ (git-ignored), with summary.json holding, per expiry: the ATM call/put quotes
from two-sided markets, and the lowest-strike puts with bids (for the put-implied default bounds).
build_cases.py turns these into implied volatilities stored in the case files. Quotes taken outside market hours can be one-sided or stale; contracts
without a positive bid and ask, or not traded on the last session, are dropped and counted.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def session():
    s = requests.Session()
    s.headers.update(UA)
    s.get("https://fc.yahoo.com", timeout=30)
    crumb = s.get("https://query2.finance.yahoo.com/v1/test/getcrumb", timeout=30).text.strip()
    return s, crumb


def get(s, crumb, sym, date=None):
    url = f"https://query2.finance.yahoo.com/v7/finance/options/{sym}?crumb={crumb}"
    if date:
        url += f"&date={date}"
    r = s.get(url, timeout=60)
    r.raise_for_status()
    time.sleep(0.7)
    return r.json()["optionChain"]["result"][0]


def summarize(sym, chains, spot, last_session):
    out = []
    for ch in chains:
        opt = ch["options"][0]
        exp = dt.datetime.fromtimestamp(opt["expirationDate"], dt.UTC).date()
        T = max((exp - dt.date.fromisoformat(last_session)).days, 1) / 365.0
        calls, puts = opt.get("calls", []), opt.get("puts", [])

        def good(c):
            return (c.get("bid") or 0) > 0 and (c.get("ask") or 0) > 0 and c["ask"] >= c["bid"]

        gc = [c for c in calls if good(c)]
        gp = [p for p in puts if good(p)]
        atm = None
        if gc and gp:
            ks = sorted({c["strike"] for c in gc} & {p["strike"] for p in gp}, key=lambda k: abs(k - spot))
            if ks:
                k = ks[0]
                c = next(x for x in gc if x["strike"] == k)
                p = next(x for x in gp if x["strike"] == k)
                atm = {"strike": k, "call_mid": 0.5 * (c["bid"] + c["ask"]), "put_mid": 0.5 * (p["bid"] + p["ask"]),
                       "call_iv_yahoo": c.get("impliedVolatility"), "put_iv_yahoo": p.get("impliedVolatility")}
        deep = sorted([p for p in puts if (p.get("bid") or 0) > 0], key=lambda p: p["strike"])[:4]
        out.append({"expiry": exp.isoformat(), "T": T, "n_calls": len(calls), "n_puts": len(puts),
                    "two_sided_calls": len(gc), "two_sided_puts": len(gp), "atm": atm,
                    "deep_puts": [{"strike": p["strike"], "bid": p.get("bid"), "ask": p.get("ask"),
                                   "last": p.get("lastPrice"), "oi": p.get("openInterest"),
                                   "last_trade": dt.datetime.fromtimestamp(p["lastTradeDate"], dt.UTC).date().isoformat()
                                   if p.get("lastTradeDate") else None} for p in deep]})
    return out


def main(symbols):
    today = dt.date.today().isoformat()
    raw_dir = ROOT / "data" / "raw" / "options" / today
    raw_dir.mkdir(parents=True, exist_ok=True)
    s, crumb = session()
    summary = {}
    for sym in symbols:
        first = get(s, crumb, sym)
        spot = first["quote"]["regularMarketPrice"]
        last_session = dt.datetime.fromtimestamp(first["quote"]["regularMarketTime"], dt.UTC).date().isoformat()
        chains = [first]
        for d in first["expirationDates"][1:]:
            chains.append(get(s, crumb, sym, d))
        (raw_dir / f"{sym}.json").write_text(json.dumps(chains))
        summary[sym] = {"spot": spot, "session": last_session, "expiries": summarize(sym, chains, spot, last_session),
                        "source": "Yahoo Finance options (bid/ask as of the last session)"}
        n2 = sum(e["two_sided_calls"] + e["two_sided_puts"] for e in summary[sym]["expiries"])
        print(f"{sym}: spot {spot} session {last_session}, {len(chains)} expiries, {n2} two-sided quotes")
    snap = raw_dir / "summary.json"   # quotes stay in the git-ignored raw folder; case files keep derived IVs
    snap.write_text(json.dumps(summary, indent=1))
    print("wrote", snap)


if __name__ == "__main__":
    main(sys.argv[1:])
