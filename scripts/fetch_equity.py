"""Daily closes for case-study equities, preferreds and bitcoin; derived volatility statistics.

Usage: python scripts/fetch_equity.py CRWV NBIS IREN MSTR WOLF STRF STRC STRK STRD BTC-USD
Raw chart JSON (Yahoo Finance chart endpoint) is cached in data/raw/yahoo/<date>/ (git-ignored).
Writes data/snapshots/equity_stats_<date>.json with last close and realised volatility only
(no price histories are redistributed).
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
UA = {"User-Agent": "Mozilla/5.0 (research script; credit-capital-structure-lab)"}


def chart(sym: str, out_dir: Path) -> dict:
    f = out_dir / f"{sym}.json"
    if f.exists():
        return json.loads(f.read_text())
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range=2y&interval=1d",
                     headers=UA, timeout=60)
    r.raise_for_status()
    f.write_text(r.text)
    time.sleep(1.0)
    return r.json()


def stats(j: dict) -> dict:
    res = j["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    close = np.array([np.nan if c is None else c for c in q["close"]], dtype=float)
    days = [dt.datetime.fromtimestamp(t, dt.UTC).date().isoformat() for t in ts]
    ok = np.isfinite(close)
    close = close[ok]
    days = [d for d, k in zip(days, ok) if k]
    lr = np.diff(np.log(close))
    meta = res["meta"]
    crypto = meta.get("instrumentType") == "CRYPTOCURRENCY"
    ann = 365 if crypto else 252

    def vol(n):
        x = lr[-n:]
        return float(np.std(x, ddof=1) * np.sqrt(ann)) if len(x) > 20 else None

    def ewma(lam=0.94):
        v = np.var(lr[:30])
        for x in lr[30:]:
            v = lam * v + (1 - lam) * x * x
        return float(np.sqrt(v * ann))

    return {"symbol": meta.get("symbol"), "currency": meta.get("currency"), "exchange": meta.get("exchangeName"),
            "last_close": float(close[-1]), "last_date": days[-1], "first_date": days[0], "n_obs": int(len(close)),
            "vol_3m": vol(63), "vol_6m": vol(126), "vol_1y": vol(252), "vol_2y": vol(504), "vol_ewma94": ewma(),
            "ann_factor": ann, "high_52w": float(np.max(close[-252:])), "low_52w": float(np.min(close[-252:])),
            "return_1m": float(close[-1] / close[-22] - 1) if len(close) > 22 else None,
            "source": "Yahoo Finance chart API (daily close), realised vol from log returns"}


def main(symbols):
    today = dt.date.today().isoformat()
    out_dir = ROOT / "data" / "raw" / "yahoo" / today
    out_dir.mkdir(parents=True, exist_ok=True)
    allstats = {}
    for s in symbols:
        try:
            allstats[s] = stats(chart(s, out_dir))
            st = allstats[s]
            print(f"{s:8} {st['last_date']} close {st['last_close']:.4f}  vol1y {st['vol_1y']}  vol3m {st['vol_3m']}")
        except Exception as exc:  # report and continue
            print(f"{s}: FAILED {exc}")
    snap = ROOT / "data" / "snapshots" / f"equity_stats_{today}.json"
    snap.write_text(json.dumps(allstats, indent=1))
    print("wrote", snap)


if __name__ == "__main__":
    main(sys.argv[1:])
