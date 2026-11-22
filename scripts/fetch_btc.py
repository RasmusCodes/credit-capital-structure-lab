"""Bitcoin price at the US equity close (21:00 UTC in winter) from Coinbase Exchange public hourly candles.

Usage: python scripts/fetch_btc.py 2026-11-20
Writes data/snapshots/btc_usd_<date>.json: the close of the 20:00-21:00 UTC candle, plus the day's
hourly range, so the Strategy case values bitcoin at the same moment as MSTR's closing price.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def main(day: str):
    d = dt.date.fromisoformat(day)
    start = dt.datetime(d.year, d.month, d.day, 0, 0, tzinfo=dt.UTC)
    end = start + dt.timedelta(hours=24)
    r = requests.get("https://api.exchange.coinbase.com/products/BTC-USD/candles",
                     params={"granularity": 3600, "start": start.isoformat(), "end": end.isoformat()},
                     headers={"User-Agent": "credit-capital-structure-lab/0.1"}, timeout=60)
    r.raise_for_status()
    rows = sorted(r.json())  # [time, low, high, open, close, volume]
    close_hour = start + dt.timedelta(hours=20)  # candle 20:00-21:00 UTC closes at the 4pm ET equity close (EST)
    c = [x for x in rows if x[0] == int(close_hour.timestamp())]
    if not c:
        raise RuntimeError("candle missing")
    out = {"date": day, "time_utc": (close_hour + dt.timedelta(hours=1)).isoformat(), "price": c[0][4],
           "day_low": min(x[1] for x in rows), "day_high": max(x[2] for x in rows),
           "source": "Coinbase Exchange BTC-USD hourly candle (close of 20:00-21:00 UTC)"}
    p = ROOT / "data" / "snapshots" / f"btc_usd_{day}.json"
    p.write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    main(sys.argv[1])
