"""Download the US Treasury daily par yield curve for one or more years (public domain data).

Usage: python scripts/fetch_treasury.py 2025 2026
Writes data/snapshots/ust_par_curve_<year>.csv (columns as published, newest first -> sorted ascending).
"""
from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
URL = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/"
       "{year}/all?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv")


def fetch(year: int) -> Path:
    r = requests.get(URL.format(year=year), timeout=60, headers={"User-Agent": "credit-capital-structure-lab/0.1"})
    r.raise_for_status()
    rows = list(csv.DictReader(io.StringIO(r.text)))
    if not rows:
        raise RuntimeError("empty response")
    for row in rows:  # MM/DD/YYYY -> ISO
        m, d, y = row["Date"].split("/")
        row["Date"] = f"{y}-{m}-{d}"
    rows.sort(key=lambda x: x["Date"])
    out = ROOT / "data" / "snapshots" / f"ust_par_curve_{year}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{out.name}: {len(rows)} days, {rows[0]['Date']} .. {rows[-1]['Date']}")
    return out


if __name__ == "__main__":
    for y in (sys.argv[1:] or ["2026"]):
        fetch(int(y))
