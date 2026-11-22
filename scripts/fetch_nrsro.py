"""Find and download S&P Global Ratings' latest Form NRSRO Exhibit 1 (performance statistics) from SEC EDGAR.

Usage: python scripts/fetch_nrsro.py [--list]
Requires the SEC_USER_AGENT environment variable (SEC fair-access policy); it is read at run time only.
The PDF goes to data/raw/nrsro/ (git-ignored); scripts/parse_nrsro.py extracts the corporate tables.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CIK = "0001650548"   # S&P Global Ratings (Form NRSRO filer)


def get(url):
    r = requests.get(url, headers={"User-Agent": os.environ["SEC_USER_AGENT"]}, timeout=60)
    r.raise_for_status()
    time.sleep(0.3)
    return r


def main(list_only=False):
    sub = get(f"https://data.sec.gov/submissions/CIK{CIK}.json").json()
    rec = sub["filings"]["recent"]
    rows = [dict(zip(rec.keys(), vals)) for vals in zip(*rec.values())]
    nrsro = [r for r in rows if r["form"].startswith("NRSRO")]
    for r in nrsro[:8]:
        print(r["form"], r["filingDate"], r["accessionNumber"], r["primaryDocument"])
    if list_only or not nrsro:
        return
    latest = next(r for r in nrsro if r["form"] == "NRSRO-CE")   # annual certification carries Exhibit 1
    acc = latest["accessionNumber"].replace("-", "")
    idx = get(f"https://www.sec.gov/Archives/edgar/data/{int(CIK)}/{acc}/index.json").json()
    out = ROOT / "data" / "raw" / "nrsro" / latest["accessionNumber"]
    out.mkdir(parents=True, exist_ok=True)
    names = [i["name"] for i in idx["directory"]["item"]]
    print("files:", names)
    for n in names:
        if n.lower().endswith(".pdf") and ("ex1" in n.lower().replace("_", "").replace(" ", "") or "exhibit1" in n.lower().replace("_", "")):
            pdf = get(f"https://www.sec.gov/Archives/edgar/data/{int(CIK)}/{acc}/{n}").content
            (out / n).write_bytes(pdf)
            print("saved", out / n, len(pdf))
    (out / "filing.json").write_text(json.dumps(latest, indent=1))


if __name__ == "__main__":
    main("--list" in sys.argv)
