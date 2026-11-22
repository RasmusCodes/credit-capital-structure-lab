"""Run the full analysis on every case (or the ones named) and save reports/cases/<id>/analysis.json.

Usage: python scripts/run_cases.py [case_id ...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from creditlab.analysis import run_case
from creditlab.case import list_cases, load_case

ROOT = Path(__file__).resolve().parents[1]


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        x = float(o)
        return x if np.isfinite(x) else None
    return o


def main(ids):
    for cid in ids or list_cases():
        t0 = time.time()
        res = run_case(load_case(cid))
        out = ROOT / "reports" / "cases" / cid
        out.mkdir(parents=True, exist_ok=True)
        (out / "analysis.json").write_text(json.dumps(clean(res), indent=1), encoding="utf-8")
        print(f"{cid}: {time.time() - t0:.1f}s -> reports/cases/{cid}/analysis.json")


if __name__ == "__main__":
    main(sys.argv[1:])
