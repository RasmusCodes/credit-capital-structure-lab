"""Extract S&P Global Ratings' corporate-issuer transition/default tables from Form NRSRO Exhibit 1.

Source: S&P Global Ratings, Form NRSRO annual certification filed on SEC EDGAR 2026-03-26
(accession 0001650548-26-000001), Exhibit 1, Tables 7-9: Corporate Issuers 1-, 3- and 10-year transition
and default rates (single cohorts ending 31 Dec 2025). Text is placed by x-position so that blank cells
stay blank. Output: data/reference/sp_corporate_transitions_2025.json (rates in percent, as published).

Usage: python scripts/parse_nrsro.py data/raw/nrsro/0001650548-26-000001/Ex1_Mar2026.pdf
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
RATINGS = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-", "B+", "B", "B-",
           "CCC+", "CCC", "CCC-", "CC", "C"]
COLS = RATINGS + ["Default", "Paid Off", "Withdrawn (Other)"]
PAGES = {"1y": (9, "December 31, 2024 through December 31, 2025"),
         "3y": (10, "December 31, 2022 through December 31, 2025"),
         "10y": (11, "December 31, 2015 through December 31, 2025")}


def chunks(page):
    items = []

    def visit(text, cm, tm, font_dict, font_size):
        t = text.strip()
        if t:
            x = tm[4] * cm[0] + cm[4] if cm else tm[4]
            y = tm[5] * cm[3] + cm[5] if cm else tm[5]
            items.append((round(y, 1), x, t))

    page.extract_text(visitor_text=visit)
    return items


def parse_page(page):
    items = chunks(page)
    # header row: the y with the most rating-label tokens
    by_y: dict = {}
    for y, x, t in items:
        by_y.setdefault(y, []).append((x, t))
    header_y = max(by_y, key=lambda y: sum(1 for _, t in by_y[y] if t in COLS or t in ("Paid", "Withdrawn")))
    header = sorted(by_y[header_y])
    col_x = []
    toks = [t for _, t in header]
    xs = [x for x, _ in header]
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == "Paid":
            col_x.append(("Paid Off", xs[i])); i += 2; continue
        if t == "Withdrawn":
            col_x.append(("Withdrawn (Other)", xs[i])); i += 2; continue
        if t in COLS:
            col_x.append((t, xs[i]))
        i += 1
    rows = {}
    for y, lst in by_y.items():
        if y >= header_y:
            continue
        lst = sorted(lst)
        if not lst or lst[0][1] not in RATINGS:
            continue
        label = lst[0][1]
        vals = [(x, t) for x, t in lst[1:]]
        nums = [(x, t) for x, t in vals if re.fullmatch(r"[\d,]+", t)]
        pcts = [(x, t) for x, t in vals if t.endswith("%")]
        row = {"n": int(nums[0][1].replace(",", "")) if nums else None, "pct": {}}
        for x, t in pcts:
            name = min(col_x, key=lambda c: abs(c[1] - x))[0]
            row["pct"][name] = float(t.rstrip("%"))
        rows[label] = row
    return rows, [c for c, _ in col_x]


def main(pdf):
    r = PdfReader(pdf)
    out = {"source": "S&P Global Ratings, Form NRSRO Exhibit 1 (EDGAR accession 0001650548-26-000001, filed 2026-03-26), "
                     "Tables 7-9: Corporate Issuers transition and default rates, single-cohort method",
           "units": "percent of ratings outstanding at the start of the period", "tables": {}}
    for key, (pg, period) in PAGES.items():
        text = r.pages[pg - 1].extract_text()
        assert period in text and "Corporate Issuers" in text, (key, pg)
        rows, cols = parse_page(r.pages[pg - 1])
        for lab, row in rows.items():
            s = sum(row["pct"].values())
            row["row_sum"] = round(s, 2)
        out["tables"][key] = {"period": period, "columns": cols, "rows": rows}
        print(key, period, len(rows), "rows; row sums", sorted({round(v["row_sum"]) for v in rows.values()}))
    p = ROOT / "data" / "reference" / "sp_corporate_transitions_2025.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main(sys.argv[1])
