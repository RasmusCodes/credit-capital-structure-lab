# Data

Snapshot: balance sheets at the latest quarter end (30 September 2026; Wolfspeed 27 September), market data at the
close of Friday 20 November 2026, bond settlement 23 November 2026.

## Sources

| Data | Source | How it is used | In the repository |
|---|---|---|---|
| Capital structures | 10-Q (CoreWeave, IREN, Strategy, Wolfspeed), 6-K interim statements (Nebius), 8-Ks after quarter end, via SEC EDGAR | `data/filings/<T>_capstructure.json`: every instrument with principal, terms, collateral, guarantees, the accession number and a verbatim quote of at most 25 words; then `cases/<id>/case.json` | Yes (facts extracted from public filings) |
| Wolfspeed 2025 restructuring | Disclosure statement filed as Ex. 99.1 to the 8-K of 1 July 2025 (liquidation analysis, Exhibit D; valuation, Exhibit F); RSA, confirmation and emergence 8-Ks | `cases/wolfspeed_2025/case.json` | Yes |
| Bond and convert prices | FINRA TRACE last sales (finra.org Corporate and Agency Bonds), queried 21 November 2026 with `scripts/trace_query.js` | Price, trade date, CUSIP per instrument in the case files | Individual prices with CUSIP and date; not the raw pull |
| Equity, preferred closes and realised volatility | Yahoo Finance daily closes | `data/snapshots/equity_stats_2026-11-21.json` (last close, volatilities) | Derived statistics only |
| Listed options | Yahoo Finance option chains at the 20 November close | ATM implied volatilities and deep-put bounds in the case files | Derived values only |
| Bitcoin price | Coinbase Exchange hourly candle closing at 21:00 UTC on 20 November | `data/snapshots/btc_usd_2026-11-20.json` | Yes (one price) |
| Risk-free curve | US Treasury daily par yield curve | `data/snapshots/ust_par_curve_2025.csv`, `..._2026.csv` | Yes (public domain) |
| CDS | Markit 5Y official close for CoreWeave on 22 October 2026 (par 1,140.598bp, conventional 1,167.011bp, upfront 19.834%, recovery 40%), from an LSEG Workspace page recorded in the October 2026 CoreWeave study | CDS-bond basis and the converter benchmark | One quote |
| Bond marks of 22 October 2026 | BondTerminal closing marks recorded in the same study | Same-day basis with the CDS quote | Four prices |
| Historical default and transition rates | S&P Global Ratings, Form NRSRO annual certification filed on EDGAR 26 March 2026 (accession 0001650548-26-000001), Exhibit 1, Tables 7-9 | `data/reference/sp_corporate_transitions_2025.json` (`scripts/parse_nrsro.py`) | Yes (regulatory disclosure, with attribution) |

## Labels

Every claim in a case file carries `status`:

- **filed**: as stated in the filing;
- **derived**: computed from filed numbers, with the method in `notes` (for example, EUR notes at the filing's FX rate);
- **estimate**: a modelling judgment, listed again under `assumptions` with its basis and, where useful, a range;
- **convention**: an industry convention (for example, six months of pre-petition interest).

Each case also lists `missing`: information a credit analyst would want that is not disclosed (for example, the size of
the collateral behind CoreWeave's revolver, or how much of Nebius's bridge is drawn).

## What is not in the repository

`data/raw/` (git-ignored) holds the raw downloads: TRACE pulls, Yahoo chart and option JSON, the S&P PDF and the cached
filings. Their providers' terms limit redistribution, and every number the analysis uses is copied into the case files
with its source and date. Re-running the fetch scripts recreates them.

## Reproducing the snapshot

```bash
python scripts/fetch_treasury.py 2025 2026
python scripts/fetch_equity.py CRWV NBIS IREN MSTR WOLF STRF STRC STRK STRD BTC-USD
python scripts/fetch_options.py CRWV NBIS IREN MSTR WOLF
python scripts/fetch_btc.py 2026-11-20
# TRACE: run scripts/trace_query.js in the browser console on FINRA's Corporate and Agency Bonds page and save the output
SEC_USER_AGENT="research contact" python scripts/fetch_nrsro.py && python scripts/parse_nrsro.py data/raw/nrsro/<accession>/Ex1_Mar2026.pdf
python scripts/build_cases.py
```

Later snapshots will differ: market data moves, and the fetch scripts take the latest values.
