# Credit and capital-structure lab

A research platform for reading a company's credit risk off every security it has issued (loans, secured and
unsecured bonds, convertibles, preferred stock, equity, CDS and listed options) and asking whether they price that risk
consistently.

**Interactive demo:** [rasmuscodes.github.io/projects/credit-lab/](https://rasmuscodes.github.io/projects/credit-lab/)

The core question: *can mispricings between equity, debt, convertibles, CDS and options be found by modelling the
company's credit risk and capital structure?* The platform answers it in a common currency (default probability at
an explicit recovery) and refuses to call any gap an arbitrage. Every signal comes with the frictions that can keep
it open: liquidity, funding, optionality, basis and model risk.

## What is in the repository

| Component | Where | What it does |
|---|---|---|
| Capital-structure engine | `src/creditlab/capital_structure.py`, `waterfall.py` | Legal entities, collateral pools, liens, guarantees, structural subordination, turnover of subordinated recoveries, lease-rejection caps, administrative costs and deviations from absolute priority; vectorised waterfall |
| Structural models | `src/creditlab/structural/` | Merton (two-equation and iterative KMV calibration, multi-tranche), Black-Cox first passage, CreditGrades, and a pricer that values every claim of a real waterfall on one asset process |
| Reduced-form models | `hazard.py`, `cds.py` | Piecewise hazard curves, risky bonds under recovery of face / Treasury / market value, bond bootstraps and curve fits, ISDA-style CDS (upfront, quoted spread, bootstrap, CS01) |
| Ratings and survival | `transition.py`, `survival.py` | Transition matrices, generators (Israel-Rosenthal-Wei), multi-year PDs, risk-neutral scaling; Kaplan-Meier, Nelson-Aalen, discrete-time hazard models |
| Instruments | `bonds.py`, `convertible.py`, `options.py` | Yield, duration, convexity, Z-spread, spread DV01, carry and roll-down; convertibles with jump to default (implied vol / hazard, bond floor); put-implied default bounds (Carr-Wu) |
| Relative value | `relative_value.py`, `spreads.py`, `scanner.py`, `scenarios.py` | Bond vs CDS basis, equity vs credit, converts vs options, curve residuals, seniority consistency, cross-issuer screens, spread decomposition, breakeven recovery, scenario and jump-to-default P&L, and a ranked scanner with risk flags |
| Case studies | `cases/`, `reports/cases/` | CoreWeave, Strategy, Nebius, IREN, Wolfspeed, and Wolfspeed's 2025 Chapter 11 as a test against a real restructuring |
| Validation | `reports/VALIDATION.md` | Benchmarks against published answers, sensitivity, restructuring scenarios, historical default experience, every model side by side |
| Browser engine and demo | `web/` | JavaScript port of the waterfall and structural pricer (206 parity checks against Python) and the interactive demo |
| Documentation | `docs/` | Methodology, assumptions and limits, data sources |

## Findings from the case studies (market data of 20 November 2026)

- **CoreWeave.** The senior notes recover 35% in the central default scenario (21-50% across the range) once the
  SPV term loans take their collateral and the notes share the operating subsidiaries' value with trade and lease
  claims. At that recovery the notes imply a 60% five-year default probability; the equity-calibrated Black-Cox model
  says 65% and prices the notes 4.7-8.4 points below where they trade. The Jan-2029 $5 puts cap the risk-neutral
  default probability by that date at 16% (bid), against 32-34% implied by the notes. On 22 October, the one date with
  a documented CDS quote, CDS traded 31-114bp wider than the notes' par-equivalent spreads.
- **Strategy.** Its assets are mostly bitcoin, so the structural model needs no calibration. With observed assets
  ($55.6bn), observed volatility (35%) and dividends paid out of the asset pool, the model values the equity at
  $32.4bn against a $32.9bn market value. Calibrating Merton to the equity instead understates assets by 22-35% and
  roughly doubles asset volatility: a measured version of the problem every structural model has when assets cannot
  be observed. The converts price about eleven times the five-year default risk the asset model implies, and the junior
  preferreds yield far more than liquidation priority explains (STRD, non-cumulative, most of all).
- **Wolfspeed 2025.** The waterfall reproduces the court-filed Chapter 7 liquidation analysis exactly. The confirmed
  plan then paid unsecured creditors 25-30% where absolute priority at the plan's own valuation gave about 1%:
  make-whole claims, rights offerings, unimpaired trade claims and a 5% gift to old equity are what real
  restructurings add.
- **Nebius and IREN.** Their convertibles imply at most 26-51% five-year default probabilities (an upper bound that
  ignores the conversion option); equity-calibrated structural models say 73-75%. IREN's converts sit structurally
  below secured GPU financings and recover 13% in the central scenario.

These are model readings on dated snapshots, not trade recommendations. Details: [reports/cases/](reports/cases/).

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e .[research,dev]     # Windows; use .venv/bin/python elsewhere
.venv/Scripts/python -m pytest                              # 49 tests, no network
node tests/js/parity.mjs                                    # JavaScript engine vs Python vectors
```

Rebuild everything from the committed inputs:

```bash
python scripts/run_cases.py        # case analyses -> reports/cases/<id>/analysis.json
python scripts/run_scanner.py      # relative-value scanner -> reports/scanner.md
python scripts/run_validation.py   # validation experiments -> reports/validation/results.json
python scripts/make_reports.py     # Markdown reports and charts
python scripts/export_web.py       # demo data + parity vectors -> web/
```

Refreshing market data (needs network): `fetch_treasury.py`, `fetch_equity.py`, `fetch_options.py`, `fetch_btc.py`,
`scripts/trace_query.js` (run on FINRA's bond page), then `build_cases.py`. SEC downloads read a contact
`SEC_USER_AGENT` from the environment, as the SEC's fair-access policy requires.

A small example:

```python
import numpy as np
from creditlab.capital_structure import CapitalStructure, Claim, CollateralPool, Entity
from creditlab.waterfall import distribute
from creditlab.structural.waterfall_pricing import calibrate_structure

cs = CapitalStructure("example", [Entity("co", "Company", None, 1.0)], [CollateralPool("all", "co", 0.8)], [
    Claim("tl", "Term loan", "term_loan", "co", 900, secured_by=(("all", 1),)),
    Claim("notes", "Senior notes", "senior_unsecured", "co", 600),
    Claim("eq", "Common", "common", "co")], admin_cost=0.05)
print(distribute(cs, np.array([1000.0])).recovery)        # recoveries if the firm defaults worth 1,000
fit = calibrate_structure(cs, E_obs=400, sigma_E_obs=0.6, T=5, r=0.04, model="black_cox", barrier_ratio=0.7)
print(fit.V0, fit.sigma, fit.pd_q, fit.values)             # asset value, vol, PD and every claim's value
```

## Data

Capital structures come from 10-Q, 6-K and 8-K filings (each number carries its accession number and a short verbatim
quote), bond prices from FINRA TRACE last sales, equity and option data from public market data, the Treasury curve
from the US Treasury, historical default rates from S&P's Form NRSRO filing on EDGAR, and one CDS quote (Markit via
LSEG, 22 October 2026). Nothing is invented: estimates are labelled as estimates, with their basis, and missing
information is listed per case. Raw downloads are not redistributed. See [docs/DATA.md](docs/DATA.md).

## Limits

Structural models assume lognormal, observable-in-principle assets, a static debt stack and frictionless markets;
real bankruptcies involve negotiation, valuation fights, make-wholes, new money and timing that no waterfall
captures. Market prices are last trades of thinly traded bonds. The documentation spells out where each of these
bites: [docs/ASSUMPTIONS_AND_LIMITS.md](docs/ASSUMPTIONS_AND_LIMITS.md).

## Documentation

- [docs/METHODOLOGY.md](docs/METHODOLOGY.md): models, formulas and conventions
- [docs/ASSUMPTIONS_AND_LIMITS.md](docs/ASSUMPTIONS_AND_LIMITS.md): what can make these models misleading
- [docs/DATA.md](docs/DATA.md): sources, dates and what is or is not in the repository
- [reports/VALIDATION.md](reports/VALIDATION.md): benchmarks, sensitivity, scenarios, historical comparisons
- [reports/scanner.md](reports/scanner.md): the relative-value scanner's output
- [reports/cases/](reports/cases/): one report per case study

## License

MIT, see [LICENSE](LICENSE). Not investment advice.
