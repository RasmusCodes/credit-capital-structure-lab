# Assumptions and limits

What can make the numbers in this repository misleading, roughly in order of how much it matters.

## 1. Asset values are not observable

Structural models price everything off the firm's asset value and volatility, yet for almost every company neither
is observed. They are backed out of the equity, which makes them only as good as two assumptions: that equity is a
call option on lognormal assets, and that the equity's volatility comes from leverage alone.

The Strategy case measures how wrong that can be. Its assets are mostly bitcoin, so value ($55.6bn) and volatility
(35%) are observable. Calibrating Merton to the equity gives assets 22-35% too low and asset volatility 2.1-2.3 times
too high, because the stock also swings with its premium or discount to the bitcoin it owns. Priced off the observed
assets instead (with dividends and coupons as a 2.9% payout), the same model values the equity within 1.5% of the
market. For companies whose assets cannot be observed this error is invisible, and default probabilities from
equity-calibrated models should be read as model outputs, not measurements.

Other choices move them a lot:

- **Volatility input and barrier.** Across implied and realised, 3-month to 2-year volatility (95-102% for CoreWeave)
  and barriers of 0.5-1.25 times the KMV default point, CoreWeave's calibrated asset volatility ranges from 38% to 63%
  and its five-year Black-Cox default probability from 53% to 65% (reports/VALIDATION.md, section 3). The calibration
  absorbs much of the barrier choice into the asset volatility, so the PD moves less than its inputs.
- **Default barrier and horizon.** The barrier (where equity holders stop supporting the firm) is unobservable; KMV's
  "short-term debt plus half of long-term debt" is a convention. All debt is treated as due at one horizon.
- **Lognormal assets.** No jumps and thin tails: structural models are known to understate short-horizon spreads (the
  "credit spread puzzle"; Huang and Huang 2012). Strategy's converts price about eleven times the default risk its
  observed bitcoin volatility implies, which is largely a statement about tails, not a free lunch.
- **Static capital structure.** Real firms issue, refinance, draw undrawn commitments and sell assets before default.

## 2. Real bankruptcies do not follow absolute priority

The waterfall is a statement of legal priority. Plans are negotiated, and the Wolfspeed case shows by how much the
outcome can differ:

| | Absolute priority at the plan's mid valuation | Confirmed plan |
|---|---|---|
| Senior secured notes | 100% | 86% of an allowed claim that included the make-whole |
| Convertible notes and Renesas | about 1% | 25-30% |
| Trade creditors | about 1% | 100% (unimpaired) |
| Old equity | 0 | 5% of the new equity |

Value from outside enterprise value (cash on hand, a backstopped rights offering), make-whole and other allowed-claim
fights, claims left unimpaired to keep the business running, gifts to equity for a fast case, and plan valuations that
differ from market valuations are all outside the model. The engine reproduces the court-filed Chapter 7
liquidation analysis exactly, which shows the mechanics are right; the plan shows the mechanics are not the whole
story.

Further simplifications:

- Value is split across entities in fixed proportions (CoreWeave: by book PP&E, using the SPV collateral the 10-Q
  discloses). Real allocations are fought over.
- Guarantee claims are processed children first and capped at what is unpaid; substantive consolidation, avoidance
  actions, equitable subordination, intercompany claims and post-petition interest are not modelled
  (`substantively_consolidated` gives the pooled alternative: it moves CoreWeave's note recovery from 35% to 29%).
- Customer contracts are assumed, so prepayments create no claims. If they were rejected, refund claims would dilute
  unsecured recoveries (CoreWeave $10.0bn, Nebius $9.4bn of deferred revenue).
- Value at default is a scenario input. Recoveries move from 21% to 50% across CoreWeave's low-to-high range; no other
  input matters as much (tornado in reports/VALIDATION.md).

## 3. Market data

- **Bond prices are last trades**, not executable quotes. TRACE prints of the same bond on different lines (144A and
  Reg S) differ by up to 2.4 points; prints before a large stock move are stale for converts (the scanner drops them).
- **One CDS point.** Only Markit's 5-year CoreWeave quote of 22 October is documented, so CDS curves are assumed flat
  and the basis is computed on that date's bond marks only.
- **Option quotes** are end-of-day bid/ask with modest open interest in the deep puts; put bounds use the bid.
- **Treasury discounting.** The CDS converter uses the Treasury curve where ISDA uses SOFR swaps (about 0.02 points of
  upfront here); no holiday calendars.
- **Access and funding.** CoreWeave's notes are 144A/Reg S (institutional only). Repo and borrow costs for shorting
  bonds are unobserved, which matters for every trade with a short bond leg.

## 4. Historical default data

S&P's Form NRSRO tables are single cohorts (one starting year per horizon) that count withdrawn issuers as
non-defaults. Powers of a one-year matrix differ from those cohorts by up to half in either direction. Rating-based
default probabilities therefore carry wide error bands before any risk premium is considered.

## 5. What a "signal" is not

A spread difference is not an arbitrage. Hedged positions across a capital structure carry recovery risk (the two legs
lose different amounts in default), basis risk (different entities, guarantees, currencies, deliverables), funding
risk (repo can be withdrawn in stress), liquidity risk, and model risk in the hedge ratio. Research on capital-structure
arbitrage finds modest, risky returns (Yu 2006; Duarte, Longstaff and Yu 2007). The scanner ranks candidates for
investigation and lists the frictions next to each one.
