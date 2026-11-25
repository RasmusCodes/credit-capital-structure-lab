# Methodology

Conventions throughout: amounts in currency millions, time in years (ACT/365F unless stated), continuously
compounded rates, risk-neutral probabilities unless labelled physical. Code references are to `src/creditlab/`.

## 1. Capital structure and the waterfall

**Data model** (`capital_structure.py`). A company is a tree of legal entities. Each entity produces a share of
enterprise value (`value_share`), may pledge part of it in collateral pools, and owes claims. A claim has an obligor,
an allowed amount at default, an unsecured rank, optional liens on pools (with lien priority), guarantors, and
optional contractual subordination with turnover. Preferred stock has a rank below debt; common equity is the
residual.

**Allowed claims.** Principal plus six months of pre-petition interest for funded debt, and a revolver drawn to 85%
of its commitment, following the S&P recovery-rating conventions the October 2026 CoreWeave study used. Lease-rejection
claims follow 11 U.S.C. 502(b)(6): the rent reserved for the greater of one year or 15% (capped at three years) of the
remaining term (`lease_rejection_claim`).

**Waterfall** (`waterfall.distribute`, vectorised over values V):

1. In default, administrative costs take `admin_cost x V`; an optional gift to old equity carves out `g x V` when
   creditors are impaired (the pattern of negotiated plans).
2. Entities are processed children first. At each entity: collateral pools pay their secured claims by lien
   priority (pro rata within a lien); unencumbered value plus pool surpluses plus subsidiaries' residual equity pays
   unsecured claims rank by rank (priority, senior incl. deficiency claims of recourse secured debt and guarantee
   claims against this entity, subordinated, preferred); subordinated claims with turnover hand their recoveries to
   the designated senior claims; the residual moves up to the parent.
3. At the top entity the residual (plus any gift) goes to common equity.

A claim guaranteed by several entities is allowed in full against each but capped at what is still unpaid when an
entity is processed. `breakpoints` finds the values at which each claim becomes paid in full (bisection; payouts are
monotone in V); `fulcrum` is the most senior impaired claim.

## 2. Structural models

**Merton (1974)** (`structural/merton.py`). Assets follow GBM with volatility s and payout q; debt of face D is due at
T. Equity is a call: E = V e^{-qT} N(d1) - D e^{-rT} N(d2); risk-neutral PD = N(-d2); spread = -ln(B / D e^{-rT}) / T;
expected recovery given default = V e^{(r-q)T} N(-d1) / (D N(-d2)). Calibration solves E(V, s) = E_obs and
N(d1) e^{-qT} s V = s_E E_obs. The iterative KMV procedure (Crosbie-Bohn; Bharath-Shumway 2008) re-estimates s from
implied asset returns. `tranche_values` prices seniority tranches as call spreads on V.

**Black-Cox (1976)** (`structural/black_cox.py`). Default at the first passage of V below H(t) = H0 e^{gt}, or at T if
V_T < D. With X = ln V - gt, drift nu = r - q - s^2/2 - g:

- P(tau <= T) = N((-a - nu T)/(s sqrt T)) + e^{-2 nu a / s^2} N((-a + nu T)/(s sqrt T)), a = ln(V0/H0);
- E[e^{-r tau} 1{tau <= T}] = e^{-a(nu + nt)/s^2} N((-a + nt T)/(s sqrt T)) + e^{-a(nu - nt)/s^2} N((-a - nt T)/(s sqrt T)),
  nt = sqrt(nu^2 + 2 r s^2);
- no-hit partial moments of V_T by the reflection principle (`lognormal.barrier_partial_moments`).

Equity is a down-and-out call; debt receives (1 - alpha) H at the barrier or min(V_T, D) at T.

**CreditGrades** (Finger et al. 2002, `structural/creditgrades.py`). Barrier L D with lognormal recovery L (mean Lbar,
log-sd lam); V = S + Lbar D, s = s_S S / V. Survival
P(t) = N(-A/2 + ln d / A) - d N(-A/2 - ln d / A), d = V e^{lam^2} / (Lbar D), A^2 = s^2 t + lam^2, and a closed-form CDS
spread (checked against numerical integration of the survival curve to 2e-8 bp).

**Pricing a whole waterfall on one asset process** (`structural/waterfall_pricing.py`). At the horizon each claim
receives its waterfall payout: going-concern priority shares if V_T is above the default point, or the default
waterfall (with costs, collateral, entities, gift) if not. Under Black-Cox, a barrier hit pays the default waterfall of
H. Payouts are smooth in V between breakpoints, so the lognormal integral is split at the breakpoints and at the
default point and done with 16-point Gauss-Legendre in ln V; above the last breakpoint every claim is paid in full and
equity is linear in V, so that tail is integrated in closed form. The pricer reproduces multi-tranche Merton to 1e-14
and Black-Cox to 1e-6. `calibrate_structure` fits (V0, s) so that the model's common equity matches market value and
volatility. `structural_hazard_curve` turns first-passage survival into a hazard curve, so coupon bonds of any maturity
can be priced off the asset process with the waterfall's recovery at the barrier (the "hybrid" used for bond-level
comparisons).

## 3. Reduced-form models

**Hazard curves** (`hazard.py`). Piecewise-constant intensity; survival Q(t) = exp(-integral lambda). Risky bonds:

- recovery of face (default): sum_k CF_k DF(t_k) Q(t_k) + R x face x sum_j DF(t_j*) (Q(t_{j-1}) - Q(t_j)) on a weekly grid;
- recovery of Treasury: sum_k CF_k DF(t_k) (Q(t_k) + R (1 - Q(t_k)));
- recovery of market value: sum_k CF_k DF(t_k) Q(t_k)^{1-R} (spread = (1 - R) lambda exactly).

`implied_flat_hazard` flags prices above the riskless value and prices below the recovery floor (a recovery
assumption inconsistent with the price). `bootstrap_hazard` fits one segment per bond; `fit_hazard_curve` fits fewer
knots than bonds by least squares weighted by 1/CS01, so residuals are in spread terms (curve relative value).

**CDS** (`cds.py`). Quarterly coupons on adjusted IMM dates, ACT/360, accrued premium on default, protection from step-in
to maturity, legs integrated daily. Clean upfront is forward-valued to T+3. Quoted (conventional) spreads convert to
upfront through a flat hazard, as the ISDA converter does. Reproduces Markit's CoreWeave 5Y close (22 Oct 2026) to 0.02
points on the Treasury curve.

**Recovery-adjusted default probability** (`spreads.py`). The same spread implies a hazard of roughly s / (1 - R):
default probabilities are always reported with their recovery. `breakeven_recovery` finds the recovery at which a
bond's implied PD equals another market's (puts, a structural model). `decompose_spread` splits a z-spread into an
expected-loss part (physical PD x physical LGD), a default-risk premium (CDS-equivalent spread minus expected loss)
and a non-default part (z-spread minus CDS-equivalent spread), leaving any piece without data unidentified.

## 4. Instruments

**Bonds** (`bonds.py`). Street-convention yields on 30/360, yield to worst over call dates, Macaulay and modified
duration, convexity, DV01, Z-spread over the zero curve, spread DV01 (CS01), and one-year carry/roll-down at unchanged
curves (income, clean-price roll along the issuer's fitted spread curve, funding).

**Convertibles** (`convertible.py`). Binomial tree with jump to default: pre-default drift r - q + lambda (1 - eta),
stock to eta S at default, holder receives max(R x face, conversion of the post-default stock). Conversion is
American; soft calls force conversion above the trigger; holder puts at par on their dates; coupons on schedule. The
hazard can depend on the stock (lambda0 (S/S0)^-p). Outputs: price, delta, gamma, bond floor, implied volatility at a
given hazard, implied hazard at a given volatility, and the option-ignoring hazard bound. The step count rises
automatically so that branch probabilities stay in [0, 1].

**Options** (`options.py`). Black-Scholes prices and implied volatility. A put pays K if the stock is worthless at
default, so P >= K e^{-rT} Q(default by T): a model-free upper bound Q <= P e^{rT} / K (valid for American puts). Carr
and Wu's (2011) unit recovery claim, (P(B) - P(A)) / (B - A), and the constant hazard it implies.

## 5. Ratings and survival

**Transition matrices** (`transition.py`). Published one-year matrices with the withdrawn column re-allocated
proportionally; multi-year PDs from matrix powers; a generator from the matrix logarithm with the Israel-Rosenthal-Wei
correction; PD-implied ratings; a one-parameter Jarrow-Lando-Turnbull scaling that matches a market-implied PD.

**Survival** (`survival.py`). Kaplan-Meier with Greenwood standard errors, Nelson-Aalen, conversions between cumulative,
marginal and average-hazard default rates, a discrete-time logit hazard model (Shumway 2001) by IRLS, and the ratio of
risk-neutral to physical default probabilities.

## 6. Relative value and scenarios

Per issuer (`analysis.py`, `relative_value.py`): bond analytics at the waterfall recovery; one hazard curve through all
bonds and the residuals; carry minus expected loss; CDS-bond basis at matched maturity; equity-calibrated structural
prices of each bond; convertibles against listed implied volatility and the issuer hazard; put bounds against bond PDs;
seniority checks (spread ratios against the waterfall's LGD ratios); cross-issuer market-vs-model PD ratios.

The scanner (`scanner.py`) turns these into signals ranked by size and discounted by risk flags (stale or dispersed
prices, recovery sensitivity, unidentified PDs, funding of short bond legs, equity borrow, illiquid options, model
dependence, dated inputs, structural inputs, restricted access, deep in-the-money converts). Confidence is high with no
flags, medium with one or two, low with three or more or any model dependence.

`scenarios.py` computes jump-to-default and repricing P&L per position, P&L across recovery scenarios, and hedge
ratios that neutralise a bond's jump-to-default loss with puts. `analysis.example_trade_section` applies them to a
note hedged with deep puts: default at the waterfall's low/central/high recoveries, equity -50%/+50% without default
(assets moved so the equity-calibrated Black-Cox model reproduces the new equity value, the note repriced on the
structural hazard curve, puts at their own implied volatility held fixed for the strike), and a year of carry.
`analysis.decomposition_section` splits each bond's par-equivalent CDS spread into expected loss (S&P's historical
default intensity for the notes' rating from the 3-year cohort, times the waterfall's loss given default), a
default-risk premium, and a non-default component that is identified only on a date with a CDS quote.

## 7. The browser engine

`web/lab-engine.js` re-implements the normal CDF (series and continued fraction), the lognormal and barrier formulas,
the waterfall, breakpoints and the waterfall pricer. `scripts/export_web.py` writes reference vectors from the Python
engine; `web/test/index.html` and `tests/js/parity.mjs` recompute them (206 checks: normal CDF to 1e-12, payouts to
1e-9, prices to 1e-6). The demo's bond prices use the hybrid of section 2; its Monte Carlo simulates weekly GBM paths
with a Brownian-bridge barrier correction and sends each default through the waterfall.

## References

Black and Cox (1976), Valuing corporate securities: some effects of bond indenture provisions, *Journal of Finance*.
Bharath and Shumway (2008), Forecasting default with the Merton distance to default model, *Review of Financial Studies*.
Carr and Wu (2011), A simple robust link between American puts and credit protection, *Review of Financial Studies*.
Duarte, Longstaff and Yu (2007), Risk and return in fixed-income arbitrage: nickels in front of a steamroller?, *Review of Financial Studies*.
Duffie and Singleton (1999), Modeling term structures of defaultable bonds, *Review of Financial Studies*.
Elton, Gruber, Agrawal and Mann (2001), Explaining the rate spread on corporate bonds, *Journal of Finance*.
Finger et al. (2002), CreditGrades technical document, RiskMetrics Group.
Huang and Huang (2012), How much of the corporate-Treasury yield spread is due to credit risk?, *Review of Asset Pricing Studies*.
Israel, Rosenthal and Wei (2001), Finding generators for Markov chains via empirical transition matrices, *Mathematical Finance*.
Jarrow, Lando and Turnbull (1997), A Markov model for the term structure of credit risk spreads, *Review of Financial Studies*.
Longstaff, Mithal and Neis (2005), Corporate yield spreads: default risk or liquidity?, *Journal of Finance*.
Merton (1974), On the pricing of corporate debt: the risk structure of interest rates, *Journal of Finance*.
Shumway (2001), Forecasting bankruptcy more accurately: a simple hazard model, *Journal of Business*.
Yu (2006), How profitable is capital structure arbitrage?, *Financial Analysts Journal*.
