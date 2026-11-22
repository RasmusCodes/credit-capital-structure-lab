"""creditlab: credit-risk and capital-structure research toolkit.

Modules
-------
daycount, curves         dates, schedules and risk-free discount curves
bonds                    fixed-rate bond analytics (yield, duration, convexity, Z-spread, DV01, carry/roll)
hazard, cds              reduced-form models: hazard-rate curves, risky bond pricing, ISDA-style CDS
structural               Merton, Black-Cox, CreditGrades and structural pricing of a whole waterfall
capital_structure        entities, collateral pools and claims of a company
waterfall                absolute-priority distribution of value in a restructuring
transition, survival     rating-transition matrices and survival analysis
spreads                  recovery-adjusted default probabilities and spread decomposition
options, convertible     equity options, put-implied default bounds, convertible bonds with default risk
relative_value, scanner  cross-security comparisons with explicit risk flags
scenarios                scenario and jump-to-default P&L
"""

__version__ = "0.1.0"
