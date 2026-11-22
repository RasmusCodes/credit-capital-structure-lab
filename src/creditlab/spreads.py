"""Recovery-adjusted default probabilities and credit-spread decomposition.

The single most important caveat in reading credit markets: a spread prices expected LOSS, roughly
hazard x (1 - recovery). The same 1,000bp spread is a 15% annual default intensity at 33% recovery and
a 50% intensity at 80% recovery. Every market-implied PD in this platform is therefore reported with
its recovery, and recoveries are derived from the waterfall rather than assumed where possible.

Spread decomposition (bp, for one bond):
    z-spread  = expected-loss spread          lambda_P x (1 - R_P), from historical (physical) data
              + default-risk premium          market default spread (CDS or bond-implied) - expected loss
              + non-default component          bond z-spread - CDS-equivalent spread (liquidity, funding,
                                               taxes, delivery option: the negative of the CDS-bond basis)
This is the Longstaff-Mithal-Neis (2005) idea (CDS as the default component) combined with an
expected-loss anchor (Elton et al. 2001; Huang-Huang 2012 show structural models explain only a small
part of investment-grade spreads).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .hazard import implied_flat_hazard


def credit_triangle_hazard(spread: float, recovery: float) -> float:
    return spread / (1.0 - recovery)


def credit_triangle_pd(spread: float, recovery: float, T: float) -> float:
    return 1.0 - np.exp(-credit_triangle_hazard(spread, recovery) * T)


def pd_recovery_table(spread: float, T: float, recoveries=(0.0, 0.2, 0.4, 0.6, 0.8)) -> list[dict]:
    return [{"recovery": R, "hazard": credit_triangle_hazard(spread, R), "pd": credit_triangle_pd(spread, R, T)}
            for R in recoveries]


def bond_pd_by_recovery(bond, clean, settle, zc, recoveries, horizon: float, convention: str = "RFV"):
    """Bond-implied cumulative PD to `horizon` for each recovery (flat hazard)."""
    out = []
    for R in recoveries:
        fit = implied_flat_hazard(bond, clean, settle, zc, R, convention)
        lam = float(fit.curve.hazards[0])
        out.append({"recovery": R, "hazard": lam, "pd": float(1 - np.exp(-lam * horizon)) if np.isfinite(lam) else 1.0,
                    "status": fit.status})
    return out


def breakeven_recovery(bond, clean, settle, zc, target_pd: float, horizon: float, convention: str = "RFV"):
    """Recovery at which the bond-implied PD to `horizon` equals `target_pd` (e.g. the PD implied by
    deep puts or by a structural model). Returns None if no recovery in [0, R_max) achieves it.

    Bond-implied PD rises with the assumed recovery (more default needed to explain the same price), so
    a target PD above the bond's PD at zero recovery cannot be matched.
    """
    def f(R):
        fit = implied_flat_hazard(bond, clean, settle, zc, R, convention)
        if fit.status == "price_below_recovery":
            return 1.0
        lam = float(fit.curve.hazards[0])
        return (1 - np.exp(-lam * horizon)) - target_pd

    dirty = clean + bond.accrued(settle)
    r_max = min(dirty / bond.face - 1e-6, 0.999)
    lo, hi = 0.0, r_max
    flo, fhi = f(lo), f(hi)
    if flo > 0:
        return None   # even at zero recovery the bond implies more default than the target
    if fhi < 0:
        return None
    return float(brentq(f, lo, hi, xtol=1e-8))


def decompose_spread(z_spread: float, cds_equiv_spread: float | None, physical_hazard: float | None,
                     physical_recovery: float | None) -> dict:
    """Split a bond's z-spread (decimal) into expected loss, default-risk premium and non-default parts.

    Any input left as None leaves the corresponding pieces unidentified (reported as None) rather than
    forcing a residual into them.
    """
    out = {"z_spread_bp": z_spread * 1e4, "expected_loss_bp": None, "default_risk_premium_bp": None,
           "non_default_bp": None, "notes": []}
    if physical_hazard is not None and physical_recovery is not None:
        out["expected_loss_bp"] = physical_hazard * (1 - physical_recovery) * 1e4
    if cds_equiv_spread is not None:
        out["non_default_bp"] = (z_spread - cds_equiv_spread) * 1e4
        if out["expected_loss_bp"] is not None:
            out["default_risk_premium_bp"] = cds_equiv_spread * 1e4 - out["expected_loss_bp"]
    else:
        out["notes"].append("no CDS: default and non-default components not separable")
        if out["expected_loss_bp"] is not None:
            out["default_risk_premium_bp"] = None
            out["notes"].append("z-spread minus expected loss mixes risk premium and non-default parts")
    return out
