"""Convertible bonds with default risk: binomial tree with jump to default.

Stock dynamics (risk-neutral, before default): dS/S = (r - q + lambda (1 - eta)) dt + sigma dW, and at
default the stock jumps to eta x S (eta = 0: worthless). The hazard can be constant or rise as the
stock falls, lambda(S) = lambda0 (S / S0)^(-p) (capped), a common way to tie credit to equity.

At each node the holder may convert (conversion ratio x S); the issuer may call (forcing conversion if
parity exceeds the call price) when the call is live and the soft-call trigger is met; the holder may
put on put dates. On default the holder receives max(recovery x face, conversion of the post-default
stock). Coupons are paid on schedule if the issuer survives. Accrued interest on conversion and
call/put notice periods are ignored (documented simplification).

`implied_vol` and `implied_hazard` invert the model against a market price; `bond_floor` prices the
same cash flows with no conversion right, which ties back to hazard.risky_dirty_price.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq


@dataclass
class Convertible:
    name: str
    coupon: float                   # annual coupon, decimal
    maturity: float                 # years to maturity
    conversion_price: float         # USD per share
    face: float = 100.0
    freq: int = 2
    call_start: float | None = None   # years from now when the issuer call becomes live
    call_price: float = 100.0
    call_trigger: float = 1.3       # soft call: stock >= trigger x conversion price
    puts: list = field(default_factory=list)   # [(time in years, put price per 100)]
    recovery: float = 0.4           # fraction of face recovered at default

    @property
    def ratio(self) -> float:
        """Shares per `face` of bonds."""
        return self.face / self.conversion_price


def _coupon_times(cb: Convertible) -> np.ndarray:
    n = int(np.floor(cb.maturity * cb.freq + 1e-9))
    return np.array([cb.maturity - k / cb.freq for k in range(n)])[::-1]


def price_convertible(cb: Convertible, S0: float, sigma: float, r: float, q: float = 0.0,
                      hazard: float = 0.0, hazard_power: float = 0.0, hazard_cap: float = 5.0,
                      eta: float = 0.0, steps: int = 400, convert: bool = True) -> dict:
    T = cb.maturity
    # the up move must exceed the default-compensated drift, or branch probabilities leave [0, 1]
    lam_max = hazard_cap if hazard_power else hazard
    need = int(np.ceil(1.2 * T * (abs(r - q) + lam_max * (1 - eta)) ** 2 / max(sigma * sigma, 1e-12))) + 1
    steps = max(steps, min(need, 20000))
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    ratio = cb.ratio if convert else 0.0
    cpn_times = _coupon_times(cb)
    cpn_amt = cb.coupon / cb.freq * cb.face
    # coupon paid at the end of step k if a coupon date falls in (t_k, t_{k+1}]
    step_cpn = np.zeros(steps + 1)
    for tc in cpn_times:
        k = int(np.clip(np.ceil(tc / dt - 1e-9), 1, steps))
        step_cpn[k] += cpn_amt
    put_at = {}
    for tp, px in cb.puts:
        if 0 < tp < T:
            put_at[int(round(tp / dt))] = px

    def lam_of(S):
        if hazard_power == 0.0:
            return np.full_like(S, hazard)
        return np.minimum(hazard * (S / S0) ** (-hazard_power), hazard_cap)

    j = np.arange(steps + 1)
    S = S0 * u ** (steps - 2 * j)
    V = np.maximum(cb.face + step_cpn[steps], ratio * S) if convert else np.full(steps + 1, cb.face + step_cpn[steps])
    for k in range(steps - 1, -1, -1):
        jj = np.arange(k + 1)
        S = S0 * u ** (k - 2 * jj)
        lam = lam_of(S)
        growth = np.exp((r - q + lam * (1 - eta)) * dt)
        p = np.clip((growth - d) / (u - d), 0.0, 1.0)
        surv = np.exp(-lam * dt)
        default_val = np.maximum(cb.recovery * cb.face, ratio * eta * S)
        cont = disc * (surv * (p * V[:-1] + (1 - p) * V[1:]) + (1 - surv) * default_val)
        t = k * dt
        if convert:
            if cb.call_start is not None and t >= cb.call_start:
                callable_now = S >= cb.call_trigger * cb.conversion_price
                cont = np.where(callable_now, np.minimum(cont, np.maximum(cb.call_price, ratio * S)), cont)
            if k in put_at:
                cont = np.maximum(cont, put_at[k])
            cont = np.maximum(cont, ratio * S)
        elif k in put_at:
            cont = np.maximum(cont, put_at[k])
        if k > 0:
            cont = cont + step_cpn[k]
        V = cont
        if k == 1:
            V1 = V.copy()
            S1 = S.copy()
        if k == 2:
            V2 = V.copy()
            S2 = S.copy()
    price = float(V[0])
    out = {"price": price, "parity": ratio * S0, "premium": price / (ratio * S0) - 1 if ratio > 0 else np.nan}
    if steps >= 3:
        delta = (V1[0] - V1[1]) / (S1[0] - S1[1])
        g_up = (V2[0] - V2[1]) / (S2[0] - S2[1])
        g_dn = (V2[1] - V2[2]) / (S2[1] - S2[2])
        gamma = (g_up - g_dn) / (0.5 * (S2[0] - S2[2]))
        out.update({"delta_per_bond": float(delta), "delta_shares": float(delta), "gamma": float(gamma),
                    "delta_pct_parity": float(delta / ratio) if ratio > 0 else np.nan})
    return out


def bond_floor(cb: Convertible, S0, sigma, r, q=0.0, hazard=0.0, hazard_power=0.0, steps=400, eta=0.0) -> float:
    return price_convertible(cb, S0, sigma, r, q, hazard, hazard_power, steps=steps, eta=eta, convert=False)["price"]


def implied_vol(cb: Convertible, price: float, S0, r, q=0.0, hazard=0.0, hazard_power=0.0, steps=300) -> float:
    f = lambda v: price_convertible(cb, S0, v, r, q, hazard, hazard_power, steps=steps)["price"] - price
    lo, hi = 0.02, 4.0
    flo, fhi = f(lo), f(hi)
    if flo > 0:
        return float("nan")  # price below the zero-vol value: cheap even with no option value
    if fhi < 0:
        return float("inf")
    return float(brentq(f, lo, hi, xtol=1e-6))


def implied_hazard(cb: Convertible, price: float, S0, sigma, r, q=0.0, hazard_power=0.0, steps=300,
                   convert: bool = True) -> float:
    """Hazard rate that reproduces `price`. With convert=False the option is ignored: the result is the
    largest hazard consistent with the price (the "straight-bond bound"), since the option is worth >= 0.
    Returns inf when even a 300% hazard leaves the model above the price (the jump-to-default model keeps
    a sizeable option value at high hazard because surviving paths drift up by the hazard rate)."""
    f = lambda h: price_convertible(cb, S0, sigma, r, q, h, hazard_power, steps=steps, convert=convert)["price"] - price
    if f(0.0) < 0:
        return 0.0
    hi = 3.0
    if f(hi) > 0:
        return float("inf")
    return float(brentq(f, 0.0, hi, xtol=1e-7))
