"""Risk-free discount curves.

ZeroCurve stores continuously compounded zero rates at pillar times (years, ACT/365F) and interpolates
log discount factors linearly (piecewise-flat forward rates). Extrapolation is flat in the forward rate
of the last segment beyond the last pillar and flat in the zero rate before the first pillar.

`from_treasury_par` bootstraps the US Treasury daily par yield curve (home.treasury.gov). Bill tenors
(<= 1 year) are treated as semiannually compounded zero yields; coupon tenors as semiannual par bonds,
with par yields interpolated linearly on a half-year grid. This is the textbook bootstrap, accurate to
a few basis points of rate, which is immaterial next to credit spreads.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TREASURY_TENORS = {
    "1 Mo": 1 / 12, "1.5 Month": 1.5 / 12, "2 Mo": 2 / 12, "3 Mo": 0.25, "4 Mo": 4 / 12, "6 Mo": 0.5,
    "1 Yr": 1.0, "2 Yr": 2.0, "3 Yr": 3.0, "5 Yr": 5.0, "7 Yr": 7.0, "10 Yr": 10.0, "20 Yr": 20.0, "30 Yr": 30.0,
}


@dataclass
class ZeroCurve:
    times: np.ndarray       # pillar times, strictly increasing, > 0
    zeros: np.ndarray       # continuously compounded zero rates at the pillars
    label: str = ""

    def __post_init__(self):
        t = np.asarray(self.times, dtype=float)
        z = np.asarray(self.zeros, dtype=float)
        order = np.argsort(t)
        self.times, self.zeros = t[order], z[order]
        if np.any(self.times <= 0):
            raise ValueError("curve pillars must be positive times")
        self._logdf = -self.zeros * self.times

    # ------------------------------------------------------------------ evaluation
    def log_df(self, t):
        t = np.asarray(t, dtype=float)
        T, L = self.times, self._logdf
        out = np.interp(t, T, L)
        # before first pillar: flat zero rate
        early = t < T[0]
        out = np.where(early, -self.zeros[0] * t, out)
        # beyond last pillar: flat forward of last segment
        if len(T) >= 2:
            f_last = -(L[-1] - L[-2]) / (T[-1] - T[-2])
        else:
            f_last = self.zeros[-1]
        late = t > T[-1]
        out = np.where(late, L[-1] - f_last * (t - T[-1]), out)
        return out

    def df(self, t):
        return np.exp(self.log_df(t))

    def zero(self, t):
        t = np.asarray(t, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = -self.log_df(t) / t
        return np.where(t > 0, z, self.zeros[0])

    def forward(self, t1, t2):
        return -(self.log_df(t2) - self.log_df(t1)) / (np.asarray(t2) - np.asarray(t1))

    def shifted(self, bp: float) -> "ZeroCurve":
        return ZeroCurve(self.times.copy(), self.zeros + bp / 1e4, label=f"{self.label}{bp:+g}bp")

    # ------------------------------------------------------------------ constructors
    @classmethod
    def flat(cls, rate: float, label: str = "flat") -> "ZeroCurve":
        return cls(np.array([0.25, 50.0]), np.array([rate, rate]), label=label)

    @classmethod
    def from_treasury_par(cls, tenors: dict[float, float], label: str = "UST par") -> "ZeroCurve":
        """tenors: {maturity in years: par yield in percent}. Returns a continuously compounded zero curve."""
        items = sorted((float(t), float(y) / 100.0) for t, y in tenors.items() if y is not None and np.isfinite(y))
        if not items:
            raise ValueError("no yields")
        ts = np.array([t for t, _ in items])
        ys = np.array([y for _, y in items])
        pillars: list[float] = []
        logdfs: list[float] = []
        # bills (<= 1y): semiannual zero yields
        for t, y in items:
            if t <= 1.0 + 1e-9:
                pillars.append(t)
                logdfs.append(-2.0 * t * np.log1p(y / 2.0))
        if not pillars:  # no bill tenors supplied: treat the first par yield as a zero yield up to 1y
            for t in (0.5, 1.0):
                pillars.append(t)
                logdfs.append(-2.0 * t * np.log1p(ys[0] / 2.0))
        if ts[-1] > 1.0 + 1e-9:
            grid = np.arange(0.5, ts[-1] + 1e-9, 0.5)
            dfs: dict[float, float] = {}

            def df_short(t):
                # discount factor from bill pillars (log-linear), used for coupon dates <= 1y
                return float(np.exp(np.interp(t, pillars, logdfs, left=None)))

            for t in grid:
                if t <= 1.0 + 1e-9:
                    dfs[round(t, 6)] = df_short(t)
                    continue
                y = float(np.interp(t, ts, ys))
                annuity = sum(dfs[round(s, 6)] for s in grid if s < t - 1e-9)
                dfs[round(t, 6)] = (1.0 - y / 2.0 * annuity) / (1.0 + y / 2.0)
            for t in grid:
                if t > 1.0 + 1e-9:
                    pillars.append(float(t))
                    logdfs.append(float(np.log(dfs[round(t, 6)])))
        p = np.array(pillars)
        z = -np.array(logdfs) / p
        return cls(p, z, label=label)


def par_yield(curve: ZeroCurve, maturity: float, freq: int = 2) -> float:
    """Par coupon (decimal, compounding = freq) of a bullet bond on `curve` (useful for checks)."""
    n = int(round(maturity * freq))
    ts = np.arange(1, n + 1) / freq
    d = curve.df(ts)
    return float(freq * (1.0 - d[-1]) / d.sum())
