"""Rating-transition models.

A TransitionMatrix holds one-period (usually one-year) migration probabilities between rating states,
with default as the last, absorbing state. Published agency matrices include a "withdrawn / NR" column;
`from_published` removes it by proportional re-allocation (the standard "NR-adjusted" treatment, which
assumes withdrawals are uninformative).

Multi-year default probabilities come either from matrix powers (discrete time, Markov) or from a
generator Q with P(t) = expm(Q t). The matrix logarithm of an empirical matrix is often not a valid
generator (negative off-diagonal rates); `generator` applies the Israel-Rosenthal-Wei (2001)
correction (set negative rates to zero and re-balance the diagonal).

Known limits of the Markov assumption: rating momentum (downgrades follow downgrades), duration
dependence and business-cycle dependence all make real cumulative default rates differ from P^n;
tests and docs compare both against the published cumulative tables where available.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, logm
from scipy.optimize import brentq


@dataclass
class TransitionMatrix:
    P: np.ndarray
    labels: list
    horizon: float = 1.0
    source: str = ""

    def __post_init__(self):
        self.P = np.asarray(self.P, dtype=float)
        n = len(self.labels)
        if self.P.shape != (n, n):
            raise ValueError("matrix shape does not match labels")
        rs = self.P.sum(axis=1)
        if np.any(np.abs(rs - 1) > 1e-6):
            raise ValueError(f"rows must sum to 1 (got {np.round(rs, 6)})")
        if np.any(self.P < -1e-12):
            raise ValueError("negative probabilities")
        if abs(self.P[-1, -1] - 1) > 1e-9:
            raise ValueError("last state (default) must be absorbing")

    @classmethod
    def from_published(cls, rows_pct, labels, nr_col: bool = True, source: str = "") -> "TransitionMatrix":
        """Build from a published table in percent. If nr_col, the last column is NR/withdrawn and is
        re-allocated proportionally across the other columns. A default row is appended if absent."""
        A = np.asarray(rows_pct, dtype=float) / 100.0
        if nr_col:
            A = A[:, :-1]
        A = A / A.sum(axis=1, keepdims=True)
        n = A.shape[1]
        if A.shape[0] == n - 1:
            A = np.vstack([A, np.eye(n)[-1]])
        return cls(A, list(labels), 1.0, source)

    @property
    def n(self) -> int:
        return len(self.labels)

    def index(self, rating: str) -> int:
        return self.labels.index(rating)

    def power(self, k: int) -> np.ndarray:
        return np.linalg.matrix_power(self.P, k)

    def cumulative_pd(self, years: int) -> np.ndarray:
        """(n_ratings - 1) x years matrix of cumulative default probabilities from matrix powers."""
        out = np.zeros((self.n - 1, years))
        M = np.eye(self.n)
        for y in range(years):
            M = M @ self.P
            out[:, y] = M[:-1, -1]
        return out

    def generator(self) -> np.ndarray:
        L = np.real(logm(self.P)) / self.horizon
        Q = L.copy()
        off = ~np.eye(self.n, dtype=bool)
        Q[off & (Q < 0)] = 0.0
        np.fill_diagonal(Q, 0.0)
        np.fill_diagonal(Q, -Q.sum(axis=1))
        Q[-1, :] = 0.0
        return Q

    def at(self, t: float) -> np.ndarray:
        return expm(self.generator() * t)

    def cumulative_pd_continuous(self, times) -> np.ndarray:
        Q = self.generator()
        return np.column_stack([expm(Q * t)[:-1, -1] for t in np.atleast_1d(times)])

    def implied_rating(self, pd: float, horizon: int = 1) -> str:
        """Rating whose cumulative PD at `horizon` years is closest to `pd` in log terms."""
        cum = self.cumulative_pd(horizon)[:, -1]
        cum = np.maximum(cum, 1e-8)
        k = int(np.argmin(np.abs(np.log(cum) - np.log(max(pd, 1e-8)))))
        return self.labels[k]

    def risk_neutral_scaling(self, rating: str, target_pd: float, horizon: float) -> float:
        """Constant multiplier l on the rating's row of the generator so that its cumulative PD at
        `horizon` matches a market-implied (risk-neutral) PD: a one-parameter version of the
        Jarrow-Lando-Turnbull (1997) risk-premium adjustment."""
        Q = self.generator()
        i = self.index(rating)

        def pd_for(l):
            Qa = Q.copy()
            Qa[i, :] *= l
            return expm(Qa * horizon)[i, -1]

        return float(brentq(lambda l: pd_for(l) - target_pd, 1e-4, 1e4))


def mobility_index(P: np.ndarray) -> float:
    """Shorrocks mobility index (n - trace)/(n - 1) over the non-default states."""
    A = P[:-1, :-1]
    n = A.shape[0]
    return float((n - np.trace(A)) / (n - 1))
