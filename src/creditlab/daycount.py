"""Dates, day counts and payment schedules.

Only weekend adjustment is modelled (no holiday calendars). For credit pricing at the precision this
platform needs (spreads of tens to thousands of basis points) the error is far below a basis point,
but it is a known simplification (see docs/ASSUMPTIONS_AND_LIMITS.md).
"""
from __future__ import annotations

import datetime as dt
from typing import Iterable

from dateutil.relativedelta import relativedelta

Date = dt.date


def to_date(x) -> Date:
    """Accept date, datetime or ISO string (YYYY-MM-DD or YYYY-MM, the latter -> 15th)."""
    if isinstance(x, dt.datetime):
        return x.date()
    if isinstance(x, dt.date):
        return x
    s = str(x).strip()
    if len(s) == 7:  # YYYY-MM: month known, day not -> mid-month
        return dt.date(int(s[:4]), int(s[5:7]), 15)
    return dt.date.fromisoformat(s[:10])


def add_months(d: Date, n: int) -> Date:
    return d + relativedelta(months=n)


def is_weekend(d: Date) -> bool:
    return d.weekday() >= 5


def adjust_following(d: Date) -> Date:
    while is_weekend(d):
        d += dt.timedelta(days=1)
    return d


def add_business_days(d: Date, n: int) -> Date:
    step = 1 if n >= 0 else -1
    left = abs(n)
    while left:
        d += dt.timedelta(days=step)
        if not is_weekend(d):
            left -= 1
    return d


def thirty360(d1: Date, d2: Date) -> float:
    """30/360 US bond basis (ISDA 2006 4.16(f)): D1=31 -> 30; D2=31 and D1>=30 -> 30."""
    y1, m1, day1 = d1.year, d1.month, d1.day
    y2, m2, day2 = d2.year, d2.month, d2.day
    if day1 == 31:
        day1 = 30
    if day2 == 31 and day1 >= 30:
        day2 = 30
    return ((y2 - y1) * 360 + (m2 - m1) * 30 + (day2 - day1)) / 360.0


def year_frac(d1: Date, d2: Date, basis: str = "ACT/365F") -> float:
    basis = basis.upper()
    if basis in ("ACT/365F", "ACT/365"):
        return (d2 - d1).days / 365.0
    if basis == "ACT/360":
        return (d2 - d1).days / 360.0
    if basis in ("30/360", "30/360US", "BOND"):
        return thirty360(d1, d2)
    if basis in ("ACT/ACT", "ACT/365.25"):
        return (d2 - d1).days / 365.25
    raise ValueError(f"unknown day-count basis {basis}")


def coupon_dates(maturity: Date, freq: int, settle: Date) -> tuple[Date, list[Date]]:
    """Regular schedule rolled back from maturity.

    Returns (previous coupon date on/before settle, [future coupon dates after settle ... maturity]).
    Short or long first coupons are not modelled; the period containing settle is treated as regular.
    """
    if settle >= maturity:
        raise ValueError("settlement on or after maturity")
    step = 12 // freq
    future: list[Date] = []
    k = 0
    d = maturity
    while d > settle:
        future.append(d)
        k += 1
        d = add_months(maturity, -step * k)
    future.reverse()
    return d, future


def imm_dates_between(start: Date, end: Date) -> list[Date]:
    """Unadjusted CDS roll dates (20 Mar/Jun/Sep/Dec) strictly after start, up to and including end."""
    out = []
    y = start.year
    while True:
        for m in (3, 6, 9, 12):
            d = dt.date(y, m, 20)
            if d > end:
                return out
            if d > start:
                out.append(d)
        y += 1


def previous_imm(d: Date) -> Date:
    """Latest unadjusted IMM date (20 Mar/Jun/Sep/Dec) on or before d."""
    cands = [dt.date(y, m, 20) for y in (d.year - 1, d.year) for m in (3, 6, 9, 12)]
    return max(c for c in cands if c <= d)


def years_between(d1: Date, d2: Date) -> float:
    return (d2 - d1).days / 365.0


def as_dates(xs: Iterable) -> list[Date]:
    return [to_date(x) for x in xs]
