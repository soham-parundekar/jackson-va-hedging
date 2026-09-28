"""Annuitant mortality on the 2012 IAM basis with Projection Scale G2.

A GMWB for Life pays for as long as the owner lives, so the horizon is set by
mortality, not by the contract. Using population mortality here would understate
the guarantee, because people who buy lifetime income guarantees live longer than
the general population. The tables used are therefore the SOA 2012 Individual
Annuity Mortality tables (SOA MORT table identities 2581/2582 for the Basic
tables, 2585/2586 for the Period tables with LATF margins) projected with Scale G2
(2583/2584), which is the basis the NAIC adopted for individual annuity valuation.

    q(x, calendar_year) = q_2012(x) * (1 - G2_x) ** (calendar_year - 2012)

The Basic tables are the best-estimate basis and are what the economic valuation
uses. The Period tables carry the regulatory margins and are closer to the basis a
reported liability would sit on, so they are available as an alternative.

Survival is blended across sexes at the survival-probability level rather than the
mortality-rate level: a 50/50 book is two populations, and averaging the rates
first would give the wrong expected number of payments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import paths

BASE_YEAR = 2012
MAX_TABLE_AGE = 120


@dataclass(frozen=True)
class MortalityBasis:
    """Generational annuitant mortality for one issue age and sex mix."""

    issue_age: int
    valuation_year: int
    male_weight: float
    qx_male: np.ndarray        # indexed by attained age, 0..MAX_TABLE_AGE
    qx_female: np.ndarray
    g2_male: np.ndarray
    g2_female: np.ndarray

    def survival(self, n_years: int) -> np.ndarray:
        """Probability of being alive at the end of each of the next n years.

        Element k-1 of the result is the probability of surviving from the valuation
        date to policy anniversary k. Anniversary k falls in calendar year
        ``valuation_year + k``, and the mortality rate applied over the year ending
        there is the rate for attained age ``issue_age + k - 1``.
        """
        if n_years < 1:
            raise ValueError("n_years must be at least 1")

        survival_by_sex = []
        for qx, g2, weight in (
            (self.qx_male, self.g2_male, self.male_weight),
            (self.qx_female, self.g2_female, 1.0 - self.male_weight),
        ):
            if weight == 0.0:
                survival_by_sex.append(None)
                continue
            rates = np.empty(n_years)
            for k in range(1, n_years + 1):
                age = min(self.issue_age + k - 1, MAX_TABLE_AGE)
                years_of_improvement = self.valuation_year + k - BASE_YEAR
                improvement = (1.0 - g2[age]) ** years_of_improvement
                rates[k - 1] = min(qx[age] * improvement, 1.0)
            survival_by_sex.append(np.cumprod(1.0 - rates))

        blended = np.zeros(n_years)
        for path, weight in zip(survival_by_sex, (self.male_weight, 1.0 - self.male_weight)):
            if path is not None:
                blended += weight * path
        return blended

    def survival_at(self, times) -> np.ndarray:
        """Survival probability at arbitrary times measured in years from the
        valuation date.

        An in-force policy is almost never valued on its anniversary, so the first
        cash flow falls a fraction of a year out. Within a policy year the force of
        mortality is held constant, which makes survival log-linear between integer
        durations.
        """
        times = np.asarray(times, dtype=float)
        if np.any(times <= 0):
            raise ValueError("times must be positive")
        integer_grid = np.arange(0, int(np.ceil(times.max())) + 1)
        annual = np.concatenate(([1.0], self.survival(max(integer_grid[-1], 1))))
        log_survival = np.log(annual[: integer_grid.size])
        return np.exp(np.interp(times, integer_grid, log_survival))


def load(
    issue_age: int,
    valuation_year: int,
    male_weight: float = 0.5,
    table: str = "basic",
    path=None,
) -> MortalityBasis:
    """Build a mortality basis from the SOA table export in data/raw."""
    if table not in ("basic", "period"):
        raise ValueError("table must be 'basic' or 'period'")
    if not 0.0 <= male_weight <= 1.0:
        raise ValueError("male_weight must be in [0, 1]")

    frame = pd.read_csv(path or paths.MORTALITY_TABLE).set_index("age")
    frame = frame.reindex(range(0, MAX_TABLE_AGE + 1))

    prefix = "iam_basic" if table == "basic" else "iam_period"
    qx_male = _fill_forward(frame[f"{prefix}_male"].to_numpy(dtype=float))
    qx_female = _fill_forward(frame[f"{prefix}_female"].to_numpy(dtype=float))

    # Scale G2 stops at age 105; the published scale has already trended to zero
    # there, so no improvement is applied above it.
    g2_male = _fill_zero(frame["g2_male"].to_numpy(dtype=float))
    g2_female = _fill_zero(frame["g2_female"].to_numpy(dtype=float))

    for name, arr in (("qx_male", qx_male), ("qx_female", qx_female)):
        if np.any(~np.isfinite(arr[40:])) or np.any(arr[40:] <= 0):
            raise ValueError(f"{name} has gaps or non-positive rates above age 40")

    return MortalityBasis(
        issue_age=int(issue_age),
        valuation_year=int(valuation_year),
        male_weight=float(male_weight),
        qx_male=qx_male,
        qx_female=qx_female,
        g2_male=g2_male,
        g2_female=g2_female,
    )


def _fill_forward(values: np.ndarray) -> np.ndarray:
    """Carry the last published rate forward. The SOA export leaves a few young
    female ages blank; ages above 40 are complete, which is all the valuation uses."""
    out = values.copy()
    last = np.nan
    for i, v in enumerate(out):
        if np.isfinite(v):
            last = v
        else:
            out[i] = last
    return out


def _fill_zero(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    out[~np.isfinite(out)] = 0.0
    return out
