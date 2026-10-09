"""Annuitant mortality on the 2012 IAM basis with Projection Scale G2.

A GMWB for Life pays for as long as the owner lives, so the horizon is set by mortality, not
by the contract. Population mortality would understate the guarantee badly, because people
who buy lifetime income guarantees live longer than the general population - that is what
they are buying. The tables are therefore the SOA 2012 Individual Annuity Mortality tables
(table identities 2581 and 2582 for the Basic tables, 2585 and 2586 for the Period tables
carrying the Life Actuarial Task Force margins) projected with Scale G2, identities 2583 and
2584, which is the basis the NAIC adopted for individual annuity valuation:

    q(x, calendar year) = q_2012(x) * (1 - G2_x) ** (calendar year - 2012)

The Basic tables are the best-estimate basis and are what the economic valuation uses. The
Period tables carry the regulatory margins and sit closer to the basis a reported liability
would use, so the GAAP lens runs on those and the difference between the two is one of the
measurable wedges between the economic and the reported number.

Survival is blended across sexes at the survival-probability level, not at the mortality-rate
level. A fifty-fifty book is two populations; averaging the rates first and then compounding
gives the wrong expected number of payments, and the error grows with duration, which is
exactly where this liability's value is.

The expanded scope needs many cohorts at once, so the interface is a matrix: give it a vector
of attained ages and it returns survival and year-of-death probabilities for all of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import paths

BASE_YEAR = 2012
MAX_TABLE_AGE = 120
# Below this age the SOA export has holes and no cohort in this book ever reaches them. At or
# above it a missing rate is a defective download rather than a feature of the table.
FIRST_USED_AGE = 40


@dataclass(frozen=True)
class MortalityTable:
    """The raw tables, loaded once and shared across every cohort."""

    qx_male: np.ndarray
    qx_female: np.ndarray
    g2_male: np.ndarray
    g2_female: np.ndarray
    basis: str

    def rates(self, attained_ages, valuation_year: int, n_years: int, male_weight: float):
        """Blended survival and year-of-death probabilities.

        ``attained_ages`` is one age per cohort. Returns two arrays of shape
        (n_cohorts, n_years): the probability of being alive at the end of policy year k, and
        the probability of dying during policy year k. Both are measured from the valuation
        date, so they already account for the cohort's age today rather than at issue.

        Deaths are what the death benefit is paid on and survival is what the withdrawal
        guarantee is paid on, so both come out of one pass.
        """
        ages = np.asarray(attained_ages, dtype=int)
        if ages.ndim != 1:
            raise ValueError("attained_ages must be one-dimensional")
        if not 0.0 <= male_weight <= 1.0:
            raise ValueError("male_weight must be in [0, 1]")
        if n_years < 1:
            raise ValueError("n_years must be at least 1")

        survival = np.zeros((ages.size, n_years))
        deaths = np.zeros((ages.size, n_years))
        for qx, g2, weight in (
            (self.qx_male, self.g2_male, male_weight),
            (self.qx_female, self.g2_female, 1.0 - male_weight),
        ):
            if weight == 0.0:
                continue
            # Attained age in year k is age + k, and the improvement runs from 2012 to the
            # calendar year that policy year ends in.
            grid_ages = np.minimum(ages[:, None] + np.arange(n_years)[None, :], MAX_TABLE_AGE)
            improvement_years = valuation_year + np.arange(1, n_years + 1)[None, :] - BASE_YEAR
            annual_q = np.minimum(
                qx[grid_ages] * (1.0 - g2[grid_ages]) ** improvement_years, 1.0
            )
            alive_start = np.concatenate(
                [np.ones((ages.size, 1)), np.cumprod(1.0 - annual_q, axis=1)[:, :-1]], axis=1
            )
            survival += weight * alive_start * (1.0 - annual_q)
            deaths += weight * alive_start * annual_q
        return survival, deaths


def load(basis: str = "basic", path=None) -> MortalityTable:
    """Read the SOA table export committed in data/raw."""
    if basis not in ("basic", "period"):
        raise ValueError("basis must be 'basic' or 'period'")

    frame = pd.read_csv(path or paths.MORTALITY_TABLE).set_index("age")
    frame = frame.reindex(range(0, MAX_TABLE_AGE + 1))

    prefix = "iam_basic" if basis == "basic" else "iam_period"
    qx_male = _published_rates(frame[f"{prefix}_male"], f"{prefix}_male")
    qx_female = _published_rates(frame[f"{prefix}_female"], f"{prefix}_female")

    # Scale G2 stops at age 105 and has already trended to zero there - 0.000 at 104 and 105
    # for both sexes - so holding no improvement above it continues the published trend rather
    # than inventing a flat spot.
    g2_male = _fill_zero(frame["g2_male"].to_numpy(dtype=float))
    g2_female = _fill_zero(frame["g2_female"].to_numpy(dtype=float))

    return MortalityTable(
        qx_male=qx_male, qx_female=qx_female,
        g2_male=g2_male, g2_female=g2_female, basis=basis,
    )


def _published_rates(column: pd.Series, name: str) -> np.ndarray:
    """Rates at and above ``FIRST_USED_AGE`` as published, with nothing carried into a hole.

    The export leaves a handful of young ages blank - 9 to 11 on the Basic female table, 8 to
    12 on the Period female - and those get the previous age's rate, since no cohort here is
    ever nine years old. Higher up, a blank is a bad download, and filling one substitutes a
    materially lighter rate while still looking like data: delete age 97 from the male Basic
    column and a carry-forward hands back 0.2192 against the published 0.2386, which is four
    per cent of that cohort's remaining lifetime at the age where the life annuity is paying.

    So the check reads the column the loader was given, before anything touches it. That order
    is the point. It used to run on the filled copy, against a fill with no limit on how far it
    would reach, which left it able to fire only on a column that was blank from age zero - a
    guard that `docs/data_sources.md` advertised and that could not catch the case it named.
    """
    values = column.to_numpy(dtype=float)
    used = values[FIRST_USED_AGE:]
    if np.any(~np.isfinite(used)):
        blanks = FIRST_USED_AGE + np.flatnonzero(~np.isfinite(used))
        raise ValueError(
            f"{name} has no rate at age {blanks[0]} ({blanks.size} blank at or above "
            f"{FIRST_USED_AGE}); repair the export rather than filling it"
        )
    if np.any(used <= 0):
        bad = FIRST_USED_AGE + np.flatnonzero(used <= 0)
        raise ValueError(f"{name} is non-positive at age {bad[0]}")

    out = values.copy()
    last = np.nan
    for age in range(FIRST_USED_AGE):
        if np.isfinite(out[age]):
            last = out[age]
        else:
            out[age] = last
    return out


def _fill_zero(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    out[~np.isfinite(out)] = 0.0
    return out
