"""Annuitant mortality: the table, the improvement scale, and how the sexes are blended.

Every property here is one an error would quietly survive. A mortality table that loads the
wrong column, improves in the wrong direction, or blends rates instead of survival produces a
liability that looks entirely reasonable and is wrong by years of expected lifetime, which on a
forty-five-year guarantee is the difference between a margin and a loss.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from tests.checks import approx, raises
from vahedge import paths
from vahedge.liability import mortality

AGE = np.array([70])


def survival(basis: str = "basic", age: int = 70, year: int = 2025, years: int = 45,
             male_weight: float = 0.5) -> np.ndarray:
    return mortality.load(basis).rates(np.array([age]), year, years, male_weight)[0][0]


def test_survival_is_a_decreasing_probability():
    curve = survival()
    assert curve.shape == (45,)
    assert np.all(curve > 0) and np.all(curve <= 1)
    assert np.all(np.diff(curve) < 0)


def test_deaths_and_survival_account_for_everyone():
    """Each year's death probability is the drop in survival over that year, exactly.

    The death benefit is paid on the deaths and the withdrawal guarantee on the survivors, so
    anything that falls between the two is a cash flow the projection never pays and never
    charges for.
    """
    alive, dead = mortality.load("basic").rates(AGE, 2025, 45, 0.5)
    opening = np.concatenate([[1.0], alive[0][:-1]])
    assert approx(opening - alive[0], abs=1e-12) == dead[0]
    assert float(dead[0].sum() + alive[0][-1]) == approx(1.0, abs=1e-12)


def test_improvement_scale_lengthens_life():
    """Thirteen more years of G2 improvement has to raise survival at every duration, not
    only on average."""
    early = survival(year=2012, years=40)
    later = survival(year=2025, years=40)
    assert np.all(later > early)
    assert later.sum() > early.sum() + 0.5


def test_period_table_carries_margins():
    """The Period table is the Basic table with the margins the Life Actuarial Task Force set,
    so it has to imply lighter mortality and a longer life. If this ever reversed, every
    reporting-basis figure in the project would be the wrong way round."""
    basic = survival("basic")
    period = survival("period")
    assert np.all(period >= basic)
    assert period.sum() > basic.sum()


def test_women_outlive_men_and_a_blend_sits_between():
    male = survival(years=40, male_weight=1.0)
    female = survival(years=40, male_weight=0.0)
    blend = survival(years=40, male_weight=0.5)
    assert np.all(female > male)
    assert np.all(blend > male) and np.all(blend < female)
    # Blending at the survival level rather than the rate level makes the blend exactly the
    # average of two single-sex books, which is what a 50/50 book of lives is.
    assert approx(0.5 * (male + female), abs=1e-12) == blend


def test_blending_rates_would_give_a_different_answer():
    """Guards the design choice. If the implementation is ever changed to average the annual
    mortality rates before compounding them, this catches it."""
    male = survival(years=40, male_weight=1.0)
    female = survival(years=40, male_weight=0.0)
    blend = survival(years=40, male_weight=0.5)

    def annual_rates(curve):
        return 1.0 - np.exp(np.diff(np.log(np.concatenate(([1.0], curve)))))

    rate_blended = np.cumprod(1.0 - 0.5 * (annual_rates(male) + annual_rates(female)))
    assert not np.allclose(blend, rate_blended, atol=1e-6)
    # Averaging rates understates survival, because survival is convex in the rate.
    assert blend[-1] > rate_blended[-1]


def test_older_lives_have_shorter_horizons():
    lifetimes = [survival(age=age, years=50).sum() for age in (60, 70, 80)]
    assert lifetimes[0] > lifetimes[1] > lifetimes[2]


def test_several_cohorts_come_back_in_the_order_they_were_asked_for():
    """The book is valued as parallel arrays, so a row that silently reordered would attach
    one cohort's mortality to another's account value."""
    ages = np.array([65, 70, 80, 70])
    alive, _ = mortality.load("basic").rates(ages, 2025, 30, 0.5)
    assert alive.shape == (4, 30)
    assert approx(alive[1], abs=1e-15) == alive[3]
    assert alive[0].sum() > alive[1].sum() > alive[2].sum()


def test_the_table_runs_out_rather_than_wrapping_round():
    """Past the end of the table the oldest row is held, so a projection that outlives its
    data dies out instead of coming back to life at a young age."""
    alive, _ = mortality.load("basic").rates(np.array([110]), 2025, 30, 0.5)
    assert np.all(np.diff(alive[0]) <= 0)
    assert alive[0][-1] < 1e-6


def _export_with(**edits) -> str:
    """The committed SOA export with individual cells blanked, written to a scratch file."""
    frame = pd.read_csv(paths.MORTALITY_TABLE).set_index("age")
    for column, ages in edits.items():
        frame.loc[list(ages), column] = np.nan
    folder = tempfile.mkdtemp()
    target = Path(folder) / "soa_edited.csv"
    frame.reset_index().to_csv(target, index=False)
    return str(target)


def test_a_missing_rate_in_the_range_the_valuation_uses_stops_the_loader():
    """The guard has to read the export, not a repaired copy of it.

    Written because it did the second thing. An unlimited carry-forward ran first and the check
    for gaps ran on its output, so a blank at age 97 arrived at the check already wearing age
    96's rate and the loader proceeded, four per cent light on mortality at an age where the
    life annuity is the whole of the liability. The only column that could still fail the check
    was one blank from age zero, which is not a failure mode a CSV export has.
    """
    for column in ("iam_basic_male", "iam_period_female"):
        with raises(ValueError, match="age 97"):
            mortality.load("basic" if "basic" in column else "period",
                           path=_export_with(**{column: [97]}))

    # Several blanks report the count, so a reader can tell one bad row from a truncated file.
    with raises(ValueError, match="(3 blank"):
        mortality.load("basic", path=_export_with(iam_basic_male=[80, 97, 104]))


def test_the_young_ages_the_export_leaves_blank_are_still_filled():
    """The fill is kept, narrowed, not removed. Those ages are real holes in the SOA export and
    no cohort in this book is ever nine years old, so refusing them would reject the only
    mortality file the repository has."""
    table = mortality.load("period")
    assert np.all(np.isfinite(table.qx_female[8:13]))
    assert float(table.qx_female[8]) == approx(float(table.qx_female[7]), abs=0.0)
    # And the committed export loads on both bases, which is the regression this pairs with.
    for basis in ("basic", "period"):
        assert np.all(np.isfinite(mortality.load(basis).qx_female[40:]))


def test_rejects_bad_arguments():
    table = mortality.load("basic")
    with raises(ValueError, match="male_weight"):
        table.rates(AGE, 2025, 10, 1.5)
    with raises(ValueError, match="n_years"):
        table.rates(AGE, 2025, 0, 0.5)
    with raises(ValueError, match="one-dimensional"):
        table.rates(np.array([[70]]), 2025, 10, 0.5)
    with raises(ValueError):
        mortality.load("nonsense")
