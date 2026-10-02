"""Annuitant mortality: the table, the improvement scale, and how the sexes are blended.

Every property here is one an error would quietly survive. A mortality table that loads the
wrong column, improves in the wrong direction, or blends rates instead of survival produces a
liability that looks entirely reasonable and is wrong by years of expected lifetime, which on a
forty-five-year guarantee is the difference between a margin and a loss.
"""

from __future__ import annotations

import numpy as np

from tests.checks import approx, raises
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
