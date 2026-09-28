"""Annuitant mortality tests."""

from __future__ import annotations

import numpy as np

from tests.checks import approx, raises

from gmwb import mortality


def test_survival_is_a_decreasing_probability():
    basis = mortality.load(70, 2025, 0.5)
    survival = basis.survival(45)
    assert survival.shape == (45,)
    assert np.all(survival > 0) and np.all(survival <= 1)
    assert np.all(np.diff(survival) < 0)


def test_improvement_scale_lengthens_life():
    """Projecting thirteen years of G2 improvement has to raise survival at every
    duration, not just on average."""
    early = mortality.load(70, 2012, 0.5).survival(40)
    later = mortality.load(70, 2025, 0.5).survival(40)
    assert np.all(later > early)
    assert later.sum() > early.sum() + 0.5


def test_period_table_carries_margins():
    """The Period table is the Basic table with the regulatory margins added, so it must
    imply lighter mortality and a longer expected lifetime."""
    basic = mortality.load(70, 2025, 0.5, table="basic").survival(45)
    period = mortality.load(70, 2025, 0.5, table="period").survival(45)
    assert np.all(period >= basic)
    assert period.sum() > basic.sum()


def test_women_outlive_men_and_a_blend_sits_between():
    male = mortality.load(70, 2025, 1.0).survival(40)
    female = mortality.load(70, 2025, 0.0).survival(40)
    blend = mortality.load(70, 2025, 0.5).survival(40)
    assert np.all(female > male)
    assert np.all(blend > male) and np.all(blend < female)
    # Blending at the survival level, not the rate level, means the blend is exactly the
    # average of the two survival curves.
    assert np.allclose(blend, 0.5 * (male + female))


def test_blending_rates_would_give_a_different_answer():
    """Guards the design choice. If the implementation is ever changed to average mortality
    rates before compounding them, this catches it."""
    basis = mortality.load(70, 2025, 0.5)
    male = mortality.load(70, 2025, 1.0).survival(40)
    female = mortality.load(70, 2025, 0.0).survival(40)

    def annual_rates(survival):
        return 1.0 - np.exp(np.diff(np.log(np.concatenate(([1.0], survival)))))

    rate_blended = np.cumprod(1.0 - 0.5 * (annual_rates(male) + annual_rates(female)))
    assert not np.allclose(basis.survival(40), rate_blended, atol=1e-6)
    # Averaging rates understates survival, because survival is convex in the rate.
    assert basis.survival(40)[-1] > rate_blended[-1]


def test_fractional_times_interpolate_between_integers():
    basis = mortality.load(70, 2025, 0.5)
    annual = basis.survival(10)
    fractional = basis.survival_at(np.arange(1.0, 11.0))
    assert np.allclose(fractional, annual, atol=1e-12)

    half = basis.survival_at([0.5])[0]
    assert annual[0] < half < 1.0
    # Constant force of mortality inside the year makes survival log-linear.
    assert half == approx(np.sqrt(annual[0]), rel=1e-12)


def test_older_lives_have_shorter_horizons():
    lifetimes = [mortality.load(age, 2025, 0.5).survival(50).sum() for age in (60, 70, 80)]
    assert lifetimes[0] > lifetimes[1] > lifetimes[2]


def test_rejects_bad_arguments():
    with raises(ValueError):
        mortality.load(70, 2025, 1.5)
    with raises(ValueError):
        mortality.load(70, 2025, 0.5, table="nonsense")
    with raises(ValueError):
        mortality.load(70, 2025, 0.5).survival(0)
    with raises(ValueError):
        mortality.load(70, 2025, 0.5).survival_at([0.0, 1.0])
