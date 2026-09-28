"""Volatility term structure tests.

Two properties have to hold everywhere rather than at a convenient point. Forward
variance must stay positive, including when a volatility spike puts the observed
short-dated level far above the long-run level, because that is the state where a
carelessly specified term structure implies negative forward variance. And a move in the
short-dated level must decay with maturity, because a curve that propagates a spike
undamped to twenty years produces a vega several times too large.
"""

from __future__ import annotations

import numpy as np

from tests.checks import approx, raises

from gmwb.volatility import VolTermStructure, decay_from_grading, realised_vol


def test_total_variance_is_monotone_everywhere():
    grid = np.arange(0.02, 61.0, 0.02)
    for front in np.arange(0.05, 1.05, 0.05):
        for long_run in np.arange(0.05, 0.45, 0.05):
            vol = VolTermStructure(front_level=front, long_run_level=long_run)
            assert np.all(np.diff(vol.total_variance(grid)) > 0), (front, long_run)


def test_closed_form_matches_quadrature():
    for front, long_run in ((0.18, 0.20), (0.67, 0.19), (0.10, 0.30)):
        vol = VolTermStructure(front_level=front, long_run_level=long_run)
        for maturity in (0.08, 0.3, 4.0, 10.0, 25.0, 45.0):
            grid = np.linspace(0.0, maturity, 200001)
            numeric = np.trapezoid(vol.forward_vol(grid) ** 2, grid)
            assert vol.total_variance(maturity) == approx(numeric, rel=1e-8)


def test_spot_starts_at_the_front_level_and_ends_at_the_long_run_level():
    vol = VolTermStructure(front_level=0.55, long_run_level=0.18)
    assert float(vol.spot_vol(1e-6)) == approx(0.55, rel=1e-4)
    grid = np.array([0.1, 1.0, 5.0, 10.0, 30.0])
    assert np.all(np.diff(vol.spot_vol(grid)) < 0)

    # Forward volatility converges on the long-run level exponentially. Spot volatility is
    # an average from zero, so it keeps a trace of the front level and converges only as one
    # over maturity: still 21.0% at sixty years against a long-run 18%.
    assert float(vol.forward_vol(60.0)) == approx(0.18, rel=1e-9)
    assert 0.18 < float(vol.spot_vol(60.0)) < 0.22
    assert float(vol.spot_vol(20_000.0)) == approx(0.18, rel=1e-3)


def test_a_spike_decays_with_maturity():
    """The defect this replaced: a fifty point move in the three month index must not move
    the twenty year volatility by fifty points."""
    calm = VolTermStructure.from_implied({0.25: 0.20}, long_run_level=0.19)
    spike = VolTermStructure.from_implied({0.25: 0.70}, long_run_level=0.19)
    short_move = spike.spot_vol(0.25) - calm.spot_vol(0.25)
    long_move = spike.spot_vol(20.0) - calm.spot_vol(20.0)
    assert short_move == approx(0.50, rel=1e-6)
    assert 0.0 < long_move < 0.15


def test_fitting_reproduces_the_observed_tenor_exactly():
    for implied in (0.12, 0.20, 0.35, 0.70):
        for long_run in (0.15, 0.19, 0.25):
            vol = VolTermStructure.from_implied({0.25: implied}, long_run_level=long_run)
            assert float(vol.spot_vol(0.25)) == approx(implied, rel=1e-12)


def test_decay_matches_the_grading_statement():
    """The decay is pinned to Jackson's description rather than fitted: the front level
    should have graded all but the residual share of the way by grade_to years."""
    kappa = decay_from_grading(10.0, 0.02)
    assert kappa == approx(-np.log(0.02) / 10.0, rel=1e-12)
    vol = VolTermStructure(front_level=0.60, long_run_level=0.20, decay=kappa)
    remaining = (vol.forward_vol(10.0) ** 2 - 0.20**2) / (0.60**2 - 0.20**2)
    assert remaining == approx(0.02, rel=1e-9)


def test_shift_front_leaves_the_long_run_level_alone():
    vol = VolTermStructure(front_level=0.18, long_run_level=0.20)
    bumped = vol.shift_front(0.01)
    assert bumped.front_level == approx(0.19)
    assert bumped.long_run_level == approx(0.20)
    parallel = vol.shift(0.01)
    assert parallel.front_level == approx(0.19)
    assert parallel.long_run_level == approx(0.21)


def test_long_run_level_dominates_a_long_dated_liability():
    """For a forty year horizon, moving the long-run level has to matter more than moving
    the short-dated level. If it ever stops being true the curve has lost its decay."""
    vol = VolTermStructure(front_level=0.18, long_run_level=0.19)
    base = vol.total_variance(40.0)
    front_bump = vol.shift_front(0.01).total_variance(40.0) - base
    long_bump = (
        VolTermStructure(front_level=0.18, long_run_level=0.20).total_variance(40.0) - base
    )
    assert long_bump > 5 * front_bump


def test_step_variances_sum_to_total_variance():
    vol = VolTermStructure(front_level=0.20, long_run_level=0.25)
    times = np.arange(1.0, 46.0)
    steps = vol.step_variances(times)
    assert steps.sum() == approx(float(vol.total_variance(times[-1])), rel=1e-12)
    assert np.all(steps > 0)


def test_step_variances_handle_a_short_first_interval():
    vol = VolTermStructure(front_level=0.20, long_run_level=0.25)
    times = 0.25 + np.arange(0.0, 40.0)
    steps = vol.step_variances(times)
    assert steps[0] == approx(float(vol.total_variance(0.25)), rel=1e-12)
    assert np.all(steps > 0)


def test_step_variances_stay_positive_through_a_spike():
    vol = VolTermStructure.from_implied({0.25: 0.70}, long_run_level=0.19)
    steps = vol.step_variances(0.3 + np.arange(0.0, 45.0))
    assert np.all(steps > 0)


def test_step_variances_reject_bad_grids():
    vol = VolTermStructure(front_level=0.20, long_run_level=0.25)
    with raises(ValueError):
        vol.step_variances(np.array([0.0, 1.0]))
    with raises(ValueError):
        vol.step_variances(np.array([2.0, 1.0]))


def test_realised_vol_recovers_a_known_diffusion():
    rng = np.random.default_rng(7)
    sigma = 0.25
    steps = rng.standard_normal(200_000) * sigma / np.sqrt(252)
    prices = 100 * np.exp(np.cumsum(steps))
    assert realised_vol(prices) == approx(sigma, rel=0.02)


def test_rejects_bad_parameters():
    with raises(ValueError):
        VolTermStructure(front_level=0.0, long_run_level=0.2)
    with raises(ValueError):
        VolTermStructure(front_level=0.2, long_run_level=0.2, decay=0.0)
    with raises(ValueError):
        decay_from_grading(0.0, 0.02)
    with raises(ValueError):
        decay_from_grading(10.0, 1.5)
    with raises(ValueError):
        VolTermStructure.from_implied({}, long_run_level=0.2)
    with raises(ValueError):
        VolTermStructure.from_implied({0.25: -0.1}, long_run_level=0.2)
