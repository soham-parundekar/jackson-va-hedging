"""Curve bootstrap tests."""

from __future__ import annotations

import numpy as np

from tests.checks import approx, raises

from gmwb import market
from gmwb.curves import ParCurveBuilder, ZeroCurve, bootstrap, flat_curve

TENORS = [1, 2, 3, 5, 7, 10, 20, 30]


def test_par_bonds_reprice_to_par():
    """A bootstrap is only correct if the par bonds it was built from price back to one."""
    yields = np.array([0.0348, 0.0347, 0.0355, 0.0373, 0.0394, 0.0418, 0.0479, 0.0484])
    curve = bootstrap(TENORS, yields)
    for tenor, quoted in zip(TENORS, yields):
        assert curve.par_equivalent(tenor) == approx(quoted, abs=1e-12)


def test_flat_par_curve_gives_the_right_zero():
    """A flat 4% semiannual par curve implies a continuously compounded zero of
    2*ln(1.02), and it should hold at every maturity."""
    curve = bootstrap([1, 30], np.array([0.04, 0.04]))
    expected = 2.0 * np.log(1.02)
    for tenor in (0.5, 1.0, 5.0, 17.5, 30.0):
        assert curve.zero(tenor) == approx(expected, abs=1e-10)


def test_discount_factors_fall_with_maturity():
    yields = np.array([0.02, 0.025, 0.028, 0.032, 0.035, 0.038, 0.042, 0.043])
    curve = bootstrap(TENORS, yields)
    times = np.arange(0.5, 30.5, 0.5)
    factors = curve.discount(times)
    assert np.all(np.diff(factors) < 0)
    assert np.all(factors > 0)


def test_forward_rates_rebuild_the_zero_curve():
    yields = np.array([0.02, 0.025, 0.028, 0.032, 0.035, 0.038, 0.042, 0.043])
    curve = bootstrap(TENORS, yields)
    grid = np.arange(1.0, 31.0)
    forwards = np.array([curve.forward(t - 1, t) for t in grid])
    rebuilt = np.cumsum(forwards) / grid
    assert np.allclose(rebuilt, curve.zero(grid), atol=1e-12)


def test_parallel_shift_moves_every_zero_the_same_way():
    yields = np.array([0.0348, 0.0347, 0.0355, 0.0373, 0.0394, 0.0418, 0.0479, 0.0484])
    builder = ParCurveBuilder(TENORS, yields)
    base = builder.build()
    up = builder.build(shift_bp=100)
    down = builder.build(shift_bp=-100)
    times = np.array([1.0, 5.0, 10.0, 30.0])
    assert np.all(up.zero(times) > base.zero(times))
    assert np.all(down.zero(times) < base.zero(times))
    # Shifting par yields by 100bp moves the zero curve by close to but not exactly
    # 100bp, which is the reason the shock is applied to par yields and re-bootstrapped.
    moved = (up.zero(times) - base.zero(times)) * 10000
    assert np.all((moved > 95) & (moved < 106))


def test_shift_is_applied_to_par_yields_not_zeros():
    yields = np.array([0.02, 0.025, 0.028, 0.032, 0.035, 0.038, 0.042, 0.043])
    builder = ParCurveBuilder(TENORS, yields)
    shifted = builder.build(shift_bp=50)
    for tenor, quoted in zip(TENORS, yields):
        assert shifted.par_equivalent(tenor) == approx(quoted + 0.005, abs=1e-12)


def test_rejects_bad_inputs():
    with raises(ValueError):
        ZeroCurve(tenors=np.array([2.0, 1.0]), zero_rates=np.array([0.02, 0.03]))
    with raises(ValueError):
        ZeroCurve(tenors=np.array([0.0, 1.0]), zero_rates=np.array([0.02, 0.03]))
    with raises(ValueError):
        ParCurveBuilder([1, 2], np.array([0.02, np.nan]))
    with raises(ValueError):
        flat_curve(0.03).forward(5.0, 5.0)


def test_every_date_in_the_sample_bootstraps():
    """The whole history has to bootstrap, not just a convenient date. Deeply inverted
    curves are the case that breaks a careless implementation."""
    frame = market.load_panel()
    dates = market.equity_dates(frame)
    worst = 0.0
    for date in dates[::25]:
        row = frame.loc[date]
        yields = np.array([row[market.PAR_YIELD_COLUMNS[t]] for t in TENORS]) / 100.0
        curve = bootstrap(TENORS, yields)
        errors = [abs(curve.par_equivalent(t) - y) for t, y in zip(TENORS, yields)]
        worst = max(worst, max(errors))
    assert worst < 1e-10
