"""The bootstrap: block length, the draws, and whether a resampled day stays in one piece.

The whole value of resampling history rather than simulating it is that a day arrives intact -
its return, its volatility, its curve and its credit spread together. That is one index
arithmetic error away from being lost, and losing it would not show up as an exception. It would
show up as a scenario set with historical marginals and no co-movement, which is exactly the
thing a model could have produced more cheaply.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tests.checks import approx, raises
from vahedge.market.curves import ZeroCurve
from vahedge.market.scenarios import (
    DailyPath,
    block_length,
    resample,
    stationary_bootstrap,
    sub_account_path,
)
from vahedge.market.simulate import SubAccountMix

# Jackson's disclosed sub-account split, near enough: the resampler rebuilds the fund from it,
# so the weights have to be a real allocation rather than all equity.
_MIX = SubAccountMix(equity=0.75, bond=0.15, balanced=0.0, money_market=0.10)


def _flat_curve(level: float) -> ZeroCurve:
    tenors = np.array([0.25, 1.0, 5.0, 10.0, 30.0])
    return ZeroCurve(tenors=tenors, zero_rates=np.full(tenors.size, level))


def _labelled_path(n: int = 60, seed: int = 4) -> DailyPath:
    """A path whose every state variable encodes its own day number.

    The returns are random so the compounding has something to get wrong, but the variance, the
    rate and the spread are day/1000 and the curve level is day/100. Anything resampled can then
    be read back as the day it came from, which is what the coherence tests need.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)
    day = np.arange(n, dtype=float)
    index = np.cumprod(np.concatenate([[100.0], np.exp(rng.normal(0.0, 0.01, n - 1))]))
    fund = np.cumprod(np.concatenate([[1.0], np.exp(rng.normal(0.0, 0.006, n - 1))]))
    return DailyPath(
        dates=dates, index=index, fund=fund,
        curves=[_flat_curve(d / 100.0) for d in day],
        zero_10y=day / 1000.0, implied_vol=0.2 + day / 1000.0, variance=day / 1000.0,
        cash_rate=day / 1000.0, own_credit_spread=0.02 + day / 1000.0, label="labelled",
    )


def test_the_block_length_of_white_noise_is_about_one():
    """No volatility clustering means no reason to draw blocks at all."""
    noise = np.random.default_rng(11).normal(0.0, 0.01, 4000)
    assert 0.9 < block_length(noise) < 2.5


def test_persistent_volatility_lengthens_the_block():
    """Squared returns that cluster are the whole reason the block is longer than a day."""
    rng = np.random.default_rng(12)
    scale = np.repeat(rng.choice([0.004, 0.025], size=200), 20)
    clustered = rng.normal(0.0, 1.0, scale.size) * scale
    assert block_length(clustered) > 4.0


def test_a_constant_series_does_not_divide_by_zero():
    assert block_length(np.full(500, 0.01)) == approx(1.0)


def test_the_draws_have_the_shape_asked_for_and_stay_in_range():
    path = _labelled_path(n=200)
    drawn = stationary_bootstrap(path, n_days=80, n_paths=5, seed=3)
    assert drawn.shape == (5, 80)
    assert drawn.min() >= 0
    # One fewer return than days, and the state is read at drawn+1, so the top index has to
    # leave a day above it.
    assert drawn.max() <= path.index.size - 2


def test_the_same_seed_draws_the_same_scenario_set():
    path = _labelled_path(n=200)
    first = stationary_bootstrap(path, n_days=50, n_paths=4, seed=8)
    second = stationary_bootstrap(path, n_days=50, n_paths=4, seed=8)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, stationary_bootstrap(path, 50, 4, seed=9))


def test_a_long_block_mostly_walks_forward_and_a_one_day_block_does_not():
    """The mean block length is the only parameter, so it had better do something."""
    path = _labelled_path(n=400)
    def share_consecutive(mean_block: float) -> float:
        drawn = stationary_bootstrap(path, n_days=300, n_paths=20, seed=5, mean_block=mean_block)
        return float(np.mean(np.diff(drawn, axis=1) == 1))
    assert share_consecutive(40.0) > 0.9
    assert share_consecutive(1.0) < 0.1


def test_a_history_too_short_to_bootstrap_is_refused():
    with raises(ValueError, match="not enough history"):
        stationary_bootstrap(_labelled_path(n=30), n_days=10, n_paths=2, seed=1)


def test_the_volatility_state_comes_from_the_day_its_return_came_from():
    """Return i is the move from day i to day i+1, so the variance beside it is day i+1's.

    The pairing is the whole point of resampling days rather than returns: a crash drawn out of
    March 2020 has to arrive with March 2020's variance.
    """
    path = _labelled_path()
    drawn = np.array([40, 3, 17, 3])
    out = resample(path, drawn, _MIX)
    expected_day = drawn + 1
    assert approx(expected_day / 1000.0) == out.variance[1:]
    assert approx(0.2 + expected_day / 1000.0) == out.implied_vol[1:]


def test_the_rate_side_stays_on_the_history_s_own_course():
    """Rates, the curve and the credit spread are levels, and reordering levels invents jumps.

    So they are not drawn at all: a bootstrap path of n days carries the first n days of the
    real rate path, whatever order the equity days came in.
    """
    path = _labelled_path()
    drawn = np.array([40, 3, 17, 3])
    out = resample(path, drawn, _MIX)
    assert approx(np.arange(5) / 1000.0) == out.cash_rate
    assert approx(np.arange(5) / 1000.0) == out.zero_10y
    assert approx(0.02 + np.arange(5) / 1000.0) == out.own_credit_spread
    assert approx(np.arange(5) / 100.0) == [float(c.zero(10.0)) for c in out.curves]


def test_the_opening_state_is_the_history_s_own_first_day():
    """Day zero is not drawn: the path has to start somewhere, and it starts where history did."""
    path = _labelled_path()
    out = resample(path, np.array([40, 3, 17]), _MIX)
    assert out.variance[0] == approx(0.0)
    assert out.own_credit_spread[0] == approx(0.02)
    assert float(out.curves[0].zero(10.0)) == approx(0.0)
    assert out.index[0] == approx(1.0)
    assert out.fund[0] == approx(1.0)


def test_the_index_compounds_exactly_the_returns_that_were_drawn():
    path = _labelled_path()
    drawn = np.array([40, 3, 17, 3, 11])
    out = resample(path, drawn, _MIX)
    steps = np.diff(np.log(path.index))[drawn]
    assert approx(np.concatenate([[1.0], np.exp(np.cumsum(steps))])) == out.index
    assert out.dates.size == drawn.size + 1
    assert len(out.curves) == drawn.size + 1


def test_the_dates_are_the_calendar_s_own_rather_than_the_drawn_ones():
    """Resampled dates would bunch anniversaries; the spacing has to stay a real calendar."""
    path = _labelled_path()
    out = resample(path, np.array([40, 3, 17]), _MIX)
    assert list(out.dates) == list(path.dates[:4])


def test_drawing_history_in_order_reproduces_history():
    """The identity draw is the one case with a known answer, so it is worth pinning."""
    path = _labelled_path()
    out = resample(path, np.arange(path.index.size - 1), _MIX)
    assert approx(path.index / path.index[0]) == out.index
    assert approx(path.variance) == out.variance
    rebuilt = sub_account_path(path.index / path.index[0], path.curves, path.cash_rate,
                               _MIX, path.year_fraction)
    assert approx(rebuilt) == out.fund


def test_a_path_with_no_credit_spread_resamples_without_one():
    """BAA10Y is optional in load_history, so the resampler cannot assume it is there."""
    path = _labelled_path()
    bare = DailyPath(
        dates=path.dates, index=path.index, fund=path.fund, curves=path.curves,
        zero_10y=path.zero_10y, implied_vol=path.implied_vol, variance=path.variance,
        cash_rate=path.cash_rate, own_credit_spread=None, label="bare",
    )
    out = resample(bare, np.array([5, 9, 2]), _MIX)
    assert out.own_credit_spread is None


def test_the_drift_control_puts_the_average_where_it_was_asked_to():
    """Re-centring is the arm of the experiment that removes the sample decade's bull market."""
    path = _labelled_path(n=300)
    ordered = np.arange(path.index.size - 1)
    days_per_year = (path.index.size - 1) / path.year_fraction[-1]
    for target in (0.04, 0.07, -0.02):
        out = resample(path, ordered, _MIX, annual_drift=target)
        realised = float(np.mean(np.diff(np.log(out.index)))) * days_per_year
        assert realised == approx(target, abs=1e-9)


def test_the_drift_shift_is_the_sample_s_and_not_each_path_s_own():
    """A path that drew a good decade has to stay a good decade relative to the others.

    Re-centring every path on the target individually would leave the ensemble with no spread
    in realised outcomes, and the spread is half of what the experiment measures.
    """
    path = _labelled_path(n=300)
    lucky = np.argsort(np.diff(np.log(path.index)))[-120:]
    unlucky = np.argsort(np.diff(np.log(path.index)))[:120]
    good = resample(path, lucky, _MIX, annual_drift=0.05)
    bad = resample(path, unlucky, _MIX, annual_drift=0.05)
    assert good.index[-1] > bad.index[-1]


def test_without_a_drift_the_returns_are_left_exactly_as_history_had_them():
    path = _labelled_path()
    drawn = np.array([7, 21, 2, 2])
    assert (approx(resample(path, drawn, _MIX).index)
            == resample(path, drawn, _MIX, annual_drift=None).index)


def test_a_re_drifted_fund_moves_by_less_than_the_index_because_it_is_not_all_equity():
    """The sub-account is rebuilt from the new index, so its sleeve mix does the scaling.

    Nothing here scales the fund by hand: the bond and money-market sleeves earn what they
    earned on those dates and only the equity sleeve carries the shift, so the fund has to move
    by the equity weight's share of it rather than one for one.
    """
    path = _labelled_path(n=300)
    ordered = np.arange(path.index.size - 1)
    base = resample(path, ordered, _MIX)
    shifted = resample(path, ordered, _MIX, annual_drift=0.20)
    index_shift = float(np.mean(np.diff(np.log(shifted.index)) - np.diff(np.log(base.index))))
    fund_shift = float(np.mean(np.diff(np.log(shifted.fund)) - np.diff(np.log(base.fund))))
    # As a share rather than a level: whether the shift is up or down depends on where this
    # short synthetic window's own mean happens to sit, and the rule being tested does not.
    share = fund_shift / index_shift
    assert approx(_MIX.equity_weight, rel=0.05) == share


def test_the_label_is_carried_so_a_scenario_can_be_identified_later():
    out = resample(_labelled_path(), np.array([1, 2, 3]), _MIX, label="path 17, re-centred")
    assert out.label == "path 17, re-centred"
