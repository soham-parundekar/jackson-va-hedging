"""The valuation engine, its Greeks, and the break-even fee.

These run on a deliberately small path count. What is being tested is not accuracy - the
convergence study does that - but the identities that have to hold whatever the noise: that the
attribution definition is the one in Note 6, that a shocked valuation reuses its base's draws,
that signs come out the way a written put requires, and that the break-even solve returns a fee
that actually zeroes the guarantee.
"""

from __future__ import annotations

import numpy as np

from tests.checks import approx, raises
from vahedge.liability import cohorts, mortality
from vahedge.liability import terms as terms_module
from vahedge.market.curves import ZeroCurve, fit_curve
from vahedge.market.heston_cos import HestonParameters
from vahedge.market.simulate import Correlations, SubAccountMix
from vahedge.valuation import breakeven, greeks
from vahedge.valuation.engine import MarketState, Valuer, _implied_attribution, _rewind_to_issue

PATHS = 2_000


def _state(**overrides):
    curve = fit_curve(ZeroCurve(tenors=np.arange(0.5, 30.5, 0.5),
                                zero_rates=np.linspace(0.035, 0.048, 60)))
    defaults = dict(
        curve=curve,
        heston=HestonParameters(v0=0.0199, kappa=4.80, theta=0.0470, xi=1.780, rho=-0.588),
        correlations=Correlations(-0.588, 0.106),
        mix=SubAccountMix(equity=0.7235, bond=0.0834, balanced=0.1832, money_market=0.0099),
        mean_reversion=0.2685,
        rate_vol=0.01141,
        valuation_year=2025,
    )
    defaults.update(overrides)
    return MarketState(**defaults)


def _small_book():
    """A three-by-two grid. Enough to exercise the aggregation, small enough to run in tests."""
    spec = cohorts.GridSpec(
        issue_ages=(60, 70), durations=(0, 6), gwb_over_av=(0.8, 1.2),
        age_weights=(0.5, 0.5), duration_weights=(0.5, 0.5), moneyness_weights=(0.5, 0.5),
        max_age=100,
    )
    core = terms_module.load()[("flex_gmwb", "single", "core")]
    return cohorts.build(spec, core, 0.0131, 0.0095, terms_module.DEATH_BENEFITS["basic"])


def _valuer():
    return Valuer(mortality.load("basic"), n_paths=PATHS, seed=99, cache_size=3)


# ---------------------------------------------------------------- the attributed fee


def test_attribution_is_claims_over_fees_capped_at_one():
    """Note 6's definition, written out. Where attributable fees cover projected claims the
    benefit starts at a fair value of zero; where they do not, the percentage caps at one and an
    MRB liability is recognised."""

    class FakeProjection:
        pv_total_claims = np.array([50.0, 200.0])
        pv_attributable_fees = np.array([100.0, 100.0])

    assert approx(np.array([0.5, 1.0])) == _implied_attribution(FakeProjection())


def test_attribution_with_no_fees_is_refused_rather_than_divided_by_zero():
    class FakeProjection:
        pv_total_claims = np.array([50.0])
        pv_attributable_fees = np.array([0.0])

    with raises(ValueError, match="no attributable fees"):
        _implied_attribution(FakeProjection())


def test_an_at_issue_book_prices_to_zero_at_its_own_attribution():
    """The definition's consequence: calibrate the attribution on a book at issue, value that
    same book with it, and the market risk benefit is zero. Any cohort whose fees fall short
    caps at one and leaves a positive liability, so the identity is one-sided."""
    book = _rewind_to_issue(_small_book())
    valuer, state = _valuer(), _state()
    attribution = _implied_attribution(
        valuer.value(book, state, attribution=np.ones(book.size)).projection
    )
    valuation = valuer.value(book, state, attribution=attribution)
    assert valuation.market_risk_benefit >= -1e-8
    uncapped = attribution < 1.0 - 1e-12
    if uncapped.all():
        assert valuation.market_risk_benefit == approx(0.0, abs=1e-6)


def test_rewinding_a_book_puts_every_cohort_back_at_premium():
    book = _small_book()
    at_issue = _rewind_to_issue(book)
    assert approx(book.premium_at_issue) == at_issue.account_value
    assert approx(book.premium_at_issue) == at_issue.benefit_base
    assert np.all(at_issue.years_since_issue == 0)
    assert approx(book.issue_age.astype(float)) == at_issue.attained_age.astype(float)
    # A cohort six years in force has six more years of projection ahead of it at issue.
    assert np.all(at_issue.projection_years == book.projection_years + book.years_since_issue)


# ---------------------------------------------------------------- common random numbers


def test_two_valuations_of_the_same_state_are_identical():
    valuer, book, state = _valuer(), _small_book(), _state()
    attribution = np.full(book.size, 0.8)
    first = valuer.value(book, state, attribution=attribution)
    second = valuer.value(book, state, attribution=attribution)
    assert first.market_risk_benefit == approx(second.market_risk_benefit, rel=1e-15)


def test_an_equity_shock_costs_no_extra_simulation():
    """Under Heston the return distribution does not depend on the index level, so shocking the
    index changes only the account value the paths start from."""
    valuer, book, state = _valuer(), _small_book(), _state()
    attribution = np.full(book.size, 0.8)
    valuer.value(book, state, attribution=attribution)
    before = valuer.simulations
    valuer.value(book, state, attribution=attribution, equity_shock=0.10)
    valuer.value(book, state, attribution=attribution, equity_shock=-0.10)
    assert valuer.simulations == before


def test_a_rate_shock_does_cost_a_simulation():
    valuer, book, state = _valuer(), _small_book(), _state()
    attribution = np.full(book.size, 0.8)
    valuer.value(book, state, attribution=attribution)
    before = valuer.simulations
    valuer.value(book, state.with_shocks(rate_shock_bp=100), attribution=attribution)
    assert valuer.simulations == before + 1


def test_the_path_cache_is_bounded():
    """An unbounded cache is tens of megabytes per market state, and a hedge backtest visits one
    per rebalance date."""
    valuer, book = Valuer(mortality.load("basic"), n_paths=500, cache_size=2), _small_book()
    attribution = np.full(book.size, 0.8)
    for shift in (0, 50, 100, 150):
        valuer.value(book, _state().with_shocks(rate_shock_bp=shift), attribution=attribution)
    assert len(valuer._paths) <= 2


def test_a_parallel_shift_moves_the_whole_curve_and_leaves_its_shape_alone():
    state = _state()
    shifted = state.with_shocks(rate_shock_bp=100)
    tenors = np.array([1.0, 5.0, 10.0, 30.0, 45.0])
    difference = shifted.curve.zero(tenors) - state.curve.zero(tenors)
    assert approx(np.full(5, 0.01), abs=1e-12) == difference
    forward_difference = (
        shifted.curve.instantaneous_forward(tenors) - state.curve.instantaneous_forward(tenors)
    )
    assert approx(np.full(5, 0.01), abs=1e-12) == forward_difference


# ---------------------------------------------------------------- signs and shapes


def test_the_guarantee_is_short_equity_and_short_rates():
    """The market risk benefit is a liability when positive, so it falls when equity rises and
    falls when rates rise. Both come out negative, which makes the hedge a short index position
    and a receive-fixed swap. A sign that disagrees with this is a bug until proven otherwise."""
    valuer, book, state = _valuer(), _small_book(), _state()
    attribution = np.full(book.size, 0.8)
    result = greeks.compute(valuer, book, state, attribution, rate_bump_bp=25.0)
    assert result.equity_exposure < 0
    assert result.rho_per_bp < 0


def test_the_liability_moves_more_on_the_way_down_than_on_the_way_up():
    """The asymmetry that Item 7A discloses and that V2 tests. It is a property of a guarantee:
    a fall puts more contracts in the money than a rally takes out."""
    valuer, book, state = _valuer(), _small_book(), _state()
    attribution = np.full(book.size, 0.8)
    table = greeks.disclosed_shocks(valuer, book, state, attribution, rate_shocks_bp=(100, -100))
    assert greeks.asymmetry(table, "equity", 10) > 1.0
    assert greeks.asymmetry(table, "rates", 100) > 1.0


def test_paired_standard_errors_are_far_smaller_than_the_level_error():
    """The whole point of common random numbers. If the paired error were the same size as the
    level error, the Greeks would be noise."""
    valuer, book, state = _valuer(), _small_book(), _state()
    attribution = np.full(book.size, 0.8)
    result = greeks.compute(valuer, book, state, attribution, rate_bump_bp=25.0)
    assert result.rho_std_error * 25.0 < result.level_std_error
    assert result.equity_std_error < result.level_std_error


def test_asymmetry_refuses_a_shock_it_was_not_given():
    valuer, book, state = _valuer(), _small_book(), _state()
    table = greeks.disclosed_shocks(valuer, book, state, np.full(book.size, 0.8),
                                    rate_shocks_bp=(100, -100))
    with raises(ValueError, match="no rates shock"):
        greeks.asymmetry(table, "rates", 50)


# ---------------------------------------------------------------- the break-even fee


def test_the_break_even_fee_zeroes_the_guarantee():
    book = _rewind_to_issue(_small_book())
    valuer, state = _valuer(), _state()
    result = breakeven.solve(valuer, book, state)
    assert result.converged
    at_fair = valuer.value(
        breakeven._rebuild_with_fee(book, result.fair), state,
        attribution=np.ones(book.size),
    )
    assert at_fair.market_risk_benefit == approx(0.0, abs=1e-4 * at_fair.account_value)


def test_the_guarantee_is_worth_less_to_the_insurer_the_less_it_charges():
    """Monotone in the fee, which is what makes the solve well posed. It is not linear, because
    a higher charge drains the account faster and brings the claim forward."""
    book = _rewind_to_issue(_small_book())
    valuer, state = _valuer(), _state()
    table = breakeven.fee_sensitivity(valuer, book, state, [0.005, 0.0125, 0.02, 0.03])
    assert np.all(np.diff(table["value"].to_numpy()) < 0)


def test_the_margin_is_the_charged_fee_less_the_fair_one():
    book = _rewind_to_issue(_small_book())
    result = breakeven.solve(_valuer(), book, _state())
    assert result.margin == approx(result.charged - result.fair, rel=1e-12)
    assert result.margin_bp == approx(result.margin * 10000.0, rel=1e-12)
