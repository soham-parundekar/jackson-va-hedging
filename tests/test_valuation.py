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
    """Eight model points, combined the way the vintage book is.

    Two issue ages, two durations and two moneyness levels is enough to exercise the aggregation
    and small enough to run in tests. Each point is a `single_contract` with its account value
    stated, which is the path the reported figures take; the horizon is cut at 100 so the
    projection stays short.
    """
    core = terms_module.load()[("flex_gmwb", "single", "core")]
    books = [
        cohorts.single_contract(
            core, issue_age, 0.0131, 0.0095,
            premium=100.0, account_value=100.0 / ratio, benefit_base=100.0,
            deferral_years=max(0, 70 - (issue_age + duration)), years_since_issue=duration,
            max_age=100, death_benefit=terms_module.DEATH_BENEFITS["basic"],
        )
        for issue_age in (60, 70)
        for duration in (0, 6)
        for ratio in (0.8, 1.2)
    ]
    return cohorts.combine(books, weights=[1.0 / len(books)] * len(books))


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


def test_the_guarantee_gets_cheaper_as_the_charge_rises_over_the_range_products_are_sold_at():
    """Falling but not linear across half a per cent to three, which is where every rider on the
    market is priced and the only range the solve has to be well posed over."""
    book = _rewind_to_issue(_small_book())
    valuer, state = _valuer(), _state()
    table = breakeven.fee_sensitivity(valuer, book, state, [0.005, 0.0125, 0.02, 0.03])
    assert np.all(np.diff(table["value"].to_numpy()) < 0)


def test_the_curve_turns_back_up_once_the_charge_starts_exhausting_the_account():
    """The reason the solve checks its bracket instead of trusting monotonicity. A charge big
    enough to empty the account ends the fee stream and leaves the guarantee to pay, so past a
    few per cent a higher fee makes the guarantee dearer rather than cheaper - and a bracket
    spanning the turn has a root on each side of it, or none.

    Six per cent against fourteen rather than adjacent points, because the two values differ by
    a couple of dollars and the Monte Carlo error on either is a few tens of cents."""
    book = _rewind_to_issue(_small_book())
    table = breakeven.fee_sensitivity(_valuer(), book, _state(), [0.06, 0.14])
    assert table["value"].iloc[1] > table["value"].iloc[0]


def test_a_bracket_with_no_root_in_it_reports_no_fee_rather_than_its_own_bound():
    """Both failure modes, and the reason each has to be separable from a solved answer.
    Returning the bound - the first version of this - put 800 basis points in the fee column of
    a table whose own convergence flag said the number was not a fee."""
    book = _rewind_to_issue(_small_book())
    valuer, state = _valuer(), _state()

    # Below the turn, where the guarantee is dear at both ends and the root is above the bracket.
    costly = breakeven.solve(valuer, book, state, bounds=(0.0005, 0.005))
    assert not costly.converged and costly.reason == "no root in the bracket"
    assert np.isnan(costly.fair) and np.isnan(costly.margin) and np.isnan(costly.margin_bp)
    assert costly.charged > 0.0

    # Above it, where the fee already covers the guarantee at the bottom of the bracket.
    cheap = breakeven.solve(valuer, book, state, bounds=(0.02, 0.08))
    assert not cheap.converged and cheap.reason == "the fair fee is below the bracket"
    assert np.isnan(cheap.fair)


def test_the_margin_is_the_charged_fee_less_the_fair_one():
    book = _rewind_to_issue(_small_book())
    result = breakeven.solve(_valuer(), book, _state())
    assert result.margin == approx(result.charged - result.fair, rel=1e-12)
    assert result.margin_bp == approx(result.margin * 10000.0, rel=1e-12)


# ---------------------------------------------------------------- Monte Carlo and truncation


def test_the_standard_error_falls_with_the_square_root_of_the_path_count():
    """Four times the paths, half the error. Not a tight check - the error of an error is
    itself noisy - but a wide one catches the mistakes that matter: a standard error computed
    across the wrong axis, or one that forgets the antithetic pairing and so reports a number
    that does not fall at all.
    """
    state, book = _state(), _small_book()
    errors = []
    for n_paths in (2_000, 8_000):
        valuer = Valuer(mortality.load("basic"), n_paths=n_paths, seed=99, cache_size=2)
        attribution = valuer.calibrate_attribution(book, state)
        errors.append(valuer.value(book, state, attribution=attribution).std_error)
    assert 0.35 < errors[1] / errors[0] < 0.75


def test_truncating_the_projection_costs_less_than_the_noise_it_is_hidden_in():
    """The truncation test only means something on common draws.

    The simulator draws its normals in one array whose width follows the horizon, so a
    forty-year run and a fifty-year run at the same seed are different worlds rather than a
    prefix and its extension. Asking for a longer path horizon than the book needs is what
    makes the two comparable, and without it this comparison reports the Monte Carlo difference
    between two independent simulations as a truncation effect.
    """
    state = _state()
    core = terms_module.load()[("flex_gmwb", "single", "core")]
    valuer = _valuer()

    def book_to(max_age: int):
        return cohorts.single_contract(core, issue_age=70, base_contract_charge=0.0131,
                                       fund_expense=0.0095, premium=100_000.0,
                                       deferral_years=5, max_age=max_age)

    longest = 120 - 70
    attribution = valuer.calibrate_attribution(book_to(115), state)
    priced = {cap: valuer.value(book_to(cap), state, attribution=attribution,
                                path_years=longest)
              for cap in (105, 115, 120)}
    noise = priced[120].std_error

    # Cutting at 115 costs nothing you could measure; cutting at 105 costs something real but
    # still small, and both are the same draws so the differences are truncation and not luck.
    assert abs(priced[115].market_risk_benefit - priced[120].market_risk_benefit) < 0.1 * noise
    assert 0 < (priced[120].market_risk_benefit - priced[105].market_risk_benefit)
    assert (priced[120].market_risk_benefit - priced[105].market_risk_benefit) < noise


def test_a_shorter_path_horizon_than_the_book_needs_is_refused():
    state, book = _state(), _small_book()
    valuer = _valuer()
    with raises(ValueError, match="shorter than"):
        valuer.value(book, state, attribution=np.ones(book.size), path_years=5)


def test_the_guarantee_is_worth_more_the_further_in_the_money_it_starts():
    """The moneyness profile the disclosed comparison is located on has to be monotone, or an
    implied moneyness read off it would not be unique."""
    state = _state()
    core = terms_module.load()[("flex_gmwb", "single", "core")]
    valuer = _valuer()
    values = []
    for ratio in (0.8, 1.0, 1.4):
        book = cohorts.single_contract(core, issue_age=70, base_contract_charge=0.0131,
                                       fund_expense=0.0095, premium=100_000.0,
                                       account_value=100_000.0, benefit_base=100_000.0 * ratio,
                                       deferral_years=5, max_age=115)
        attribution = valuer.calibrate_attribution(book, state)
        values.append(valuer.value(book, state, attribution=attribution).market_risk_benefit)
    assert values[0] < values[1] < values[2]
