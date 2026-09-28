"""Greeks and shock repricing tests.

Bump and revalue is only trustworthy if the bumps share random draws with the base
valuation, so one test checks that the noise really does cancel: the same delta computed
from two different seeds should agree far more closely than the standard error on the
level would allow if it did not.
"""

from __future__ import annotations

import numpy as np

from tests.checks import approx, raises

from gmwb import mortality
from gmwb.contract import GmwbContract
from gmwb.curves import ParCurveBuilder
from gmwb.engine import make_normals, projection_years, value_rider
from gmwb.sensitivities import compute_greeks, disclosed_shock_repricing, moneyness_profile
from gmwb.volatility import VolTermStructure

TENORS = [1, 2, 3, 5, 7, 10, 20, 30]
YIELDS = np.array([0.0348, 0.0347, 0.0355, 0.0373, 0.0394, 0.0418, 0.0479, 0.0484])
MAX_AGE = 110


def setup(n_paths=40_000, seed=17, **overrides):
    defaults = dict(
        premium=100_000.0,
        issue_age=70,
        gawa_pct=0.0575,
        rider_charge_pct=0.0125,
        base_contract_charge=0.0131,
        fund_expense=0.0095,
    )
    defaults.update(overrides)
    contract = GmwbContract(**defaults)
    builder = ParCurveBuilder(TENORS, YIELDS)
    vol = VolTermStructure.from_implied({0.25: 0.18}, long_run_level=0.19)
    basis = mortality.load(contract.issue_age, 2025, 0.5)
    n_years = projection_years(contract, MAX_AGE)
    normals = make_normals(n_paths, n_years, seed, True)
    return contract, builder, vol, basis, normals, {"max_age": MAX_AGE}


def test_greek_signs():
    contract, builder, vol, basis, normals, state = setup()
    greeks = compute_greeks(contract, builder, vol, basis, normals, state)
    # A written guarantee falls in value when the index rises.
    assert greeks.equity_exposure < 0
    # It is convex in the index: the loss from a fall exceeds the gain from a rise.
    assert greeks.equity_gamma > 0
    # A long-dated liability discounts away faster when rates rise.
    assert greeks.rho_per_bp < 0
    # More volatility makes a guarantee more expensive.
    assert greeks.vega_per_point > 0
    assert greeks.vega_long_run > 0


def test_long_dated_volatility_carries_more_of_the_exposure():
    """Listed index options run out well before the long-run level starts to matter, so a
    desk can trade only part of the volatility exposure. If the tradeable part ever
    dominates, the term structure has lost its decay."""
    contract, builder, vol, basis, normals, state = setup()
    greeks = compute_greeks(contract, builder, vol, basis, normals, state)
    assert greeks.vega_long_run > greeks.vega_per_point


def test_common_random_numbers_make_the_delta_stable():
    """The level has a standard error of tens of dollars. The delta should agree across
    seeds far more tightly than that, which only happens if the draws cancel."""
    deltas, errors = [], []
    for seed in (11, 22, 33):
        contract, builder, vol, basis, normals, state = setup(seed=seed)
        greeks = compute_greeks(contract, builder, vol, basis, normals, state)
        deltas.append(greeks.equity_exposure)
        errors.append(greeks.std_error)
    spread = max(deltas) - min(deltas)
    assert spread < 0.05 * abs(np.mean(deltas))
    assert np.mean(errors) > 0


def test_delta_predicts_a_small_shock():
    """A first-order prediction from the delta should match a full repricing under a small
    shock to within the convexity term."""
    contract, builder, vol, basis, normals, state = setup(n_paths=80_000)
    greeks = compute_greeks(contract, builder, vol, basis, normals, state)
    curve = builder.build()
    base = value_rider(contract, curve, vol, basis, normals, **state)
    shock = -0.02
    actual = value_rider(contract, curve, vol, basis, normals, equity_shock=shock,
                         **state).net_value - base.net_value
    log_move = np.log(1.0 + shock)
    predicted = greeks.equity_exposure * log_move
    convexity = 0.5 * greeks.equity_gamma * log_move**2
    assert actual == approx(predicted + convexity, rel=0.05)


def test_rho_predicts_a_small_rate_shock():
    contract, builder, vol, basis, normals, state = setup(n_paths=80_000)
    greeks = compute_greeks(contract, builder, vol, basis, normals, state)
    base = value_rider(contract, builder.build(), vol, basis, normals, **state)
    actual = value_rider(contract, builder.build(shift_bp=10), vol, basis, normals,
                         **state).net_value - base.net_value
    assert actual == approx(greeks.rho_per_bp * 10, rel=0.03)


def test_disclosed_shocks_are_convex():
    """Both features Jackson's table shows: a down shock moves the liability more than an
    up shock, and the 100bp impact is more than twice the 50bp impact on the downside and
    less than twice on the upside."""
    contract, builder, vol, basis, normals, state = setup(n_paths=80_000)
    result = disclosed_shock_repricing(contract, builder, vol, basis, normals, state)

    assert result["equity_down_10pct"] > 0 > result["equity_up_10pct"]
    assert result["equity_down_10pct"] > abs(result["equity_up_10pct"])

    assert result["rates_up_100bp"] < result["rates_up_50bp"] < 0
    assert result["rates_down_100bp"] > result["rates_down_50bp"] > 0
    assert result["rates_down_100bp"] / result["rates_down_50bp"] > 2.0
    assert result["rates_up_100bp"] / result["rates_up_50bp"] < 2.0


def test_moneyness_profile_is_monotone_in_value():
    contract, builder, vol, basis, normals, state = setup(n_paths=20_000)
    rows = moneyness_profile(contract, builder, vol, basis, normals, state,
                             ratios=(0.8, 1.0, 1.2, 1.4))
    values = [row["base_value"] for row in rows]
    assert np.all(np.diff(values) > 0)
    # A larger benefit base relative to the account value means a bigger equity response.
    responses = [row["equity_down_10pct"] for row in rows]
    assert np.all(np.diff(responses) > 0)


def test_moneyness_profile_holds_the_account_value_fixed():
    contract, builder, vol, basis, normals, state = setup(n_paths=10_000)
    rows = moneyness_profile(contract, builder, vol, basis, normals, state,
                             ratios=(0.8, 1.2))
    assert rows[0]["account_value"] == approx(rows[1]["account_value"])
    assert rows[0]["account_value"] == approx(contract.premium)


def test_reporting_basis_is_more_expensive_than_the_economic_basis():
    contract, builder, vol, basis, normals, state = setup()
    period = mortality.load(contract.issue_age, 2025, 0.5, table="period")
    economic = compute_greeks(contract, builder, vol, basis, normals, state)
    reporting = compute_greeks(contract, builder, vol, period, normals, state)
    assert reporting.base_value > economic.base_value
    assert abs(reporting.rho_per_bp) > abs(economic.rho_per_bp)


def test_rejects_an_unknown_state_key():
    contract, builder, vol, basis, normals, _ = setup(n_paths=2_000)
    with raises(TypeError):
        compute_greeks(contract, builder, vol, basis, normals, {"not_a_real_argument": 1})
