"""Valuation engine tests.

The two that carry the most weight are the martingale test and the fee-collection
identity. The first checks the risk-neutral drift by pricing the account value itself as
a traded asset: with no charges and no withdrawals, the discounted expected account value
has to come back to where it started. The second checks the closed form used to measure
continuously accruing charges against a brute-force calculation that sub-steps the year.
Neither can pass by accident.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from tests.checks import approx, raises

from gmwb import mortality
from gmwb.contract import GmwbContract
from gmwb.curves import flat_curve
from gmwb.engine import (
    calibrate_attribution,
    make_normals,
    projection_years,
    value_rider,
)
from gmwb.volatility import VolTermStructure

RATE = 0.04
MAX_AGE = 110


def base_contract(**overrides) -> GmwbContract:
    defaults = dict(
        premium=100_000.0,
        issue_age=70,
        gawa_pct=0.0575,
        rider_charge_pct=0.0125,
        base_contract_charge=0.0131,
        fund_expense=0.0095,
    )
    defaults.update(overrides)
    return GmwbContract(**defaults)


def fixtures(n_paths=40_000, contract=None, max_age=MAX_AGE):
    contract = contract or base_contract()
    curve = flat_curve(RATE)
    vol = VolTermStructure(front_level=0.18, long_run_level=0.18)
    basis = mortality.load(contract.issue_age, 2025, 0.5)
    n_years = projection_years(contract, max_age)
    normals = make_normals(n_paths, n_years, 11, True)
    return contract, curve, vol, basis, normals


def test_discounted_account_value_is_a_martingale():
    """With no charges and no withdrawals the account value is a traded asset, so its
    discounted expectation must equal its starting value. This is what checks that the
    drift is the risk-free rate and that the lognormal step is centred correctly."""
    contract = base_contract(
        gawa_pct=1e-12, rider_charge_pct=0.0, base_contract_charge=0.0, fund_expense=0.0
    )
    _, curve, vol, basis, normals = fixtures(200_000, contract)
    result = value_rider(contract, curve, vol, basis, normals, max_age=MAX_AGE)

    horizons = [1, 5, 10, 20, 39]
    for k in horizons:
        discounted = result.mean_account_value[k - 1] * float(curve.discount(result.times[k - 1]))
        assert discounted == approx(contract.premium, rel=3e-3), f"year {k}"


def test_continuous_charge_collection_matches_sub_stepping():
    """The engine collects charges with a closed form instead of sub-stepping the year.
    This reproduces the same quantity by brute force on a single path of the same
    lognormal increments and checks they agree."""
    drag = 0.0226
    insurer_share = 0.0131 / drag
    rng = np.random.default_rng(5)
    for _ in range(200):
        start = float(rng.uniform(1_000, 200_000))
        total_return = float(np.exp(rng.normal(0.02, 0.2)))
        av_pre_drag = start * total_return

        closed_form = insurer_share * av_pre_drag * (1.0 - np.exp(-drag))

        # Brute force: split the year into many intervals, let the balance grow smoothly
        # at the realised rate and deduct the charge each interval, then accumulate the
        # deductions forward at the same gross rate.
        steps = 20_000
        dt = 1.0 / steps
        growth = total_return ** dt
        balance = start
        accumulated = 0.0
        for _ in range(steps):
            balance *= growth
            deduction = balance * (1.0 - np.exp(-drag * dt))
            balance -= deduction
            accumulated = accumulated * growth + deduction * insurer_share
        assert closed_form == approx(accumulated, rel=2e-4)


def test_account_value_floors_at_zero_and_stays_there():
    """A contract drawing far more than it can earn must exhaust, must not go negative,
    and must keep paying once it has."""
    contract = base_contract(gawa_pct=0.20)
    _, curve, vol, basis, normals = fixtures(20_000, contract)
    result = value_rider(contract, curve, vol, basis, normals, max_age=MAX_AGE)

    assert np.all(result.mean_account_value >= 0)
    # A 20% draw against a 4% drift and 2.26% of charges cannot last: exhausted on a fifth
    # of paths by year three and on essentially all of them by year ten.
    assert result.exhaustion_prob[2] > 0.02
    assert result.exhaustion_prob[9] > 0.997
    assert np.all(np.diff(result.exhaustion_prob) >= -1e-12)
    # Claims continue for as long as anyone is alive.
    assert result.claims_by_year[-1] > 0


def test_benefit_base_never_falls():
    contract = base_contract()
    _, curve, vol, basis, normals = fixtures(20_000, contract)
    result = value_rider(contract, curve, vol, basis, normals, max_age=MAX_AGE)
    assert np.all(np.diff(result.mean_benefit_base) >= -1e-9)
    assert result.mean_benefit_base[0] >= contract.premium - 1e-9


def test_no_guarantee_means_no_claims():
    contract = base_contract(gawa_pct=1e-12)
    _, curve, vol, basis, normals = fixtures(20_000, contract)
    result = value_rider(contract, curve, vol, basis, normals, max_age=MAX_AGE)
    assert result.pv_claims == approx(0.0, abs=1e-6)
    # Fees keep coming, so the net position is an asset to the insurer.
    assert result.net_value < 0


def test_higher_volatility_raises_the_guarantee():
    contract, curve, _, basis, normals = fixtures(60_000)
    values = []
    for level in (0.10, 0.18, 0.30, 0.45):
        vol = VolTermStructure(front_level=level, long_run_level=level)
        values.append(value_rider(contract, curve, vol, basis, normals,
                                  max_age=MAX_AGE).pv_claims)
    assert np.all(np.diff(values) > 0)


def test_higher_rates_lower_the_liability():
    contract, _, vol, basis, normals = fixtures(60_000)
    values = []
    for rate in (0.01, 0.03, 0.05, 0.07):
        values.append(value_rider(contract, flat_curve(rate), vol, basis, normals,
                                  max_age=MAX_AGE).pv_claims)
    assert np.all(np.diff(values) < 0)


def test_step_up_can_only_help_the_policyholder():
    contract, curve, vol, basis, normals = fixtures(40_000)
    with_step_up = value_rider(contract, curve, vol, basis, normals, max_age=MAX_AGE)
    without = value_rider(replace(contract, annual_step_up=False), curve, vol, basis,
                          normals, max_age=MAX_AGE)
    assert with_step_up.pv_claims > without.pv_claims


def test_attribution_calibration_zeroes_the_benefit_when_it_can():
    contract, curve, vol, basis, normals = fixtures(60_000)
    at_issue = value_rider(contract, curve, vol, basis, normals, max_age=MAX_AGE)
    alpha = calibrate_attribution(at_issue)
    assert 0 < alpha <= 1
    calibrated = value_rider(contract, curve, vol, basis, normals,
                             fee_attribution=alpha, max_age=MAX_AGE)
    if alpha < 1.0:
        assert calibrated.net_value == approx(0.0, abs=1e-6)
    else:
        assert calibrated.net_value > 0


def test_attribution_is_capped_at_one():
    contract = base_contract(gawa_pct=0.12)
    _, curve, vol, basis, normals = fixtures(20_000, contract)
    at_issue = value_rider(contract, curve, vol, basis, normals, max_age=MAX_AGE)
    assert calibrate_attribution(at_issue) == approx(1.0)


def test_partial_utilisation_lowers_the_guarantee():
    contract, curve, vol, basis, normals = fixtures(40_000)
    values = []
    for utilisation in (1.0, 0.8, 0.6, 0.4):
        result = value_rider(replace(contract, utilisation=utilisation), curve, vol, basis,
                             normals, max_age=MAX_AGE)
        values.append(result.pv_claims)
    assert np.all(np.diff(values) < 0)
    assert values[-1] < 0.5 * values[0]


def test_lapse_reduces_both_claims_and_fees():
    contract, curve, vol, basis, normals = fixtures(40_000)
    no_lapse = value_rider(contract, curve, vol, basis, normals, max_age=MAX_AGE)
    lapsing = value_rider(replace(contract, lapse_rate=0.04), curve, vol, basis, normals,
                          max_age=MAX_AGE)
    assert lapsing.pv_claims < no_lapse.pv_claims
    assert lapsing.pv_rider_fees < no_lapse.pv_rider_fees
    # Claims are longer dated than fees, so lapse takes proportionally more off the claims.
    claim_drop = 1 - lapsing.pv_claims / no_lapse.pv_claims
    fee_drop = 1 - lapsing.pv_rider_fees / no_lapse.pv_rider_fees
    assert claim_drop > fee_drop


def test_equity_shock_moves_the_account_not_the_benefit_base():
    contract, curve, vol, basis, normals = fixtures(20_000)
    shocked = value_rider(contract, curve, vol, basis, normals, equity_shock=-0.10,
                          max_age=MAX_AGE)
    assert shocked.account_value == approx(contract.premium * 0.9)
    assert shocked.benefit_base == approx(contract.premium)
    base = value_rider(contract, curve, vol, basis, normals, max_age=MAX_AGE)
    assert shocked.net_value > base.net_value


def test_equity_beta_scales_the_shock():
    contract = base_contract(fund_equity_beta=0.5)
    _, curve, vol, basis, normals = fixtures(20_000, contract)
    shocked = value_rider(contract, curve, vol, basis, normals, equity_shock=-0.10,
                          max_age=MAX_AGE)
    assert shocked.account_value == approx(contract.premium * 0.95)


def test_credit_spread_lowers_the_reported_value():
    contract, curve, vol, basis, normals = fixtures(40_000)
    plain = value_rider(contract, curve, vol, basis, normals, max_age=MAX_AGE)
    spread = value_rider(contract, curve, vol, basis, normals, credit_spread=0.01,
                         max_age=MAX_AGE)
    assert spread.net_value < plain.net_value


def test_a_short_first_step_prices_between_neighbouring_grids():
    contract, curve, vol, basis, normals = fixtures(40_000)
    full = value_rider(contract, curve, vol, basis, normals, max_age=MAX_AGE,
                       first_step_years=1.0)
    short = value_rider(contract, curve, vol, basis, normals, max_age=MAX_AGE,
                        first_step_years=0.05)
    # Bringing the next withdrawal forward makes the guarantee more expensive.
    assert short.pv_claims > full.pv_claims
    assert short.times[0] == approx(0.05)


def test_truncating_the_projection_costs_less_than_simulation_noise():
    contract = base_contract()
    curve = flat_curve(RATE)
    vol = VolTermStructure(front_level=0.18, long_run_level=0.18)
    basis = mortality.load(70, 2025, 0.5)
    values = {}
    for cap in (110, 115, 120):
        n_years = projection_years(contract, cap)
        normals = make_normals(100_000, n_years, 11, True)
        result = value_rider(contract, curve, vol, basis, normals, max_age=cap)
        values[cap] = result
    spread = abs(values[120].pv_claims - values[110].pv_claims)
    assert spread < 3 * values[115].std_error


def test_standard_error_falls_with_the_square_root_of_paths():
    contract, curve, vol, basis, _ = fixtures()
    n_years = projection_years(contract, MAX_AGE)
    errors = []
    for n in (10_000, 40_000, 160_000):
        normals = make_normals(n, n_years, 3, True)
        errors.append(value_rider(contract, curve, vol, basis, normals,
                                  max_age=MAX_AGE).std_error)
    assert errors[0] / errors[1] == approx(2.0, rel=0.25)
    assert errors[1] / errors[2] == approx(2.0, rel=0.25)


def test_antithetic_sampling_beats_independent_draws():
    contract, curve, vol, basis, _ = fixtures()
    n_years = projection_years(contract, MAX_AGE)
    paired = value_rider(contract, curve, vol, basis,
                         make_normals(40_000, n_years, 3, True), max_age=MAX_AGE)
    independent = value_rider(contract, curve, vol, basis,
                              make_normals(40_000, n_years, 3, False),
                              antithetic=False, max_age=MAX_AGE)
    assert paired.std_error < independent.std_error


def test_rejects_bad_arguments():
    contract, curve, vol, basis, normals = fixtures(2_000)
    with raises(ValueError):
        value_rider(contract, curve, vol, basis, normals, fee_attribution=1.5,
                    max_age=MAX_AGE)
    with raises(ValueError):
        value_rider(contract, curve, vol, basis, normals, first_step_years=0.0,
                    max_age=MAX_AGE)
    with raises(ValueError):
        value_rider(contract, curve, vol, basis, normals, max_age=MAX_AGE + 40)
    with raises(ValueError):
        value_rider(replace(contract, utilisation=1.5), curve, vol, basis, normals,
                    max_age=MAX_AGE)
    with raises(ValueError):
        make_normals(999, 10, 1, True)
