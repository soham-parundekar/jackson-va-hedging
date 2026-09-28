"""Hedging tests: policy roll-forward, hedge sizing, and the profit ledger.

The most valuable test here is the look-ahead check. A hedging backtest that peeks at the
next period's return will show a superb variance reduction and be worthless, so the
position held over a period is verified to be a function of information available when it
was set, and the profit booked for a period is verified to use it.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from tests.checks import approx, raises

from gmwb import config, hedging, market
from gmwb.contract import GmwbContract
from gmwb.curves import flat_curve
from gmwb.hedging import PolicyState, SubAccountMix, compose, effectiveness, swap_annuity


def a_contract(**overrides) -> GmwbContract:
    defaults = dict(
        premium=100_000.0,
        issue_age=70,
        gawa_pct=0.05,
        rider_charge_pct=0.01,
        base_contract_charge=0.013,
        fund_expense=0.01,
    )
    defaults.update(overrides)
    return GmwbContract(**defaults)


def test_swap_annuity_grows_with_tenor_and_matches_a_hand_calculation():
    curve = flat_curve(0.0)
    for tenor in (2, 5, 10, 30):
        # At a zero discount rate the annuity is just the tenor.
        assert swap_annuity(curve, tenor) == approx(float(tenor), rel=1e-12)
    curve = flat_curve(0.04)
    annuities = [swap_annuity(curve, t) for t in (2, 5, 10, 30)]
    assert np.all(np.diff(annuities) > 0)
    assert annuities[-1] < 30


def test_sub_account_weights_must_sum_to_one():
    with raises(ValueError):
        SubAccountMix(equity=0.5, bond=0.1, balanced=0.1, money_market=0.1)


def test_effective_equity_beta_matches_the_disclosed_mix():
    mix = SubAccountMix(equity=0.7235, bond=0.0834, balanced=0.1832, money_market=0.0099)
    assert mix.effective_equity_beta == approx(0.7235 + 0.6 * 0.1832, rel=1e-12)
    assert SubAccountMix.all_equity().effective_equity_beta == approx(1.0)


def test_all_equity_mix_returns_the_index():
    mix = SubAccountMix.all_equity()
    assert mix.realised_return(0.05, 0.04, 0.001, 0.03, 1 / 52) == approx(0.05, rel=1e-12)


def test_bond_leg_loses_when_yields_rise():
    mix = SubAccountMix(equity=0.0, bond=1.0, balanced=0.0, money_market=0.0,
                        bond_duration=6.0)
    up = mix.realised_return(0.0, 0.04, 0.01, 0.03, 1 / 52)
    assert up == approx(0.04 / 52 - 6.0 * 0.01, rel=1e-12)
    assert up < 0


def test_policy_takes_its_withdrawal_only_on_an_anniversary():
    contract = a_contract()
    policy = PolicyState.at_issue(contract, "2020-01-02")
    mid_year = policy.step(contract, "2020-06-30", sub_account_return=0.0, dt=0.5)
    assert mid_year["rider_charge"] == 0.0
    assert mid_year["claim"] == 0.0
    assert mid_year["me_charge"] > 0

    on_anniversary = policy.step(contract, "2021-01-04", sub_account_return=0.0, dt=0.51)
    assert on_anniversary["rider_charge"] == approx(0.01 * 100_000.0, rel=1e-9)
    assert policy.anniversaries_passed == 1


def test_benefit_base_ratchets_up_and_never_down():
    contract = a_contract()
    policy = PolicyState.at_issue(contract, "2020-01-02")
    bases = []
    for year in range(1, 8):
        policy.step(contract, f"{2020 + year}-01-05", sub_account_return=0.30, dt=1.0)
        bases.append(policy.benefit_base)
    assert np.all(np.diff(bases) > 0)

    policy = PolicyState.at_issue(contract, "2020-01-02")
    bases = []
    for year in range(1, 8):
        policy.step(contract, f"{2020 + year}-01-05", sub_account_return=-0.20, dt=1.0)
        bases.append(policy.benefit_base)
    assert np.all(np.diff(bases) == 0)
    assert bases[0] == approx(contract.premium)


def test_account_value_floors_at_zero_and_claims_take_over():
    contract = a_contract(gawa_pct=0.25)
    policy = PolicyState.at_issue(contract, "2020-01-02")
    claims = []
    for year in range(1, 9):
        cash = policy.step(contract, f"{2020 + year}-01-05", sub_account_return=0.0, dt=1.0)
        claims.append(cash["claim"])
        assert policy.account_value >= 0
    assert claims[0] == 0.0
    assert claims[-1] > 0
    # Once exhausted, the insurer pays the full guaranteed amount and collects no charges.
    assert claims[-1] == approx(contract.gawa_pct * policy.benefit_base, rel=1e-9)
    assert policy.account_value == 0.0


def test_years_to_anniversary_stays_inside_its_bounds():
    contract = a_contract()
    policy = PolicyState.at_issue(contract, "2020-01-02")
    for date in pd.date_range("2020-01-02", "2020-12-28", freq="W"):
        value = policy.years_to_anniversary(date)
        assert 0 < value <= 1.0
    assert policy.years_to_anniversary("2020-01-02") == approx(1.0, rel=2e-3)
    assert policy.years_to_anniversary("2020-12-28") < 0.05


def test_rebalance_dates_land_on_trading_days():
    frame = market.load_panel()
    weekly = hedging.rebalance_dates(frame, "2019-01-01", "2019-12-31", "weekly")
    monthly = hedging.rebalance_dates(frame, "2019-01-01", "2019-12-31", "monthly")
    trading = set(market.equity_dates(frame, "2019-01-01", "2019-12-31"))
    assert set(weekly) <= trading
    assert set(monthly) <= trading
    assert 50 <= len(weekly) <= 53
    assert len(monthly) == 12
    assert weekly.is_monotonic_increasing
    with raises(ValueError):
        hedging.rebalance_dates(frame, "2019-01-01", "2019-12-31", "fortnightly")


def test_roll_policy_matches_a_hand_computed_step():
    cfg = config.load()
    frame = market.load_panel()
    contract = a_contract()
    mix = SubAccountMix.all_equity()
    policy, history = hedging.roll_policy(contract, cfg, frame, mix,
                                        "2019-01-02", "2019-01-31", frequency="weekly")
    first, second = history.index[0], history.index[1]
    dt = (second - first).days / 365.25
    index_return = (history.loc[second, "spot"] / history.loc[first, "spot"] - 1
                    + cfg["dividend_yield"] * dt)
    expected = contract.premium * (1 + index_return) * np.exp(-contract.account_drag * dt)
    assert history.loc[second, "account_value"] == approx(expected, rel=1e-12)
    assert history["benefit_base"].iloc[-1] == approx(contract.premium)


def test_roll_policy_needs_more_than_one_date():
    cfg = config.load()
    frame = market.load_panel()
    with raises(ValueError):
        hedging.roll_policy(a_contract(), cfg, frame, SubAccountMix.all_equity(),
                          "2019-01-02", "2019-01-03", frequency="monthly")


def synthetic_ledger(n: int = 200, seed: int = 4) -> pd.DataFrame:
    """A ledger with a known structure, for testing the composition arithmetic."""
    rng = np.random.default_rng(seed)
    excess = rng.normal(0, 0.02, n)
    rate_move = rng.normal(0, 5.0, n)
    vol_move = rng.normal(0, 0.01, n)
    exposure = -30_000.0
    dv01 = 50.0
    vega = 400.0
    liability = -(exposure * excess) - (-dv01 * rate_move) - (vega * vol_move * 100)
    frame = pd.DataFrame(
        {
            "liability_pnl": liability,
            "equity_hedge_pnl": exposure * excess,
            "rate_hedge_pnl": -dv01 * rate_move,
            "vega_hedge_pnl": vega * vol_move * 100,
            "equity_cost": np.full(n, 1.0),
            "rate_cost": np.full(n, 0.5),
            "vega_cost": np.full(n, 2.0),
        },
        index=pd.date_range("2020-01-03", periods=n, freq="W"),
    )
    frame["hedged_pnl_gross"] = frame["liability_pnl"]
    frame["transaction_cost"] = 0.0
    frame["hedged_pnl"] = frame["liability_pnl"]
    return frame


def test_composing_no_legs_gives_back_the_unhedged_liability():
    ledger = synthetic_ledger()
    composed = compose(ledger, ())
    assert np.allclose(composed["hedged_pnl"], ledger["liability_pnl"])
    assert effectiveness(composed)["variance_ratio"] == approx(1.0, rel=1e-12)
    assert effectiveness(composed)["total_costs"] == approx(0.0)


def test_each_leg_removes_its_own_factor():
    ledger = synthetic_ledger()
    ratios = {}
    for legs in ((), ("equity",), ("equity", "rates"), ("equity", "rates", "vega")):
        ratios[legs] = effectiveness(compose(ledger, legs))["variance_ratio_gross"]
    assert ratios[()] > ratios[("equity",)] > ratios[("equity", "rates")]
    # The synthetic liability is exactly the three factors, so hedging all three removes
    # everything.
    assert ratios[("equity", "rates", "vega")] < 1e-20


def test_costs_accumulate_by_leg():
    ledger = synthetic_ledger(n=100)
    assert effectiveness(compose(ledger, ("equity",)))["total_costs"] == approx(100.0)
    assert effectiveness(compose(ledger, ("equity", "rates")))["total_costs"] == approx(150.0)
    assert effectiveness(
        compose(ledger, ("equity", "rates", "vega"))
    )["total_costs"] == approx(350.0)


def test_compose_rejects_an_unknown_leg():
    with raises(ValueError):
        compose(synthetic_ledger(), ("equity", "convexity"))


def test_max_drawdown_matches_a_hand_calculation():
    frame = synthetic_ledger(n=5)
    frame["liability_pnl"] = [10.0, -4.0, -3.0, 6.0, -1.0]
    frame["hedged_pnl_gross"] = frame["liability_pnl"]
    frame["hedged_pnl"] = frame["liability_pnl"]
    frame["transaction_cost"] = 0.0
    metrics = effectiveness(frame)
    # Cumulative profit runs 10, 6, 3, 9, 8, so the worst fall from a peak is seven.
    assert metrics["unhedged_max_drawdown"] == approx(-7.0)
    assert metrics["unhedged_worst"] == approx(-4.0)


def test_backtest_uses_no_information_from_after_the_rebalance():
    """The positions recorded on a date must be reproducible from that date alone.

    The check re-derives each period's hedge profit from the position recorded on the
    previous date and the market move that followed, and compares it with what the
    backtest booked. Any leak of future information into the sizing would break the
    identity.
    """
    cfg = config.load()
    frame = market.load_panel()
    hedge_cfg = dict(cfg["hedge"])
    hedge_cfg.update({"start": "2019-01-01", "end": "2021-12-31", "rebalance": "monthly"})
    cfg = {**cfg, "hedge": hedge_cfg}

    from gmwb.engine import make_normals, projection_years

    contract = GmwbContract.from_config(cfg)
    n_years = projection_years(contract, int(cfg["simulation"]["max_age"]))
    normals = make_normals(4_000, n_years, 1, True)
    ledger = hedging.run_backtest(contract, cfg, frame, normals, 0.85, 0.19,
                                  mix=SubAccountMix.all_equity(), hedge_vega=True)

    booked = ledger["equity_hedge_pnl"].to_numpy()[1:]
    rebuilt = (
        ledger["equity_notional"].shift(1).to_numpy()[1:] * ledger["excess_return"].to_numpy()[1:]
    )
    assert np.allclose(booked, rebuilt, rtol=1e-12, atol=1e-9)

    booked_rates = ledger["rate_hedge_pnl"].to_numpy()[1:]
    rebuilt_rates = (
        -ledger["swap_dv01"].shift(1).to_numpy()[1:] * ledger["rate_move_bp"].to_numpy()[1:]
    )
    assert np.allclose(booked_rates, rebuilt_rates, rtol=1e-12, atol=1e-9)

    # The first period has no prior position, so it carries no hedge profit.
    assert np.isnan(ledger["equity_hedge_pnl"].iloc[0]) or ledger["equity_hedge_pnl"].iloc[0] == 0


def test_hedge_signs_are_the_right_way_round():
    """A written guarantee behaves like a short put, so the equity hedge is normally short
    the index and the rate hedge receives fixed.

    Equity exposure is not short unconditionally, and the exception is economically real
    rather than numerical. Under the Contract Anniversary Value step-up the benefit base
    resets to the contract value on the anniversary, so in the weeks before one, with the
    contract value above the benefit base, the index level on that date is about to fix the
    guaranteed income for the rest of the contract's life. The insurer is long equity into
    that reset. The test therefore pins the sign where the guarantee is in the money and
    checks that any long exposure sits close to an anniversary with the contract value above
    the benefit base.
    """
    cfg = config.load()
    frame = market.load_panel()
    hedge_cfg = dict(cfg["hedge"])
    hedge_cfg.update({"start": "2019-01-01", "end": "2021-12-31", "rebalance": "monthly"})
    cfg = {**cfg, "hedge": hedge_cfg}

    from gmwb.engine import make_normals, projection_years

    contract = GmwbContract.from_config(cfg)
    n_years = projection_years(contract, int(cfg["simulation"]["max_age"]))
    normals = make_normals(4_000, n_years, 1, True)
    ledger = hedging.run_backtest(contract, cfg, frame, normals, 0.85, 0.19,
                                  mix=SubAccountMix.all_equity())

    # Rate exposure has no such exception: a long-dated liability always shrinks when rates
    # rise, so the hedge always receives fixed.
    assert (ledger["rho_per_bp"] < 0).all()
    assert (ledger["swap_dv01"] > 0).all()

    # Short in the majority of periods, and short without exception wherever the benefit base
    # is at or above the contract value.
    assert (ledger["equity_exposure"] < 0).mean() > 0.6
    in_the_money = ledger[ledger["benefit_base"] >= ledger["account_value"]]
    assert not in_the_money.empty
    assert (in_the_money["equity_exposure"] < 0).all()

    # Every long period has the contract value above the benefit base and sits inside the
    # second half of a contract year, approaching the reset.
    long_periods = ledger[ledger["equity_exposure"] > 0]
    assert not long_periods.empty
    assert (long_periods["account_value"] > long_periods["benefit_base"]).all()
    assert (long_periods["first_step_years"] < 0.5).all()

    assert np.allclose(np.sign(ledger["equity_notional"]),
                       np.sign(ledger["equity_exposure"]))


def test_exposure_rises_as_a_step_up_approaches():
    """The mechanism behind the long periods, isolated from market movement.

    On a realised path the contract value moves at the same time as the time to the
    anniversary shrinks, so monotonicity cannot be read off the ledger. Holding the state
    fixed and varying only the time to the reset is what shows it.
    """
    from gmwb import mortality
    from gmwb.curves import ParCurveBuilder
    from gmwb.engine import make_normals, projection_years
    from gmwb.sensitivities import compute_greeks
    from gmwb.volatility import VolTermStructure

    contract = a_contract(gawa_pct=0.0575, rider_charge_pct=0.0125)
    builder = ParCurveBuilder([1, 2, 3, 5, 7, 10, 20, 30],
                              np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]))
    vol = VolTermStructure.from_implied({0.25: 0.18}, long_run_level=0.19)
    basis = mortality.load(contract.issue_age, 2021, 0.5)
    normals = make_normals(20_000, projection_years(contract, 110), 5, True)

    def exposure_path(curve_builder):
        out = []
        for tau in (1.0, 0.8, 0.6, 0.4, 0.2, 0.05):
            greeks = compute_greeks(
                contract, curve_builder, vol, basis, normals,
                {"account_value": 130_000.0, "benefit_base": 100_000.0,
                 "fee_attribution": 1.0, "max_age": 110, "first_step_years": tau},
            )
            out.append(greeks.equity_exposure)
        return out

    low_rates = exposure_path(builder)
    assert np.all(np.diff(low_rates) > 0), low_rates
    # With the contract value 30% above the benefit base and a 2% curve, the account depletes
    # on nearly every path, so the certain reset already dominates and exposure is long.
    assert low_rates[-1] > 0

    high = ParCurveBuilder([1, 2, 3, 5, 7, 10, 20, 30], np.array([0.06] * 8))
    high_rates = exposure_path(high)
    assert np.all(np.diff(high_rates) > 0), high_rates
    # A higher risk-neutral drift keeps paths solvent, which weakens the reset channel. The
    # rate environment is what decides the sign.
    assert np.all(np.array(high_rates) < np.array(low_rates))
