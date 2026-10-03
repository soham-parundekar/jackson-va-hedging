"""The hedging engine: instrument pricing, sizing conventions, the ledger and the attribution.

The sign tests are the ones that earn their keep. A hedging result is a small number left over
from two large ones, and a sign error anywhere in the chain produces a plausible-looking table
that is exactly wrong - a hedge that doubles the risk reports the same order of magnitude as one
that removes it. So the conventions are tested as identities: a written guarantee is hedged
short the index and long duration, a partial hedge leaves exactly its own share of every
exposure, and the attribution's pieces add to the total with nothing left over.

Four of these exist because of specific failures. The liability's delta is per move in the
contract value and the instruments' is per move in the index, and the step between them is the
sub-account's equity weight; leaving it out oversized every hedge by twenty per cent, and the
second convention arrived later - the full Greeks already differentiate in the index, so putting
them through the weight as well undersized a hedge by a sixth. The sizing ridge has to act on
normalised columns or it is scaled by the equity future's hundred
dollars of delta per unit, which left the rate exposure essentially unhedged while the delta
looked closed. And the attribution needs a bar for the contract's own anniversary, because a
Taylor expansion in market variables cannot explain a jump caused by a charge.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from tests.checks import approx, raises
from vahedge.hedge import attribution, instruments as inst, simulator, sizing, strategies
from vahedge.market.curves import ZeroCurve, fit_curve
from vahedge.valuation import lsmc
from vahedge.valuation.greeks import EQUITY_CURVATURE_STEP, Greeks

EQUITY_WEIGHT = 0.8334


def _curve():
    return fit_curve(ZeroCurve(tenors=np.arange(0.5, 30.5, 0.5),
                               zero_rates=np.linspace(0.035, 0.048, 60)))


def _smile():
    """A downward-sloping smile in strike over spot, which is what an equity index has."""
    return inst.SmileShape(
        strike_over_spot=np.array([0.6, 0.8, 0.9, 1.0, 1.1, 1.3]),
        multiplier=np.array([1.60, 1.30, 1.15, 1.00, 0.90, 0.82]),
    )


def _market(index: float = 100.0, volatility: float = 0.20):
    return inst.HedgeMarket(index=index, curve=_curve(), volatility=volatility,
                            smile=_smile(), dividend_yield=0.015, financing_rate=0.04)


def _guarantee_greeks(**overrides):
    """A written guarantee as the Greeks module reports it: a liability, so delta and rho are
    negative and the insurer is on the other side of both."""
    defaults = dict(
        value=4.0, account_value=100.0, equity_exposure=-30.0, equity_gamma=25.0,
        rho_per_bp=-0.08, vega_current=0.02, vega_long_run=0.60,
        equity_std_error=0.5, rho_std_error=0.001, vega_std_error=0.05, level_std_error=0.1,
    )
    defaults.update(overrides)
    return Greeks(**defaults)


def _proxy_greeks(**overrides):
    """The same guarantee as the regression proxy reports it: a dictionary, and its delta and
    gamma are per log move in the account value rather than in the index."""
    defaults = dict(value=4.0, delta=-30.0 / EQUITY_WEIGHT,
                    gamma=25.0 / EQUITY_WEIGHT ** 2, rho_per_bp=-0.08, vega=0.02,
                    outside_design=0.0)
    defaults.update(overrides)
    return {key: np.array([value]) for key, value in defaults.items()}


# ---------------------------------------------------------------- instruments


def test_a_future_is_worth_nothing_when_struck_and_carries_the_forward_as_delta():
    market = _market()
    future = inst.EquityFuture(maturity=0.25)
    assert approx(0.0, abs=1e-12) == future.value(market)
    assert approx(market.forward(0.25), rel=1e-12) == future.exposures(market).delta
    assert future.exposures(market).gamma == 0.0


def test_the_forward_carries_at_the_curve_less_the_dividend():
    market = _market()
    expected = 100.0 * np.exp((float(market.curve.zero(1.0)) - 0.015) * 1.0)
    assert approx(expected, rel=1e-12) == market.forward(1.0)


def test_a_put_is_priced_at_its_own_strike_s_volatility_not_the_at_the_money_level():
    """The whole point of a skew. A ten per cent out-of-the-money put priced at the
    at-the-money level is cheap by about the multiplier."""
    market = _market()
    put = inst.IndexPut(maturity=1.0, strike_over_spot=0.90)
    assert approx(0.20 * 1.15, rel=1e-12) == market.strike_volatility(0.90)
    flat = replace(market, smile=inst.SmileShape(
        strike_over_spot=np.array([0.5, 1.5]), multiplier=np.array([1.0, 1.0])))
    assert put.value(market) > put.value(flat)


def test_the_put_s_delta_carries_the_smile_term():
    """As the index moves, the strike slides along the smile and its volatility moves with it.
    Ignoring that while marking at the smile price leaves the attribution unable to reconcile,
    and at a one-year ten per cent out-of-the-money strike it halves the hedge ratio."""
    market = _market()
    put = inst.IndexPut(maturity=1.0, strike_over_spot=0.90).struck(market)
    step = 0.002
    numeric = (put.value(replace(market, index=market.index * np.exp(step)))
               - put.value(replace(market, index=market.index * np.exp(-step)))) / (2 * step)
    assert approx(numeric, rel=1e-3) == put.exposures(market).delta

    # Against the same option with no skew at all, the smile term is worth a large share of it.
    flat = replace(market, smile=inst.SmileShape(
        strike_over_spot=np.array([0.5, 1.5]), multiplier=np.array([1.0, 1.0])))
    assert abs(put.exposures(market).delta) < 0.75 * abs(put.exposures(flat).delta)


def test_a_long_put_is_long_gamma_and_long_vega():
    market = _market()
    exposures = inst.IndexPut(maturity=1.0, strike_over_spot=0.90).exposures(market)
    assert exposures.delta < 0
    assert exposures.gamma > 0
    assert exposures.vega > 0


def test_the_put_s_gamma_survives_being_differenced():
    """At a basis-point step the quantity being differenced is a part in ten million of a
    premium of order one, which is double precision's floor; the first version of this returned
    a gamma twenty times too large. Checked against a hand-rolled wide difference."""
    market = _market()
    put = inst.IndexPut(maturity=1.0, strike_over_spot=0.90).struck(market)
    step = EQUITY_CURVATURE_STEP
    here = put.value(market)
    wide = (put.value(replace(market, index=market.index * np.exp(step)))
            - 2.0 * here
            + put.value(replace(market, index=market.index * np.exp(-step)))) / step ** 2
    assert approx(wide, rel=1e-9) == put.exposures(market).gamma


def test_the_put_and_the_liability_measure_curvature_the_same_way():
    """A solve that matches one definition of gamma against another is sizing off a unit error.

    The two were different for a while - two per cent on the option, ten on the regression - and
    nothing failed, because both numbers are perfectly well conditioned. At a one-year ten per
    cent out-of-the-money strike they differ by about five per cent of the gamma, which goes
    straight into the put position.
    """
    assert lsmc.GAMMA_STEP == EQUITY_CURVATURE_STEP
    market = _market()
    put = inst.IndexPut(maturity=1.0, strike_over_spot=0.90).struck(market)
    narrow = put.exposures(market, gamma_step=0.02).gamma
    assert put.exposures(market).gamma != approx(narrow, rel=1e-3)


def test_a_receive_fixed_swap_gains_when_rates_fall():
    market = _market()
    swap = inst.InterestRateSwap(tenor=10.0, receive_fixed=True)
    assert swap.exposures(market).rho < 0
    assert inst.InterestRateSwap(tenor=10.0, receive_fixed=False).exposures(market).rho > 0
    # Its sensitivity is the annuity of its own fixed leg rather than an assumed duration.
    assert approx(-swap.annuity(market) * 1e-4, rel=1e-12) == swap.exposures(market).rho


def test_a_longer_rate_instrument_carries_more_rate_risk():
    market = _market()
    short = inst.InterestRateSwap(tenor=5.0).exposures(market).rho
    long = inst.InterestRateSwap(tenor=30.0).exposures(market).rho
    assert abs(long) > abs(short)


# ---------------------------------------------------------------- sizing conventions


def test_the_insurer_is_long_equity_through_the_guarantee():
    """Which is the sign the whole hedge hangs off. The market risk benefit is a liability when
    positive, so a rally shrinks it and the insurer gains; the hedge therefore has to be short."""
    exposure = sizing.insurer_exposures(_guarantee_greeks(), equity_weight=1.0)
    assert exposure.delta > 0
    assert exposure.rho > 0


def test_the_equity_weight_converts_the_proxy_contract_delta_into_an_index_delta():
    """A contract four-fifths in equity funds does not move one for one with the index. The
    proxy differentiates in the account value, so leaving the conversion out oversizes every
    delta hedge by one over the weight."""
    greeks = _proxy_greeks()
    unconverted = sizing.insurer_exposures(greeks, equity_weight=1.0)
    converted = sizing.insurer_exposures(greeks, equity_weight=EQUITY_WEIGHT)
    assert approx(unconverted.delta * EQUITY_WEIGHT, rel=1e-12) == converted.delta
    assert approx(unconverted.gamma * EQUITY_WEIGHT ** 2, rel=1e-12) == converted.gamma
    # Volatility and rates are the same variable on both sides, so neither is rescaled.
    assert approx(unconverted.vega, rel=1e-12) == converted.vega
    assert approx(unconverted.rho, rel=1e-12) == converted.rho


def test_the_full_greeks_are_already_in_the_index_and_are_not_converted_again():
    """greeks.compute bumps the index and divides by the index log span, so its delta is the one
    the instruments want. Rescaling it by the equity weight as well undersizes the equity hedge
    by that factor, which is a short position missing a sixth of its size - and it was what this
    function did until the two conventions were told apart."""
    greeks = _guarantee_greeks()
    for weight in (1.0, EQUITY_WEIGHT, 0.5):
        exposure = sizing.insurer_exposures(greeks, equity_weight=weight)
        assert approx(-greeks.equity_exposure, rel=1e-12) == exposure.delta
        assert approx(-greeks.equity_gamma, rel=1e-12) == exposure.gamma


def test_both_greek_sources_describe_the_same_index_exposure():
    """The identity that would have caught the convention clash. One liability, two routes to
    its exposure vector: the proxy's contract-space derivatives put through the equity weight,
    and the full Greeks straight. They have to land on the same index delta."""
    contract_delta, contract_gamma = -36.0, 36.0
    proxy = _proxy_greeks(delta=contract_delta, gamma=contract_gamma)
    full = _guarantee_greeks(equity_exposure=contract_delta * EQUITY_WEIGHT,
                             equity_gamma=contract_gamma * EQUITY_WEIGHT ** 2)
    from_proxy = sizing.insurer_exposures(proxy, equity_weight=EQUITY_WEIGHT)
    from_full = sizing.insurer_exposures(full, equity_weight=EQUITY_WEIGHT)
    assert approx(from_proxy.delta, rel=1e-12) == from_full.delta
    assert approx(from_proxy.gamma, rel=1e-12) == from_full.gamma


def test_the_long_run_vega_is_not_what_a_traded_option_is_sized_against():
    greeks = _guarantee_greeks()
    tradeable = sizing.insurer_exposures(greeks, vega="current")
    long_run = sizing.insurer_exposures(greeks, vega="long_run")
    assert abs(long_run.vega) > 10.0 * abs(tradeable.vega)
    assert sizing.insurer_exposures(greeks, vega="none").vega == 0.0
    with raises(ValueError, match="vega must be"):
        sizing.insurer_exposures(greeks, vega="whatever")


def test_a_written_guarantee_is_hedged_short_the_index():
    market, exposure = _market(), sizing.insurer_exposures(_guarantee_greeks())
    positions = sizing.solve(strategies.equity_only(), market, exposure)
    assert positions.units[0] < 0
    # Not closed to the last decimal, and that is the weighted solve doing its job rather than
    # failing: a future carries a little rate exposure through its own financing, rho is
    # weighted as heavily as delta, and with one instrument and two exposures to cover the
    # solve gives up a fraction of a per cent of the delta to take some of the rho.
    left = sizing.effectiveness(exposure, positions)
    assert left["delta_removed"] > 0.99


def test_the_rate_leg_is_actually_hedged_and_not_crushed_by_the_ridge():
    """The failure this test exists for: with the ridge on unnormalised columns it was scaled by
    the equity future's hundred dollars of delta per unit, the rate positions came back a
    thousandth of their right size, and the hedge reported a closed delta while removing one per
    cent of the rate exposure."""
    market, exposure = _market(), sizing.insurer_exposures(_guarantee_greeks())
    positions = sizing.solve(
        strategies.equity_only() + strategies.rate_instruments(), market, exposure
    )
    left = sizing.effectiveness(exposure, positions)
    assert left["rho_removed"] > 0.99
    assert left["delta_removed"] > 0.99
    assert positions.units[1] > 0          # long duration against a liability that grows as
    #                                        rates fall


def test_a_partial_hedge_leaves_its_own_share_of_every_exposure():
    """Not ninety per cent of whichever exposures the instruments happened to cover. S5 is the
    strategy that tests whether the last tenth of coverage is worth its cost, and that question
    only means something if the tenth is taken off everything."""
    market, exposure = _market(), sizing.insurer_exposures(_guarantee_greeks())
    instruments = strategies.equity_only() + strategies.rate_instruments()
    positions = sizing.solve(instruments, market, exposure, hedge_ratio=0.90)
    left = sizing.effectiveness(exposure, positions)
    assert approx(0.10 * exposure.delta, rel=1e-3) == left["delta_after"]
    assert approx(0.10 * exposure.rho, rel=1e-3) == left["rho_after"]


def test_the_unhedged_baseline_keeps_the_whole_exposure():
    market, exposure = _market(), sizing.insurer_exposures(_guarantee_greeks())
    positions = sizing.solve((), market, exposure)
    assert positions.units.size == 0
    assert approx(exposure.as_array(), rel=1e-12) == positions.residual.as_array()
    assert positions.notional(market) == 0.0


def test_a_put_leg_removes_convexity_that_futures_cannot_touch():
    market, exposure = _market(), sizing.insurer_exposures(_guarantee_greeks())
    linear = strategies.equity_only() + strategies.rate_instruments()
    without = sizing.effectiveness(exposure, sizing.solve(linear, market, exposure))
    with_puts = sizing.effectiveness(
        exposure, sizing.solve(linear + strategies.put_leg(), market, exposure)
    )
    assert approx(0.0, abs=1e-9) == without["gamma_removed"]        # nothing removed at all
    assert with_puts["gamma_removed"] > 0.5


def test_position_limits_bind_without_abandoning_the_rest_of_the_hedge():
    market, exposure = _market(), sizing.insurer_exposures(_guarantee_greeks())
    instruments = strategies.equity_only() + strategies.rate_instruments()
    limits = np.array([0.05, 1e9])
    positions = sizing.solve(instruments, market, exposure, limits=limits)
    assert approx(-0.05, rel=1e-9) == positions.units[0]
    assert sizing.effectiveness(exposure, positions)["rho_removed"] > 0.99


# ---------------------------------------------------------------- the strategy matrix


def test_the_strategy_matrix_adds_one_thing_at_a_time():
    matrix = strategies.matrix()
    assert matrix["S0"].instruments == ()
    assert len(matrix["S1"].instruments) == 1
    assert set(matrix["S1"].instruments) < set(matrix["S2"].instruments)
    assert set(matrix["S2"].instruments) < set(matrix["S3"].instruments)
    assert matrix["S4"].rebalance == "band" and matrix["S4"].band > 0
    assert matrix["S5"].hedge_ratio < 1.0
    assert matrix["S6"].overlay and not matrix["S4"].overlay


def test_only_one_rate_instrument_is_used_because_the_others_are_the_same_risk():
    """Under a one-factor short-rate model a note future, a bond forward and a swap differ only
    by a scalar, so holding two makes the design matrix singular and the split between them is
    decided by the ridge rather than by anything financial."""
    market = _market()
    columns = inst.exposure_matrix(
        (inst.RateFuture(), inst.BondForward(), inst.InterestRateSwap()), market
    )
    first = columns[:, 0] / np.linalg.norm(columns[:, 0])
    for other in range(1, 3):
        scaled = columns[:, other] / np.linalg.norm(columns[:, other])
        assert approx(first, abs=1e-12) == scaled
    assert len(strategies.rate_instruments()) == 1


def test_the_macro_overlay_is_a_size_rather_than_a_solve():
    """Handed to the least-squares fit it exploited the near-collinearity of the two legs,
    taking fifty units long against seventy-six short to manufacture gamma at a hundred and
    thirty times the account value in notional."""
    overlay = strategies.macro_leg(0.25)
    assert len(overlay) == 2
    assert approx(0.25, rel=1e-12) == overlay[0][1]
    assert approx(-0.25, rel=1e-12) == overlay[1][1]
    assert overlay[0][0].strike_over_spot > overlay[1][0].strike_over_spot
    assert not set(strategies.matrix()["S6"].instruments) & {i for i, _ in overlay}


def test_the_rebalance_calendar_counts_what_it_says():
    dates = pd.bdate_range("2022-01-03", "2022-12-30")
    daily = strategies.rebalance_dates(dates, "daily").sum()
    weekly = strategies.rebalance_dates(dates, "weekly").sum()
    monthly = strategies.rebalance_dates(dates, "monthly").sum()
    assert daily == dates.size
    assert 50 <= weekly <= 53
    assert monthly == 12
    # A band rule cannot be decided from a calendar, so every date is offered and the
    # simulator applies the band.
    assert strategies.rebalance_dates(dates, "band").all()
    with raises(ValueError, match="unknown rebalance"):
        strategies.rebalance_dates(dates, "fortnightly")


# ---------------------------------------------------------------- the fee the insurer collects


def test_the_two_fee_bases_have_opposite_equity_sensitivity():
    """The thing about this product that catches people out: the rider charge is a percentage of
    the benefit base and does not fall when markets do, while the contract charge is a
    percentage of the account value and does."""
    from vahedge.liability import cohorts
    from vahedge.liability import terms as terms_module

    book = cohorts.single_contract(
        terms_module.load()[("flex_gmwb", "single", "core")], issue_age=70,
        base_contract_charge=0.0131, fund_expense=0.0095, premium=100.0, max_age=100,
    )
    high = simulator.fee_rate(100.0, 100.0, book)
    crashed = simulator.fee_rate(100.0, 50.0, book)
    assert crashed < high
    # The benefit-base part survives the crash untouched.
    assert crashed > float(book.rider_charge_pct[0]) * 100.0


# ---------------------------------------------------------------- the attribution


def _synthetic_ledger(n: int = 60, seed: int = 4) -> pd.DataFrame:
    """A ledger with the columns the attribution reads, and nothing else going on.

    Built rather than simulated so the test is about the decomposition's arithmetic. The
    Greeks, the state and the cash are all arbitrary but internally consistent, and the total is
    whatever the net worth column says it is - which is the point: the attribution has to
    reconcile to the total it is given, not to one it recomputes.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    index = np.cumprod(1.0 + rng.normal(0.0002, 0.012, n))
    frame = pd.DataFrame(index=dates)
    frame["index"] = index
    frame["account_value"] = 100.0 * index ** EQUITY_WEIGHT
    frame["pre_account_value"] = frame["account_value"]
    frame["benefit_base"] = 100.0
    frame["variance"] = 0.04 + 0.01 * rng.normal(0, 1, n).cumsum() / 20.0
    frame["variance"] = frame["variance"].clip(lower=0.005)
    frame["implied_vol"] = np.sqrt(frame["variance"]) * 1.05
    frame["zero_10y"] = 0.04 + 0.0002 * rng.normal(0, 1, n).cumsum()
    frame["liability"] = 5.0 - 20.0 * np.log(index)
    frame["liability_pre"] = frame["liability"]
    frame["is_anniversary"] = False
    frame["delta"] = -30.0
    frame["gamma"] = 20.0
    frame["vega"] = 0.05
    frame["rho_per_bp"] = -0.08
    frame["theta"] = -1.2
    frame["hedge_delta"] = 25.0
    frame["hedge_gamma"] = 0.0
    frame["hedge_vega"] = 0.0
    frame["hedge_rho_per_bp"] = 0.0
    frame["cash_flow"] = 0.01
    frame["interest"] = 0.0
    frame["trade_cost"] = 0.0
    frame["carry_cost"] = 0.0
    frame["net_worth"] = -frame["liability"]
    frame["pnl"] = frame["net_worth"].diff().fillna(0.0)
    return frame


def test_the_attributed_pieces_and_the_residual_add_to_the_total_exactly():
    """By construction, which is the only way an attribution is worth reading. A decomposition
    that nearly adds up hides its own errors in the rounding."""
    pieces = attribution.attribute(_synthetic_ledger(), EQUITY_WEIGHT)
    names = ["delta", "basis", "gamma", "vega", "rate", "theta", "anniversary", "carry",
             "cost", "residual"]
    assert approx(pieces["total"].to_numpy(), abs=1e-12) == pieces[names].sum(axis=1).to_numpy()
    assert attribution.summarise(pieces, 100.0)["reconciliation_error"] < 1e-12


def test_a_short_gamma_book_loses_on_realised_variance_whichever_way_the_market_went():
    """The sign that makes a convexity hedge worth discussing. The insurer is short the
    guarantee's curvature, the term is one half gamma times the squared move, and a square has
    no sign to argue about."""
    pieces = attribution.attribute(_synthetic_ledger(), EQUITY_WEIGHT)
    assert pieces["gamma"].sum() < 0


def test_the_anniversary_is_its_own_bar_rather_than_residual():
    """A Taylor expansion in market variables cannot explain a jump caused by a charge. Before
    this bar existed, the three largest unexplained days across 2019 to 2021 were the three
    policy anniversaries rather than anything in the crash."""
    frame = _synthetic_ledger()
    day = 30
    frame.loc[frame.index[day], "liability_pre"] = frame["liability"].iloc[day] - 3.0
    frame.loc[frame.index[day], "is_anniversary"] = True
    pieces = attribution.attribute(frame, EQUITY_WEIGHT)
    assert approx(-3.0, rel=1e-12) == pieces["anniversary"].iloc[day]
    assert approx(0.0, abs=1e-12) == (
        pieces["anniversary"].drop(pieces.index[day]).abs().sum()
    )


def test_the_volatility_comparison_is_the_price_of_the_gamma_bar():
    frame = _synthetic_ledger()
    comparison = attribution.volatility_comparison(frame)
    assert comparison["realised_vol"] > 0
    assert approx(comparison["realised_less_implied"], rel=1e-12) == (
        comparison["realised_vol"] - comparison["implied_vol_mean"]
    )


# ---------------------------------------------------------------- the simulator


_CACHE: dict = {}


def _backtest_fixture():
    """A small proxy, a short synthetic path, and an in-force policy to roll along it.

    The path is a geometric random walk at constant volatility and a flat curve, which is the
    world a delta hedge is supposed to work in. Everything is sized for a test rather than for
    accuracy: the point is the simulator's bookkeeping, not the numbers it produces.
    """
    if "fixture" in _CACHE:
        return _CACHE["fixture"]
    from vahedge.liability import cohorts, gmwb, mortality
    from vahedge.liability import terms as terms_module
    from vahedge.market.heston_cos import HestonParameters
    from vahedge.market.scenarios import DailyPath
    from vahedge.market.simulate import Correlations, SubAccountMix, simulate
    from vahedge.valuation import lsmc

    curve = _curve()
    heston = HestonParameters(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.6)
    mix = SubAccountMix(equity=0.75, bond=0.15, balanced=0.0, money_market=0.10)
    correlations = Correlations(-0.6, 0.1)
    core = terms_module.load()[("flex_gmwb", "single", "core")]

    at_issue = cohorts.single_contract(
        core, issue_age=70, base_contract_charge=0.0131, fund_expense=0.0095,
        premium=100.0, deferral_years=3, max_age=90, lapse_rate=0.03,
    )
    horizon = int(at_issue.projection_years.max())
    survival, deaths = mortality.load("basic").rates(at_issue.attained_age, 2025, horizon, 0.5)
    market_paths = simulate(heston, __import__(
        "vahedge.market.hull_white", fromlist=["HullWhite"]
    ).HullWhite(a=0.25, sigma=0.011, curve=curve), correlations, mix,
        n_years=horizon, n_paths=3_000, seed=7)
    proxy = lsmc.fit(gmwb.project(at_issue, market_paths, survival, deaths,
                                  equity_weight=mix.equity_weight, record=True))

    policy = cohorts.single_contract(
        core, issue_age=70, base_contract_charge=0.0131, fund_expense=0.0095,
        premium=100.0, account_value=95.0, benefit_base=100.0, deferral_years=1,
        years_since_issue=2, max_age=90, lapse_rate=0.03,
    )
    policy_survival, policy_deaths = mortality.load("basic").rates(
        policy.attained_age, 2025, int(policy.projection_years.max()), 0.5
    )

    n = 420
    dates = pd.bdate_range("2023-01-02", periods=n)
    rng = np.random.default_rng(99)
    index = np.cumprod(np.concatenate([[1.0], np.exp(rng.normal(0.0002, 0.011, n - 1))]))
    years = (dates - dates[0]).days.to_numpy(dtype=float) / 365.0
    cash = np.full(n, 0.04)
    from vahedge.market.scenarios import sub_account_path
    curves = [curve] * n
    path = DailyPath(
        dates=dates, index=index,
        fund=sub_account_path(index, curves, cash, mix, years),
        curves=curves, zero_10y=np.full(n, float(curve.zero(10.0))),
        implied_vol=np.full(n, 0.20), variance=np.full(n, 0.04), cash_rate=cash,
        label="synthetic walk",
    )
    _CACHE["fixture"] = (path, policy, policy_survival, policy_deaths, proxy, mix)
    return _CACHE["fixture"]


def test_the_ledger_on_a_date_does_not_depend_on_anything_after_it():
    """The look-ahead check, and the most valuable test here. A backtest that peeks is the
    classic way to produce a hedging result nobody can repeat, and it does not announce itself:
    the numbers look plausible and are simply better than reality allowed. So the same path is
    run twice, once truncated, and every row the two have in common has to agree exactly.
    """
    path, policy, survival, deaths, proxy, mix = _backtest_fixture()
    strategy = strategies.matrix()["S2"]
    full = simulator.run(path, policy, survival, deaths, proxy, strategy, _smile(),
                         years_at_start=2.0, equity_weight=mix.equity_weight)
    cut = 300
    short = simulator.run(path.window(path.dates[0], path.dates[cut], label="cut"),
                          policy, survival, deaths, proxy, strategy, _smile(),
                          years_at_start=2.0, equity_weight=mix.equity_weight)

    common = short.ledger.index.intersection(full.ledger.index)[:-1]
    for column in ("liability", "delta", "hedge_delta", "hedge_mark", "cash", "net_worth",
                   "account_value", "benefit_base"):
        assert approx(full.ledger.loc[common, column].to_numpy(), rel=1e-12, abs=1e-12) == (
            short.ledger.loc[common, column].to_numpy()
        ), column


def test_rebalancing_more_often_leaves_less_delta_open():
    """The property that makes a rebalancing frequency a choice rather than a free parameter.
    It does not continue to zero: the floor is the proxy's own delta error, which is what the
    nested validation measures and why the frequency frontier in E3 has a floor in it."""
    path, policy, survival, deaths, proxy, mix = _backtest_fixture()
    base = strategies.matrix()["S2"]
    left = {}
    for rule in ("monthly", "weekly", "daily"):
        run = simulator.run(path, policy, survival, deaths, proxy,
                            base.with_rebalance(rule), _smile(),
                            years_at_start=2.0, equity_weight=mix.equity_weight)
        left[rule] = run.summary["mean_abs_delta_left_pct"]
    assert left["daily"] < left["weekly"] < left["monthly"]


def test_the_ledger_reconciles_cash_the_hedge_and_the_liability_into_net_worth():
    path, policy, survival, deaths, proxy, mix = _backtest_fixture()
    run = simulator.run(path, policy, survival, deaths, proxy, strategies.matrix()["S3"],
                        _smile(), years_at_start=2.0, equity_weight=mix.equity_weight)
    ledger = run.ledger
    assert approx(ledger["net_worth"].to_numpy(), rel=1e-12, abs=1e-12) == (
        ledger["cash"] + ledger["hedge_mark"] - ledger["liability"]
    ).to_numpy()
    assert approx(float(ledger["pnl"].sum()), rel=1e-9, abs=1e-9) == float(
        ledger["net_worth"].iloc[-1] - ledger["net_worth"].iloc[0]
    )


def test_a_hedge_removes_more_of_the_daily_variation_than_it_leaves():
    """The weakest claim worth asserting about a hedge, and the one that catches a sign error:
    on a walk with no drift to speak of, hedging delta and rho has to reduce the day-to-day
    variation of the result rather than add to it."""
    path, policy, survival, deaths, proxy, mix = _backtest_fixture()
    matrix = strategies.matrix()
    unhedged = simulator.run(path, policy, survival, deaths, proxy, matrix["S0"], _smile(),
                             years_at_start=2.0, equity_weight=mix.equity_weight)
    hedged = simulator.run(path, policy, survival, deaths, proxy, matrix["S2"], _smile(),
                           years_at_start=2.0, equity_weight=mix.equity_weight)
    assert hedged.summary["pnl_sd_pct"] < 0.5 * unhedged.summary["pnl_sd_pct"]
    assert hedged.summary["mean_abs_delta_left_pct"] < 0.1 * (
        unhedged.summary["mean_abs_delta_left_pct"]
    )


def test_a_solved_put_rolls_when_its_strike_drifts_but_an_overlay_does_not():
    """The difference between a hedge and a floor, and a defect the macro frontier found.

    A put the solve uses wants to stay near its target moneyness. A macro spread bought at 80% of
    spot is meant to pay when the index falls 20%, and at that moment its held strike is at the
    money - a drift of 0.20 against a band of 0.05. Rolling there closes the position at the first
    sign of the event it was bought for and re-strikes 20% below the new spot.
    """
    market = _market()
    put = inst.IndexPut(maturity=2.0, strike_over_spot=0.80).struck(market)
    crashed = replace(market, index=market.index * 0.75)
    solved = simulator._Position(instrument=put, units=1.0, reference=None,
                                 opened_at=0.0, role="solved")
    overlay = simulator._Position(instrument=put, units=1.0, reference=None,
                                  opened_at=0.0, role="overlay")
    assert simulator._needs_roll(solved, crashed, 0.1)
    assert not simulator._needs_roll(overlay, crashed, 0.1)


def test_an_overlay_still_rolls_on_time():
    """Held through the event, not held past expiry."""
    market = _market()
    put = inst.IndexPut(maturity=2.0, strike_over_spot=0.80).struck(market)
    overlay = simulator._Position(instrument=put, units=1.0, reference=None,
                                  opened_at=0.0, role="overlay")
    assert not simulator._needs_roll(overlay, market, 1.0)
    assert simulator._needs_roll(overlay, market, 1.6)
