"""The hedging experiments: crisis replays, the cost frontier, and the cost of model risk.

Four of the six experiments the project design calls for are here, and the two that are not are
not ready rather than skipped: the real-world Monte Carlo and the variable-annuity-plus-RILA
netting both need the capital work in W5 to say anything, and the rider-economics experiment
closes the loop with the break-even fee, which is the same.

E1, crisis replays. Every strategy over every stress window the free data reaches, with the
unhedged baseline beside it and the attribution behind it. The windows are chosen by what
happened rather than by what makes a hedge look good, and they are deliberately different in
kind: a volatility shock with little index damage, a grinding fall into a year end, the fastest
crash on record, and the one year in the sample when equity and rates fell together.

E3, the frequency and cost frontier. Residual standard deviation against total cost for daily,
weekly, monthly and band rebalancing, with every cost assumption at half, one and two times its
base level. A conclusion that survives the sweep is worth something; one quoted at a single cost
level is not, because the costs are assumptions.

E4, model misspecification. The world is whatever history did; the hedge is sized from Greeks
that are deliberately wrong. Two kinds of wrong: a flat-volatility model, which has no skew and
so understates how much a guarantee moves in a fall, and the right model with the wrong
long-run variance, which is the parameter no option expires at and the one the valuation is most
exposed to. The extra residual is the cost of model risk in hedging, and it is the honest answer
to "how much does the calibration matter", which no amount of fit diagnostics can give.

Two tables come out alongside. The first is the share of rebalances on which the proxy was asked
for a state outside its design, which is not an experiment but a condition on all of them: a
hedging result built on a quarter of its decisions being extrapolated is not a result. The second
compares the two places the liability's curvature can come from, since the regression's second
derivative is wrong by about its own size and the strategies holding puts are sized off a nested
surface instead. That substitution has to earn its keep, and the table is where it does or
does not.

Usage:  python -m scripts.run_hedge_experiments
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from vahedge import paths
from vahedge.hedge import attribution, instruments as inst, simulator, strategies
from vahedge.liability import cohorts, gmwb, mortality
from vahedge.liability import terms as terms_module
from vahedge.market import scenarios
from vahedge.market import state as market_state
from vahedge.market.heston_cos import HestonParameters
from vahedge.market.simulate import simulate
from vahedge.capital import statutory
from vahedge.valuation import lsmc
from vahedge.valuation.convexity import ConvexitySurface, with_nested_gamma
from vahedge.valuation.engine import MarketState

ISSUE_AGE = 70
DEFERRAL_YEARS = 5
MAX_AGE = 105
PREMIUM = 100.0
# The policy the backtest follows: three years in force, and started just below its step-up
# ceiling rather than exactly at it, because a contract whose account equals its benefit base
# at an anniversary is a state the recursion cannot produce - the charges come out after the
# step-up.
DURATION_AT_START = 3
OPENING_MONEYNESS = 0.94

FIT_PATHS = 20_000
FIT_SEED = 20251231
DIVIDEND_YIELD = 0.015        # assumption; the S&P 500 series on FRED is a price index
STRATEGY_ORDER = ("S0", "S1", "S2", "S3", "S4", "S5", "S6")
COST_MULTIPLES = (0.5, 1.0, 2.0)
FREQUENCIES = ("daily", "weekly", "monthly")
# The option leg's two free parameters. The short end brackets the 0.24-year average term of
# Jackson's own equity option book at 31 December 2025; the long end is what S3 assumes.
PUT_TENORS = (0.25, 0.5, 1.0)
PUT_STRIKES = (0.95, 0.90, 0.85)


def build(backtest_start: str = "2016-09-26"):
    """The proxy, the policy and the replay path.

    The proxy is fitted off the curve as at the start of the backtest rather than at the
    valuation date, and that is not a detail. A hedging programme that ran from 2016 was priced
    off 2016's rates; a proxy fitted off December 2025's curve has a design distribution for
    the ten-year zero of roughly three to six and a half per cent, while the decade being
    replayed spent most of its time between half a per cent and three. Seventy per cent of the
    backtest's days were being valued by a fit that had never simulated a rate anywhere near
    them. Refitting off the start of the window is what a programme would actually have done.

    The Heston parameters stay at the current calibration because no free historical option
    data exists to recalibrate them, and that is a stated approximation rather than a choice:
    the volatility level along the path comes from the index volatility series, so what is held
    fixed is the shape of the surface and the speed of mean reversion, not the level.
    """
    calibration = market_state.load()
    panel = pd.read_csv(paths.FRED_PANEL, comment="#", parse_dates=["date"]).set_index("date")
    opening_curve = market_state.treasury_curve(panel, pd.Timestamp(backtest_start))
    calibration = replace(calibration, curve=opening_curve)
    state = MarketState.from_calibration(calibration)
    core = terms_module.load()[("flex_gmwb", "single", "core")]

    at_issue = cohorts.single_contract(
        core, issue_age=ISSUE_AGE, base_contract_charge=0.0131, fund_expense=0.0095,
        premium=PREMIUM, deferral_years=DEFERRAL_YEARS, max_age=MAX_AGE,
        lapse_rate=0.04, lapse_beta=1.2, lapse_floor=0.01,
    )
    horizon = int(at_issue.projection_years.max())
    survival, deaths = mortality.load("basic").rates(
        at_issue.attained_age, state.valuation_year, horizon, 0.5
    )
    market_paths = simulate(state.heston, state.hull_white(), state.correlations, state.mix,
                            n_years=horizon, n_paths=FIT_PATHS, seed=FIT_SEED)
    proxy = lsmc.fit(gmwb.project(at_issue, market_paths, survival, deaths,
                                  equity_weight=state.mix.equity_weight, record=True))

    policy = cohorts.single_contract(
        core, issue_age=ISSUE_AGE, base_contract_charge=0.0131, fund_expense=0.0095,
        premium=PREMIUM, account_value=PREMIUM * OPENING_MONEYNESS, benefit_base=PREMIUM,
        deferral_years=max(DEFERRAL_YEARS - DURATION_AT_START, 0),
        years_since_issue=DURATION_AT_START, max_age=MAX_AGE,
        lapse_rate=0.04, lapse_beta=1.2, lapse_floor=0.01,
    )
    policy_survival, policy_deaths = mortality.load("basic").rates(
        policy.attained_age, state.valuation_year, int(policy.projection_years.max()), 0.5
    )

    history = scenarios.load_history(panel, calibration.heston, calibration.mix,
                                     dividend_yield=DIVIDEND_YIELD)
    smile = inst.smile_from_heston(calibration.heston, calibration.curve, maturity=1.0)
    surface = (ConvexitySurface.from_csv(paths.GAMMA_SURFACE)
               if paths.GAMMA_SURFACE.exists() else None)
    return {
        "calibration": calibration, "state": state, "proxy": proxy, "policy": policy,
        "survival": policy_survival, "deaths": policy_deaths,
        "history": history, "smile": smile, "surface": surface,
    }


def _run(setup, path, strategy, cost_multiple: float = 1.0, greeks_override=None,
         nested_gamma: bool = True):
    """One strategy over one path.

    ``nested_gamma`` decides where the liability's curvature comes from. The proxy's own second
    derivative is off by about the size of the quantity at every horizon, so by default it is
    replaced by the tabulated surface built in scripts/run_convexity_surface.py - for every
    strategy, not only the ones holding puts, because the curvature is a property of the
    liability rather than of the hedge. Futures and swaps carry no gamma, so on S1 and S2 the
    substitution changes the target and not the trade.

    E4 turns it off. Its arms are all misspecified in a named way and measured against the
    calibrated arm, and a surface built under the calibrated model would hand every arm one
    correct Greek, which is the opposite of what that experiment is asking.
    """
    source = greeks_override or setup["proxy"]
    if nested_gamma and setup["surface"] is not None:
        source = with_nested_gamma(source, setup["surface"])
    elif greeks_override is None:
        source = None             # the simulator's own default path
    return simulator.run(
        path, setup["policy"], setup["survival"], setup["deaths"], setup["proxy"], strategy,
        setup["smile"], years_at_start=float(DURATION_AT_START),
        equity_weight=setup["state"].mix.equity_weight, dividend_yield=DIVIDEND_YIELD,
        cost_multiple=cost_multiple, greeks_override=source,
    )


def crisis_replays(setup) -> tuple:
    """E1. Every strategy on every stress window the data reaches, with the attribution."""
    episodes = scenarios.available_episodes(setup["history"])
    matrix = strategies.matrix()
    rows, bars = [], []
    for name, (start, end, why) in episodes["covered"].items():
        path = setup["history"].window(start, end, label=name)
        baseline = None
        for key in STRATEGY_ORDER:
            run = _run(setup, path, matrix[key])
            pieces = attribution.attribute(run.ledger, setup["state"].mix.equity_weight)
            summary = run.summary
            if key == "S0":
                baseline = summary
            rows.append({
                "episode": name, "why": why, "strategy": key,
                "days": summary["days"], "rebalances": summary["rebalances"],
                "index_return": float(path.index[-1] - 1.0),
                "fund_return": float(path.fund[-1] - 1.0),
                "rate_move_bp": float(1e4 * (path.zero_10y[-1] - path.zero_10y[0])),
                "peak_implied_vol": float(path.implied_vol.max()),
                "total_pnl_pct": summary["total_pnl_pct"],
                "pnl_sd_pct": summary["pnl_sd_pct"],
                "worst_day_pct": summary["worst_day_pct"],
                "worst_drawdown_pct": summary["worst_drawdown_pct"],
                "cost_pct": summary["total_cost_pct"],
                "delta_left_pct": summary["mean_abs_delta_left_pct"],
                # The number the experiment exists for: how much of the unhedged variation the
                # strategy removed. Reported on the standard deviation rather than on the total,
                # because a total over a few weeks is one draw and a standard deviation is not.
                "variance_reduction": 1.0 - (summary["pnl_sd_pct"] / baseline["pnl_sd_pct"]) ** 2
                if baseline["pnl_sd_pct"] > 0 else np.nan,
                "outside_design_share": run.summary["outside_design_share"],
            })
            bars.append({
                "episode": name, "strategy": key,
                **attribution.summarise(pieces, PREMIUM),
                **attribution.volatility_comparison(run.ledger),
            })
    return pd.DataFrame(rows), pd.DataFrame(bars), episodes


def frequency_and_cost(setup) -> pd.DataFrame:
    """E3. Residual variation against what it cost to get there.

    Run on the whole replay window rather than on a crisis, because a frontier is a statement
    about the average cost of a rebalancing rule and a crisis is the opposite of average.
    """
    matrix = strategies.matrix()
    window = setup["history"]
    rows = []
    for key in ("S1", "S2", "S3"):
        for rule in FREQUENCIES:
            for multiple in COST_MULTIPLES:
                run = _run(setup, window, matrix[key].with_rebalance(rule), multiple)
                rows.append({
                    "strategy": key, "rebalance": rule, "cost_multiple": multiple,
                    "rebalances": run.summary["rebalances"],
                    "pnl_sd_pct": run.summary["pnl_sd_pct"],
                    "total_cost_pct": run.summary["total_cost_pct"],
                    "total_pnl_pct": run.summary["total_pnl_pct"],
                    "delta_left_pct": run.summary["mean_abs_delta_left_pct"],
                    "outside_design_share": run.summary["outside_design_share"],
                })
        for multiple in COST_MULTIPLES:
            banded = matrix[key].with_rebalance("band", strategies.BAND_SHARE)
            run = _run(setup, window, banded, multiple)
            rows.append({
                "strategy": key, "rebalance": f"band {strategies.BAND_SHARE:.1%}",
                "cost_multiple": multiple, "rebalances": run.summary["rebalances"],
                "pnl_sd_pct": run.summary["pnl_sd_pct"],
                "total_cost_pct": run.summary["total_cost_pct"],
                "total_pnl_pct": run.summary["total_pnl_pct"],
                "delta_left_pct": run.summary["mean_abs_delta_left_pct"],
                "outside_design_share": run.summary["outside_design_share"],
            })
    return pd.DataFrame(rows)


def put_design(setup) -> pd.DataFrame:
    """E3, second half. What the option leg's tenor and strike cost, which S3 asserts.

    S3's first version held a one-year put ten per cent out of the money and nothing chose those
    numbers; the comment in strategies.py had promised a sweep since it was written. This is it.
    The quantity that drives the answer is gamma per unit of trading cost, because the spread on
    a listed option is charged on its vega and a short-dated option carries far more curvature per
    unit of vega than a long one.

    Two measurements per design, because one of them alone picks the wrong point, and that is not
    a hypothetical. On the whole replay window's two axes the three one-year designs are all
    dominated and the sweep reads as a clean verdict for short tenors. Add the worst window in the
    sample as a third axis and nothing is dominated: the long-dated puts pay more through the
    crash, which is what they were bought for. The first version of this function measured only
    the window and the conclusion it supported was an artefact of the ten years in which nothing
    happened.

    The disclosure is the reason this is worth running rather than a tidy-up. Jackson's equity
    option book averaged 0.24 years of remaining term at 31 December 2025 - three months, against
    the twelve first assumed here - so a sweep that puts the short tenors anywhere near the
    frontier agrees with what the issuer does, and one that does not needs an explanation.
    """
    matrix = strategies.matrix()
    base = matrix["S3"]
    linear = strategies.equity_only() + strategies.rate_instruments()
    episodes = scenarios.available_episodes(setup["history"])
    stress_name = max(
        episodes["covered"],
        key=lambda name: -setup["history"].window(
            *episodes["covered"][name][:2], label=name).index[-1],
    )
    stress = setup["history"].window(*episodes["covered"][stress_name][:2], label=stress_name)
    rows = []
    for tenor in PUT_TENORS:
        for strike in PUT_STRIKES:
            variant = replace(base, instruments=linear + strategies.put_leg(tenor, strike))
            run = _run(setup, setup["history"], variant)
            crisis = _run(setup, stress, variant)
            put = inst.IndexPut(maturity=tenor, strike_over_spot=strike)
            reference = inst.HedgeMarket(
                index=100.0, curve=setup["calibration"].curve, volatility=0.18,
                smile=inst.smile_from_heston(setup["calibration"].heston,
                                             setup["calibration"].curve, maturity=tenor),
            )
            exposure = put.struck(reference).exposures(reference)
            rows.append({
                "tenor_years": tenor, "strike_over_spot": strike,
                "pnl_sd_pct": run.summary["pnl_sd_pct"],
                "total_cost_pct": run.summary["total_cost_pct"],
                "total_pnl_pct": run.summary["total_pnl_pct"],
                "worst_day_pct": run.summary["worst_day_pct"],
                "rebalances": run.summary["rebalances"],
                "stress_episode": stress_name,
                "stress_pnl_sd_pct": crisis.summary["pnl_sd_pct"],
                "stress_total_pnl_pct": crisis.summary["total_pnl_pct"],
                "stress_worst_day_pct": crisis.summary["worst_day_pct"],
                # Why the sweep comes out the way it does, at one reference market rather than
                # along the path: curvature bought per dollar of spread paid.
                "gamma_per_cost": exposure.gamma / (exposure.vega * inst.OPTION_COST_VOL_POINTS),
                "premium_pct_of_index": put.struck(reference).value(reference) / 100.0,
            })
    table = pd.DataFrame(rows)
    table["on_frontier"] = [
        not ((table["total_cost_pct"] <= row["total_cost_pct"])
             & (table["pnl_sd_pct"] <= row["pnl_sd_pct"])
             & (table["stress_total_pnl_pct"] >= row["stress_total_pnl_pct"])
             & (table.index != index)).any()
        for index, row in table.iterrows()
    ]
    return table


def surrender_floor(setup) -> pd.DataFrame:
    """What the cash surrender value floor costs a working hedge, over the replay window.

    The hedge is sized against the economic liability and the 8-K says what that costs when the
    statutory reserve cannot follow it: "non-economic hedging costs". The mechanism is a sign
    change rather than a level. Unfloored, capital is cash plus the hedge mark less the guarantee
    and a working hedge keeps it flat. Floored, the reserve tracks the surrender value, the
    guarantee leaves the reserve entirely, and the hedge's loss in a rally has two per cent of the
    account rise against it instead of the guarantee getting cheaper.

    Run on the whole window rather than on a crisis, because the floor is a rally problem. 2016 to
    2026 is one long rally with the guarantee a net asset most of the way through, which is the
    regime that made the floor worth a captive.
    """
    matrix = strategies.matrix()
    rows = []
    for key in STRATEGY_ORDER:
        run = _run(setup, setup["history"], matrix[key])
        marked = statutory.statutory_capital(run.ledger)
        rows.append({
            "strategy": key,
            **statutory.floor_summary(marked, float(setup["policy"].account_value[0])),
        })
    table = pd.DataFrame(rows).set_index("strategy")
    # Against the unhedged book on each basis separately, which is the only comparison that
    # separates the two. The level gap cannot: the hedge cancels out of it.
    for basis in ("economic", "statutory"):
        base = table.loc["S0", f"{basis}_sd_pct"]
        table[f"{basis}_variance_removed"] = 1.0 - (table[f"{basis}_sd_pct"] / base) ** 2
    return table.reset_index()


def convexity_source(setup) -> pd.DataFrame:
    """What sizing the option leg off nested curvature is worth, against the regression's own.

    The surface costs a few minutes of nested valuation and a file on disk, and the case for it
    is an error measurement rather than a hedging result: the proxy's gamma is wrong by roughly
    its own size. Whether that matters to a hedge is a separate question, because gamma enters
    the solve at a tenth of the weight of delta and the put leg is the only instrument that
    answers to it. This runs the strategies that hold puts both ways over the same paths.

    Run only where the surface exists, which is the usual case; without it the experiments fall
    back to the regression and this table is empty rather than fabricated.
    """
    if setup["surface"] is None:
        return pd.DataFrame()
    matrix = strategies.matrix()
    episodes = scenarios.available_episodes(setup["history"])
    rows = []
    for name in list(episodes["covered"]) + ["whole window"]:
        path = (setup["history"] if name == "whole window"
                else setup["history"].window(*episodes["covered"][name][:2], label=name))
        for key in ("S3", "S4", "S5", "S6"):
            both = {
                source: _run(setup, path, matrix[key], nested_gamma=(source == "nested"))
                for source in ("regression", "nested")
            }
            rows.append({
                "episode": name, "strategy": key,
                "sd_regression_pct": both["regression"].summary["pnl_sd_pct"],
                "sd_nested_pct": both["nested"].summary["pnl_sd_pct"],
                "cost_regression_pct": both["regression"].summary["total_cost_pct"],
                "cost_nested_pct": both["nested"].summary["total_cost_pct"],
                "worst_day_regression_pct": both["regression"].summary["worst_day_pct"],
                "worst_day_nested_pct": both["nested"].summary["worst_day_pct"],
            })
    table = pd.DataFrame(rows)
    table["sd_change_pct"] = table["sd_nested_pct"] - table["sd_regression_pct"]
    table["cost_change_pct"] = table["cost_nested_pct"] - table["cost_regression_pct"]
    return table


def _wrong_model_greeks(setup, heston: HestonParameters, fixed_variance: float | None = None):
    """Greeks from a proxy fitted under a different market model, on the same contract.

    This is what makes E4 a measurement rather than an assertion. The hedge is sized from a
    model that is wrong in a named way and marked against the same realised history, so the
    extra residual is the cost of that particular error - not of model risk in the abstract.

    ``fixed_variance`` is what makes the flat-volatility variant an honest one. A model with no
    volatility of volatility produces a design distribution for the variance that is a single
    point, so feeding it the realised variance asks it about states it has never seen and the
    answer is extrapolation rather than misspecification - the first version of this reported an
    extra standard deviation of five and a half percentage points a day against a calibrated
    hedge's two tenths, which is not a finding about model risk. A hedger with no stochastic
    volatility does not observe a volatility state in the first place: they compute their Greeks
    at their own constant level and leave them there. Passing that level here is what the
    experiment is supposed to mean.
    """
    state = setup["state"]
    core = terms_module.load()[("flex_gmwb", "single", "core")]
    at_issue = cohorts.single_contract(
        core, issue_age=ISSUE_AGE, base_contract_charge=0.0131, fund_expense=0.0095,
        premium=PREMIUM, deferral_years=DEFERRAL_YEARS, max_age=MAX_AGE,
        lapse_rate=0.04, lapse_beta=1.2, lapse_floor=0.01,
    )
    horizon = int(at_issue.projection_years.max())
    survival, deaths = mortality.load("basic").rates(
        at_issue.attained_age, state.valuation_year, horizon, 0.5
    )
    wrong = replace(state, heston=heston,
                    correlations=replace(state.correlations, equity_variance=heston.rho))
    market_paths = simulate(wrong.heston, wrong.hull_white(), wrong.correlations, wrong.mix,
                            n_years=horizon, n_paths=FIT_PATHS, seed=FIT_SEED)
    proxy = lsmc.fit(gmwb.project(at_issue, market_paths, survival, deaths,
                                  equity_weight=wrong.mix.equity_weight, record=True))
    if fixed_variance is None:
        return lambda **kwargs: proxy.greeks_at(**kwargs)

    def at_own_variance(**kwargs):
        return proxy.greeks_at(**{**kwargs, "variance": fixed_variance})

    return at_own_variance


def model_risk(setup) -> pd.DataFrame:
    """E4. Hedge the realised world with Greeks from a model that is wrong on purpose."""
    calibrated = setup["calibration"].heston
    variants = {
        "calibrated": (None, None),
        # No skew and no volatility of volatility: the flat-volatility world, which is what a
        # Black-Scholes hedge assumes. Matched on the starting variance so the two models agree
        # on today's at-the-money level and disagree about everything else.
        "flat volatility": (HestonParameters(
            v0=calibrated.v0, kappa=4.0, theta=calibrated.v0, xi=0.01, rho=-0.01,
        ), calibrated.v0),
        # The right model with the long-run variance put where ten years of realised history
        # would put it rather than where the option surface does. That is the parameter no
        # listed option expires at and the one a forty-year liability is most exposed to.
        "long-run vol 17%": (replace(calibrated, theta=0.17 ** 2), None),
        "long-run vol 26%": (replace(calibrated, theta=0.26 ** 2), None),
    }
    matrix = strategies.matrix()
    episodes = scenarios.available_episodes(setup["history"])
    rows = []
    for label, (heston, fixed) in variants.items():
        override = (None if heston is None
                    else _wrong_model_greeks(setup, heston, fixed))
        for name in list(episodes["covered"]) + ["whole window"]:
            path = (setup["history"] if name == "whole window"
                    else setup["history"].window(*episodes["covered"][name][:2], label=name))
            for key in ("S1", "S2", "S3"):
                run = _run(setup, path, matrix[key], greeks_override=override,
                           nested_gamma=False)
                rows.append({
                    "model": label, "episode": name, "strategy": key,
                    "pnl_sd_pct": run.summary["pnl_sd_pct"],
                    "total_pnl_pct": run.summary["total_pnl_pct"],
                    "worst_day_pct": run.summary["worst_day_pct"],
                    "delta_left_pct": run.summary["mean_abs_delta_left_pct"],
                    "outside_design_share": run.summary["outside_design_share"],
                })
    table = pd.DataFrame(rows)
    base = table[table["model"] == "calibrated"].set_index(["episode", "strategy"])
    table["extra_sd_pct"] = [
        row["pnl_sd_pct"] - float(base.loc[(row["episode"], row["strategy"]), "pnl_sd_pct"])
        for _, row in table.iterrows()
    ]
    return table


def main() -> None:
    paths.ensure_output_dirs()
    setup = build()

    replays, bars, episodes = crisis_replays(setup)
    replays.to_csv(paths.TABLES / "hedge_crisis_replays.csv", index=False)
    bars.to_csv(paths.TABLES / "hedge_attribution.csv", index=False)
    pd.DataFrame([
        {"episode": name, "start": start, "end": end, "why": why, "covered": True}
        for name, (start, end, why) in episodes["covered"].items()
    ] + [
        {"episode": name, "start": start, "end": end, "why": why, "covered": False,
         "dates_available": inside, "dates_needed": span}
        for name, (start, end, why, inside, span) in episodes["missing"].items()
    ]).to_csv(paths.TABLES / "hedge_episodes.csv", index=False)

    print("E1 crisis replays")
    for name in episodes["covered"]:
        block = replays[replays["episode"] == name]
        head = block.iloc[0]
        print(f"\n  {name}: index {100*head['index_return']:+.1f}%  "
              f"fund {100*head['fund_return']:+.1f}%  rates {head['rate_move_bp']:+.0f}bp  "
              f"peak implied vol {100*head['peak_implied_vol']:.0f}%  {head['days']:.0f} days")
        for _, row in block.iterrows():
            print(f"    {row['strategy']}  total {100*row['total_pnl_pct']:+7.2f}%  "
                  f"sd {100*row['pnl_sd_pct']:6.3f}%  worst day {100*row['worst_day_pct']:+7.2f}%  "
                  f"variance removed {100*row['variance_reduction']:5.1f}%  "
                  f"cost {100*row['cost_pct']:5.2f}%  "
                  f"delta left {100*row['delta_left_pct']:5.2f}%  "
                  f"outside design {100*row['outside_design_share']:4.0f}%")
    if episodes["missing"]:
        print("\n  not reachable with the free index history: "
              + ", ".join(episodes["missing"]))

    frontier = frequency_and_cost(setup)
    frontier.to_csv(paths.TABLES / "hedge_frequency_frontier.csv", index=False)
    print("\nE3 frequency and cost, over the whole replay window")
    base = frontier[frontier["cost_multiple"] == 1.0]
    for key in ("S1", "S2", "S3"):
        for _, row in base[base["strategy"] == key].iterrows():
            print(f"  {key} {row['rebalance']:<10s} {row['rebalances']:4.0f} trades  "
                  f"sd {100*row['pnl_sd_pct']:6.3f}%  cost {100*row['total_cost_pct']:6.2f}%  "
                  f"delta left {100*row['delta_left_pct']:5.2f}%  "
                  f"outside design {100*row['outside_design_share']:4.0f}%")

    design = put_design(setup)
    design.to_csv(paths.TABLES / "hedge_put_design.csv", index=False)
    stress_name = design["stress_episode"].iloc[0]
    print(f"\nE3, option leg: the put's tenor and strike, over the whole window and in "
          f"{stress_name}")
    for _, row in design.sort_values(["tenor_years", "strike_over_spot"]).iterrows():
        print(f"  {row['tenor_years']:.2f}y  K/S {row['strike_over_spot']:.2f}  "
              f"sd {100*row['pnl_sd_pct']:6.3f}%  cost {100*row['total_cost_pct']:6.2f}%  "
              f"{stress_name} total {100*row['stress_total_pnl_pct']:+6.2f}% "
              f"worst day {100*row['stress_worst_day_pct']:+6.2f}%  "
              f"gamma per unit of cost {row['gamma_per_cost']:7.0f}"
              f"{'  <- frontier' if row['on_frontier'] else ''}")
    held = design[(design.tenor_years == strategies.PUT_TENOR_YEARS)
                  & (design.strike_over_spot == strategies.PUT_STRIKE)]
    if not held.empty:
        row = held.iloc[0]
        print(f"  S3 holds {row['tenor_years']:.2f}y at K/S {row['strike_over_spot']:.2f}: "
              f"on the frontier {bool(row['on_frontier'])}")
    dominated = design[~design["on_frontier"]]
    if not dominated.empty:
        print("  dominated on all three of cost, residual and the stress outcome: "
              + ", ".join(f"{r['tenor_years']:.2f}y/{r['strike_over_spot']:.2f}"
                          for _, r in dominated.iterrows()))

    floor = surrender_floor(setup)
    floor.to_csv(paths.TABLES / "hedge_surrender_floor.csv", index=False)
    print("\nThe cash surrender value floor against the hedge, over the whole window")
    print(f"  floor binds on {100*floor['share_of_days_floored'].iloc[0]:.0f}% of days; it costs "
          f"the block {100*floor['basis_gap_total_pct'].iloc[0]:+.2f}% of account value over ten "
          f"years ({100*floor['basis_gap_per_year_pct'].iloc[0]:+.2f}%/yr), the same whatever the "
          f"hedge, because the hedge mark is in both capital series")
    print("  variance each strategy removes, by basis:")
    for _, row in floor.iterrows():
        print(f"  {row['strategy']}  economic total {100*row['economic_total_pct']:+7.2f}%  "
              f"statutory {100*row['statutory_total_pct']:+7.2f}%   "
              f"sd {100*row['economic_sd_pct']:5.3f}% -> {100*row['statutory_sd_pct']:5.3f}%   "
              f"variance removed {100*row['economic_variance_removed']:5.1f}% economic vs "
              f"{100*row['statutory_variance_removed']:5.1f}% statutory")

    sources = convexity_source(setup)
    if not sources.empty:
        sources.to_csv(paths.TABLES / "hedge_convexity_source.csv", index=False)
        print("\nCurvature from the nested surface against the regression's own, on the "
              "strategies that hold puts")
        for _, row in sources[sources["episode"] == "whole window"].iterrows():
            print(f"  {row['strategy']}  sd {100*row['sd_regression_pct']:6.3f}% -> "
                  f"{100*row['sd_nested_pct']:6.3f}%   "
                  f"cost {100*row['cost_regression_pct']:5.2f}% -> "
                  f"{100*row['cost_nested_pct']:5.2f}%")
        worst = sources.loc[sources["sd_change_pct"].abs().idxmax()]
        print(f"  largest move in any episode: {worst['strategy']} in {worst['episode']}, "
              f"sd {100*worst['sd_change_pct']:+.3f} points")
    else:
        print("\nNo curvature surface on disk; the option leg is sized off the regression. "
              "Run scripts/run_convexity_surface.py first.")

    misspecified = model_risk(setup)
    misspecified.to_csv(paths.TABLES / "hedge_model_risk.csv", index=False)
    print("\nE4 model risk, extra standard deviation against the calibrated hedge")
    pivot = misspecified[misspecified["episode"] == "whole window"].pivot(
        index="model", columns="strategy", values="extra_sd_pct"
    )
    print((100 * pivot).round(4).to_string())
    print(f"\nwrote eight tables under {paths.TABLES}")


if __name__ == "__main__":
    main()
