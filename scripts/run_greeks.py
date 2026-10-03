"""The risk numbers a desk would hedge against, and how they change as the book moves.

Four tables. The first two are the contract at issue; the last two are the reason a single-policy
Greek is not enough on its own.

*Greeks on both mortality bases.* The economic basis is what the hedge is sized on; the reporting
basis carries the margins the fair value includes. Every Greek is reported with the standard error
of its own bumped difference rather than of the two levels, because the levels are a hundred-odd
dollars noisy at twenty thousand paths and the paired difference is not. A rho quoted without that
number is not checkable.

*Positions.* What the Greeks mean in instruments: a short index notional, a swap DV01 and the
notional that carries it, and a volatility exposure split into the part a listed option reaches
and the part it does not.

*The moneyness profile.* One policy at issue sits at a benefit base equal to its account value.
Jackson's book does not: after a decade of rising markets most contracts sit well below one on
that ratio. The profile is what lets a disclosed sensitivity be located on the model's own curve
instead of being compared to a single point it was never going to match.

*Step-up proximity.* The Core option steps the benefit base up to the contract value on the
anniversary, so for a contract already above its benefit base the index level on that one date
fixes the guaranteed income for life. Approaching it the insurer stops being short the index and
becomes long it, because a higher index means a permanently larger guarantee to fund. This is read
off the regression proxy rather than the engine, since the proxy is what the hedge actually
consults between anniversaries, and its pre-event and post-event fits are what make a fractional
policy time meaningful at all.

Usage:  python -m scripts.run_greeks
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vahedge import paths
from vahedge.hedge import instruments as inst
from vahedge.liability import gmwb, mortality
from vahedge.market.simulate import simulate
from vahedge.valuation import greeks as greeks_module
from vahedge.valuation import lsmc
from vahedge.valuation.engine import Valuer

from scripts.run_valuation import (
    DEFERRAL_YEARS,
    ISSUE_AGE,
    MAX_AGE,
    PREMIUM,
    base_valuation,
    build,
)

EQUITY_BUMP = 0.01
RATE_BUMP_BP = 10.0
VOL_BUMP = 0.01
SWAP_TENOR_YEARS = 10.0
# Benefit base over account value. One is a contract at issue; the disclosed book sits below it
# after a decade of rising markets, and the upper end is where a contract whose account has been
# drained by withdrawals ends up.
MONEYNESS_GRID = (0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0)
# Fractions of a policy year still to run before the next anniversary.
TIME_TO_ANNIVERSARY = (0.9, 0.5, 0.1)
# Account value over benefit base. 0.90 sits inside both of the proxy's fit families; 1.20 sits
# inside the pre-event family alone, because a post-event state cannot have an account above its
# base once the step-up has run.
ACCOUNT_OVER_BASE = (0.90, 1.20)
# Policy years the proxy can be trusted on for a delta. The proxy validation puts its delta error
# against nested simulation at 36% of the nested delta in policy year 1 and 21% in year 2, because
# the fitting paths have barely dispersed by then and the whole design piles up in a narrow band;
# it is 10% by year 5 and 7% by year 9, which is the best the fit gets. A step-up effect read off
# year 1 would be reading the fit's own error.
PROXY_RELIABLE_YEARS = (5, 9)
PROXY_PATHS = 20_000
PROXY_SEED = 20251231


def greeks_table(setup) -> pd.DataFrame:
    book, state, valuer = setup["book"], setup["state"], setup["valuer"]
    attribution = base_valuation(setup)[0]
    rows = []
    for label, table, note in (("economic, Basic table", "basic", "hedging basis"),
                               ("reporting, Period table", "period", "margins included")):
        priced = greeks_module.compute(
            Valuer(mortality.load(table), n_paths=valuer.n_paths, seed=valuer.seed,
                   male_weight=valuer.male_weight),
            book, state, attribution,
            equity_bump=EQUITY_BUMP, rate_bump_bp=RATE_BUMP_BP, vol_bump=VOL_BUMP,
        )
        rows.append({
            "basis": label,
            "value": priced.value,
            "equity_exposure": priced.equity_exposure,
            "equity_exposure_pct_av": 100 * priced.equity_exposure_pct_of_account,
            "equity_exposure_std_error": priced.equity_std_error,
            "equity_gamma": priced.equity_gamma,
            "rho_per_bp": priced.rho_per_bp,
            "rho_per_100bp_pct_av": 100 * priced.rho_per_bp * 100 / priced.account_value,
            "rho_std_error": priced.rho_std_error,
            "vega_current": priced.vega_current,
            "vega_long_run": priced.vega_long_run,
            "vega_std_error": priced.vega_std_error,
            "level_std_error": priced.level_std_error,
            "note": note,
        })
    return pd.DataFrame(rows)


def positions(setup) -> pd.DataFrame:
    """The Greeks expressed as the trades that carry them."""
    book, state, valuer = setup["book"], setup["state"], setup["valuer"]
    priced = greeks_module.compute(
        valuer, book, state, base_valuation(setup)[0],
        equity_bump=EQUITY_BUMP, rate_bump_bp=RATE_BUMP_BP, vol_bump=VOL_BUMP,
    )
    swap = inst.InterestRateSwap(tenor=SWAP_TENOR_YEARS, receive_fixed=True)
    market = inst.HedgeMarket(index=1.0, curve=state.curve,
                              volatility=float(np.sqrt(state.heston.v0)),
                              smile=inst.smile_from_heston(state.heston, state.curve))
    # The swap's own rate sensitivity per dollar of notional, taken from the curve's annuity
    # rather than from an assumed duration, so the notional below is the one that actually
    # carries the liability's DV01 on this curve.
    per_dollar = swap.exposures(market).rho
    rows = [
        ("short index futures, notional", priced.equity_exposure),
        ("as a multiple of account value", priced.equity_exposure / priced.account_value),
        (f"receive-fixed {SWAP_TENOR_YEARS:.0f}-year swap, target per basis point",
         -priced.rho_per_bp),
        ("the notional that carries it", priced.rho_per_bp / per_dollar
         if per_dollar else np.nan),
        ("long volatility, per point of current variance", priced.vega_current),
        ("volatility exposure beyond any listed maturity, per point", priced.vega_long_run),
    ]
    return pd.DataFrame(rows, columns=["position", "size"])


def moneyness_profile(setup) -> pd.DataFrame:
    """Value and shock responses across the benefit-base-to-account ratio.

    The attribution percentage comes out the same at every ratio, and that is the accounting
    rather than an oversight: it is fixed when the contract is written, and a contract sitting at
    a benefit base 40% above its account value today was written with the two equal. So this grid
    is one contract at seven points of its life, not seven contracts - which is exactly what makes
    it the right curve to locate a disclosed sensitivity on.
    """
    valuer, state, terms = setup["valuer"], setup["state"], setup["terms"]
    from vahedge.liability import cohorts
    rows = []
    for ratio in MONEYNESS_GRID:
        book = cohorts.single_contract(
            terms, issue_age=ISSUE_AGE, base_contract_charge=0.0131, fund_expense=0.0095,
            premium=PREMIUM, account_value=PREMIUM, benefit_base=PREMIUM * ratio,
            deferral_years=DEFERRAL_YEARS, max_age=MAX_AGE,
        )
        attribution = valuer.calibrate_attribution(book, state)
        shocks = greeks_module.disclosed_shocks(valuer, book, state, attribution)
        wide = shocks.set_index("shock")["change_pct_of_account"] * 100
        row = {"gwb_over_av": ratio,
               "value_pct_av": 100 * float(shocks.loc[0, "value"]) / PREMIUM,
               "attribution": float(np.ravel(attribution)[0]),
               "equity_down_over_up": greeks_module.asymmetry(shocks, "equity", 10),
               "rates_down_over_up": greeks_module.asymmetry(shocks, "rates", 100)}
        # The _pct_av suffix matches what the validation scripts write for the same quantity.
        # Two tables of the same number under two column names is how a figure ends up reading
        # the wrong one.
        row.update({f"{name}_pct_av": float(value) for name, value in wide.items()
                    if name != "base"})
        rows.append(row)
    return pd.DataFrame(rows)


def step_up_proximity(setup) -> pd.DataFrame:
    """Equity exposure against time to the next anniversary, from the proxy the hedge uses.

    A book with anniversaries spread across the calendar averages this away. One policy does not,
    which is one reason a single-policy hedge turnover statistic is an overstatement.

    Two moneyness levels, chosen against the fit's own design range rather than for roundness.
    At 0.90 of the benefit base both fit families contain the state and the numbers are
    interpolation. At 1.20 only the pre-event family does, and that is not a gap in the fit but a
    fact about the contract: the step-up resets the base to the contract value every anniversary,
    so no post-event state ever has an account above its base, while a contract halfway through a
    good year does. The flag is reported rather than suppressed, and the 1.20 rows should be read
    as the pre-event fit's answer with the post-event end of the bracket extrapolated.
    """
    book, state = setup["book"], setup["state"]
    n_years = int(book.projection_years.max())
    survival, deaths = setup["valuer"].mortality_for(book, state.valuation_year, n_years)
    market_paths = simulate(state.heston, state.hull_white(), state.correlations, state.mix,
                            n_years=n_years, n_paths=PROXY_PATHS, seed=PROXY_SEED)
    fit = lsmc.fit(gmwb.project(book, market_paths, survival, deaths,
                                equity_weight=state.mix.equity_weight, record=True))

    attribution = float(np.ravel(base_valuation(setup)[0])[0])
    rows = []
    for account_over_base in ACCOUNT_OVER_BASE:
        for policy_year in PROXY_RELIABLE_YEARS:
            for tau in TIME_TO_ANNIVERSARY:
                # years_since_issue of (year - tau) sits tau of a year before the anniversary
                # that closes policy year `year`.
                at = float(policy_year - tau)
                priced = fit.greeks_at(
                    years_since_issue=at, account_value=PREMIUM * account_over_base,
                    benefit_base=PREMIUM, variance=state.heston.v0,
                    zero_10y=float(state.curve.zero(10.0)), attribution=attribution,
                )
                rows.append({
                    "policy_year": policy_year,
                    "years_to_anniversary": tau,
                    "account_over_base": account_over_base,
                    "value": float(np.ravel(priced["value"])[0]),
                    "equity_exposure": float(np.ravel(priced["delta"])[0]),
                    "equity_gamma": float(np.ravel(priced["gamma"])[0]),
                    "vega": float(np.ravel(priced["vega"])[0]),
                    "rho_per_bp": float(np.ravel(priced["rho_per_bp"])[0]),
                    "theta_per_year": float(np.ravel(priced["theta"])[0]),
                    "outside_design": bool(np.ravel(priced["outside_design"])[0]),
                })
    return pd.DataFrame(rows)


def main() -> None:
    paths.ensure_output_dirs()
    # The shock grid revisits the base state and the four rate shifts at every moneyness,
    # so the cache has to be large enough to hold all five or it re-simulates each of them seven
    # times over.
    setup = build(cache_size=8)

    table = greeks_table(setup)
    table.to_csv(paths.TABLES / "greeks.csv", index=False, float_format="%.4f")
    print("Greeks at issue, both mortality bases")
    for _, row in table.iterrows():
        print(f"  {row['basis']}")
        print(f"    value {row['value']:>12,.0f}  (level standard error "
              f"{row['level_std_error']:,.0f})")
        print(f"    equity exposure {row['equity_exposure']:>12,.0f} "
              f"({row['equity_exposure_pct_av']:+.2f}% of account, standard error "
              f"{row['equity_exposure_std_error']:,.0f})")
        print(f"    rho per 100bp   {100*row['rho_per_bp']:>12,.0f} "
              f"({row['rho_per_100bp_pct_av']:+.2f}% of account, standard error "
              f"{100*row['rho_std_error']:,.0f})")
        print(f"    vega, current {row['vega_current']:>10,.0f} and long run "
              f"{row['vega_long_run']:,.0f} per point (standard error "
              f"{row['vega_std_error']:,.0f})")

    sizing = positions(setup)
    sizing.to_csv(paths.TABLES / "hedge_sizing.csv", index=False, float_format="%.4f")
    print("\nWhat that is, in trades")
    for _, row in sizing.iterrows():
        print(f"  {row['position']:<58s} {row['size']:>14,.2f}")

    profile = moneyness_profile(setup)
    profile.to_csv(paths.TABLES / "moneyness_profile.csv", index=False, float_format="%.4f")
    print("\nAcross the benefit-base-to-account ratio, % of account value")
    print("  GWB/AV   value   alpha   eq -10%  eq +10%  down/up   +100bp   -100bp  down/up")
    for _, row in profile.iterrows():
        print(f"  {row['gwb_over_av']:6.2f} {row['value_pct_av']:7.2f} "
              f"{row['attribution']:7.4f} {row['equity_down_10pct_pct_av']:9.2f} "
              f"{row['equity_up_10pct_pct_av']:8.2f} {row['equity_down_over_up']:8.2f} "
              f"{row['rates_up_100bp_pct_av']:8.2f} {row['rates_down_100bp_pct_av']:8.2f} "
              f"{row['rates_down_over_up']:8.2f}")

    proximity = step_up_proximity(setup)
    proximity.to_csv(paths.TABLES / "step_up_proximity.csv", index=False, float_format="%.4f")
    print("\nApproaching the step-up, from the proxy")
    print("  AV/GB  year  tau        value        delta        gamma  theta  outside design")
    for _, row in proximity.iterrows():
        print(f"  {row['account_over_base']:5.2f} {int(row['policy_year']):5d} "
              f"{row['years_to_anniversary']:4.1f} {row['value']:12,.0f} "
              f"{row['equity_exposure']:12,.0f} {row['equity_gamma']:12,.0f} "
              f"{row['theta_per_year']:6,.0f}  {str(bool(row['outside_design'])):>14s}")
    for (level, year), block in proximity.groupby(["account_over_base", "policy_year"],
                                                  sort=False):
        far = float(block.loc[block["years_to_anniversary"].idxmax(), "equity_exposure"])
        near = float(block.loc[block["years_to_anniversary"].idxmin(), "equity_exposure"])
        turned = " and changes sign" if far * near < 0 else ""
        print(f"    AV/GB {level:.2f}, policy year {int(year)}: the exposure runs from "
              f"{far:+,.0f} with most of the year to run to {near:+,.0f} on its eve, a "
              f"{abs(near / far - 1):.0%} move with no market move behind it{turned}")
    print(f"\nwrote four tables to {paths.TABLES}")


if __name__ == "__main__":
    main()
