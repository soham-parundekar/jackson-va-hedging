"""Statutory requirement for the guarantee block, and what the surrender value floor costs.

The valuation workstream answers what the guarantees are worth. This answers how much has to be
held, which on this book is a number of the opposite sign: the market risk benefit is a net asset
at issue on the base-case grid, while a tail expectation of the accumulated deficiency is firmly
positive. Both are correct and they are not reconcilable, because one is a price and the other is
a percentile.

Three things come out.

The requirement itself, at CTE(70) and CTE(90), on real-world paths. Reported against a sweep of
the equity risk premium rather than at one value, because the premium is the assumption the
answer is most exposed to and there is no free data that pins it down. Two levels rather than
one, because the gap between them says whether a hedge flattens the mean of the tail or its
shape.

Whether the cash surrender value floor binds at all, across the scenarios. The first version of
this script reported the floor's cost as a level, by comparing the accumulated deficiency against
the surrender value, and got 115 per cent of premium out of two quantities that do not belong on
one axis - one measured net of the assets already held, the other gross. Jackson's 8-K names the
cost correctly: "non-economic hedging costs". The floor is a minimum on the total policy reserve
and the separate account already holds the account value, so the floor's bite is in the reserve's
sensitivity rather than its level, and measuring it needs the hedge ledger rather than a scenario
reserve. What this reports is the input to that: how often, and in which states, 98 per cent of
the account value exceeds the account value plus the guarantee reserve.

Where in the projection the tail scenarios run out of money. A requirement is one number and it
hides its horizon, and a block whose tail peaks at year five is a different problem from one
whose tail peaks at year twenty-five.

Usage:  python -m scripts.run_statutory
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vahedge import paths
from vahedge.capital import statutory
from vahedge.liability import cohorts, gmwb, mortality
from vahedge.liability import terms as terms_module
from vahedge.market import state as market_state
from vahedge.market.simulate import simulate
from vahedge.valuation.engine import MarketState

ISSUE_AGE = 70
DEFERRAL_YEARS = 5
MAX_AGE = 105
PREMIUM = 100.0

N_PATHS = 20_000
SEED = 20260101
# Arithmetic equity risk premium over the short rate, applied to the equity sleeve only, as a
# stated sweep rather than an estimate. Nothing available here identifies it: the free equity
# history reaches back ten years, which is far too short a window to say anything about a
# long-horizon premium, and no free source gives one for a specific date. So the requirement is
# reported at three levels spanning the range the argument is usually had over, and the spread
# between them is part of the result rather than something to average away. Reading the middle
# figure as an estimate would be reading more into it than the data supports.
PREMIUM_SWEEP = (0.02, 0.04, 0.06)
BASE_PREMIUM = 0.04
LEVELS = (0.70, 0.90)


def build(equity_risk_premium: float):
    calibration = market_state.load()
    state = MarketState.from_calibration(calibration)
    core = terms_module.load()[("flex_gmwb", "single", "core")]
    book = cohorts.single_contract(
        core, issue_age=ISSUE_AGE, base_contract_charge=0.0131, fund_expense=0.0095,
        premium=PREMIUM, deferral_years=DEFERRAL_YEARS, max_age=MAX_AGE,
        lapse_rate=0.04, lapse_beta=1.2, lapse_floor=0.01,
    )
    horizon = int(book.projection_years.max())
    survival, deaths = mortality.load("basic").rates(
        book.attained_age, state.valuation_year, horizon, 0.5
    )
    market_paths = simulate(
        state.heston, state.hull_white(), state.correlations, state.mix,
        n_years=horizon, n_paths=N_PATHS, seed=SEED,
        equity_risk_premium=equity_risk_premium,
    )
    projection = gmwb.project(
        book, market_paths, survival, deaths,
        equity_weight=state.mix.equity_weight, record_deficiency=True,
    )
    return book, state, market_paths, projection


def sweep() -> tuple:
    """The requirement against the equity risk premium, plus the floor's reach and its
    sensitivity to the surrender share, both at the base case."""
    rows, floor, profile, shares = [], None, None, None
    for premium in PREMIUM_SWEEP:
        _, _, market_paths, projection = build(premium)
        table = statutory.requirement(projection.deficiency_pv, levels=LEVELS)
        table["equity_risk_premium"] = premium
        rows.append(table)
        if premium == BASE_PREMIUM:
            floor = floor_reach(projection)
            profile = statutory.deficiency_profile(projection.deficiency_pv, level=0.90)
            shares = floor_share_sensitivity(projection)
    return pd.concat(rows, ignore_index=True), floor, profile, shares


def floor_reach(projection) -> pd.DataFrame:
    """Where the surrender value minimum binds, by policy year.

    The guarantee reserve at a year end is the remaining accumulated deficiency on that path,
    re-expressed in money of that date: the present value of everything still to come, which is
    what a reserve is. The floor binds where the guarantee is enough of an asset that 98 per cent
    of the account value exceeds the account plus that reserve.

    Re-expressing is the step this used to skip, and ``statutory.reserve_at_each_year`` is where
    the arithmetic and the reason for it now live, beside the floor comparison that needs it.
    """
    account = projection.in_force_account
    remaining = statutory.reserve_at_each_year(
        projection.deficiency_pv, projection.deficiency_discount
    )
    rows = []
    for year in range(remaining.shape[1]):
        live = account[:, year] > 1e-9
        if not live.any():
            continue
        out = statutory.floored_reserve(remaining[live, year], account[live, year])
        rows.append({
            "policy_year": year + 1,
            "paths_in_force": int(live.sum()),
            "share_floor_binds": float(out["floor_binds"].mean()),
            # Ratio of the sums, not the mean of the ratios. A path whose account has run down
            # to nothing still carries a guarantee reserve, so its own ratio goes to infinity and
            # the average of the ratios reported 290 per cent at year eight. The block-level
            # ratio is the quantity the floor actually compares.
            "guarantee_reserve_pct_of_account": float(
                remaining[live, year].sum() / account[live, year].sum()
            ),
            "median_guarantee_reserve_pct_of_account": float(
                np.median(remaining[live, year] / np.maximum(account[live, year], 1e-9))
            ),
        })
    return pd.DataFrame(rows)


# Every share Jackson has disclosed over the four year-ends in data/raw/jackson_book_statistics.csv,
# plus the 98.0% the model holds. The level moved 0.73 points across those dates, which sounds
# immaterial and is not: what the floor's bite depends on is one minus the share, and that ran
# 1.99% to 2.71% - a range of more than a third of itself on the quantity that drives the result.
DISCLOSED_SHARES = (0.9729, 0.9773, 0.9795, 0.9801)


def floor_share_sensitivity(projection) -> pd.DataFrame:
    """How much of the floor's reach is the surrender share rather than the block.

    The model holds the share at the latest disclosed year end and replays a decade with it. The
    ratio is not a constant of the business: it rose through the four disclosed dates as the book
    aged out of its surrender charge period. Reporting the reach at each disclosed level is what
    turns a stated constant into a bounded one.
    """
    account = projection.in_force_account
    remaining = statutory.reserve_at_each_year(
        projection.deficiency_pv, projection.deficiency_discount
    )
    rows = []
    for share in sorted({*DISCLOSED_SHARES, statutory.SURRENDER_VALUE_SHARE}):
        binds, weighted, years = [], [], 0
        for year in range(remaining.shape[1]):
            live = account[:, year] > 1e-9
            if not live.any():
                continue
            out = statutory.floored_reserve(remaining[live, year], account[live, year],
                                            surrender_share=share)
            binds.append(float(out["floor_binds"].mean()))
            # What the floor adds over the unfloored reserve, as a share of the account it is
            # measured against. This is the quantity the basis gap in the hedge ledger is built
            # from, so its sensitivity to the share is the sensitivity of that gap.
            added = out["reserve"] - (account[live, year] + remaining[live, year])
            weighted.append(float(added.sum() / account[live, year].sum()))
            years += 1
        rows.append({
            "surrender_share": share,
            "policy_years": years,
            "mean_share_floor_binds": float(np.mean(binds)),
            "first_year_share_floor_binds": binds[0],
            "last_year_share_floor_binds": binds[-1],
            "mean_floor_addition_pct_of_account": float(np.mean(weighted)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    paths.ensure_output_dirs()
    requirement, floor, profile, shares = sweep()
    requirement.to_csv(paths.TABLES / "statutory_requirement.csv", index=False)
    floor.to_csv(paths.TABLES / "statutory_floor_reach.csv", index=False)
    profile.to_csv(paths.TABLES / "statutory_deficiency_profile.csv", index=False)
    shares.to_csv(paths.TABLES / "statutory_floor_share_sensitivity.csv", index=False)

    print(f"Requirement as a share of premium, {N_PATHS:,} real-world paths")
    for _, row in requirement.iterrows():
        print(f"  premium {row['equity_risk_premium']:.0%}  CTE({row['cte_level']:.0%})  "
              f"{row['requirement']:7.2f}   worst scenario {row['worst_scenario']:7.2f}   "
              f"median {row['median_scenario']:7.2f}   "
              f"scenarios in deficit {row['share_positive']:5.1%}   "
              f"tail size {row['scenarios_in_tail']:.0f}")

    print("\nWhere the surrender value minimum binds, at a 4% premium")
    binding = floor[floor["share_floor_binds"] > 0.0]
    if binding.empty:
        print("  nowhere: the guarantee reserve is never a large enough asset for 98% of the "
              "account value to exceed the account plus the reserve")
    else:
        for _, row in binding.head(12).iterrows():
            print(f"  policy year {row['policy_year']:3.0f}  binds on "
                  f"{row['share_floor_binds']:5.1%} of in-force paths   guarantee reserve "
                  f"{row['guarantee_reserve_pct_of_account']:+6.1%} of account "
                  f"(median path {row['median_guarantee_reserve_pct_of_account']:+6.1%})")
        print(f"  binds somewhere in {len(binding)} of {len(floor)} policy years; "
              f"worst year {binding.loc[binding['share_floor_binds'].idxmax(), 'policy_year']:.0f} "
              f"at {binding['share_floor_binds'].max():.1%}")

    print("\nThe floor's reach against the surrender share, over every level Jackson has "
          "disclosed")
    for _, row in shares.iterrows():
        flag = "  <- held by the model" if row["surrender_share"] == (
            statutory.SURRENDER_VALUE_SHARE) else ""
        print(f"  share {row['surrender_share']:.2%}  binds on "
              f"{row['mean_share_floor_binds']:5.1%} of in-force paths on average "
              f"({row['first_year_share_floor_binds']:5.1%} at year one, "
              f"{row['last_year_share_floor_binds']:5.1%} at the end)   the floor adds "
              f"{row['mean_floor_addition_pct_of_account']:5.2%} of account{flag}")

    peak = profile.loc[profile["share_of_tail_peaking_here"].idxmax()]
    print(f"\nWhere the 10% tail runs out of money: most scenarios peak at policy year "
          f"{peak['policy_year']:.0f} ({peak['share_of_tail_peaking_here']:.1%} of the tail), "
          f"worst mean accumulated deficiency {profile['mean_accumulated_tail'].max():.2f} "
          f"of premium against {profile['mean_accumulated_all'].max():.2f} across all scenarios")
    print(f"\nwrote four tables under {paths.TABLES}")


if __name__ == "__main__":
    main()
