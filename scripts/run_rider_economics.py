"""E6. What the rider actually earned, after hedging it, in basis points of benefit base a year.

The valuation says what the guarantee is worth and the backtest says how tightly it can be
hedged. Neither says whether the product made money. This does, on the one basis that lets a
hundred-dollar model policy stand for a book: the rider charge is levied on the benefit base, so
everything is reported against the benefit base and annualised.

The decomposition is an identity rather than an attribution, which matters because an
attribution with a residual in it can hide the answer in the residual. Over any window the
hedged book's profit is

    net  =  fee income  +  interest on cash  -  hedge cost  -  uncovered cost

where the uncovered cost is what the guarantee cost that the hedge did not recover:

    uncovered cost  =  claims paid  +  change in the guarantee's value  -  the hedge's result

with the hedge's result taken as its closing mark plus whatever cash the rebalancing threw off,
so nothing is left over. On the unhedged row there is no hedge to recover anything and the term
is the whole cost of the guarantee, which is what makes the two rows comparable. The four terms
add to the ledger's own profit to the last cent and a test holds them to it.

Interest is a term in its own right because it does not behave the way a fee-income statement
would suggest. A hedge that is losing money funds those losses with borrowed cash, so on every
hedged arm the interest line is negative and large enough to matter: a fifth of the rider's fee
income over the replayed decade.

*By cohort*, because the balance between the terms changes completely across a contract's life.
A rider still deferring collects its charge on a benefit base that is growing and pays no claims
at all; one twelve years in is paying withdrawals out of an account that may no longer cover
them, on a benefit base that stopped growing. The fee is the same percentage throughout.

*By scenario*, because the margin on the decade that happened is one draw. The crisis windows
say what the product earns when it is tested, and they are reported at an annual rate like
everything else, which on a window of a few weeks is a rate and not an outcome.

What was meant to close the loop is the break-even fee, and the loop does not close the way the
research design expected it to. Off the curve this backtest starts from there is no fee that
prices the base contract at all: the guarantee's cost falls as the charge rises, bottoms out
near a five per cent charge and turns back up, never reaching zero. So the comparison this makes
is against the cheapest the guarantee can be made rather than against a fair fee, and the fee
curve is published so the shape can be read rather than taken on trust.

Usage:  python -m scripts.run_rider_economics
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vahedge import paths
from vahedge.hedge import attribution, strategies
from vahedge.liability import cohorts, mortality
from vahedge.liability import terms as terms_module
from vahedge.market import scenarios
from vahedge.valuation import breakeven
from vahedge.valuation.engine import Valuer

from scripts.run_hedge_experiments import (
    DEFERRAL_YEARS,
    ISSUE_AGE,
    MAX_AGE,
    PREMIUM,
    _run,
    build,
)

# Where a contract can be when the replay starts: years since issue, and the account value as a
# share of the benefit base. The pairs are not independent - a contract twelve years in has been
# drawing for seven of them, so its account is far below its base - and they are set together
# rather than swept as a grid for that reason. The benefit base is held at the premium in every
# case, which is the normalisation rather than an assumption about roll-up: every figure here is
# per unit of benefit base.
COHORTS = (
    ("at issue", 0, 1.00),
    ("deferring", 3, 0.94),
    ("first income", 6, 0.85),
    ("drawing", 9, 0.70),
    ("late", 12, 0.52),
)
# The unhedged book, the delta-and-rho hedge, and the hedge that adds puts. The three that differ
# in kind; the variants on S3 differ in sizing and would tell the same story five times.
COMPARED = ("S0", "S2", "S3")
HEADLINE = "S2"
WHOLE_REPLAY = "the whole replay"
# The fee grid for the curve behind the solve. Dense where riders are actually priced, sparse
# out at the charges that exist only to locate the turning point.
FEE_GRID = (0.0005, 0.002, 0.005, 0.0125, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08)
# The pricing map. One issue age either side of the base case, and income starting at the
# contract's own landmarks: immediately, at the midpoint of the bonus period, and at its end.
ISSUE_AGES = (65, 70, 75)
INCOME_START_AGES = (70, 75, 80)
TERMS_KEY = ("flex_gmwb", "single", "core")


def policy_at(duration: int, moneyness: float, valuation_year: int):
    """One cohort's contract and its own mortality, on the terms the rest of the project uses.

    Every cohort draws at the percentage a contract issued at 70 and deferring five years locks
    in, whatever age it has since reached, because that is what these five rows are: one contract
    seen at five points in its life rather than five contracts. Leaving the percentage to follow
    the attained age would hand the twelve-year-old cohort the 81-and-over band and make its
    guarantee 4% dearer than the same contract's at any earlier point.
    """
    core = terms_module.load()[TERMS_KEY]
    book = cohorts.single_contract(
        core, issue_age=ISSUE_AGE, base_contract_charge=0.0131, fund_expense=0.0095,
        premium=PREMIUM, account_value=PREMIUM * moneyness, benefit_base=PREMIUM,
        deferral_years=max(DEFERRAL_YEARS - duration, 0),
        years_since_issue=duration, max_age=MAX_AGE,
        lapse_rate=0.04, lapse_beta=1.2, lapse_floor=0.01,
        first_withdrawal_age=ISSUE_AGE + DEFERRAL_YEARS,
    )
    survival, deaths = mortality.load("basic").rates(
        book.attained_age, valuation_year, int(book.projection_years.max()), 0.5,
    )
    return book, survival, deaths


def economics(run, years: float) -> dict:
    """The decomposition, with the window it was measured over and the base it is quoted on."""
    return {
        "days": int(run.ledger.shape[0]),
        "years": years,
        **attribution.economics(run.ledger),
        "mean_benefit_base": float(run.ledger["benefit_base"].mean()),
    }


REPORTED = ("fee_income", "cash_interest", "hedge_cost", "claims_paid", "liability_change",
            "hedge_result", "uncovered_cost", "net")


def per_year_bp(row: dict) -> dict:
    """The same terms as an annual rate on the benefit base, which is what the fee is quoted in."""
    scale = 10000.0 / (row["mean_benefit_base"] * row["years"])
    return {f"{name}_bp": row[name] * scale for name in REPORTED}


def across_cohorts(setup, window, label: str, books: dict) -> pd.DataFrame:
    """Every cohort against every strategy, on one window."""
    matrix = strategies.matrix()
    years = float(window.year_fraction[-1] - window.year_fraction[0])
    rows = []
    for name, duration, moneyness in COHORTS:
        book, survival, deaths = books[name]
        cohort_setup = dict(setup, policy=book, survival=survival, deaths=deaths)
        for key in COMPARED:
            run = _run(cohort_setup, window, matrix[key], years_at_start=float(duration))
            figures = economics(run, years)
            rows.append({
                "scenario": label,
                "cohort": name,
                "years_since_issue": duration,
                "opening_moneyness": moneyness,
                "strategy": key,
                **figures,
                **per_year_bp(figures),
            })
    return pd.DataFrame(rows)


def fee_curve(setup, valuer) -> tuple:
    """The guarantee's value across the fee grid, and the solve's verdict on the base contract.

    Solved at issue rather than per cohort, because a break-even fee on a contract already
    part-way through its life is a different and weaker question: the fees already collected are
    sunk, so the answer would say the rider is nearly free. What this gives is the price the
    product needed at the point it was sold.

    Off the curve at the start of the backtest, not at the valuation date, for the same reason
    the proxy is refitted there: a programme that ran from 2016 priced its contracts on 2016's
    rates. That is also why the solve fails here and succeeds in the at-issue valuation - the
    difference is three hundred basis points of ten-year rate, not a modelling choice.
    """
    curve = breakeven.fee_sensitivity(valuer, setup["at_issue"], setup["state"], FEE_GRID)
    return curve, breakeven.solve(valuer, setup["at_issue"], setup["state"])


def pricing_map(setup, valuer) -> pd.DataFrame:
    """The fair fee across issue age and income start, to say whether the failure is general.

    If the base contract cannot be priced, the next question is whether that is this cell or the
    product: a guarantee sold five years older pays for fewer years, and one that defers to the
    end of the bonus period pays more per year but for fewer of them still.
    """
    core = terms_module.load()[TERMS_KEY]

    def builder(issue_age: int, income_start_age: int):
        return cohorts.single_contract(
            core, issue_age=issue_age, base_contract_charge=0.0131, fund_expense=0.0095,
            premium=PREMIUM, deferral_years=income_start_age - issue_age, max_age=MAX_AGE,
            lapse_rate=0.04, lapse_beta=1.2, lapse_floor=0.01,
        )

    return breakeven.margin_map(
        valuer, builder, setup["state"],
        issue_ages=ISSUE_AGES, income_start_ages=INCOME_START_AGES,
    )


def main() -> None:
    print("Fitting the proxy and the market history", flush=True)
    setup = build()
    history = setup["history"]

    windows = [(WHOLE_REPLAY, history)]
    episodes = scenarios.available_episodes(history)
    for name, (start, end, _) in episodes["covered"].items():
        windows.append((name, history.window(start, end, label=name)))

    # Built once and replayed through every window, so the five cohorts are literally the same
    # five contracts in the decade and in each crisis rather than five rebuilt each time.
    books = {name: policy_at(duration, moneyness, setup["state"].valuation_year)
             for name, duration, moneyness in COHORTS}

    frames = []
    for label, window in windows:
        print(f"  {label}: {len(COHORTS)} cohorts x {len(COMPARED)} strategies", flush=True)
        frames.append(across_cohorts(setup, window, label, books))
    table = pd.concat(frames, ignore_index=True)
    table.to_csv(paths.TABLES / "rider_economics.csv", index=False, float_format="%.6f")

    # The identity, checked rather than assumed. If the four terms do not add to the ledger's own
    # profit the decomposition is telling a story the ledger does not support.
    drift = (table["net"] - table["ledger_pnl"]).abs().max()
    scale = table["mean_benefit_base"].max()
    if drift > 1e-8 * scale:
        raise ValueError(
            f"the decomposition misses the ledger by {drift:.3e} on a base of {scale:.0f}"
        )
    leak = table["cash_recursion_error"].abs().max()
    if leak > 1e-8 * scale:
        raise ValueError(f"cash moved by {leak:.3e} that no ledger column accounts for")
    print(f"  the four terms add to the ledger's own profit, worst gap {drift:.2e} "
          f"on a benefit base of {scale:,.0f}")

    whole = table[table["scenario"] == WHOLE_REPLAY]
    print("\nOver the decade, basis points of benefit base a year")
    print("  cohort         strategy    fee  interest  hedge cost  uncovered      net")
    for _, row in whole.sort_values(["years_since_issue", "strategy"]).iterrows():
        print(f"  {row['cohort']:<14} {row['strategy']:<8} {row['fee_income_bp']:6.0f} "
              f"{row['cash_interest_bp']:9.0f} {row['hedge_cost_bp']:11.1f} "
              f"{row['uncovered_cost_bp']:10.0f} {row['net_bp']:8.0f}")

    print("\nPricing the guarantee at the start of the backtest", flush=True)
    # One set of paths at a time: a forty-year projection over twenty thousand paths is a large
    # array and the map walks three different horizons.
    valuer = Valuer(mortality.load("basic"), cache_size=1)
    curve, solved = fee_curve(setup, valuer)
    curve.to_csv(paths.TABLES / "rider_fee_curve.csv", index=False, float_format="%.6f")
    as_pct = 100.0 * curve["value"] / PREMIUM
    cheapest = int(curve["value"].idxmin())
    print(f"  charged {10000.0 * solved.charged:.0f}bp of benefit base; the solve returns "
          f"{solved.reason}")
    print(f"  the guarantee costs {as_pct.iloc[0]:.1f}% of premium at a "
          f"{10000.0 * curve['fee_pct'].iloc[0]:.0f}bp charge, least at "
          f"{10000.0 * curve['fee_pct'].iloc[cheapest]:.0f}bp where it still costs "
          f"{as_pct.iloc[cheapest]:.1f}%, and {as_pct.iloc[-1]:.1f}% at "
          f"{10000.0 * curve['fee_pct'].iloc[-1]:.0f}bp")

    pricing = pricing_map(setup, valuer)
    pricing.to_csv(paths.TABLES / "rider_break_even.csv", index=False, float_format="%.6f")
    # The map's base cell is the contract the solve above priced, so the two have to agree. They
    # are built by different code paths and this is the only place that notices if they drift.
    base = pricing[(pricing["issue_age"] == ISSUE_AGE)
                   & (pricing["income_start_age"] == ISSUE_AGE + DEFERRAL_YEARS)]
    if base.empty or base["reason"].iloc[0] != solved.reason:
        raise ValueError("the pricing map's base cell disagrees with the solve on the same "
                         f"contract: {base['reason'].tolist()} against {solved.reason}")
    priced = int(pricing["fair_pct"].notna().sum())
    print(f"  across {len(pricing)} issue-age and income-start cells, {priced} admit a fee that "
          f"prices the guarantee")
    for _, row in pricing.iterrows():
        fair = "-" if np.isnan(row["fair_pct"]) else f"{10000.0 * row['fair_pct']:.0f}bp"
        print(f"    issued at {int(row['issue_age'])}, income at "
              f"{int(row['income_start_age'])}: {fair:>8}   {row['reason']}")

    print(f"\nWhat the crises cost on {HEADLINE}, at an annual rate against the decade's own")
    pivot = (table[table["strategy"] == HEADLINE]
             .pivot_table(index=["years_since_issue", "cohort"], columns="scenario",
                          values="net_bp"))
    for (_, cohort), line in pivot.iterrows():
        crises = line.drop(labels=[WHOLE_REPLAY], errors="ignore").dropna()
        if crises.empty:
            continue
        print(f"  {cohort:<14} decade {line.get(WHOLE_REPLAY, np.nan):8.0f}   "
              f"worst window {crises.idxmin()} {crises.min():8.0f}")

    print(f"\nwrote three tables to {paths.TABLES}")


if __name__ == "__main__":
    main()
