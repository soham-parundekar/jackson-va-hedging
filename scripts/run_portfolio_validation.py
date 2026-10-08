"""Five policy vintages added up, against the disclosed book.

One policy cannot match Jackson's sensitivities on scale, and the single-policy comparison shows
why: matching the book's moneyness and matching its weighted-average attained age of 70 pull in
opposite directions. A contract issued in 2016 has ridden the market up and so sits out of the
money, which is right, but it is also nine years older than the book's average, which shortens
the guarantee and takes duration out of it.

Five vintages fix that without turning this into a seriatim valuation. Issue dates run from 2016
to 2024 and the issue ages are chosen so the attained ages in 2025 straddle 70. Because the rate
sheet bands the guaranteed withdrawal percentage by the age at the first withdrawal, the vintages
carry different withdrawal rates, which is the point rather than a nuisance: a real book draws at
a blended rate below the rate a new 70-year-old is quoted.

Two vintages are still in their deferral period at the last disclosed date, and that is
deliberate. The earlier single-policy comparison left deferral-phase contracts out because the
bonus accruing on their benefit base was not modelled, and said so; it is modelled here, so they
are in. Their presence pulls the portfolio's moneyness up and its duration down, both in the
direction the disclosed book sits.

Everything is valued as one book on one simulation rather than vintage by vintage: the cohorts
share their paths, each carries the attribution percentage its own issue date calibrates to, and
the aggregate is what the disclosure reports. Valuing them separately and adding would be the same
arithmetic with five times the Monte Carlo noise and five times the runtime.

Usage:  python -m scripts.run_portfolio_validation
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vahedge import paths
from vahedge.hedge import simulator
from vahedge.liability import cohorts, mortality
from vahedge.market import scenarios

from scripts.run_shock_validation import (
    DISCLOSED_DATES,
    DIVIDEND_YIELD,
    load_disclosed,
    market_at,
    shock_row,
)
from scripts.run_valuation import (
    BASE_CONTRACT_CHARGE,
    DEFERRAL_YEARS,
    FUND_EXPENSE,
    MAX_AGE,
    PREMIUM,
    build,
)

# Issue date, issue age, and the share of premium written in that vintage. Dates are mid-year or
# the start of the index history; ages are set so the attained ages in 2025 sit either side of the
# disclosed weighted average of 70. Equal premium shares, because nothing in the disclosure gives
# the book's issue-year distribution and inventing one would be the least defensible part of this.
VINTAGES = (
    ("2016-09-26", 60, 0.20),
    ("2018-06-29", 63, 0.20),
    ("2020-06-30", 66, 0.20),
    ("2022-06-30", 69, 0.20),
    ("2024-06-28", 72, 0.20),
)
SHOCKS = ("equity_up_10pct", "equity_down_10pct", "rates_up_50bp", "rates_down_50bp",
          "rates_up_100bp", "rates_down_100bp")


def vintage_contract(terms, issue_age: int, years_since_issue: int, account_value: float,
                     benefit_base: float, utilisation: float = 1.0, lapse_rate: float = 0.0,
                     lapse_beta: float = 0.0, lapse_floor: float = 0.0):
    """One vintage as a cohort, at a whole number of policy years since its own issue.

    The behaviour arguments default to the static benchmark - draw the full guaranteed amount
    every year, never surrender - which is the most expensive case for the insurer and therefore
    the right base case for a validation that expects to sit above the disclosure. The behaviour
    reconciliation moves them.
    """
    return cohorts.single_contract(
        terms, issue_age=issue_age, base_contract_charge=BASE_CONTRACT_CHARGE,
        fund_expense=FUND_EXPENSE, premium=PREMIUM, account_value=account_value,
        benefit_base=benefit_base,
        deferral_years=max(DEFERRAL_YEARS - years_since_issue, 0),
        years_since_issue=years_since_issue, max_age=MAX_AGE,
        utilisation=utilisation, lapse_rate=lapse_rate, lapse_beta=lapse_beta,
        lapse_floor=lapse_floor,
    )


def roll_vintage(history, terms, issue_date: str, issue_age: int, target,
                 equity_weight: float) -> dict:
    """The vintage's account value and benefit base on the target date, from its own history."""
    at_issue = vintage_contract(terms, issue_age, 0, PREMIUM, PREMIUM)
    window = history.window(issue_date, target, label=f"{issue_date} to {target}")
    survival, deaths = mortality.load("basic").rates(
        at_issue.attained_age, pd.Timestamp(issue_date).year,
        int(at_issue.projection_years.max()), 0.5,
    )
    rolled = simulator.roll_contract(at_issue, window, survival, deaths,
                                     equity_weight=equity_weight)
    daily = simulator.daily_state(window, rolled, at_issue, PREMIUM, PREMIUM)
    return {"account_value": float(daily["account_value"][-1]),
            "benefit_base": float(daily["benefit_base"][-1]),
            "elapsed_years": float(window.year_fraction[-1]),
            "date": window.dates[-1]}


def vintage_attributions(valuer, panel, history, calibration, terms) -> dict:
    """Each vintage's attribution percentage, calibrated on the market of its own issue date.

    This is the piece that cannot be shortcut. The percentage is fixed when the contract is
    written, so a 2016 vintage carries the percentage a 1.6% ten-year rate implied and a 2024
    vintage carries the percentage a 4.3% rate implied. Using one percentage for the whole book
    would erase the single largest difference between the vintages.
    """
    out = {}
    for issue_date, issue_age, _ in VINTAGES:
        state = market_at(panel, history, calibration, issue_date)
        book = vintage_contract(terms, issue_age, 0, PREMIUM, PREMIUM)
        out[issue_date] = float(np.ravel(valuer.calibrate_attribution(book, state))[0])
    return out


def portfolio_at(setup, history, target, attributions, behaviour=None) -> tuple:
    """The book at one disclosed date, and the per-vintage detail behind it.

    ``behaviour`` overrides utilisation and lapse on the valued cohorts only. The roll that
    produces each vintage's account value and benefit base stays on the benchmark assumptions,
    because the account value a contract actually has today is a fact about the past rather than
    an assumption about the future, and letting the sweep rewrite history would confound the two.
    """
    terms = setup["terms"]
    behaviour = behaviour or {}
    equity_weight = float(setup["state"].mix.equity_weight)
    # Only the vintages that existed by the valuation date. A book grows by new business, so its
    # composition at the end of 2022 is not its composition at the end of 2025, and rolling a
    # 2024 vintage back to 2022 would be valuing a contract that had not been sold. Shares are
    # renormalised over the vintages in force, which keeps equal premium per vintage written.
    in_force = [(date, age, share) for date, age, share in VINTAGES
                if pd.Timestamp(date) <= pd.Timestamp(target)]
    if not in_force:
        raise ValueError(f"no vintage had been written by {target}")
    total_share = sum(share for _, _, share in in_force)
    books, weights, attribution, detail = [], [], [], []
    for issue_date, issue_age, premium_share in in_force:
        share = premium_share / total_share
        rolled = roll_vintage(history, terms, issue_date, issue_age, target,
                              equity_weight)
        duration = int(np.floor(rolled["elapsed_years"]))
        book = vintage_contract(terms, issue_age, duration, rolled["account_value"],
                                rolled["benefit_base"], **behaviour)
        books.append(book)
        weights.append(share)
        attribution.append(attributions[issue_date])
        detail.append({
            "as_of": str(pd.Timestamp(rolled["date"]).date()),
            "issue_date": issue_date,
            "issue_age": issue_age,
            "years_since_issue": duration,
            "attained_age": int(book.attained_age[0]),
            "deferring": bool(book.deferral_years[0] > 0),
            "gawa_pct": 100 * float(book.gawa_pct[0]),
            "account_value": rolled["account_value"],
            "benefit_base": rolled["benefit_base"],
            "gwb_over_av": rolled["benefit_base"] / rolled["account_value"],
            "attribution": attributions[issue_date],
            "premium_share": share,
        })
    return (cohorts.combine(books, weights), np.asarray(attribution, dtype=float),
            pd.DataFrame(detail))


def compare(setup, history, panel, disclosed: pd.DataFrame, attributions) -> tuple:
    valuer, calibration = setup["valuer"], setup["calibration"]
    rows, details = [], []
    for target in DISCLOSED_DATES:
        book, attribution, detail = portfolio_at(setup, history, target, attributions)
        state = market_at(panel, history, calibration, detail["as_of"].iloc[0])
        shocks = shock_row(valuer, book, state, attribution)
        account = float(book.total_account_value)
        rows.append({
            "as_of": detail["as_of"].iloc[0],
            "account_value": account,
            "benefit_base": float(book.total_benefit_base),
            "gwb_over_av": float(book.total_benefit_base / account),
            "weighted_attained_age": float(book.weighted_attained_age),
            "weighted_gawa_pct": float(100 * np.sum(book.weight * book.benefit_base
                                                    * book.gawa_pct)
                                       / np.sum(book.weight * book.benefit_base)),
            "deferring_share": float(np.sum(book.weight[book.deferral_years > 0])
                                     / np.sum(book.weight)),
            "net_amount_at_risk_pct_av": 100 * float(book.net_amount_at_risk) / account,
            "model_value_pct_av": shocks["value_pct_av"],
            **{f"model_{name}": shocks[f"{name}_pct_av"] for name in SHOCKS},
        })
        details.append(detail)

    model = pd.DataFrame(rows)
    wide = disclosed.pivot_table(index="as_of", columns="shock", values="impact_pct_of_av")
    wide = (100 * wide).add_prefix("disclosed_").reset_index()
    fair = (disclosed.drop_duplicates(subset=["as_of"])[["as_of", "fair_value_pct_of_av"]]
            .assign(disclosed_value_pct_av=lambda f: 100 * f["fair_value_pct_of_av"])
            .drop(columns="fair_value_pct_of_av"))
    for frame in (model, wide, fair):
        frame["year"] = pd.to_datetime(frame["as_of"]).dt.year
    merged = model.merge(wide.drop(columns="as_of"), on="year", how="left")
    merged = merged.merge(fair.drop(columns="as_of"), on="year", how="left").drop(columns="year")
    return merged, pd.concat(details, ignore_index=True)


def summarise(comparison: pd.DataFrame) -> pd.DataFrame:
    """Model over disclosed, shock by shock and date by date.

    A ratio rather than a difference, because the whole expectation is that the level will not
    match and the question is whether the gap is stable. A gap that holds the same multiple across
    four years and six shocks has one cause; a gap that wanders has several.
    """
    rows = []
    for shock in SHOCKS:
        both = comparison[["as_of", f"model_{shock}", f"disclosed_{shock}"]].dropna()
        if both.empty:
            continue
        ratio = both[f"model_{shock}"] / both[f"disclosed_{shock}"]
        rows.append({"shock": shock, "comparisons": len(both),
                     "model_over_disclosed_min": float(ratio.min()),
                     "model_over_disclosed_max": float(ratio.max()),
                     "model_over_disclosed_mean": float(ratio.mean()),
                     "signs_agree": int((np.sign(both[f"model_{shock}"])
                                         == np.sign(both[f"disclosed_{shock}"])).sum())})
    return pd.DataFrame(rows)


def main() -> None:
    paths.ensure_output_dirs()
    setup = build(cache_size=5)
    panel = pd.read_csv(paths.FRED_PANEL, comment="#", parse_dates=["date"]).set_index("date")
    history = scenarios.load_history(panel, setup["calibration"].heston,
                                    setup["calibration"].mix, dividend_yield=DIVIDEND_YIELD)
    disclosed = load_disclosed()

    attributions = vintage_attributions(setup["valuer"], panel, history,
                                        setup["calibration"], setup["terms"])
    print("Attribution percentage by vintage, each on the market of its own issue date")
    for issue_date, issue_age, _ in VINTAGES:
        state = market_at(panel, history, setup["calibration"], issue_date)
        print(f"  {issue_date}  issued at {issue_age}  ten-year zero "
              f"{state.curve.zero(10.0):.2%}  attribution {attributions[issue_date]:.4f}")

    comparison, detail = compare(setup, history, panel, disclosed, attributions)
    comparison.to_csv(paths.TABLES / "portfolio_vs_disclosed.csv", index=False,
                      float_format="%.4f")
    detail.to_csv(paths.TABLES / "portfolio_vintage_detail.csv", index=False,
                  float_format="%.4f")

    print("\nThe portfolio at each disclosed date")
    print("  as_of        age  GWB/AV  GAWA  deferring  NAR %AV   value    eq-10%   +100bp")
    for _, row in comparison.iterrows():
        print(f"  {row['as_of']}  {row['weighted_attained_age']:4.1f} "
              f"{row['gwb_over_av']:7.3f} {row['weighted_gawa_pct']:5.2f} "
              f"{row['deferring_share']:9.0%} {row['net_amount_at_risk_pct_av']:8.2f} "
              f"{row['model_value_pct_av']:8.2f} {row['model_equity_down_10pct']:8.2f} "
              f"{row['model_rates_up_100bp']:8.2f}")
    print("  disclosed, same columns")
    for _, row in comparison.iterrows():
        print(f"  {row['as_of']}  {'':4s} {'':7s} {'':5s} {'':9s} {'':8s} "
              f"{row['disclosed_value_pct_av']:8.2f} "
              f"{row['disclosed_equity_down_10pct']:8.2f} "
              f"{row['disclosed_rates_up_100bp']:8.2f}")

    summary = summarise(comparison)
    summary.to_csv(paths.TABLES / "portfolio_comparison_summary.csv", index=False,
                   float_format="%.4f")
    print("\nModel over disclosed, by shock")
    for _, row in summary.iterrows():
        print(f"  {row['shock']:<18s} {int(row['comparisons'])} dates, "
              f"{row['model_over_disclosed_mean']:5.2f}x on average and "
              f"{row['model_over_disclosed_min']:.2f} to "
              f"{row['model_over_disclosed_max']:.2f} across them, signs agree "
              f"{int(row['signs_agree'])} of {int(row['comparisons'])}")
    equity = summary[summary["shock"].str.startswith("equity")]
    rates = summary[summary["shock"].str.startswith("rates")]
    print(f"\n  the equity shocks come out {equity['model_over_disclosed_mean'].mean():.2f} times "
          f"the disclosed figure and the rate shocks "
          f"{rates['model_over_disclosed_mean'].mean():.2f} times, against "
          f"{len(VINTAGES)} vintages at a weighted attained age of "
          f"{comparison['weighted_attained_age'].iloc[-1]:.1f} and a blended withdrawal rate of "
          f"{comparison['weighted_gawa_pct'].iloc[-1]:.2f}%")
    print(f"\nwrote three tables to {paths.TABLES}")


if __name__ == "__main__":
    main()
