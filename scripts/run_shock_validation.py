"""The model against Jackson's disclosed sensitivity table: sign, shape, and where the level sits.

Item 7A runs the same two shocks every year: a 10% move in equity prices and a parallel shift in
risk-free rates, 50bp through the FY2024 filing and 100bp from FY2025. Those four year-ends are
the primary validation target, and they are compared in increasing order of difficulty.

*Sign.* A guarantee gets dearer when equity markets fall, and a long-dated liability shrinks when
discount rates rise. Both signs have to come out of the model without being put in.

*Shape.* The disclosed impacts are asymmetric, and where a year gives both shock sizes the ratio
of the 100bp impact to the 50bp impact is above two on the downside and below two on the upside.
Reproducing that convexity is a stronger test than reproducing a sign, because it cannot be got
right by accident.

*Level.* One $100,000 policy against a $236bn book spread over issue years, ages, benefit options
and withdrawal status. Matching the level is not a fair expectation and the project's own
hypothesis four says it should not happen. What is reported instead is where the disclosed figure
falls on the model's own moneyness curve, and therefore what would have to be true of the book for
it to sit there.

The in-force comparison rolls one policy from its 2016 inception along realised market history to
each disclosed balance-sheet date, with the attribution percentage fixed at inception. Holding that
percentage static is what the accounting requires, and it is the mechanism that turns a guarantee
written at a 1.6% ten-year rate into a net asset once rates are above 4%.

One approximation, stated because it is structural rather than incidental. The contract recursion
steps in whole policy years, so a contract can only be valued at an integer duration, while the
disclosed dates are year-ends and the inception is in September. Each date is therefore valued at
the anniversary on or before it, carrying the account value and benefit base the roll produces on
the disclosed date itself and that date's market. The remaining horizon is then up to a year long,
which is the same integer-duration approximation the cohort portfolio uses everywhere else. Its
size is measured rather than asserted: every date is also valued at the following anniversary and
the spread between the two is reported alongside.

Usage:  python -m scripts.run_shock_validation
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from vahedge import paths
from vahedge.hedge import simulator
from vahedge.liability import cohorts, mortality
from vahedge.market import scenarios
from vahedge.market import state as market_state
from vahedge.valuation import greeks as greeks_module
from vahedge.valuation.engine import MarketState

from scripts.run_valuation import (
    BASE_CONTRACT_CHARGE,
    DEFERRAL_YEARS,
    FUND_EXPENSE,
    ISSUE_AGE,
    MAX_AGE,
    PREMIUM,
    build,
)

INCEPTION = "2016-09-26"       # the first S&P 500 observation FRED carries
DISCLOSED_DATES = ("2022-12-30", "2023-12-29", "2024-12-31", "2025-12-31")
DIVIDEND_YIELD = 0.015
# Moneyness grid for locating the disclosed figure. Finer than the Greeks script's grid and
# reaching further down, because the disclosed book sits well below one and the interpolation
# wants points either side of it.
LOCATE_GRID = (0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6)
CONVEXITY_RATIOS = (0.8, 1.0, 1.2)
# Guaranteed withdrawal rates for the diagnostic. The grid reaches far below the rate sheet's
# lowest band because that is where the disclosed sensitivities turned out to sit, and a grid that
# stopped short reported its own endpoint by silently clamping the interpolation. These are
# effective rates for a whole book rather than contractual ones: a book where a large share of
# contracts have not started withdrawing draws less than any band.
GAWA_GRID = (0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.0575, 0.065)


def load_disclosed() -> pd.DataFrame:
    frame = pd.read_csv(paths.DATA_PROCESSED / "disclosed_scaled.csv")
    return frame.query("line_item == 'market_risk_benefits'").drop_duplicates(
        subset=["as_of", "shock"]
    )


def convexity_from_disclosure(disclosed: pd.DataFrame) -> pd.DataFrame:
    """Ratio of the 100bp impact to the 50bp impact, where a year discloses both."""
    pivot = disclosed.pivot_table(index="as_of", columns="shock", values="impact_musd")
    rows = []
    for as_of, row in pivot.iterrows():
        if {"rates_up_50bp", "rates_up_100bp"} <= set(row.dropna().index):
            rows.append({"as_of": as_of,
                         "up_100_over_50": row["rates_up_100bp"] / row["rates_up_50bp"],
                         "down_100_over_50": row["rates_down_100bp"] / row["rates_down_50bp"]})
    return pd.DataFrame(rows)


def contract_at(account_value: float, benefit_base: float, years_since_issue: int, terms):
    return cohorts.single_contract(
        terms, issue_age=ISSUE_AGE, base_contract_charge=BASE_CONTRACT_CHARGE,
        fund_expense=FUND_EXPENSE, premium=PREMIUM, account_value=account_value,
        benefit_base=benefit_base, deferral_years=max(DEFERRAL_YEARS - years_since_issue, 0),
        years_since_issue=years_since_issue, max_age=MAX_AGE,
    )


def shock_row(valuer, book, state, attribution) -> dict:
    """The disclosed shocks at one state, as shares of account value."""
    frame = greeks_module.disclosed_shocks(valuer, book, state, attribution)
    out = {"account_value": float(book.account_value[0]),
           "value": float(frame.loc[0, "value"])}
    for _, row in frame.iterrows():
        if row["shock"] != "base":
            out[row["shock"]] = float(row["change"])
    out["value_pct_av"] = 100 * out["value"] / out["account_value"]
    for name in [c for c in out if c.startswith(("equity_", "rates_"))]:
        out[f"{name}_pct_av"] = 100 * out[name] / out["account_value"]
    return out


def model_convexity(setup, ratios=CONVEXITY_RATIOS) -> pd.DataFrame:
    """The 100-over-50 ratio the model produces, at three levels of moneyness."""
    valuer, state, terms = setup["valuer"], setup["state"], setup["terms"]
    rows = []
    for ratio in ratios:
        book = contract_at(PREMIUM, PREMIUM * ratio, 0, terms)
        shocks = shock_row(valuer, book, state, valuer.calibrate_attribution(book, state))
        rows.append({
            "gwb_over_av": ratio,
            "up_100_over_50": shocks["rates_up_100bp"] / shocks["rates_up_50bp"],
            "down_100_over_50": shocks["rates_down_100bp"] / shocks["rates_down_50bp"],
            "equity_down_over_up": -shocks["equity_down_10pct"] / shocks["equity_up_10pct"],
        })
    return pd.DataFrame(rows)


def rolled_states(history, terms, equity_weight: float) -> dict:
    """Account value and benefit base on each disclosed date, from one policy rolled from 2016.

    The roll is the contract's own recursion run along realised history rather than a
    reconstruction: the same code the hedge backtest uses, so the two cannot drift apart.
    """
    at_issue = contract_at(PREMIUM, PREMIUM, 0, terms)
    out = {}
    for target in DISCLOSED_DATES:
        window = history.window(INCEPTION, target, label=f"inception to {target}")
        survival, deaths = mortality.load("basic").rates(
            at_issue.attained_age, pd.Timestamp(INCEPTION).year,
            int(at_issue.projection_years.max()), 0.5,
        )
        rolled = simulator.roll_contract(at_issue, window, survival, deaths,
                                        equity_weight=equity_weight)
        daily = simulator.daily_state(window, rolled, at_issue, PREMIUM, PREMIUM)
        elapsed = float(window.year_fraction[-1])
        out[target] = {
            "account_value": float(daily["account_value"][-1]),
            "benefit_base": float(daily["benefit_base"][-1]),
            "elapsed_years": elapsed,
            "date": window.dates[-1],
        }
    return out


def market_at(panel: pd.DataFrame, history, calibration, date) -> MarketState:
    """The market state on a past date: that date's curve and variance, today's surface shape.

    No free historical option data exists to recalibrate the surface, so what moves with the date
    is the curve and the observable instantaneous variance, and what is held is the speed of mean
    reversion, the long-run level and the skew. That is stated rather than hidden: it means the
    in-force comparison is run with the volatility surface of December 2025 attached to the rate
    environment of each disclosed year.
    """
    stamp = pd.Timestamp(date)
    position = int(np.searchsorted(history.dates, stamp, side="right")) - 1
    variance = float(history.variance[max(position, 0)])
    dated = replace(calibration, curve=market_state.treasury_curve(panel, stamp),
                    heston=replace(calibration.heston, v0=max(variance, 1e-6)))
    return replace(MarketState.from_calibration(dated), valuation_year=stamp.year)


def in_force_comparison(setup, disclosed: pd.DataFrame, panel, history) -> pd.DataFrame:
    """Roll one policy to each disclosed date and reprice the shocks there."""
    valuer, terms, calibration = setup["valuer"], setup["terms"], setup["calibration"]

    # The percentage is set at inception, on the market of that day.
    inception_state = market_at(panel, history, calibration, INCEPTION)
    at_issue = contract_at(PREMIUM, PREMIUM, 0, terms)
    alpha = valuer.calibrate_attribution(at_issue, inception_state)
    print(f"  attribution fixed at inception {INCEPTION}: {float(alpha[0]):.4f} "
          f"(ten-year zero then {inception_state.curve.zero(10.0):.2%}, volatility "
          f"{np.sqrt(inception_state.heston.v0):.2%})")

    states = rolled_states(history, terms, setup["state"].mix.equity_weight)
    rows = []
    for target, rolled in states.items():
        state = market_at(panel, history, calibration, rolled["date"])
        elapsed = rolled["elapsed_years"]
        for duration, label in ((int(np.floor(elapsed)), "anniversary on or before"),
                                (int(np.floor(elapsed)) + 1, "the following anniversary")):
            book = contract_at(rolled["account_value"], rolled["benefit_base"], duration, terms)
            shocks = shock_row(valuer, book, state, alpha)
            rows.append({
                "as_of": str(pd.Timestamp(rolled["date"]).date()),
                "duration_basis": label,
                "years_since_issue": duration,
                "elapsed_years": elapsed,
                "attained_age": int(book.attained_age[0]),
                "account_value": rolled["account_value"],
                "benefit_base": rolled["benefit_base"],
                "gwb_over_av": rolled["benefit_base"] / rolled["account_value"],
                "ten_year_zero_pct": 100 * float(state.curve.zero(10.0)),
                "model_value_pct_av": shocks["value_pct_av"],
                **{f"model_{name}": shocks[f"{name}_pct_av"]
                   for name in ("equity_up_10pct", "equity_down_10pct", "rates_up_50bp",
                                "rates_down_50bp", "rates_up_100bp", "rates_down_100bp")},
            })

    model = pd.DataFrame(rows)
    wide = disclosed.pivot_table(index="as_of", columns="shock", values="impact_pct_of_av")
    wide = (100 * wide).add_prefix("disclosed_").reset_index()
    fair = (disclosed.drop_duplicates(subset=["as_of"])[["as_of", "fair_value_pct_of_av"]]
            .assign(disclosed_value_pct_av=lambda f: 100 * f["fair_value_pct_of_av"])
            .drop(columns="fair_value_pct_of_av"))
    for frame in (model, wide, fair):
        frame["year"] = pd.to_datetime(frame["as_of"]).dt.year
    merged = model.merge(wide.drop(columns="as_of"), on="year", how="left")
    return merged.merge(fair.drop(columns="as_of"), on="year", how="left").drop(columns="year")


def sign_agreement(comparison: pd.DataFrame) -> pd.DataFrame:
    """Does the model's shock response point the same way as the disclosure's, shock by shock."""
    primary = comparison[comparison["duration_basis"] == "anniversary on or before"]
    rows = []
    for shock in ("equity_down_10pct", "equity_up_10pct", "rates_up_100bp", "rates_down_100bp",
                  "rates_up_50bp", "rates_down_50bp"):
        both = primary[[f"model_{shock}", f"disclosed_{shock}"]].dropna()
        if both.empty:
            continue
        agree = int((np.sign(both[f"model_{shock}"])
                     == np.sign(both[f"disclosed_{shock}"])).sum())
        rows.append({"shock": shock, "comparisons": len(both), "signs_agree": agree})
    return pd.DataFrame(rows)


def locate_disclosure_on_curve(profile: pd.DataFrame, disclosed: pd.DataFrame) -> pd.DataFrame:
    """For each disclosed year, the moneyness at which the model would match it.

    Linear interpolation along the model's own curve. Where the disclosed figure falls outside
    the range the model reaches at any moneyness, that is reported rather than extrapolated,
    because an extrapolated moneyness is a number with no contract behind it.
    """
    rows = []
    pivot = disclosed.pivot_table(index="as_of", columns="shock", values="impact_pct_of_av")
    for shock in ("equity_down_10pct", "equity_up_10pct", "rates_up_100bp", "rates_down_100bp"):
        column = f"{shock}_pct_av"
        if shock not in pivot.columns or column not in profile:
            continue
        curve = profile[["gwb_over_av", column]].dropna().sort_values(column)
        for as_of, value in pivot[shock].dropna().items():
            target = 100 * value
            inside = curve[column].min() <= target <= curve[column].max()
            rows.append({
                "as_of": as_of, "shock": shock, "disclosed_pct_av": target,
                "model_range_low": curve[column].min(),
                "model_range_high": curve[column].max(),
                "implied_gwb_over_av": (float(np.interp(target, curve[column],
                                                       curve["gwb_over_av"]))
                                        if inside else np.nan),
                "within_model_range": inside,
            })
    return pd.DataFrame(rows)


def locate_profile(setup) -> pd.DataFrame:
    """The model's shock responses across moneyness, on the grid the locator interpolates over."""
    valuer, state, terms = setup["valuer"], setup["state"], setup["terms"]
    rows = []
    for ratio in LOCATE_GRID:
        book = contract_at(PREMIUM, PREMIUM * ratio, 0, terms)
        shocks = shock_row(valuer, book, state, valuer.calibrate_attribution(book, state))
        rows.append({"gwb_over_av": ratio,
                     **{name: value for name, value in shocks.items()
                        if name.endswith("_pct_av")}})
    return pd.DataFrame(rows)


def withdrawal_rate_diagnostic(setup) -> pd.DataFrame:
    """How the rate sensitivity depends on the guaranteed withdrawal rate.

    The model's rate sensitivity sits well above the disclosed figure, and this is the reason. A
    contract drawing 5.75% a year against a risk-neutral drift near 4.2% less 2.26% of charges
    exhausts its account on most paths, which turns the guarantee into a long-dated life annuity
    and gives it that annuity's duration. A book where many contracts have not started
    withdrawals, or draw at the lower rates the older benefit options carry, has far less of it.
    """
    valuer, state, terms = setup["valuer"], setup["state"], setup["terms"]
    rows = []
    for gawa in GAWA_GRID:
        book = contract_at(PREMIUM, PREMIUM, 0, terms)
        book = replace(book, gawa_pct=np.array([gawa]))
        attribution = valuer.calibrate_attribution(book, state)
        priced = valuer.value(book, state, attribution=attribution)
        shocks = shock_row(valuer, book, state, attribution)
        rows.append({
            "gawa_pct": 100 * gawa,
            "attribution": float(np.ravel(attribution)[0]),
            "pv_claims_pct_av": 100 * priced.total_claims / priced.account_value,
            "prob_exhausted_20y": float(priced.projection.exhaustion_prob[0, 19]),
            "equity_down_10pct_pct_av": shocks["equity_down_10pct_pct_av"],
            "rates_up_100bp_pct_av": shocks["rates_up_100bp_pct_av"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    paths.ensure_output_dirs()
    # Five, which is the number of market states the shock grid cycles through: the base and the
    # four rate shifts. Four evicts one of them on every pass and re-simulates it; eight holds
    # several dates' worth at fifty megabytes each and takes the process past its memory.
    setup = build(cache_size=5)
    panel = pd.read_csv(paths.FRED_PANEL, comment="#", parse_dates=["date"]).set_index("date")
    history = scenarios.load_history(panel, setup["calibration"].heston,
                                    setup["calibration"].mix, dividend_yield=DIVIDEND_YIELD)
    disclosed = load_disclosed()

    print("Disclosed sensitivities, scaled by variable annuity account value (%)")
    view = disclosed.pivot_table(index="as_of", columns="shock", values="impact_pct_of_av")
    print((100 * view).to_string(float_format=lambda v: f"{v:,.4f}"))

    print("\nConvexity: the 100bp impact over the 50bp impact")
    disclosed_convexity = convexity_from_disclosure(disclosed)
    print(disclosed_convexity.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    model = model_convexity(setup)
    print("  the model, at three levels of moneyness")
    print(model.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    disclosed_convexity.to_csv(paths.TABLES / "convexity_disclosed.csv", index=False,
                               float_format="%.4f")
    model.to_csv(paths.TABLES / "convexity_model.csv", index=False, float_format="%.4f")

    print("\nOne policy rolled from inception to each disclosed date")
    comparison = in_force_comparison(setup, disclosed, panel, history)
    comparison.to_csv(paths.TABLES / "in_force_vs_disclosed.csv", index=False,
                      float_format="%.4f")
    primary = comparison[comparison["duration_basis"] == "anniversary on or before"]
    print("  as_of        age   AV      GWB/AV  10y     value    eq-10%   eq+10%  "
          "+100bp  -100bp")
    for _, row in primary.iterrows():
        print(f"  {row['as_of']}  {row['attained_age']:3d} {row['account_value']:8,.0f} "
              f"{row['gwb_over_av']:7.3f} {row['ten_year_zero_pct']:5.2f} "
              f"{row['model_value_pct_av']:8.2f} {row['model_equity_down_10pct']:8.2f} "
              f"{row['model_equity_up_10pct']:8.2f} {row['model_rates_up_100bp']:7.2f} "
              f"{row['model_rates_down_100bp']:7.2f}")
    print("  disclosed, same columns")
    for _, row in primary.iterrows():
        print(f"  {row['as_of']}  {'':3s} {'':8s} {'':7s} {'':5s} "
              f"{row['disclosed_value_pct_av']:8.2f} "
              f"{row['disclosed_equity_down_10pct']:8.2f} "
              f"{row['disclosed_equity_up_10pct']:8.2f} "
              f"{row['disclosed_rates_up_100bp']:7.2f} "
              f"{row['disclosed_rates_down_100bp']:7.2f}")

    spread = (comparison.pivot_table(index="as_of", columns="duration_basis",
                                     values="model_equity_down_10pct"))
    gap = (spread["the following anniversary"] - spread["anniversary on or before"]).abs()
    print(f"  the integer-duration approximation moves the equity-down figure by "
          f"{gap.min():.2f} to {gap.max():.2f} points of account value across the four dates, "
          f"against a disclosed figure of about "
          f"{primary['disclosed_equity_down_10pct'].mean():.2f}")

    checks = sign_agreement(comparison)
    checks.to_csv(paths.TABLES / "sign_agreement.csv", index=False, float_format="%.0f")
    print("\nSign agreement")
    for _, row in checks.iterrows():
        print(f"  {row['shock']:<20s} {int(row['signs_agree'])} of "
              f"{int(row['comparisons'])}")

    profile = locate_profile(setup)
    located = locate_disclosure_on_curve(profile, disclosed)
    located.to_csv(paths.TABLES / "disclosure_located_on_curve.csv", index=False,
                   float_format="%.4f")
    print("\nWhere the disclosed figure sits on the model's moneyness curve")
    for _, row in located.iterrows():
        where = (f"{row['implied_gwb_over_av']:.3f}" if row["within_model_range"]
                 else f"outside [{row['model_range_low']:.2f}, {row['model_range_high']:.2f}]")
        print(f"  {row['as_of']}  {row['shock']:<18s} disclosed "
              f"{row['disclosed_pct_av']:+7.3f}  implied GWB/AV {where}")

    diagnostic = withdrawal_rate_diagnostic(setup)
    diagnostic.to_csv(paths.TABLES / "withdrawal_rate_diagnostic.csv", index=False,
                      float_format="%.4f")
    print("\nWhy the rate sensitivity is larger than the disclosure")
    print("  GAWA   alpha  claims %AV  exhausted by 20y   eq-10%  +100bp")
    for _, row in diagnostic.iterrows():
        print(f"  {row['gawa_pct']:5.2f} {row['attribution']:7.4f} "
              f"{row['pv_claims_pct_av']:11.2f} {row['prob_exhausted_20y']:18.3f} "
              f"{row['equity_down_10pct_pct_av']:8.2f} {row['rates_up_100bp_pct_av']:7.2f}")
    # The number that makes the structural explanation quantitative rather than directional:
    # the withdrawal rate at which the model's rate sensitivity would equal the disclosure's.
    # Both averages over the same dates - the two year-ends that disclose a 100bp shock -
    # because averaging the equity figure over four years and the rate figure over two compares
    # a book that was further in the money against one that was not.
    comparable = primary.dropna(subset=["disclosed_rates_up_100bp"])
    disclosed_rate = float(comparable["disclosed_rates_up_100bp"].mean())
    disclosed_equity = float(comparable["disclosed_equity_down_10pct"].mean())
    rate_curve, equity_curve = diagnostic, diagnostic
    def monotone_branch(curve: pd.DataFrame, column: str) -> pd.DataFrame:
        """The part of the diagnostic where the response still moves with the withdrawal rate.

        Below about one and a half per cent the account stops exhausting at all - the share of
        paths with no contract value left by year twenty falls from 76% at the contractual rate
        to 13% - and the living benefit stops being what the sensitivity is made of. The death
        benefit and the fee stream take over and the response turns back up, so the curve is not
        monotone down there and reading an implied rate off it would be reading the turn.
        """
        ordered = curve.sort_values("gawa_pct", ascending=False).reset_index(drop=True)
        steps = np.sign(np.diff(ordered[column].to_numpy(dtype=float)))
        direction = steps[0] if steps.size else 0.0
        keep = len(ordered)
        for position, step in enumerate(steps, start=1):
            if step != direction:
                keep = position
                break
        return ordered.iloc[:keep]

    def implied(target: float, curve: pd.DataFrame, column: str) -> str:
        """The withdrawal rate that would match a disclosed figure, over the monotone branch.

        np.interp holds its end values flat outside the data, so a target beyond the branch comes
        back as the nearest point and reads as a solution. Saying which side it fell off is the
        difference between a reconciliation and a coincidence.
        """
        branch = monotone_branch(curve, column).sort_values(column)
        low, high = float(branch[column].iloc[0]), float(branch[column].iloc[-1])
        if low <= target <= high:
            return f"{float(np.interp(target, branch[column], branch['gawa_pct'])):.2f}%"
        nearest = branch.iloc[0] if abs(target - low) < abs(target - high) else branch.iloc[-1]
        rate = float(nearest["gawa_pct"])
        side = "at or below" if rate <= float(branch["gawa_pct"].min()) else "above"
        return f"{side} {rate:.2f}%"

    base_row = diagnostic[diagnostic["gawa_pct"] == 5.75].iloc[0]
    rate_factor = (float(np.interp(1.69, diagnostic["gawa_pct"],
                                   diagnostic["rates_up_100bp_pct_av"]))
                   / float(base_row["rates_up_100bp_pct_av"]))
    equity_factor = (float(np.interp(1.69, diagnostic["gawa_pct"],
                                     diagnostic["equity_down_10pct_pct_av"]))
                     / float(base_row["equity_down_10pct_pct_av"]))
    print(f"\n  matching the disclosed rate sensitivity of {disclosed_rate:.2f} takes a "
          f"withdrawal rate of "
          f"{implied(disclosed_rate, rate_curve, 'rates_up_100bp_pct_av')}, and matching the "
          f"disclosed equity sensitivity of {disclosed_equity:.2f} takes "
          f"{implied(disclosed_equity, equity_curve, 'equity_down_10pct_pct_av')}, against the "
          f"5.75% the rate sheet gives a 70-year-old drawing from age 75, both averaged over the "
          f"{len(comparable)} year-ends that disclose a 100bp shock.")
    print(f"  One lever moves both towards the disclosure, from "
          f"{base_row['rates_up_100bp_pct_av']:.2f} and "
          f"{base_row['equity_down_10pct_pct_av']:.2f}, but not by the same factor: at a 1.7% "
          f"draw the rate sensitivity is {rate_factor:.2f} of its contractual-rate value and the "
          f"equity sensitivity {equity_factor:.2f} of its own. The withdrawal rate therefore "
          f"narrows the gap rather than closing it, and the behaviour reconciliation measures "
          f"how far it gets at the portfolio level.")
    print("  It is an effective rate rather than a contractual one: a book where a share of "
          "contracts have not started withdrawing draws less than any of its rate sheet bands, "
          "and the vintage portfolio models that share directly rather than through this lever.")
    print(f"\nwrote six tables to {paths.TABLES}")


if __name__ == "__main__":
    main()
