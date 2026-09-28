"""Compare the model's shock responses against Jackson's disclosed sensitivity table.

Jackson runs the same two shocks every year in Item 7A: a 10% move in equity prices
and a parallel shift in risk-free rates, 50bp through the FY2024 filing and 100bp from
the FY2025 filing. The disclosed impacts are the primary validation target for this
project.

A single $100,000 policy cannot be compared to a $236bn book in dollars, so every
figure is scaled by account value. Three things are being tested, in increasing order
of difficulty.

*Sign.* Guarantees get more expensive when equity markets fall, and a long-dated
liability shrinks when discount rates rise. Both signs should come out of the model
without being put in.

*Shape.* The disclosed impacts are asymmetric: the down shock moves the liability by
more than the up shock. Where a year discloses both shock sizes, the ratio of the
100bp to the 50bp impact is above two on the downside and below two on the upside.
Reproducing that convexity is a stronger test than reproducing a sign.

*Scale.* The model is one policy with one set of terms. Matching the level of a book
spread across issue years, ages, benefit elections and withdrawal status is not a fair
expectation, so what is reported is where on the model's own moneyness curve the
disclosed figure sits, and what has to be true of the book for it to sit there.

The in-force comparison rolls one policy from a 2016 inception along realised market
history to each disclosed balance-sheet date, with the fee attribution percentage fixed
at inception. Holding that percentage static is what the accounting requires, and it is
the mechanism that turns a guarantee written in a 1.6% rate environment into a net asset
once rates are above 4%.

Usage:  python -m scripts.run_shock_validation
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from gmwb import figures, market, mortality, paths, session
from gmwb.engine import calibrate_attribution, make_normals, projection_years, value_rider
from gmwb.hedging import SubAccountMix, roll_policy
from gmwb.sensitivities import disclosed_shock_repricing, moneyness_profile

INCEPTION = "2016-09-26"   # the first S&P 500 observation FRED carries
DISCLOSED_DATES = ["2022-12-30", "2023-12-29", "2024-12-31", "2025-12-31"]


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
            rows.append(
                {
                    "as_of": as_of,
                    "up_100_over_50": row["rates_up_100bp"] / row["rates_up_50bp"],
                    "down_100_over_50": row["rates_down_100bp"] / row["rates_down_50bp"],
                }
            )
    return pd.DataFrame(rows)


def model_convexity(s: session.Session, ratios=(0.8, 1.0, 1.2)) -> pd.DataFrame:
    rows = []
    for ratio in ratios:
        result = disclosed_shock_repricing(
            s.contract, s.state.curve_builder, s.state.vol, s.mortality, s.normals,
            s.valuation_state(account_value=s.contract.premium,
                             benefit_base=s.contract.premium * ratio),
        )
        rows.append(
            {
                "gwb_over_av": ratio,
                "up_100_over_50": result["rates_up_100bp"] / result["rates_up_50bp"],
                "down_100_over_50": result["rates_down_100bp"] / result["rates_down_50bp"],
                "equity_down_over_up": -result["equity_down_10pct"] / result["equity_up_10pct"],
            }
        )
    return pd.DataFrame(rows)


def in_force_comparison(s: session.Session, disclosed: pd.DataFrame) -> pd.DataFrame:
    """Roll one policy from 2016 to each disclosed date and reprice the shocks there."""
    cfg = s.cfg
    mix = SubAccountMix.from_config(cfg)
    max_age = int(cfg["simulation"]["max_age"])
    male_weight = float(cfg["contract"]["sex_mix"]["male"])
    n_paths = s.normals.shape[0]

    # The attribution percentage is set at inception, on the market of that day.
    inception_state = market.state_at(s.panel, INCEPTION, cfg, s.long_run_vol)
    inception_basis = mortality.load(s.contract.issue_age,
                                    pd.Timestamp(INCEPTION).year, male_weight)
    at_inception = value_rider(
        s.contract, inception_state.curve_builder.build(), inception_state.vol,
        inception_basis, s.normals, max_age=max_age,
    )
    alpha = calibrate_attribution(at_inception)
    print(f"  attribution fixed at inception {INCEPTION}: {alpha:.4f} "
          f"(10-year zero then {100 * inception_state.curve_builder.build().zero(10):.2f}%, "
          f"S&P 500 {inception_state.spot:,.0f})")

    rows = []
    for target in DISCLOSED_DATES:
        target = market.equity_dates(s.panel, None, target)[-1]
        policy, _ = roll_policy(s.contract, cfg, s.panel, mix, INCEPTION, target)
        state = market.state_at(s.panel, target, cfg, s.long_run_vol)
        attained = policy.attained_age(s.contract, target)
        aged = replace(s.contract, issue_age=attained,
                       fund_equity_beta=mix.effective_equity_beta)
        basis = mortality.load(attained, target.year, male_weight)
        n_years = projection_years(aged, max_age)
        normals = s.normals[:, :n_years]

        result = disclosed_shock_repricing(
            aged, state.curve_builder, state.vol, basis, normals,
            {
                "account_value": policy.account_value,
                "benefit_base": policy.benefit_base,
                "fee_attribution": alpha,
                "max_age": max_age,
                "first_step_years": policy.years_to_anniversary(target),
            },
        )
        av = result["account_value"]
        rows.append(
            {
                "as_of": str(target.date()),
                "attained_age": attained,
                "account_value": av,
                "benefit_base": policy.benefit_base,
                "gwb_over_av": policy.benefit_base / av,
                "model_value_pct_av": 100 * result["base_value"] / av,
                "model_equity_up_10pct": 100 * result["equity_up_10pct"] / av,
                "model_equity_down_10pct": 100 * result["equity_down_10pct"] / av,
                "model_rates_up_50bp": 100 * result["rates_up_50bp"] / av,
                "model_rates_down_50bp": 100 * result["rates_down_50bp"] / av,
                "model_rates_up_100bp": 100 * result["rates_up_100bp"] / av,
                "model_rates_down_100bp": 100 * result["rates_down_100bp"] / av,
            }
        )

    model = pd.DataFrame(rows)
    wide = disclosed.pivot_table(index="as_of", columns="shock", values="impact_pct_of_av")
    wide = (100 * wide).add_prefix("disclosed_").reset_index()
    wide["as_of"] = wide["as_of"].astype(str)
    fair = (
        disclosed.drop_duplicates(subset=["as_of"])[["as_of", "fair_value_pct_of_av"]]
        .assign(disclosed_value_pct_av=lambda f: 100 * f["fair_value_pct_of_av"])
        .drop(columns="fair_value_pct_of_av")
    )
    fair["as_of"] = fair["as_of"].astype(str)

    model["as_of_key"] = pd.to_datetime(model["as_of"]).dt.to_period("Y").astype(str)
    wide["as_of_key"] = pd.to_datetime(wide["as_of"]).dt.to_period("Y").astype(str)
    fair["as_of_key"] = pd.to_datetime(fair["as_of"]).dt.to_period("Y").astype(str)
    merged = model.merge(wide.drop(columns="as_of"), on="as_of_key", how="left")
    merged = merged.merge(fair.drop(columns="as_of"), on="as_of_key", how="left")
    return merged.drop(columns="as_of_key")


def locate_disclosure_on_curve(profile: pd.DataFrame, disclosed: pd.DataFrame) -> pd.DataFrame:
    """For each disclosed year, the moneyness at which the model matches it.

    Linear interpolation along the model's own curve. Where the disclosed figure falls
    outside the range the model produces at any moneyness, that is reported rather than
    extrapolated.
    """
    rows = []
    pivot = disclosed.pivot_table(index="as_of", columns="shock", values="impact_pct_of_av")
    for shock, column in (
        ("equity_down_10pct", "equity_down_10pct_pct_av"),
        ("equity_up_10pct", "equity_up_10pct_pct_av"),
        ("rates_up_100bp", "rates_up_100bp_pct_av"),
        ("rates_down_100bp", "rates_down_100bp_pct_av"),
    ):
        if shock not in pivot.columns:
            continue
        curve = profile[["gwb_over_av", column]].dropna().sort_values(column)
        for as_of, value in pivot[shock].dropna().items():
            target = 100 * value
            inside = curve[column].min() <= target <= curve[column].max()
            implied = (
                float(np.interp(target, curve[column], curve["gwb_over_av"]))
                if inside else np.nan
            )
            rows.append(
                {
                    "as_of": as_of,
                    "shock": shock,
                    "disclosed_pct_av": target,
                    "model_range_low": curve[column].min(),
                    "model_range_high": curve[column].max(),
                    "implied_gwb_over_av": implied,
                    "within_model_range": inside,
                }
            )
    return pd.DataFrame(rows)


def withdrawal_rate_diagnostic(s: session.Session) -> pd.DataFrame:
    """How the rate sensitivity depends on the guaranteed withdrawal rate.

    The model's rate sensitivity is well above the disclosed figure, and this is the
    reason. A contract drawing 5.75% a year against a risk-neutral drift near 3.7% less
    2.26% of charges exhausts its account value on nearly every path, which makes the
    guarantee a long-dated life annuity and gives it the duration to match. A book
    where many contracts have not started withdrawals, or draw at the lower rates older
    benefit options carry, has far less of that duration.
    """
    rows = []
    max_age = int(s.cfg["simulation"]["max_age"])
    for gawa in (0.03, 0.04, 0.05, 0.0575, 0.065):
        contract = replace(s.contract, gawa_pct=gawa)
        at_issue = value_rider(contract, s.state.curve_builder.build(), s.state.vol,
                               s.mortality, s.normals, max_age=max_age)
        alpha = calibrate_attribution(at_issue)
        result = disclosed_shock_repricing(
            contract, s.state.curve_builder, s.state.vol, s.mortality, s.normals,
            {"fee_attribution": alpha, "max_age": max_age},
        )
        av = result["account_value"]
        rows.append(
            {
                "gawa_pct": 100 * gawa,
                "attribution": alpha,
                "pv_claims_pct_av": 100 * at_issue.pv_claims / av,
                "prob_exhausted_20y": at_issue.exhaustion_prob[19],
                "equity_down_10pct_pct_av": 100 * result["equity_down_10pct"] / av,
                "rates_up_100bp_pct_av": 100 * result["rates_up_100bp"] / av,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    s = session.start()
    disclosed = load_disclosed()

    print("Disclosed market risk benefit sensitivities, scaled by variable annuity "
          "account value (%)")
    view = disclosed.pivot_table(index="as_of", columns="shock", values="impact_pct_of_av")
    print((100 * view).to_string(float_format=lambda v: f"{v:,.4f}"))

    print("\nConvexity in the disclosure: ratio of the 100bp impact to the 50bp impact")
    disc_convex = convexity_from_disclosure(disclosed)
    print(disc_convex.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print("Model, at three moneyness levels")
    mod_convex = model_convexity(s)
    print(mod_convex.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    session.write_table(disc_convex, "convexity_disclosed")
    session.write_table(mod_convex, "convexity_model")

    print("\nIn-force policy rolled from inception to each disclosed date")
    comparison = in_force_comparison(s, disclosed)
    columns = ["as_of", "attained_age", "account_value", "benefit_base", "gwb_over_av",
               "model_value_pct_av", "disclosed_value_pct_av",
               "model_equity_down_10pct", "disclosed_equity_down_10pct",
               "model_equity_up_10pct", "disclosed_equity_up_10pct"]
    print(comparison[columns].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    rate_columns = ["as_of", "model_rates_up_50bp", "disclosed_rates_up_50bp",
                    "model_rates_down_50bp", "disclosed_rates_down_50bp",
                    "model_rates_up_100bp", "disclosed_rates_up_100bp",
                    "model_rates_down_100bp", "disclosed_rates_down_100bp"]
    print(comparison[rate_columns].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    session.write_table(comparison, "in_force_vs_disclosed")

    print("\nSign agreement")
    pairs = [
        ("model_equity_down_10pct", "disclosed_equity_down_10pct"),
        ("model_equity_up_10pct", "disclosed_equity_up_10pct"),
        ("model_rates_up_100bp", "disclosed_rates_up_100bp"),
        ("model_rates_down_100bp", "disclosed_rates_down_100bp"),
        ("model_rates_up_50bp", "disclosed_rates_up_50bp"),
        ("model_rates_down_50bp", "disclosed_rates_down_50bp"),
    ]
    checks = []
    for model_col, disc_col in pairs:
        both = comparison[[model_col, disc_col]].dropna()
        if both.empty:
            continue
        agree = int((np.sign(both[model_col]) == np.sign(both[disc_col])).sum())
        checks.append({"shock": disc_col.replace("disclosed_", ""),
                       "comparisons": len(both), "signs_agree": agree})
    checks = pd.DataFrame(checks)
    print(checks.to_string(index=False))
    session.write_table(checks, "sign_agreement", "%.0f")

    print("\nWhere the disclosed figure sits on the model's moneyness curve")
    rows = moneyness_profile(
        s.contract, s.state.curve_builder, s.state.vol, s.mortality, s.normals,
        s.valuation_state(), ratios=(0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6),
    )
    profile = pd.DataFrame(rows)
    for column in ("base_value", "equity_up_10pct", "equity_down_10pct",
                   "rates_up_100bp", "rates_down_100bp"):
        profile[f"{column}_pct_av"] = 100 * profile[column] / profile["account_value"]
    located = locate_disclosure_on_curve(profile, disclosed)
    print(located.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    session.write_table(located, "disclosure_located_on_curve")

    print("\nWhy the rate sensitivity is larger than the disclosure")
    diag = withdrawal_rate_diagnostic(s)
    print(diag.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    session.write_table(diag, "withdrawal_rate_diagnostic")

    latest = disclosed[disclosed["as_of"] == disclosed["as_of"].max()]
    print("\nFigure")
    print("  " + figures.shock_comparison(profile, latest))


if __name__ == "__main__":
    main()
