"""Aggregate a handful of policy vintages and compare the result with the book.

One policy cannot match Jackson's disclosed sensitivities on scale, and the reason is
visible in the single-policy results: matching the book's moneyness and matching its
weighted-average attained age pull the answer in opposite directions. A contract issued
in 2016 has ridden the market up and is out of the money, which is right, but it is also
nine years older than the book's average attained age of 70, which shortens the
guarantee and understates its duration.

Five vintages fix that without turning the project into a seriatim valuation. Issue
dates run from 2016 to 2024, and each issue age is chosen so the attained ages in 2025
straddle 70. Because the rate sheet bands the guaranteed withdrawal percentage by age,
the vintages also carry different withdrawal rates, which is a feature rather than a
nuisance: a real book draws at a blended rate below the rate a new 70-year-old gets.

What this still does not represent: contracts that have not started withdrawals. Those
accrue a bonus to the benefit base that is not modelled here, so they are left out
rather than modelled wrongly. Their absence biases the portfolio towards more guarantee
duration than the book has, which is the direction the residual gap runs.

Usage:  python -m scripts.run_portfolio_validation
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from gmwb import market, mortality, paths, session
from gmwb.engine import calibrate_attribution, make_normals, projection_years, value_rider
from gmwb.hedging import SubAccountMix, roll_policy
from gmwb.sensitivities import disclosed_shock_repricing

# (issue date, issue age, GAWA% from the rate sheet band for that age, premium weight)
VINTAGES = [
    ("2016-09-26", 60, 0.0400, 1.0),
    ("2018-06-29", 63, 0.0400, 1.0),
    ("2020-06-30", 66, 0.0555, 1.0),
    ("2022-06-30", 69, 0.0555, 1.0),
    ("2024-06-28", 72, 0.0575, 1.0),
]

DISCLOSED_DATES = ["2022-12-30", "2023-12-29", "2024-12-31", "2025-12-31"]

SHOCK_KEYS = [
    "equity_up_10pct",
    "equity_down_10pct",
    "rates_up_50bp",
    "rates_down_50bp",
    "rates_up_100bp",
    "rates_down_100bp",
]


def value_vintage(s: session.Session, normals: np.ndarray, mix: SubAccountMix,
                  issue_date: str, issue_age: int, gawa: float,
                  target: pd.Timestamp) -> dict[str, float]:
    """Roll one vintage to the target date and reprice the disclosed shocks there."""
    cfg = s.cfg
    max_age = int(cfg["simulation"]["max_age"])
    male_weight = float(cfg["contract"]["sex_mix"]["male"])
    contract = replace(s.contract, issue_age=issue_age, gawa_pct=gawa)

    # Attribution percentage, fixed on the market of the issue date.
    inception = market.equity_dates(s.panel, issue_date, None)[0]
    inception_state = market.state_at(s.panel, inception, cfg, s.long_run_vol)
    inception_basis = mortality.load(issue_age, inception.year, male_weight)
    n_years_issue = projection_years(contract, max_age)
    at_inception = value_rider(
        replace(contract, fund_equity_beta=mix.effective_equity_beta),
        inception_state.curve_builder.build(),
        inception_state.vol,
        inception_basis,
        normals[:, :n_years_issue],
        max_age=max_age,
    )
    alpha = calibrate_attribution(at_inception)

    policy, _ = roll_policy(contract, cfg, s.panel, mix, inception, target)
    state = market.state_at(s.panel, target, cfg, s.long_run_vol)
    attained = policy.attained_age(contract, target)
    aged = replace(contract, issue_age=attained, fund_equity_beta=mix.effective_equity_beta)
    basis = mortality.load(attained, target.year, male_weight)
    n_years = projection_years(aged, max_age)

    result = disclosed_shock_repricing(
        aged, state.curve_builder, state.vol, basis, normals[:, :n_years],
        {
            "account_value": policy.account_value,
            "benefit_base": policy.benefit_base,
            "fee_attribution": alpha,
            "max_age": max_age,
            "first_step_years": policy.years_to_anniversary(target),
        },
    )
    result.update(
        {
            "issue_date": str(inception.date()),
            "issue_age": issue_age,
            "attained_age": attained,
            "gawa_pct": 100 * gawa,
            "attribution": alpha,
            "benefit_base": policy.benefit_base,
        }
    )
    return result


def aggregate(rows: list[dict], weights: list[float]) -> dict[str, float]:
    """Add the vintages up. Dollar amounts add; the reported figures are then scaled by
    total account value, which is how the disclosure presents them."""
    weights = np.asarray(weights, dtype=float)
    total_av = float(sum(w * r["account_value"] for w, r in zip(weights, rows)))
    total_bb = float(sum(w * r["benefit_base"] for w, r in zip(weights, rows)))
    out = {
        "account_value": total_av,
        "benefit_base": total_bb,
        "gwb_over_av": total_bb / total_av,
        "value_pct_av": 100 * sum(w * r["base_value"] for w, r in zip(weights, rows)) / total_av,
        "weighted_attained_age": float(
            sum(w * r["account_value"] * r["attained_age"] for w, r in zip(weights, rows))
            / total_av
        ),
        "weighted_gawa_pct": float(
            sum(w * r["benefit_base"] * r["gawa_pct"] for w, r in zip(weights, rows)) / total_bb
        ),
    }
    for key in SHOCK_KEYS:
        out[f"model_{key}"] = 100 * sum(w * r[key] for w, r in zip(weights, rows)) / total_av
    return out


def main() -> None:
    s = session.start()
    mix = SubAccountMix.from_config(s.cfg)

    # The youngest vintage is issued well below the base-case issue age, so it needs a
    # longer projection than the session's normals cover. One wider array is drawn here
    # and every vintage takes the leading columns it needs, which keeps the draws common
    # across vintages and across the shocked revaluations.
    youngest = min(age for _, age, _, _ in VINTAGES)
    max_years = projection_years(replace(s.contract, issue_age=youngest),
                                int(s.cfg["simulation"]["max_age"]))
    normals = make_normals(s.normals.shape[0], max_years, int(s.cfg["simulation"]["seed"]),
                           bool(s.cfg["simulation"]["antithetic"]))
    print(f"{s.normals.shape[0]:,} paths, {max_years} projection years for the youngest vintage")
    disclosed = pd.read_csv(paths.DATA_PROCESSED / "disclosed_scaled.csv")
    disclosed = disclosed.query("line_item == 'market_risk_benefits'").drop_duplicates(
        subset=["as_of", "shock"]
    )
    wide = 100 * disclosed.pivot_table(index="as_of", columns="shock", values="impact_pct_of_av")
    levels = disclosed.drop_duplicates(subset=["as_of"]).set_index("as_of")[
        "fair_value_pct_of_av"
    ] * 100

    detail_rows = []
    aggregate_rows = []
    for target_label in DISCLOSED_DATES:
        target = market.equity_dates(s.panel, None, target_label)[-1]
        rows, weights = [], []
        for issue_date, issue_age, gawa, weight in VINTAGES:
            if pd.Timestamp(issue_date) >= target:
                continue
            result = value_vintage(s, normals, mix, issue_date, issue_age, gawa, target)
            result["as_of"] = str(target.date())
            detail_rows.append(result)
            rows.append(result)
            weights.append(weight)
        agg = aggregate(rows, weights)
        agg["as_of"] = str(target.date())
        agg["vintages"] = len(rows)
        aggregate_rows.append(agg)

    detail = pd.DataFrame(detail_rows)
    detail_view = detail[["as_of", "issue_date", "issue_age", "attained_age", "gawa_pct",
                          "attribution", "account_value", "benefit_base", "base_value",
                          "equity_down_10pct", "rates_up_100bp"]]
    print("Vintage detail")
    print(detail_view.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    session.write_table(detail_view, "portfolio_vintage_detail", "%.3f")

    frame = pd.DataFrame(aggregate_rows).set_index("as_of")
    year_key = pd.to_datetime(frame.index).to_period("Y").astype(str)
    disclosed_key = pd.to_datetime(wide.index).to_period("Y").astype(str)
    wide = wide.set_index(disclosed_key)
    levels.index = disclosed_key
    frame["year"] = year_key
    for key in SHOCK_KEYS:
        frame[f"disclosed_{key}"] = frame["year"].map(
            wide[key] if key in wide.columns else pd.Series(dtype=float)
        )
    frame["disclosed_value_pct_av"] = frame["year"].map(levels)

    print("\nPortfolio against the disclosure, all figures % of account value")
    columns = ["vintages", "weighted_attained_age", "weighted_gawa_pct", "gwb_over_av",
               "value_pct_av", "disclosed_value_pct_av"]
    print(frame[columns].to_string(float_format=lambda v: f"{v:,.3f}"))

    for label, model_col, disc_col in (
        ("equity -10%", "model_equity_down_10pct", "disclosed_equity_down_10pct"),
        ("equity +10%", "model_equity_up_10pct", "disclosed_equity_up_10pct"),
        ("rates +50bp", "model_rates_up_50bp", "disclosed_rates_up_50bp"),
        ("rates -50bp", "model_rates_down_50bp", "disclosed_rates_down_50bp"),
        ("rates +100bp", "model_rates_up_100bp", "disclosed_rates_up_100bp"),
        ("rates -100bp", "model_rates_down_100bp", "disclosed_rates_down_100bp"),
    ):
        both = frame[[model_col, disc_col]].dropna()
        if both.empty:
            continue
        ratio = (both[model_col] / both[disc_col]).round(2).to_dict()
        print(f"\n{label}")
        print(both.to_string(float_format=lambda v: f"{v:,.3f}"))
        print(f"  model over disclosed: {ratio}")

    session.write_table(frame.reset_index(), "portfolio_vs_disclosed", "%.4f")

    ratios = []
    for model_col, disc_col in (
        ("model_equity_down_10pct", "disclosed_equity_down_10pct"),
        ("model_equity_up_10pct", "disclosed_equity_up_10pct"),
        ("model_rates_up_50bp", "disclosed_rates_up_50bp"),
        ("model_rates_down_50bp", "disclosed_rates_down_50bp"),
        ("model_rates_up_100bp", "disclosed_rates_up_100bp"),
        ("model_rates_down_100bp", "disclosed_rates_down_100bp"),
    ):
        both = frame[[model_col, disc_col]].dropna()
        for as_of, row in both.iterrows():
            ratios.append(
                {
                    "as_of": as_of,
                    "shock": disc_col.replace("disclosed_", ""),
                    "model_pct_av": row[model_col],
                    "disclosed_pct_av": row[disc_col],
                    "ratio": row[model_col] / row[disc_col],
                    "sign_agrees": bool(np.sign(row[model_col]) == np.sign(row[disc_col])),
                }
            )
    summary = pd.DataFrame(ratios)
    print("\nAll comparisons")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print(f"\nsigns agreeing: {int(summary['sign_agrees'].sum())} of {len(summary)}")
    print(f"median ratio of model to disclosed: {summary['ratio'].median():.2f}")
    print(f"ratio range: {summary['ratio'].min():.2f} to {summary['ratio'].max():.2f}")
    session.write_table(summary, "portfolio_comparison_summary", "%.4f")


if __name__ == "__main__":
    main()
