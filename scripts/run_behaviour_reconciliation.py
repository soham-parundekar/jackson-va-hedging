"""What behaviour assumption would reconcile the model with the disclosed sensitivities.

The vintage portfolio matches the sign of every disclosed shock and tracks the year on
year decline in sensitivity, but its sensitivities are about 2.4 times the disclosed
figures, and that factor is close to constant across six shocks and four balance-sheet
dates. A single near-constant multiple points at one structural assumption rather than a
pile of small errors.

The candidate is policyholder behaviour. The base case draws the full guaranteed amount
every year and never surrenders, which Bauer, Kling and Russ (2008) use as the
benchmark case precisely because it is the most expensive one for the insurer. Jackson's
own fair value is built on assumed "benefit utilization by policyholders, lapse,
mortality, and withdrawal rates", so its liability reflects contract holders who draw
less than the maximum and some of whom surrender.

This script sweeps utilisation and lapse and reports which combinations bring the
portfolio's sensitivities into line. The point is not to fit the disclosure - that would
be reverse engineering a number rather than modelling a liability. The point is to
establish whether the gap can be closed by behaviour assumptions inside the range the
literature and the filings support, or whether something else has to be wrong.

Two biases run the other way and are worth holding in mind while reading the output. A
static lapse rate overstates surrender in exactly the states where the guarantee is
valuable, because real lapse falls when a guarantee is deep in the money. And roughly a
quarter of Jackson's variable annuity account value carries no living benefit at all,
which scales the disclosed sensitivity down without any behaviour assumption doing work.

Usage:  python -m scripts.run_behaviour_reconciliation
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from gmwb import market, mortality, paths, session
from gmwb.engine import calibrate_attribution, make_normals, projection_years, value_rider
from gmwb.hedging import SubAccountMix, roll_policy
from gmwb.sensitivities import disclosed_shock_repricing

from scripts.run_portfolio_validation import VINTAGES, SHOCK_KEYS, aggregate

TARGET_DATE = "2025-12-31"

# Share of Jackson's variable annuity account value carrying a withdrawal guarantee:
# GMWB for Life 72% plus GMWB 3% at 31 December 2025 (FY2025 10-K, Item 1).
GUARANTEED_SHARE = 0.75


def portfolio_at(s: session.Session, normals: np.ndarray, mix: SubAccountMix,
                 target: pd.Timestamp, utilisation: float, lapse: float) -> dict[str, float]:
    """Aggregate the vintages under one behaviour assumption.

    The attribution percentage is recalibrated at each vintage's own inception under the
    same behaviour assumption, because that is when the accounting fixes it and the
    behaviour assumption was already in the pricing at that point.
    """
    cfg = s.cfg
    max_age = int(cfg["simulation"]["max_age"])
    male_weight = float(cfg["contract"]["sex_mix"]["male"])
    rows, weights = [], []

    for issue_date, issue_age, gawa, weight in VINTAGES:
        if pd.Timestamp(issue_date) >= target:
            continue
        contract = replace(
            s.contract,
            issue_age=issue_age,
            gawa_pct=gawa,
            utilisation=utilisation,
            lapse_rate=lapse,
            fund_equity_beta=mix.effective_equity_beta,
        )
        inception = market.equity_dates(s.panel, issue_date, None)[0]
        inception_state = market.state_at(s.panel, inception, cfg, s.long_run_vol)
        n_years_issue = projection_years(contract, max_age)
        at_inception = value_rider(
            contract,
            inception_state.curve_builder.build(),
            inception_state.vol,
            mortality.load(issue_age, inception.year, male_weight),
            normals[:, :n_years_issue],
            max_age=max_age,
        )
        alpha = calibrate_attribution(at_inception)

        policy, _ = roll_policy(contract, cfg, s.panel, mix, inception, target)
        state = market.state_at(s.panel, target, cfg, s.long_run_vol)
        attained = policy.attained_age(contract, target)
        aged = replace(contract, issue_age=attained)
        n_years = projection_years(aged, max_age)
        result = disclosed_shock_repricing(
            aged, state.curve_builder, state.vol,
            mortality.load(attained, target.year, male_weight),
            normals[:, :n_years],
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
                "attained_age": attained,
                "gawa_pct": 100 * gawa,
                "benefit_base": policy.benefit_base,
                "attribution": alpha,
            }
        )
        rows.append(result)
        weights.append(weight)

    return aggregate(rows, weights)


def main() -> None:
    s = session.start()
    mix = SubAccountMix.from_config(s.cfg)
    youngest = min(age for _, age, _, _ in VINTAGES)
    max_years = projection_years(replace(s.contract, issue_age=youngest),
                                int(s.cfg["simulation"]["max_age"]))
    normals = make_normals(s.normals.shape[0], max_years, int(s.cfg["simulation"]["seed"]),
                           bool(s.cfg["simulation"]["antithetic"]))
    target = market.equity_dates(s.panel, None, TARGET_DATE)[-1]

    disclosed = pd.read_csv(paths.DATA_PROCESSED / "disclosed_scaled.csv")
    disclosed = disclosed.query(
        "line_item == 'market_risk_benefits' and as_of == '2025-12-31'"
    ).drop_duplicates(subset=["shock"]).set_index("shock")["impact_pct_of_av"] * 100

    print(f"Reconciling against the {TARGET_DATE} disclosure, "
          f"{s.normals.shape[0]:,} paths per valuation")
    print(f"Guaranteed share of account value used for scaling: {GUARANTEED_SHARE:.2f}")

    grid = []
    for utilisation in (1.0, 0.9, 0.8, 0.7, 0.6):
        for lapse in (0.0, 0.02, 0.04):
            agg = portfolio_at(s, normals, mix, target, utilisation, lapse)
            row = {
                "utilisation": utilisation,
                "lapse_rate": lapse,
                "value_pct_av": agg["value_pct_av"],
                "gwb_over_av": agg["gwb_over_av"],
            }
            for key in ("equity_down_10pct", "equity_up_10pct",
                        "rates_up_100bp", "rates_down_100bp"):
                model = agg[f"model_{key}"] * GUARANTEED_SHARE
                row[f"model_{key}"] = model
                row[f"ratio_{key}"] = model / disclosed[key] if key in disclosed else np.nan
            ratios = [row[f"ratio_{k}"] for k in
                      ("equity_down_10pct", "equity_up_10pct",
                       "rates_up_100bp", "rates_down_100bp")]
            row["mean_ratio"] = float(np.nanmean(ratios))
            row["max_abs_log_ratio"] = float(np.nanmax(np.abs(np.log(ratios))))
            grid.append(row)
            print(f"  utilisation {utilisation:.2f}, lapse {lapse:.2f}: "
                  f"mean ratio {row['mean_ratio']:.2f}")

    frame = pd.DataFrame(grid)
    print("\nFull grid, model over disclosed after scaling for the guaranteed share")
    display = frame[["utilisation", "lapse_rate", "value_pct_av",
                     "ratio_equity_down_10pct", "ratio_equity_up_10pct",
                     "ratio_rates_up_100bp", "ratio_rates_down_100bp", "mean_ratio"]]
    print(display.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    session.write_table(frame, "behaviour_reconciliation", "%.4f")

    best = frame.iloc[frame["max_abs_log_ratio"].idxmin()]
    print(
        f"\nClosest combination: utilisation {best['utilisation']:.2f}, "
        f"lapse {best['lapse_rate']:.2f}, every shock within "
        f"{100 * (np.exp(best['max_abs_log_ratio']) - 1):.0f}% of the disclosed figure"
    )
    print("Disclosed, % of account value:")
    print(disclosed.to_string(float_format=lambda v: f"{v:,.4f}"))


if __name__ == "__main__":
    main()
