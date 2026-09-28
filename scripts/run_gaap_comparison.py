"""Why reported earnings still move when the economic hedge is doing its job.

Jackson's Item 7A says it directly: "We do not directly use hedging to offset the
movement in our U.S. GAAP liabilities as market conditions change from period to period,
which has resulted, and may continue to result, in U.S. GAAP net income volatility." This
script measures that gap on the same hedge rather than describing it.

The hedge positions come from the economic basis: best-estimate annuitant mortality,
discounted on the Treasury curve. The reported liability is the same contract on the
reporting basis: the margin-loaded annuity table, discounted on the Treasury curve plus
Jackson's own non-performance spread. Under the market risk benefit rules the movement
attributable to own non-performance risk is reported in other comprehensive income, so it
is separated out and the rest is what reaches net income.

Usage:  python -m scripts.run_gaap_comparison
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gmwb import accounting, figures, hedging, paths, session

LEDGER = paths.DATA_PROCESSED / "hedge_ledger_book_mix.csv"


def load_ledger() -> pd.DataFrame:
    if not LEDGER.exists():
        raise FileNotFoundError(
            f"{LEDGER.name} is missing. Run scripts.run_hedge_backtest first."
        )
    return pd.read_csv(LEDGER, parse_dates=["date"]).set_index("date")


def basis_gap(reported: pd.DataFrame) -> pd.DataFrame:
    """How far the reporting basis sits from the economic basis, and why."""
    frame = pd.DataFrame(
        {
            "economic_value": reported["rider_value"],
            "reporting_basis_value": reported["mrb_reporting_basis"],
            "with_own_credit": reported["mrb_with_own_credit"],
            "own_credit_adjustment": reported["own_credit_adjustment"],
            "own_credit_spread_pct": 100 * reported["own_credit_spread"],
        }
    )
    frame["margin_effect"] = frame["reporting_basis_value"] - frame["economic_value"]
    return frame


def main() -> None:
    s = session.start()
    ledger = load_ledger()
    print(f"Loaded {len(ledger)} weekly observations from {LEDGER.name}")

    # The hedge sized on the economic basis, delta and rho only, which is what Jackson
    # describes its core dynamic programme as covering.
    composed = hedging.compose(ledger, ("equity", "rates"))
    reported = accounting.reported_earnings(composed, s.fee_attribution)

    print("\nMeasurement bases")
    gap = basis_gap(reported)
    print(gap.describe().loc[["mean", "50%", "min", "max"]].to_string(
        float_format=lambda v: f"{v:,.1f}"))
    session.write_table(gap.reset_index(), "accounting_basis_gap", "%.2f")

    summary = accounting.summarise(reported)
    print("\nVolatility of weekly outcomes on one hedge")
    rows = [
        ("unhedged guarantee", summary["unhedged_std"]),
        ("economic, hedged", summary["economic_hedged_std"]),
        ("reported net income", summary["reported_net_income_std"]),
        ("reported comprehensive income", summary["reported_comprehensive_std"]),
        ("the OCI piece on its own", summary["oci_std"]),
    ]
    table = pd.DataFrame(rows, columns=["measure", "weekly_std"])
    table["annualised_std"] = table["weekly_std"] * np.sqrt(52.0)
    table["variance_ratio_vs_unhedged"] = (
        table["weekly_std"] ** 2 / summary["unhedged_std"] ** 2
    )
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    session.write_table(table, "accounting_volatility", "%.4f")

    print(
        f"\nThe hedge removes {100 * (1 - summary['economic_variance_ratio']):.1f}% of the "
        f"variance of the economic liability and "
        f"{100 * (1 - summary['reported_variance_ratio']):.1f}% of the variance that reaches "
        f"net income."
    )
    print(
        f"Reported net income is {summary['reported_over_economic_std']:.2f} times as "
        f"volatile as the economic outcome under the same hedge."
    )
    print(
        f"The own-credit adjustment averages {summary['mean_own_credit_adjustment']:,.0f} "
        f"and reaches {summary['max_own_credit_adjustment']:,.0f} at its largest, all of it "
        f"outside net income."
    )
    print(
        f"The margin loading in the reporting basis adds "
        f"{gap['margin_effect'].mean():,.0f} on average to the liability."
    )

    print("\nPeriods where the economic hedge worked and reported earnings still moved")
    worst = accounting.worst_periods(reported, n=8)
    print(worst.to_string(float_format=lambda v: f"{v:,.1f}"))
    session.write_table(worst.reset_index(), "accounting_worst_periods", "%.2f")

    yearly = reported.groupby(reported.index.year).agg(
        weeks=("reported_net_income", "size"),
        economic_hedged=("hedged_pnl", "sum"),
        reported_net_income=("reported_net_income", "sum"),
        oci=("oci_pnl", "sum"),
        economic_std=("hedged_pnl", "std"),
        reported_std=("reported_net_income", "std"),
    )
    yearly["reported_over_economic_std"] = yearly["reported_std"] / yearly["economic_std"]
    print("\nBy calendar year")
    print(yearly.to_string(float_format=lambda v: f"{v:,.2f}"))
    session.write_table(yearly.reset_index(names="year"), "accounting_by_year", "%.2f")

    print("\nFigure")
    print("  " + figures.economic_versus_reported(reported))


if __name__ == "__main__":
    main()
