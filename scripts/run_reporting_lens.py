"""Economic against reported earnings over the replay window, and the OCI split.

The hedging experiments say a delta-and-rho hedge removes 91 per cent of the daily variation in
economic net worth. Jackson says in Item 7A that it does not hedge its US GAAP liabilities and
that this has produced net income volatility. Both can be true, and this measures the gap: the
same hedge, the same decade, marked on three bases.

What comes out, strategy by strategy:

The **risk margin**, the margin-loaded mortality table against the best-estimate one. Not a
loading this project invented - the NAIC-adopted annuity table ships in both versions and Note 6
says the fair value uses best estimate plus margins, so the difference is a margin someone else
calibrated. It moves with the market, so a hedge sized on the economic basis is the wrong size
for it, and that mismatch reaches net income.

The **own-credit adjustment**, the reported liability discounted at Treasury plus the insurer's
own spread against Treasury alone. Under ASU 2018-12 its movement goes to other comprehensive
income rather than through net income. Across the replay window the spread ran 1.4 to 4.3 per
cent and through covid alone it went 2.1 to 4.3, so this line is large, and it is the one line a
hedge has no reason to target.

The **ratio of reported to economic volatility**, which is the number the exercise exists for. A
value above one says the programme removed economic variation and left more reported variation
than it took out.

Four regression fits rather than one, sharing a single simulation: the paths do not depend on the
mortality table or on the discount spread, so what repeats is the projection and the regression.
Expect eight to ten minutes before the first strategy runs.

Usage:  python -m scripts.run_reporting_lens
"""

from __future__ import annotations

import pandas as pd

from vahedge import paths
from vahedge.capital import reporting
from vahedge.hedge import simulator, strategies
from vahedge.valuation.convexity import with_nested_gamma

from scripts.run_hedge_experiments import (
    DIVIDEND_YIELD,
    DURATION_AT_START,
    STRATEGY_ORDER,
    build,
)


def run(setup, strategy):
    surface = setup["surface"]
    source = with_nested_gamma(setup["proxy"], surface) if surface is not None else None
    return simulator.run(
        setup["history"], setup["policy"], setup["survival"], setup["deaths"], setup["proxy"],
        strategy, setup["smile"], years_at_start=float(DURATION_AT_START),
        equity_weight=setup["state"].mix.equity_weight, dividend_yield=DIVIDEND_YIELD,
        greeks_override=source,
    )


def lens(setup, bases) -> tuple:
    matrix = strategies.matrix()
    account_value = float(setup["policy"].account_value[0])
    rows, detail = [], None
    for key in STRATEGY_ORDER:
        marked = reporting.mark(run(setup, matrix[key]).ledger, setup["history"], bases)
        rows.append({"strategy": key,
                     **reporting.summarise(marked, account_value)})
        if key == "S2":
            # The one strategy worth keeping day by day: Jackson's own book is mostly delta and
            # rho by notional, so S2 is the closest thing in the matrix to what the disclosure
            # describes.
            detail = marked.assign(strategy=key)
    return pd.DataFrame(rows), detail


def main() -> None:
    paths.ensure_output_dirs()
    setup = build()
    if setup["surface"] is None:
        print("No curvature surface on disk; run scripts/run_convexity_surface.py first.")

    bases = reporting.fit_bases(
        setup["at_issue"], setup["state"], setup["market_paths"],
    )
    table, detail = lens(setup, bases)
    table.to_csv(paths.TABLES / "reporting_lens.csv", index=False)
    detail.to_csv(paths.TABLES / "reporting_lens_daily_s2.csv")

    print("Economic against reported earnings, whole replay window, % of account value")
    print("                 totals                        daily standard deviation")
    print("  strat  economic  net income    OCI    economic  net income  ratio    OCI   compreh")
    for _, row in table.iterrows():
        print(f"  {row['strategy']:<5s} {100*row['economic_total_pct']:+9.2f} "
              f"{100*row['net_income_total_pct']:+11.2f} {100*row['oci_total_pct']:+7.2f}  "
              f"{100*row['economic_sd_pct']:9.3f} {100*row['net_income_sd_pct']:10.3f} "
              f"{row['net_income_sd_multiple']:6.2f} {100*row['oci_sd_pct']:7.3f} "
              f"{100*row['comprehensive_sd_pct']:8.3f}")

    first = table.iloc[0]
    print(f"\n  risk margin averages {100*first['mean_risk_margin_pct']:+.2f}% of account value "
          f"and the own-credit adjustment {100*first['mean_own_credit_pct']:+.2f}%; both are "
          f"properties of the block, so they are the same on every row")
    hedged = table[table["strategy"] != "S0"]
    best = hedged.loc[hedged["net_income_sd_multiple"].idxmin()]
    worst = hedged.loc[hedged["net_income_sd_multiple"].idxmax()]
    print(f"  reported-to-economic volatility ratio runs {best['net_income_sd_multiple']:.2f} "
          f"({best['strategy']}) to {worst['net_income_sd_multiple']:.2f} ({worst['strategy']}) "
          f"across the hedged strategies, against {first['net_income_sd_multiple']:.2f} unhedged")

    # The two comparisons the table is actually for, and neither is the ratio above.
    tightest = hedged.loc[hedged["net_income_sd_pct"].idxmin()]
    print(f"  the OCI line, which no hedge targets, has a daily standard deviation of "
          f"{100*first['oci_sd_pct']:.3f}% - larger than the whole of {tightest['strategy']}'s "
          f"hedged net income at {100*tightest['net_income_sd_pct']:.3f}%")
    offset = hedged[hedged["comprehensive_sd_pct"] < hedged["net_income_sd_pct"]]
    if not offset.empty:
        print(f"  and comprehensive income is steadier than net income on "
              f"{len(offset)} of {len(hedged)} hedged strategies, so own credit moves against "
              f"the guarantee rather than with it: spreads widen as markets fall, the reported "
              f"liability shrinks, and the rule books that gain outside net income")
    print(f"\nwrote {paths.TABLES / 'reporting_lens.csv'} and the S2 daily detail")


if __name__ == "__main__":
    main()
