"""What behaviour assumption would reconcile the model with the disclosed sensitivities.

The vintage portfolio matches the sign of every disclosed shock and tracks the year-on-year
decline in sensitivity, but its sensitivities sit well above the disclosed figures by a multiple
that is close to constant across six shocks and four balance-sheet dates. One near-constant
multiple points at a single structural assumption rather than a pile of small errors.

The candidate is policyholder behaviour. The base case draws the full guaranteed amount every
year and never surrenders, which is the benchmark case in the literature precisely because it is
the most expensive one for the insurer. Jackson's own fair value is built on assumed benefit
utilisation, lapse, mortality and withdrawal rates, so its liability reflects contract holders who
draw less than the maximum and some of whom leave.

This sweeps utilisation and the base lapse rate and reports which combinations bring the
portfolio into line. The point is not to fit the disclosure - that would be reverse-engineering a
number rather than modelling a liability - but to establish whether the gap closes inside the
range the literature and the filings support, or whether something else has to be wrong.

Two things the sweep handles outright rather than listing as caveats on a static comparison:

*Lapse is dynamic.* A static rate surrenders contracts at the same pace whatever the guarantee is
worth, which is wrong in exactly the states that matter: real lapse collapses when a guarantee is
deep in the money, and a static assumption therefore flatters the insurer. The damping is on, so
the rate quoted in the sweep is the rate at the money and the effective rate falls from there.

*The guaranteed share is applied explicitly.* Roughly a quarter of Jackson's variable annuity
account value carries no living benefit at all, so the disclosed sensitivity is already divided by
a denominator that includes contracts with nothing to be sensitive about. Scaling the model by
that share is arithmetic rather than an assumption, and it does part of the work no behaviour
assumption should be asked to do.

One bias still runs the other way and is left in rather than corrected: the roll that produces
each vintage's account value today keeps the benchmark behaviour, because what a contract is worth
now is a fact about the past. A book whose holders had been drawing less would have more account
value today and less moneyness, so the sweep understates how far behaviour alone could go.

Usage:  python -m scripts.run_behaviour_reconciliation
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vahedge import paths
from vahedge.market import scenarios

from scripts.run_portfolio_validation import (
    SHOCKS,
    VINTAGES,
    portfolio_at,
    vintage_attributions,
)
from scripts.run_shock_validation import DIVIDEND_YIELD, market_at, shock_row
from scripts.run_valuation import build

TARGET_DATE = "2025-12-31"
# Share of Jackson's variable annuity account value carrying a withdrawal guarantee: GMWB for
# Life 72% plus GMWB 3% at 31 December 2025, FY2025 10-K Item 1.
GUARANTEED_SHARE = 0.75
UTILISATION_GRID = (1.0, 0.9, 0.8, 0.7, 0.6)
LAPSE_GRID = (0.0, 0.02, 0.04)
# Dynamic lapse damping, the same parameters the hedging backtest carries. The quoted rate is the
# rate at the money; above it the rate decays with this exponent and stops at the floor.
LAPSE_BETA = 1.2
LAPSE_FLOOR = 0.01
COMPARED = ("equity_down_10pct", "equity_up_10pct", "rates_up_100bp", "rates_down_100bp")


def disclosed_at(date: str) -> pd.Series:
    frame = pd.read_csv(paths.DATA_PROCESSED / "disclosed_scaled.csv")
    rows = frame.query("line_item == 'market_risk_benefits' and as_of == @date")
    if rows.empty:
        raise ValueError(f"no disclosed market risk benefit sensitivities at {date}")
    return rows.drop_duplicates(subset=["shock"]).set_index("shock")["impact_pct_of_av"] * 100


def sweep(setup, history, panel, disclosed: pd.Series, attributions) -> pd.DataFrame:
    valuer, calibration = setup["valuer"], setup["calibration"]
    rows = []
    for utilisation in UTILISATION_GRID:
        for lapse in LAPSE_GRID:
            behaviour = {"utilisation": utilisation, "lapse_rate": lapse,
                         "lapse_beta": LAPSE_BETA if lapse > 0 else 0.0,
                         "lapse_floor": LAPSE_FLOOR if lapse > 0 else 0.0}
            book, attribution, detail = portfolio_at(setup, history, TARGET_DATE, attributions,
                                                     behaviour=behaviour)
            state = market_at(panel, history, calibration, detail["as_of"].iloc[0])
            shocks = shock_row(valuer, book, state, attribution)
            row = {"utilisation": utilisation, "lapse_rate": lapse,
                   "value_pct_av": shocks["value_pct_av"] * GUARANTEED_SHARE,
                   "gwb_over_av": float(book.total_benefit_base / book.total_account_value)}
            ratios = []
            for shock in SHOCKS:
                scaled = shocks[f"{shock}_pct_av"] * GUARANTEED_SHARE
                row[f"model_{shock}"] = scaled
                if shock in disclosed.index and shock in COMPARED:
                    ratio = scaled / disclosed[shock]
                    row[f"ratio_{shock}"] = ratio
                    ratios.append(ratio)
            row["mean_ratio"] = float(np.mean(ratios))
            # The two shock families separately, because the sweep's main result is that they do
            # not move together: utilisation takes duration out of the guarantee and so collapses
            # the rate sensitivity, while the benefit base is still there and the equity
            # sensitivity barely notices.
            row["equity_ratio"] = float(np.mean(
                [row[f"ratio_{s}"] for s in COMPARED if s.startswith("equity")]))
            row["rate_ratio"] = float(np.mean(
                [row[f"ratio_{s}"] for s in COMPARED if s.startswith("rates")]))
            row["equity_over_rate"] = row["equity_ratio"] / row["rate_ratio"]
            # The widest miss across the four compared shocks, in log space so that two times too
            # big and two times too small count the same. Ranking on the mean would let a
            # combination that overshoots one shock and undershoots another look like a fit.
            row["max_abs_log_ratio"] = float(np.max(np.abs(np.log(np.abs(ratios)))))
            rows.append(row)
            print(f"  utilisation {utilisation:.2f}, lapse {lapse:.2f} at the money: "
                  f"mean ratio {row['mean_ratio']:.2f}, widest miss "
                  f"{100 * (np.exp(row['max_abs_log_ratio']) - 1):.0f}%", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    paths.ensure_output_dirs()
    setup = build(cache_size=5)
    panel = pd.read_csv(paths.FRED_PANEL, comment="#", parse_dates=["date"]).set_index("date")
    history = scenarios.load_history(panel, setup["calibration"].heston,
                                    setup["calibration"].mix, dividend_yield=DIVIDEND_YIELD)
    disclosed = disclosed_at(TARGET_DATE)

    print(f"Reconciling against the {TARGET_DATE} disclosure, {setup['valuer'].n_paths:,} paths")
    print(f"  {len(VINTAGES)} vintages, model scaled by the {GUARANTEED_SHARE:.0%} of account "
          f"value that carries a withdrawal guarantee")
    attributions = vintage_attributions(setup["valuer"], panel, history,
                                        setup["calibration"], setup["terms"])

    frame = sweep(setup, history, panel, disclosed, attributions)
    frame.to_csv(paths.TABLES / "behaviour_reconciliation.csv", index=False, float_format="%.4f")

    print("\nModel over disclosed, after scaling for the guaranteed share")
    print("  util  lapse   value  GWB/AV   eq-10%   eq+10%   +100bp   -100bp    mean")
    for _, row in frame.iterrows():
        print(f"  {row['utilisation']:.2f} {row['lapse_rate']:6.2f} "
              f"{row['value_pct_av']:7.2f} {row['gwb_over_av']:7.3f} "
              f"{row['ratio_equity_down_10pct']:8.2f} {row['ratio_equity_up_10pct']:8.2f} "
              f"{row['ratio_rates_up_100bp']:8.2f} {row['ratio_rates_down_100bp']:8.2f} "
              f"{row['mean_ratio']:7.2f}")

    best = frame.loc[frame["max_abs_log_ratio"].idxmin()]
    baseline = frame[(frame["utilisation"] == 1.0) & (frame["lapse_rate"] == 0.0)].iloc[0]
    print(f"\nClosest combination: utilisation {best['utilisation']:.2f} with lapse "
          f"{best['lapse_rate']:.2f} at the money, every compared shock within "
          f"{100 * (np.exp(best['max_abs_log_ratio']) - 1):.0f}% of the disclosed figure, "
          f"against {baseline['mean_ratio']:.2f} times the disclosure on the static benchmark")

    # The result the sweep exists to produce, and it is not the closest combination.
    print(f"\n  But the two shock families do not move together. Across the grid the equity "
          f"multiple runs {frame['equity_ratio'].max():.2f} down to "
          f"{frame['equity_ratio'].min():.2f} while the rate multiple runs "
          f"{frame['rate_ratio'].max():.2f} down to {frame['rate_ratio'].min():.2f}, so their "
          f"ratio goes from {baseline['equity_over_rate']:.2f} on the static benchmark to "
          f"{frame['equity_over_rate'].max():.2f} at the far corner.")
    print("  Drawing less takes duration out of the guarantee - fewer paths exhaust, so it is "
          "less of a long-dated annuity - and the rate sensitivity collapses with it. The "
          "benefit base is still there whatever the owner draws, so the equity sensitivity "
          "barely moves. No single behaviour assumption closes both: by the time utilisation "
          f"is low enough to match the rate figure the model is at "
          f"{frame.loc[frame['rate_ratio'].sub(1.0).abs().idxmin(), 'equity_ratio']:.2f} times "
          "the disclosed equity figure.")
    print("  What is left for the equity gap is moneyness rather than behaviour: the shock "
          f"locator puts the disclosed book at a benefit base to account ratio near 0.85, and "
          f"this portfolio sits at {baseline['gwb_over_av']:.3f}.")
    print("\nDisclosed, % of account value")
    for shock in COMPARED:
        if shock in disclosed.index:
            print(f"  {shock:<18s} {disclosed[shock]:+7.4f}")
    print(f"\nwrote {paths.TABLES / 'behaviour_reconciliation.csv'}")


if __name__ == "__main__":
    main()
