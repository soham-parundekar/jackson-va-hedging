"""What the macro put spread buys, measured on the lens it was bought for.

S6 holds a deep out-of-the-money put spread in a fixed size outside the Greek solve, and the
hedging experiments judge it the same way they judge everything else: by how much daily variance
it removes. On that measure it looks like a worse S3 - it costs more and the residual is no
better - which is the wrong verdict from the wrong test. A tail hedge is not bought to flatten
the Greeks week to week. It is bought so that a severe fall does not take the statutory capital
position with it, and that is a different number on a different basis.

So this sweeps the spread's size and its strikes and reports two things against each other: what
it costs over the whole replay window, and how far capital falls through covid with it in place.

Both bases are reported and in covid they coincide, which was not the expectation and is worth
stating rather than hiding. The surrender value floor binds when the guarantee is a net asset,
and a thirty-three per cent fall in the index makes the guarantee a large net liability, so the
floor is slack on every day of that window - 62 per cent of days across the whole decade, none of
them in the crash. The floor is a rally problem. A tail hedge therefore has no separate statutory
story in a crisis, and the case for it has to be made on the economic drawdown it avoids.

The spread is held outside the solve for a reason recorded in strategies.py: handed to a
least-squares fit it exploited the near-collinearity of its two legs, taking fifty units long
against seventy-six short to manufacture gamma at a hundred and thirty times account value in
notional. Size is a decision, not an output.

What this cannot say is whether the capital protection is worth its price, because that needs a
cost of capital and the project has no defensible figure for one. What it gives instead is the
exchange rate: basis points of running cost per point of capital drawdown avoided, so the
judgement is explicit rather than buried.

One limitation decides how far the strike sweep can be read. The worst window the free index
history reaches is covid, where the index bottomed at 0.66 of its starting level. A spread struck
below that never pays in this sample, so the grid's verdict on the far-out strikes is a statement
about the crises available and not about far-out strikes. With 2008 reachable - it is not; the
S&P 500 series on FRED is licensed to the trailing ten years - the 0.70/0.50 pair would have an
event to pay into. The near strikes are the ones this data can speak to.

Usage:  python -m scripts.run_macro_frontier
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from vahedge import paths
from vahedge.capital import statutory
from vahedge.hedge import simulator, strategies
from vahedge.market import scenarios
from vahedge.valuation.convexity import ConvexitySurface, with_nested_gamma

from scripts.run_hedge_experiments import DIVIDEND_YIELD, DURATION_AT_START, build

# Notional of the spread as a share of account value. Zero is S3 with no tail hedge at all and is
# the baseline every other row is read against.
MACRO_SIZES = (0.0, 0.125, 0.25, 0.50)
# Strike pairs, long the first and short the second. The 80/60 pair is what S6 holds; 90/70 is
# nearer the money and pays earlier; 70/50 is further out and cheaper. The width is held at
# twenty points of spot in all three so the payout cap moves with the strikes rather than
# independently of them, which keeps the sweep one-dimensional in moneyness.
MACRO_STRIKE_PAIRS = ((0.90, 0.70), (0.80, 0.60), (0.70, 0.50))
MACRO_TENORS = (2.0,)
STRESS_EPISODE = "covid"


def variant(size: float, strikes: tuple, tenor: float):
    """S3 plus a macro overlay of the given size, or S3 itself when the size is zero."""
    base = strategies.matrix()["S3"]
    if size <= 0.0:
        return replace(base, name="no overlay", overlay=())
    return replace(
        base, name=f"{strikes[0]:.2f}/{strikes[1]:.2f} at {size:.3f}",
        overlay=strategies.macro_leg(size, strikes, tenor),
    )


def run(setup, path, strategy):
    surface = setup["surface"]
    source = (with_nested_gamma(setup["proxy"], surface) if surface is not None else None)
    return simulator.run(
        path, setup["policy"], setup["survival"], setup["deaths"], setup["proxy"], strategy,
        setup["smile"], years_at_start=float(DURATION_AT_START),
        equity_weight=setup["state"].mix.equity_weight, dividend_yield=DIVIDEND_YIELD,
        greeks_override=source,
    )


def drawdown(series: pd.Series, account_value: float) -> float:
    """Worst peak-to-trough fall in a capital series, as a share of the starting account value.

    Peak to trough rather than start to end, because a capital constraint is breached at the
    trough and does not care that the position recovered afterwards.
    """
    running = series.cummax()
    return float((series - running).min() / account_value)


def frontier(setup) -> pd.DataFrame:
    episodes = scenarios.available_episodes(setup["history"])
    if STRESS_EPISODE not in episodes["covered"]:
        raise ValueError(f"{STRESS_EPISODE!r} is not in the replay window")
    stress = setup["history"].window(
        *episodes["covered"][STRESS_EPISODE][:2], label=STRESS_EPISODE
    )
    account_value = float(setup["policy"].account_value[0])

    def measure(size, strikes, tenor) -> dict:
        strategy = variant(size, strikes, tenor)
        whole = run(setup, setup["history"], strategy)
        crisis = run(setup, stress, strategy)
        marked = statutory.statutory_capital(crisis.ledger)
        return {
            "tenor_years": tenor,
            "long_strike": strikes[0], "short_strike": strikes[1],
            "notional_share": size,
            "cost_whole_window_pct": whole.summary["total_cost_pct"],
            "sd_whole_window_pct": whole.summary["pnl_sd_pct"],
            "crisis_economic_total_pct": crisis.summary["total_pnl_pct"],
            "crisis_economic_drawdown_pct": drawdown(
                marked["economic_capital"], account_value),
            "crisis_statutory_drawdown_pct": drawdown(
                marked["statutory_capital"], account_value),
            "crisis_statutory_worst_day_pct": float(
                marked["statutory_pnl"].min() / account_value),
            "share_of_crisis_days_floored": float(marked["floor_binds"].mean()),
        }

    # The no-overlay row does not depend on the strikes, so it runs once rather than once per
    # pair. Its strike columns carry nan rather than a borrowed pair, which would read as a
    # spread that happened to cost nothing.
    rows = [{**measure(0.0, (np.nan, np.nan), MACRO_TENORS[0])}]
    for tenor in MACRO_TENORS:
        for strikes in MACRO_STRIKE_PAIRS:
            for size in MACRO_SIZES:
                if size <= 0.0:
                    continue
                rows.append(measure(size, strikes, tenor))

    table = pd.DataFrame(rows)
    baseline = table[table["notional_share"] == 0.0].iloc[0]
    table["extra_cost_pct"] = table["cost_whole_window_pct"] - baseline["cost_whole_window_pct"]
    # Drawdowns are negative, so a smaller fall is a larger number and the improvement is the
    # variant less the baseline. The first version had this the other way round and read every
    # worsening as protection bought, which made the worst row in the grid the recommended one.
    table["drawdown_avoided_pct"] = (
        table["crisis_statutory_drawdown_pct"] - baseline["crisis_statutory_drawdown_pct"]
    )
    # Basis points of running cost over ten years per point of statutory drawdown avoided. Blank
    # where the overlay avoided nothing, because a ratio against zero is not a price.
    table["bp_of_cost_per_point_avoided"] = np.where(
        table["drawdown_avoided_pct"] > 1e-6,
        1e4 * table["extra_cost_pct"] / (100.0 * table["drawdown_avoided_pct"]),
        np.nan,
    )
    return table


def main() -> None:
    paths.ensure_output_dirs()
    setup = build()
    if setup["surface"] is None:
        print("No curvature surface on disk. Run scripts/run_convexity_surface.py first, or the "
              "option leg is sized off a gamma the proxy validation says is unusable.")
    table = frontier(setup)
    table.to_csv(paths.TABLES / "macro_hedge_frontier.csv", index=False)

    print(f"Macro put spread against statutory capital in {STRESS_EPISODE}, "
          f"{100*table['share_of_crisis_days_floored'].iloc[0]:.0f}% of crisis days floored")
    print("  strikes      size   cost 10y   crisis drawdown: economic  statutory   "
          "avoided   bp per point")
    for _, row in table.iterrows():
        label = ("no overlay" if row["notional_share"] == 0.0
                 else f"{row['long_strike']:.2f}/{row['short_strike']:.2f}")
        price = ("" if not np.isfinite(row["bp_of_cost_per_point_avoided"])
                 else f"{row['bp_of_cost_per_point_avoided']:11.0f}")
        print(f"  {label:<11s} {row['notional_share']:5.3f}  "
              f"{100*row['cost_whole_window_pct']:8.2f}%  "
              f"{100*row['crisis_economic_drawdown_pct']:+20.2f}% "
              f"{100*row['crisis_statutory_drawdown_pct']:+10.2f}% "
              f"{100*row['drawdown_avoided_pct']:+8.2f}%{price}")

    paid = table[np.isfinite(table["bp_of_cost_per_point_avoided"])]
    if paid.empty:
        print("\n  No overlay in the grid reduced the statutory drawdown. On this crisis the "
              "spread is premium with nothing bought, and the strikes are the reason to check "
              "before the size is.")
    else:
        best = paid.loc[paid["bp_of_cost_per_point_avoided"].idxmin()]
        print(f"\n  Cheapest protection in the grid: {best['long_strike']:.2f}/"
              f"{best['short_strike']:.2f} at {best['notional_share']:.3f} of account value, "
              f"{100*best['drawdown_avoided_pct']:.2f} points of statutory drawdown avoided for "
              f"{100*best['extra_cost_pct']:.2f}% of ten years' cost")
    print(f"\nwrote {paths.TABLES / 'macro_hedge_frontier.csv'}")


if __name__ == "__main__":
    main()
