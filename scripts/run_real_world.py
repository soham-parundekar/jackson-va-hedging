"""E2. The hedging result against the orderings of the decade that did not happen.

Every hedging number in this project so far comes from one path: the 2,493 trading days between
September 2016 and September 2026, in the order they arrived. That is one observation. It is a
real one, which is its whole value, and it is also a decade in which the index compounded at
15.4 per cent a year and the two bad stretches were short. A delta-and-rho hedge that removes
91 per cent of the daily variation in that decade may be removing 91 per cent of the variation
in a market that happened not to test it.

So the days are resampled. A stationary bootstrap with an 11-day mean block, set from the
integrated autocorrelation of squared returns rather than by eye, reorders the decade's equity
days while keeping each one paired with its own volatility state. The rate path, the curve and
the credit spread are not resampled - ``scenarios.resample`` explains at length why a reordered
level is not a rate scenario - so every path faces the decade of rates that actually happened
and the only thing varying is the equity sequence. The joint equity-and-rates tail is therefore
out of scope here and belongs to the crisis replays, which keep every day whole.

Two arms, on the same draws so the comparison is paired:

**As drawn.** The ordering changes, the average does not. This answers the narrow question: did
the hedging conclusion depend on the particular sequence, or only on the decade's statistics?

**Re-centred.** The same orderings with the equity drift reset to the window's mean financing
rate plus the 4 per cent equity risk premium the statutory work uses as its base case. This
answers the question that matters more, because a guarantee's whole exposure depends on where
the account sits relative to the benefit base: in a 15 per cent world the rider drifts out of
the money and there is progressively less to hedge, while at 6 per cent it stays near the money
for a decade. If the hedge looks worse here, the 91 per cent was partly a gift from the market.

Three things come out that a single path cannot give:

1. The *distribution* of residual volatility, and where the realised decade sits inside it.
2. Whether the ranking survives. On the realised path the put leg bought about 4 per cent of
   residual standard deviation for 13.7 per cent of account value in cost. Four per cent on one
   path is not a finding; the share of reorderings on which it holds is.
3. How often the proxy is asked for a state it was never fitted at. The realised path already
   extrapolates on 36 per cent of rebalances, and a resampled path wanders further. Reported
   with the result rather than checked once, because a conclusion drawn from mostly extrapolated
   Greeks is not a conclusion.

Usage:  python -m scripts.run_real_world
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vahedge import paths
from vahedge.hedge import simulator, strategies
from vahedge.market import scenarios

from scripts.run_hedge_experiments import build, _run
from scripts.run_statutory import BASE_PREMIUM

# Thirty orderings on each arm, which supports a median and a tenth-to-ninetieth range and
# nothing finer; a 5th percentile would be the second-smallest draw and would read as a tail it
# has no claim to. The arms share their draws, so the drift comparison - the main result here -
# is thirty paired differences rather than two independent samples, which is where the precision
# in this experiment actually comes from. The cost is the constraint: one ten-year run of one
# strategy takes six to twelve seconds and the grid is four strategies on two arms.
N_PATHS = 30
SEED = 20260202
# S5 and S6 are left out deliberately. The question here is how much variation a hedge removes
# and whether the ranking holds, which S0 to S3 answer by adding one instrument class at a time;
# a partial ratio and a macro overlay are scalings of that answer and their own experiments are
# elsewhere.
STRATEGIES = ("S0", "S1", "S2", "S3")
HEDGED = STRATEGIES[1:]
QUANTILES = (0.10, 0.50, 0.90)


def drift_target(history) -> float:
    """The re-centred arm's expected equity return, as a continuously compounded annual rate.

    Financing plus a risk premium rather than a round number, and the same 4 per cent premium
    the statutory requirement is reported at, so the two pieces of work are not quietly assuming
    different worlds. The financing leg is the window's own mean overnight rate, which is what
    the paths will actually earn on cash.
    """
    return float(BASE_PREMIUM + np.mean(history.cash_rate))


def one_path(setup, scenario, matrix) -> list:
    account_value = float(setup["policy"].account_value[0])
    equity_weight = setup["state"].mix.equity_weight
    rows = []
    for key in STRATEGIES:
        summary = simulator.summarise(
            _run(setup, scenario, matrix[key]).ledger, account_value, equity_weight
        )
        rows.append({
            "strategy": key,
            "pnl_sd_pct": summary["pnl_sd_pct"],
            "total_pnl_pct": summary["total_pnl_pct"],
            "worst_day_pct": summary["worst_day_pct"],
            "worst_drawdown_pct": summary["worst_drawdown_pct"],
            "total_cost_pct": summary["total_cost_pct"],
            "outside_design_share": summary["outside_design_share"],
            "rebalances": summary["rebalances"],
            "index_end": float(scenario.index[-1]),
            "fund_end": float(scenario.fund[-1]),
            "liability_end_pct": summary["liability_end_pct"],
        })
    return rows


def variance_removed(frame: pd.DataFrame) -> pd.DataFrame:
    """Share of daily P&L variance each strategy removed, path by path.

    Paired against that path's own unhedged run rather than against an average, because the
    paths differ by a factor of five in how much variance there is to remove and a ratio of
    means would be dominated by the worst of them. The worst day is paired the same way.
    """
    unhedged = frame[frame["strategy"] == "S0"].set_index(["arm", "path"])
    hedged = frame[frame["strategy"] != "S0"].copy()
    key = hedged.set_index(["arm", "path"]).index
    baseline_sd = np.asarray(key.map(unhedged["pnl_sd_pct"]), dtype=float)
    baseline_worst = np.asarray(key.map(unhedged["worst_day_pct"]), dtype=float)
    hedged["variance_removed"] = 1.0 - (hedged["pnl_sd_pct"].to_numpy() / baseline_sd) ** 2
    hedged["worst_day_improvement_pct"] = hedged["worst_day_pct"].to_numpy() - baseline_worst
    return hedged


def removed_on_realised(realised: pd.DataFrame, key: str) -> float:
    by_key = realised.set_index("strategy")
    return float(1.0 - (by_key.loc[key, "pnl_sd_pct"] / by_key.loc["S0", "pnl_sd_pct"]) ** 2)


def summarise(removed: pd.DataFrame, realised: pd.DataFrame, sample: str) -> pd.DataFrame:
    """One row per arm and strategy: the distribution, and where the realised decade sits."""
    rows = []
    for (arm, key), block in removed.groupby(["arm", "strategy"], sort=False):
        low, median, high = np.quantile(block["variance_removed"], QUANTILES)
        realised_removed = removed_on_realised(realised, key)
        rows.append({
            "sample": sample, "arm": arm, "strategy": key, "paths": int(block.shape[0]),
            "variance_removed_p10": low,
            "variance_removed_median": median,
            "variance_removed_p90": high,
            "realised_variance_removed": realised_removed,
            # Where the one decade that happened falls in the bootstrap distribution. Near the
            # top means the realised backtest flattered the hedge.
            "realised_percentile": float(
                np.mean(block["variance_removed"].to_numpy() <= realised_removed)
            ),
            "residual_sd_median_pct": float(block["pnl_sd_pct"].median()),
            "residual_sd_p90_pct": float(np.quantile(block["pnl_sd_pct"], 0.90)),
            "cost_median_pct": float(block["total_cost_pct"].median()),
            "worst_day_median_pct": float(block["worst_day_pct"].median()),
            "worst_day_p10_pct": float(np.quantile(block["worst_day_pct"], 0.10)),
            # Variance is a whole-sample measure and a guarantee is a tail problem, so the two
            # are reported side by side: a hedge can take out most of the variance and leave the
            # single worst day where it was, which is what happens on the paths that end furthest
            # out of the money.
            "worst_day_improved_share": float(np.mean(block["worst_day_improvement_pct"] > 0.0)),
            "total_pnl_median_pct": float(block["total_pnl_pct"].median()),
            "outside_design_median": float(block["outside_design_share"].median()),
            "beat_unhedged_share": float(np.mean(block["variance_removed"] > 0.0)),
        })
    return pd.DataFrame(rows)


def ranking_stability(frame: pd.DataFrame, realised: pd.DataFrame) -> pd.DataFrame:
    """How often the richer hedge's edge over the simpler one survives a reordering.

    Pairs are read richer first, so a share near one says adding that instrument class tightened
    the residual on nearly every ordering. The cost gap sits beside it, because on the pairs
    involving S3 the edge is a few hundredths of a per cent of residual standard deviation
    against a double-digit option premium, and only the two together say whether it is worth it.
    """
    wide = frame.pivot_table(index=["arm", "path"], columns="strategy",
                             values=["pnl_sd_pct", "total_cost_pct"])
    realised_by_key = realised.set_index("strategy")
    pairs = [(HEDGED[i], HEDGED[j]) for i in range(len(HEDGED)) for j in range(i)]
    rows = []
    for arm, block in wide.groupby(level="arm", sort=False):
        for richer, simpler in pairs:
            edge = block[("pnl_sd_pct", simpler)] - block[("pnl_sd_pct", richer)]
            gap = block[("total_cost_pct", richer)] - block[("total_cost_pct", simpler)]
            rows.append({
                "arm": arm, "richer": richer, "simpler": simpler,
                "share_tighter": float((edge > 0.0).mean()),
                "median_sd_edge_pct": float(edge.median()),
                "median_cost_gap_pct": float(gap.median()),
                "realised_sd_edge_pct": float(realised_by_key.loc[simpler, "pnl_sd_pct"]
                                              - realised_by_key.loc[richer, "pnl_sd_pct"]),
                "realised_cost_gap_pct": float(realised_by_key.loc[richer, "total_cost_pct"]
                                               - realised_by_key.loc[simpler, "total_cost_pct"]),
            })
    return pd.DataFrame(rows)


def run_every_path() -> pd.DataFrame:
    """Every run this experiment makes, the realised path included, as one table.

    The realised path is a row of the same table rather than a separate object, under the arm
    name "realised", because everything downstream compares against it and a comparison that
    has to join two files is a comparison that can be done against the wrong one. It also means
    the reporting half can be rerun off the saved table without repeating an hour of
    simulation.
    """
    setup = build()
    if setup["surface"] is None:
        print("No curvature surface on disk; run scripts/run_convexity_surface.py first.")
    history, mix = setup["history"], setup["state"].mix
    matrix = strategies.matrix()

    rows = [{"arm": "realised", "path": -1, **row}
            for row in one_path(setup, history, matrix)]
    block = scenarios.block_length(np.diff(np.log(history.index)))
    drawn = scenarios.stationary_bootstrap(
        history, n_days=len(history) - 1, n_paths=N_PATHS, seed=SEED, mean_block=block
    )
    target = drift_target(history)
    sample_drift = float(np.mean(np.diff(np.log(history.index)))
                         * (len(history) - 1) / history.year_fraction[-1])
    print(f"{N_PATHS} paths of {len(history)} days, {block:.2f}-day mean block\n"
          f"  as drawn   expected equity drift {sample_drift:.2%} a year, the window's own\n"
          f"  re-centred expected equity drift {target:.2%} a year "
          f"({np.mean(history.cash_rate):.2%} financing plus {BASE_PREMIUM:.0%})", flush=True)

    for arm, annual_drift in (("as drawn", None), ("re-centred", target)):
        for index in range(N_PATHS):
            scenario = scenarios.resample(
                history, drawn[index], mix, label=f"{arm} {index}", annual_drift=annual_drift
            )
            for row in one_path(setup, scenario, matrix):
                rows.append({"arm": arm, "path": index, **row})
            if (index + 1) % 10 == 0:
                print(f"  {arm}: {index + 1} of {N_PATHS}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(paths.TABLES / "real_world_paths.csv", index=False)
    return frame


def report(frame: pd.DataFrame) -> None:
    """Everything the experiment says, from the table of runs and nothing else."""
    realised = frame[frame["arm"] == "realised"]
    frame = frame[frame["arm"] != "realised"]
    removed = variance_removed(frame)
    realised_outside = float(realised["outside_design_share"].iloc[0])
    # The second sample is a condition rather than a selection: the rest of the project already
    # lives with the realised path's 36 per cent extrapolation share, and a path that exceeds it
    # is measuring the regression's behaviour outside its fit as much as the hedge's.
    comparable = removed[removed["outside_design_share"] <= realised_outside]
    table = pd.concat([
        summarise(removed, realised, "all"),
        summarise(comparable, realised, "comparable"),
    ], ignore_index=True)
    stability = ranking_stability(frame, realised)
    table.to_csv(paths.TABLES / "real_world.csv", index=False)
    stability.to_csv(paths.TABLES / "real_world_ranking.csv", index=False)

    print(f"\nResidual and variance removed, % of account value, {N_PATHS} reorderings.")
    print(f"  \"comparable\" keeps only the paths that extrapolate no more than the realised "
          f"path's {realised_outside:.0%}.")
    print("                                variance removed          residual sd     cost"
          "    worst day")
    print("  sample     arm         strat  n   p10  median   p90  real   median    p90"
          "  median  median    p10")
    for _, row in table.iterrows():
        print(f"  {row['sample']:<10s} {row['arm']:<11s} {row['strategy']:<4s} "
              f"{row['paths']:3d} {100*row['variance_removed_p10']:5.1f} "
              f"{100*row['variance_removed_median']:6.1f} {100*row['variance_removed_p90']:5.1f} "
              f"{100*row['realised_variance_removed']:5.1f} "
              f"{100*row['residual_sd_median_pct']:7.3f} {100*row['residual_sd_p90_pct']:6.3f} "
              f"{100*row['cost_median_pct']:7.2f} {100*row['worst_day_median_pct']:7.2f} "
              f"{100*row['worst_day_p10_pct']:6.2f}")

    print("\nWhere the decade that happened sits in the distribution")
    for _, row in table[table["sample"] == "all"].iterrows():
        print(f"  {row['arm']:<11s} {row['strategy']:<4s} "
              f"realised {100*row['realised_variance_removed']:.1f}% removed, the "
              f"{row['realised_percentile']:.0%} percentile of the reorderings; the hedge took "
              f"out variance on {row['beat_unhedged_share']:.0%} of paths and improved the worst "
              f"day on {row['worst_day_improved_share']:.0%}")

    print("\nDoes the ranking survive a reordering?")
    for _, row in stability.iterrows():
        print(f"  {row['arm']:<11s} {row['richer']} tighter than {row['simpler']} on "
              f"{row['share_tighter']:.0%} of paths, by a median "
              f"{100*row['median_sd_edge_pct']:.4f}% of residual against "
              f"{100*row['realised_sd_edge_pct']:.4f}% realised, for "
              f"{100*row['median_cost_gap_pct']:+.2f}% more cost")

    print(f"\nExtrapolation: the realised path asks the proxy for a state outside its design on "
          f"{realised_outside:.0%} of rebalances.")
    for arm, block in frame.groupby("arm", sort=False):
        share = block["outside_design_share"]
        print(f"  {arm:<11s} median {share.median():.0%}, 90th percentile "
              f"{np.quantile(share, 0.90):.0%}, and {np.mean(share <= realised_outside):.0%} of "
              f"paths at or below the realised path's share")
    print(f"\nwrote {paths.TABLES / 'real_world.csv'}, the per-path detail and the ranking table")


def main() -> None:
    paths.ensure_output_dirs()
    report(run_every_path())


if __name__ == "__main__":
    main()
