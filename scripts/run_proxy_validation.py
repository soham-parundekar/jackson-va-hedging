"""How far the regression proxy can be trusted, measured against nested simulation.

The hedge backtest needs the liability's value and its Greeks at every rebalance date on every
path, and computing those by simulation at each node is the nested calculation the proxy exists
to avoid. So the proxy's error has to be established before any hedging result built on it
means anything. The standard it is measured against is the expensive answer: value the contract
at a sample of the states the paths actually visit, with a full inner simulation at each one.

Two things are reported for every year, and both matter. The error over all nodes is what a
backtest would suffer if it wandered anywhere; the error over the nodes the fit has data behind
is what it will actually suffer, because a state the paths effectively never reached is flagged
when the proxy is asked for it. Quoting only the first understates the proxy; quoting only the
second hides where it should not be trusted.

Nodes are chosen by stratifying on moneyness rather than at random. The states that decide a
hedge are the ones near the boundary of the guarantee biting, and a random sample of a
lognormal puts almost nothing there.

Usage:  python -m scripts.run_proxy_validation
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vahedge import paths
from vahedge.liability import cohorts, gmwb, mortality
from vahedge.liability import terms as terms_module
from vahedge.market import state as market_state
from vahedge.market.simulate import simulate
from vahedge.valuation import lsmc, nested
from vahedge.valuation.engine import MarketState, Valuer

# The representative contract for the hedging work: a 70-year-old on the Core option with a
# five-year deferral, which is the weighted-average attained age Jackson discloses and a
# deferral long enough that the bonus and the step-up both matter.
ISSUE_AGE = 70
DEFERRAL_YEARS = 5
MAX_AGE = 105
PREMIUM = 100.0

FIT_PATHS = 40_000
FIT_SEED = 20251231
INNER_PATHS = 5_000
INNER_SEED = 4242
NODES = 50
# Years to test. Spread across the contract rather than clustered, because the fit's problem
# changes with the horizon: early on almost nothing is exhausted and the low-moneyness end has
# no data; late on almost everything is, and the live end has none.
TEST_YEARS = (1, 2, 5, 9, 14, 20, 25, 30)
# The delta check is the second half of the proxy's job and the half a level comparison does
# not cover: a fit whose value is right and whose slope is wrong hedges badly while valuing
# correctly. It is run at every tested year because it costs almost nothing - the bumped
# valuations reuse each node's own draws - but on fewer nodes, since what matters is whether
# the slope is right across the moneyness range rather than resolving it point by point.
DELTA_NODES = 20
EQUITY_BUMP = 0.02


def build():
    calibration = market_state.load()
    state = MarketState.from_calibration(calibration)
    core = terms_module.load()[("flex_gmwb", "single", "core")]
    book = cohorts.single_contract(
        core, issue_age=ISSUE_AGE, base_contract_charge=0.0131, fund_expense=0.0095,
        premium=PREMIUM, deferral_years=DEFERRAL_YEARS, max_age=MAX_AGE,
        lapse_rate=0.04, lapse_beta=1.2, lapse_floor=0.01,
    )
    years = int(book.projection_years.max())
    survival, deaths = mortality.load("basic").rates(book.attained_age, state.valuation_year,
                                                     years, 0.5)
    market_paths = simulate(state.heston, state.hull_white(), state.correlations, state.mix,
                            n_years=years, n_paths=FIT_PATHS, seed=FIT_SEED)
    projection = gmwb.project(book, market_paths, survival, deaths,
                              equity_weight=state.mix.equity_weight, record=True)
    return book, state, projection


def main() -> None:
    paths.ensure_output_dirs()
    book, state, projection = build()
    proxy = lsmc.fit(projection)
    proxy.diagnostics.assign(
        rmse_pct_of_premium=proxy.diagnostics["rmse"] / PREMIUM,
        exhausted_share=proxy.diagnostics["n_exhausted"] / FIT_PATHS,
    ).to_csv(paths.TABLES / "proxy_fit_diagnostics.csv", index=False)

    inner = Valuer(mortality.load("basic"), n_paths=INNER_PATHS, seed=INNER_SEED, cache_size=1)
    rows, delta_rows = [], []
    for year in TEST_YEARS:
        if year not in proxy.fits:
            continue
        truth = nested.gold_standard(inner, book, state, projection.recorded,
                                     year=year, n_nodes=NODES)
        comparison = nested.compare(proxy, truth)
        summary = nested.summarise(comparison, premium=PREMIUM)
        summary["year"] = year
        summary["mean_nested_value"] = float(truth["nested_value"].mean())
        summary["exhausted_share"] = proxy.fits[year].n_exhausted / FIT_PATHS
        rows.append(summary)

        bumped = nested.compare(proxy, nested.gold_standard(
            inner, book, state, projection.recorded, year=year,
            n_nodes=DELTA_NODES, equity_bump=EQUITY_BUMP,
        ))
        inside = bumped[~bumped["outside_design_range"]]
        summary["delta_rmse_share_of_premium_in_range"] = float(
            np.sqrt((inside["delta_error"] ** 2).mean()) / PREMIUM
        ) if not inside.empty else np.nan
        summary["gamma_rmse_share_of_premium_in_range"] = float(
            np.sqrt((inside["gamma_error"] ** 2).mean()) / PREMIUM
        ) if not inside.empty else np.nan
        summary["mean_abs_nested_delta"] = float(inside["nested_delta"].abs().mean())
        summary["mean_abs_nested_gamma"] = float(inside["nested_gamma"].abs().mean())
        delta_rows.append(bumped.assign(year=year))
        print(
            f"year {year:2d}  all rmse {100*summary['rmse_share_of_premium']:5.2f}% of premium"
            f"  R2 {summary['r_squared']:.4f}"
            f" | in range {summary['nodes_in_range']:2d} nodes"
            f"  rmse {100*summary['rmse_share_of_premium_in_range']:5.2f}%"
            f"  R2 {summary['r_squared_in_range']:.4f}"
            f" | flagged {100*summary['share_outside_design_range']:3.0f}%"
            f"  delta rmse {100*summary['delta_rmse_share_of_premium_in_range']:5.2f}%"
            f" of {summary['mean_abs_nested_delta']:5.1f}"
            f"  gamma rmse {100*summary['gamma_rmse_share_of_premium_in_range']:6.2f}%"
            f" of {summary['mean_abs_nested_gamma']:6.1f}"
            f"  inner se {summary['mean_nested_std_error']:.3f}",
            flush=True,
        )

    table = pd.DataFrame(rows)[[
        "year", "nodes", "r_squared", "rmse", "rmse_share_of_premium", "worst_share_of_premium",
        "nodes_in_range", "r_squared_in_range", "rmse_share_of_premium_in_range",
        "worst_share_of_premium_in_range", "delta_rmse_share_of_premium_in_range",
        "gamma_rmse_share_of_premium_in_range", "mean_abs_nested_delta",
        "mean_abs_nested_gamma", "share_outside_design_range",
        "mean_nested_std_error", "mean_nested_value", "exhausted_share",
    ]]
    table.to_csv(paths.TABLES / "proxy_accuracy.csv", index=False)
    pd.concat(delta_rows, ignore_index=True)[[
        "year", "path", "account_value", "benefit_base", "variance", "zero_10y",
        "nested_value", "proxy_value", "nested_delta", "proxy_delta", "delta_error",
        "nested_gamma", "proxy_gamma", "gamma_error", "outside_design_range",
    ]].to_csv(paths.TABLES / "proxy_delta_nodes.csv", index=False)

    usable = table[table["rmse_share_of_premium_in_range"] < 0.03]["year"]
    print(f"\nproxy within 3% of premium inside its design range through year "
          f"{int(usable.max()) if len(usable) else 0}")
    print(f"wrote {paths.TABLES / 'proxy_accuracy.csv'}")


if __name__ == "__main__":
    main()
