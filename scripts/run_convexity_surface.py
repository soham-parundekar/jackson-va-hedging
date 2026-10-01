"""Build the nested convexity surface the hedge's option leg is sized from, and test it.

The regression proxy's second derivative is not usable - the validation puts its error at about
a hundred per cent of the quantity at every horizon - so the strategies that buy convexity get
their gamma from nested valuations instead. This builds that surface once and writes it to
data/processed, where the hedging experiments read it.

Three things come out alongside, and each answers a question the surface alone leaves open.

The first is the comparison against the proxy's own curvature at the same nodes, which is what
justifies doing this at all. If the two agreed there would be no reason for the file to exist.

The second is a sensitivity, and it is read off the nodes themselves rather than from a second
run. Each node carries the variance of the path it was taken from, so the spread of the
curvature at a given moneyness across those is exactly the size of what the surface leaves out
by tabulating against moneyness alone. Rebuilding the whole surface at a shocked volatility was
the first attempt and it measured nothing: every node takes its own variance from its own
recorded path, so shocking the market state the nodes are drawn into does not reach them.

The third is the shape itself, which is worth looking at rather than only summarising. A
guarantee's convexity is a bell: near zero far out of the money, near zero once the contract is
spent and the liability is a life annuity, and at its largest where the account value is close
to the point at which the guarantee starts to bite. The first build of this surface could not
show the far side of that bell, because every node is an anniversary state and the ratchet
leaves no live anniversary with an account above its benefit base. The hedge sees moneyness
above one on most days of a rising market, so the nodes near the base are now valued along a
ladder of equity shifts that carries the curve to about 1.45.

Usage:  python -m scripts.run_convexity_surface
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from vahedge import paths
from vahedge.liability import cohorts, gmwb, mortality
from vahedge.liability import terms as terms_module
from vahedge.market import state as market_state
from vahedge.market.simulate import simulate
from vahedge.valuation import lsmc, nested
from vahedge.valuation.convexity import ConvexitySurface
from vahedge.valuation.engine import MarketState, Valuer

ISSUE_AGE = 70
DEFERRAL_YEARS = 5
MAX_AGE = 105
PREMIUM = 100.0
BACKTEST_START = "2016-09-26"     # the surface is built off the same curve the proxy is

FIT_PATHS = 20_000
FIT_SEED = 20251231
INNER_PATHS = 5_000
INNER_SEED = 4242
NODES_PER_YEAR = 25
EQUITY_BUMP = 0.10
# The policy years the hedge will visit: the backtest starts at duration three and runs ten
# years, so the surface has to cover three to thirteen, with a margin either side.
SURFACE_YEARS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)


def build(curve_date: str = BACKTEST_START):
    calibration = market_state.load()
    panel = pd.read_csv(paths.FRED_PANEL, comment="#", parse_dates=["date"]).set_index("date")
    calibration = replace(
        calibration, curve=market_state.treasury_curve(panel, pd.Timestamp(curve_date))
    )
    state = MarketState.from_calibration(calibration)
    core = terms_module.load()[("flex_gmwb", "single", "core")]
    book = cohorts.single_contract(
        core, issue_age=ISSUE_AGE, base_contract_charge=0.0131, fund_expense=0.0095,
        premium=PREMIUM, deferral_years=DEFERRAL_YEARS, max_age=MAX_AGE,
        lapse_rate=0.04, lapse_beta=1.2, lapse_floor=0.01,
    )
    horizon = int(book.projection_years.max())
    survival, deaths = mortality.load("basic").rates(
        book.attained_age, state.valuation_year, horizon, 0.5
    )
    market_paths = simulate(state.heston, state.hull_white(), state.correlations, state.mix,
                            n_years=horizon, n_paths=FIT_PATHS, seed=FIT_SEED)
    projection = gmwb.project(book, market_paths, survival, deaths,
                              equity_weight=state.mix.equity_weight, record=True)
    return book, state, projection, lsmc.fit(projection)


def surface_for(book, state, projection, years=SURFACE_YEARS) -> pd.DataFrame:
    valuer = Valuer(mortality.load("basic"), n_paths=INNER_PATHS, seed=INNER_SEED, cache_size=1)
    return nested.gamma_surface(
        valuer, book, state, projection.recorded, years=years,
        n_nodes=NODES_PER_YEAR, equity_bump=EQUITY_BUMP,
    )


def against_the_proxy(table: pd.DataFrame, proxy) -> pd.DataFrame:
    """The proxy's own curvature at the same nodes, which is why the surface exists.

    The design flag comes along, because the nodes are stratified across the whole live
    moneyness range and the ends of that range are where the fit has nothing. Judging the proxy
    on all of them would be judging it on states it declines to answer for.
    """
    rows = []
    for _, row in table.iterrows():
        year = int(row["policy_year"]) - 1
        if year not in proxy.fits:
            continue
        greeks = proxy.greeks(
            year, row["moneyness"] * row["benefit_base"], row["benefit_base"],
            row["variance"], row["zero_10y"], gamma_step=EQUITY_BUMP,
        )
        rows.append({
            "policy_year": row["policy_year"], "moneyness": row["moneyness"],
            "nested_gamma": row["gamma"], "proxy_gamma": float(np.ravel(greeks["gamma"])[0]),
            "nested_value": row["value"], "proxy_value": float(np.ravel(greeks["value"])[0]),
            "outside_design": float(np.ravel(greeks["outside_design"])[0]) > 0.5,
        })
    frame = pd.DataFrame(rows)
    frame["gamma_error"] = frame["proxy_gamma"] - frame["nested_gamma"]
    frame["value_error"] = frame["proxy_value"] - frame["nested_value"]
    return frame


def volatility_spread(table: pd.DataFrame, window: float = 0.08) -> pd.DataFrame:
    """How much the curvature at a given moneyness varies with the node's own volatility.

    The surface is indexed by policy year and moneyness, so everything the volatility does to
    the curvature is averaged into the curve. The nodes carry it, so the size of that
    approximation is measurable without another run: take the nodes within a narrow band of
    moneyness, split them at the median variance, and compare. A gap that is small relative to
    the curvature means the simplification costs little; a large one is a reason to index the
    surface on volatility too, at a hundred times the compute.

    Only the rungs a path actually produced count here. A walked-up point inherits the variance
    of the node it was walked from, so a band made of several rungs of one node is one draw of
    the variance wearing five hats, and splitting it at its own median puts everything on one
    side.
    """
    rows = []
    for year, block in table[table["rung"] == 0].groupby("policy_year"):
        middle = block[(block["moneyness"] > 0.5 - window) & (block["moneyness"] < 0.5 + window)]
        near = block[block["moneyness"] > 0.85]
        for label, chosen in (("moneyness near 0.5", middle), ("moneyness above 0.85", near)):
            if chosen.shape[0] < 4:
                continue
            ordered = chosen.sort_values("variance")
            half = ordered.shape[0] // 2
            quiet = ordered.iloc[:half]["gamma_per_unit"].mean()
            busy = ordered.iloc[-half:]["gamma_per_unit"].mean()
            if not np.isfinite(quiet) or not np.isfinite(busy):
                continue
            level = max(abs(quiet), abs(busy), 1e-9)
            rows.append({
                "policy_year": year, "where": label, "nodes": chosen.shape[0],
                "quiet_half": quiet, "busy_half": busy,
                "gap_share_of_level": abs(busy - quiet) / level,
            })
    return pd.DataFrame(rows)


def main() -> None:
    paths.ensure_output_dirs()
    book, state, projection, proxy = build()

    table = surface_for(book, state, projection)
    surface = ConvexitySurface(
        table=table[["policy_year", "moneyness", "gamma_per_unit"]],
        equity_bump=EQUITY_BUMP, inner_paths=INNER_PATHS,
    )
    surface.to_csv(paths.GAMMA_SURFACE)
    table.to_csv(paths.TABLES / "convexity_surface.csv", index=False)

    comparison = against_the_proxy(table, proxy)
    comparison.to_csv(paths.TABLES / "convexity_proxy_comparison.csv", index=False)

    walked = table[table["rung"] > 0].shape[0]
    print(f"surface: {table.shape[0]} points across {table['policy_year'].nunique()} policy "
          f"years, of which {walked} sit above the ratchet's cap and were reached by walking "
          f"a node up; moneyness spans {table['moneyness'].min():.2f} to "
          f"{table['moneyness'].max():.2f}")
    shape = table.groupby("policy_year").apply(
        lambda block: pd.Series({
            "peak_gamma_per_unit": block["gamma_per_unit"].max(),
            "at_peak_moneyness": block.loc[block["gamma_per_unit"].idxmax(), "moneyness"],
            "at_exhaustion": block.sort_values("moneyness")["gamma_per_unit"].iloc[0],
            "at_the_money": float(np.interp(
                1.0, block.sort_values("moneyness")["moneyness"],
                block.sort_values("moneyness")["gamma_per_unit"])),
            # The point of extending the ladder: what the surface used to hold flat at its
            # top end, against what is there once the account is walked above the base.
            "at_1_25": float(np.interp(
                1.25, block.sort_values("moneyness")["moneyness"],
                block.sort_values("moneyness")["gamma_per_unit"])),
        }), include_groups=False,
    )
    print("\nshape of the curvature, per unit of benefit base")
    print(shape.round(4).to_string())

    print("\nproxy curvature against nested, at the nodes inside the proxy's design")
    inside = comparison[~comparison["outside_design"]]
    summary = inside.groupby("policy_year").apply(
        lambda block: pd.Series({
            "nodes": float(block.shape[0]),
            "mean_abs_nested": block["nested_gamma"].abs().mean(),
            "rmse_gamma": float(np.sqrt((block["gamma_error"] ** 2).mean())),
            "rmse_value": float(np.sqrt((block["value_error"] ** 2).mean())),
        }), include_groups=False,
    )
    summary["gamma_error_share"] = summary["rmse_gamma"] / summary["mean_abs_nested"]
    print(summary.round(3).to_string())
    print(f"  nodes flagged outside the design: {comparison['outside_design'].mean():.0%}")

    print("\nwhat tabulating against moneyness alone leaves out")
    print(volatility_spread(table).round(3).to_string())

    print(f"\nwrote {paths.GAMMA_SURFACE}")


if __name__ == "__main__":
    main()
