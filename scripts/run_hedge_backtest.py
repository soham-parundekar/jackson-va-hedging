"""Hedge the rider weekly through ten years of realised markets and measure the result.

One policy, issued on the first date FRED carries an S&P 500 close, hedged every week
until the end of the sample. The window contains the February 2018 volatility spike, the
February and March 2020 collapse, the 2022 bond and equity drawdown, and the recovery
since, which is as much stress as a decade of free data provides.

Every leg is computed on each date, so the effect of adding delta, then rho, then a
volatility overlay is read off one run rather than four. The sizing of each leg depends
only on its own Greek, and no hedge position uses information from after the date it is
set on.

Two runs are reported. The first puts the contract holder in a single broad equity
sub-account, which removes fund basis risk and isolates what discrete rebalancing and
curve reshaping cost on their own. The second uses Jackson's disclosed fund mix, so the
account value follows a blend of equity, bond and balanced funds while the hedge still
trades one index.

A caution that belongs next to the headline number rather than in a footnote: the
liability is revalued with the same model that produced the hedge ratios. Any risk factor
the model represents and the hedge covers will be removed almost completely, limited only
by convexity between rebalances. What this backtest measures is the cost of hedging
discretely with imperfect instruments. What it cannot measure is model error, and on a
forty-year guarantee that is the larger risk.

Usage:  python -m scripts.run_hedge_backtest
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gmwb import figures, hedging, market, mortality, paths, session
from gmwb.engine import calibrate_attribution, make_normals, projection_years, value_rider
from gmwb.hedging import SubAccountMix

LEG_SETS = [
    ("unhedged", ()),
    ("delta", ("equity",)),
    ("delta and rho", ("equity", "rates")),
    ("delta, rho and vega", ("equity", "rates", "vega")),
]


def inception_attribution(s: session.Session, normals: np.ndarray, mix: SubAccountMix,
                          start) -> float:
    """Attribution percentage fixed on the market of the day the policy was written."""
    from dataclasses import replace

    contract = replace(s.contract, fund_equity_beta=mix.effective_equity_beta)
    date = market.equity_dates(s.panel, start, None)[0]
    state = market.state_at(s.panel, date, s.cfg, s.long_run_vol)
    basis = mortality.load(contract.issue_age, date.year,
                           float(s.cfg["contract"]["sex_mix"]["male"]))
    at_inception = value_rider(
        contract, state.curve_builder.build(), state.vol, basis, normals,
        max_age=int(s.cfg["simulation"]["max_age"]),
    )
    alpha = calibrate_attribution(at_inception)
    print(f"  inception {date.date()}: S&P 500 {state.spot:,.0f}, "
          f"10-year zero {100 * state.curve_builder.build().zero(10):.2f}%, "
          f"3-month implied {100 * state.implied_vol_3m:.1f}%, attribution {alpha:.4f}")
    return alpha


def leg_table(ledger: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict]]:
    rows = []
    results = {}
    for label, legs in LEG_SETS:
        composed = hedging.compose(ledger, legs)
        metrics = hedging.effectiveness(composed)
        results[label] = metrics
        rows.append(
            {
                "hedge": label,
                "weekly_std": metrics["hedged_std"],
                "variance_ratio": metrics["variance_ratio"],
                "variance_reduction_pct": 100 * metrics["variance_reduction"],
                "worst_week": metrics["hedged_worst"],
                "max_drawdown": metrics["hedged_max_drawdown"],
                "cumulative_pnl": metrics["hedged_total"],
                "total_costs": metrics["total_costs"],
            }
        )
    return pd.DataFrame(rows), results


def stress_windows(ledger: pd.DataFrame) -> pd.DataFrame:
    """Hedge performance inside the periods that actually hurt."""
    windows = {
        "Feb 2018 volatility spike": ("2018-01-26", "2018-02-28"),
        "Q4 2018 drawdown": ("2018-09-28", "2018-12-31"),
        "Feb-Mar 2020": ("2020-02-14", "2020-03-31"),
        "2022 full year": ("2022-01-01", "2022-12-31"),
        "rest of the sample": (None, None),
    }
    stress_index = pd.DatetimeIndex([])
    rows = []
    for label, (start, end) in windows.items():
        if start is None:
            subset = ledger.drop(index=stress_index, errors="ignore")
        else:
            subset = ledger.loc[start:end]
            stress_index = stress_index.union(subset.index)
        composed = hedging.compose(subset, ("equity", "rates"))
        valid = composed.dropna(subset=["liability_pnl"])
        if valid.empty:
            continue
        with_vega = hedging.compose(subset, ("equity", "rates", "vega")).dropna(
            subset=["liability_pnl"]
        )
        rows.append(
            {
                "window": label,
                "weeks": len(valid),
                "index_return_pct": 100 * (valid["spot"].iloc[-1] / valid["spot"].iloc[0] - 1),
                "unhedged_total": valid["liability_pnl"].sum(),
                "delta_rho_total": valid["hedged_pnl"].sum(),
                "plus_vega_total": with_vega["hedged_pnl"].sum(),
                "unhedged_worst_week": valid["liability_pnl"].min(),
                "delta_rho_worst_week": valid["hedged_pnl"].min(),
            }
        )
    return pd.DataFrame(rows)


def exposure_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    valid = ledger.dropna(subset=["liability_pnl"])
    rows = []
    for column, label in (
        ("rider_value", "rider value, $"),
        ("account_value", "account value, $"),
        ("benefit_base", "benefit base, $"),
        ("equity_exposure", "equity exposure, $"),
        ("rho_per_bp", "rho, $ per bp"),
        ("vega_per_point", "vega, $ per vol point"),
        ("equity_gamma", "equity gamma, $"),
        ("swap_notional", "swap notional, $"),
    ):
        series = valid[column]
        rows.append(
            {
                "quantity": label,
                "start": series.iloc[0],
                "min": series.min(),
                "median": series.median(),
                "max": series.max(),
                "end": series.iloc[-1],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    s = session.start()
    cfg = s.cfg
    hcfg = cfg["hedge"]
    n_paths = int(hcfg["n_paths"])
    normals = make_normals(n_paths, s.n_years, int(cfg["simulation"]["seed"]),
                           bool(cfg["simulation"]["antithetic"]))
    book_mix = SubAccountMix.from_config(cfg)

    print(f"Weekly rebalancing from {hcfg['start']} to {hcfg['end']}, "
          f"{n_paths:,} paths per valuation")

    ledgers = {}
    for label, mix in (("single equity sub-account", SubAccountMix.all_equity()),
                       ("disclosed fund mix", book_mix)):
        print(f"\n{label}")
        alpha = inception_attribution(s, normals, mix, hcfg["start"])
        ledger = hedging.run_backtest(
            s.contract, cfg, s.panel, normals, alpha, s.long_run_vol,
            mix=mix, hedge_equity=True, hedge_rates=True, hedge_vega=True,
            with_gaap=(mix is book_mix),
        )
        ledgers[label] = ledger
        table, results = leg_table(ledger)
        print(table.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
        suffix = "all_equity" if mix is not book_mix else "book_mix"
        session.write_table(table, f"hedge_effectiveness_{suffix}", "%.4f")
        ledger.to_csv(paths.DATA_PROCESSED / f"hedge_ledger_{suffix}.csv",
                      float_format="%.6f")
        if mix is book_mix:
            print("\nFigure")
            print("  " + figures.leg_comparison(results))

    ledger = ledgers["disclosed fund mix"]
    print("\nExposures over the window, disclosed fund mix")
    exposures = exposure_summary(ledger)
    print(exposures.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    session.write_table(exposures, "hedge_exposures", "%.2f")

    print("\nStress windows, delta and rho hedged")
    stress = stress_windows(ledger)
    print(stress.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    session.write_table(stress, "hedge_stress_windows", "%.2f")

    print("\nWhat the delta and rho hedge leaves behind")
    composed = hedging.compose(ledger, ("equity", "rates"))
    attribution = hedging.residual_attribution(composed)
    print(attribution.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    session.write_table(attribution, "hedge_residual_attribution")

    annual_factor = np.sqrt(52.0)
    metrics = hedging.effectiveness(composed)
    print(f"\nWeekly unhedged standard deviation {metrics['unhedged_std']:,.0f}, "
          f"annualised {metrics['unhedged_std'] * annual_factor:,.0f}")
    print(f"Weekly hedged standard deviation   {metrics['hedged_std']:,.0f}, "
          f"annualised {metrics['hedged_std'] * annual_factor:,.0f}")
    print(f"Variance reduction {100 * metrics['variance_reduction']:.1f}% "
          f"over {metrics['periods']} weeks")
    print(f"Transaction costs over the window {metrics['total_costs']:,.0f} "
          f"on a {s.contract.premium:,.0f} policy")

    print("\nFigure")
    print("  " + figures.hedge_performance(composed))


if __name__ == "__main__":
    main()
