"""Check the committed inputs, bootstrap the curve history, and write the scaled disclosure.

Run this first. It fails loudly on the data problems that would otherwise surface as a wrong
number twenty minutes into a backtest: a par curve that cannot be bootstrapped, a volatility
index missing on a day the market traded, a mortality table whose margins run the wrong way, a
disclosed sensitivity that two filings do not agree on.

Nothing here is a model. Every check is either arithmetic that has to close or a comparison
between two things the filings say, which is what makes a failure here unambiguous.

Usage:  python -m scripts.build_dataset
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vahedge import paths
from vahedge.liability import mortality
from vahedge.market import scenarios
from vahedge.market.curves import bootstrap
from vahedge.market.heston_cos import HestonParameters
from vahedge.market.state import PAR_SERIES

# Round-trip tolerance on the par bootstrap, as a decimal yield. A tenth of a basis point is
# far above floating-point error and far below anything that would move a valuation, so a
# failure here means the bootstrap is wrong rather than imprecise.
PAR_TOLERANCE = 1e-6
EQUITY_SERIES = "SP500"
VOLATILITY_SERIES = ("VIXCLS", "VXVCLS")
CREDIT_SERIES = "BAA10Y"
# Overnight financing and the three-month bill, which are what the sub-account's money-market
# sleeve and the backtest's carry are built from. The long short-rate history the Hull-White fit
# uses lives in its own file and is checked by the calibration step rather than here.
CASH_SERIES = ("DFF", "DTB3")
# Zero tenors kept in the committed curve history. Enough to see the shape move without
# writing a file the size of the panel.
KEPT_TENORS = (1, 2, 5, 10, 20, 30)


def load_panel() -> pd.DataFrame:
    return pd.read_csv(paths.FRED_PANEL, comment="#", parse_dates=["date"]).set_index("date")


def equity_days(panel: pd.DataFrame) -> pd.DatetimeIndex:
    return panel.index[panel[EQUITY_SERIES].notna()]


def series_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in panel.columns:
        series = panel[column].dropna()
        rows.append({
            "series": column,
            "observations": int(series.size),
            "first": str(series.index.min().date()) if series.size else "",
            "last": str(series.index.max().date()) if series.size else "",
            "min": float(series.min()) if series.size else np.nan,
            "max": float(series.max()) if series.size else np.nan,
            "missing_in_window": int(panel[column].isna().sum()),
        })
    return pd.DataFrame(rows)


def calendar_check(panel: pd.DataFrame) -> pd.DataFrame:
    """How often each input is missing on a day the equity index traded.

    The equity calendar is the one that matters because it is the one the hedge rebalances on.
    A Treasury holiday that is not an equity holiday is a day with a price and no curve, and
    the loader drops it rather than carrying the previous curve forward - so the count here is
    the number of rebalance dates the replay loses, which belongs in the write-up.
    """
    trading = equity_days(panel)
    rows = []
    for column in list(PAR_SERIES.values()) + list(VOLATILITY_SERIES) + [CREDIT_SERIES] + list(
        CASH_SERIES
    ):
        if column not in panel:
            rows.append({"series": column, "equity_trading_days": int(trading.size),
                         "missing_on_trading_days": np.nan, "first_five_gaps": "not in the panel"})
            continue
        missing = panel.loc[trading, column].isna()
        rows.append({
            "series": column,
            "equity_trading_days": int(trading.size),
            "missing_on_trading_days": int(missing.sum()),
            "first_five_gaps": ", ".join(str(d.date()) for d in trading[missing][:5]),
        })
    return pd.DataFrame(rows)


def zero_curve_history(panel: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap a zero curve on every date with a complete par curve, and check each one.

    The check is a round trip: price the par bonds back off the zeros the bootstrap produced
    and compare with the quotes it started from. A bootstrap that does not reprice its own
    inputs is wrong in a way no downstream test would catch, because every later step would
    simply use the wrong curve consistently.
    """
    tenors = list(PAR_SERIES)
    columns = [PAR_SERIES[tenor] for tenor in tenors]
    usable = panel.loc[equity_days(panel), columns]
    complete = usable.dropna()
    records, worst = [], 0.0
    for date, row in complete.iterrows():
        quotes = row.to_numpy(dtype=float) / 100.0
        curve = bootstrap(tenors, quotes)
        errors = [abs(curve.par_equivalent(tenor) - quote)
                  for tenor, quote in zip(tenors, quotes)]
        worst = max(worst, max(errors))
        record = {"date": date, "worst_par_roundtrip": max(errors)}
        record.update({f"zero_{tenor}y": float(curve.zero(tenor)) for tenor in KEPT_TENORS})
        records.append(record)

    if worst > PAR_TOLERANCE:
        raise ValueError(f"par bootstrap round-trip error {worst:.2e} exceeds {PAR_TOLERANCE:.0e}")
    dropped = len(usable) - len(complete)
    print(f"  bootstrapped {len(records):,} curves, worst round-trip {worst:.2e}; "
          f"{dropped} trading days had an incomplete par curve and were dropped")
    return pd.DataFrame(records).set_index("date")


def volatility_check(panel: pd.DataFrame, heston: HestonParameters) -> pd.DataFrame:
    """The 30-day index against what the calibrated variance curve implies at 30 days.

    A genuine out-of-sample check rather than a restatement: the instantaneous variance is
    implied from the three-month index, and the one-month level the Heston forward curve then
    produces is compared with the one-month index, which was not used. Mean reversion cannot
    reproduce a steep one-to-three-month slope, so a gap is expected on the days when the front
    of the curve is dislocated; a gap that is large on ordinary days, or one-sided across the
    decade, would mean the variance mapping is wrong.
    """
    quotes = panel[list(VOLATILITY_SERIES)].dropna()
    three_month = quotes["VXVCLS"].to_numpy(dtype=float) / 100.0
    one_month = quotes["VIXCLS"].to_numpy(dtype=float) / 100.0
    implied = scenarios.implied_at_tenor(three_month, heston, quoted_tenor=0.25,
                                         target_tenor=scenarios.VIX_TENOR_YEARS)
    error = implied - one_month
    return pd.DataFrame({
        "observations": [int(error.size)],
        "mean_error_vol_points": [100 * float(error.mean())],
        "median_error_vol_points": [100 * float(np.median(error))],
        "sd_error_vol_points": [100 * float(error.std(ddof=1))],
        "worst_over_vol_points": [100 * float(error.max())],
        "worst_under_vol_points": [100 * float(error.min())],
        "share_within_two_points": [float(np.mean(np.abs(error) <= 0.02))],
    })


def mortality_check() -> pd.DataFrame:
    """Survival on both tables, and the direction the margins run.

    The Period table is the Basic table with the margins the Life Actuarial Task Force set, so
    it has to imply a longer life. If it ever did not, the two files would have been swapped
    somewhere and every reporting-basis number in the project would be the wrong way round.
    """
    rows = []
    for basis in ("basic", "period"):
        table = mortality.load(basis)
        for year in (2012, 2025):
            survival, _ = table.rates(np.array([70]), year, 45, 0.5)
            rows.append({
                "table": basis,
                "valuation_year": year,
                "expected_future_lifetime": float(survival[0].sum()),
                "survival_to_10y": float(survival[0][9]),
                "survival_to_20y": float(survival[0][19]),
                "survival_to_30y": float(survival[0][29]),
            })
    frame = pd.DataFrame(rows)
    latest = frame[frame["valuation_year"] == 2025].set_index("table")
    if (latest.loc["period", "expected_future_lifetime"]
            <= latest.loc["basic", "expected_future_lifetime"]):
        raise ValueError("the Period table carries margins, so it has to imply the longer life")
    improvement = (frame[frame["valuation_year"] == 2025]["expected_future_lifetime"].mean()
                   - frame[frame["valuation_year"] == 2012]["expected_future_lifetime"].mean())
    print(f"  G2 improvement from 2012 to 2025 adds {improvement:.2f} years of expected "
          f"lifetime at age 70")
    return frame


def scaled_disclosure() -> pd.DataFrame:
    """Reconcile the overlapping filings, then scale every figure by account value.

    Consecutive 10-Ks repeat a year: FY2023 and FY2024 both report 31 December 2023, and FY2024
    and FY2025 both report 31 December 2024. Where the shock size is the same the two have to
    agree, and a disagreement would mean a transcription error in the committed input, which is
    the one error this project cannot detect any other way.
    """
    disclosed = pd.read_csv(paths.DISCLOSED_SENSITIVITIES, comment="#")
    repeated = (disclosed.groupby(["as_of", "line_item", "shock"])["impact_musd"]
                .agg(["nunique", "count"]).reset_index())
    conflicts = repeated[(repeated["count"] > 1) & (repeated["nunique"] > 1)]
    if not conflicts.empty:
        raise ValueError(f"filings disagree on a disclosed figure:\n{conflicts}")
    overlapping = int((repeated["count"] > 1).sum())
    print(f"  {overlapping} disclosed figures appear in two filings and agree in every one")

    statistics = pd.read_csv(paths.BOOK_STATISTICS, comment="#")
    scaled = disclosed.merge(statistics[["as_of", "va_separate_account_musd"]], on="as_of",
                             how="left")
    # The pre-LDTI guarantee liability is disclosed for a date the book statistics do not
    # cover, and the file that carries it says why: it is there for the scale-free asymmetry
    # test and never for a level comparison, so it does not need an account value and is left
    # unscaled. A market risk benefit without one is a different matter - every level
    # comparison in the project divides by it - so that is still an error.
    unscalable = scaled[scaled["va_separate_account_musd"].isna()]
    blocking = unscalable[unscalable["line_item"] == "market_risk_benefits"]
    if not blocking.empty:
        raise ValueError(f"no separate account value for market risk benefits at "
                         f"{sorted(blocking['as_of'].unique())}")
    if not unscalable.empty:
        print(f"  {len(unscalable)} pre-LDTI rows at "
              f"{', '.join(sorted(unscalable['as_of'].unique()))} are left unscaled, which is "
              f"what they are for")
    scaled["impact_pct_of_av"] = scaled["impact_musd"] / scaled["va_separate_account_musd"]
    scaled["fair_value_pct_of_av"] = (scaled["fair_value_musd"]
                                      / scaled["va_separate_account_musd"])
    return scaled


def main() -> None:
    paths.ensure_output_dirs()
    panel = load_panel()
    print(f"Panel: {len(panel):,} dates, {panel.index.min().date()} to "
          f"{panel.index.max().date()}, {len(equity_days(panel)):,} equity trading days")

    summary = series_summary(panel)
    summary.to_csv(paths.TABLES / "data_series_summary.csv", index=False, float_format="%.4f")
    print(f"  {len(summary)} series checked")

    calendar = calendar_check(panel)
    calendar.to_csv(paths.TABLES / "data_calendar_check.csv", index=False, float_format="%.0f")
    worst = calendar.dropna(subset=["missing_on_trading_days"]).nlargest(
        1, "missing_on_trading_days")
    if not worst.empty:
        row = worst.iloc[0]
        print(f"  worst calendar gap: {row['series']} missing on "
              f"{int(row['missing_on_trading_days'])} of "
              f"{int(row['equity_trading_days'])} equity trading days")

    curves = zero_curve_history(panel)
    curves.to_csv(paths.ZERO_CURVES, float_format="%.8f")
    panel.to_csv(paths.MARKET_PANEL)

    mortality_table = mortality_check()
    mortality_table.to_csv(paths.TABLES / "mortality_check.csv", index=False,
                           float_format="%.4f")

    # The surface's own parameters are not available until the calibration step has run, so
    # this check uses the saved calibration when there is one and says so when there is not.
    if paths.MARKET_CALIBRATION.exists():
        from vahedge.market import state as market_state
        volatility = volatility_check(panel, market_state.load().heston)
        volatility.to_csv(paths.TABLES / "volatility_curve_check.csv", index=False,
                          float_format="%.4f")
        row = volatility.iloc[0]
        print(f"  one-month volatility implied from the three-month index is "
              f"{row['mean_error_vol_points']:+.2f} points off the one-month index on average, "
              f"within two points on {row['share_within_two_points']:.0%} of "
              f"{int(row['observations']):,} days")
    else:
        print("  volatility check skipped: run scripts/run_calibration.py first")

    scaled = scaled_disclosure()
    scaled.to_csv(paths.DISCLOSED_SCALED, index=False)
    mrb = scaled.query("line_item == 'market_risk_benefits'")
    print(f"  scaled {len(mrb)} market risk benefit sensitivities across "
          f"{mrb['as_of'].nunique()} balance-sheet dates")
    print(f"\nwrote the curve history, the scaled disclosure and four checks to "
          f"{paths.DATA_PROCESSED} and {paths.TABLES}")


if __name__ == "__main__":
    main()
