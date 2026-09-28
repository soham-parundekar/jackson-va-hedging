"""Check the raw inputs, bootstrap the curve history, and write the processed panel.

Run this first. It fails loudly on the data problems that would otherwise turn into a
wrong number twenty minutes later: a par curve that cannot be bootstrapped, a stale
volatility index, a mortality table with a hole in it above age 40, a disclosed
sensitivity that does not reconcile between two filings.

Usage:  python -m scripts.build_dataset
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gmwb import config, market, mortality, paths, session
from gmwb.volatility import realised_vol


def check_panel(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in frame.columns:
        series = frame[column].dropna()
        rows.append(
            {
                "series": column,
                "observations": int(series.size),
                "first": str(series.index.min().date()) if series.size else "",
                "last": str(series.index.max().date()) if series.size else "",
                "min": float(series.min()) if series.size else np.nan,
                "max": float(series.max()) if series.size else np.nan,
                "missing_in_window": int(frame[column].isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def check_calendar(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """How often each input is missing on a day the equity index traded."""
    equity_days = market.equity_dates(frame)
    rows = []
    for column in list(market.PAR_YIELD_COLUMNS.values()) + ["VIXCLS", "VXVCLS", "BAA10Y", "DTB3"]:
        missing = frame.loc[equity_days, column].isna()
        rows.append(
            {
                "series": column,
                "equity_trading_days": int(equity_days.size),
                "missing_after_fill": int(missing.sum()),
                "worst_gap_dates": ", ".join(
                    str(d.date()) for d in equity_days[missing][:5]
                ),
            }
        )
    return pd.DataFrame(rows)


def build_zero_curves(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Bootstrap a zero curve on every date with a complete par curve.

    Every bootstrap is checked by repricing the par bonds off the zeros it produced.
    A round-trip error above a tenth of a basis point means the bootstrap is wrong,
    not merely imprecise.
    """
    tenors = list(cfg["curve"]["par_tenors"])
    records = []
    worst_error = 0.0
    failures = []

    for date in market.equity_dates(frame):
        row = frame.loc[date]
        yields = np.array([row[market.PAR_YIELD_COLUMNS[t]] for t in tenors], dtype=float)
        if np.any(~np.isfinite(yields)):
            failures.append((str(date.date()), "missing par yield"))
            continue
        state = market.state_at(frame, date, cfg, 0.2)
        curve = state.curve_builder.build()
        errors = [abs(curve.par_equivalent(t) - yields[i] / 100.0) for i, t in enumerate(tenors)]
        worst_error = max(worst_error, max(errors))
        record = {"date": date, "worst_par_roundtrip": max(errors)}
        for t in (1, 2, 5, 10, 20, 30):
            record[f"zero_{t}y"] = float(curve.zero(t))
        records.append(record)

    if worst_error > 1e-6:
        raise ValueError(f"par bootstrap round-trip error {worst_error:.2e} is too large")
    if failures:
        print(f"  skipped {len(failures)} dates without a complete par curve")
        for date, reason in failures[:5]:
            print(f"    {date}: {reason}")
    print(f"  bootstrapped {len(records)} curves, worst round-trip {worst_error:.2e}")
    return pd.DataFrame(records).set_index("date")


def volatility_check(frame: pd.DataFrame, cfg: dict, long_run: float) -> pd.DataFrame:
    """Compare the fitted curve's one-month volatility with the 30-day index.

    The curve is fitted to the 3-month index only, so the 30-day index is a genuine
    out-of-sample check on the shape rather than a restatement of an input. A
    mean-reverting curve cannot reproduce a steep one-month to three-month slope, so a
    gap is expected; what would be a problem is a gap that is large or one-sided.
    """
    rows = []
    vcfg = cfg["volatility"]
    short = float(vcfg["short_tenor_years"])
    for date in market.equity_dates(frame)[::5]:
        state = market.state_at(frame, date, cfg, long_run)
        if not np.isfinite(state.vix):
            continue
        rows.append(
            {
                "date": date,
                "vix_observed": state.vix / 100.0,
                "model_one_month": float(state.vol.spot_vol(short)),
                "vix3m_observed": state.implied_vol_3m,
                "model_three_month": float(state.vol.spot_vol(
                    float(vcfg["mid_tenor_years"]))),
            }
        )
    frame_out = pd.DataFrame(rows).set_index("date")
    frame_out["one_month_error"] = frame_out["model_one_month"] - frame_out["vix_observed"]
    frame_out["three_month_error"] = (
        frame_out["model_three_month"] - frame_out["vix3m_observed"]
    )
    worst_fit = frame_out["three_month_error"].abs().max()
    if worst_fit > 1e-9:
        raise ValueError(
            f"the curve should reproduce its fitting point exactly, worst error {worst_fit:.2e}"
        )
    return frame_out


def check_mortality() -> pd.DataFrame:
    """Survival on the Basic and Period tables, and the effect of G2 improvement."""
    rows = []
    for table in ("basic", "period"):
        for year in (2012, 2025):
            basis = mortality.load(70, year, 0.5, table=table)
            survival = basis.survival(45)
            rows.append(
                {
                    "table": table,
                    "valuation_year": year,
                    "expected_future_lifetime": float(survival.sum()),
                    "survival_to_10y": float(survival[9]),
                    "survival_to_20y": float(survival[19]),
                    "survival_to_30y": float(survival[29]),
                }
            )
    frame = pd.DataFrame(rows)

    basic_2025 = frame.query("table == 'basic' and valuation_year == 2025")
    period_2025 = frame.query("table == 'period' and valuation_year == 2025")
    if (
        period_2025["expected_future_lifetime"].iloc[0]
        <= basic_2025["expected_future_lifetime"].iloc[0]
    ):
        raise ValueError(
            "the Period table carries margins, so it must imply longer expected lifetime "
            "than the Basic table"
        )
    return frame


def check_disclosures() -> pd.DataFrame:
    """Reconcile the overlapping year in consecutive filings.

    The FY2023 and FY2024 filings both report 31 December 2023, and the FY2024 and
    FY2025 filings both report 31 December 2024. Where the shock size is the same the
    numbers have to agree, and the equity shock size never changed.
    """
    disclosed = pd.read_csv(paths.DISCLOSED_SENSITIVITIES)
    overlaps = (
        disclosed.groupby(["as_of", "line_item", "shock"])["impact_musd"]
        .agg(["nunique", "count", "min", "max"])
        .reset_index()
    )
    conflicts = overlaps[(overlaps["count"] > 1) & (overlaps["nunique"] > 1)]
    if not conflicts.empty:
        raise ValueError(f"disclosed values disagree across filings:\n{conflicts}")
    repeated = overlaps[overlaps["count"] > 1]
    print(f"  {len(repeated)} disclosed figures appear in two filings and agree in all of them")

    stats = pd.read_csv(paths.BOOK_STATISTICS)
    mrb = disclosed.query("line_item == 'market_risk_benefits'").drop_duplicates(
        subset=["as_of", "shock"]
    )
    merged = mrb.merge(stats[["as_of", "va_separate_account_musd"]], on="as_of", how="left")
    if merged["va_separate_account_musd"].isna().any():
        missing = merged[merged["va_separate_account_musd"].isna()]["as_of"].unique()
        raise ValueError(f"no account value for {missing}")
    merged["impact_pct_of_av"] = merged["impact_musd"] / merged["va_separate_account_musd"]
    merged["fair_value_pct_of_av"] = merged["fair_value_musd"] / merged["va_separate_account_musd"]
    return merged


def main() -> None:
    cfg = config.load()
    frame = market.load_panel()
    paths.ensure_output_dirs()

    print("Market panel")
    summary = check_panel(frame)
    print(summary.to_string(index=False))
    session.write_table(summary, "data_series_summary", "%.4f")

    print("\nCalendar alignment")
    calendar = check_calendar(frame, cfg)
    print(calendar.to_string(index=False))
    session.write_table(calendar, "data_calendar_check", "%.0f")

    print("\nCurve bootstrap")
    curves = build_zero_curves(frame, cfg)
    curves.to_csv(paths.ZERO_CURVES, float_format="%.6f")

    print("\nMortality")
    mort = check_mortality()
    print(mort.to_string(index=False))
    session.write_table(mort, "mortality_check", "%.4f")

    print("\nDisclosed sensitivities")
    disclosed = check_disclosures()
    disclosed.to_csv(paths.DATA_PROCESSED / "disclosed_scaled.csv", index=False,
                     float_format="%.6f")
    view = disclosed.query("shock.str.startswith('equity')", engine="python")[
        ["as_of", "shock", "impact_musd", "impact_pct_of_av"]
    ]
    print(view.to_string(index=False))

    prices = frame["SP500"].dropna()
    vcfg = cfg["volatility"]
    long_run = market.long_run_vol(frame, int(vcfg["long_run_lookback_years"]),
                                  float(vcfg["long_run_risk_margin"]))
    print("\nVolatility levels")
    print(f"  realised over the full S&P 500 history: {realised_vol(prices.to_numpy()):.4f}")
    print(f"  long-run level used, including the risk margin: {long_run:.4f}")

    check = volatility_check(frame, cfg, long_run)
    print("\nVolatility curve against the 30-day index it was not fitted to")
    print(check.describe().loc[["count", "mean", "50%", "min", "max"]].to_string(
        float_format=lambda v: f"{v:,.4f}"))
    session.write_table(check.reset_index(), "volatility_curve_check", "%.4f")

    frame.to_csv(paths.MARKET_PANEL, float_format="%.6f")
    print(f"\nWrote {paths.MARKET_PANEL.name} and {paths.ZERO_CURVES.name} to data/processed")


if __name__ == "__main__":
    main()
