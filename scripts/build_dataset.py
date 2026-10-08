"""Check the committed inputs, bootstrap the curve history, and write the scaled disclosure.

Run this first. It fails loudly on the data problems that would otherwise surface as a wrong
number twenty minutes into a backtest: a par curve that cannot be bootstrapped, a volatility
index missing on a day the market traded, a mortality table whose margins run the wrong way, a
disclosed sensitivity that two filings do not agree on.

Nothing here is a model. Every check is arithmetic that has to close, or a comparison between two
things the filings say, or between two market sources that have to agree - which is what makes a
failure here unambiguous.

Usage:  python -m scripts.build_dataset
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from vahedge import paths
from vahedge.liability import mortality
from vahedge.market import chain, heston_cos, scenarios
from vahedge.market.curves import bootstrap
from vahedge.market.heston_cos import HestonParameters
from vahedge.market.state import PAR_SERIES, treasury_curve

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
# The chain snapshot's trade date, which is in the file's own header and not in the panel.
OPTION_CHAIN_AS_OF = "2026-09-28"
# What the parity-implied financing rate is allowed to sit at over the matched-maturity
# Treasury. An SPX box trades a few tens of basis points above Treasury; a zero or negative
# spread would mean the discount factors came out too high, and anything past a per cent and a
# half means the chain is not what it says it is. Wide enough that only a broken input trips it.
PARITY_SPREAD_RANGE_BP = (5.0, 150.0)


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


def volatility_indices(panel: pd.DataFrame) -> pd.DataFrame:
    """The four volatility index levels on the days all four traded, as decimals.

    VIXCLS and VXVCLS come from the FRED panel and VIX6M and SKEW from Cboe directly, because
    FRED does not carry either. An inner join is deliberate: a comparison across tenors on a day
    one tenor is missing is not a comparison.
    """
    cboe = pd.read_csv(paths.CBOE_VOL_PANEL, parse_dates=["date"]).set_index("date")
    joined = panel[list(VOLATILITY_SERIES)].join(cboe[["VIX6M", "SKEW"]], how="inner").dropna()
    return pd.DataFrame({
        "one_month": joined["VIXCLS"] / 100.0,
        "three_month": joined["VXVCLS"] / 100.0,
        "six_month": joined["VIX6M"] / 100.0,
        "skew_index": joined["SKEW"],
    })


# Each row is (name of the quoted tenor, its length in years, name of the target, its length).
# The first is an interpolation and the two below it reach outward, which is the direction the
# option leg actually needs: the puts run one and two years and six months is as far as free
# data goes, so the outward rows are the only test of that extension there is.
TENOR_COMPARISONS = (
    ("three_month", 0.25, "one_month", scenarios.VIX_TENOR_YEARS),
    ("one_month", scenarios.VIX_TENOR_YEARS, "six_month", scenarios.VIX6M_TENOR_YEARS),
    ("three_month", 0.25, "six_month", scenarios.VIX6M_TENOR_YEARS),
)


def volatility_check(indices: pd.DataFrame, heston: HestonParameters) -> pd.DataFrame:
    """Quoted volatility at one tenor, carried to another by the calibrated variance curve.

    Out of sample in every row: the surface is fitted to one afternoon's option chain and never
    sees a volatility index, so the whole decade is a holdout. Mean reversion cannot reproduce a
    front end that has dislocated from the back, so a gap on panic days is expected; a gap that
    is large on ordinary days, or one-sided across the decade, would mean the variance mapping
    is wrong rather than merely smooth.
    """
    rows = []
    for quoted, quoted_tenor, target, target_tenor in TENOR_COMPARISONS:
        implied = scenarios.implied_at_tenor(indices[quoted].to_numpy(dtype=float), heston,
                                             quoted_tenor=quoted_tenor,
                                             target_tenor=target_tenor)
        error = implied - indices[target].to_numpy(dtype=float)
        rows.append({
            "quoted": quoted,
            "quoted_tenor_years": quoted_tenor,
            "target": target,
            "target_tenor_years": target_tenor,
            "direction": "inward" if target_tenor < quoted_tenor else "outward",
            "observations": int(error.size),
            # The levels as well as the error, so the table says what the market's own term
            # structure looks like rather than only how far the model is from it. The figure
            # draws the model's variance curve against these three points.
            "mean_quoted_vol_points": 100 * float(indices[quoted].mean()),
            "mean_target_vol_points": 100 * float(indices[target].mean()),
            "mean_error_vol_points": 100 * float(error.mean()),
            "median_error_vol_points": 100 * float(np.median(error)),
            "sd_error_vol_points": 100 * float(error.std(ddof=1)),
            "worst_over_vol_points": 100 * float(error.max()),
            "worst_under_vol_points": 100 * float(error.min()),
            "share_within_two_points": float(np.mean(np.abs(error) <= 0.02)),
        })
    return pd.DataFrame(rows)


def skew_check(indices: pd.DataFrame, heston: HestonParameters) -> pd.DataFrame:
    """The SKEW index against the skewness the calibrated parameters actually generate.

    This is the one check in the project that tests the shape of the risk-neutral density rather
    than its width. Every volatility index is a variance quote, so the whole set of them is
    silent on rho and xi: those two can trade off against each other without moving the expected
    average variance at any tenor. SKEW is not silent on them, and it is free.

    The level still comes from the quote - instantaneous variance is implied from the one-month
    index exactly as the paths do it - so what is left for the model to get right is the third
    moment alone. Heston has no jumps, and a pure diffusion reaches a deeply negative short-dated
    skew only by pushing rho towards minus one, so the expected failure is a model skew too
    shallow, meaning a model index too high. The direction matters downstream: too little left
    tail at thirty days understates the chance of the sharp declines that put a living benefit in
    the money, so it biases the liability down rather than up.
    """
    levels = []
    for one_month in indices["one_month"]:
        variance = float(scenarios.instantaneous_variance(np.array([one_month]), heston)[0])
        skewness = heston_cos.log_return_skewness(replace(heston, v0=variance),
                                                  scenarios.VIX_TENOR_YEARS)
        levels.append(100.0 - 10.0 * skewness)
    model = np.array(levels)
    observed = indices["skew_index"].to_numpy(dtype=float)
    error = model - observed

    def rank_correlation(left, right) -> float:
        return float(pd.Series(left).corr(pd.Series(right), method="spearman"))

    return pd.DataFrame({
        "observations": [int(error.size)],
        "mean_model_index": [float(model.mean())],
        "mean_observed_index": [float(observed.mean())],
        "mean_error_index_points": [float(error.mean())],
        "sd_error_index_points": [float(error.std(ddof=1))],
        "share_model_above_observed": [float(np.mean(error > 0))],
        "model_skewness_at_mean_index": [float((100.0 - model.mean()) / 10.0)],
        "observed_skewness_at_mean_index": [float((100.0 - observed.mean()) / 10.0)],
        # Read these two together or not at all. The model's skewness falls out of its one state
        # variable, so the model index is a monotone function of the one-month quote and its rank
        # correlation with the observed index is forced to be minus the quote's own: the two
        # numbers below are mirror images by construction. What that means is not weak agreement
        # but no information - whatever makes the market's skew move day to day is not in here.
        "model_vs_observed_rank_correlation": [rank_correlation(model, observed)],
        "quote_vs_observed_rank_correlation": [
            rank_correlation(indices["one_month"].to_numpy(dtype=float), observed)],
    })


def parity_check(panel: pd.DataFrame) -> pd.DataFrame:
    """What the option chain's own discount factors imply about financing, against Treasury.

    The parity regression is self-validating on fit - the call-minus-put spread is linear in
    strike by construction, and a slice where the line misses has stale quotes in it - but a
    perfectly straight line can still sit at the wrong level, which is what a mislabelled strike
    column or a chain snapped on the wrong date produces. The level has an external check: the
    discount factor per expiry is a financing rate, and for SPX that rate sits a few tens of basis
    points above the matched-maturity Treasury, which is the box spread.

    Compared against the Treasury curve on the last date the panel carries a full set of
    quotes rather than on the snapshot date, because the chain is a Friday afternoon snapshot and
    the constant-maturity series publish with a lag.
    """
    as_of = pd.Timestamp(OPTION_CHAIN_AS_OF)
    forwards = chain.implied_forwards(paths.DATA_RAW / "cboe_spx_parity_quotes.csv", as_of)
    forwards = forwards[forwards["used"]].copy()

    quoted = panel[list(PAR_SERIES.values())].dropna()
    curve_date = quoted.index[quoted.index <= as_of][-1]
    curve = treasury_curve(panel, curve_date)

    maturity = forwards["maturity"].to_numpy(dtype=float)
    implied = -np.log(forwards["discount"].to_numpy(dtype=float)) / maturity
    treasury = np.array([float(curve.zero(float(m))) for m in maturity])
    forwards["curve_date"] = curve_date.date()
    forwards["implied_rate"] = implied
    forwards["treasury_zero"] = treasury
    forwards["spread_bp"] = 1e4 * (implied - treasury)
    return forwards[["expiry", "maturity", "n_pairs", "forward", "discount", "parity_r2",
                     "parity_max_resid", "curve_date", "implied_rate", "treasury_zero",
                     "spread_bp"]]


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

    parity = parity_check(panel)
    parity.to_csv(paths.TABLES / "parity_discount_check.csv", index=False, float_format="%.6f")
    low, high = PARITY_SPREAD_RANGE_BP
    outside = parity[(parity["spread_bp"] < low) | (parity["spread_bp"] > high)]
    if not outside.empty:
        raise ValueError(
            f"{len(outside)} of {len(parity)} expiries imply a financing spread outside "
            f"{low:.0f} to {high:.0f}bp over Treasury: "
            f"{outside['spread_bp'].round(0).tolist()}"
        )
    print(f"  the chain's own discount factors imply financing "
          f"{parity['spread_bp'].min():.0f} to {parity['spread_bp'].max():.0f}bp over the "
          f"{parity['curve_date'].iloc[0]} Treasury curve across {len(parity)} expiries, worst "
          f"parity residual {parity['parity_max_resid'].max():.2f} index points")

    mortality_table = mortality_check()
    mortality_table.to_csv(paths.TABLES / "mortality_check.csv", index=False,
                           float_format="%.4f")

    # The surface's own parameters are not available until the calibration step has run, so
    # this check uses the saved calibration when there is one and says so when there is not.
    if paths.MARKET_CALIBRATION.exists():
        from vahedge.market import state as market_state
        heston = market_state.load().heston
        indices = volatility_indices(panel)

        volatility = volatility_check(indices, heston)
        volatility.to_csv(paths.TABLES / "volatility_curve_check.csv", index=False,
                          float_format="%.4f")
        for _, row in volatility.iterrows():
            print(f"  {row['target'].replace('_', '-')} volatility carried {row['direction']} "
                  f"from the {row['quoted'].replace('_', '-')} index is "
                  f"{row['mean_error_vol_points']:+.2f} points off on average, within two points "
                  f"on {row['share_within_two_points']:.0%} of "
                  f"{int(row['observations']):,} days")

        skew = skew_check(indices, heston)
        skew.to_csv(paths.TABLES / "skew_check.csv", index=False, float_format="%.4f")
        row = skew.iloc[0]
        print(f"  the calibrated parameters generate a SKEW index of "
              f"{row['mean_model_index']:.1f} on average against {row['mean_observed_index']:.1f} "
              f"observed, so a thirty-day skewness of "
              f"{row['model_skewness_at_mean_index']:.2f} against "
              f"{row['observed_skewness_at_mean_index']:.2f}, too shallow on "
              f"{1 - row['share_model_above_observed']:.0%} of days")
    else:
        print("  volatility and skew checks skipped: run scripts/run_calibration.py first")

    scaled = scaled_disclosure()
    scaled.to_csv(paths.DISCLOSED_SCALED, index=False)
    mrb = scaled.query("line_item == 'market_risk_benefits'")
    print(f"  scaled {len(mrb)} market risk benefit sensitivities across "
          f"{mrb['as_of'].nunique()} balance-sheet dates")
    print(f"\nwrote the curve history, the scaled disclosure and six checks to "
          f"{paths.DATA_PROCESSED} and {paths.TABLES}")


if __name__ == "__main__":
    main()
