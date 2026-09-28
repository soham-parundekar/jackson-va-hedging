"""Market data: load the FRED panel, clean it, and turn a date into model inputs.

Everything comes from one daily panel pulled from FRED (see docs/data_sources.md
for series identifiers and the reason each one is there). The awkward parts of the
panel are all calendar related. Treasury yields are published on federal business
days, the S&P 500 on NYSE trading days, and the volatility indices occasionally lag
by a day or two. Columbus Day and Veterans Day are the reliable offenders: the bond
market is shut and the stock market is open.

The rule used here is to key off S&P 500 observations, since those are the dates a
hedge actually gets rebalanced, and to carry rates and volatility forward by at
most a few days. A gap longer than that is a data problem, so it raises rather than
quietly filling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import paths
from .curves import ParCurveBuilder
from .volatility import VolTermStructure, realised_vol

PAR_YIELD_COLUMNS = {
    1: "DGS1",
    2: "DGS2",
    3: "DGS3",
    5: "DGS5",
    7: "DGS7",
    10: "DGS10",
    20: "DGS20",
    30: "DGS30",
}

MAX_STALENESS_DAYS = 5


@dataclass(frozen=True)
class MarketState:
    """Everything the valuation needs to know about one date."""

    date: pd.Timestamp
    spot: float
    curve_builder: ParCurveBuilder
    vol: VolTermStructure
    implied_vol_3m: float
    vix: float
    credit_spread: float


def load_panel(path=None) -> pd.DataFrame:
    """Read the raw panel, coerce to numeric, and forward-fill the slow-moving series."""
    frame = pd.read_csv(path or paths.FRED_PANEL, parse_dates=["date"])
    frame = frame.set_index("date").sort_index()
    frame = frame.apply(pd.to_numeric, errors="coerce")

    if frame.index.has_duplicates:
        raise ValueError("duplicate dates in the FRED panel")

    fillable = list(PAR_YIELD_COLUMNS.values()) + ["VIXCLS", "VXVCLS", "BAA10Y", "AAA10Y", "DTB3"]
    for column in fillable:
        if column not in frame.columns:
            raise ValueError(f"expected column {column} in the FRED panel")
    frame[fillable] = frame[fillable].ffill(limit=MAX_STALENESS_DAYS)

    if "SP500" not in frame.columns:
        raise ValueError("expected column SP500 in the FRED panel")
    return frame


def equity_dates(frame: pd.DataFrame, start=None, end=None) -> pd.DatetimeIndex:
    """Dates with an S&P 500 close, inside the requested window."""
    dates = frame.index[frame["SP500"].notna()]
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]
    return pd.DatetimeIndex(dates)


def long_run_vol(frame: pd.DataFrame, lookback_years: int, risk_margin: float) -> float:
    """Realised volatility of the equity index over the available history, plus the
    explicit risk margin Jackson says its long-run level carries."""
    prices = frame["SP500"].dropna()
    window = min(len(prices) - 1, int(lookback_years * 252))
    return realised_vol(prices.to_numpy(), window_days=window) + risk_margin


def state_at(frame: pd.DataFrame, date, cfg: dict, long_run: float) -> MarketState:
    """Build the curve and volatility inputs for one date."""
    date = pd.Timestamp(date)
    if date not in frame.index:
        raise KeyError(f"{date.date()} is not in the panel")
    row = frame.loc[date]

    spot = row["SP500"]
    if not np.isfinite(spot):
        raise ValueError(f"no S&P 500 close on {date.date()}")

    tenors = list(cfg["curve"]["par_tenors"])
    yields = np.array([row[PAR_YIELD_COLUMNS[t]] for t in tenors], dtype=float) / 100.0
    if np.any(~np.isfinite(yields)):
        missing = [t for t, y in zip(tenors, yields) if not np.isfinite(y)]
        raise ValueError(f"missing par yields {missing} on {date.date()}")

    implied_3m = row["VXVCLS"]
    vix = row["VIXCLS"]
    if not np.isfinite(implied_3m):
        raise ValueError(f"no 3-month implied volatility on {date.date()}")

    # The curve is fitted to the 3-month index alone. It is the longer of the two free
    # implied observations and therefore the more informative one for a liability whose
    # cash flows run decades out, and a one-parameter curve fitted to both would match
    # neither. The 30-day index is carried through as an independent check: the spot
    # volatility the fitted curve implies at one month can be compared with it, and
    # build_dataset reports that comparison.
    vcfg = cfg["volatility"]
    vol = VolTermStructure.from_implied(
        {float(vcfg["mid_tenor_years"]): float(implied_3m) / 100.0},
        long_run_level=float(long_run),
        grade_to=float(vcfg["grade_to_years"]),
        residual=float(vcfg["grading_residual"]),
    )

    spread_series = cfg["accounting"]["spread_series"]
    spread = row[spread_series]
    spread = float(spread) / 100.0 if np.isfinite(spread) else float("nan")

    return MarketState(
        date=date,
        spot=float(spot),
        curve_builder=ParCurveBuilder(tenors, yields, frequency=cfg["curve"]["coupon_frequency"]),
        vol=vol,
        implied_vol_3m=float(implied_3m) / 100.0,
        vix=float(vix) if np.isfinite(vix) else float("nan"),
        credit_spread=spread,
    )
