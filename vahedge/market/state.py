"""One valuation date's market, assembled from the committed data and written down.

Everything downstream - the valuation engine, the Greeks, the proxy, the hedge backtest, the
capital lenses - prices off the same five objects: a discount curve, a Heston surface, a
Hull-White short-rate model, the correlations between them, and the fund mix the contract
holder actually owns. This is where those come from, and it exists so that they come from one
place rather than from whatever a script happened to construct.

The expensive step is the Heston fit: several hundred least-squares iterations over seventeen
hundred quotes, a couple of minutes. So the calibration is saved to JSON under data/processed
and reloaded afterwards. That file is an output of ``scripts/run_calibration.py``, not an
input, and deleting it costs nothing but time.

The rate model is fitted to history rather than to swaptions, and the reason is worth stating
plainly: there is no free swaption surface. An AR(1) on sixty-four years of three-month bills
gives a mean reversion and a volatility that are honest about what they are - real-world
estimates used as risk-neutral parameters - and the curve itself is fitted exactly to today's
Treasuries, so the model reprices bonds regardless. What the historical fit decides is the
width of the rate distribution, which matters for the guarantee and is reported as an
assumption rather than a calibration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths as project_paths
from .chain import load_chain
from .curves import NelsonSiegelSvensson, ZeroCurve, bootstrap, fit_curve
from .heston_cos import HestonParameters, calibrate
from .hull_white import HullWhite, calibrate_historical
from .simulate import Correlations, SubAccountMix

# Treasury constant-maturity series, and the tenor each one is. The three-month bill is left
# out of the curve fit: it is a discount instrument rather than a par bond, and the bootstrap
# treats its input as a par yield.
PAR_SERIES = {1: "DGS1", 2: "DGS2", 3: "DGS3", 5: "DGS5", 7: "DGS7",
              10: "DGS10", 20: "DGS20", 30: "DGS30"}
SHORT_RATE_SERIES = "DGS3MO"
TRADING_DAYS = 252

# Separate-account split at 31 December 2025, from the FY2025 10-K: equity 171,046, bond
# 19,711, balanced 43,317, money market 2,332, of 236,406, in millions.
BOOK_MIX = SubAccountMix(equity=0.7235, bond=0.0834, balanced=0.1832, money_market=0.0099)


@dataclass(frozen=True)
class MarketCalibration:
    """Everything one valuation date is priced off, and the diagnostics to judge it by."""

    as_of: pd.Timestamp
    curve: NelsonSiegelSvensson
    heston: HestonParameters
    mean_reversion: float
    rate_vol: float
    correlations: Correlations
    mix: SubAccountMix
    diagnostics: dict

    def hull_white(self) -> HullWhite:
        return HullWhite(a=self.mean_reversion, sigma=self.rate_vol, curve=self.curve)

    def to_json(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "as_of": self.as_of.strftime("%Y-%m-%d"),
            "curve": {name: float(getattr(self.curve, name))
                      for name in ("beta0", "beta1", "beta2", "beta3", "tau1", "tau2")},
            "heston": {name: float(getattr(self.heston, name))
                       for name in ("v0", "kappa", "theta", "xi", "rho")},
            "mean_reversion": float(self.mean_reversion),
            "rate_vol": float(self.rate_vol),
            "correlations": {"equity_variance": float(self.correlations.equity_variance),
                             "equity_rate": float(self.correlations.equity_rate)},
            "mix": {name: float(getattr(self.mix, name))
                    for name in ("equity", "bond", "balanced", "money_market",
                                 "bond_duration", "balanced_equity_share", "tracking_error")},
            "diagnostics": _plain(self.diagnostics),
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path

    @classmethod
    def from_json(cls, path) -> "MarketCalibration":
        payload = json.loads(Path(path).read_text())
        return cls(
            as_of=pd.Timestamp(payload["as_of"]),
            curve=NelsonSiegelSvensson(**payload["curve"]),
            heston=HestonParameters(**payload["heston"]),
            mean_reversion=payload["mean_reversion"],
            rate_vol=payload["rate_vol"],
            correlations=Correlations(**payload["correlations"]),
            mix=SubAccountMix(**payload["mix"]),
            diagnostics=payload["diagnostics"],
        )


def _plain(value):
    """JSON cannot hold numpy scalars or timestamps, and the diagnostics are full of both."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def treasury_curve(panel: pd.DataFrame, as_of: pd.Timestamp) -> NelsonSiegelSvensson:
    """Par yields on the date, bootstrapped to zeros and fitted.

    The fit is to the quoted tenors with flat anchors past thirty years, because an
    unconstrained Svensson curve fitted to three decimal places inside thirty years will do
    anything it likes at fifty, and a contract written on a seventy-year-old still has a
    forty-year tail. See fit_curve.
    """
    row = panel.loc[as_of]
    missing = [name for name in PAR_SERIES.values() if not np.isfinite(row.get(name, np.nan))]
    if missing:
        raise ValueError(f"{as_of.date()} is missing Treasury quotes for {missing}")
    par = np.array([row[name] for name in PAR_SERIES.values()], dtype=float) / 100.0
    return fit_curve(bootstrap(list(PAR_SERIES), par))


def equity_rate_correlation(panel: pd.DataFrame, lookback_years: int = 10) -> tuple[float, dict]:
    """Daily index return against the daily change in the ten-year yield.

    This is the correlation that decides whether a sell-off arrives with rates falling, which
    is the state this liability is worst in: the guarantee moves further into the money at the
    same time as the discount rate drops. It is not stable across regimes - it was strongly
    negative through the 2010s and turned positive in 2022 - so it is estimated on a fixed
    recent window and the sensitivity to that choice is a robustness test rather than a claim.
    """
    frame = panel[["SP500", "DGS10"]].dropna()
    if lookback_years:
        frame = frame.loc[frame.index >= frame.index.max() - pd.DateOffset(years=lookback_years)]
    equity = np.diff(np.log(frame["SP500"].to_numpy(dtype=float)))
    rates = np.diff(frame["DGS10"].to_numpy(dtype=float)) / 100.0
    rho = float(np.corrcoef(equity, rates)[0, 1])
    return rho, {
        "observations": int(equity.size),
        "start": frame.index.min(),
        "end": frame.index.max(),
        "lookback_years": lookback_years,
    }


def calibrate_market(
    as_of: str | pd.Timestamp = "2025-12-31",
    mix: SubAccountMix = BOOK_MIX,
    lookback_years: int = 10,
) -> MarketCalibration:
    """Fit the whole market state from the committed raw data.

    Runs in minutes rather than seconds, almost all of it in the Heston fit. Callers that only
    need the answer should use ``load`` and let ``scripts/run_calibration.py`` do this once.
    """
    as_of = pd.Timestamp(as_of)
    panel = pd.read_csv(project_paths.FRED_PANEL, parse_dates=["date"]).set_index("date")
    curve = treasury_curve(panel, as_of)

    history = pd.read_csv(project_paths.DATA_RAW / "fred_long_rate_history.csv",
                          comment="#", parse_dates=["date"])
    short_rates = history[SHORT_RATE_SERIES].dropna().to_numpy(dtype=float) / 100.0
    mean_reversion, rate_vol, rate_diagnostics = calibrate_historical(
        short_rates, dt=1.0 / TRADING_DAYS
    )

    snapshot = load_chain(project_paths.OPTION_CHAIN,
                          project_paths.DATA_RAW / "cboe_spx_parity_quotes.csv")
    heston, heston_diagnostics = calibrate(snapshot.quotes)

    rho_equity_rate, correlation_diagnostics = equity_rate_correlation(panel, lookback_years)

    return MarketCalibration(
        as_of=as_of,
        curve=curve,
        heston=heston,
        mean_reversion=mean_reversion,
        rate_vol=rate_vol,
        correlations=Correlations(equity_variance=heston.rho, equity_rate=rho_equity_rate),
        mix=mix,
        diagnostics={
            "chain_as_of": snapshot.as_of,
            "chain_spot": snapshot.spot,
            "chain_quotes": int(snapshot.quotes.shape[0]),
            "chain_expiries": int(snapshot.quotes["maturity"].nunique()),
            "chain_max_maturity": float(snapshot.quotes["maturity"].max()),
            "heston": heston_diagnostics,
            "short_rate": rate_diagnostics,
            "equity_rate_correlation": correlation_diagnostics,
            "curve_zero_10y": float(curve.zero(10.0)),
            "curve_zero_30y": float(curve.zero(30.0)),
        },
    )


def load(path=None) -> MarketCalibration:
    """The saved calibration, with a message that says how to make one if it is not there."""
    path = Path(path or project_paths.MARKET_CALIBRATION)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; run `python -m scripts.run_calibration` to build it"
        )
    return MarketCalibration.from_json(path)


def with_tracking_error(calibration: MarketCalibration, tracking_error: float):
    """The same market with a different sub-account basis, for the basis-risk experiment."""
    return replace(calibration, mix=replace(calibration.mix, tracking_error=tracking_error))
