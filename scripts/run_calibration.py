"""Fit the market state once and write it down, with the diagnostics needed to judge it.

Everything downstream reads data/processed/market_calibration.json. This is the only step that
builds it, and it is slow enough - a couple of minutes, almost all of it in the Heston fit -
that running it from every script would be the difference between a project that can be
explored and one that cannot.

Three tables come out alongside it, and each answers a question the calibration alone does not.
The surface fit says whether the five Heston parameters actually reproduce the quotes they were
fitted to, by expiry, in volatility points rather than in price. The curve check reprices the
Treasuries the curve was fitted to and shows what the fitted curve does past thirty years,
which is where an unconstrained Svensson fit goes wrong and where a forty-year liability lives.
The rate table records the AR(1) estimate and whether the mean-reversion floor bound.

Usage:  python -m scripts.run_calibration
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vahedge import paths
from vahedge.market import state as market_state
from vahedge.market.chain import fit_report
from vahedge.market.curves import bootstrap
from vahedge.market.state import PAR_SERIES


def surface_fit(calibration) -> pd.DataFrame:
    """Model against market volatility, collapsed to one row per expiry.

    Reported in volatility points because that is the unit the fit is weighted in and the unit
    anyone reads a surface in: a price error of eighty cents means nothing without knowing
    which option it was on. The bias column is what catches a fit that has traded the front of
    the surface against the back, which a single pooled error hides.
    """
    from vahedge.market.chain import load_chain

    snapshot = load_chain(paths.OPTION_CHAIN, paths.DATA_RAW / "cboe_spx_parity_quotes.csv")
    quotes = fit_report(snapshot, calibration.heston)
    quotes["squared"] = quotes["vol_error"] ** 2
    by_expiry = quotes.groupby("maturity").agg(
        n_quotes=("vol_error", "size"),
        mean_market_vol=("market_vol", "mean"),
        bias_vol=("vol_error", "mean"),
        rmse_vol=("squared", lambda s: float(np.sqrt(s.mean()))),
        worst_vol=("vol_error", lambda s: float(s.abs().max())),
    ).reset_index()
    return by_expiry


def curve_check(calibration, as_of) -> pd.DataFrame:
    """The fitted curve against the par yields it came from, and what it does past the data.

    The round trip is the test that matters: bootstrap the quoted par yields to zeros, fit the
    smooth curve, then ask the smooth curve what par yield it implies at each quoted tenor. A
    fit that reprices its own inputs to a basis point is doing its job. The rows beyond thirty
    years have no quote to compare against and are there to be looked at: a Svensson curve
    fitted only inside thirty years is free to send the fifty-year forward anywhere, and an
    earlier version of this sent it to minus twenty-six per cent.
    """
    panel = pd.read_csv(paths.FRED_PANEL, parse_dates=["date"]).set_index("date")
    quoted = np.array([panel.loc[as_of, name] for name in PAR_SERIES.values()]) / 100.0
    tenors = np.array(list(PAR_SERIES), dtype=float)
    boot = bootstrap(list(PAR_SERIES), quoted)

    rows = []
    for tenor, par in zip(tenors, quoted):
        rows.append({
            "tenor_years": tenor,
            "quoted_par": par,
            "bootstrap_zero": float(boot.zero(tenor)),
            "fitted_zero": float(calibration.curve.zero(tenor)),
            "fitted_par": float(calibration.curve.par_equivalent(tenor)),
            "par_error_bp": 1e4 * (float(calibration.curve.par_equivalent(tenor)) - par),
        })
    for tenor in (35.0, 40.0, 50.0, 60.0):
        rows.append({
            "tenor_years": tenor,
            "quoted_par": np.nan,
            "bootstrap_zero": np.nan,
            "fitted_zero": float(calibration.curve.zero(tenor)),
            "fitted_par": float(calibration.curve.par_equivalent(tenor)),
            "par_error_bp": np.nan,
        })
    return pd.DataFrame(rows)


def rate_model(calibration) -> pd.DataFrame:
    diagnostics = calibration.diagnostics["short_rate"]
    return pd.DataFrame([{
        "mean_reversion": calibration.mean_reversion,
        "rate_vol": calibration.rate_vol,
        "half_life_years": float(np.log(2.0) / calibration.mean_reversion),
        "ten_year_sd_bp": 1e4 * calibration.rate_vol * np.sqrt(
            (1.0 - np.exp(-2.0 * calibration.mean_reversion * 10.0))
            / (2.0 * calibration.mean_reversion)
        ),
        **{str(k): v for k, v in diagnostics.items()},
    }])


def main() -> None:
    paths.ensure_output_dirs()
    calibration = market_state.calibrate_market()
    written = calibration.to_json(paths.MARKET_CALIBRATION)

    surface = surface_fit(calibration)
    curve = curve_check(calibration, calibration.as_of)
    rates = rate_model(calibration)
    surface.to_csv(paths.TABLES / "surface_fit.csv", index=False)
    curve.to_csv(paths.TABLES / "curve_fit.csv", index=False)
    rates.to_csv(paths.TABLES / "rate_model_fit.csv", index=False)

    heston = calibration.heston
    print(f"wrote {written}")
    print(f"  Heston   v0={heston.v0:.4f} kappa={heston.kappa:.3f} theta={heston.theta:.4f} "
          f"xi={heston.xi:.3f} rho={heston.rho:.3f}")
    print(f"  Feller   2*kappa*theta - xi^2 = {2*heston.kappa*heston.theta - heston.xi**2:+.3f}")
    quotes = surface["n_quotes"]
    pooled_rmse = float(np.sqrt((surface["rmse_vol"] ** 2 * quotes).sum() / quotes.sum()))
    print(f"  surface  {int(quotes.sum())} quotes, rmse {1e2 * pooled_rmse:.2f} vol points")
    print(f"  curve    worst par error {curve['par_error_bp'].abs().max():.2f}bp, "
          f"50y zero {float(calibration.curve.zero(50.0)):.4f}")
    print(f"  rates    a={calibration.mean_reversion:.4f} sigma={calibration.rate_vol:.5f}")
    print(f"  equity-rate correlation {calibration.correlations.equity_rate:+.3f}")


if __name__ == "__main__":
    main()
