"""One place to assemble the objects every script needs.

Each analysis script starts the same way: read the parameter file, load the market
panel, fix the long-run volatility level, build the contract and the mortality basis,
draw the normals, and calibrate the fee attribution percentage at issue. Doing that
once here keeps the scripts about their own analysis and guarantees that the
attribution percentage and the random draws are identical across them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as config_module
from . import market, mortality as mortality_module, paths
from .contract import GmwbContract
from .engine import (
    RiderValuation,
    calibrate_attribution,
    make_normals,
    projection_years,
    value_rider,
)
from .market import MarketState
from .mortality import MortalityBasis


@dataclass
class Session:
    cfg: dict
    panel: pd.DataFrame
    long_run_vol: float
    contract: GmwbContract
    mortality: MortalityBasis
    normals: np.ndarray
    state: MarketState
    at_issue: RiderValuation
    fee_attribution: float

    @property
    def valuation_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.cfg["valuation_date"])

    @property
    def n_years(self) -> int:
        return projection_years(self.contract, int(self.cfg["simulation"]["max_age"]))

    def valuation_state(self, **overrides) -> dict:
        """Arguments describing the policy and the discounting basis, for the engine."""
        state = {
            "fee_attribution": self.fee_attribution,
            "max_age": int(self.cfg["simulation"]["max_age"]),
        }
        state.update(overrides)
        return state


def start(config_path=None, n_paths: int | None = None) -> Session:
    cfg = config_module.load(config_path)
    panel = market.load_panel()
    paths.ensure_output_dirs()

    vcfg = cfg["volatility"]
    long_run = market.long_run_vol(
        panel, int(vcfg["long_run_lookback_years"]), float(vcfg["long_run_risk_margin"])
    )

    contract = GmwbContract.from_config(cfg)
    valuation_date = pd.Timestamp(cfg["valuation_date"])
    basis = mortality_module.load(
        contract.issue_age,
        valuation_date.year,
        float(cfg["contract"]["sex_mix"]["male"]),
    )

    n_years = projection_years(contract, int(cfg["simulation"]["max_age"]))
    sim = cfg["simulation"]
    normals = make_normals(
        int(n_paths or sim["n_paths"]),
        n_years,
        int(sim["seed"]),
        bool(sim["antithetic"]),
    )

    state = market.state_at(panel, valuation_date, cfg, long_run)
    at_issue = value_rider(
        contract,
        state.curve_builder.build(),
        state.vol,
        basis,
        normals,
        max_age=int(sim["max_age"]),
    )
    attribution = calibrate_attribution(at_issue)

    return Session(
        cfg=cfg,
        panel=panel,
        long_run_vol=long_run,
        contract=contract,
        mortality=basis,
        normals=normals,
        state=state,
        at_issue=at_issue,
        fee_attribution=attribution,
    )


def write_table(frame: pd.DataFrame, name: str, float_format: str = "%.4f") -> None:
    """Save a results table to reports/tables as CSV.

    CSV only, deliberately. The scripts already print a readable version to the console, and a
    second fixed-width copy of every table on disk is two files to keep in step for no gain.
    """
    paths.ensure_output_dirs()
    frame.to_csv(paths.TABLES / f"{name}.csv", index=False, float_format=float_format)
