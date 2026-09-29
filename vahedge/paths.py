"""Repository paths, resolved from this file rather than the working directory.

Scripts get run from the repository root, from ``make``, and from a notebook a directory
down. Anchoring on ``__file__`` means all three find the same data.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIG = ROOT / "config"
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
REFERENCES = ROOT / "references"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"

# Raw inputs. Everything here was retrieved once through a browser and committed, because
# the sandbox this was built in cannot reach sec.gov, FRED or Cboe from a script. See
# docs/data_sources.md and references/README.md.
FRED_PANEL = DATA_RAW / "fred_daily_panel.csv"
FRED_FINANCING = DATA_RAW / "fred_financing_rates.csv"
CBOE_VOL_PANEL = DATA_RAW / "cboe_vix6m_skew.csv"
OPTION_CHAIN = DATA_RAW / "cboe_spx_option_chain.csv"
MORTALITY_TABLE = DATA_RAW / "soa_2012_iam_g2.csv"
DISCLOSED_SENSITIVITIES = DATA_RAW / "jackson_disclosed_sensitivities.csv"
DERIVATIVE_SENSITIVITIES = DATA_RAW / "jackson_derivative_sensitivities.csv"
BOOK_STATISTICS = DATA_RAW / "jackson_book_statistics.csv"
RIDER_TERMS = DATA_RAW / "jackson_rider_terms.csv"
XBRL_QUARTERLY = DATA_RAW / "jackson_xbrl_quarterly.csv"

# Built by `make data`.
MARKET_PANEL = DATA_PROCESSED / "market_panel.csv"
ZERO_CURVES = DATA_PROCESSED / "zero_curves.csv"
NSS_PARAMETERS = DATA_PROCESSED / "nss_parameters.csv"
CLEAN_CHAIN = DATA_PROCESSED / "option_chain_clean.csv"
DISCLOSED_SCALED = DATA_PROCESSED / "disclosed_scaled.csv"

# Built by the calibration step and read by everything downstream.
HESTON_FIT = DATA_PROCESSED / "heston_calibration.json"
HULL_WHITE_FIT = DATA_PROCESSED / "hull_white_calibration.json"


def ensure_output_dirs() -> None:
    for directory in (DATA_INTERIM, DATA_PROCESSED, FIGURES, TABLES):
        directory.mkdir(parents=True, exist_ok=True)
