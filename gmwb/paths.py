"""Repository paths, resolved from this file rather than the working directory.

Scripts get run from the repository root, from ``make``, and from a notebook a
directory down. Anchoring on ``__file__`` means all three find the same data.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIG = ROOT / "config"
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"

FRED_PANEL = DATA_RAW / "fred_daily_panel.csv"
MORTALITY_TABLE = DATA_RAW / "soa_2012_iam_g2.csv"
DISCLOSED_SENSITIVITIES = DATA_RAW / "jackson_disclosed_sensitivities.csv"
BOOK_STATISTICS = DATA_RAW / "jackson_book_statistics.csv"

MARKET_PANEL = DATA_PROCESSED / "market_panel.csv"
ZERO_CURVES = DATA_PROCESSED / "zero_curves.csv"


def ensure_output_dirs() -> None:
    for directory in (DATA_PROCESSED, FIGURES, TABLES):
        directory.mkdir(parents=True, exist_ok=True)
