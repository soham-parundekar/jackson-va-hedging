"""Redraw every figure from the committed tables.

Separate from the analysis scripts on purpose. The figures depend on the tables and nothing
else, so they redraw in a couple of seconds whenever a number changes, and a figure can never
be left disagreeing with the table printed beside it in the report. Any figure whose table has
not been written yet is reported as skipped rather than crashing the run.

Usage:  python -m scripts.run_figures
"""

from __future__ import annotations

from vahedge import paths
from vahedge.report import figures


def main() -> None:
    paths.ensure_output_dirs()
    for line in figures.draw_all():
        print(f"  {line}")


if __name__ == "__main__":
    main()
