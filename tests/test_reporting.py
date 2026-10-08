"""The reporting lens: the spread loading, the three marks, and the net income / OCI split.

The arithmetic here is all checkable by hand, which matters because an accounting split that is
wrong by a sign produces a plausible earnings series rather than an error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tests.checks import approx, raises
from vahedge.capital import reporting
from vahedge.market import scenarios
from vahedge.market.scenarios import DailyPath
from vahedge.market.simulate import MarketPaths


def paths(n_paths: int = 3, n_years: int = 4) -> MarketPaths:
    """Paths whose only interesting field is the discount factor."""
    ones = np.ones((n_paths, n_years))
    discount = np.cumprod(np.full((n_paths, n_years), 0.96), axis=1)
    return MarketPaths(
        fund_growth=ones * 1.05, index_growth=ones * 1.06, discount=discount,
        short_rate=ones * 0.04, variance=ones * 0.04, zero_10y=ones * 0.03,
        realised_variance=ones * 0.04, seed=1, antithetic=False,
    )


def test_the_spread_loads_only_the_discount_factor():
    original = paths()
    loaded = reporting.discounted_at_spread(original, 0.02)
    years = np.arange(1, original.n_years + 1, dtype=float)
    # approx on the left: NumPy's own __eq__ gets first refusal and would return an array.
    assert approx(original.discount * np.exp(-0.02 * years)[None, :]) == loaded.discount
    for field in ("fund_growth", "index_growth", "short_rate", "variance", "zero_10y"):
        assert getattr(loaded, field) is getattr(original, field)


def test_a_zero_spread_returns_the_paths_untouched():
    original = paths()
    assert reporting.discounted_at_spread(original, 0.0) is original


def test_the_loading_compounds_by_year_not_by_step():
    """A year-four cash flow is discounted four years of spread, not one."""
    loaded = reporting.discounted_at_spread(paths(), 0.03)
    ratio = loaded.discount[0] / paths().discount[0]
    assert approx(np.exp(-0.03 * np.array([1.0, 2.0, 3.0, 4.0]))) == ratio


class FlatFit:
    """A fit whose value is a constant, so interpolation across the grid is readable."""

    def __init__(self, value: float):
        self.value = value

    def greeks_at(self, **kwargs):
        return {"value": np.array([self.value])}


def basis(economic=10.0, at_spread=(12.0, 11.0, 9.5)) -> reporting.ReportingBasis:
    """Reporting liability falling as the spread rises, which is what discounting more does."""
    return reporting.ReportingBasis(
        economic=FlatFit(economic),
        reporting=dict(zip(reporting.SPREAD_GRID, [FlatFit(v) for v in at_spread])),
        attribution=1.0,
    )


def state(spread: float) -> dict:
    return {
        "years_since_issue": 5.0, "account_value": 90.0, "benefit_base": 100.0,
        "variance": 0.04, "zero_10y": 0.03, "own_credit_spread": spread,
    }


def test_the_three_marks_come_back_separately():
    out = basis().marks(**state(0.0))
    assert out["economic"] == approx(10.0)
    assert out["reporting_ex_own_credit"] == approx(12.0)
    assert out["reporting"] == approx(12.0)


def test_the_no_own_credit_basis_is_the_zero_spread_member():
    """By construction rather than by a second calculation, which is why the grid starts at zero."""
    for spread in (0.0, 0.015, 0.03, 0.045):
        out = basis().marks(**state(spread))
        assert out["reporting_ex_own_credit"] == approx(12.0)


def test_the_spread_is_interpolated_across_the_grid():
    """Read off the grid rather than written out, so retuning the levels to a different spread
    range does not silently turn this into a test of the old ones."""
    low, mid = reporting.SPREAD_GRID[0], reporting.SPREAD_GRID[1]
    out = basis().marks(**state(0.5 * (low + mid)))
    assert out["reporting"] == approx(11.5)


def test_a_spread_outside_the_grid_is_held_not_extrapolated():
    """np.interp holds the end value, which is the conservative behaviour for a fitted object."""
    assert basis().marks(**state(0.20))["reporting"] == approx(9.5)


def test_own_credit_reduces_the_reported_liability():
    """Discounting a liability at a higher rate makes it smaller, so the adjustment is negative."""
    out = basis().marks(**state(0.045))
    assert out["reporting"] < out["reporting_ex_own_credit"]


def ledger(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame({
        "policy_year": np.full(n, 5.0),
        "account_value": np.full(n, 90.0),
        "benefit_base": np.full(n, 100.0),
        "variance": np.full(n, 0.04),
        "zero_10y": np.full(n, 0.03),
        "hedge_mark": np.zeros(n),
        "cash": np.zeros(n),
    }, index=pd.date_range("2020-01-01", periods=n, freq="D"))


def spread_path(values) -> DailyPath:
    values = np.asarray(values, dtype=float)
    n = values.size
    return DailyPath(
        dates=pd.date_range("2020-01-01", periods=n, freq="D"),
        index=np.ones(n), fund=np.ones(n), curves=[None] * n,
        zero_10y=np.full(n, 0.03), implied_vol=np.full(n, 0.2),
        variance=np.full(n, 0.04), cash_rate=np.full(n, 0.01),
        own_credit_spread=values, label="test",
    )


def test_the_margin_and_the_own_credit_adjustment_separate():
    marked = reporting.mark(ledger(), spread_path([0.0, 0.0, 0.0, 0.0]), basis())
    assert marked["risk_margin"].iloc[0] == approx(2.0)          # 12 reporting less 10 economic
    assert marked["own_credit_adjustment"].iloc[0] == approx(0.0)


def test_a_widening_spread_goes_to_oci_and_not_to_net_income():
    """The whole of the accounting rule, in two rows.

    Nothing moves but the spread. The economic liability is unchanged, the reporting basis before
    own credit is unchanged, so net income is flat and the entire movement is the OCI line.
    """
    marked = reporting.mark(ledger(2), spread_path([0.0, reporting.SPREAD_GRID[1]]), basis())
    assert marked["economic_pnl"].iloc[1] == approx(0.0)
    assert marked["net_income"].iloc[1] == approx(0.0)
    # Liability fell from 12.0 to 11.0, which is a gain, so OCI is positive.
    assert marked["oci"].iloc[1] == approx(1.0)
    assert marked["comprehensive_income"].iloc[1] == approx(1.0)


def test_comprehensive_income_is_the_two_lines_added():
    rng = np.random.default_rng(3)
    marked = reporting.mark(ledger(6), spread_path(rng.uniform(0.0, 0.04, 6)), basis())
    assert np.allclose(marked["comprehensive_income"],
                       marked["net_income"] + marked["oci"])


def test_the_mark_refuses_a_path_with_no_spread():
    bare = spread_path([0.0, 0.0])
    bare = DailyPath(**{**bare.__dict__, "own_credit_spread": None})
    with raises(ValueError, match="own-credit spread"):
        reporting.mark(ledger(2), bare, basis())


def test_the_mark_refuses_a_length_mismatch():
    with raises(ValueError, match="rows"):
        reporting.mark(ledger(4), spread_path([0.0, 0.0]), basis())


def test_the_summary_scales_and_reports_the_multiple():
    marked = reporting.mark(ledger(3), spread_path([0.0, 0.01, 0.02]), basis())
    out = reporting.summarise(marked, account_value=100.0)
    assert out["days"] == 3.0
    assert out["mean_risk_margin_pct"] == approx(0.02)
    # Economic P&L is identically zero here, so the multiple is undefined rather than infinite.
    assert np.isnan(out["net_income_sd_multiple"])
    assert out["oci_total_pct"] > 0.0


def test_the_spread_grid_brackets_the_replay_window():
    """BAA10Y runs 1.36% to 4.31% across the replay window and the own-credit share of it is what
    the liability is discounted at, so the grid has to reach past that rather than past the raw
    index. Derived from the share so that retuning either one keeps the test honest."""
    assert min(reporting.SPREAD_GRID) == 0.0
    assert max(reporting.SPREAD_GRID) >= scenarios.OWN_CREDIT_SHARE * 0.0431
