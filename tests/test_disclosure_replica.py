"""The two pieces of the disclosure replica that are logic rather than arithmetic.

The fourth-quarter derivation and the p-value both decide how a result reads, and both are easy
to get subtly wrong in a way that produces a plausible table.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from tests.checks import approx
from scripts import run_disclosure_replica as replica


def test_the_p_value_matches_a_known_case():
    """A correlation of 0.5 on 30 observations sits just inside five per cent."""
    p = replica.two_sided_p(0.5, 30)
    assert 0.003 < p < 0.02


def test_the_p_value_rises_as_the_sample_shrinks():
    """The same coefficient on four observations says nothing, which is the point of reporting it."""
    assert replica.two_sided_p(-0.821, 4) > 0.2
    assert replica.two_sided_p(-0.821, 40) < 0.001


def test_the_approximation_errs_conservatively_on_the_rows_the_finding_rests_on():
    """Fisher's z stands in for the exact t survival function, and its direction is not uniform.

    The filed series is the only part of this comparison that is evidence rather than
    arithmetic, so what matters is that the approximation does not manufacture significance
    there. It does the opposite on both filed rows. The exact two-sided values are computed here
    from the t statistic by Simpson's rule on the t density, which needs nothing the repository
    does not already import.
    """
    def exact(r: float, n: int) -> float:
        t = abs(r) * np.sqrt((n - 2) / (1.0 - r * r))
        df = n - 2
        log_norm = (math.lgamma((df + 1) / 2) - math.lgamma(df / 2)
                    - 0.5 * math.log(df * math.pi))
        # The tail beyond t, out to where the density is dead, on an even grid.
        upper = t + 60.0
        steps = 200_001
        x = np.linspace(t, upper, steps)
        density = np.exp(log_norm - (df + 1) / 2 * np.log1p(x * x / df))
        weights = np.ones(steps)
        weights[1:-1:2], weights[2:-1:2] = 4.0, 2.0
        return float(2 * (upper - t) / (3 * (steps - 1)) * np.sum(weights * density))

    # Sanity on the quadrature before it is used to judge anything: a correlation of 0.5 on 30
    # observations is a textbook case, two-sided p just under five per cent.
    assert exact(0.5, 30) == approx(0.0049, abs=5e-4)

    for correlation, periods in ((0.006387, 18), (-0.821236, 4)):
        assert replica.two_sided_p(correlation, periods) > exact(correlation, periods)

    # And it is optimistic where the coefficient is strong, which is the half of the split the
    # docstring has to keep stating, since both of those rows are zero to any reading.
    assert replica.two_sided_p(-0.985827, 41) < exact(-0.985827, 41)


def test_the_p_value_refuses_a_sample_too_small_to_speak():
    for periods in (0, 1, 2, 3):
        assert np.isnan(replica.two_sided_p(0.9, periods))


def test_the_p_value_handles_a_perfect_correlation():
    """A degenerate coefficient would divide by zero in Fisher's transform."""
    assert np.isnan(replica.two_sided_p(1.0, 20))
    assert np.isnan(replica.two_sided_p(-1.0, 20))
    assert np.isnan(replica.two_sided_p(float("nan"), 20))


def test_a_zero_correlation_is_a_p_value_of_one():
    assert replica.two_sided_p(0.0, 25) == approx(1.0)


def test_the_detectable_threshold_is_the_one_the_script_quotes():
    """At eighteen observations the five per cent threshold is about 0.47."""
    assert float(np.tanh(1.96 / np.sqrt(18 - 3.0))) == approx(0.47, abs=0.01)


def test_offset_stats_signs_a_perfect_hedge_as_one():
    liability = pd.Series([10.0, -4.0, 6.0], index=pd.to_datetime(
        ["2021-03-31", "2021-06-30", "2021-09-30"]))
    out = replica.offset_stats(liability, -liability, "test")
    assert out["offset_ratio"] == approx(1.0)
    assert out["correlation"] == approx(-1.0)
    assert out["share_opposite_sign"] == approx(1.0)
    assert out["residual_total"] == approx(0.0)


def test_offset_stats_reports_too_few_periods_rather_than_a_number():
    liability = pd.Series([1.0, 2.0], index=pd.to_datetime(["2021-03-31", "2021-06-30"]))
    out = replica.offset_stats(liability, -liability, "test")
    assert out["periods"] == 2.0
    assert "correlation" not in out


def test_offset_stats_aligns_on_the_index_rather_than_by_position():
    """A series missing a quarter must drop that quarter, not shift the other side onto it."""
    dates = pd.to_datetime(["2021-03-31", "2021-06-30", "2021-09-30", "2021-12-31"])
    liability = pd.Series([10.0, -4.0, 6.0, 2.0], index=dates)
    hedge = pd.Series([-10.0, 4.0, -6.0], index=dates[[0, 1, 2]])
    out = replica.offset_stats(liability, hedge, "test")
    assert out["periods"] == 3.0
    assert out["offset_ratio"] == approx(1.0)


def test_the_fourth_quarter_is_derived_from_the_year():
    """The 10-K reports the year, so Q4 is the year less the three tagged quarters."""
    rows = []
    for start, end, val in [("2022-01-01", "2022-03-31", 100.0),
                            ("2022-04-01", "2022-06-30", 200.0),
                            ("2022-07-01", "2022-09-30", 50.0)]:
        rows.append({"concept": "X", "period_type": "quarter", "start": start, "end": end,
                     "val_usd": val * 1e6, "form": "10-Q", "filed": "2022-11-01"})
    rows.append({"concept": "X", "period_type": "year", "start": "2022-01-01",
                 "end": "2022-12-31", "val_usd": 500.0 * 1e6, "form": "10-K",
                 "filed": "2023-02-01"})
    frame = pd.DataFrame(rows)
    frame["start"] = pd.to_datetime(frame["start"])
    frame["end"] = pd.to_datetime(frame["end"])

    original = pd.read_csv
    try:
        pd.read_csv = lambda *args, **kwargs: frame.copy()
        out = replica.with_fourth_quarters("X")
    finally:
        pd.read_csv = original

    assert out.size == 4
    assert out.loc[pd.Timestamp("2022-12-31")] == approx(150.0)   # 500 less 100, 200, 50


def test_a_tagged_fourth_quarter_is_left_alone():
    """Deriving over a figure the filing already gave would overwrite the filed number."""
    rows = [{"concept": "X", "period_type": "quarter", "start": f"2022-{m:02d}-01",
             "end": e, "val_usd": v * 1e6, "form": "10-Q", "filed": "2023-01-01"}
            for m, e, v in [(1, "2022-03-31", 100.0), (4, "2022-06-30", 200.0),
                            (7, "2022-09-30", 50.0), (10, "2022-12-31", 999.0)]]
    rows.append({"concept": "X", "period_type": "year", "start": "2022-01-01",
                 "end": "2022-12-31", "val_usd": 500.0 * 1e6, "form": "10-K",
                 "filed": "2023-02-01"})
    frame = pd.DataFrame(rows)
    frame["start"] = pd.to_datetime(frame["start"])
    frame["end"] = pd.to_datetime(frame["end"])

    original = pd.read_csv
    try:
        pd.read_csv = lambda *args, **kwargs: frame.copy()
        out = replica.with_fourth_quarters("X")
    finally:
        pd.read_csv = original

    assert out.loc[pd.Timestamp("2022-12-31")] == approx(999.0)


def test_the_model_periods_sign_the_liability_the_filings_way():
    """A liability that got cheaper is a gain, so the sign flips against the raw change."""
    dates = pd.date_range("2021-01-01", periods=120, freq="D")
    daily = pd.DataFrame({
        "reporting": np.linspace(10.0, 4.0, dates.size),      # liability falling
        "hedge_mark": np.zeros(dates.size),
        "cash": np.zeros(dates.size),
        "oci": np.zeros(dates.size),
        "own_credit_spread": np.full(dates.size, 0.02),
    }, index=dates)
    out = replica.model_periods(daily, "QE")
    assert (out["liability"] > 0).all()
