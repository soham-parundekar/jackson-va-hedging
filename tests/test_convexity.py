"""The curvature surface: interpolation, scaling, and the substitution into a hedge's Greeks.

Every case is one whose answer can be worked out by hand. A surface that interpolates slightly
wrong produces a plausible gamma rather than an error, and the hedge would size an option leg
off it without complaint.

Comparisons go through ``float(np.ravel(...)[0])`` rather than comparing arrays to ``approx``
directly, because the surface returns arrays and NumPy's own ``__eq__`` gets first refusal on
``array == approx(...)``, which makes the result depend on shape in ways a test should not.
"""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pandas as pd

from tests.checks import approx
from vahedge.valuation.convexity import ConvexitySurface, with_nested_gamma


def bell(years=(4.0, 8.0), grid=np.linspace(0.2, 1.4, 25)) -> ConvexitySurface:
    """A surface shaped like the real one: a hump at the money, decaying either side.

    The level halves between the two policy years, which is what the real surface does as the
    guarantee runs down, so a test can tell interpolation in time from interpolation in
    moneyness by which number comes back.
    """
    rows = []
    for index, year in enumerate(years):
        height = 1.0 / (index + 1.0)
        for moneyness in grid:
            rows.append({
                "policy_year": year,
                "moneyness": float(moneyness),
                "gamma_per_unit": height * float(np.exp(-((moneyness - 0.95) / 0.25) ** 2)),
            })
    return ConvexitySurface(table=pd.DataFrame(rows), equity_bump=0.10, inner_paths=5_000)


def at(surface, year, moneyness) -> float:
    return float(np.ravel(surface.per_unit(year, moneyness))[0])


def test_reads_back_the_tabulated_value():
    surface = bell(years=(4.0, 8.0), grid=np.array([0.5, 0.95, 1.4]))
    assert at(surface, 4.0, 0.95) == approx(1.0, abs=1e-12)
    assert at(surface, 8.0, 0.95) == approx(0.5, abs=1e-12)


def test_interpolates_in_time_between_years():
    surface = bell(years=(4.0, 8.0), grid=np.array([0.5, 0.95, 1.4]))
    assert at(surface, 6.0, 0.95) == approx(0.75, abs=1e-12)


def test_holds_the_end_value_outside_the_tabulated_range():
    """A bell extrapolated linearly goes negative, so the ends are held instead."""
    surface = bell()
    assert at(surface, 4.0, 2.5) == approx(at(surface, 4.0, 1.4))
    assert at(surface, 4.0, 0.0) == approx(at(surface, 4.0, 0.2))


def test_holds_the_end_year_outside_the_tabulated_years():
    surface = bell()
    assert at(surface, 1.0, 0.95) == approx(at(surface, 4.0, 0.95))
    assert at(surface, 30.0, 0.95) == approx(at(surface, 8.0, 0.95))


def test_gamma_is_homogeneous_of_degree_one():
    """Doubling the account and the base together doubles the gamma and leaves per-unit alone.

    This is why the surface can be tabulated against moneyness at all: a contract twice the size
    in the same state is two of the same contract.
    """
    surface = bell()
    small = float(np.ravel(surface.gamma(5.0, 90.0, 100.0))[0])
    large = float(np.ravel(surface.gamma(5.0, 180.0, 200.0))[0])
    assert large == approx(2.0 * small, rel=1e-12)


def test_gamma_handles_an_exhausted_contract():
    """Benefit base at zero would divide by zero in the moneyness; the floor catches it."""
    surface = bell()
    assert np.isfinite(surface.gamma(5.0, 0.0, 0.0)).all()


def test_vectorises_over_states():
    surface = bell()
    out = surface.per_unit([4.0, 6.0, 8.0], [0.95, 0.95, 0.95])
    assert out.shape == (3,)
    assert out[0] > out[1] > out[2]


def test_survives_a_round_trip_through_csv():
    """A temporary directory rather than a fixture, because tests/run_tests.py calls each test
    function with no arguments and that runner is what ``make test`` uses."""
    surface = bell()
    with tempfile.TemporaryDirectory() as folder:
        path = pathlib.Path(folder) / "surface.csv"
        surface.to_csv(path)
        back = ConvexitySurface.from_csv(path)
    assert back.equity_bump == surface.equity_bump
    assert back.inner_paths == surface.inner_paths
    assert at(back, 6.0, 1.1) == approx(at(surface, 6.0, 1.1))


class FakeProxy:
    """Stands in for a fitted proxy: answers with markers so a substitution is visible."""

    def greeks_at(self, **kwargs):
        return {"value": 12.0, "delta": -3.0, "gamma": 999.0, "vega": 0.4, "rho": -1.2}


def test_substitution_replaces_only_the_gamma():
    proxy, surface = FakeProxy(), bell()
    out = with_nested_gamma(proxy, surface)(
        years_since_issue=5.0, account_value=95.0, benefit_base=100.0,
        variance=0.04, zero_10y=0.03,
    )
    expected = float(np.ravel(surface.gamma(5.0, 95.0, 100.0))[0])
    assert float(np.ravel(out["gamma"])[0]) == approx(expected)
    assert float(np.ravel(out["gamma"])[0]) != approx(999.0)
    for name in ("value", "delta", "vega", "rho"):
        assert out[name] == proxy.greeks_at()[name]


def test_substitution_accepts_a_bare_callable():
    """The misspecification experiment passes a function, not a fitted object."""
    surface = bell()
    override = lambda **kwargs: {"value": 1.0, "gamma": -7.0}
    out = with_nested_gamma(override, surface)(
        years_since_issue=5.0, account_value=95.0, benefit_base=100.0,
    )
    expected = float(np.ravel(surface.gamma(5.0, 95.0, 100.0))[0])
    assert out["value"] == 1.0
    assert float(np.ravel(out["gamma"])[0]) == approx(expected)


def test_substitution_does_not_mutate_what_the_source_returned():
    proxy, surface = FakeProxy(), bell()
    with_nested_gamma(proxy, surface)(
        years_since_issue=5.0, account_value=95.0, benefit_base=100.0,
    )
    assert proxy.greeks_at()["gamma"] == 999.0
