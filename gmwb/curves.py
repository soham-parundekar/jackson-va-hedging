"""Treasury par yields to continuously compounded zero rates.

Treasury publishes par yields, not zero rates, so a 30-year cash flow discounted
straight off the 30-year par yield is wrong by enough to matter for a liability
whose cash flows run 40 years out. Everything downstream discounts off the
bootstrapped zero curve.

The shock in Jackson's Item 7A is described as a parallel shift in risk-free
interest rates. That is applied here the way a desk would apply it: shift the par
yields, then bootstrap again. Shifting the zero curve instead gives slightly
different discount factors, which is why ``shift`` re-runs the bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ZeroCurve:
    """Continuously compounded zero rates on an increasing grid of maturities."""

    tenors: np.ndarray       # years, strictly increasing, first element > 0
    zero_rates: np.ndarray   # continuously compounded, decimal

    def __post_init__(self) -> None:
        if self.tenors.ndim != 1 or self.tenors.shape != self.zero_rates.shape:
            raise ValueError("tenors and zero_rates must be 1-D and the same length")
        if np.any(np.diff(self.tenors) <= 0):
            raise ValueError("tenors must be strictly increasing")
        if self.tenors[0] <= 0:
            raise ValueError("tenors must be positive")

    def zero(self, t):
        """Zero rate at t, linear in the rate between knots and flat outside them."""
        t = np.asarray(t, dtype=float)
        return np.interp(t, self.tenors, self.zero_rates)

    def discount(self, t):
        t = np.asarray(t, dtype=float)
        return np.exp(-self.zero(t) * t)

    def forward(self, t1, t2):
        """Continuously compounded forward rate over [t1, t2]."""
        t1 = np.asarray(t1, dtype=float)
        t2 = np.asarray(t2, dtype=float)
        if np.any(t2 <= t1):
            raise ValueError("t2 must exceed t1")
        return (self.zero(t2) * t2 - self.zero(t1) * t1) / (t2 - t1)

    def par_equivalent(self, tenor: float, frequency: int = 2) -> float:
        """The par coupon this curve implies at a given tenor. Used to check that a
        bootstrap round-trips."""
        times = np.arange(1, int(round(tenor * frequency)) + 1) / frequency
        dfs = self.discount(times)
        return float(frequency * (1.0 - dfs[-1]) / dfs.sum())


class ParCurveBuilder:
    """Holds the observed par yields so a shocked curve can be re-bootstrapped."""

    def __init__(self, tenors, par_yields, frequency: int = 2):
        self.tenors = np.asarray(tenors, dtype=float)
        self.par_yields = np.asarray(par_yields, dtype=float)
        self.frequency = int(frequency)
        if np.any(np.isnan(self.par_yields)):
            raise ValueError("par yields contain NaN; fill the curve before bootstrapping")
        if np.any(np.diff(self.tenors) <= 0):
            raise ValueError("par tenors must be strictly increasing")

    def build(self, shift_bp: float = 0.0) -> ZeroCurve:
        return bootstrap(
            self.tenors,
            self.par_yields + shift_bp / 10000.0,
            frequency=self.frequency,
        )


def bootstrap(tenors, par_yields, frequency: int = 2) -> ZeroCurve:
    """Bootstrap continuously compounded zeros from par yields.

    Par yields are interpolated onto the coupon grid first (flat below the shortest
    quoted tenor, which for Treasury data is one year). Each par bond prices to one:

        1 = (c/m) * sum_{i<=n} D(t_i) + D(t_n)

    solved forward for D(t_n).
    """
    tenors = np.asarray(tenors, dtype=float)
    par_yields = np.asarray(par_yields, dtype=float)
    m = int(frequency)
    if m < 1:
        raise ValueError("frequency must be at least 1")

    n_steps = int(round(tenors[-1] * m))
    grid = np.arange(1, n_steps + 1) / m
    coupons = np.interp(grid, tenors, par_yields)   # np.interp is flat outside the range

    dfs = np.empty(n_steps)
    running_sum = 0.0
    for i in range(n_steps):
        c = coupons[i] / m
        dfs[i] = (1.0 - c * running_sum) / (1.0 + c)
        if dfs[i] <= 0:
            raise ValueError(
                f"bootstrap produced a non-positive discount factor at {grid[i]:.2f}y; "
                "the input par curve is not arbitrage-free"
            )
        running_sum += dfs[i]

    zeros = -np.log(dfs) / grid
    return ZeroCurve(tenors=grid, zero_rates=zeros)


def flat_curve(rate: float, max_tenor: float = 60.0) -> ZeroCurve:
    """A flat continuously compounded curve. Used in tests and in closed-form checks."""
    tenors = np.array([0.5, max_tenor])
    return ZeroCurve(tenors=tenors, zero_rates=np.array([rate, rate]))
