"""Treasury par yields to zero rates, and a smooth curve Hull-White can differentiate.

Two curve objects live here and they do different jobs.

``ZeroCurve`` is the bootstrap. Treasury publishes par yields, so a forty-year cash flow
discounted straight off the thirty-year par yield is wrong by enough to matter. The
bootstrap is exact forward substitution on the semi-annual coupon grid and is what every
discounting in the project runs off. Jackson's Item 7A shock is described as a parallel
shift in risk-free rates, and that is applied the way a desk applies it: shift the par
yields, bootstrap again. Shifting the bootstrapped zeros instead gives slightly different
discount factors, which is why ``ParCurveBuilder.build`` re-runs the whole bootstrap.

``NelsonSiegelSvensson`` is the smooth fit, and it exists for one reason: Hull-White needs
the instantaneous forward rate f(0,t), which is the derivative of t*R(t). A bootstrapped
curve is piecewise linear in the zero rate, so its derivative is a step function, and a
step function in f(0,t) puts visible kinks into simulated bond prices. NSS has an analytic
forward and extrapolates sensibly past thirty years, which a forty-year liability needs.

The fit is to the bootstrapped zero rates, not to the par yields. NSS is a model of the
spot curve; fitting it to par yields and then treating the output as a spot curve is a
common shortcut and it is wrong by roughly the par-zero spread, which at thirty years and
a 5% curve is about fifteen basis points.
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
        """The par coupon this curve implies at a given tenor. Used to check a round trip."""
        return par_equivalent(self, tenor, frequency)


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

    Par yields are interpolated onto the coupon grid first, flat below the shortest quoted
    tenor, which for Treasury data is one year. Each par bond prices to one:

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
    """A flat continuously compounded curve. Used in tests and closed-form checks."""
    tenors = np.array([0.5, max_tenor])
    return ZeroCurve(tenors=tenors, zero_rates=np.array([rate, rate]))


@dataclass(frozen=True)
class NelsonSiegelSvensson:
    """Six-parameter smooth zero curve with an analytic instantaneous forward.

    Zero rate:

        R(t) = b0 + b1*g1(t/tau1) + b2*g2(t/tau1) + b3*g2(t/tau2)

    with g1(x) = (1 - exp(-x)) / x and g2(x) = g1(x) - exp(-x).

    Instantaneous forward, differentiating t*R(t):

        f(0,t) = b0 + b1*exp(-t/tau1) + b2*(t/tau1)*exp(-t/tau1) + b3*(t/tau2)*exp(-t/tau2)

    b0 is the level the curve flattens to, which is what a forty-year projection sits on
    long after the last quoted Treasury tenor.
    """

    beta0: float
    beta1: float
    beta2: float
    beta3: float
    tau1: float
    tau2: float
    rmse_bp: float = float("nan")   # fit error against the input, in basis points

    def zero(self, t):
        t = np.asarray(t, dtype=float)
        basis = _nss_basis(t, self.tau1, self.tau2)
        flat = basis @ np.array([self.beta0, self.beta1, self.beta2, self.beta3])
        return flat.reshape(t.shape)

    def discount(self, t):
        t = np.asarray(t, dtype=float)
        return np.exp(-self.zero(t) * t)

    def instantaneous_forward(self, t):
        t = np.asarray(t, dtype=float)
        x1 = t / self.tau1
        x2 = t / self.tau2
        return (
            self.beta0
            + self.beta1 * np.exp(-x1)
            + self.beta2 * x1 * np.exp(-x1)
            + self.beta3 * x2 * np.exp(-x2)
        )

    def as_zero_curve(self, max_tenor: float = 60.0, step: float = 0.25) -> ZeroCurve:
        """Sample onto a grid, for code that wants the ``ZeroCurve`` interface."""
        tenors = np.arange(step, max_tenor + step / 2, step)
        return ZeroCurve(tenors=tenors, zero_rates=np.asarray(self.zero(tenors), dtype=float))

    def par_equivalent(self, tenor: float, frequency: int = 2) -> float:
        return par_equivalent(self, tenor, frequency)


def par_equivalent(curve, tenor: float, frequency: int = 2) -> float:
    """The par coupon a curve implies at a tenor, from its own discount factors.

    This is the round trip that says whether a fitted curve reproduces the Treasuries it came
    from. Both curve types answer it the same way, so the arithmetic lives here once.
    """
    times = np.arange(1, int(round(tenor * frequency)) + 1) / frequency
    dfs = np.asarray(curve.discount(times), dtype=float)
    return float(frequency * (1.0 - dfs[-1]) / dfs.sum())


def _nss_basis(t: np.ndarray, tau1: float, tau2: float) -> np.ndarray:
    """Design matrix for the four linear coefficients at the given decays.

    t = 0 is a removable singularity in g1: the limit is 1 for g1 and 0 for g2. Maturities
    are positive everywhere this is called, but the guard keeps a zero from reaching the
    divide.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    safe = np.where(t <= 0, 1e-8, t)
    out = np.empty((safe.size, 4))
    out[:, 0] = 1.0
    for column, tau in ((1, tau1), (3, tau2)):
        x = safe / tau
        g1 = (1.0 - np.exp(-x)) / x
        g2 = g1 - np.exp(-x)
        if column == 1:
            out[:, 1] = g1
            out[:, 2] = g2
        else:
            out[:, 3] = g2
    return out


def fit_nss(
    tenors,
    zero_rates,
    tau_grid=None,
    weights=None,
) -> NelsonSiegelSvensson:
    """Fit NSS by grid search on the two decays with least squares on the four betas.

    The objective is not convex in tau1 and tau2, and a gradient optimiser started
    anywhere reasonable lands in a different local minimum depending on the day's curve
    shape. Because the betas solve in closed form once the decays are fixed, the whole fit
    is a two-dimensional search over a grid of a few hundred points, which is both fast and
    deterministic. Determinism matters here: this runs on every date of a ten-year backtest
    and a fit that jitters between neighbouring optima would put a sawtooth into the
    simulated bond prices.

    The two decays are kept apart by a factor of 1.5 so the third and fourth basis
    functions stay distinguishable; when they collide the design matrix is near-singular
    and the betas blow up while the fit barely improves.
    """
    tenors = np.asarray(tenors, dtype=float)
    zero_rates = np.asarray(zero_rates, dtype=float)
    if tenors.shape != zero_rates.shape:
        raise ValueError("tenors and zero_rates must be the same length")
    if tenors.size < 6:
        raise ValueError("NSS has six parameters; give it at least six points")

    if tau_grid is None:
        # Two grids rather than one. The third basis function is a short-dated hump and the
        # fourth a long-dated one, and letting both decays roam the same wide range lets the
        # search put the short hump at fifteen years, which fits the quoted tenors fine and
        # then extrapolates to nonsense. Splitting the ranges is what keeps beta0, the level
        # the curve flattens to, interpretable.
        tau_grid = (np.geomspace(0.3, 8.0, 32), np.geomspace(3.0, 30.0, 32))
    short_grid, long_grid = (
        (np.asarray(tau_grid[0], dtype=float), np.asarray(tau_grid[1], dtype=float))
        if isinstance(tau_grid, tuple)
        else (np.asarray(tau_grid, dtype=float), np.asarray(tau_grid, dtype=float))
    )

    w = np.ones_like(tenors) if weights is None else np.asarray(weights, dtype=float)
    sqrt_w = np.sqrt(w)
    target = zero_rates * sqrt_w

    best = None
    for tau1 in short_grid:
        for tau2 in long_grid:
            if tau2 < 1.5 * tau1:
                continue
            design = _nss_basis(tenors, tau1, tau2) * sqrt_w[:, None]
            betas, *_ = np.linalg.lstsq(design, target, rcond=None)
            residual = design @ betas - target
            sse = float(residual @ residual)
            if best is None or sse < best[0]:
                best = (sse, betas, tau1, tau2)

    if best is None:
        raise ValueError("no admissible (tau1, tau2) pair in the grid")

    sse, betas, tau1, tau2 = best
    rmse = np.sqrt(sse / tenors.size) * 10000.0
    return NelsonSiegelSvensson(
        beta0=float(betas[0]),
        beta1=float(betas[1]),
        beta2=float(betas[2]),
        beta3=float(betas[3]),
        tau1=float(tau1),
        tau2=float(tau2),
        rmse_bp=float(rmse),
    )


def fit_curve(
    curve: ZeroCurve,
    tail_years=(35.0, 40.0, 45.0, 50.0, 60.0),
    tail_weight: float = 0.5,
) -> NelsonSiegelSvensson:
    """Fit NSS to a bootstrapped curve, with the extrapolation pinned down.

    Treasury stops quoting at thirty years and this liability runs past forty-five, so
    something has to be assumed about the tail. The assumption made here is the same one the
    bootstrap makes when asked for a rate past its last knot: the zero rate is flat beyond
    the last quoted tenor. Writing that in as weighted anchor points rather than leaving it
    implicit is what stops the fit from extrapolating to a long-run level of minus twenty-six
    percent, which an unanchored NSS on this curve does while fitting the quoted tenors to
    three basis points. A curve can be right everywhere you can check it and absurd where you
    cannot, and this liability lives where you cannot.

    The anchors are down-weighted relative to the quoted points because they are an
    assumption, not an observation.
    """
    tenors = np.asarray(curve.tenors, dtype=float)
    rates = np.asarray(curve.zero_rates, dtype=float)
    tail = np.asarray([t for t in tail_years if t > tenors[-1]], dtype=float)
    if tail.size:
        tenors = np.concatenate([tenors, tail])
        rates = np.concatenate([rates, np.full(tail.size, rates[-1])])
    weights = np.concatenate(
        [np.ones(curve.tenors.size), np.full(tail.size, tail_weight)]
    )
    return fit_nss(tenors, rates, weights=weights)
