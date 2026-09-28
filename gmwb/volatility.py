"""Volatility term structure for the sub-account.

Jackson describes its own fair-value input this way: implied market volatility for
durations up to five years, grading to a historical volatility level by year ten, where
the long-run historical level carries an explicit risk margin (FY2025 10-K, Note 6).

Reproducing that with free data takes some care. FRED carries the 30-day VIX and the
3-month VIX and nothing longer, so three months is the furthest tenor with an observed
implied volatility. The obvious shortcut is to hold the 3-month level flat out to five
years and grade from there, and it is wrong in a way that matters: a volatility spike
would then propagate undamped across the whole front of the surface. In March 2020 the
3-month index went from roughly 20 to roughly 70, and a flat-to-five-years reading of
that implies the five-year implied volatility also rose fifty points, which no
volatility surface has ever done. The consequence is a vega several times too large,
which then dominates a hedging backtest.

What is used instead is a mean-reverting forward variance curve, the standard shape for
a volatility term structure:

    forward variance v(t) = v_long + (v_front - v_long) * exp(-kappa * t)

Total variance integrates in closed form, forward variance is positive by construction,
and a move in the observed short-dated level decays with maturity the way a real surface
does. The decay is pinned to Jackson's own description rather than fitted: kappa is set
so the front level has graded ``1 - residual`` of the way to the long-run level by
``grade_to`` years, with grade_to at ten years and residual at 2%.

The front level is then fitted to the observed implied volatilities. With one parameter
and two observations the fit is the average of the two implied front levels, which keeps
both the 30-day and the 3-month index in the answer rather than discarding one.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

TRADING_DAYS = 252
DEFAULT_GRADE_TO = 10.0
DEFAULT_RESIDUAL = 0.02


def decay_from_grading(grade_to: float = DEFAULT_GRADE_TO,
                       residual: float = DEFAULT_RESIDUAL) -> float:
    """Mean reversion speed implied by grading to the long-run level by a given year."""
    if grade_to <= 0:
        raise ValueError("grade_to must be positive")
    if not 0 < residual < 1:
        raise ValueError("residual must be in (0, 1)")
    return float(-np.log(residual) / grade_to)


@dataclass(frozen=True)
class VolTermStructure:
    """Mean-reverting forward variance, described by its two levels and a decay."""

    front_level: float        # instantaneous volatility at the front of the curve
    long_run_level: float     # realised long-run volatility plus the risk margin
    decay: float = 0.3912     # kappa; the default is decay_from_grading(10, 0.02)
    observed_tenors: tuple = (0.0833, 0.25)   # what the front level was fitted to

    def __post_init__(self) -> None:
        if self.front_level <= 0 or self.long_run_level <= 0:
            raise ValueError("volatility levels must be positive")
        if self.decay <= 0:
            raise ValueError("decay must be positive")

    @classmethod
    def from_implied(
        cls,
        observations: dict[float, float],
        long_run_level: float,
        grade_to: float = DEFAULT_GRADE_TO,
        residual: float = DEFAULT_RESIDUAL,
    ) -> "VolTermStructure":
        """Fit the front level to observed spot implied volatilities.

        ``observations`` maps maturity in years to implied volatility as a decimal. For
        each one, inverting the closed-form total variance gives the front variance that
        would reproduce it exactly; the fitted front variance is their average.
        """
        if not observations:
            raise ValueError("need at least one implied volatility observation")
        kappa = decay_from_grading(grade_to, residual)
        v_long = long_run_level**2

        estimates = []
        for tenor, implied in observations.items():
            if tenor <= 0 or implied <= 0:
                raise ValueError(f"bad observation {tenor}y at {implied}")
            annuity = (1.0 - np.exp(-kappa * tenor)) / kappa
            estimates.append(v_long + (implied**2 * tenor - v_long * tenor) / annuity)
        v_front = float(np.mean(estimates))
        if v_front <= 0:
            # Only reachable if an observed implied volatility is far below the long-run
            # level at a very short tenor. Falling back to the shortest observation keeps
            # the curve well defined and is reported rather than hidden.
            v_front = min(observations.values()) ** 2
        return cls(
            front_level=float(np.sqrt(v_front)),
            long_run_level=float(long_run_level),
            decay=kappa,
            observed_tenors=tuple(sorted(observations)),
        )

    def forward_vol(self, t):
        """Instantaneous volatility at time t."""
        t = np.asarray(t, dtype=float)
        v_long = self.long_run_level**2
        v_front = self.front_level**2
        return np.sqrt(v_long + (v_front - v_long) * np.exp(-self.decay * t))

    def total_variance(self, t):
        """Integral of forward variance from zero to t, in closed form."""
        t = np.asarray(t, dtype=float)
        v_long = self.long_run_level**2
        v_front = self.front_level**2
        annuity = (1.0 - np.exp(-self.decay * t)) / self.decay
        return v_long * t + (v_front - v_long) * annuity

    def spot_vol(self, t):
        """Volatility of the return to maturity t."""
        t = np.asarray(t, dtype=float)
        return np.sqrt(self.total_variance(t) / t)

    def step_variances(self, times) -> np.ndarray:
        """Variance of each interval in an increasing sequence of maturities.

        The engine steps between anniversaries, so it needs forward variance. These are
        positive by construction; the check stays because a change to the shape could
        break that silently.
        """
        times = np.asarray(times, dtype=float)
        if times[0] <= 0:
            raise ValueError("times must start after zero")
        if np.any(np.diff(times) <= 0):
            raise ValueError("times must be strictly increasing")
        steps = np.diff(np.concatenate(([0.0], self.total_variance(times))))
        if np.any(steps <= 0):
            bad = times[steps <= 0]
            raise ValueError(f"non-positive forward variance at maturities {bad}")
        return steps

    def shift(self, points: float) -> "VolTermStructure":
        """Move the whole curve by an absolute number of volatility points."""
        return replace(
            self,
            front_level=self.front_level + points,
            long_run_level=self.long_run_level + points,
        )

    def shift_front(self, points: float) -> "VolTermStructure":
        """Move only the short-dated level, leaving the long-run level alone.

        This is the shift that corresponds to something tradeable. Listed index options
        run out long before the long-run level starts to matter, and across a backtest
        the long-run level is an assumption that does not move week to week while the
        implied indices do.
        """
        return replace(self, front_level=self.front_level + points)

    # Kept as an alias because "implied" is how the observable is described elsewhere.
    shift_implied = shift_front

    @property
    def implied_level(self) -> float:
        """Spot volatility at the shortest observed tenor, for reporting."""
        return float(self.spot_vol(self.observed_tenors[0]))


def realised_vol(prices, window_days: int | None = None) -> float:
    """Annualised volatility of log returns, ignoring gaps in the series."""
    prices = np.asarray(prices, dtype=float)
    prices = prices[np.isfinite(prices)]
    if window_days is not None:
        prices = prices[-(window_days + 1):]
    if prices.size < 30:
        raise ValueError(f"need at least 30 observations, got {prices.size}")
    log_returns = np.diff(np.log(prices))
    return float(log_returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
