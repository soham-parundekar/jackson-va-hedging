"""One-factor Hull-White short rate, fitted to today's curve.

A lifetime withdrawal guarantee pays for forty years, so what it is worth depends on the
discount curve at least as much as on equity. Holding rates deterministic understates the
liability's rho and leaves the rate hedge with nothing to be wrong about, which is the
opposite of useful. Hull-White is the smallest model that is arbitrage-free against the
curve that is actually quoted:

    dr(t) = (theta(t) - a*r(t)) dt + sigma dW(t)

theta(t) is not a free parameter. It is whatever reproduces the initial curve, and the
standard way to write that is to split the short rate into a deterministic level and a
zero-mean Ornstein-Uhlenbeck process:

    r(t) = x(t) + alpha(t),   dx = -a*x dt + sigma dW,   x(0) = 0
    alpha(t) = f(0,t) + (sigma^2 / (2a^2)) * (1 - exp(-a*t))^2

where f(0,t) is the instantaneous forward from the NSS fit. The second term is the convexity
correction that keeps E[exp(-integral r)] equal to the quoted discount factor rather than
merely close to it.

Simulation is exact, not Euler. Over one step the pair (x at the end of the step, the
integral of x across the step) is jointly normal with moments that integrate in closed form,
so a monthly grid introduces no discretisation error in the rate at all. That matters
because the integral is what discounts the cash flows, and a scheme that gets the rate right
but the integral wrong prices a forty-year annuity badly.

Calibration is historical. Swaption volatilities are not free data, so a and sigma come from
an AR(1) regression on the short rate, which estimates them under the real-world measure and
says nothing about the market price of rate risk. Both are therefore carried as scenario
parameters and swept, and the sweep is reported next to the headline rather than in an
appendix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .curves import NelsonSiegelSvensson

MIN_MEAN_REVERSION = 1e-6


@dataclass(frozen=True)
class HullWhite:
    """Hull-White with the initial curve baked in."""

    a: float                        # mean reversion speed, per year
    sigma: float                    # absolute volatility of the short rate, per root year
    curve: NelsonSiegelSvensson

    def __post_init__(self) -> None:
        if self.a <= MIN_MEAN_REVERSION:
            raise ValueError(
                f"mean reversion {self.a} is at or below zero; the B and variance formulas "
                "divide by it and the process stops reverting"
            )
        if self.sigma < 0:
            raise ValueError("sigma must not be negative")

    # ---------------------------------------------------------------- closed forms

    def B(self, tau):
        """(1 - exp(-a*tau)) / a, the duration factor in the bond price."""
        tau = np.asarray(tau, dtype=float)
        return (1.0 - np.exp(-self.a * tau)) / self.a

    def alpha(self, t):
        """Deterministic level of the short rate at t."""
        t = np.asarray(t, dtype=float)
        return self.curve.instantaneous_forward(t) + (
            self.sigma**2 / (2.0 * self.a**2)
        ) * (1.0 - np.exp(-self.a * t)) ** 2

    def bond_price(self, t, maturity, r):
        """P(t, T) given the short rate at t.

            P(t,T) = A(t,T) * exp(-B(T-t) * r(t))
            ln A   = ln(P(0,T)/P(0,t)) + B(T-t)*f(0,t)
                     - (sigma^2 / (4a)) * (1 - exp(-2at)) * B(T-t)^2

        t and maturity broadcast against r, so a whole path of bond prices costs one call.
        """
        t = np.asarray(t, dtype=float)
        maturity = np.asarray(maturity, dtype=float)
        if np.any(maturity < t):
            raise ValueError("maturity must not precede the valuation time")
        b = self.B(maturity - t)
        log_a = (
            np.log(self.curve.discount(maturity) / self.curve.discount(t))
            + b * self.curve.instantaneous_forward(t)
            - (self.sigma**2 / (4.0 * self.a)) * (1.0 - np.exp(-2.0 * self.a * t)) * b**2
        )
        return np.exp(log_a - b * np.asarray(r, dtype=float))

    def zero_rate(self, t, tenor, r):
        """Continuously compounded zero rate of the given tenor, seen from time t."""
        tenor = float(tenor)
        if tenor <= 0:
            raise ValueError("tenor must be positive")
        return -np.log(self.bond_price(t, np.asarray(t, dtype=float) + tenor, r)) / tenor

    # ---------------------------------------------------------------- simulation

    def step_moments(self, dt: float) -> tuple[float, float, float, float]:
        """Moments of (x at the end of the step, integral of x over the step).

        Returns (decay, var_x, var_integral, covariance). With x(t) known, the end-of-step
        x has mean x*decay and the integral has mean x*B(dt); both are exact.
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        a, s = self.a, self.sigma
        decay = float(np.exp(-a * dt))
        var_x = s**2 * (1.0 - np.exp(-2.0 * a * dt)) / (2.0 * a)
        var_int = (s**2 / a**2) * (
            dt - 2.0 * (1.0 - np.exp(-a * dt)) / a + (1.0 - np.exp(-2.0 * a * dt)) / (2.0 * a)
        )
        cov = (s**2 / a) * (
            (1.0 - np.exp(-a * dt)) / a - (1.0 - np.exp(-2.0 * a * dt)) / (2.0 * a)
        )
        return decay, float(var_x), float(var_int), float(cov)

    def deterministic_drift_integral(self, t0: float, t1: float) -> float:
        """Integral of alpha from t0 to t1, in closed form.

        The forward part integrates to the log ratio of discount factors and the convexity
        part integrates to half the variance of the integrated OU process. Writing it this
        way rather than quadrature is what makes the martingale test pass to Monte Carlo
        error instead of to the accuracy of a quadrature rule.
        """
        forward_part = np.log(self.curve.discount(t0) / self.curve.discount(t1))
        return float(forward_part + 0.5 * (self._int_var(t1) - self._int_var(t0)))

    def _int_var(self, t: float) -> float:
        """Variance of the integral of x from 0 to t."""
        a, s = self.a, self.sigma
        if t <= 0:
            return 0.0
        return float(
            (s**2 / a**2)
            * (t - 2.0 * (1.0 - np.exp(-a * t)) / a + (1.0 - np.exp(-2.0 * a * t)) / (2.0 * a))
        )


def calibrate_historical(
    short_rates,
    dt: float,
    min_mean_reversion: float = 0.02,
) -> tuple[float, float, dict]:
    """Estimate (a, sigma) by AR(1) on an observed short-rate series.

    Discretely, x_{t+dt} = phi*x_t + noise with phi = exp(-a*dt) and noise variance
    sigma^2 * (1 - phi^2) / (2a). Regressing the level on its own lag recovers both.

    The honest problem with this, stated here because it changes how the output should be
    read: on any post-2008 sample the short rate is close to a unit root, so phi sits within
    a standard error or two of one and the implied mean reversion is barely identified. A
    long sample helps and this is run on the longest free history available, but the
    estimate still deserves a floor. Below ``min_mean_reversion`` the fitted a is replaced by
    the floor and the diagnostics record that it bound, because an a of 0.002 puts a
    thirty-year rate distribution three hundred basis points wide and nothing downstream
    survives that.

    Returns (a, sigma, diagnostics).
    """
    rates = np.asarray(short_rates, dtype=float)
    rates = rates[np.isfinite(rates)]
    if rates.size < 250:
        raise ValueError(f"need at least 250 observations to fit an AR(1), got {rates.size}")
    if dt <= 0:
        raise ValueError("dt must be positive")

    lagged, current = rates[:-1], rates[1:]
    design = np.column_stack([np.ones_like(lagged), lagged])
    coefficients, *_ = np.linalg.lstsq(design, current, rcond=None)
    intercept, phi = float(coefficients[0]), float(coefficients[1])
    residuals = current - design @ coefficients
    residual_var = float(residuals @ residuals) / (residuals.size - 2)

    bound = False
    if phi >= 1.0 or phi <= 0.0:
        a = min_mean_reversion
        bound = True
    else:
        a = -np.log(phi) / dt
        if a < min_mean_reversion:
            a = min_mean_reversion
            bound = True

    # Invert the stationary AR(1) noise variance for sigma at the a actually used.
    phi_used = np.exp(-a * dt)
    sigma = float(np.sqrt(residual_var * 2.0 * a / (1.0 - phi_used**2)))

    diagnostics = {
        "observations": int(rates.size),
        "dt_years": float(dt),
        "ar1_phi": phi,
        "ar1_intercept": intercept,
        "implied_long_run_level": float(intercept / (1.0 - phi)) if phi < 1 else float("nan"),
        "residual_std": float(np.sqrt(residual_var)),
        "mean_reversion_floor_bound": bound,
        "half_life_years": float(np.log(2.0) / a),
    }
    return float(a), sigma, diagnostics
