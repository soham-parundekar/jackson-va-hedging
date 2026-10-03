"""Heston pricing by Fourier-cosine expansion, and calibration to a listed SPX chain.

Why stochastic volatility at all for this liability. A lifetime withdrawal guarantee is a
long-dated put struck at a level that ratchets. Under flat volatility the deep out-of-the-
money tail that decides whether the guarantee ever pays is priced off the same number as the
at-the-money options, and the index skew says that is wrong by several volatility points
exactly where it matters. Heston produces a skew from one mechanism - volatility rises when
the index falls - so the shape it produces is tied to a parameter that can be hedged rather
than fitted freely.

    dS = (r - q) S dt + sqrt(v) S dW_S
    dv = kappa (theta - v) dt + xi sqrt(v) dW_v,   d<W_S, W_v> = rho dt

The characteristic function is the "little trap" form of Albrecher, Mayer, Schoutens and
Tistaert. The textbook form has a branch cut that the complex logarithm crosses for large
maturities, which shows up as prices that oscillate wildly past a few years. On this chain
the longest expiry is December 2031, five and a quarter years out, so the trap is not
hypothetical: the naive form is unusable at the far end.

Pricing is the COS method of Fang and Oosterlee. The density is recovered from the
characteristic function on a truncated interval by a cosine expansion, and for a European
payoff the expansion coefficients are analytic, so a whole strike slice costs one matrix
product. It converges geometrically in the number of terms, which is what makes a calibration
over a few thousand quotes finish in seconds rather than minutes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.stats import norm


@dataclass(frozen=True)
class HestonParameters:
    v0: float       # initial variance
    kappa: float    # mean reversion speed of variance
    theta: float    # long-run variance
    xi: float       # volatility of variance
    rho: float      # correlation between the index and its variance

    def __post_init__(self) -> None:
        if self.v0 <= 0 or self.theta <= 0 or self.kappa <= 0 or self.xi <= 0:
            raise ValueError("v0, kappa, theta and xi must all be positive")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must lie strictly inside (-1, 1)")

    @property
    def feller(self) -> float:
        """2*kappa*theta - xi^2. Non-negative means variance cannot reach zero."""
        return 2.0 * self.kappa * self.theta - self.xi**2

    @property
    def feller_satisfied(self) -> bool:
        return bool(self.feller >= 0.0)

    @property
    def long_run_vol(self) -> float:
        return float(np.sqrt(self.theta))

    def as_dict(self) -> dict:
        return {
            "v0": self.v0,
            "kappa": self.kappa,
            "theta": self.theta,
            "xi": self.xi,
            "rho": self.rho,
            "feller": self.feller,
            "feller_satisfied": self.feller_satisfied,
            "long_run_vol": self.long_run_vol,
            "initial_vol": float(np.sqrt(self.v0)),
        }


def characteristic_function(u, params: HestonParameters, maturity: float) -> np.ndarray:
    """E[exp(i*u*log(S_T/S_0))] under Q, excluding the drift.

    The drift is left out so the caller can supply its own forward. Writing it this way
    keeps the forward in one place: the chain gives a forward per expiry from put-call
    parity, and threading it through the characteristic function instead would make the
    parity fit and the pricer disagree about what the drift is.
    """
    u = np.asarray(u, dtype=complex)
    kappa, theta, xi, rho, v0 = (
        params.kappa,
        params.theta,
        params.xi,
        params.rho,
        params.v0,
    )
    iu = 1j * u
    m = kappa - rho * xi * iu
    d = np.sqrt(m**2 + xi**2 * (iu + u**2))
    # The little trap: g = (m - d) / (m + d) rather than its reciprocal. Both are algebraically
    # the same price; only this one keeps |g| <= 1 so the principal branch of the log is the
    # right one for every maturity.
    g = (m - d) / (m + d)
    exp_dt = np.exp(-d * maturity)

    term_theta = (kappa * theta / xi**2) * (
        (m - d) * maturity - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g))
    )
    term_v0 = (v0 / xi**2) * (m - d) * (1.0 - exp_dt) / (1.0 - g * exp_dt)
    return np.exp(term_theta + term_v0)


def log_return_skewness(params: HestonParameters, maturity: float, step: float = 1e-3) -> float:
    """Risk-neutral skewness of log(S_T/S_0), which is the quantity the SKEW index reports.

    Cboe publishes SKEW as 100 - 10 * skewness of the thirty-day return, so this is directly
    comparable to a quoted index level and is the only free observation that speaks to the
    correlation and the volatility of variance. The variance term structure the volatility
    indices pin down is blind to both: rho and xi can move together with no effect on the
    expected average variance at any tenor.

    Taken off the characteristic function by central differences rather than from a published
    cumulant expression, because the second cumulant already in this file is the truncation
    approximation from the COS literature and sits about one and a half per cent away from the
    true value - harmless for setting an integration range, wrong for a third standardised
    moment. The default step is flat to six figures across four decades either side of it.

    Drift is excluded from the characteristic function, which costs nothing here: shifting a
    distribution leaves its skewness alone.
    """
    offsets = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) * step
    psi = np.log(characteristic_function(offsets, params, maturity))
    second = (psi[3] - 2.0 * psi[2] + psi[1]) / step**2
    third = (psi[4] - 2.0 * psi[3] + 2.0 * psi[1] - psi[0]) / (2.0 * step**3)
    variance = float(np.real(-second))
    if variance <= 0.0:
        raise ValueError(f"non-positive variance {variance:.3e} at maturity {maturity}")
    return float(np.real(1j * third)) / variance**1.5


def _cumulants(params: HestonParameters, maturity: float) -> tuple[float, float]:
    """First and second cumulants of log(S_T/S_0), drift excluded.

    These set the truncation range. They do not have to be exact - the range only has to
    contain essentially all of the density - but a range that is too tight loses mass and a
    range that is too wide wastes terms, and both show up as a price that will not settle as
    N grows.
    """
    kappa, theta, xi, rho, v0 = (
        params.kappa,
        params.theta,
        params.xi,
        params.rho,
        params.v0,
    )
    t = maturity
    ekt = np.exp(-kappa * t)
    c1 = (1.0 - ekt) * (theta - v0) / (2.0 * kappa) - 0.5 * theta * t
    c2 = (1.0 / (8.0 * kappa**3)) * (
        xi * t * kappa * ekt * (v0 - theta) * (8.0 * kappa * rho - 4.0 * xi)
        + kappa * rho * xi * (1.0 - ekt) * (16.0 * theta - 8.0 * v0)
        + 2.0 * theta * kappa * t * (-4.0 * kappa * rho * xi + xi**2 + 4.0 * kappa**2)
        + xi**2 * ((theta - 2.0 * v0) * np.exp(-2.0 * kappa * t) + theta * (6.0 * ekt - 7.0) + 2.0 * v0)
        + 8.0 * kappa**2 * (v0 - theta) * (1.0 - ekt)
    )
    return float(c1), float(c2)


def cos_price(
    params: HestonParameters,
    forward: float,
    strikes,
    maturity: float,
    discount: float,
    is_call=True,
    n_terms: int = 256,
    truncation: float = 12.0,
) -> np.ndarray:
    """European option prices on one expiry slice.

    ``forward`` and ``discount`` come from the chain, not from a dividend assumption.
    ``is_call`` may be a scalar or an array matching ``strikes``, which is how an OTM-only
    slice gets priced in one call.
    """
    strikes = np.atleast_1d(np.asarray(strikes, dtype=float))
    is_call = np.broadcast_to(np.asarray(is_call), strikes.shape)
    if maturity <= 0:
        raise ValueError("maturity must be positive")
    if forward <= 0 or np.any(strikes <= 0):
        raise ValueError("forward and strikes must be positive")

    c1, c2 = _cumulants(params, maturity)
    width = truncation * np.sqrt(abs(c2))
    x = np.log(forward / strikes)          # log-moneyness of the forward
    lower = x + c1 - width
    upper = x + c1 + width
    span = upper - lower

    k = np.arange(n_terms)
    # Shape (n_strikes, n_terms): each strike has its own truncation range, so the frequency
    # grid differs by strike and the characteristic function has to be evaluated per strike.
    u = k[None, :] * np.pi / span[:, None]
    cf = characteristic_function(u, params, maturity)
    unit = np.real(cf * np.exp(1j * u * (x[:, None] - lower[:, None])))
    unit[:, 0] *= 0.5                       # the primed sum halves the first term

    coefficients = _payoff_coefficients(lower, upper, k, is_call)
    return discount * np.sum(unit * coefficients, axis=1) * strikes


def _payoff_coefficients(lower, upper, k, is_call) -> np.ndarray:
    """Cosine coefficients of the call and put payoffs, in units of the strike."""
    span = (upper - lower)[:, None]
    call = np.asarray(is_call, dtype=bool)[:, None]
    zeros = np.zeros_like(lower)
    chi_c, psi_c = _chi_psi(lower, upper, zeros, upper, k)
    chi_p, psi_p = _chi_psi(lower, upper, lower, zeros, k)
    return np.where(call, 2.0 / span * (chi_c - psi_c), 2.0 / span * (psi_p - chi_p))


def _chi_psi(lower, upper, c, d, k) -> tuple[np.ndarray, np.ndarray]:
    """The two analytic integrals behind the payoff coefficients.

    chi integrates exp(y) against the cosine basis and psi integrates one against it. Both
    are in Fang and Oosterlee (2009), equations 22 and 23.
    """
    lower = lower[:, None]
    upper = upper[:, None]
    c = np.asarray(c, dtype=float)[:, None]
    d = np.asarray(d, dtype=float)[:, None]
    span = upper - lower
    omega = k[None, :] * np.pi / span

    cos_d = np.cos(omega * (d - lower))
    cos_c = np.cos(omega * (c - lower))
    sin_d = np.sin(omega * (d - lower))
    sin_c = np.sin(omega * (c - lower))
    exp_d = np.exp(d)
    exp_c = np.exp(c)

    chi = (1.0 / (1.0 + omega**2)) * (
        cos_d * exp_d - cos_c * exp_c + omega * sin_d * exp_d - omega * sin_c * exp_c
    )

    psi = np.empty_like(chi)
    psi[:, 1:] = (sin_d[:, 1:] - sin_c[:, 1:]) / omega[:, 1:]
    psi[:, 0] = (d - c)[:, 0]
    return chi, psi


# ------------------------------------------------------------------ Black-76 helpers


def black76(forward, strike, maturity, vol, discount, is_call=True):
    """Undiscounted-forward Black price, discounted. Used for implied volatility only."""
    forward = np.asarray(forward, dtype=float)
    strike = np.asarray(strike, dtype=float)
    vol = np.asarray(vol, dtype=float)
    total = vol * np.sqrt(maturity)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(forward / strike) + 0.5 * total**2) / total
        d2 = d1 - total
    call = discount * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
    put = discount * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))
    return np.where(np.asarray(is_call, dtype=bool), call, put)


def implied_vol(price, forward, strike, maturity, discount, is_call=True,
                lo: float = 1e-4, hi: float = 5.0) -> float:
    """Black implied volatility by bisection on a bracketed root.

    Returns NaN rather than raising when the price sits outside the no-arbitrage bounds,
    because a stale quote in a chain should drop out of a fit diagnostic instead of stopping
    it.
    """
    intrinsic = discount * max(
        (forward - strike) if is_call else (strike - forward), 0.0
    )
    upper_bound = discount * (forward if is_call else strike)
    if not (intrinsic - 1e-10 < price < upper_bound):
        return float("nan")

    def objective(vol):
        return float(black76(forward, strike, maturity, vol, discount, is_call)) - price

    try:
        return float(brentq(objective, lo, hi, xtol=1e-8, maxiter=200))
    except ValueError:
        return float("nan")


# ------------------------------------------------------------------ calibration


DEFAULT_BOUNDS = {
    "v0": (1e-4, 1.0),
    "kappa": (0.05, 10.0),
    "theta": (1e-4, 0.5),
    "xi": (0.01, 3.0),
    "rho": (-0.995, 0.10),
}

DEFAULT_STARTS = (
    HestonParameters(v0=0.030, kappa=1.5, theta=0.045, xi=0.60, rho=-0.70),
    HestonParameters(v0=0.020, kappa=0.8, theta=0.060, xi=0.90, rho=-0.80),
    HestonParameters(v0=0.045, kappa=3.0, theta=0.035, xi=0.40, rho=-0.60),
    HestonParameters(v0=0.025, kappa=0.4, theta=0.080, xi=1.20, rho=-0.85),
)


def _pack(p: HestonParameters) -> np.ndarray:
    return np.array([p.v0, p.kappa, p.theta, p.xi, p.rho])


def _unpack(z: np.ndarray) -> HestonParameters:
    return HestonParameters(v0=z[0], kappa=z[1], theta=z[2], xi=z[3], rho=z[4])


def calibrate(
    quotes,
    starts=DEFAULT_STARTS,
    bounds=None,
    n_terms: int = 192,
) -> tuple[HestonParameters, dict]:
    """Fit the five parameters to a cleaned chain.

    ``quotes`` is a DataFrame with columns maturity, strike, forward, discount, is_call,
    price and vega. Errors are weighted by one over vega, which turns a price error into an
    approximate volatility error: a far out-of-the-money option worth eighty cents and an
    at-the-money option worth four hundred dollars then count roughly equally, which is what
    a fit meant to reproduce a surface should do. Weighting prices directly would fit the
    at-the-money options and ignore the tail that this liability lives in.

    Several starts are run because the objective has local minima that differ mainly in how
    they split the skew between rho and xi. The best by cost wins and the spread across
    starts is reported, since a wide spread means the surface does not identify the split.
    """
    import pandas as pd  # local import keeps the module importable without pandas

    if not isinstance(quotes, pd.DataFrame):
        raise TypeError("quotes must be a DataFrame")
    needed = {"maturity", "strike", "forward", "discount", "is_call", "price", "vega"}
    missing = needed - set(quotes.columns)
    if missing:
        raise ValueError(f"quotes is missing {sorted(missing)}")
    if quotes.empty:
        raise ValueError("no quotes to calibrate to")

    bounds = bounds or DEFAULT_BOUNDS
    lower = np.array([bounds[k][0] for k in ("v0", "kappa", "theta", "xi", "rho")])
    upper = np.array([bounds[k][1] for k in ("v0", "kappa", "theta", "xi", "rho")])

    slices = [
        (float(t), group)
        for t, group in quotes.groupby("maturity", sort=True)
    ]
    weights = 1.0 / np.maximum(quotes["vega"].to_numpy(dtype=float), 1e-6)
    scale = np.sqrt(weights.sum())

    def residuals(z):
        params = _unpack(np.clip(z, lower, upper))
        out = []
        for maturity, group in slices:
            model = cos_price(
                params,
                float(group["forward"].iloc[0]),
                group["strike"].to_numpy(dtype=float),
                maturity,
                float(group["discount"].iloc[0]),
                group["is_call"].to_numpy(dtype=bool),
                n_terms=n_terms,
            )
            error = (model - group["price"].to_numpy(dtype=float)) / np.maximum(
                group["vega"].to_numpy(dtype=float), 1e-6
            )
            out.append(error)
        return np.concatenate(out) / scale

    attempts = []
    for start in starts:
        try:
            fit = least_squares(
                residuals,
                _pack(start),
                bounds=(lower, upper),
                xtol=1e-10,
                ftol=1e-10,
                max_nfev=400,
            )
        except ValueError:
            continue
        attempts.append((float(fit.cost), _unpack(fit.x), int(fit.nfev)))

    if not attempts:
        raise RuntimeError("every calibration start failed")
    attempts.sort(key=lambda row: row[0])
    cost, best, nfev = attempts[0]

    diagnostics = {
        "n_quotes": int(len(quotes)),
        "n_expiries": int(len(slices)),
        "cost": cost,
        "function_evaluations": nfev,
        "starts_tried": len(attempts),
        "cost_spread_across_starts": float(attempts[-1][0] - attempts[0][0]),
        "theta_spread_across_starts": float(
            max(a[1].theta for a in attempts) - min(a[1].theta for a in attempts)
        ),
        "rho_spread_across_starts": float(
            max(a[1].rho for a in attempts) - min(a[1].rho for a in attempts)
        ),
    }
    return best, diagnostics
