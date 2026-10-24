"""Known-answer tests for the market models.

Each of these has an answer that does not come from the code being tested. The COS pricer is
checked against Black-Scholes in the limit where Heston becomes Black-Scholes; the Hull-White
bond formula is checked against the curve it was built from and against its own simulation;
the simulator is checked against the two martingale identities and against the COS pricer with
rates switched off. A test that only checks the code against itself would pass on a sign error
in the characteristic function, which is the error this module exists to catch.
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np

from tests.checks import approx, raises
from vahedge.market.curves import ZeroCurve, bootstrap, fit_curve, fit_nss
from vahedge.market.heston_cos import (
    HestonParameters,
    black76,
    characteristic_function,
    cos_price,
    implied_vol,
    log_return_skewness,
)
from vahedge.market.hull_white import HullWhite, calibrate_historical
from vahedge.market.simulate import (
    Correlations,
    MarketPaths,
    SubAccountMix,
    martingale_report,
    simulate,
)

FLAT_PAR = np.array([0.04] * 8)
PAR_TENORS = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])


def _curve():
    return fit_curve(bootstrap(PAR_TENORS, np.array([0.035, 0.037, 0.039, 0.042, 0.044,
                                                     0.045, 0.048, 0.049])))


def _heston():
    return HestonParameters(v0=0.0199, kappa=4.80, theta=0.0470, xi=1.780, rho=-0.588)


# ---------------------------------------------------------------- curves


def test_bootstrap_round_trips_to_the_par_yields_it_was_built_from():
    curve = bootstrap(PAR_TENORS, FLAT_PAR)
    for tenor in PAR_TENORS:
        assert curve.par_equivalent(float(tenor)) == approx(0.04, abs=1e-10)


def test_bootstrap_rejects_a_curve_that_cannot_price():
    with raises(ValueError, match="non-positive discount factor"):
        bootstrap(np.array([1.0, 2.0]), np.array([0.05, 5.0]))


def test_nss_forward_is_the_derivative_of_the_zero_curve():
    """f(0,t) = d/dt [t * R(t)]. Checked numerically, which catches a sign or factor error in
    the analytic forward that the fit itself would never reveal."""
    curve = _curve()
    t = np.array([0.5, 1.0, 3.0, 10.0, 25.0, 40.0])
    h = 1e-5
    numerical = ((t + h) * curve.zero(t + h) - (t - h) * curve.zero(t - h)) / (2 * h)
    # approx has to sit on the left: with an ndarray on the left numpy's own comparison
    # wins and returns an array of booleans, which is truthy element by element.
    assert approx(numerical, abs=2e-8) == curve.instantaneous_forward(t)


def test_nss_extrapolates_near_the_last_quoted_zero_rate():
    """An unanchored fit on a real curve extrapolated to a long-run level of minus twenty-six
    percent while fitting the quoted tenors to three basis points. The anchored fit has to
    stay near the thirty-year rate well past it."""
    curve = _curve()
    thirty = float(curve.zero(30.0))
    for tenor in (40.0, 50.0, 60.0):
        assert abs(float(curve.zero(tenor)) - thirty) < 0.005


def test_nss_scalar_input_gives_a_scalar():
    curve = _curve()
    assert np.ndim(curve.zero(5.0)) == 0
    assert np.ndim(curve.discount(5.0)) == 0


def test_fit_nss_needs_more_points_than_parameters():
    with raises(ValueError, match="at least six"):
        fit_nss(np.array([1.0, 2.0]), np.array([0.03, 0.04]))


def test_the_rate_shock_is_parallel_in_both_the_zero_rate_and_the_forward():
    """What makes shifting the fitted level the right way to apply Item 7A's shock.

    The alternative - shift the quoted par yields and bootstrap again - is what a desk does
    and it cannot be done to this curve without re-fitting. The re-fit is the problem: on the
    committed curve a 100bp par shift arrives as 98bp of zero-rate shift at five years and
    105bp at thirty, and the two legs of a central difference land on fits of different
    quality, so part of the measured sensitivity is the fitting algorithm. A level shift moves
    R(t) and f(0,t) by exactly the applied amount at every maturity and is exactly symmetric,
    and this test is what would notice if it stopped doing either.
    """
    from dataclasses import replace

    base = _curve()
    maturities = np.array([0.5, 1.0, 5.0, 10.0, 30.0, 45.0, 60.0])
    for bp in (50.0, 100.0):
        up = replace(base, beta0=base.beta0 + bp / 10000.0)
        down = replace(base, beta0=base.beta0 - bp / 10000.0)
        assert approx(base.zero(maturities) + bp / 10000.0, abs=1e-15) == up.zero(maturities)
        assert approx(base.instantaneous_forward(maturities) + bp / 10000.0,
                      abs=1e-15) == up.instantaneous_forward(maturities)
        centred = 0.5 * (up.zero(maturities) - down.zero(maturities))
        assert approx(np.full(maturities.size, bp / 10000.0), abs=1e-15) == centred


# ---------------------------------------------------------------- Heston, COS


def test_cos_matches_black_scholes_when_volatility_of_variance_vanishes():
    """Heston with xi to zero and v0 equal to theta is Black-Scholes at sqrt(theta). This is
    the one test that would catch a wrong characteristic function."""
    forward, maturity = 7702.53, 1.0
    discount = float(np.exp(-0.04 * maturity))
    strikes = np.array([5800.0, 6500.0, 7000.0, 7702.53, 8200.0, 8450.0])
    is_call = strikes >= forward
    for vol in (0.12, 0.20, 0.35):
        params = HestonParameters(v0=vol**2, kappa=2.0, theta=vol**2, xi=1e-5, rho=-0.5)
        model = cos_price(params, forward, strikes, maturity, discount, is_call, n_terms=512)
        closed_form = black76(forward, strikes, maturity, vol, discount, is_call)
        assert np.max(np.abs(model - closed_form) / closed_form) < 5e-4


def test_cos_holds_put_call_parity_out_to_thirty_years():
    """C - P = D (F - K), across the strike range and out past every expiry in the chain.

    The textbook characteristic function fails this past a few years because its complex
    logarithm crosses a branch cut; the little-trap form does not, and the longest expiry in
    the chain is over five years. The tolerance is a hundredth of a basis point of the forward,
    which is what the pricer delivers at these parameters - an earlier version of this
    test allowed half a per cent of the price, thirty times looser, and a tolerance
    that loose is a test that cannot fail for the reason it exists.
    """
    params, forward = _heston(), 7702.53
    for maturity in (0.05, 1.0, 5.25, 10.0, 30.0):
        discount = float(np.exp(-0.04 * maturity))
        strikes = forward * np.array([0.6, 0.8, 0.95, 1.0, 1.05, 1.25, 1.6])
        call = cos_price(params, forward, strikes, maturity, discount, True, n_terms=512)
        put = cos_price(params, forward, strikes, maturity, discount, False, n_terms=512)
        residual = call - put - discount * (forward - strikes)
        assert np.abs(residual).max() < 1e-5 * forward


def test_cos_prices_nothing_impossible_anywhere_the_calibration_can_search():
    """No negative price and no NaN across the fit's own parameter bounds.

    This is the test the parity check above cannot be: parity at the calibrated point says
    nothing about the corners a least-squares search passes through, and the call payoff's
    cosine coefficient integrates exp(y) over the upper half of the truncation range, so at a
    dispersed enough variance distribution it comes out enormous and the sum over terms cannot
    cancel it back to a price. Before cos_price learned to check its own call against parity,
    a third of the corners below broke parity by more than a basis point of the forward, a
    sixth returned a negative price, and the far ones overflowed to NaN - all of it reaching a
    least-squares objective as a residual it would steer by.
    """
    from vahedge.market.heston_cos import DEFAULT_BOUNDS, PARITY_TOLERANCE

    forward = 7702.53
    corners = [(DEFAULT_BOUNDS[name][i] for name in ("v0", "kappa", "theta", "xi", "rho"))
               for i in (0, 1)]
    grid = itertools.product(
        (0.005, 0.08, 0.30), (0.05, 1.0, 10.0), (0.005, 0.20, 0.50),
        (0.05, 1.78, 3.0), (-0.995, 0.10),
    )
    for v0, kappa, theta, xi, rho in list(grid) + [tuple(c) for c in corners]:
        params = HestonParameters(v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)
        for maturity in (0.142466, 1.0, 3.230137):
            discount = float(np.exp(-0.04 * maturity))
            strikes = forward * np.array([0.75, 1.0, 1.10])
            call = cos_price(params, forward, strikes, maturity, discount, True, n_terms=192)
            put = cos_price(params, forward, strikes, maturity, discount, False, n_terms=192)
            where = f"v0={v0} kappa={kappa} theta={theta} xi={xi} rho={rho} T={maturity}"
            assert np.all(np.isfinite(call)) and np.all(np.isfinite(put)), where
            assert call.min() >= 0.0 and put.min() >= 0.0, where
            residual = call - put - discount * (forward - strikes)
            assert np.abs(residual).max() <= PARITY_TOLERANCE * discount * forward, where


def test_the_call_is_priced_directly_everywhere_the_committed_chain_asks():
    """The parity fallback must not be quietly standing in for the whole chain.

    Parity loses relative precision on a small out-of-the-money call, so it is the fallback and
    not the method. This asserts the direct expansion is what every committed quote is priced
    with, which is also what makes the guard free of any effect on a committed number.
    """
    from vahedge.market.heston_cos import PARITY_TOLERANCE

    params = _heston()
    forward, discount = 7702.53, 0.98
    for maturity in (0.142466, 1.0, 3.230137):
        strikes = forward * np.array([1.0, 1.02, 1.05, 1.10])
        call = cos_price(params, forward, strikes, maturity, discount, True, n_terms=192)
        put = cos_price(params, forward, strikes, maturity, discount, False, n_terms=192)
        implied = put + discount * (forward - strikes)
        gap = np.abs(call - implied)
        # Non-zero means the direct value was kept; zero would mean parity was substituted.
        assert gap.min() > 0.0
        assert gap.max() < PARITY_TOLERANCE * discount * forward


def test_cos_converges_in_the_number_of_terms():
    params, forward, maturity = _heston(), 7702.53, 2.0
    discount = float(np.exp(-0.08))
    reference = float(cos_price(params, forward, np.array([6000.0]), maturity, discount,
                                False, n_terms=1024)[0])
    for n_terms in (192, 256, 512):
        value = float(cos_price(params, forward, np.array([6000.0]), maturity, discount,
                                False, n_terms=n_terms)[0])
        assert abs(value / reference - 1.0) < 1e-6


def test_characteristic_function_is_one_at_zero():
    """phi(0) is the expectation of one. A characteristic function that misses it is
    mis-normalised, and every price built on it is wrong by the same factor."""
    value = complex(characteristic_function(np.array([0.0]), _heston(), 3.0)[0])
    assert approx(1.0, abs=1e-12) == value.real
    assert approx(0.0, abs=1e-12) == value.imag


def test_skewness_from_the_characteristic_function_matches_a_simulated_sample():
    """The skew check run beside the calibration reads a third moment off the characteristic
    function, and the simulator reaches the same number by a completely different route. Across
    ten seeds at this path count the sample skewness scatters with a standard deviation of 0.046
    and sits
    no further than 0.085 from the closed form, so the tolerance here is wide enough to be
    quiet and far too tight to survive a sign error or a mis-scaled rho."""
    params = _heston()
    nss = fit_curve(ZeroCurve(tenors=np.arange(0.5, 30.5, 0.5), zero_rates=np.full(60, 0.04)))
    paths = simulate(params, HullWhite(a=0.27, sigma=1e-9, curve=nss),
                     Correlations(params.rho, 0.0), SubAccountMix.all_equity(),
                     n_years=1, n_paths=200000, seed=23)
    sample = np.log(paths.index_growth[:, 0])
    centred = sample - sample.mean()
    simulated = float((centred**3).mean() / (centred**2).mean() ** 1.5)
    assert log_return_skewness(params, 1.0) == approx(simulated, abs=0.15)


def test_skewness_vanishes_as_the_volatility_of_variance_does():
    """With a deterministic variance the log return is exactly normal, whatever rho is set to,
    so the skewness has to go to zero in proportion to xi. Nothing in the stochastic-volatility
    part of the characteristic function is exercised in that limit, which is the point.

    The sweep starts at a tenth rather than at the calibrated 1.78 because proportionality is
    only the leading term: at the fitted value the higher-order terms take ten per cent out of
    the ratio, which is a fact about Heston and not something to test around.
    """
    previous = None
    for xi in (0.1, 0.01, 0.001):
        skewness = abs(log_return_skewness(
            HestonParameters(v0=0.0199, kappa=4.80, theta=0.0470, xi=xi, rho=-0.588), 1.0))
        if previous is not None:
            assert skewness == approx(0.1 * previous, rel=0.05)
        previous = skewness
    assert previous < 0.002


def test_skewness_steepens_as_the_leverage_correlation_does():
    """A one-sided check on the one parameter the volatility indices cannot see. The variance
    term structure is identical along this sweep; only the shape of the density moves."""
    values = [log_return_skewness(
        HestonParameters(v0=0.0199, kappa=4.80, theta=0.0470, xi=1.780, rho=rho), 0.5)
        for rho in (-0.9, -0.588, -0.2)]
    assert values[0] < values[1] < values[2] < 0.0


def test_skewness_does_not_depend_on_the_difference_step():
    """The default step is the middle of a flat region. A step that drifts out of it would move
    the number without failing anything, so pin the width of the region rather than the value.

    The region's edges are pinned too, and that half is the newer one: the docstring used to
    claim the number was flat to six figures four decades either side of the default, and it is
    flat over about one decade. Four decades below, the third difference has lost the
    cancellation entirely - the second difference comes back negative and the function refuses
    rather than return a skewness built on it. Asserting where the flatness stops is what keeps
    that claim from being made again.
    """
    params = _heston()
    reference = log_return_skewness(params, 30.0 / 365.0)
    for step in (1e-2, 3e-3, 3e-4, 1e-4):
        assert log_return_skewness(params, 30.0 / 365.0, step=step) == approx(reference, rel=1e-4)

    with raises(ValueError, match="non-positive variance"):
        log_return_skewness(params, 30.0 / 365.0, step=1e-7)
    far = log_return_skewness(params, 30.0 / 365.0, step=1e-6)
    assert abs(far / reference - 1.0) > 1e-2, (
        f"a step a thousandth of the default agreed to {abs(far / reference - 1.0):.1e}, so the "
        "flat region is wider than the docstring now says and the prose needs remeasuring"
    )


def test_the_parity_guard_has_less_headroom_than_it_reads_like():
    """Both sides of the tolerance, because only one of them had ever been checked.

    `test_cos_call_and_put_satisfy_parity_across_the_fit_bounds` asserts the gap stays under
    PARITY_TOLERANCE. Nothing asserted it stays anywhere near it, and the comment beside the
    constant claimed four orders of magnitude of headroom when the true figure is a factor of
    seven - at the 0.39-year expiry the direct call sits 0.137 basis points of the forward from
    parity against a tolerance of one. That matters in one direction only, and it is the
    direction a well-meaning change goes: dropping the term count or tightening the truncation
    costs accuracy here, and at a factor of seven the guard starts substituting parity on quotes
    that were being priced directly - silently, since parity is a legitimate answer.

    So this fails if the gap grows past a third of the tolerance, and also if someone widens the
    tolerance far enough that it stops discriminating.
    """
    from vahedge.market.heston_cos import PARITY_TOLERANCE

    params, forward = _heston(), 7702.53
    strikes = forward * np.array([0.75, 0.85, 0.95, 1.00, 1.05, 1.10])
    worst = 0.0
    for maturity in (0.049315, 0.142466, 0.391781, 1.0, 2.0, 3.230137, 5.227397):
        discount = float(np.exp(-0.04 * maturity))
        call = cos_price(params, forward, strikes, maturity, discount, True, n_terms=192)
        put = cos_price(params, forward, strikes, maturity, discount, False, n_terms=192)
        gap = np.abs(call - put - discount * (forward - strikes)) / (discount * forward)
        worst = max(worst, float(gap.max()))

    assert worst == approx(1.3708e-05, rel=0.02), (
        f"the direct call's worst distance from parity moved to {worst:.4e} of the forward"
    )
    assert worst < PARITY_TOLERANCE / 3.0, (
        f"the gap is {worst:.2e} against a tolerance of {PARITY_TOLERANCE:.0e}, so the guard is "
        "close enough to firing that parity may be standing in for quotes that do not need it"
    )
    assert PARITY_TOLERANCE < 100 * worst, (
        f"a tolerance of {PARITY_TOLERANCE:.0e} against a gap of {worst:.2e} is slack enough "
        "that a badly conditioned call would pass the check"
    )


def test_the_put_floor_is_reached_inside_the_fit_bounds():
    """The floor in `cos_price` is a live branch, not a defensive one.

    Three audit passes found a public function with no caller and a docstring citing it as the
    reason for a convention, so a guarded branch whose guard never fires is worth proving one way
    or the other. Over a grid on the fit's own bounds, at the strikes and expiries this chain
    has, the expansion comes out under the put's intrinsic value at three and a half per cent of
    the points, the worst of them 229 index points under - which is the number the comment beside
    the floor now quotes. The old comment said forty-five.

    The floored output is checked at every point as well, so this covers the branch's effect and
    not only its reachability.
    """
    from vahedge.market.heston_cos import DEFAULT_BOUNDS, _cumulants, _put_coefficients

    def expansion_only(params, forward, strikes, maturity, discount, n_terms=192):
        """The put before the floor, rebuilt from the same pieces cos_price uses."""
        c1, c2 = _cumulants(params, maturity)
        width = 12.0 * np.sqrt(abs(c2))
        x = np.log(forward / strikes)
        lower, upper = x + c1 - width, x + c1 + width
        k = np.arange(n_terms)
        u = k[None, :] * np.pi / (upper - lower)[:, None]
        with np.errstate(over="ignore", invalid="ignore"):
            density = characteristic_function(u, params, maturity)
            unit = np.real(density * np.exp(1j * u * (x[:, None] - lower[:, None])))
            unit[:, 0] *= 0.5
            return discount * np.sum(unit * _put_coefficients(lower, upper, k), axis=1) * strikes

    forward = 7702.53
    strikes = forward * np.array([0.75, 0.85, 0.95, 1.00, 1.05, 1.10])
    grid = itertools.product(
        (DEFAULT_BOUNDS["v0"][0], 0.08, DEFAULT_BOUNDS["v0"][1]),
        (DEFAULT_BOUNDS["kappa"][0], 1.0, DEFAULT_BOUNDS["kappa"][1]),
        (DEFAULT_BOUNDS["theta"][0], 0.20, DEFAULT_BOUNDS["theta"][1]),
        (DEFAULT_BOUNDS["xi"][0], 1.78, DEFAULT_BOUNDS["xi"][1]),
        (DEFAULT_BOUNDS["rho"][0], -0.5, DEFAULT_BOUNDS["rho"][1]),
    )
    under, evaluated, worst = 0, 0, 0.0
    for v0, kappa, theta, xi, rho in grid:
        params = HestonParameters(v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)
        for maturity in (0.142466, 0.391781, 1.0, 3.230137):
            discount = float(np.exp(-0.04 * maturity))
            intrinsic = discount * np.maximum(strikes - forward, 0.0)
            shortfall = expansion_only(params, forward, strikes, maturity, discount) - intrinsic
            floored = cos_price(params, forward, strikes, maturity, discount, False, n_terms=192)
            assert np.all(floored >= intrinsic - 1e-9), f"floor failed at {params}, T={maturity}"
            evaluated += len(strikes)
            under += int((shortfall < -1e-9).sum())
            worst = min(worst, float(np.nanmin(shortfall)))

    share = under / evaluated
    assert 0.02 < share < 0.06, (
        f"the floor binds at {share:.1%} of {evaluated} points, against the three and a half "
        "per cent the comment in cos_price quotes"
    )
    assert worst == approx(-228.8, rel=0.02), (
        f"the worst shortfall under intrinsic moved to {worst:.1f} index points"
    )


def test_implied_vol_inverts_black76():
    forward, strike, maturity, discount = 7702.53, 6500.0, 1.5, 0.94
    for vol in (0.10, 0.22, 0.45):
        price = float(black76(forward, strike, maturity, vol, discount, False))
        assert implied_vol(price, forward, strike, maturity,
                           discount, False) == approx(vol, abs=1e-6)


def test_implied_vol_returns_nan_outside_the_arbitrage_bounds():
    assert np.isnan(implied_vol(-1.0, 7702.53, 6500.0, 1.0, 0.96, False))


def test_heston_rejects_impossible_parameters():
    with raises(ValueError, match="positive"):
        HestonParameters(v0=-0.01, kappa=1.0, theta=0.04, xi=0.5, rho=-0.7)
    with raises(ValueError, match="rho"):
        HestonParameters(v0=0.04, kappa=1.0, theta=0.04, xi=0.5, rho=-1.0)


# ---------------------------------------------------------------- Hull-White


def test_hull_white_reproduces_the_initial_curve_exactly():
    """P(0,T) from the bond formula has to be the curve's own discount factor. If it is not,
    the model is not arbitrage-free against the curve it was fitted to."""
    curve = _curve()
    model = HullWhite(a=0.27, sigma=0.0114, curve=curve)
    maturities = np.array([1.0, 5.0, 10.0, 30.0, 45.0])
    r0 = float(model.alpha(0.0))
    assert approx(curve.discount(maturities), rel=1e-12) == model.bond_price(0.0, maturities, r0)


def test_hull_white_bond_price_matches_its_own_simulation():
    """Monte Carlo E[exp(-integral r)] against the closed form. This is the test that catches
    a wrong deterministic drift or a wrong variance in the exact stepping scheme."""
    curve = _curve()
    model = HullWhite(a=0.27, sigma=0.0114, curve=curve)
    heston = HestonParameters(v0=0.04, kappa=2.0, theta=0.04, xi=1e-5, rho=-0.5)
    paths = simulate(heston, model, Correlations(-0.5, 0.0), SubAccountMix.all_equity(),
                     n_years=20, n_paths=40000, seed=11, steps_per_year=12)
    report = martingale_report(paths, model, SubAccountMix.all_equity())
    bond = report[report["quantity"] == "bond"]
    assert float(bond["z_score"].abs().max()) < 3.0


def test_hull_white_bond_price_is_a_martingale_from_a_future_date():
    """exp(-int_0^t r) P(t,T) has time-zero expectation P(0,T), evaluated in closed form.

    The simulation test above only reaches bonds seen from today, and the nested valuation
    rebuilds the whole curve at a node from that node's short rate - so A(t,T) for t well
    inside the projection is load-bearing and nothing was checking it. The moments of the
    state and of its integral are written out here from the SDE rather than taken from the
    module, so the two sides are independent.
    """
    curve = _curve()
    for a, sigma in ((0.27, 0.0114), (0.05, 0.02)):
        model = HullWhite(a=a, sigma=sigma, curve=curve)
        for t in (0.5, 10.0, 25.0):
            var_x = sigma**2 * (1.0 - np.exp(-2 * a * t)) / (2 * a)
            var_integral = (sigma**2 / a**2) * (
                t - 2 * (1 - np.exp(-a * t)) / a + (1 - np.exp(-2 * a * t)) / (2 * a)
            )
            covariance = (sigma**2 / a) * (
                (1 - np.exp(-a * t)) / a - (1 - np.exp(-2 * a * t)) / (2 * a)
            )
            # exp(-int_0^t alpha), with alpha(s) = f(0,s) + sigma^2 (1 - e^{-as})^2 / (2a^2)
            grid = np.linspace(0.0, t, 100_001)
            convexity = (sigma**2 / (2 * a**2)) * np.trapezoid(
                (1.0 - np.exp(-a * grid)) ** 2, grid
            )
            deterministic = float(curve.discount(t)) * np.exp(-convexity)

            alpha = float(model.alpha(t))
            for maturity in (t + 0.5, t + 20.0):
                # Recover A and B from the module's own prices at two short rates.
                near = float(np.atleast_1d(model.bond_price(t, np.array([maturity]), alpha))[0])
                far = float(np.atleast_1d(
                    model.bond_price(t, np.array([maturity]), alpha + 0.01))[0])
                b = -np.log(far / near) / 0.01
                expectation = (
                    deterministic * near
                    * np.exp(0.5 * (var_integral + b**2 * var_x + 2.0 * b * covariance))
                )
                assert expectation == approx(float(curve.discount(maturity)), rel=1e-9)


def test_hull_white_rejects_zero_mean_reversion():
    with raises(ValueError, match="mean reversion"):
        HullWhite(a=0.0, sigma=0.01, curve=_curve())


def test_historical_calibration_recovers_a_known_process():
    """Simulate an AR(1) with a known mean reversion and volatility, then fit it back."""
    a_true, sigma_true, dt = 0.30, 0.012, 1.0 / 252
    rng = np.random.default_rng(4)
    n = 60000
    phi = np.exp(-a_true * dt)
    noise_sd = sigma_true * np.sqrt((1.0 - phi**2) / (2.0 * a_true))
    rates = np.empty(n)
    rates[0] = 0.04
    for i in range(1, n):
        rates[i] = 0.04 + phi * (rates[i - 1] - 0.04) + noise_sd * rng.standard_normal()
    a_fit, sigma_fit, diagnostics = calibrate_historical(rates, dt=dt)
    assert a_fit == approx(a_true, rel=0.15)
    assert sigma_fit == approx(sigma_true, rel=0.05)
    assert not diagnostics["mean_reversion_floor_bound"]


# ---------------------------------------------------------------- the joint simulator


def test_discounted_index_and_sub_account_are_martingales():
    curve = _curve()
    model = HullWhite(a=0.27, sigma=0.0114, curve=curve)
    mix = SubAccountMix(equity=0.7235, bond=0.0834, balanced=0.1832, money_market=0.0099)
    paths = simulate(_heston(), model, Correlations(-0.588, 0.106), mix,
                     n_years=30, n_paths=40000, seed=5)
    report = martingale_report(paths, model, mix)
    assert float(report["z_score"].abs().max()) < 3.5


def test_simulator_matches_the_cos_pricer_with_rates_switched_off():
    """With the short rate held still, the simulated index is the model the COS pricer prices,
    so a European put has to come out the same both ways. This is the joint test: it fails if
    the variance scheme, the log-price update or the martingale correction is wrong.

    It is also the only check on ``steps_per_year``, which every valuation in the project runs
    at its default of 24. The module docstring used to justify that default with a measured
    discretisation bias at 12 steps and cite a script this repository does not contain; carried
    out to three million paths at one, five and ten years, no step count from 12 to 104
    separates from any other, so the default rests on the variance process's mean-reversion
    time constant and this test's job is only to show that 24 carries no bias the project can
    see. At the path count here that means inside about 1.2 standard errors.

    The pairing is ``antithetic_pairs``, not the first half against the second. With ten
    simulation blocks the two differ by only a few per cent on a put payoff, so this is not a
    tolerance that was hiding anything - but the inflated error was the looser branch of the
    tolerance below, and a test should not take its own slack from a bug the module it tests
    documents.
    """
    nss = fit_curve(ZeroCurve(tenors=np.arange(0.5, 30.5, 0.5),
                              zero_rates=np.full(60, 0.04)))
    model = HullWhite(a=0.27, sigma=1e-9, curve=nss)
    params = _heston()
    paths = simulate(params, model, Correlations(params.rho, 0.0), SubAccountMix.all_equity(),
                     n_years=5, n_paths=200000, seed=7)

    for years in (1, 5):
        discount = float(nss.discount(float(years)))
        forward = 1.0 / discount
        level = np.cumprod(paths.index_growth[:, :years], axis=1)[:, -1]
        for moneyness in (0.80, 1.00):
            strike = moneyness * forward
            pairs = paths.antithetic_pairs(np.maximum(strike - level, 0.0))
            monte_carlo = discount * float(pairs.mean())
            std_error = discount * float(pairs.std(ddof=1) / np.sqrt(pairs.size))
            closed_form = float(cos_price(params, forward, np.array([strike]), float(years),
                                          discount, False, n_terms=512)[0])
            assert abs(monte_carlo - closed_form) < max(3.0 * std_error, 0.006 * closed_form)


def test_variance_never_goes_negative():
    """The calibrated parameters violate the Feller condition, so the variance reaches zero.
    An Euler scheme would take it below."""
    model = HullWhite(a=0.27, sigma=0.0114, curve=_curve())
    paths = simulate(_heston(), model, Correlations(-0.588, 0.106), SubAccountMix.all_equity(),
                     n_years=20, n_paths=20000, seed=9)
    assert float(paths.variance.min()) >= 0.0


def test_antithetic_pairs_are_found_inside_their_own_simulation_block():
    """A path pairs with its mirror, which is not the row half an array away.

    Paths are simulated in independently seeded blocks and each block mirrors its own first
    half into its own second half, so above one block the halves of the whole array are not
    pairs at all. Pairing them anyway leaves the mean untouched - averaging pairs and averaging
    rows are the same sum - and throws away the variance reduction, which is the one thing the
    antithetic draw was for. It read as a Monte Carlo error a tenth too large on the
    forty-thousand-path row of the convergence table, and off the trend the rows below it set.
    """
    curve, heston = _curve(), _heston()
    model = HullWhite(a=0.27, sigma=0.0114, curve=curve)
    block = 1_000
    for n_paths in (block, 2 * block, 5 * block):
        paths = simulate(heston, model, Correlations(-0.5, 0.0), SubAccountMix.all_equity(),
                         n_years=3, n_paths=n_paths, seed=3, steps_per_year=4,
                         chunk_paths=block)
        assert paths.antithetic_block == min(block, n_paths)
        sample = paths.fund_growth.prod(axis=1)
        paired = paths.antithetic_pairs(sample)
        assert paired.size == n_paths // 2
        # Whichever way they are paired the mean is the same, which is why this went unnoticed.
        halves = 0.5 * (sample[:n_paths // 2] + sample[n_paths // 2:])
        assert paired.mean() == approx(halves.mean(), rel=1e-12)
        # And the real pairing is the tighter one as soon as there is more than one block.
        if n_paths > block:
            assert paired.std(ddof=1) < 0.95 * halves.std(ddof=1)
        else:
            assert approx(halves, rel=1e-12) == paired

    # Hand-built paths carry no block, so the whole array is one and the old behaviour stands.
    ones = np.ones((4, 2))
    bare = MarketPaths(fund_growth=ones, index_growth=ones, discount=ones, short_rate=ones,
                       variance=ones, zero_10y=ones, realised_variance=ones, seed=0,
                       antithetic=True)
    assert bare.antithetic_block == 0
    assert bare.antithetic_pairs(np.array([1.0, 2.0, 3.0, 4.0])).tolist() == [2.0, 3.0]
    assert replace(bare, antithetic=False).antithetic_pairs(np.arange(4.0)) is None


def test_the_martingale_report_takes_its_error_across_the_right_pairs():
    """The guard everything downstream rests on, measured against the right error bars.

    `martingale_report` fails a simulator at a z-score of three, so its standard error is the
    threshold and not a decoration. It took the first half of the path array against the second,
    which pairs a path with an unrelated one from another simulation block - the same defect
    `MarketPaths.antithetic_pairs` was written for, left in two more places. On the bond
    identity, where the antithetic pairing is nearly perfect, the wrong pairing left the error
    almost ten times too wide and every z-score about eight times too small. Nothing was ever
    reported wrong by it; it simply could not have reported anything wrong.

    So this asserts the direction that matters: on paths drawn in several blocks the reported
    error is strictly tighter than the across-halves figure. A report that drifts back to the
    old pairing cannot satisfy it.
    """
    curve, heston = _curve(), _heston()
    model = HullWhite(a=0.27, sigma=0.0114, curve=curve)
    block = 1_000
    paths = simulate(heston, model, Correlations(-0.5, 0.0), SubAccountMix.all_equity(),
                     n_years=3, n_paths=5 * block, seed=4, steps_per_year=4, chunk_paths=block)
    assert paths.antithetic_block == block
    report = martingale_report(paths, model, SubAccountMix.all_equity())

    for year in range(paths.n_years):
        sample = paths.discount[:, year]
        half = sample.size // 2
        across = 0.5 * (sample[:half] + sample[half:])
        loose = float(across.std(ddof=1) / np.sqrt(half))
        row = report[(report["quantity"] == "bond")
                     & (report["maturity"] == float(year + 1))].iloc[0]
        # The mean is pairing-invariant, which is exactly why this went unnoticed for so long.
        assert float(row["estimate"]) == approx(float(across.mean()), rel=1e-12)
        assert float(row["std_error"]) < 0.5 * loose, (
            f"at maturity {year + 1} the report's error is {row['std_error']:.3e} against "
            f"{loose:.3e} across halves, so the pairing has drifted back"
        )

    assert report["z_score"].abs().max() < 3.0


def test_antithetic_sampling_needs_an_even_path_count():
    model = HullWhite(a=0.27, sigma=0.0114, curve=_curve())
    with raises(ValueError, match="antithetic"):
        simulate(_heston(), model, Correlations(-0.588, 0.0), SubAccountMix.all_equity(),
                 n_years=2, n_paths=1001, seed=1)


def test_unreachable_equity_rate_correlation_is_refused():
    """With rho at -0.588 the equity Brownian has only sqrt(1 - rho^2) left to correlate with
    rates, so a correlation above that ceiling cannot be imposed and must not be faked."""
    model = HullWhite(a=0.27, sigma=0.0114, curve=_curve())
    with raises(ValueError, match="out of reach"):
        simulate(_heston(), model, Correlations(-0.588, 0.9), SubAccountMix.all_equity(),
                 n_years=2, n_paths=1000, seed=1, antithetic=False)


def test_sub_account_weights_must_sum_to_one():
    with raises(ValueError, match="sum to 1"):
        SubAccountMix(equity=0.5, bond=0.2, balanced=0.2, money_market=0.2)


def test_sub_account_equity_weight_includes_the_balanced_sleeve():
    mix = SubAccountMix(equity=0.7235, bond=0.0834, balanced=0.1832, money_market=0.0099)
    assert mix.equity_weight == approx(0.7235 + 0.6 * 0.1832)
    assert mix.equity_weight + mix.bond_weight + mix.money_market == approx(1.0)


def test_every_date_in_the_committed_panel_bootstraps():
    """A data-level regression, not a property of the algorithm.

    The curve history is built from a committed file, so the thing that can break is the file:
    a transcribed yield with a digit missing, or a date whose par curve is non-monotone in a
    way the bootstrap cannot price. Running it over every date once is cheap and it is the only
    check that would catch a bad input before it reached a valuation.
    """
    import pandas as pd

    from vahedge import paths
    from vahedge.market.curves import bootstrap
    from vahedge.market.state import PAR_SERIES

    panel = pd.read_csv(paths.FRED_PANEL, comment="#", parse_dates=["date"]).set_index("date")
    tenors = list(PAR_SERIES)
    quotes = panel.loc[panel["SP500"].notna(), [PAR_SERIES[t] for t in tenors]].dropna()
    assert len(quotes) > 2_000

    worst = 0.0
    for _, row in quotes.iterrows():
        par = row.to_numpy(dtype=float) / 100.0
        curve = bootstrap(tenors, par)
        worst = max(worst, max(abs(curve.par_equivalent(t) - p) for t, p in zip(tenors, par)))
    assert worst < 1e-6
