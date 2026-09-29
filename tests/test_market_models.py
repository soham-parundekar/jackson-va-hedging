"""Known-answer tests for the market models.

Each of these has an answer that does not come from the code being tested. The COS pricer is
checked against Black-Scholes in the limit where Heston becomes Black-Scholes; the Hull-White
bond formula is checked against the curve it was built from and against its own simulation;
the simulator is checked against the two martingale identities and against the COS pricer with
rates switched off. A test that only checks the code against itself would pass on a sign error
in the characteristic function, which is the error this module exists to catch.
"""

from __future__ import annotations

import numpy as np

from tests.checks import approx, raises
from vahedge.market.curves import ZeroCurve, bootstrap, fit_curve, fit_nss, flat_curve
from vahedge.market.heston_cos import (
    HestonParameters,
    black76,
    characteristic_function,
    cos_price,
    implied_vol,
)
from vahedge.market.hull_white import HullWhite, calibrate_historical
from vahedge.market.simulate import (
    Correlations,
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
    """At the forward a call and a put are worth the same. The textbook characteristic
    function fails this past a few years because its complex logarithm crosses a branch cut;
    the little-trap form does not, and the longest expiry in the chain is over five years."""
    params, forward = _heston(), 7702.53
    for maturity in (0.05, 1.0, 5.25, 10.0, 30.0):
        discount = float(np.exp(-0.04 * maturity))
        call = cos_price(params, forward, np.array([forward]), maturity, discount, True, n_terms=512)
        put = cos_price(params, forward, np.array([forward]), maturity, discount, False, n_terms=512)
        assert float(call[0] - put[0]) == approx(0.0, abs=5e-3 * max(1.0, float(call[0])))


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


def test_implied_vol_inverts_black76():
    forward, strike, maturity, discount = 7702.53, 6500.0, 1.5, 0.94
    for vol in (0.10, 0.22, 0.45):
        price = float(black76(forward, strike, maturity, vol, discount, False))
        assert implied_vol(price, forward, strike, maturity, discount, False) == approx(vol, abs=1e-6)


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
    the variance scheme, the log-price update or the martingale correction is wrong."""
    curve = flat_curve(0.04)
    nss = fit_curve(ZeroCurve(tenors=np.arange(0.5, 30.5, 0.5),
                              zero_rates=np.full(60, 0.04)))
    model = HullWhite(a=0.27, sigma=1e-9, curve=nss)
    params = _heston()
    paths = simulate(params, model, Correlations(params.rho, 0.0), SubAccountMix.all_equity(),
                     n_years=5, n_paths=200000, seed=7)
    del curve

    for years in (1, 5):
        discount = float(nss.discount(float(years)))
        forward = 1.0 / discount
        level = np.cumprod(paths.index_growth[:, :years], axis=1)[:, -1]
        for moneyness in (0.80, 1.00):
            strike = moneyness * forward
            payoff = np.maximum(strike - level, 0.0)
            half = payoff.size // 2
            pairs = 0.5 * (payoff[:half] + payoff[half:])
            monte_carlo = discount * float(pairs.mean())
            std_error = discount * float(pairs.std(ddof=1) / np.sqrt(half))
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
