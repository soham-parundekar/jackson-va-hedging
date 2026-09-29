"""The index-linked annuity: payoff boundaries, closed forms, and the cap that funds it.

The payoff boundaries are the cheapest possible test and they catch the sign errors that matter:
a buffer implemented as a floor, or a cap applied to the wrong side. The closed form matters
more, because the credited return decomposes exactly into three European options and there is
no excuse for a simulation that disagrees with that.
"""

from __future__ import annotations

import numpy as np

from tests.checks import approx, raises
from vahedge.liability import rila
from vahedge.market.curves import ZeroCurve, fit_curve
from vahedge.market.heston_cos import HestonParameters, black76
from vahedge.market.hull_white import HullWhite
from vahedge.market.simulate import Correlations, SubAccountMix, simulate


def _heston():
    return HestonParameters(v0=0.0199, kappa=4.80, theta=0.0470, xi=1.780, rho=-0.588)


def _flat_curve(rate: float = 0.04):
    return fit_curve(ZeroCurve(tenors=np.arange(0.5, 30.5, 0.5), zero_rates=np.full(60, rate)))


def test_credited_return_hits_every_boundary_in_the_specification():
    """Buffer 10%, cap 12%. Index returns of -30%, -10%, -5%, 0, 5% and 20% must credit
    -20%, 0, 0, 0, 5% and 12%."""
    returns = np.array([-0.30, -0.10, -0.05, 0.0, 0.05, 0.20])
    expected = np.array([-0.20, 0.0, 0.0, 0.0, 0.05, 0.12])
    assert approx(expected, abs=1e-12) == rila.credited_return(returns, 0.10, 0.12)


def test_the_buffer_absorbs_exactly_its_own_depth_and_no_more():
    """At any loss beyond the buffer the insurer has absorbed the buffer and nothing further."""
    for buffer in (0.10, 0.20, 0.30):
        for index_return in (-0.35, -0.50, -0.80):
            credited = float(rila.credited_return(index_return, buffer, 0.12))
            assert credited == approx(index_return + buffer, abs=1e-12)


def test_a_bigger_buffer_is_always_worth_more_to_the_contract_holder():
    returns = np.linspace(-0.6, 0.6, 121)
    small = rila.credited_return(returns, 0.10, 0.15)
    large = rila.credited_return(returns, 0.25, 0.15)
    assert np.all(large >= small - 1e-12)


def test_the_credited_payoff_is_a_call_spread_less_a_put():
    """credited = call(1) - call(1 + c) - put(1 - b), checked payoff by payoff rather than by
    pricing it, so the identity is established before any model touches it."""
    buffer, cap = 0.15, 0.25
    index = np.linspace(0.3, 2.0, 341)
    returns = index - 1.0
    decomposition = (
        np.maximum(index - 1.0, 0.0)
        - np.maximum(index - (1.0 + cap), 0.0)
        - np.maximum((1.0 - buffer) - index, 0.0)
    )
    assert approx(decomposition, abs=1e-12) == rila.credited_return(returns, buffer, cap)


def test_the_black_scholes_value_matches_the_three_options_it_is_built_from():
    terms = rila.RilaTerms(buffer=0.10, cap=0.25, term_years=3.0)
    forward, discount, vol = 1.12, 0.89, 0.20
    by_hand = (
        float(black76(forward, 1.0, terms.term_years, vol, discount, True))
        - float(black76(forward, 1.25, terms.term_years, vol, discount, True))
        - float(black76(forward, 0.90, terms.term_years, vol, discount, False))
    )
    assert rila.value_black_scholes(forward, discount, vol, terms) == approx(by_hand, rel=1e-12)


def test_an_uncapped_contract_with_no_buffer_is_worth_the_forward_less_the_discount():
    """With the cap effectively infinite and no buffer, the credited return is just the index
    return, so its value collapses to D*(F-1). That is the one point on the surface where the
    answer is arithmetic rather than an option price."""
    terms = rila.RilaTerms(buffer=1e-9, cap=50.0, term_years=2.0)
    forward, discount = 1.09, 0.93
    value = rila.value_black_scholes(forward, discount, 0.20, terms)
    assert value == approx(discount * (forward - 1.0), abs=1e-6)


def test_the_simulation_reproduces_the_closed_form_with_rates_held_still():
    curve = _flat_curve()
    model = HullWhite(a=0.27, sigma=1e-9, curve=curve)
    params = _heston()
    paths = simulate(params, model, Correlations(params.rho, 0.0), SubAccountMix.all_equity(),
                     n_years=6, n_paths=200_000, seed=3)
    terms = rila.RilaTerms(buffer=0.10, cap=0.55, term_years=6.0)
    discount = float(curve.discount(6.0))
    simulated = rila.project(paths, terms)
    closed_form = rila.value_heston(params, 1.0 / discount, discount, terms)
    z_score = (simulated["embedded_derivative"] - closed_form) / simulated["std_error"]
    assert abs(z_score) < 3.0


def test_a_term_that_is_not_a_whole_number_of_years_is_refused():
    curve = _flat_curve()
    model = HullWhite(a=0.27, sigma=1e-9, curve=curve)
    params = _heston()
    paths = simulate(params, model, Correlations(params.rho, 0.0), SubAccountMix.all_equity(),
                     n_years=3, n_paths=1000, seed=3)
    with raises(ValueError, match="whole number"):
        rila.project(paths, rila.RilaTerms(buffer=0.1, cap=0.2, term_years=2.5))


def test_the_break_even_cap_makes_the_contract_worth_its_premium():
    """The solved cap has to satisfy the identity it was solved from, which is worth asserting
    separately because the identity is where the economics live."""
    terms = rila.RilaTerms(buffer=0.20, cap=0.3, term_years=1.0)
    forward, discount, vol = 1.03, 0.9662, 0.20
    funding_discount = 0.9562          # a general account earning a spread over Treasuries
    price = lambda cap: rila.value_black_scholes(
        forward, discount, vol, rila.RilaTerms(terms.buffer, cap, terms.term_years)
    )
    cap = rila.breakeven_cap(funding_discount, terms, price)
    assert price(cap) == approx(1.0 - funding_discount, abs=1e-9)
    assert 0.02 < cap < 1.0


def test_a_wider_buffer_forces_a_lower_cap():
    """The buffer and the cap are paid for out of the same funding, so protecting more of the
    downside has to cost upside. This is the trade the product is built around."""
    forward, discount, vol, funding = 1.03, 0.9662, 0.20, 0.9562
    caps = []
    for buffer in (0.10, 0.20, 0.30):
        price = lambda cap, b=buffer: rila.value_black_scholes(
            forward, discount, vol, rila.RilaTerms(b, cap, 1.0)
        )
        caps.append(rila.breakeven_cap(funding, rila.RilaTerms(buffer, 0.3, 1.0), price))
    assert caps[0] > caps[1] > caps[2]


def test_no_funding_means_no_cap_rather_than_a_silently_wrong_one():
    terms = rila.RilaTerms(buffer=0.10, cap=0.2, term_years=1.0)
    price = lambda cap: rila.value_black_scholes(1.03, 0.9662, 0.20,
                                                 rila.RilaTerms(0.10, cap, 1.0))
    with raises(ValueError, match="nothing to fund"):
        rila.breakeven_cap(1.0, terms, price)


def test_impossible_terms_are_refused():
    with raises(ValueError, match="buffer"):
        rila.RilaTerms(buffer=1.0, cap=0.1, term_years=1.0)
    with raises(ValueError, match="cap"):
        rila.RilaTerms(buffer=0.1, cap=0.0, term_years=1.0)
    with raises(ValueError, match="term_years"):
        rila.RilaTerms(buffer=0.1, cap=0.1, term_years=0.0)


def test_the_insurer_gains_on_a_rally_where_the_guarantee_book_loses():
    """The netting claim, made precise. The credited payoff rises with the index, so the
    insurer's obligation on a RILA rises in a rally - the opposite direction to a withdrawal
    guarantee, whose value to the insurer improves when markets rise."""
    terms = rila.RilaTerms(buffer=0.10, cap=0.25, term_years=3.0)
    discount, vol = 0.89, 0.20
    up = rila.value_black_scholes(1.12 * 1.10, discount, vol, terms)
    down = rila.value_black_scholes(1.12 * 0.90, discount, vol, terms)
    assert up > down
