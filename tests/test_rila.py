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


# ---------------------------------------------------------------- in-force segments and delta


def _paths(n_years: int = 6, n_paths: int = 200_000, rate: float = 0.04, seed: int = 3):
    params = _heston()
    return simulate(params, HullWhite(a=0.27, sigma=1e-9, curve=_flat_curve(rate)),
                    Correlations(params.rho, 0.0), SubAccountMix.all_equity(),
                    n_years=n_years, n_paths=n_paths, seed=seed)


def test_a_segment_partway_through_its_term_is_the_closed_form_on_what_is_left():
    """A point-to-point credit looks at two index levels and nothing between them, so three
    years into a six-year term the contract is the same contract on an index that has already
    moved. The growth to date goes into the forward, not into the buffer and cap: the payoff is
    homogeneous in the index, so scaling the forward scales the whole thing. Rescaling the
    strikes instead is the obvious move and it is wrong, which is how this test earned its
    keep."""
    curve, params = _flat_curve(), _heston()
    terms = rila.RilaTerms(buffer=0.10, cap=0.55, term_years=6.0)
    discount = float(curve.discount(3.0))
    remaining = rila.RilaTerms(buffer=terms.buffer, cap=terms.cap, term_years=3.0)
    paths = _paths()
    for realised in (0.80, 1.00, 1.30):
        simulated = rila.project(paths, terms, realised_growth=realised, elapsed_years=3)
        closed_form = rila.value_heston(params, realised / discount, discount, remaining)
        z_score = (simulated["embedded_derivative"] - closed_form) / simulated["std_error"]
        assert abs(z_score) < 3.0, f"realised {realised}: z {z_score:.2f}"


def test_a_segment_at_the_end_of_its_term_has_no_optionality_left():
    """With nothing remaining the credit is known, so the value is the credited return itself
    and the delta is zero. A path count of one would do; the point is that the code does not
    index past the end of the simulation."""
    terms = rila.RilaTerms(buffer=0.10, cap=0.55, term_years=6.0)
    for realised, expected in ((1.80, 0.55), (1.20, 0.20), (0.95, 0.0), (0.70, -0.20)):
        done = rila.project(_paths(n_paths=1000), terms,
                            realised_growth=realised, elapsed_years=6)
        assert approx(expected, abs=1e-12) == done["embedded_derivative"]
        assert approx(0.0, abs=1e-12) == done["std_error"]


def test_the_insurer_owes_more_when_the_index_rises():
    """The sign the whole netting argument rests on. A withdrawal guarantee gets cheaper in a
    rally; this gets dearer, so the exposure is positive on the same convention."""
    terms = rila.RilaTerms(buffer=0.10, cap=0.55, term_years=6.0)
    exposure = rila.equity_exposure(_paths(), terms)
    assert exposure["equity_exposure"] > 0.0


def test_the_cap_kills_the_exposure_and_the_buffer_does_not():
    """The asymmetry the netting argument depends on, and the one that reads backwards. A
    buffer absorbs the first slice of a loss and the holder bears everything past it, so a
    segment well below its buffer is still fully exposed; a segment above its cap has stopped
    moving entirely. At a 55% cap with a year to run, a segment up 90% is 90% capped and keeps
    about a tenth of its exposure, while one down 30% keeps two thirds - and one just under the
    cap carries more than a unit of account value, because the measure is per log move and
    carries the index ratio with it."""
    terms = rila.RilaTerms(buffer=0.10, cap=0.55, term_years=6.0)
    paths = _paths()
    at = {g: rila.equity_exposure(paths, terms, realised_growth=g, elapsed_years=5)
          for g in (0.70, 1.20, 1.90)}
    assert at[1.90]["share_capped"] > 0.85
    assert at[1.90]["equity_exposure_pct_of_account"] < 0.2
    # Through the buffer and still carrying most of its exposure, which is the point.
    assert at[0.70]["share_through_buffer"] > 0.9
    assert at[0.70]["equity_exposure_pct_of_account"] > 0.5
    assert at[1.20]["equity_exposure_pct_of_account"] > 1.0
    assert all(v["equity_exposure_pct_of_account"] > 0.0 for v in at.values())


def test_an_elapsed_term_longer_than_the_segment_is_refused():
    terms = rila.RilaTerms(buffer=0.10, cap=0.55, term_years=6.0)
    with raises(ValueError, match="outside a 6-year term"):
        rila.project(_paths(n_paths=1000), terms, elapsed_years=7)
    with raises(ValueError, match="positive"):
        rila.project(_paths(n_paths=1000), terms, realised_growth=0.0)


def test_the_ten_per_cent_repricing_is_not_the_delta_times_ten_per_cent():
    """Item 7A runs a 10% shock and a credit with a cap and a buffer in it is not linear over a
    move that size, so the two have to be reported separately. They also carry different units -
    a delta is per log move and a disclosed impact is per 10% move, a factor of one over ln(1.1)
    - and confusing them inflates a comparison by 10.5, which is large enough to look like a
    finding. This pins both the units and the curvature."""
    terms = rila.RilaTerms(buffer=0.10, cap=0.55, term_years=6.0)
    point = rila.equity_exposure(_paths(), terms, realised_growth=1.15, elapsed_years=4)

    # The insurer owes more when the index rises and less when it falls, on both measures.
    assert point["equity_up_10pct_pct_of_account"] > 0.0
    assert point["equity_down_10pct_pct_of_account"] < 0.0
    assert point["equity_exposure_pct_of_account"] > 0.0

    # A linear read of the delta would land near delta * ln(1.1); the true repricing is below it
    # on the way up, because the cap takes the top off a large move and nothing takes the
    # equivalent off a large fall.
    linear = point["equity_exposure_pct_of_account"] * np.log(1.1)
    assert point["equity_up_10pct_pct_of_account"] < linear
    assert abs(point["equity_down_10pct_pct_of_account"]) > point["equity_up_10pct_pct_of_account"]

    # And the scale: a 10% repricing is about a tenth of a per-log-move delta, never equal to it.
    assert point["equity_up_10pct_pct_of_account"] < 0.3 * point["equity_exposure_pct_of_account"]
