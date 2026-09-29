"""The regression proxy and the nested standard that is the only real test of it.

Two tests here exist because of specific failures, and both failures had the same shape: the
proxy disagreed with the nested values by a large, one-directional amount, and every fit
diagnostic stayed green throughout because the fit was reproducing its target faithfully. The
target was wrong.

The first was conditioning. The recorded cash flows carry the probability that the contract is
still there to pay them, so a continuation value has to divide that out along with the discount
factor. Dividing out only the discount values a contract that might already have lapsed or
died, which by year twenty is about half the contract standing at the node.

The second was state. Rebuilding a node needs every path-dependent quantity the recursion
carries, not just the contract value and the benefit base. The death benefit base is cut by
withdrawals on its own schedule and ends up far below the guaranteed withdrawal base; setting
the two equal at a node turned an exhausted contract worth 0.34 of its base into one worth
0.96.

So the tests below are identities and independent recalculations rather than numbers copied
from a run. The strongest is the annuity one: an exhausted contract is a life annuity, a life
annuity has a closed form, and both the projection and the rebuilt node have to produce it.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from tests.checks import approx, raises
from vahedge.liability import behaviour, cohorts, gmwb, mortality
from vahedge.liability import terms as terms_module
from vahedge.market.curves import ZeroCurve, fit_curve
from vahedge.market.heston_cos import HestonParameters
from vahedge.market.simulate import Correlations, SubAccountMix, simulate
from vahedge.valuation import lsmc, nested
from vahedge.valuation.engine import MarketState, Valuer

PATHS = 4_000
SEED = 5150
_CACHE: dict = {}


def _state() -> MarketState:
    curve = fit_curve(ZeroCurve(tenors=np.arange(0.5, 30.5, 0.5),
                                zero_rates=np.linspace(0.035, 0.048, 60)))
    return MarketState(
        curve=curve,
        heston=HestonParameters(v0=0.0199, kappa=4.80, theta=0.0470, xi=1.780, rho=-0.588),
        correlations=Correlations(-0.588, 0.106),
        mix=SubAccountMix(equity=0.7235, bond=0.0834, balanced=0.1832, money_market=0.0099),
        mean_reversion=0.2685,
        rate_vol=0.01141,
        valuation_year=2025,
    )


def _projection():
    """One recorded projection, reused by every test in the file.

    A 70-year-old on a five-year deferral reaches exhaustion on a useful share of paths inside
    a twenty-five year horizon, which is what gives the fit something to fit at both ends.
    """
    if "projection" in _CACHE:
        return _CACHE["projection"]
    core = terms_module.load()[("flex_gmwb", "single", "core")]
    book = cohorts.single_contract(
        core, issue_age=70, base_contract_charge=0.0131, fund_expense=0.0095,
        premium=100.0, deferral_years=5, max_age=95, lapse_rate=0.04,
        lapse_beta=1.2, lapse_floor=0.01,
    )
    state = _state()
    years = int(book.projection_years.max())
    survival, deaths = mortality.load("basic").rates(book.attained_age, 2025, years, 0.5)
    paths = simulate(state.heston, state.hull_white(), state.correlations, state.mix,
                     n_years=years, n_paths=PATHS, seed=SEED)
    projection = gmwb.project(book, paths, survival, deaths,
                              equity_weight=state.mix.equity_weight, record=True)
    _CACHE["projection"] = (book, state, survival, deaths, paths, projection)
    return _CACHE["projection"]


def _proxy():
    if "proxy" not in _CACHE:
        _CACHE["proxy"] = lsmc.fit(_projection()[5])
    return _CACHE["proxy"]


def _exhausted_node(min_year: int = 6, min_years_left: int = 6):
    """A path and year where the contract value has run out and the guarantee is all there is.

    Bounded away from both ends of the projection: early enough that the owner is still paying
    charges when the account goes, late enough that there is a stream left to value.
    """
    book, _, _, _, _, projection = _projection()
    recorded = projection.recorded
    spent = recorded["account_value"] <= 1e-9
    spent[:, :min_year] = False
    spent[:, int(book.projection_years[0]) - min_years_left:] = False
    paths, years = np.where(spent)
    if paths.size == 0:
        raise AssertionError("no path reached exhaustion; the fixture is not exercising the test")
    order = np.argsort(recorded["benefit_base"][paths, years])
    pick = order[order.size // 2]
    return int(paths[pick]), int(years[pick])


# ---------------------------------------------------------------- what the recording means


def test_the_recorded_legs_rebuild_the_projection_totals():
    """The two recorded legs are the projection, split by year and by path. If they were not,
    every continuation value built from them would be measuring a different contract."""
    _, _, _, _, _, projection = _projection()
    recorded = projection.recorded
    assert approx(float(projection.pv_total_claims[0]), rel=1e-12) == float(
        recorded["pv_claim"].sum(axis=1).mean()
    )
    assert approx(float(projection.pv_attributable_fees[0]), rel=1e-12) == float(
        recorded["pv_fee"].sum(axis=1).mean()
    )


def test_persistency_is_the_chance_the_contract_is_still_there_to_pay():
    """Rebuilt from the survival curve and the lapse rule, with no reference to the recording.

    This is the factor that has to come out of a continuation value, and getting it from the
    same loop that wrote it would test nothing.
    """
    book, _, survival, _, _, projection = _projection()
    recorded = projection.recorded
    in_force = np.ones(PATHS)
    for year in range(recorded["persistency"].shape[1]):
        expected = survival[0, year] * in_force
        assert approx(expected, rel=1e-12) == recorded["persistency"][:, year]
        account = recorded["account_value"][:, year]
        lapse = behaviour.dynamic_lapse(
            recorded["benefit_base"][:, year], account,
            float(book.lapse_rate[0]), float(book.lapse_beta[0]), float(book.lapse_floor[0]),
        )
        in_force = in_force * np.where(account > 0.0, 1.0 - lapse, 1.0)


def test_the_death_benefit_base_drifts_away_from_the_guaranteed_withdrawal_base():
    """Which is why a node cannot reconstruct one from the other.

    The withdrawal base holds flat through the income phase while the death base is cut by
    every withdrawal, so by the time the contract is spent they are nowhere near each other.
    """
    path, year = _exhausted_node()
    recorded = _projection()[5].recorded
    assert recorded["death_rollup_base"][path, year] < 0.5 * recorded["benefit_base"][path, year]


# ---------------------------------------------------------------- rebuilding a node


def test_a_node_restarts_from_the_recorded_state_rather_than_a_guess():
    book, _, _, _, _, projection = _projection()
    recorded = projection.recorded
    path, year = _exhausted_node()
    node = nested._advance(book, recorded, path, year)

    for field, column in (
        ("benefit_base", "benefit_base"),
        ("bonus_base", "bonus_base"),
        ("death_benefit_base", "death_rollup_base"),
        ("death_ratchet_base", "death_ratchet_base"),
    ):
        assert approx(float(recorded[column][path, year]), rel=1e-12) == float(
            getattr(node, field)[0]
        )
    assert int(node.attained_age[0]) == int(book.attained_age[0]) + year + 1
    assert int(node.projection_years[0]) == int(book.projection_years[0]) - year - 1
    assert int(node.deferral_years[0]) == max(int(book.deferral_years[0]) - year - 1, 0)


def test_a_rebuilt_exhausted_node_is_worth_its_life_annuity():
    """The independent recalculation that caught the node-rebuilding error.

    Once the contract value is gone the insurer pays the guaranteed amount every year until the
    owner dies and collects nothing, so the value is a life annuity certain in form: the
    guaranteed withdrawal times the sum of survival-weighted zero-coupon bonds off the curve the
    node sits on. Nothing about the equity market enters it. A rebuilt node that disagrees with
    that sum is carrying state it should not have.
    """
    book, state, survival, _, _, projection = _projection()
    recorded = projection.recorded
    path, year = _exhausted_node()

    node_curve = nested.curve_at_node(
        state.hull_white(), float(year + 1), float(recorded["short_rate"][path, year])
    )
    node = nested._advance(book, recorded, path, year)
    node_state = replace(
        state, curve=node_curve, valuation_year=state.valuation_year + year + 1,
        heston=replace(state.heston, v0=float(max(recorded["variance"][path, year], 1e-8))),
    )
    valuation = Valuer(mortality.load("basic"), n_paths=2_000, seed=31, cache_size=1).value(
        node, node_state, attribution=np.array([1.0])
    )

    horizon = int(node.projection_years[0])
    ahead = np.arange(1, horizon + 1, dtype=float)
    conditional = survival[0, year + 1:year + 1 + horizon] / survival[0, year]
    bonds = np.exp(-node_curve.zero(ahead) * ahead)
    annuity = float(book.gawa_pct[0]) * float(node.benefit_base[0]) * float(
        (bonds * conditional).sum()
    )
    assert valuation.market_risk_benefit == approx(annuity, rel=0.02)


# ---------------------------------------------------------------- the proxy's own properties


def test_the_proxy_is_homogeneous_of_degree_one():
    """Double the contract value and the benefit base together and every cash flow doubles.
    This is the property the fit is built on, so it has to hold exactly, not approximately."""
    proxy, year = _proxy(), 8
    single, _ = proxy.value(year, 80.0, 120.0, 0.02, 0.045)
    doubled, _ = proxy.value(year, 160.0, 240.0, 0.02, 0.045)
    assert approx(2.0 * single[0], rel=1e-12) == doubled[0]


def test_the_proxy_is_affine_in_the_attribution():
    """One fit serves any attribution percentage, which is only true if the legs stay separate.
    The GAAP lens, the economic lens and the break-even solve all use different ones."""
    proxy, year = _proxy(), 8
    at_zero, _ = proxy.value(year, 80.0, 120.0, 0.02, 0.045, attribution=0.0)
    at_half, _ = proxy.value(year, 80.0, 120.0, 0.02, 0.045, attribution=0.5)
    at_one, _ = proxy.value(year, 80.0, 120.0, 0.02, 0.045, attribution=1.0)
    assert approx(0.5 * (at_zero[0] + at_one[0]), rel=1e-12) == at_half[0]
    assert at_one[0] < at_zero[0]          # attributing fees can only reduce the liability


def test_the_guarantee_is_worth_more_the_further_the_contract_value_has_fallen():
    proxy, year = _proxy(), 8
    values, _ = proxy.value(year, np.array([130.0, 100.0, 70.0, 40.0]), 120.0, 0.02, 0.045)
    assert np.all(np.diff(values) > 0)


def test_the_proxy_is_short_the_equity_market():
    proxy, year = _proxy(), 8
    assert proxy.delta(year, np.array([90.0]), np.array([120.0]), 0.02, 0.045)[0] < 0


def test_a_state_the_fit_never_visited_is_flagged():
    proxy, year = _proxy(), 8
    _, outside = proxy.value(year, 3_000.0, 120.0, 0.02, 0.045)
    assert bool(outside[0])
    _, inside = proxy.value(year, 100.0, 120.0, 0.02, 0.045)
    assert not bool(inside[0])


def test_a_state_inside_the_range_but_with_no_paths_around_it_is_still_flagged():
    """The failure a plain range check cannot see. By the last years of the contract almost
    every path sits at exhaustion, so the range runs from zero to the cap while the middle of
    the interval is empty; a node there came back unflagged and wrong by forty per cent of the
    account value. The flag has to follow where the paths are, not how far they reached."""
    proxy = _proxy()
    late = max(proxy.years) - 4
    fit_year = proxy.fits[late]
    empty = np.where(fit_year.support < fit_year.min_support)[0]
    interior = empty[(empty > 0) & (empty * lsmc.SUPPORT_BIN < fit_year.upper[0])]
    assert interior.size > 0, "expected a thin interior bin in the last years of the contract"

    moneyness = (interior[0] + 0.5) * lsmc.SUPPORT_BIN
    assert fit_year.lower[0] <= moneyness <= fit_year.upper[0]
    _, outside = proxy.value(late, 100.0 * moneyness, 100.0, 0.02, 0.045)
    assert bool(outside[0])


def test_the_exhausted_replicates_are_capped_and_the_live_rows_are_not():
    """A spent contract is one state repeated, so thirty thousand of them carry no more
    information than a couple of thousand, and left at full weight they take the fit over."""
    account = np.concatenate([np.zeros(30_000), np.linspace(1.0, 200.0, 500)])
    weights = lsmc._replicate_weights(account)
    assert approx(1.0, rel=1e-12) == weights[-1]
    assert approx(float(lsmc.EXHAUSTED_EFFECTIVE_ROWS), rel=1e-9) == float(weights[:30_000].sum())

    few = lsmc._replicate_weights(np.zeros(10))
    assert approx(np.ones(10), rel=1e-12) == few


def test_fitting_needs_a_recorded_projection():
    _, _, survival, deaths, paths, _ = _projection()
    book = _projection()[0]
    plain = gmwb.project(book, paths, survival, deaths, equity_weight=0.7235)
    with raises(ValueError, match="record=True"):
        lsmc.fit(plain)


def test_a_year_that_was_never_fitted_is_refused_rather_than_extrapolated():
    with raises(KeyError, match="no fit for year"):
        _proxy().value(10_000, 80.0, 120.0, 0.02, 0.045)


# ---------------------------------------------------------------- against the nested standard


def test_the_proxy_agrees_with_a_full_nested_valuation():
    """The test that matters. Everything above can hold while the proxy values the wrong thing.

    Six nodes at two thousand inner paths each is a coarse standard - the inner standard error
    is a meaningful share of the tolerance here - but it is an independent calculation of the
    same quantity, and it is what the two failures in this file's opening were caught by. The
    production sweep runs fifty nodes at five thousand paths across seven years and reports
    under one per cent of account value inside the design range.
    """
    book, state, _, _, _, projection = _projection()
    inner = Valuer(mortality.load("basic"), n_paths=2_000, seed=77, cache_size=1)
    truth = nested.gold_standard(inner, book, state, projection.recorded, year=8, n_nodes=6)
    comparison = nested.compare(_proxy(), truth)
    summary = nested.summarise(comparison, account_value=100.0)
    assert summary["nodes_in_range"] >= 4
    assert summary["rmse_pct_of_account_in_range"] < 0.05
