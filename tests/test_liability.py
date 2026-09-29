"""Contract mechanics, checked against the prospectus rather than against the code.

Most of these run the projection on a market that does nothing - growth of exactly one, a
discount factor of exactly one - so the answer is arithmetic that can be written down by hand
from the prospectus text. That is the only way to catch the errors that matter here, which are
not numerical: they are reading the contract wrong. The bonus compounding when it should be
simple was caught by the first test in this file and would have inflated a ten-year deferral
by 19% of the benefit base while every convergence and martingale check stayed green.

The death-benefit tests reproduce the prospectus's own worked examples, including the two that
show a withdrawal cutting the benefit base by more than the cash withdrawn.
"""

from __future__ import annotations

import numpy as np

from tests.checks import approx, raises
from vahedge.liability import cohorts, gmwb
from vahedge.liability import terms as terms_module
from vahedge.market.simulate import MarketPaths

YEARS = 14
PATHS = 4


def _flat_market(growth: float = 1.0, years: int = YEARS, n_paths: int = PATHS) -> MarketPaths:
    """A market where nothing happens, so the contract arithmetic is the whole answer."""
    ones = np.ones((n_paths, years))
    return MarketPaths(
        fund_growth=np.full((n_paths, years), growth),
        index_growth=np.full((n_paths, years), growth),
        discount=ones,
        short_rate=np.zeros((n_paths, years)),
        variance=np.zeros((n_paths, years)),
        zero_10y=np.zeros((n_paths, years)),
        realised_variance=np.zeros((n_paths, years)),
        seed=0,
        antithetic=False,
    )


def _core():
    return terms_module.load()[("flex_gmwb", "single", "core")]


def _contract(**overrides):
    defaults = dict(
        terms=_core(), issue_age=60, base_contract_charge=0.0, fund_expense=0.0,
        premium=100.0, deferral_years=YEARS, max_age=60 + YEARS,
    )
    defaults.update(overrides)
    return cohorts.single_contract(**defaults)


def _survival(n_cohorts: int = 1, years: int = YEARS):
    return np.ones((n_cohorts, years))


# ---------------------------------------------------------------- the rate sheet


def test_withdrawal_rate_follows_the_age_band_at_the_first_withdrawal():
    """Core option, single life, from the 27 April 2026 rate sheet."""
    core = _core()
    ages = np.array([58, 62, 67, 72, 78, 84])
    expected = np.array([0.0400, 0.0400, 0.0555, 0.0575, 0.0595, 0.0620])
    assert approx(expected) == core.withdrawal_rate(ages)


def test_an_age_outside_every_band_is_refused_rather_than_guessed():
    with raises(ValueError, match="no withdrawal band"):
        _core().withdrawal_rate(np.array([30]))


def test_the_three_benefit_options_differ_the_way_the_rate_sheet_says():
    loaded = terms_module.load()
    value = loaded[("flex_gmwb", "single", "value")]
    core = loaded[("flex_gmwb", "single", "core")]
    plus = loaded[("flex_gmwb", "single", "plus")]
    assert (value.charge_pct, core.charge_pct, plus.charge_pct) == (0.0030, 0.0125, 0.0170)
    assert (value.bonus_pct, core.bonus_pct, plus.bonus_pct) == (0.05, 0.06, 0.07)
    assert (value.gwb_adjustment_pct, plus.gwb_adjustment_pct) == (1.05, 2.00)
    assert core.annual_step_up and not plus.annual_step_up   # Plus steps up quarterly


# ---------------------------------------------------------------- the bonus


def test_the_bonus_is_simple_on_the_bonus_base_not_compound_on_the_benefit_base():
    """The prospectus: a bonus equal to a percentage of the Bonus Base is applied to the GWB,
    and the Bonus Base moves only with premiums and step-ups. With a flat market there are no
    step-ups, so a 6% bonus adds 6 a year to a base of 100 rather than compounding it. Treating
    it as a roll-up gives 179 after ten years instead of 160."""
    projection = gmwb.project(_contract(), _flat_market(), _survival())
    benefit_base = projection.mean_benefit_base[0]
    expected = np.minimum(100.0 + 6.0 * np.arange(1, YEARS + 1), 160.0)
    assert approx(expected, abs=1e-9) == benefit_base


def test_the_bonus_stops_after_ten_contract_years_without_a_step_up():
    projection = gmwb.project(_contract(), _flat_market(), _survival())
    benefit_base = projection.mean_benefit_base[0]
    assert float(benefit_base[9]) == approx(160.0)
    assert float(benefit_base[-1]) == approx(160.0)


def test_a_step_up_restarts_the_bonus_period_and_lifts_the_bonus_base():
    """In a rising market the step-up raises the GWB to the contract value, the Bonus Base
    follows it, and the ten-year clock starts again. Year one: the account grows 15% to 115,
    the rider charge of 1.25% of a benefit base of 100 takes 1.25, the step-up lifts the
    benefit base to 113.75, and the bonus adds 6% of the new base."""
    projection = gmwb.project(_contract(), _flat_market(growth=1.15), _survival())
    assert float(projection.mean_account_value[0][0]) == approx(113.75, rel=1e-9)
    assert float(projection.mean_benefit_base[0][0]) == approx(113.75 * 1.06, rel=1e-9)
    # Still growing in year thirteen, which a ten-year period without restarts would forbid.
    assert projection.mean_benefit_base[0][12] > projection.mean_benefit_base[0][11]


# ---------------------------------------------------------------- the GWB adjustment


def test_the_adjustment_date_is_the_later_of_age_seventy_and_twelve_years():
    """A 55-year-old reaches the seventieth-birthday anniversary at contract year 15, which is
    later than year 12, so that is the date. A 65-year-old reaches it at year 5, so year 12
    governs."""
    assert int(_contract(issue_age=55, deferral_years=20, max_age=90).adjustment_year[0]) == 15
    assert int(_contract(issue_age=65, deferral_years=20, max_age=90).adjustment_year[0]) == 12


def test_a_withdrawal_on_the_adjustment_date_kills_the_provision():
    """The prospectus voids the adjustment for any withdrawal taken on or prior to the date, so
    deferring exactly to it is not enough."""
    assert int(_contract(issue_age=60, deferral_years=12, max_age=90).adjustment_year[0]) == -1
    assert int(_contract(issue_age=60, deferral_years=13, max_age=90).adjustment_year[0]) == 12


def test_the_adjustment_floors_the_benefit_base_in_a_falling_market():
    """With the Plus option's 200% adjustment and a market that halves the account every year,
    the benefit base is floored at twice the original premium on the adjustment date."""
    plus = terms_module.load()[("flex_gmwb", "single", "plus")]
    book = cohorts.single_contract(
        terms=plus, issue_age=60, base_contract_charge=0.0, fund_expense=0.0,
        premium=100.0, deferral_years=YEARS, max_age=60 + YEARS,
    )
    assert int(book.adjustment_year[0]) == 12
    projection = gmwb.project(book, _flat_market(growth=0.5), _survival())
    before = float(projection.mean_benefit_base[0][11])
    after = float(projection.mean_benefit_base[0][12])
    assert before < 200.0
    assert after == approx(200.0, rel=1e-9)


# ---------------------------------------------------------------- withdrawals and claims


def test_a_contract_that_never_withdraws_never_claims():
    projection = gmwb.project(_contract(), _flat_market(), _survival())
    assert float(projection.pv_claims[0]) == 0.0


def test_the_account_runs_down_by_the_charge_plus_the_guaranteed_withdrawal():
    """Income phase, flat market, no fund or contract charges. Each year the rider charge of
    1.25% of the benefit base comes out, then 5.75% of it is withdrawn, and neither figure
    moves because the benefit base cannot step up in a flat market."""
    book = _contract(issue_age=70, deferral_years=0, max_age=70 + YEARS)
    projection = gmwb.project(book, _flat_market(), _survival())
    gawa = float(book.gawa_pct[0])
    annual = 100.0 * (0.0125 + gawa)
    expected = np.maximum(100.0 - annual * np.arange(1, YEARS + 1), 0.0)
    assert approx(expected, abs=1e-9) == projection.mean_account_value[0]


def test_claims_begin_only_once_the_account_is_exhausted():
    """A flat market drains the account at 1.25% plus 5.75% of a benefit base that never
    moves, so it lasts a little over fourteen years. The horizon has to outrun that or the
    test proves nothing."""
    horizon = 20
    book = _contract(issue_age=70, deferral_years=0, max_age=70 + horizon)
    projection = gmwb.project(book, _flat_market(years=horizon), _survival(years=horizon))
    exhausted = projection.exhaustion_prob[0] > 0
    claiming = projection.claims_by_year[0] > 0
    assert not np.any(claiming & ~exhausted)
    assert np.any(claiming)


def test_the_account_value_never_goes_negative():
    book = _contract(issue_age=70, deferral_years=0, max_age=70 + YEARS)
    projection = gmwb.project(book, _flat_market(growth=0.7), _survival())
    assert float(projection.mean_account_value[0].min()) >= 0.0


def test_partial_utilisation_lowers_both_the_draw_and_the_claim():
    full = _contract(issue_age=70, deferral_years=0, max_age=70 + YEARS, utilisation=1.0)
    partial = _contract(issue_age=70, deferral_years=0, max_age=70 + YEARS, utilisation=0.8)
    market, survival = _flat_market(growth=0.85), _survival()
    assert (
        float(gmwb.project(partial, market, survival).pv_claims[0])
        < float(gmwb.project(full, market, survival).pv_claims[0])
    )


def test_lapse_only_bites_while_there_is_a_contract_value_to_surrender():
    """An exhausted contract has nothing left to surrender, so its persistency stops decaying.
    Folding lapse into the survival curve instead would keep surrendering contracts that are
    already in claim, which is the state the guarantee exists for."""
    lapsing = _contract(issue_age=70, deferral_years=0, max_age=70 + YEARS, lapse_rate=0.05)
    static = _contract(issue_age=70, deferral_years=0, max_age=70 + YEARS, lapse_rate=0.0)
    market, survival = _flat_market(growth=0.85), _survival()
    with_lapse = float(gmwb.project(lapsing, market, survival).pv_claims[0])
    without = float(gmwb.project(static, market, survival).pv_claims[0])
    assert 0.0 < with_lapse < without


# ---------------------------------------------------------------- the death benefit


def test_the_rollup_benefit_base_reproduces_the_prospectus_withdrawal_example():
    """Appendix E, Roll Up example 3a: a withdrawal of 24,000 against a contract value of
    150,000 and a benefit base of 200,000 leaves a base of 171,000. The first 5% of the base
    comes off dollar for dollar and the rest proportionally against what is left."""
    base, contract_value, withdrawal, free_rate = 200_000.0, 150_000.0, 24_000.0, 0.05
    free = free_rate * base
    dollar_part = min(withdrawal, free)
    excess = withdrawal - dollar_part
    result = (base - dollar_part) * (1.0 - excess / (contract_value - dollar_part))
    assert result == approx(171_000.0, rel=1e-12)


def test_a_withdrawal_inside_the_free_amount_only_costs_its_own_dollars():
    """Example 3b: 5,000 withdrawn against a 200,000 base leaves 195,000."""
    base, withdrawal, free = 200_000.0, 5_000.0, 0.05 * 200_000.0
    dollar_part = min(withdrawal, free)
    excess = withdrawal - dollar_part
    assert excess == 0.0
    assert base - dollar_part == approx(195_000.0)


def test_the_rollup_compounds_and_freezes_at_the_anniversary_before_eighty_one():
    """A 6% roll-up on a base of 100 compounds while the owner is 80 or younger and stops
    after that. Issue age 75 gives six accruals, at the anniversaries where the attained age
    is 75 through 80."""
    book = _contract(issue_age=75, deferral_years=YEARS, max_age=75 + YEARS,
                     death_benefit=terms_module.DEATH_BENEFITS["rollup"])
    deaths = np.zeros((1, YEARS))
    projection = gmwb.project(book, _flat_market(), _survival(), deaths=deaths,
                              death_benefit=terms_module.DEATH_BENEFITS["rollup"])
    accrued = projection.mean_death_benefit[0]
    assert float(accrued[5]) == approx(100.0 * 1.05**6, rel=1e-9)   # age 75 elects at 5%
    assert float(accrued[-1]) == approx(float(accrued[5]), rel=1e-9)


def test_the_death_benefit_pays_only_the_excess_over_the_account_value():
    """A market that rises fast leaves the contract value above any roll-up, so the insurer's
    death-benefit cost is zero even though the benefit itself is large."""
    book = _contract(issue_age=60, deferral_years=YEARS, max_age=60 + YEARS,
                     death_benefit=terms_module.DEATH_BENEFITS["rollup"])
    deaths = np.full((1, YEARS), 0.01)
    projection = gmwb.project(book, _flat_market(growth=1.30), _survival(), deaths=deaths,
                              death_benefit=terms_module.DEATH_BENEFITS["rollup"])
    assert float(projection.pv_death_claims[0]) == approx(0.0, abs=1e-9)


def test_the_combination_benefit_is_worth_at_least_each_component():
    deaths = np.full((1, YEARS), 0.01)
    market, survival = _flat_market(growth=0.95), _survival()
    values = {}
    for name in ("rollup", "highest_anniversary", "combination"):
        benefit = terms_module.DEATH_BENEFITS[name]
        book = _contract(issue_age=60, deferral_years=YEARS, max_age=60 + YEARS,
                         death_benefit=benefit)
        values[name] = float(
            gmwb.project(book, market, survival, deaths=deaths, death_benefit=benefit)
            .pv_death_claims[0]
        )
    assert values["combination"] >= values["rollup"] - 1e-9
    assert values["combination"] >= values["highest_anniversary"] - 1e-9


# ---------------------------------------------------------------- the cohort grid


def test_the_grid_weights_sum_to_one_and_span_both_phases():
    book = cohorts.build(cohorts.GridSpec(), _core(), 0.0131, 0.0095)
    assert float(book.weight.sum()) == approx(1.0)
    assert np.any(book.deferral_years > 0) and np.any(book.deferral_years == 0)


def test_grid_weights_that_do_not_sum_to_one_are_refused():
    spec = cohorts.GridSpec(age_weights=(0.5, 0.2, 0.1, 0.1, 0.05))
    with raises(ValueError, match="weights must sum to 1"):
        cohorts.build(spec, _core(), 0.0131, 0.0095)


def test_rescaling_the_book_preserves_its_shape():
    book = cohorts.build(cohorts.GridSpec(), _core(), 0.0131, 0.0095)
    scaled = book.rescale_to(236_406.0)
    assert float(scaled.total_account_value) == approx(236_406.0, rel=1e-9)
    assert scaled.weighted_attained_age == approx(book.weighted_attained_age, rel=1e-12)


def test_net_amount_at_risk_counts_only_contracts_in_the_money():
    book = cohorts.build(cohorts.GridSpec(), _core(), 0.0131, 0.0095)
    in_the_money = book.benefit_base > book.account_value
    assert np.any(in_the_money) and np.any(~in_the_money)
    assert book.net_amount_at_risk > 0
    assert book.net_amount_at_risk < book.total_benefit_base
