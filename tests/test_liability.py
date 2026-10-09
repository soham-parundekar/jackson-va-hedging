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


def _scalar_recursion(book, growth, discount, years):
    """The seven anniversary steps of docs/methodology.md, written out one year at a time.

    A second implementation of the same contract, deliberately. Every other test here fixes one
    mechanic against the prospectus; none of them can catch the mechanics being right and their
    *order* being wrong, because reordering the charge, the withdrawal and the step-up leaves
    each one individually correct. This reads the order off the document rather than off the
    loop, so a reordering of either has to show up as a disagreement.
    """
    account = float(book.account_value[0])
    base = float(book.benefit_base[0])
    bonus_base = float(book.bonus_base[0])
    bonus_end = float(book.bonus_years_remaining[0])
    adjustment_year = int(book.adjustment_year[0])
    adjustment_live = adjustment_year >= 0
    gawa = float(book.gawa_pct[0])
    charge = float(book.rider_charge_pct[0])
    bonus_rate = float(book.bonus_pct[0])
    steps_up = bool(book.annual_step_up[0])
    utilisation = float(book.utilisation[0])
    deferral = int(book.deferral_years[0])
    drag_factor = float(np.exp(-float(book.account_drag[0])))
    insurer_share = float(book.insurer_drag_share[0])
    attained = int(book.attained_age[0])

    claims = fees = 0.0
    for year in range(years):
        before_drag = account * growth[year]
        account = before_drag * drag_factor
        base_charge = insurer_share * before_drag * (1.0 - drag_factor)

        rider = min(charge * base, max(account, 0.0))          # 1
        account -= rider
        deferring = year < deferral
        drawn = 0.0 if deferring else utilisation * gawa * base   # 3
        from_account = min(drawn, max(account, 0.0))
        claim = drawn - from_account
        account -= from_account
        if steps_up and account > base:                        # 4
            if account > bonus_base and attained + year <= gmwb.BONUS_RESTART_MAX_AGE:
                bonus_base = account
                bonus_end = year + 1 + gmwb.BONUS_PERIOD_YEARS
            base = account
        if deferring and year < bonus_end and account > 0.0:   # 5
            base += bonus_rate * bonus_base
        if adjustment_live and year == adjustment_year:        # 6
            base = max(base, float(book.adjustment_amount[0]))
            adjustment_live = False
        adjustment_live = adjustment_live and deferring

        claims += discount[year] * claim
        fees += discount[year] * (rider + base_charge)
    return claims, fees


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


def test_a_contract_already_drawing_keeps_the_percentage_it_locked_in():
    """The percentage is set at the first withdrawal and never moves again. A contract issued at
    70 that started income at 75 is on 5.95% at 82 as well, and letting the attained age pick the
    band instead hands it the 81-and-over rate it never qualified for - a guarantee 4% dearer
    than the same contract's at any earlier point in its own life."""
    drawing = dict(terms=_core(), issue_age=70, base_contract_charge=0.0, fund_expense=0.0,
                   premium=100.0, deferral_years=0, years_since_issue=12, max_age=105)
    aged_into_the_band = cohorts.single_contract(**drawing)
    locked = cohorts.single_contract(**drawing, first_withdrawal_age=75)
    assert float(aged_into_the_band.gawa_pct[0]) == approx(0.0620)
    assert float(locked.gawa_pct[0]) == approx(0.0595)
    # The default is still right for a contract that has not started: deferral carries it to the
    # age it will actually first draw at.
    deferring = cohorts.single_contract(
        terms=_core(), issue_age=70, base_contract_charge=0.0, fund_expense=0.0,
        premium=100.0, deferral_years=5, years_since_issue=0, max_age=105,
    )
    assert float(deferring.gawa_pct[0]) == approx(0.0595)


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


def test_the_recursion_matches_an_independent_reading_of_the_anniversary_order():
    """The vectorised projection against the scalar one, on a path that exercises everything.

    The growth path falls far enough to exhaust the account and so reach the claim branch,
    rises far enough early to trigger a step-up and restart the bonus clock, and runs long
    enough for the withdrawal base adjustment to land. If it agreed only on a path where
    nothing happened it would be testing the discounting.
    """
    years = 30
    growth = np.array([1.08, 0.78, 1.21, 1.05, 0.94, 1.11, 1.03, 1.17, 0.88, 1.06,
                       1.02, 1.09, 0.97, 1.04, 0.71, 0.93, 1.02, 0.88, 0.95, 1.01,
                       0.84, 0.97, 1.03, 0.91, 0.99, 1.05, 0.93, 1.00, 0.96, 1.02])
    discount = np.cumprod(np.full(years, 1.0 / 1.035))
    book = cohorts.single_contract(
        terms=_core(), issue_age=65, base_contract_charge=0.0131, fund_expense=0.0095,
        premium=100.0, deferral_years=5, max_age=65 + years,
    )
    paths = MarketPaths(
        fund_growth=growth.reshape(1, -1), index_growth=growth.reshape(1, -1),
        discount=discount.reshape(1, -1), short_rate=np.full((1, years), 0.035),
        variance=np.full((1, years), 0.04), zero_10y=np.full((1, years), 0.04),
        realised_variance=np.full((1, years), 0.04), seed=0, antithetic=False,
    )
    projected = gmwb.project(book, paths, np.ones((1, years)))
    claims, fees = _scalar_recursion(book, growth, discount, years)

    assert projected.pv_claims[0] > 1.0, "the path has to reach the claim branch to test it"
    assert projected.exhaustion_prob[0][-1] == 1.0
    assert projected.pv_claims[0] == approx(claims, rel=1e-13)
    assert projected.pv_attributable_fees[0] == approx(fees, rel=1e-13)


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


# ------------------------------------------------- book-level totals over several model points


def _mixed_book():
    """Four vintages spanning both phases and both sides of the money.

    Assembled the way `run_portfolio_validation.py` assembles the vintage book, so these totals
    are exercised on the construction the reported figures come from.
    """
    return cohorts.combine(
        [_vintage(60, 9, 130.0, 100.0),    # deep out of the money, already drawing
         _vintage(66, 3, 105.0, 100.0),    # out of the money, still deferring
         _vintage(72, 1, 92.0, 100.0),     # in the money, drawing
         _vintage(58, 2, 80.0, 100.0)],    # in the money, deferring
        weights=[0.4, 0.3, 0.2, 0.1],
    )


def test_the_book_spans_both_phases_and_its_weights_sum_to_one():
    book = _mixed_book()
    assert float(book.weight.sum()) == approx(1.0)
    assert np.any(book.deferral_years > 0) and np.any(book.deferral_years == 0)


def test_rescaling_the_book_preserves_its_shape():
    book = _mixed_book()
    scaled = book.rescale_to(236_406.0)
    assert float(scaled.total_account_value) == approx(236_406.0, rel=1e-9)
    assert scaled.weighted_attained_age == approx(book.weighted_attained_age, rel=1e-12)


def test_net_amount_at_risk_counts_only_contracts_in_the_money():
    book = _mixed_book()
    in_the_money = book.benefit_base > book.account_value
    assert np.any(in_the_money) and np.any(~in_the_money)
    assert book.net_amount_at_risk > 0
    assert book.net_amount_at_risk < book.total_benefit_base


def _vintage(issue_age: int, duration: int, account_value: float, benefit_base: float):
    return cohorts.single_contract(
        _core(), issue_age=issue_age, base_contract_charge=0.0131, fund_expense=0.0095,
        premium=100.0, account_value=account_value, benefit_base=benefit_base,
        deferral_years=max(5 - duration, 0), years_since_issue=duration, max_age=115,
    )


def test_combining_vintages_keeps_each_one_s_own_terms():
    """The vintage comparison lives on this: two cohorts whose withdrawal rates and horizons
    differ have to stay different after they are added into one book."""
    young = _vintage(60, 9, 130.0, 100.0)
    old = _vintage(72, 1, 105.0, 100.0)
    book = cohorts.combine([young, old], weights=[0.7, 0.3])
    assert book.size == 2
    assert approx([0.7, 0.3]) == book.weight
    assert approx([69, 73]) == book.attained_age
    assert book.gawa_pct[0] != book.gawa_pct[1]
    assert approx([float(young.projection_years[0]), float(old.projection_years[0])]) == \
        book.projection_years


def test_combining_scales_the_totals_by_the_weights_and_nothing_else():
    young = _vintage(60, 9, 130.0, 100.0)
    old = _vintage(72, 1, 105.0, 100.0)
    book = cohorts.combine([young, old], weights=[0.7, 0.3])
    assert book.total_account_value == approx(0.7 * 130.0 + 0.3 * 105.0)
    assert book.total_benefit_base == approx(100.0)
    # Weighted by account value, which is how a book's average age is quoted.
    assert book.weighted_attained_age == approx(
        (0.7 * 130.0 * 69 + 0.3 * 105.0 * 73) / (0.7 * 130.0 + 0.3 * 105.0)
    )


def test_combining_without_weights_leaves_each_book_s_own_weight():
    book = cohorts.combine([_vintage(60, 9, 130.0, 100.0), _vintage(72, 1, 105.0, 100.0)])
    assert approx([1.0, 1.0]) == book.weight


def test_combining_refuses_a_weight_per_book_mismatch():
    with raises(ValueError, match="weights"):
        cohorts.combine([_vintage(60, 9, 130.0, 100.0)], weights=[0.5, 0.5])


def test_combining_nothing_is_refused_rather_than_returning_an_empty_book():
    with raises(ValueError, match="no books"):
        cohorts.combine([])


def test_combining_refuses_books_that_disagree_on_an_optional_field():
    """A field set on one book and absent on another would give the combined book a short array,
    which the projection would read as a different book rather than as an error."""
    from dataclasses import replace
    plain = _vintage(60, 9, 130.0, 100.0)
    with_ratchet = replace(plain, death_ratchet_base=np.array([100.0]))
    with raises(ValueError, match="death_ratchet_base"):
        cohorts.combine([plain, with_ratchet])


# ---------------------------------------------------------------- policyholder behaviour


def test_dynamic_lapse_only_damps_once_the_guarantee_is_in_the_money():
    """At or below a moneyness of one the base rate is untouched; above it the rate decays and
    stops at the floor. A model that lets lapse fall to zero for a deep in-the-money contract
    ignores the surrenders that happen for reasons unrelated to the guarantee."""
    from vahedge.liability.behaviour import dynamic_lapse

    account = np.array([100.0, 100.0, 100.0, 100.0])
    base_bases = np.array([80.0, 100.0, 120.0, 160.0])
    rates = dynamic_lapse(base_bases, account, base_lapse=0.06, beta=4.0, floor=0.01)
    assert float(rates[0]) == approx(0.06)          # out of the money, base rate
    assert float(rates[1]) == approx(0.06)          # at the money, base rate
    assert 0.01 < float(rates[2]) < 0.06            # in the money, damped
    assert float(rates[3]) == approx(0.01)          # deep in the money, at the floor


def test_damping_lapse_makes_the_guarantee_more_expensive():
    """This is the direction that matters. A guarantee is worth more to the insurer when
    policyholders leave, so a behaviour model that keeps in-the-money contracts in force has to
    raise the claim, and by enough to notice."""
    static = _contract(issue_age=70, deferral_years=0, max_age=90, lapse_rate=0.06)
    damped = _contract(issue_age=70, deferral_years=0, max_age=90,
                       lapse_rate=0.06, lapse_beta=4.0, lapse_floor=0.01)
    market, survival = _flat_market(growth=0.9, years=20), _survival(years=20)
    static_claim = float(gmwb.project(static, market, survival).pv_claims[0])
    damped_claim = float(gmwb.project(damped, market, survival).pv_claims[0])
    assert damped_claim > static_claim * 1.1


def test_the_behaviour_grid_is_admissible():
    from vahedge.liability.behaviour import BEHAVIOUR_GRID, BehaviourAssumptions

    assert len(BEHAVIOUR_GRID) >= 5
    assert BEHAVIOUR_GRID[0].utilisation == 1.0 and BEHAVIOUR_GRID[0].base_lapse == 0.0
    with raises(ValueError, match="lapse_floor"):
        BehaviourAssumptions(base_lapse=0.02, lapse_floor=0.05)
    with raises(ValueError, match="lapse_beta"):
        BehaviourAssumptions(lapse_beta=-1.0)


def test_the_insurer_s_share_of_a_continuous_drag_is_collected_exactly():
    """The closed form against brute-force sub-stepping.

    Attributing fees means knowing how much of a continuous proportional drag was the
    insurer's revenue rather than the funds'. For a charge c out of a total drag m over an
    interval of length one, the end-of-interval value of what was collected is exactly
    (c/m) * AV * (1 - exp(-m)), and the implementation uses that rather than sub-stepping.
    An error here would move the attribution percentage and with it every market risk benefit
    figure in the project, and it would not show up anywhere else: the account value would
    still be right, because the drag comes out of it either way.
    """
    base, fund = 0.0131, 0.0095
    drag = base + fund
    book = _contract(base_contract_charge=base, fund_expense=fund)
    projection = gmwb.project(book, _flat_market(), _survival(), record=True)

    # Year one, before any withdrawal: the account has only decayed by the drag.
    collected = float(projection.recorded["pv_fee"][0, 0])
    rider = float(book.rider_charge_pct[0]) * 100.0
    closed_form = (base / drag) * 100.0 * (1.0 - np.exp(-drag))
    assert collected - rider == approx(closed_form, rel=1e-12)

    # And the closed form is the integral, which is what the sub-stepping converges to.
    steps = 20_000
    delta = 1.0 / steps
    account, brute = 100.0, 0.0
    for _ in range(steps):
        brute += base * account * delta
        account *= np.exp(-drag * delta)
    assert brute == approx(closed_form, rel=1e-4)
