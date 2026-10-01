"""The statutory measures: GPVAD, the tail expectation, and the surrender value floor.

Every case here is one whose answer can be worked out by hand, because a reserve is a quantile
of a simulated distribution and a bug in it produces a plausible number rather than an error.
"""

from __future__ import annotations

import numpy as np
import pytest

from vahedge.capital import statutory


def test_gpvad_takes_the_worst_point_not_the_end():
    """A block that costs money and then earns it back still has to be funded through the middle."""
    # One scenario: out 10, out 5, back 20. Accumulated: 10, 15, -5. Terminal says -5.
    deficiency = np.array([[10.0, 5.0, -20.0]])
    assert statutory.greatest_pv_deficiency(deficiency)[0] == pytest.approx(15.0)


def test_gpvad_can_be_negative_and_is_not_floored_by_default():
    """Flooring each scenario would hide a hedge that turns losses into profits."""
    deficiency = np.array([[-3.0, -4.0]])
    assert statutory.greatest_pv_deficiency(deficiency)[0] == pytest.approx(-3.0)
    assert statutory.greatest_pv_deficiency(deficiency, floor_at_zero=True)[0] == 0.0


def test_gpvad_is_per_scenario():
    deficiency = np.array([[1.0, 1.0], [5.0, -1.0], [0.0, 0.0]])
    assert statutory.greatest_pv_deficiency(deficiency).tolist() == [2.0, 5.0, 0.0]


def test_cte_averages_the_worst_tail():
    values = np.arange(1.0, 11.0)          # 1 to 10
    measure = statutory.cte(values, level=0.70)
    assert measure.scenarios_in_tail == 3
    assert measure.value == pytest.approx((8 + 9 + 10) / 3)
    assert measure.worst_scenario == 10.0
    assert measure.median_scenario == pytest.approx(5.5)


def test_cte_tail_size_rounds_up():
    """CTE(90) on 999 scenarios averages 100, not 99.9 of them."""
    assert statutory.cte(np.arange(999.0), level=0.90).scenarios_in_tail == 100
    assert statutory.cte(np.arange(1000.0), level=0.90).scenarios_in_tail == 100


def test_cte_tail_size_survives_binary_rounding():
    """ceil(n * (1 - level)) puts one scenario too many in the tail at every round level.

    1 - 0.70 is 0.30000000000000004, so ten times it ceilings to four. Counting from the kept
    side instead gets 300 out of 1,000 rather than 301, which is what CTE(70) means.
    """
    for size, level, expected in [(10, 0.70, 3), (1_000, 0.70, 300), (1_000, 0.80, 200),
                                  (100, 0.70, 30), (2_000, 0.90, 200), (5_000, 0.98, 100)]:
        assert statutory.cte(np.arange(float(size)), level).scenarios_in_tail == expected


def test_cte_at_zero_is_the_mean():
    values = np.array([2.0, 4.0, 9.0, 1.0])
    assert statutory.cte(values, level=0.0).value == pytest.approx(values.mean())


def test_cte_always_keeps_at_least_one_scenario():
    assert statutory.cte(np.arange(10.0), level=0.999).scenarios_in_tail == 1


def test_cte_is_monotone_in_the_level():
    values = np.random.default_rng(4).normal(size=5_000)
    levels = [0.0, 0.5, 0.70, 0.90, 0.98]
    tails = [statutory.cte(values, level).value for level in levels]
    assert tails == sorted(tails)


def test_cte_rejects_an_impossible_level():
    with pytest.raises(ValueError):
        statutory.cte(np.arange(10.0), level=1.0)
    with pytest.raises(ValueError):
        statutory.cte(np.arange(10.0), level=-0.1)


def test_cte_rejects_an_empty_sample():
    with pytest.raises(ValueError):
        statutory.cte(np.array([]))


def test_the_floor_does_not_bind_while_the_surrender_value_is_below_the_assets():
    """The separate account already holds the account value, so the floor adds nothing in level.

    This is the correction. The first version compared the accumulated deficiency against the
    surrender value, bound on almost every scenario and reported 115 per cent of premium. The
    floor is a minimum on the total policy reserve, and 98 per cent of the account value is below
    the account value.
    """
    out = statutory.floored_reserve(guarantee_reserve=np.array([2.0]),
                                    account_value=np.array([100.0]))
    assert not out["floor_binds"][0]
    assert out["reserve"][0] == pytest.approx(102.0)


def test_the_floor_binds_only_when_the_guarantee_is_a_large_enough_asset():
    """A guarantee worth minus three of a hundred leaves 97 against a 98 floor."""
    out = statutory.floored_reserve(guarantee_reserve=np.array([-3.0]),
                                    account_value=np.array([100.0]))
    assert out["floor_binds"][0]
    assert out["reserve"][0] == pytest.approx(98.0)


def test_the_reserve_delta_loses_the_guarantee_when_the_floor_binds():
    """Which is the whole of the effect the captive transaction was for."""
    out = statutory.floored_reserve(guarantee_reserve=np.array([-5.0, 5.0]),
                                    account_value=np.array([100.0, 100.0]))
    assert out["reserve_delta_floored"].tolist() == [
        statutory.SURRENDER_VALUE_SHARE, 1.0
    ]
    assert out["reserve_delta_unfloored"].tolist() == [1.0, 1.0]


def test_a_working_hedge_is_flat_economically_and_not_statutorily_when_floored():
    """A rally: the guarantee gets cheaper, the hedge loses, the floored reserve does not release."""
    account_change = np.array([10.0])
    guarantee_change = np.array([-4.0])     # the liability got cheaper by four
    hedge_pnl = np.array([-4.0])            # the short equity hedge lost exactly that
    out = statutory.non_economic_hedge_cost(
        account_change, guarantee_change, hedge_pnl, floor_binds=np.array([True]),
    )
    assert out["economic"][0] == pytest.approx(0.0)
    # Reserve moves by 0.98 of the account rise and the guarantee is not in it, so the hedge loss
    # sits against a reserve release of 0.98 * 10 against assets up 10.
    assert out["statutory"][0] == pytest.approx(-4.0 + 10.0 - 0.98 * 10.0)
    assert out["unoffset"][0] == pytest.approx(out["statutory"][0])


def test_the_two_bases_agree_when_the_floor_does_not_bind():
    rng = np.random.default_rng(21)
    account_change = rng.normal(scale=5.0, size=200)
    guarantee_change = rng.normal(scale=2.0, size=200)
    hedge_pnl = rng.normal(scale=2.0, size=200)
    out = statutory.non_economic_hedge_cost(
        account_change, guarantee_change, hedge_pnl, floor_binds=np.zeros(200, dtype=bool),
    )
    assert np.allclose(out["economic"], out["statutory"])
    assert np.allclose(out["unoffset"], 0.0)
    assert out["share_of_periods_floored"] == 0.0


def test_requirement_reports_both_levels_and_the_tail_is_worse_at_ninety():
    rng = np.random.default_rng(5)
    deficiency = rng.normal(loc=0.5, scale=2.0, size=(2_000, 15))
    table = statutory.requirement(deficiency)
    assert list(table["cte_level"]) == [0.70, 0.90]
    assert table.loc[1, "requirement"] > table.loc[0, "requirement"]


def test_the_profile_finds_where_the_tail_peaks():
    """Two scenarios peaking at different years, so the share column has to split them."""
    deficiency = np.array([
        [10.0, -20.0, 0.0],      # peaks at year 1
        [1.0, 1.0, 1.0],         # peaks at year 3
    ])
    profile = statutory.deficiency_profile(deficiency, level=0.0)
    assert list(profile["policy_year"]) == [1, 2, 3]
    assert profile["share_of_tail_peaking_here"].sum() == pytest.approx(1.0)
    assert profile.loc[0, "share_of_tail_peaking_here"] == pytest.approx(0.5)
    assert profile.loc[2, "share_of_tail_peaking_here"] == pytest.approx(0.5)


def test_the_profile_tail_is_at_least_as_bad_as_the_whole_sample():
    rng = np.random.default_rng(13)
    deficiency = rng.normal(loc=0.2, scale=1.0, size=(800, 20))
    profile = statutory.deficiency_profile(deficiency, level=0.90)
    assert profile["mean_accumulated_tail"].max() > profile["mean_accumulated_all"].max()


def test_the_surrender_share_matches_the_disclosure():
    """231,711 of cash surrender value against 236,406 of separate account at 31 December 2025."""
    assert statutory.SURRENDER_VALUE_SHARE == pytest.approx(231_711 / 236_406, abs=5e-4)
