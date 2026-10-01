"""The statutory measures: GPVAD, the tail expectation, and the surrender value floor.

Every case here is one whose answer can be worked out by hand, because a reserve is a quantile
of a simulated distribution and a bug in it produces a plausible number rather than an error.
"""

from __future__ import annotations

import numpy as np

from tests.checks import approx, raises
from vahedge.capital import statutory


def test_gpvad_takes_the_worst_point_not_the_end():
    """A block that costs money and then earns it back still has to be funded through the middle."""
    # One scenario: out 10, out 5, back 20. Accumulated: 10, 15, -5. Terminal says -5.
    deficiency = np.array([[10.0, 5.0, -20.0]])
    assert statutory.greatest_pv_deficiency(deficiency)[0] == approx(15.0)


def test_gpvad_can_be_negative_and_is_not_floored_by_default():
    """Flooring each scenario would hide a hedge that turns losses into profits."""
    deficiency = np.array([[-3.0, -4.0]])
    assert statutory.greatest_pv_deficiency(deficiency)[0] == approx(-3.0)
    assert statutory.greatest_pv_deficiency(deficiency, floor_at_zero=True)[0] == 0.0


def test_gpvad_is_per_scenario():
    deficiency = np.array([[1.0, 1.0], [5.0, -1.0], [0.0, 0.0]])
    assert statutory.greatest_pv_deficiency(deficiency).tolist() == [2.0, 5.0, 0.0]


def test_cte_averages_the_worst_tail():
    values = np.arange(1.0, 11.0)          # 1 to 10
    measure = statutory.cte(values, level=0.70)
    assert measure.scenarios_in_tail == 3
    assert measure.value == approx((8 + 9 + 10) / 3)
    assert measure.worst_scenario == 10.0
    assert measure.median_scenario == approx(5.5)


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
    assert statutory.cte(values, level=0.0).value == approx(values.mean())


def test_cte_always_keeps_at_least_one_scenario():
    assert statutory.cte(np.arange(10.0), level=0.999).scenarios_in_tail == 1


def test_cte_is_monotone_in_the_level():
    values = np.random.default_rng(4).normal(size=5_000)
    levels = [0.0, 0.5, 0.70, 0.90, 0.98]
    tails = [statutory.cte(values, level).value for level in levels]
    assert tails == sorted(tails)


def test_cte_rejects_an_impossible_level():
    with raises(ValueError):
        statutory.cte(np.arange(10.0), level=1.0)
    with raises(ValueError):
        statutory.cte(np.arange(10.0), level=-0.1)


def test_cte_rejects_an_empty_sample():
    with raises(ValueError):
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
    assert out["reserve"][0] == approx(102.0)


def test_the_floor_binds_only_when_the_guarantee_is_a_large_enough_asset():
    """A guarantee worth minus three of a hundred leaves 97 against a 98 floor."""
    out = statutory.floored_reserve(guarantee_reserve=np.array([-3.0]),
                                    account_value=np.array([100.0]))
    assert out["floor_binds"][0]
    assert out["reserve"][0] == approx(98.0)


def test_the_reserve_delta_loses_the_guarantee_when_the_floor_binds():
    """Which is the whole of the effect the captive transaction was for."""
    out = statutory.floored_reserve(guarantee_reserve=np.array([-5.0, 5.0]),
                                    account_value=np.array([100.0, 100.0]))
    assert out["reserve_delta_floored"].tolist() == [
        statutory.SURRENDER_VALUE_SHARE, 1.0
    ]
    assert out["reserve_delta_unfloored"].tolist() == [1.0, 1.0]


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
    assert profile["share_of_tail_peaking_here"].sum() == approx(1.0)
    assert profile.loc[0, "share_of_tail_peaking_here"] == approx(0.5)
    assert profile.loc[2, "share_of_tail_peaking_here"] == approx(0.5)


def test_the_profile_tail_is_at_least_as_bad_as_the_whole_sample():
    rng = np.random.default_rng(13)
    deficiency = rng.normal(loc=0.2, scale=1.0, size=(800, 20))
    profile = statutory.deficiency_profile(deficiency, level=0.90)
    assert profile["mean_accumulated_tail"].max() > profile["mean_accumulated_all"].max()


def test_the_surrender_share_matches_the_disclosure():
    """231,711 of cash surrender value against 236,406 of separate account at 31 December 2025."""
    assert statutory.SURRENDER_VALUE_SHARE == approx(231_711 / 236_406, abs=5e-4)


def _ledger(account, guarantee, hedge, cash):
    """A minimal hedge-run ledger with only the columns the two-basis mark reads."""
    import pandas as pd
    return pd.DataFrame({
        "account_value": account, "liability": guarantee,
        "hedge_mark": hedge, "cash": cash,
    }, index=pd.RangeIndex(len(account), name="date"))


def test_the_two_bases_agree_while_the_floor_is_slack():
    """Unfloored, the account legs cancel and statutory capital is the economic net worth."""
    marked = statutory.statutory_capital(_ledger(
        account=[100.0, 104.0], guarantee=[3.0, 2.0], hedge=[0.0, -1.0], cash=[1.0, 1.0],
    ))
    assert not marked["floor_binds"].any()
    assert np.allclose(marked["economic_capital"], marked["statutory_capital"])
    assert np.allclose(marked["basis_gap_pnl"], 0.0)


def test_a_rally_with_the_floor_binding_leaves_the_hedge_loss_unoffset():
    """The whole of the captive transaction, in two rows.

    The account rises ten, the guarantee gets four cheaper, the short hedge loses exactly four.
    Economically that is flat. With the reserve pinned at 98 per cent of the account, the
    guarantee's gain is not in the reserve and only two per cent of the rise offsets the hedge.
    """
    marked = statutory.statutory_capital(_ledger(
        account=[100.0, 110.0], guarantee=[-5.0, -9.0], hedge=[0.0, -4.0], cash=[0.0, 0.0],
    ))
    assert marked["floor_binds"].all()
    assert marked["economic_pnl"].iloc[1] == approx(0.0)
    assert marked["statutory_pnl"].iloc[1] == approx(-4.0 + 0.02 * 10.0)
    assert marked["basis_gap_pnl"].iloc[1] < 0.0


def test_the_basis_gap_does_not_depend_on_the_hedge():
    """Which is why it is not the cost of hedging, and the first version of the experiment was wrong.

    The hedge mark is in both capital series, so it cancels out of the difference and the gap
    comes out identical for an unhedged book and a hedged one. The experiment reported the same
    23.56 per cent for all seven strategies before this was noticed.
    """
    account, guarantee = [100.0, 110.0], [-5.0, -9.0]
    unhedged = statutory.statutory_capital(
        _ledger(account, guarantee, hedge=[0.0, 0.0], cash=[0.0, 0.0]))
    hedged = statutory.statutory_capital(
        _ledger(account, guarantee, hedge=[0.0, -4.0], cash=[0.0, 0.0]))
    assert np.allclose(unhedged["basis_gap_pnl"], hedged["basis_gap_pnl"])
    assert not np.allclose(unhedged["economic_pnl"], hedged["economic_pnl"])


def test_the_floored_reserve_carries_no_guarantee_sensitivity():
    """Two ledgers differing only in the guarantee give the same statutory capital when floored."""
    one = statutory.statutory_capital(_ledger([100.0], [-6.0], [0.0], [0.0]))
    two = statutory.statutory_capital(_ledger([100.0], [-20.0], [0.0], [0.0]))
    assert one["floor_binds"].all() and two["floor_binds"].all()
    assert one["statutory_capital"].iloc[0] == approx(two["statutory_capital"].iloc[0])
    assert one["economic_capital"].iloc[0] != approx(two["economic_capital"].iloc[0])


def test_the_mark_refuses_a_ledger_that_is_not_one():
    import pandas as pd
    with raises(ValueError, match="hedge_mark"):
        statutory.statutory_capital(pd.DataFrame({
            "account_value": [1.0], "liability": [1.0], "cash": [1.0],
        }))


def test_the_summary_scales_by_the_starting_account_value():
    marked = statutory.statutory_capital(_ledger(
        account=[100.0, 110.0, 120.0], guarantee=[-5.0, -9.0, -13.0],
        hedge=[0.0, -4.0, -8.0], cash=[0.0, 0.0, 0.0],
    ))
    summary = statutory.floor_summary(marked, account_value=100.0)
    assert summary["days"] == 3.0
    assert summary["share_of_days_floored"] == 1.0
    assert summary["basis_gap_total_pct"] < 0.0
    assert summary["economic_total_pct"] == approx(0.0)
