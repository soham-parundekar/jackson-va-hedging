"""Statutory capital for the guarantee block: a conditional tail expectation, not a price.

Everything upstream of this module prices. A price is an expectation under a measure chosen so
that hedges are self-financing, and it is the right answer to "what is this worth". A reserve
answers a different question - how much has to be held so that the block survives a bad outcome -
and the answer is a percentile of a real-world distribution. The two differ in sign as well as
in level here: on the base-case grid the market risk benefit is a net asset at issue, while the
statutory requirement is positive, and no amount of care with one produces the other.

What this implements is VM-21's shape, not VM-21. The structure is the real thing: project the
block on real-world scenarios, accumulate the deficiency, take the greatest present value of it
over the projection horizon on each scenario, and average the worst tail of those. What is
missing is everything prescribed. The scenarios are this project's own Heston and Hull-White
model with an equity risk premium rather than the Academy generator; the discount rate is the
scenario's own short rate rather than a net asset earned rate computed off a modelled asset
portfolio; there is no standard projection amount, no prescribed reinvestment, no revenue
sharing, no deterministic reserve floor and no aggregation across the rest of the company's
business. The output is therefore comparable with itself across scenarios and assumptions, which
is what the experiments need, and is not a filed number.

Two levels are reported rather than one. CTE(70) is the stochastic reserve level in VM-21;
CTE(90) stands in for the capital question, where the requirement sits further out in the tail.
The gap between them is the interesting quantity for a hedging study, because a hedge that
flattens the mean of the tail and a hedge that flattens its shape are different hedges.

The surrender value floor is the part worth the trouble. Statutory reserves for this business
are floored at the cash surrender value, so a block whose guarantees are worth nothing - deep
out of the money, every account value far above every benefit base - still has to be reserved
at roughly the account value. Jackson said in December 2023, in its own words, that this floor
imposes a cost that is not economic and that moving the riders to a captive was meant to remove
it. ``with_surrender_floor`` measures that cost: the same block, the same scenarios, the reserve
with and without the floor binding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Share of account value a surrendering policyholder receives, across the block. Taken from the
# disclosure rather than from a surrender charge schedule: at 31 December 2025 Jackson reported
# 231,711 million of cash surrender value against 236,406 million of variable annuity separate
# account value, so 98.0 per cent, and the ratio has moved less than a point across the four
# year-ends in data/raw/jackson_book_statistics.csv. A schedule would be more precise per
# contract and less accurate for the block, because most of this business is long past its
# surrender charge period and the few contracts still inside it are a small share of the value.
SURRENDER_VALUE_SHARE = 0.980


@dataclass(frozen=True)
class TailMeasure:
    """One conditional tail expectation and the pieces needed to argue about it."""

    level: float                  # 0.70 for CTE(70)
    value: float                  # the tail mean itself
    scenarios_in_tail: int
    worst_scenario: float
    median_scenario: float
    mean_scenario: float
    share_of_scenarios_positive: float

    def as_row(self, **extra) -> dict:
        return {
            "cte_level": self.level, "requirement": self.value,
            "scenarios_in_tail": self.scenarios_in_tail,
            "worst_scenario": self.worst_scenario,
            "median_scenario": self.median_scenario,
            "mean_scenario": self.mean_scenario,
            "share_positive": self.share_of_scenarios_positive,
            **extra,
        }


def greatest_pv_deficiency(deficiency_pv: np.ndarray, floor_at_zero: bool = False) -> np.ndarray:
    """GPVAD per scenario: the worst the accumulated deficiency ever gets, in time-zero money.

    ``deficiency_pv`` is (n_paths, n_years), each entry the present value of that year's net
    outgo, so the accumulated deficiency is a running sum and the greatest present value is its
    maximum over the horizon. Taking the maximum over the path rather than the value at the end
    is the whole point of the measure: a block that costs money for fifteen years and then earns
    it back has to be funded through the fifteen years, and the terminal figure says nothing
    about that.

    ``floor_at_zero`` is off by default, so a scenario that never runs a deficiency contributes
    its negative number. VM-21 allows that before aggregation and it matters for a hedging study:
    flooring every scenario individually would hide a hedge that turns losing scenarios into
    profitable ones, which is the comparison the experiments are built to make.
    """
    accumulated = np.cumsum(np.asarray(deficiency_pv, dtype=float), axis=1)
    # A zero-year horizon has no deficiency; guard rather than let max() raise on an empty axis.
    if accumulated.shape[1] == 0:
        return np.zeros(accumulated.shape[0])
    worst = accumulated.max(axis=1)
    return np.maximum(worst, 0.0) if floor_at_zero else worst


def cte(values: np.ndarray, level: float = 0.70) -> TailMeasure:
    """Mean of the worst ``1 - level`` of the distribution.

    Computed by sorting and averaging rather than from a fitted quantile, because the tail is
    the object of interest and a parametric fit to it would be the assumption doing the work.
    The tail size rounds up, so CTE(90) on 999 scenarios averages the worst 100 rather than
    99.9 of them.

    The tail size is counted from the kept side, and that is not stylistic. Taking
    ``ceil(n * (1 - level))`` puts one extra scenario in the tail at every round level, because
    ``1 - 0.70`` is 0.30000000000000004 in binary and the ceiling of ten times it is four rather
    than three. One scenario in three hundred is a small bias in the level and a visible one in
    the gap between CTE(70) and CTE(90), which is the quantity the hedging comparison turns on.
    """
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("no scenarios to take a tail expectation over")
    if not 0.0 <= level < 1.0:
        raise ValueError(f"level must be in [0, 1), got {level}")
    count = max(values.size - int(np.floor(values.size * level + 1e-9)), 1)
    tail = np.sort(values)[-count:]
    return TailMeasure(
        level=level,
        value=float(tail.mean()),
        scenarios_in_tail=count,
        worst_scenario=float(values.max()),
        median_scenario=float(np.median(values)),
        mean_scenario=float(values.mean()),
        share_of_scenarios_positive=float((values > 0.0).mean()),
    )


def floored_reserve(
    guarantee_reserve,
    account_value,
    surrender_share: float = SURRENDER_VALUE_SHARE,
) -> dict:
    """The total policy reserve with the cash surrender value minimum applied, and its delta.

    What the floor applies to matters more than how it is applied, and the first version of this
    got it wrong in a way worth recording. It compared the accumulated deficiency - a quantity
    measured net of the assets already held - against the surrender value, which is gross. The
    floor then bound on 99.8 per cent of scenarios and came out at 115 per cent of premium, a
    number that says only that two different balance-sheet quantities were put on one axis.

    The 8-K is explicit about the mechanism. The floor sits on the base contract, Jackson retains
    the base contract, and the floor stays with it; what the captive takes is the guarantee rider.
    So the floor is a minimum on the TOTAL policy reserve - separate account assets plus any
    additional general-account reserve for the guarantee - not on the guarantee reserve alone. The
    separate account already holds the account value, so while the surrender value sits below it
    the floor adds nothing to the level.

    The cost the filing names is therefore not a level. It is a sensitivity: "non-economic hedging
    costs". When the floor binds the reserve stops following the economic liability and starts
    following the account value, so its derivative with respect to equity changes sign. The hedge
    is short equity against a guarantee that gets cheaper as markets rise; the floored reserve
    rises instead of falling, so the hedge loss is realised against a reserve that does not
    release. This returns the reserve, whether the floor binds, and the reserve's equity delta on
    each basis, so the comparison can be run off a ledger rather than asserted.
    """
    guarantee = np.asarray(guarantee_reserve, dtype=float)
    account = np.asarray(account_value, dtype=float)
    unfloored = account + guarantee
    floor = surrender_share * account
    binds = floor > unfloored
    return {
        "reserve": np.where(binds, floor, unfloored),
        "floor_binds": binds,
        # Derivative with respect to a move in the account value. Unfloored, the separate account
        # moves one for one and the guarantee moves with its own delta. Floored, only the
        # surrender value moves and the guarantee's delta leaves the reserve entirely, which is
        # the whole of the effect.
        "reserve_delta_unfloored": np.ones_like(account),
        "reserve_delta_floored": np.where(binds, surrender_share, 1.0),
    }


def non_economic_hedge_cost(
    account_change,
    guarantee_change,
    hedge_pnl,
    floor_binds,
    surrender_share: float = SURRENDER_VALUE_SHARE,
) -> dict:
    """How much hedge P&L the floor leaves with nothing on the other side, period by period.

    On the economic basis net worth moves by the hedge result less the change in the guarantee
    liability, and a working hedge makes that roughly nothing. On the statutory basis what moves
    is the floored total reserve against the separate account assets, and when the floor binds the
    guarantee's movement is not in the reserve at all. The gap between the two is the hedge P&L
    with no offset.

    All four inputs are period changes on one grid, signed so a positive ``guarantee_change`` is
    the liability getting more expensive.
    """
    account_change = np.asarray(account_change, dtype=float)
    guarantee_change = np.asarray(guarantee_change, dtype=float)
    hedge_pnl = np.asarray(hedge_pnl, dtype=float)
    binds = np.asarray(floor_binds, dtype=bool)

    economic = hedge_pnl - guarantee_change
    reserve_change = np.where(binds, surrender_share * account_change,
                              account_change + guarantee_change)
    statutory_basis = hedge_pnl + account_change - reserve_change
    return {
        "economic": economic,
        "statutory": statutory_basis,
        "unoffset": statutory_basis - economic,
        "share_of_periods_floored": float(binds.mean()),
    }


def requirement(
    deficiency_pv: np.ndarray,
    levels=(0.70, 0.90),
    floor_at_zero: bool = False,
) -> pd.DataFrame:
    """The reserve at each tail level, with the distribution behind it."""
    gpvad = greatest_pv_deficiency(deficiency_pv, floor_at_zero=floor_at_zero)
    return pd.DataFrame([cte(gpvad, level).as_row(basis="economic") for level in levels])


def deficiency_profile(deficiency_pv: np.ndarray, level: float = 0.90) -> pd.DataFrame:
    """Where in the projection the tail scenarios actually run out of money.

    A reserve is one number and it hides the horizon. Two blocks with the same CTE behave very
    differently if one peaks at year four and the other at year twenty-five, because the first is
    a liquidity problem and the second is a duration problem. This reports, for the scenarios in
    the tail, the mean accumulated deficiency by year and the distribution of the year at which
    each one peaks.
    """
    accumulated = np.cumsum(np.asarray(deficiency_pv, dtype=float), axis=1)
    gpvad = accumulated.max(axis=1)
    count = max(int(np.ceil(gpvad.size * (1.0 - level))), 1)
    in_tail = np.argsort(gpvad)[-count:]
    peak_year = accumulated[in_tail].argmax(axis=1) + 1
    return pd.DataFrame({
        "policy_year": np.arange(1, accumulated.shape[1] + 1),
        "mean_accumulated_tail": accumulated[in_tail].mean(axis=0),
        "mean_accumulated_all": accumulated.mean(axis=0),
        "share_of_tail_peaking_here": np.bincount(
            peak_year - 1, minlength=accumulated.shape[1]
        ) / count,
    })
