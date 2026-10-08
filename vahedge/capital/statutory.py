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

from ..hedge.simulator import daily_profit

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


def statutory_capital(
    ledger: pd.DataFrame,
    surrender_share: float = SURRENDER_VALUE_SHARE,
) -> pd.DataFrame:
    """Mark a hedge backtest on both bases, so the floor's cost in a rally is a measurement.

    The balance sheet is short enough to write out, and writing it out is what makes the result
    arguable rather than asserted. Assets are the separate account, the hedge mark and the cash
    the programme has accumulated; liabilities are the separate account liability, which equals
    the account value, plus the additional general-account reserve for the guarantee. So

        capital = account + hedge + cash - max(account + guarantee, share * account)

    and the two regimes fall out of the max. Unfloored, the account legs cancel and capital is
    ``cash + hedge - guarantee``, which is the economic net worth the simulator already tracks and
    which a working hedge keeps flat. Floored, the guarantee leaves the expression entirely and
    capital is ``cash + hedge + (1 - share) * account``: the hedge is short equity, so in a rally
    it loses, and what used to offset that loss - the guarantee getting cheaper - is no longer in
    the reserve at all. Two per cent of the account rising is all that remains on the other side.

    The gap between the two capital series is reported as ``basis_gap`` and not as a hedging
    cost, because the hedge mark sits in both expressions and cancels out of the difference. The
    gap is ``(1 - share) * account + guarantee`` on floored days and zero elsewhere, so it comes
    out the same for an unhedged book as for a fully hedged one. Calling it the cost of hedging
    would be a mislabelling, and the first version of the experiment did exactly that: it
    reported an identical 23.56 per cent of account value for all seven strategies, including S0,
    which has no hedge to be unoffset.

    The cost the 8-K names shows up instead in how much of each basis the hedge removes. The same
    position takes 91 to 96 per cent of the variance out of the economic series and 48 to 58 per
    cent out of the statutory one, so the programme pays its full price and collects under half of
    the benefit on the basis that drives capital. That asymmetry is the measurement; the level gap
    is the floor's own cost and belongs to the block rather than to the hedge.

    Returns the ledger's columns plus the two capital series and their difference, indexed the
    same way, so the daily P&L reconciles against ``ledger["pnl"]`` on the economic side by
    construction.
    """
    for column in ("account_value", "liability", "hedge_mark", "cash"):
        if column not in ledger.columns:
            raise ValueError(f"ledger has no {column!r}; it is not a hedge run ledger")

    account = ledger["account_value"].to_numpy(dtype=float)
    guarantee = ledger["liability"].to_numpy(dtype=float)
    reserve = floored_reserve(guarantee, account, surrender_share=surrender_share)

    out = pd.DataFrame(index=ledger.index)
    out["account_value"] = account
    out["guarantee_reserve"] = guarantee
    out["floor_binds"] = reserve["floor_binds"]
    out["total_reserve"] = reserve["reserve"]
    out["economic_capital"] = ledger["cash"] + ledger["hedge_mark"] - ledger["liability"]
    out["statutory_capital"] = (
        account + ledger["hedge_mark"].to_numpy(dtype=float)
        + ledger["cash"].to_numpy(dtype=float) - reserve["reserve"]
    )
    # Before the programme there is no hedge and no cash, so the opening economic position is
    # minus the guarantee and the opening statutory one is the account less its reserve. Both
    # first days then carry the cost of striking the book, which keeps the basis gap at zero on
    # a day when nothing but the hedge changed.
    out["economic_pnl"] = daily_profit(out["economic_capital"], -guarantee[0])
    out["statutory_pnl"] = daily_profit(
        out["statutory_capital"], account[0] - float(reserve["reserve"][0])
    )
    out["basis_gap_pnl"] = out["statutory_pnl"] - out["economic_pnl"]
    return out


def floor_summary(marked: pd.DataFrame, account_value: float) -> dict:
    """The headline numbers from a two-basis mark, as shares of the starting account value."""
    days_per_year = 252.0
    years = max(marked.shape[0] / days_per_year, 1e-9)
    return {
        "days": float(marked.shape[0]),
        "share_of_days_floored": float(marked["floor_binds"].mean()),
        "economic_total_pct": float(
            (marked["economic_capital"].iloc[-1] - marked["economic_capital"].iloc[0])
            / account_value
        ),
        "statutory_total_pct": float(
            (marked["statutory_capital"].iloc[-1] - marked["statutory_capital"].iloc[0])
            / account_value
        ),
        # The floor's own cost to the block. Identical across strategies by construction, because
        # the hedge mark is in both capital series and cancels; see statutory_capital.
        "basis_gap_total_pct": float(marked["basis_gap_pnl"].sum() / account_value),
        "basis_gap_per_year_pct": float(marked["basis_gap_pnl"].sum() / account_value / years),
        "economic_sd_pct": float(marked["economic_pnl"].std(ddof=0) / account_value),
        "statutory_sd_pct": float(marked["statutory_pnl"].std(ddof=0) / account_value),
        "economic_worst_day_pct": float(marked["economic_pnl"].min() / account_value),
        "statutory_worst_day_pct": float(marked["statutory_pnl"].min() / account_value),
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
