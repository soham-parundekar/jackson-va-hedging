"""The fee the guarantee is actually worth, and the margin between that and the fee charged.

The rider charge is 1.25% of the benefit base on the Core option. Whether that is a good price
is the first question anyone asks about a guarantee, and it has an exact answer: the fee that
makes the rider worth zero at issue. Above it the product earns; below it the product is sold
at a loss and has to be made back somewhere else.

    find phi* such that  E[ PV(claims) - phi* * PV(benefit base exposure) ] = 0

The solve is on the fee itself rather than on a scaling of the fee's value, because the fee
changes the contract. Charging more takes more out of the account every year, which brings
exhaustion forward, which raises the claim. The function is monotone - a higher fee is always
worth more to the insurer over any range that matters - but it is not linear, and treating it
as linear overstates the break-even fee by enough to matter on a long deferral.

This is also where the attributed fee in the GAAP lens comes from. Under ASU 2018-12 the
attribution percentage is set so the market risk benefit is zero at inception where the fees
can cover the benefits, which is the same calculation read from the other end.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import brentq


@dataclass(frozen=True)
class BreakEven:
    """One cohort's fair fee, against what it is charged."""

    charged: float
    fair: float
    margin: float
    iterations: int
    converged: bool

    @property
    def margin_bp(self) -> float:
        return self.margin * 10000.0


def _rebuild_with_fee(book, fee: float):
    """The same book with a different rider charge, and nothing else changed."""
    fields = dict(book.__dict__)
    fields["rider_charge_pct"] = np.full_like(book.rider_charge_pct, fee)
    from ..liability.cohorts import CohortBook
    return CohortBook(**fields)


def solve(
    valuer,
    book,
    state,
    bounds: tuple = (0.0005, 0.08),
    tolerance: float = 1e-6,
) -> BreakEven:
    """The rider charge that makes the guarantee worth zero at issue, for the whole book.

    The book passed in should be at issue: account value and benefit base both at premium, with
    the full deferral ahead of it. Solving on an in-force book answers a different and less
    useful question, because most of the fee that would have paid for the guarantee has already
    been collected.

    Monte Carlo noise makes the objective slightly ragged, so the bracket is checked before the
    solve rather than letting Brent wander into a sign change that is not there. The same paths
    are used at every fee, which is what keeps the objective smooth enough to solve at all.
    """
    charged = float(np.average(book.rider_charge_pct, weights=book.weight))
    calls = {"n": 0}

    def objective(fee: float) -> float:
        calls["n"] += 1
        candidate = _rebuild_with_fee(book, fee)
        # Attribution is one here by construction: this asks what the guarantee costs against
        # the fees it is charged, not what share of them accounting attributes to it.
        valuation = valuer.value(
            candidate, state, attribution=np.ones(candidate.size)
        )
        return valuation.market_risk_benefit

    low, high = bounds
    at_low, at_high = objective(low), objective(high)
    if at_low < 0:
        return BreakEven(charged=charged, fair=low, margin=charged - low,
                         iterations=calls["n"], converged=False)
    if at_high > 0:
        return BreakEven(charged=charged, fair=high, margin=charged - high,
                         iterations=calls["n"], converged=False)

    fair = float(brentq(objective, low, high, xtol=tolerance, maxiter=60))
    return BreakEven(charged=charged, fair=fair, margin=charged - fair,
                     iterations=calls["n"], converged=True)


def margin_map(
    valuer,
    book_builder,
    state,
    issue_ages=(55, 60, 65, 70, 75),
    income_start_ages=(65, 70, 75),
) -> pd.DataFrame:
    """Fair fee against charged fee across the grid, which is the headline pricing result.

    Two dimensions matter and they pull in opposite directions. A later income start raises the
    withdrawal percentage and keeps the bonus accruing, which makes the guarantee more
    expensive; it also shortens the payment stream, which makes it cheaper. Where the two
    balance is not obvious in advance and is the point of drawing the map.
    """
    rows = []
    for issue_age in issue_ages:
        for income_start in income_start_ages:
            if income_start < issue_age:
                continue
            book = book_builder(issue_age=issue_age, income_start_age=income_start)
            result = solve(valuer, book, state)
            rows.append({
                "issue_age": issue_age,
                "income_start_age": income_start,
                "gawa_pct": float(book.gawa_pct[0]),
                "charged_pct": result.charged,
                "fair_pct": result.fair,
                "margin_bp": result.margin_bp,
                "converged": result.converged,
            })
    return pd.DataFrame(rows)


def fee_sensitivity(valuer, book, state, fees) -> pd.DataFrame:
    """Value of the guarantee across a range of fees, which is what the solve walks.

    Worth reporting on its own: it shows the objective is monotone, shows how far from linear it
    is, and makes the solved root checkable by eye rather than only by assertion.
    """
    rows = []
    for fee in fees:
        candidate = _rebuild_with_fee(book, float(fee))
        valuation = valuer.value(candidate, state, attribution=np.ones(candidate.size))
        rows.append({
            "fee_pct": float(fee),
            "value": valuation.market_risk_benefit,
            "value_pct_of_account": valuation.mrb_pct_of_account,
            "pv_claims": valuation.total_claims,
            "pv_fees": valuation.pv_fees,
            "std_error": valuation.std_error,
        })
    return pd.DataFrame(rows)
