"""The fee the guarantee is actually worth, and the margin between that and the fee charged.

The rider charge is 1.25% of the benefit base on the Core option. Whether that is a good price
is the first question anyone asks about a guarantee, and it has an exact answer: the fee that
makes the rider worth zero at issue. Above it the product earns; below it the product is sold
at a loss and has to be made back somewhere else.

    find phi* such that  E[ PV(claims) - phi* * PV(benefit base exposure) ] = 0

The solve is on the fee itself rather than on a scaling of the fee's value, because the fee
changes the contract. Charging more takes more out of the account every year, which brings
exhaustion forward, which raises the claim. Over the range a product is actually sold at the
revenue wins and the guarantee gets cheaper as the charge rises, but not in proportion, and
treating the relationship as linear overstates the break-even fee by enough to matter on a long
deferral.

Past a few per cent the two effects change places and the curve turns back up: a charge large
enough to exhaust the account ends the fee stream and leaves the insurer paying the guaranteed
withdrawal for the rest of a life with nothing to charge it against. So the objective is
U-shaped over a wide enough bracket, there can be two roots or none, and a bracket that spans
the turning point will find neither. ``solve`` therefore checks both ends and refuses rather
than returning a bound that reads like a price - which is exactly what an earlier version did,
and the refusal is how the September 2016 curve turned out to admit no break-even fee at all.

This is also where the attributed fee in the GAAP lens comes from. Under ASU 2018-12 the
attribution percentage is set so the market risk benefit is zero at inception where the fees
can cover the benefits, which is the same calculation read from the other end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq


@dataclass(frozen=True)
class BreakEven:
    """One cohort's fair fee, against what it is charged.

    ``fair`` and ``margin`` are NaN unless ``reason`` is ``"solved"``. The first version of this
    put the failing bracket bound in ``fair`` instead, and the first caller to meet an unsolvable
    contract printed 800 basis points as the fee the guarantee was worth. A NaN cannot be quoted
    by accident.
    """

    charged: float
    fair: float
    margin: float
    iterations: int
    reason: str

    @property
    def converged(self) -> bool:
        return self.reason == "solved"

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

    Both ends positive means no fee in the bracket prices the guarantee, which is a result and
    not a failure - see the module docstring for why the curve can do that - and the bracket's
    top is not reported as the answer. ``fee_sensitivity`` is what to call next: it draws the
    curve the solve only sampled at two points.
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
    unresolved = lambda reason: BreakEven(charged=charged, fair=float("nan"),
                                          margin=float("nan"), iterations=calls["n"],
                                          reason=reason)
    if at_low < 0:
        # Already worth less than the fees it attracts at the smallest charge in the bracket, so
        # the fair fee is below the bracket rather than inside it.
        return unresolved("the fair fee is below the bracket")
    if at_high > 0:
        # Costly at both ends. Whether that means no fee prices this contract at all or only
        # that the bracket sits on one side of the turn is a question for fee_sensitivity, so it
        # is not answered here.
        return unresolved("no root in the bracket")

    fair = float(brentq(objective, low, high, xtol=tolerance, maxiter=60))
    return BreakEven(charged=charged, fair=fair, margin=charged - fair,
                     iterations=calls["n"], reason="solved")


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

    Cells the solve cannot resolve carry NaN and their own reason rather than a bound, so the map
    distinguishes a contract priced too cheaply from one no charge can price.
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
                "reason": result.reason,
            })
    return pd.DataFrame(rows)


def fee_sensitivity(valuer, book, state, fees) -> pd.DataFrame:
    """Value of the guarantee across a range of fees, which is what the solve walks.

    Worth reporting on its own: it shows where the curve falls and where it turns, how far from
    linear it is, and whether a solved root is a root at all rather than an artefact of a
    bracket. Where ``solve`` refuses, this is the table that says why.
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
