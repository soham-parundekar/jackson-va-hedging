"""Model points, one or many, as the parallel arrays the projection steps through.

A `CohortBook` is a set of model points with one array per contract field, so one contract and
a hundred cost the same code path and the projection never learns which it is holding.
`single_contract` makes one; `combine` concatenates several into a book whose cohorts are valued
on one simulation, with each carrying its own attribution percentage.

Two objects get built out of this, and they answer different questions.

The *reference contract* is one model point: Perspective II with the Flex GMWB Single rider on
the Core option, $100,000 single premium, issue age 70. Every mechanic in the contract is stated
on it, the robustness table varies it one assumption at a time, and the hedging backtest rolls
it along realised history. It is an illustration and says so.

The *vintage book* is five of those, issued at two-year intervals from 2016 to 2024, each rolled
to a disclosure date along the index history that actually happened and then added up with
`combine`. That is
what the filing's aggregate gets compared against, because the comparison needs properties a
single policy cannot have: it reaches a weighted attained age of 70.85 against the disclosed 70,
a blended withdrawal rate of 5.73%, and a deferring share that falls from 0.75 to 0.40 across
the four dates.

Rolling rather than assuming is the point of doing it that way. Issue age, duration and
moneyness are not three dials to be weighted independently - what a 2016 contract is worth today
is a fact about the decade it lived through, and the roll produces its account value, its benefit
base and therefore its moneyness together. `scripts/run_portfolio_validation.py` carries the
construction and the issuance weights.

What the vintage book does not do is close the gap to the filing, and claiming otherwise would be
easy and wrong. On the four 2025 shocks it runs 2.3 to 2.8 times the disclosed figure where the
reference contract runs 1.7 to 2.1, so representativeness moves the multiple *up*.
`docs/validation.md` takes that apart: behaviour accounts for the rate half and moneyness for
most of the equity half, and neither is a property of how many model points there are.

Phase matters enough to call out, since it is the one dimension that is not a market variable. A
deferring contract is still earning a bonus, still eligible for the GWB adjustment that floors
its benefit base at 105% of premium, and has not begun drawing its account down. Three-quarters
of the vintage book is still deferring at the 2022 date, so this is not a simplification at the
margin.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np
import pandas as pd

from .terms import DEATH_BENEFITS, DeathBenefitTerms, RiderTerms

GWB_ADJUSTMENT_AGE = 70          # the anniversary on or following this birthday
GWB_ADJUSTMENT_MIN_YEARS = 12    # ...or the twelfth contract anniversary, whichever is later
BONUS_PERIOD_YEARS = 10


@dataclass(frozen=True)
class CohortBook:
    """Model points and everything the projection needs, as parallel arrays."""

    issue_age: np.ndarray
    years_since_issue: np.ndarray
    attained_age: np.ndarray
    account_value: np.ndarray
    benefit_base: np.ndarray
    bonus_base: np.ndarray
    premium_at_issue: np.ndarray
    weight: np.ndarray
    deferral_years: np.ndarray        # policy years from today until the first withdrawal
    bonus_years_remaining: np.ndarray
    adjustment_year: np.ndarray       # -1 once the provision is dead
    adjustment_amount: np.ndarray
    gawa_pct: np.ndarray
    rider_charge_pct: np.ndarray
    bonus_pct: np.ndarray
    annual_step_up: np.ndarray
    utilisation: np.ndarray
    lapse_rate: np.ndarray
    lapse_beta: np.ndarray
    lapse_floor: np.ndarray
    account_drag: np.ndarray
    insurer_drag_share: np.ndarray
    projection_years: np.ndarray
    option: np.ndarray
    death_benefit_base: np.ndarray
    db_rollup_pct: np.ndarray
    db_ratchet: np.ndarray
    db_charge_pct: np.ndarray
    db_free_withdrawal_pct: np.ndarray
    death_benefit: np.ndarray
    # The highest-anniversary base, which starts equal to the roll-up base and then follows its
    # own path. Only a book restarted part-way through its life needs to set it, so it defaults
    # to None and the projection reads the roll-up base instead.
    death_ratchet_base: np.ndarray | None = None

    @property
    def size(self) -> int:
        return int(self.issue_age.size)

    @property
    def total_account_value(self) -> float:
        return float(np.sum(self.weight * self.account_value))

    @property
    def total_benefit_base(self) -> float:
        return float(np.sum(self.weight * self.benefit_base))

    @property
    def net_amount_at_risk(self) -> float:
        """The living-benefit measure: benefit base above contract value, floored at zero.

        Jackson defines the disclosed net amount at risk as the greater of the death-benefit
        and living-benefit measures, so this is a lower bound on the disclosed figure by
        construction, and V3 reports it as such rather than pretending otherwise.
        """
        return float(np.sum(self.weight * np.maximum(self.benefit_base - self.account_value, 0.0)))

    @property
    def weighted_attained_age(self) -> float:
        """Weighted by account value, which is how Jackson's disclosure is constructed."""
        value = self.weight * self.account_value
        return float(np.sum(value * self.attained_age) / np.sum(value))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "issue_age": self.issue_age,
                "years_since_issue": self.years_since_issue,
                "attained_age": self.attained_age,
                "option": self.option,
                "account_value": self.account_value,
                "benefit_base": self.benefit_base,
                # A spent contract has no moneyness rather than an infinite one, and a node taken
                # off a projection is often exactly that.
                "gwb_over_av": np.where(
                    self.account_value > 0.0,
                    self.benefit_base / np.where(self.account_value > 0.0, self.account_value, 1.0),
                    np.nan,
                ),
                "deferral_years": self.deferral_years,
                "adjustment_year": self.adjustment_year,
                "gawa_pct": self.gawa_pct,
                "weight": self.weight,
            }
        )

    def rescale_to(self, total_account_value: float) -> "CohortBook":
        """Scale the weights so the book's account value matches a disclosed total."""
        if total_account_value <= 0:
            raise ValueError("total account value must be positive")
        factor = total_account_value / self.total_account_value
        return CohortBook(**{**self.__dict__, "weight": self.weight * factor})


def combine(books, weights=None) -> CohortBook:
    """One book out of several, for a portfolio assembled model point by model point.

    The vintage comparison builds each vintage with ``single_contract``, rolls it along its own
    history and then needs all of them valued together: one simulation, one projection, and one
    attribution percentage per cohort rather than per run. Concatenating the arrays is all that
    takes, and it belongs here rather than in the script so that a book which grows a field does
    not silently lose it in whichever caller did its own concatenation.

    ``weights`` multiplies each book's own weight, so a vintage's share of the portfolio is set
    here and the per-contract scale stays where ``single_contract`` put it.
    """
    books = list(books)
    if not books:
        raise ValueError("no books to combine")
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.size != len(books):
            raise ValueError(f"{weights.size} weights for {len(books)} books")
        books = [replace(book, weight=book.weight * float(weight))
                 for book, weight in zip(books, weights)]
    values = {}
    for field in fields(CohortBook):
        parts = [getattr(book, field.name) for book in books]
        present = [part is not None for part in parts]
        if not any(present):
            values[field.name] = None
        elif not all(present):
            # Silently dropping it would give the combined book a shorter array than its
            # cohorts, which the projection would read as a different book.
            raise ValueError(f"{field.name} is set on some of these books and not others")
        else:
            values[field.name] = np.concatenate([np.atleast_1d(part) for part in parts])
    return CohortBook(**values)


def single_contract(
    terms: RiderTerms,
    issue_age: int,
    base_contract_charge: float,
    fund_expense: float,
    premium: float = 100000.0,
    account_value: float | None = None,
    benefit_base: float | None = None,
    deferral_years: int = 0,
    years_since_issue: int = 0,
    max_age: int = 115,
    utilisation: float = 1.0,
    lapse_rate: float = 0.0,
    lapse_beta: float = 0.0,
    lapse_floor: float = 0.0,
    death_benefit: DeathBenefitTerms | None = None,
    first_withdrawal_age: int | None = None,
) -> CohortBook:
    """A one-cohort book, for the at-issue valuation and for the hedging backtest.

    The backtest rolls one representative policy along realised history, so it needs the same
    projection machinery the book uses, pointed at a single model point.

    ``first_withdrawal_age`` is what the guaranteed withdrawal percentage is read off, and it
    defaults to the age the contract will reach when its remaining deferral ends. That default is
    right for a contract that has not started drawing and wrong for one that has: the percentage
    locks at the first withdrawal and never moves again, so a contract twelve years into a life
    that began drawing at 75 keeps 5.95% rather than picking up the 81-and-over band's 6.20% by
    having aged into it. Pass it explicitly whenever the account value says withdrawals have
    already been happening.
    """
    death_benefit = death_benefit or DEATH_BENEFITS["basic"]
    attained = issue_age + years_since_issue
    anniversary_at_70 = max(1, GWB_ADJUSTMENT_AGE - issue_age)
    adjustment_contract_year = max(anniversary_at_70, GWB_ADJUSTMENT_MIN_YEARS)
    adjustment_year = adjustment_contract_year - years_since_issue
    if adjustment_year < 0 or deferral_years <= adjustment_year:
        adjustment_year = -1

    drag = base_contract_charge + fund_expense
    if first_withdrawal_age is None:
        first_withdrawal_age = attained + deferral_years
    array = lambda value, dtype=float: np.array([value], dtype=dtype)
    return CohortBook(
        issue_age=array(issue_age, int),
        years_since_issue=array(years_since_issue, int),
        attained_age=array(attained, int),
        account_value=array(premium if account_value is None else account_value),
        benefit_base=array(premium if benefit_base is None else benefit_base),
        bonus_base=array(premium if benefit_base is None else benefit_base),
        premium_at_issue=array(premium),
        weight=array(1.0),
        deferral_years=array(deferral_years, int),
        bonus_years_remaining=array(max(0, BONUS_PERIOD_YEARS - years_since_issue), int),
        adjustment_year=array(adjustment_year, int),
        adjustment_amount=array(terms.gwb_adjustment_pct * premium),
        gawa_pct=array(float(terms.withdrawal_rate(np.array([first_withdrawal_age]))[0])),
        rider_charge_pct=array(terms.charge_pct),
        bonus_pct=array(terms.bonus_pct),
        annual_step_up=np.array([terms.annual_step_up], dtype=bool),
        utilisation=array(utilisation),
        lapse_rate=array(lapse_rate),
        lapse_beta=array(lapse_beta),
        lapse_floor=array(lapse_floor),
        account_drag=array(drag),
        insurer_drag_share=array(base_contract_charge / drag if drag > 0 else 0.0),
        projection_years=array(max_age - attained, int),
        option=np.array([terms.option], dtype=object),
        death_benefit_base=array(premium),
        db_rollup_pct=array(float(death_benefit.rollup_rate(issue_age))),
        db_ratchet=np.array([death_benefit.highest_anniversary], dtype=bool),
        db_charge_pct=array(death_benefit.charge_pct),
        db_free_withdrawal_pct=array(death_benefit.free_withdrawal_pct),
        death_benefit=np.array([death_benefit.name], dtype=object),
    )
