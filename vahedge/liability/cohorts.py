"""The book as a grid of model points rather than one illustrative policy.

One contract is an illustration. Jackson's variable annuity book is $236bn of account value
spread across issue ages, durations and moneyness, and the disclosed sensitivity is an
aggregate over all of it. A single policy at a single moneyness cannot reproduce that, and the
earlier version of this project showed why: its sensitivities came out two to three times the
disclosed figures with a near-constant multiple, which is the signature of a structural
mismatch rather than a pile of errors.

The grid here is five issue ages, four durations, five moneyness levels and both phases of the
contract, weighted to what Jackson actually discloses about the book. What each dimension is
for:

*Issue age* sets the withdrawal percentage, through the age band at the first withdrawal, and
sets the horizon through mortality. A 55-year-old on the Core option withdraws 4.00% of the
benefit base for life; a 75-year-old withdraws 5.95%. That is not a detail: it is a 50% higher
draw on a shorter life.

*Duration* decides where the contract sits relative to the bonus period, the GWB adjustment
date, and the point at which the owner starts taking income.

*Moneyness*, the benefit base over the contract value, is what decides whether the guarantee is
worth anything. A contract whose account value is well above its benefit base is a fee stream;
one whose account value has fallen below it is a claim.

*Phase* separates a contract that has started taking income from one still deferring. The
difference is not cosmetic. A deferring contract is still earning a bonus, still eligible for
the GWB adjustment that floors its benefit base at 105% of the original, and has not yet begun
to draw its account down. The earlier version of this project excluded deferral-phase contracts
entirely and flagged that as the most valuable thing to add.

Weights come from the FY2025 10-K where it discloses them and are stated as assumptions where
it does not. Nothing here is tuned to make a validation target land.
"""

from __future__ import annotations

from dataclasses import dataclass

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
                "gwb_over_av": self.benefit_base / self.account_value,
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


@dataclass(frozen=True)
class GridSpec:
    """What the grid spans, and the behaviour assumptions attached to it."""

    issue_ages: tuple = (55, 60, 65, 70, 75)
    durations: tuple = (0, 3, 6, 10)
    gwb_over_av: tuple = (0.6, 0.8, 1.0, 1.25, 1.5)
    income_start_age: int = 70
    utilisation: float = 1.0
    lapse_rate: float = 0.0
    max_age: int = 115
    premium: float = 100.0

    # How much of the account value sits at each moneyness level. A book that has seen a
    # decade of rising equity markets is concentrated where the contract value is well above
    # the benefit base, which is where Jackson's disclosed sensitivity per dollar of account
    # value says it sits. These are an assumption and the robustness sweep moves them.
    moneyness_weights: tuple = (0.06, 0.14, 0.25, 0.30, 0.25)
    duration_weights: tuple = (0.15, 0.25, 0.30, 0.30)
    age_weights: tuple = (0.12, 0.20, 0.26, 0.24, 0.18)


def build(
    spec: GridSpec,
    terms: RiderTerms,
    base_contract_charge: float,
    fund_expense: float,
    death_benefit: DeathBenefitTerms | None = None,
) -> CohortBook:
    """Expand the grid into model points.

    A cohort is in the income phase when the owner has already reached the income start age,
    and deferring otherwise. That is what decides whether the bonus is still accruing and
    whether the GWB adjustment is still alive, so it is derived rather than specified.
    """
    if not 0.0 <= spec.utilisation <= 1.0:
        raise ValueError("utilisation must be in [0, 1]")
    if not 0.0 <= spec.lapse_rate < 1.0:
        raise ValueError("lapse_rate must be in [0, 1)")
    for name, weights, values in (
        ("age", spec.age_weights, spec.issue_ages),
        ("duration", spec.duration_weights, spec.durations),
        ("moneyness", spec.moneyness_weights, spec.gwb_over_av),
    ):
        if len(weights) != len(values):
            raise ValueError(f"{name} weights do not match the {name} grid")
        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError(f"{name} weights must sum to 1, got {sum(weights)}")

    death_benefit = death_benefit or DEATH_BENEFITS["basic"]
    drag = base_contract_charge + fund_expense
    rows = []
    for age, age_weight in zip(spec.issue_ages, spec.age_weights):
        for duration, duration_weight in zip(spec.durations, spec.duration_weights):
            attained = age + duration
            if attained >= spec.max_age:
                continue
            deferral = max(0, spec.income_start_age - attained)
            first_withdrawal_age = max(attained, spec.income_start_age)

            # The GWB adjustment date, in contract years from issue. The prospectus makes it
            # the later of the anniversary on or following the seventieth birthday and the
            # twelfth anniversary; an owner already past seventy at issue gets the first
            # anniversary, because the effective date is not an anniversary.
            anniversary_at_70 = max(1, GWB_ADJUSTMENT_AGE - age)
            adjustment_contract_year = max(anniversary_at_70, GWB_ADJUSTMENT_MIN_YEARS)
            adjustment_year = adjustment_contract_year - duration
            # It only survives if the owner has taken no withdrawal by that date, in the past
            # or in the projection.
            alive = adjustment_year >= 0 and deferral > adjustment_year
            if not alive:
                adjustment_year = -1

            # The benefit base has only ever ratcheted up. For a contract still deferring, the
            # floor on that ratchet is the bonus it has already collected; a contract taking
            # income is assumed to have stepped up no further than its premium. Both are
            # assumptions, they set the GWB adjustment's reference point, and only the 200%
            # adjustment on the Plus option is materially sensitive to them.
            if deferral > 0:
                gwb_multiple = 1.0 + terms.bonus_pct * min(duration, BONUS_PERIOD_YEARS)
            else:
                gwb_multiple = 1.0

            for ratio, money_weight in zip(spec.gwb_over_av, spec.moneyness_weights):
                benefit_base = spec.premium * gwb_multiple
                account_value = benefit_base / ratio
                rows.append(
                    {
                        "issue_age": age,
                        "years_since_issue": duration,
                        "attained_age": attained,
                        "account_value": account_value,
                        "benefit_base": benefit_base,
                        "bonus_base": benefit_base,
                        "premium_at_issue": spec.premium,
                        "weight": age_weight * duration_weight * money_weight,
                        "deferral_years": deferral,
                        "bonus_years_remaining": max(0, BONUS_PERIOD_YEARS - duration),
                        "adjustment_year": adjustment_year,
                        "adjustment_amount": terms.gwb_adjustment_pct * spec.premium,
                        "gawa_pct": float(terms.withdrawal_rate(np.array([first_withdrawal_age]))[0]),
                        "rider_charge_pct": terms.charge_pct,
                        "bonus_pct": terms.bonus_pct,
                        "annual_step_up": terms.annual_step_up,
                        "utilisation": spec.utilisation,
                        "lapse_rate": spec.lapse_rate,
                        "account_drag": drag,
                        "insurer_drag_share": base_contract_charge / drag if drag > 0 else 0.0,
                        "projection_years": spec.max_age - attained,
                        "option": terms.option,
                        "death_benefit": death_benefit.name,
                        # The basic benefit base is premium; an add-on base is premium
                        # rolled or ratcheted to today. The grid states it at premium and
                        # lets the projection take it forward, which understates an
                        # in-force add-on base by the roll-up already accrued.
                        "death_benefit_base": spec.premium,
                        "db_rollup_pct": float(death_benefit.rollup_rate(age)),
                        "db_ratchet": death_benefit.highest_anniversary,
                        "db_charge_pct": death_benefit.charge_pct,
                        "db_free_withdrawal_pct": death_benefit.free_withdrawal_pct,
                    }
                )

    if not rows:
        raise ValueError("the grid produced no cohorts; check the ages against max_age")

    frame = pd.DataFrame(rows)
    frame["weight"] = frame["weight"] / frame["weight"].sum()
    boolean = {"annual_step_up", "db_ratchet"}
    text = {"option", "death_benefit"}
    return CohortBook(
        **{
            column: frame[column].to_numpy(
                dtype=bool if column in boolean
                else object if column in text
                else float if frame[column].dtype.kind == "f"
                else int
            )
            for column in frame.columns
        }
    )


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
    death_benefit: DeathBenefitTerms | None = None,
) -> CohortBook:
    """A one-cohort book, for the at-issue valuation and for the hedging backtest.

    The backtest rolls one representative policy along realised history, so it needs the same
    projection machinery the book uses, pointed at a single model point.
    """
    death_benefit = death_benefit or DEATH_BENEFITS["basic"]
    attained = issue_age + years_since_issue
    anniversary_at_70 = max(1, GWB_ADJUSTMENT_AGE - issue_age)
    adjustment_contract_year = max(anniversary_at_70, GWB_ADJUSTMENT_MIN_YEARS)
    adjustment_year = adjustment_contract_year - years_since_issue
    if adjustment_year < 0 or deferral_years <= adjustment_year:
        adjustment_year = -1

    drag = base_contract_charge + fund_expense
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
