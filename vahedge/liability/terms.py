"""Contract terms, read from the rate sheet rather than assumed.

Everything in ``data/raw/jackson_rider_terms.csv`` came out of the Rate Sheet Prospectus
Supplement dated 27 April 2026. That matters more than it sounds. The Perspective II
prospectus itself quotes a maximum rider charge of 3.00% of the benefit base and defers the
actual numbers to the rate sheet, so a model built from the prospectus alone charges more
than twice the fee Jackson charges, and fee income is half of what a market risk benefit is.

The withdrawal percentage is banded by attained age at the first withdrawal and then fixed
for life. A cohort grid spanning issue ages 55 to 75 therefore spans four of the six bands,
and using a single withdrawal rate across it would misprice the young cohorts by more than a
hundred basis points of benefit base a year.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import paths

AGE_BANDS = ((35, 59), (60, 64), (65, 69), (70, 74), (75, 80), (81, 120))


@dataclass(frozen=True)
class RiderTerms:
    """One benefit option of one rider, with its full age-banded withdrawal table."""

    benefit: str
    life_basis: str
    option: str
    charge_pct: float
    bonus_pct: float
    step_up: str
    gwb_adjustment_pct: float
    age_low: np.ndarray
    age_high: np.ndarray
    gawa_pct: np.ndarray

    def withdrawal_rate(self, attained_age):
        """Guaranteed annual withdrawal percentage at the age of the first withdrawal."""
        age = np.asarray(attained_age)
        rates = np.full(age.shape, np.nan, dtype=float)
        for low, high, rate in zip(self.age_low, self.age_high, self.gawa_pct):
            rates = np.where((age >= low) & (age <= high), rate, rates)
        if np.any(~np.isfinite(rates)):
            bad = np.unique(age[~np.isfinite(rates)])
            raise ValueError(f"no withdrawal band covers attained age(s) {bad.tolist()}")
        return rates

    @property
    def annual_step_up(self) -> bool:
        return self.step_up == "annual_contract_value"


def load(path=None) -> dict[tuple[str, str, str], RiderTerms]:
    """Every benefit option in the rate sheet, keyed by (benefit, life basis, option)."""
    frame = pd.read_csv(path or paths.RIDER_TERMS, comment="#")
    out: dict[tuple[str, str, str], RiderTerms] = {}
    for key, group in frame.groupby(["benefit", "life_basis", "option"]):
        group = group.sort_values("age_low")
        for column in ("charge_pct", "bonus_pct", "step_up", "gwb_adjustment_pct"):
            if group[column].nunique() != 1:
                raise ValueError(f"{key} has more than one {column} in the rate sheet")
        out[tuple(key)] = RiderTerms(
            benefit=key[0],
            life_basis=key[1],
            option=key[2],
            charge_pct=float(group["charge_pct"].iloc[0]),
            bonus_pct=float(group["bonus_pct"].iloc[0]),
            step_up=str(group["step_up"].iloc[0]),
            gwb_adjustment_pct=float(group["gwb_adjustment_pct"].iloc[0]),
            age_low=group["age_low"].to_numpy(dtype=int),
            age_high=group["age_high"].to_numpy(dtype=int),
            gawa_pct=group["gawa_pct"].to_numpy(dtype=float),
        )
    if not out:
        raise ValueError("the rider terms file produced no benefit options")
    return out


@dataclass(frozen=True)
class DeathBenefitTerms:
    """The death benefits, with the mechanics taken from the prospectus rather than assumed.

    Every contract carries the basic benefit for nothing: the greater of the contract value and
    total premiums, reduced for prior withdrawals "in the same proportion that the Contract
    Value was reduced on the date of the withdrawal". Proportional, not dollar for dollar -
    which means a withdrawal taken when the contract is down cuts the death benefit by more
    than the cash taken out.

    Three add-ons replace that benefit base, at 0.90%, 0.30% and 1.00% of the base per the
    current rate sheet:

    *Roll-up.* The base starts at premium and compounds annually at 6% for an owner 69 or
    younger at election and 5% from 70, per the rate sheet. Compounding stops at the contract
    anniversary immediately preceding the owner's 81st birthday. Withdrawals cut the base
    dollar for dollar up to 5% of it and proportionally beyond that. (The prospectus's own
    worked examples use 5% and 4%, which were the rates when those examples were written; the
    rate sheet is the current schedule and is what is used here.)

    *Highest quarterly anniversary value.* The base ratchets to the contract value at each
    quarterly anniversary until the same 81st-birthday cut-off, and falls proportionally with
    withdrawals.

    *Combination.* The base is the greater of the two components.

    The ratchet is modelled annually rather than quarterly. Every living-benefit event lands on
    a policy anniversary, so the recursion steps a year at a time, and re-stepping it quarterly
    for the death benefit would quadruple the cost of the loop that dominates every valuation.
    The direction of the error is known and worth stating: an annual ratchet is a floor on a
    quarterly one, so the highest-anniversary base here is understated by however much the
    contract value peaked and fell back within a year.
    """

    name: str
    charge_pct: float
    rollup_pct_below_70: float = 0.0
    rollup_pct_from_70: float = 0.0
    highest_anniversary: bool = False
    free_withdrawal_pct: float = 0.0   # cut dollar for dollar up to this share of the base

    def rollup_rate(self, age_at_election):
        age = np.asarray(age_at_election)
        return np.where(age < 70, self.rollup_pct_below_70, self.rollup_pct_from_70)


DEATH_BENEFITS = {
    "basic": DeathBenefitTerms(name="basic", charge_pct=0.0),
    "rollup": DeathBenefitTerms(
        name="rollup", charge_pct=0.0090,
        rollup_pct_below_70=0.06, rollup_pct_from_70=0.05, free_withdrawal_pct=0.05,
    ),
    "highest_anniversary": DeathBenefitTerms(
        name="highest_anniversary", charge_pct=0.0030, highest_anniversary=True,
    ),
    "combination": DeathBenefitTerms(
        name="combination", charge_pct=0.0100,
        rollup_pct_below_70=0.06, rollup_pct_from_70=0.05,
        highest_anniversary=True, free_withdrawal_pct=0.05,
    ),
}
