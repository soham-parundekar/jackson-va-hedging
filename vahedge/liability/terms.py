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
    """The GMDB variants Jackson sells alongside the withdrawal rider.

    The rate sheet gives three: a roll-up benefit base growing at 6% below age 70 and 5% from
    70, a highest quarterly anniversary value benefit, and the combination of the two, charged
    at 0.90%, 0.30% and 1.00% of the death benefit base. The return-of-premium benefit is
    included at no charge in the base contract.

    The highest-anniversary variant is modelled on annual anniversaries, not quarterly. The
    liability recursion steps a policy year at a time because every living-benefit event does,
    and re-stepping it quarterly to catch the death benefit's ratchet would cost four times as
    much for a benefit that is a small fraction of the book's value. The direction of the
    error is known and stated: an annual ratchet is a floor on a quarterly one, so this
    understates the highest-anniversary death benefit.
    """

    name: str
    charge_pct: float
    rollup_pct_below_70: float = 0.0
    rollup_pct_from_70: float = 0.0
    highest_anniversary: bool = False
    rollup_cap_multiple: float = 2.0

    def rollup_rate(self, attained_age):
        age = np.asarray(attained_age)
        return np.where(age < 70, self.rollup_pct_below_70, self.rollup_pct_from_70)


DEATH_BENEFITS = {
    "return_of_premium": DeathBenefitTerms(name="return_of_premium", charge_pct=0.0),
    "rollup": DeathBenefitTerms(
        name="rollup", charge_pct=0.0090,
        rollup_pct_below_70=0.06, rollup_pct_from_70=0.05,
    ),
    "highest_anniversary": DeathBenefitTerms(
        name="highest_anniversary", charge_pct=0.0030, highest_anniversary=True,
    ),
    "combination": DeathBenefitTerms(
        name="combination", charge_pct=0.0100,
        rollup_pct_below_70=0.06, rollup_pct_from_70=0.05, highest_anniversary=True,
    ),
}
