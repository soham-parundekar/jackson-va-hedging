"""Policyholder behaviour, which moves this liability more than the market model does.

Two decisions matter and neither is observable from outside the company.

*Whether the owner keeps the contract.* A guarantee is worth nothing to an insurer whose
policyholders surrender, and worth a great deal when they do not. Surrender is not random: an
owner whose benefit base sits well above the contract value is holding something valuable and
knows it. The standard shape for that is a base lapse rate damped by how far the guarantee is
in the money,

    lapse = max(floor, base * exp(-beta * (GWB / AV - 1)^+))

which leaves the base rate alone while the contract value is above the benefit base and drives
the rate toward the floor as the guarantee moves into the money. The floor matters: some
surrenders happen for reasons that have nothing to do with the guarantee - death of a spouse,
a house purchase, an adviser moving firms - and a model that lets lapse fall to zero for a deep
in-the-money contract overstates the liability.

*When the owner starts taking income.* Later is worth more, because the withdrawal percentage
rises with the age band, the bonus keeps accruing, and the GWB adjustment survives.

Jackson calls both of these out as model inputs in Note 6 - "benefit utilization by
policyholders, lapse, mortality, and withdrawal rates" - and publishes none of them. There is
no free data that identifies them either. So nothing here is calibrated, and this module does
not pretend otherwise: it provides the shape and a grid of plausible values, and the results
are reported as ranges over that grid rather than as a point estimate with a decoration of
uncertainty around it.

The size of the effect is the reason this is not a footnote. The earlier version of this
project found the model's sensitivities running 2.1 to 2.9 times Jackson's disclosed figures
with a near-constant multiple, and the gap closed at 90% utilisation and 4% lapse - assumptions
that are unremarkable in isolation and that move the answer by more than the choice between
Black-Scholes and Heston does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BehaviourAssumptions:
    """One point in the behaviour grid.

    ``lapse_beta`` of zero gives a static lapse rate, which is what the base case uses so that
    the dynamic version can be measured against it rather than assumed to be better.
    """

    utilisation: float = 1.0
    base_lapse: float = 0.0
    lapse_beta: float = 0.0
    lapse_floor: float = 0.0
    income_start_age: int = 70

    def __post_init__(self) -> None:
        if not 0.0 <= self.utilisation <= 1.0:
            raise ValueError("utilisation must be in [0, 1]")
        if not 0.0 <= self.base_lapse < 1.0:
            raise ValueError("base_lapse must be in [0, 1)")
        if self.lapse_beta < 0:
            raise ValueError("lapse_beta must not be negative; it damps lapse, never lifts it")
        if not 0.0 <= self.lapse_floor <= self.base_lapse:
            raise ValueError("lapse_floor must be between zero and the base lapse rate")
        if not 50 <= self.income_start_age <= 90:
            raise ValueError("income_start_age outside the range the rate sheet bands cover")

    @property
    def is_dynamic(self) -> bool:
        return self.lapse_beta > 0.0


def dynamic_lapse(benefit_base, account_value, base_lapse, beta, floor):
    """Lapse rate given how far the guarantee is in the money.

    ``base_lapse``, ``beta`` and ``floor`` broadcast against the state arrays, so a book with
    different assumptions per cohort costs the same as one assumption for all of them.

    An exhausted contract has nothing to surrender. That case is handled by the caller rather
    than here, because it is a property of the contract rather than of the behaviour model.
    """
    account = np.maximum(np.asarray(account_value, dtype=float), 1e-12)
    moneyness = np.asarray(benefit_base, dtype=float) / account
    in_the_money = np.maximum(moneyness - 1.0, 0.0)
    return np.maximum(floor, base_lapse * np.exp(-beta * in_the_money))


# The grid the sensitivity tables sweep. The base case sits first and is the static
# full-utilisation benchmark that Bauer, Kling and Russ use, which is an upper bound on the
# guarantee's value by construction rather than a central estimate.
BEHAVIOUR_GRID = (
    BehaviourAssumptions(utilisation=1.00, base_lapse=0.00),
    BehaviourAssumptions(utilisation=1.00, base_lapse=0.02),
    BehaviourAssumptions(utilisation=1.00, base_lapse=0.04),
    BehaviourAssumptions(utilisation=0.90, base_lapse=0.04),
    BehaviourAssumptions(utilisation=0.80, base_lapse=0.04),
    BehaviourAssumptions(utilisation=0.90, base_lapse=0.06, lapse_beta=2.0, lapse_floor=0.01),
    BehaviourAssumptions(utilisation=0.90, base_lapse=0.06, lapse_beta=4.0, lapse_floor=0.01),
    BehaviourAssumptions(utilisation=1.00, base_lapse=0.04, income_start_age=65),
    BehaviourAssumptions(utilisation=1.00, base_lapse=0.04, income_start_age=75),
)
