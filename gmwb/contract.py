"""The representative contract.

Terms follow Perspective II with the Flex GMWB (Single) rider on the Core benefit
option, elected at issue. Sources: the Perspective II statutory prospectus
(485BPOS, filed 2026-04-21) for the mechanics, and the Rate Sheet Prospectus
Supplement dated 2026-04-27 (Form 497, filed 2026-04-09) for the current charges
and rates.

Mechanics that matter for the valuation, with the prospectus language behind each:

* The rider charge is a percentage of the GWB, not of the contract value. Jackson's
  10-K gives the reason: charging on the benefit base "supports our hedging program
  by stabilizing the guarantee fees we earn".
* The charge is paid "through the earlier date that you annuitize the Contract or
  your Contract Value is zero". Fee income stops precisely when claims start, which
  is most of what makes the guarantee expensive.
* Step-up: on each contract anniversary, if the contract value is above the GWB,
  the GWB is reset to the contract value. The Core option uses the Contract
  Anniversary Value method.
* With the For Life Guarantee in effect, which it is at any age past 59 1/2, once
  the contract value is exhausted "the Owner will receive annual payments of the
  GAWA until the death of the Designated Life". The tail is a life annuity, not a
  run-off of the remaining GWB.
* No death benefit is payable under this rider on its own: "If you die, all rights
  under your Contract cease." The Flex DB death benefit is a separate election and
  is out of scope here.
* Taking a withdrawal in the first contract year voids the GWB adjustment provision
  and forfeits the bonus, so a policy that draws income from the start has neither.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GmwbContract:
    premium: float
    issue_age: int
    gawa_pct: float
    rider_charge_pct: float
    base_contract_charge: float
    fund_expense: float
    annual_step_up: bool = True
    fund_equity_beta: float = 1.0
    first_withdrawal_year: int = 1

    # Policyholder behaviour. The base case is the static assumption the academic
    # literature uses as its reference point: the owner draws the full guaranteed amount
    # every year and never surrenders. Bauer, Kling and Russ (2008) treat that as the
    # benchmark precisely because it maximises the guarantee's value, which makes it an
    # upper bound rather than a central estimate.
    utilisation: float = 1.0   # share of the guaranteed annual amount actually withdrawn
    lapse_rate: float = 0.0    # annual surrender rate, applied only while the contract
                               # value is above zero, since an exhausted contract has no
                               # surrender value left to take

    @property
    def account_drag(self) -> float:
        """Continuous proportional drag on the contract value. The base contract
        charge is deducted from contract value and fund expenses are netted inside
        the funds; both scale with the balance, so they combine."""
        return self.base_contract_charge + self.fund_expense

    @classmethod
    def from_config(cls, cfg: dict) -> "GmwbContract":
        c = cfg["contract"]
        return cls(
            premium=float(c["premium"]),
            issue_age=int(c["issue_age"]),
            gawa_pct=float(c["gawa_pct"]),
            rider_charge_pct=float(c["rider_charge_pct"]),
            base_contract_charge=float(c["base_contract_charge"]),
            fund_expense=float(c["fund_expense"]),
            annual_step_up=bool(c["annual_step_up"]),
            fund_equity_beta=float(c["fund_equity_beta"]),
            first_withdrawal_year=int(c["first_withdrawal_year"]),
            utilisation=float(c.get("utilisation", 1.0)),
            lapse_rate=float(c.get("lapse_rate", 0.0)),
        )
