"""Risk-neutral Monte Carlo valuation of the GMWB rider.

What gets valued is the insurer's position on the definition Jackson uses for a
market risk benefit: the present value of projected guarantee payments less the
present value of the attributed fees. Matching that definition is what makes a
single-policy number comparable in sign to the disclosed sensitivity table, and it
is why a disclosed variable annuity guarantee can sit on the balance sheet as a net
asset.

Three parts of the implementation are worth reading before the loop.

*Annual stepping is exact.* Between anniversaries nothing happens to the contract
value except growth and a proportional fee drag, so it is a geometric Brownian
motion over each year. Stepping annually with exact lognormal increments introduces
no discretisation error in the account value. Every path-dependent event - the
step-up, the point of exhaustion, the switch from fee income to claims - lands on an
anniversary, which is a date being stepped to.

*The drift is the risk-free rate less the fee drag.* The contract value earns the
sub-account's total return, and under the risk-neutral measure any traded portfolio
earns the risk-free rate. Dividends never appear here. They appear in the hedge,
where the instrument is a futures position on a price index.

*Continuous charges are collected exactly, not approximated.* The charges that
accrue on average daily value are modelled as a continuous drag, but the amount
collected still has to be measured to attribute it. For a proportional charge c out
of a total drag m, the year-end accumulated value of the charges collected during
the year is exactly (c/m) * AV_before_drag * (1 - exp(-m)). Charges left invested in
the sub-account grow at the risk-free rate under Q, so discounting that year-end
amount gives the same present value as discounting the continuous stream. No
sub-stepping is needed and no approximation enters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contract import GmwbContract
from .curves import ZeroCurve
from .mortality import MortalityBasis
from .volatility import VolTermStructure


@dataclass(frozen=True)
class RiderValuation:
    """Present values and path diagnostics for one valuation state."""

    pv_claims: float           # PV of guarantee payments made by the insurer
    pv_rider_fees: float       # PV of the explicit GMWB charge on the GWB
    pv_me_fees: float          # PV of the base contract charge (the insurer's share)
    fee_attribution: float     # share of total projected fees attributed to the rider
    net_value: float           # pv_claims - attribution * (rider + M&E fees)
    std_error: float           # Monte Carlo standard error of net_value
    account_value: float
    benefit_base: float
    times: np.ndarray
    survival: np.ndarray
    discounts: np.ndarray
    claims_by_year: np.ndarray
    rider_fees_by_year: np.ndarray
    me_fees_by_year: np.ndarray
    exhaustion_prob: np.ndarray
    mean_account_value: np.ndarray
    mean_benefit_base: np.ndarray

    @property
    def pv_total_fees(self) -> float:
        return self.pv_rider_fees + self.pv_me_fees

    @property
    def net_pct_of_av(self) -> float:
        return self.net_value / self.account_value

    @property
    def gross_pct_of_av(self) -> float:
        return self.pv_claims / self.account_value


def make_normals(n_paths: int, n_years: int, seed: int, antithetic: bool = True) -> np.ndarray:
    """Standard normals of shape (n_paths, n_years).

    One array is reused across every revaluation in a run. That is what makes a
    bumped price comparable to the base price: the difference is a sensitivity rather
    than the gap between two independent simulations. Under antithetic sampling the
    second half of the array is the negative of the first.
    """
    rng = np.random.default_rng(seed)
    if antithetic:
        if n_paths % 2:
            raise ValueError("antithetic sampling needs an even n_paths")
        half = rng.standard_normal((n_paths // 2, n_years))
        return np.concatenate([half, -half], axis=0)
    return rng.standard_normal((n_paths, n_years))


def projection_years(contract: GmwbContract, max_age: int) -> int:
    """Years to project. The cap is on attained age; survival past it is small enough
    that truncating costs less than the Monte Carlo error, which the tests check."""
    if max_age <= contract.issue_age:
        raise ValueError("max_age must exceed the issue age")
    return int(max_age - contract.issue_age)


def value_rider(
    contract: GmwbContract,
    curve: ZeroCurve,
    vol: VolTermStructure,
    mortality: MortalityBasis,
    normals: np.ndarray,
    account_value: float | None = None,
    benefit_base: float | None = None,
    equity_shock: float = 0.0,
    antithetic: bool = True,
    credit_spread: float = 0.0,
    fee_attribution: float = 1.0,
    max_age: int = 115,
    first_step_years: float = 1.0,
) -> RiderValuation:
    """Value the rider from the state (account_value, benefit_base).

    ``equity_shock`` is an instantaneous proportional move in the equity index,
    passed through to the contract value by the sub-account's equity beta. The
    benefit base does not move with it, which is the entire point of the guarantee.

    ``credit_spread`` is added to the zero curve. Zero gives the economic value the
    hedging programme targets. A positive spread gives the own-credit adjusted value
    that a reported market risk benefit carries.

    ``fee_attribution`` is the share of total projected fees attributed to the
    guarantee. Jackson fixes this percentage at inception and holds it static for the
    life of the contract; ``calibrate_attribution`` reproduces that calculation.

    ``first_step_years`` is the time to the next contract anniversary. An in-force
    policy is rarely valued on its anniversary, and treating every date as one would
    put a sawtooth into a weekly hedging backtest. Set it to one at issue.
    """
    n_paths, n_years_available = normals.shape
    n_years = projection_years(contract, max_age)
    if n_years_available < n_years:
        raise ValueError(
            f"normals provide {n_years_available} years, the projection needs {n_years}"
        )
    if not 0.0 <= fee_attribution <= 1.0:
        raise ValueError("fee_attribution must be in [0, 1]")
    if not 0.0 < first_step_years <= 1.0:
        raise ValueError("first_step_years must be in (0, 1]")

    times = first_step_years + np.arange(n_years, dtype=float)
    intervals = np.diff(np.concatenate(([0.0], times)))
    survival = mortality.survival_at(times)

    zero_rates = curve.zero(times) + credit_spread
    discounts = np.exp(-zero_rates * times)
    # Integrated forward rate over each interval, which is what the lognormal step
    # needs; the first interval is shorter than a year for an in-force policy.
    forwards = np.diff(np.concatenate(([0.0], zero_rates * times)))

    beta = contract.fund_equity_beta
    step_var = vol.step_variances(times) * beta**2
    step_vol = np.sqrt(step_var)

    drag = contract.account_drag
    if drag < 0:
        raise ValueError("account_drag must not be negative")
    # The share of the drag that is the insurer's revenue rather than the funds'. With no
    # drag at all there is nothing to split, which is the configuration the martingale
    # test in tests/ uses to check the risk-neutral drift.
    insurer_drag_share = contract.base_contract_charge / drag if drag > 0 else 0.0
    drag_factor = np.exp(-drag * intervals)
    collected_per_unit = 1.0 - drag_factor

    start_av = (account_value if account_value is not None else contract.premium) * (
        1.0 + beta * equity_shock
    )
    start_gwb = benefit_base if benefit_base is not None else contract.premium
    av = np.full(n_paths, start_av, dtype=float)
    gwb = np.full(n_paths, start_gwb, dtype=float)

    claims_by_year = np.zeros(n_years)
    rider_fees_by_year = np.zeros(n_years)
    me_fees_by_year = np.zeros(n_years)
    exhaustion_prob = np.zeros(n_years)
    mean_av = np.zeros(n_years)
    mean_gwb = np.zeros(n_years)

    claim_pv = np.zeros(n_paths)
    rider_fee_pv = np.zeros(n_paths)
    me_fee_pv = np.zeros(n_paths)

    # Persistency runs alongside mortality. Surrender is only possible while there is a
    # contract value to surrender, so once a path exhausts, its persistency stops
    # decaying. That makes the decrement path-dependent, which is why it is carried per
    # path rather than folded into the survival curve.
    in_force = np.ones(n_paths)
    lapse_survival = 1.0 - contract.lapse_rate
    if not 0.0 <= contract.lapse_rate < 1.0:
        raise ValueError("lapse_rate must be in [0, 1)")
    if not 0.0 <= contract.utilisation <= 1.0:
        raise ValueError("utilisation must be in [0, 1]")

    for k in range(n_years):
        av_pre_drag = av * np.exp(
            forwards[k] - 0.5 * step_var[k] + step_vol[k] * normals[:, k]
        )
        av = av_pre_drag * drag_factor[k]
        me_charge = insurer_drag_share * av_pre_drag * collected_per_unit[k]

        # Explicit GMWB charge, taken on the GWB at the start of the contract year and
        # only to the extent the contract value can cover it. It stops when the
        # contract value reaches zero.
        rider_charge = np.minimum(contract.rider_charge_pct * gwb, av)
        av -= rider_charge

        # Guaranteed withdrawal, off the GWB before this anniversary's step-up. The
        # owner draws exactly the guaranteed amount every year.
        if k + 1 >= contract.first_withdrawal_year:
            # Withdrawing less than the guaranteed amount forfeits the difference, so
            # partial utilisation both delays exhaustion and lowers the claim when it
            # comes.
            drawn = contract.utilisation * contract.gawa_pct * gwb
            from_account = np.minimum(drawn, av)
            claim = drawn - from_account
            av -= from_account
        else:
            claim = np.zeros(n_paths)

        if contract.annual_step_up:
            np.maximum(gwb, av, out=gwb)

        weight = discounts[k] * survival[k] * in_force
        claim_pv += weight * claim
        rider_fee_pv += weight * rider_charge
        me_fee_pv += weight * me_charge

        claims_by_year[k] = (claim * in_force).mean()
        rider_fees_by_year[k] = (rider_charge * in_force).mean()
        me_fees_by_year[k] = (me_charge * in_force).mean()
        exhaustion_prob[k] = float((av <= 0.0).mean())
        mean_av[k] = av.mean()
        mean_gwb[k] = gwb.mean()

        if lapse_survival < 1.0:
            in_force = np.where(av > 0.0, in_force * lapse_survival, in_force)

    net_path = claim_pv - fee_attribution * (rider_fee_pv + me_fee_pv)
    std_error = _standard_error(net_path, antithetic)

    return RiderValuation(
        pv_claims=float(claim_pv.mean()),
        pv_rider_fees=float(rider_fee_pv.mean()),
        pv_me_fees=float(me_fee_pv.mean()),
        fee_attribution=float(fee_attribution),
        net_value=float(net_path.mean()),
        std_error=std_error,
        account_value=float(start_av),
        benefit_base=float(start_gwb),
        times=times,
        survival=survival,
        discounts=discounts,
        claims_by_year=claims_by_year,
        rider_fees_by_year=rider_fees_by_year,
        me_fees_by_year=me_fees_by_year,
        exhaustion_prob=exhaustion_prob,
        mean_account_value=mean_av,
        mean_benefit_base=mean_gwb,
    )


def calibrate_attribution(valuation: RiderValuation) -> float:
    """The attribution percentage implied by an at-issue valuation.

    Jackson's description: at inception a portion of total projected fees is
    attributed to the benefit to offset projected claims, expressed as a percentage
    of total projected fees, and that percentage may not exceed 100%. Where the
    attributable fees cover the projected claims the benefit starts at a fair value
    of zero. That is exactly ``min(1, PV(claims) / PV(total fees))``.
    """
    total_fees = valuation.pv_total_fees
    if total_fees <= 0:
        raise ValueError("no attributable fees; cannot calibrate an attribution percentage")
    return float(min(1.0, valuation.pv_claims / total_fees))


def _standard_error(net_path: np.ndarray, antithetic: bool) -> float:
    """Standard error of the mean. Antithetic pairs are dependent, so the variance has
    to be taken across pair averages rather than across paths."""
    n = net_path.size
    if antithetic and n % 2 == 0:
        half = n // 2
        pair_means = 0.5 * (net_path[:half] + net_path[half:])
        return float(pair_means.std(ddof=1) / np.sqrt(half))
    return float(net_path.std(ddof=1) / np.sqrt(n))
