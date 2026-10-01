"""The lifetime withdrawal guarantee, projected a policy year at a time.

What the rider promises. The owner puts a premium into sub-accounts. The rider guarantees a
withdrawal every year for life, equal to a percentage of a benefit base, and keeps paying
after the contract value is exhausted. The insurer pays the shortfall. Two features make the
benefit base ratchet upward and never down: an annual step-up to the contract value, and a
bonus during deferral. The rider fee is charged on that benefit base rather than on the
contract value, so fee income does not fall when markets do while the guarantee is getting
more expensive. Jackson says why in its own 10-K: charging on the benefit base "supports our
hedging program by stabilizing the guarantee fees we earn".

Everything below comes from the Perspective II prospectus and its rate sheet, not from a
textbook GMWB. Three mechanics in particular are specific enough to be worth stating.

*The bonus is simple, not compound.* It is a percentage of the Bonus Base applied to the GWB
at the end of each contract year with no withdrawal. The Bonus Base is not the GWB: it starts
equal to it, and afterwards moves only with premiums and step-ups. So a 6% bonus on a
deferring contract adds 6% of the original benefit base each year, not 6% compounding. Treating
it as a roll-up on the GWB, which is what most GMWB descriptions assume, overstates a ten-year
deferral by roughly a third.

*The bonus period restarts on a step-up.* It runs ten contract years from the endorsement, or
ten years from the most recent step-up that raised the Bonus Base, whichever is later, and
only while the owner is 80 or younger. A contract that keeps ratcheting can therefore collect
a bonus for far longer than ten years, which is exactly the path where the guarantee is least
valuable, so the feature costs less than its headline suggests.

*The GWB adjustment is a floor on the benefit base, paid for by never withdrawing.* On the
later of the anniversary following the owner's seventieth birthday and the twelfth contract
anniversary, a contract that has taken no withdrawal at all has its GWB reset to the greater
of its current value and a fixed percentage of the GWB at issue - 105% on the Core option,
200% on Plus. Any withdrawal before that date kills the provision outright. It is the single
largest reason a deferral-phase cohort is worth modelling separately from an income-phase one.

The projection steps a policy year at a time because every one of those events lands on an
anniversary. Within a year the only things that happen are growth and a proportional fee drag,
and the market simulator has already integrated both. The recursion is vectorised over
cohorts and paths together, so the Python loop runs once per year rather than once per cohort.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import behaviour

BONUS_PERIOD_YEARS = 10
BONUS_RESTART_MAX_AGE = 80
# Roll-up accrual and the anniversary ratchet both stop at "the Contract Anniversary
# immediately preceding the oldest Covered Life's 81st birthday", which is the last
# anniversary at which the owner is 80.
DEATH_BENEFIT_FREEZE_AGE = 80


@dataclass(frozen=True)
class GmwbProjection:
    """Present values per cohort, and the diagnostics needed to read them."""

    pv_claims: np.ndarray          # (n_cohorts,) PV of living-benefit payments
    pv_death_claims: np.ndarray    # (n_cohorts,) PV of death-benefit payments above the account
    pv_rider_fees: np.ndarray      # (n_cohorts,) PV of the explicit charge on the GWB
    pv_base_fees: np.ndarray       # (n_cohorts,) PV of the insurer's share of the account drag
    pv_death_fees: np.ndarray      # (n_cohorts,) PV of the explicit charge for the GMDB
    claims_by_year: np.ndarray     # (n_cohorts, n_years) expected, discounted
    death_claims_by_year: np.ndarray
    fees_by_year: np.ndarray       # (n_cohorts, n_years) expected, discounted
    exhaustion_prob: np.ndarray    # (n_cohorts, n_years) share of paths with no contract value
    mean_account_value: np.ndarray
    mean_benefit_base: np.ndarray
    mean_death_benefit: np.ndarray
    claim_paths: np.ndarray        # (n_cohorts, n_paths) PV per path, for standard errors
    fee_paths: np.ndarray          # (n_cohorts, n_paths) PV per path
    recorded: dict | None = None   # per-path state by year, only when record=True
    # (n_paths, n_years), summed over cohorts, only when record_deficiency=True. The first is
    # the time-zero present value of net outgo in each year; the second the in-force account
    # value at each year end, for the surrender value floor.
    deficiency_pv: np.ndarray | None = None
    in_force_account: np.ndarray | None = None

    @property
    def pv_attributable_fees(self) -> np.ndarray:
        return self.pv_rider_fees + self.pv_base_fees + self.pv_death_fees

    @property
    def pv_total_claims(self) -> np.ndarray:
        return self.pv_claims + self.pv_death_claims


def project(
    book,
    paths,
    survival: np.ndarray,
    deaths: np.ndarray | None = None,
    death_benefit=None,
    equity_shock: float = 0.0,
    equity_weight: float = 1.0,
    record: bool = False,
    record_deficiency: bool = False,
) -> GmwbProjection:
    """Project every cohort in the book across every path.

    ``survival`` is (n_cohorts, n_years): the probability of the owner being alive at the end
    of each policy year, already measured from the valuation date. Lapse is separate and path
    dependent, because a contract with no value left has no surrender value to take, so it
    cannot lapse; folding lapse into the survival curve would quietly keep surrendering
    contracts that have already exhausted.

    ``equity_shock`` scales the starting account value by the sub-account's equity weight. A
    10% index fall does not take 10% off a contract value that is a fifth in bonds, and
    applying the shock to the whole account value is the most common way to get a disclosed
    equity sensitivity wrong by the weight of the bond sleeve.

    The death benefit rides along in the same loop rather than in a module of its own, and
    that is deliberate. The two guarantees are written on one account: a withdrawal reduces
    the death benefit floor, an exhausted contract has no account value for the beneficiary to
    receive, and the same step-up feeds both. Running them separately would mean two
    projections that agree only by construction, and would double the cost of the loop that
    dominates every valuation. ``deaths`` is the probability of dying during each policy year,
    from the same mortality pass that produced ``survival``.
    """
    n_cohorts = book.size
    n_paths, n_years_available = paths.fund_growth.shape
    n_years = int(book.projection_years.max())
    if n_years_available < n_years:
        raise ValueError(
            f"paths carry {n_years_available} years, the oldest cohort needs {n_years}"
        )
    if survival.shape != (n_cohorts, n_years_available) and survival.shape[1] < n_years:
        raise ValueError(f"survival is {survival.shape}, needs at least {n_cohorts} x {n_years}")

    shock_factor = 1.0 + equity_weight * equity_shock
    if shock_factor <= 0:
        raise ValueError("the equity shock wipes out the contract value; check its sign")

    # ---- per-cohort constants, shaped (n_cohorts, 1) so they broadcast against paths
    col = lambda values: np.asarray(values, dtype=float).reshape(-1, 1)
    gawa = col(book.gawa_pct)
    rider_rate = col(book.rider_charge_pct)
    bonus_rate = col(book.bonus_pct)
    utilisation = col(book.utilisation)
    drag = col(book.account_drag)
    insurer_share = col(book.insurer_drag_share)
    step_up = np.asarray(book.annual_step_up, dtype=bool).reshape(-1, 1)
    horizon = np.asarray(book.projection_years, dtype=int).reshape(-1, 1)
    deferral_years = np.asarray(book.deferral_years, dtype=int).reshape(-1, 1)
    attained_age = np.asarray(book.attained_age, dtype=int).reshape(-1, 1)
    adjustment_year = np.asarray(book.adjustment_year, dtype=int).reshape(-1, 1)
    adjustment_amount = col(book.adjustment_amount)
    lapse_rate = col(book.lapse_rate)
    lapse_beta = col(book.lapse_beta)
    lapse_floor = col(book.lapse_floor)
    dynamic = bool(np.any(lapse_beta > 0.0))

    drag_factor = np.exp(-drag)
    collected_per_unit = 1.0 - drag_factor

    # ---- death benefit. Every contract carries the basic benefit for nothing; the add-ons
    # replace its benefit base with a rolled-up or ratcheted one. All three variants pay the
    # excess of the benefit base over the contract value, so the code differs only in how the
    # base moves.
    has_death = deaths is not None
    if has_death:
        if deaths.shape[0] != n_cohorts:
            raise ValueError("deaths must have one row per cohort")
        rollup_rate = col(book.db_rollup_pct)
        ratchets = np.asarray(book.db_ratchet, dtype=bool).reshape(-1, 1)
        free_withdrawal = col(book.db_free_withdrawal_pct)
        death_charge_rate = col(book.db_charge_pct)
        rollup_base = np.broadcast_to(col(book.death_benefit_base), (n_cohorts, n_paths)).copy()
        ratchet_base = (
            rollup_base.copy()
            if getattr(book, "death_ratchet_base", None) is None
            else np.broadcast_to(col(book.death_ratchet_base), (n_cohorts, n_paths)).copy()
        )
        death_claim_pv = np.zeros((n_cohorts, n_paths))
        death_fee_pv = np.zeros((n_cohorts, n_paths))
        death_claims_by_year = np.zeros((n_cohorts, n_years))
        mean_death_benefit = np.zeros((n_cohorts, n_years))
    else:
        death_claim_pv = np.zeros((n_cohorts, n_paths))
        death_fee_pv = np.zeros((n_cohorts, n_paths))
        death_claims_by_year = np.zeros((n_cohorts, n_years))
        mean_death_benefit = np.zeros((n_cohorts, n_years))

    # ---- state, (n_cohorts, n_paths)
    account = np.broadcast_to(col(book.account_value) * shock_factor, (n_cohorts, n_paths)).copy()
    benefit_base = np.broadcast_to(col(book.benefit_base), (n_cohorts, n_paths)).copy()
    bonus_base = np.broadcast_to(col(book.bonus_base), (n_cohorts, n_paths)).copy()
    in_force = np.ones((n_cohorts, n_paths))
    # The bonus period ends this many policy years from the valuation date. It restarts on a
    # step-up that raises the Bonus Base, so it has to be carried per path.
    bonus_end = np.broadcast_to(
        np.asarray(book.bonus_years_remaining, dtype=float).reshape(-1, 1), (n_cohorts, n_paths)
    ).copy()
    # The GWB adjustment survives only while no withdrawal has ever been taken. Every cohort
    # already in the income phase has lost it, which book.adjustment_year encodes as -1.
    adjustment_live = np.broadcast_to(adjustment_year >= 0, (n_cohorts, n_paths)).copy()

    claim_pv = np.zeros((n_cohorts, n_paths))
    fee_pv = np.zeros((n_cohorts, n_paths))
    rider_fee_pv = np.zeros((n_cohorts, n_paths))

    # Recording keeps the per-path state and cash flow at every anniversary, which is what
    # the regression proxy regresses on. It is off by default because it costs one array per
    # recorded quantity per year, and every valuation would pay for it.
    if record:
        if n_cohorts != 1:
            raise ValueError("recording is for one cohort at a time; the arrays are per path")
        recorded = {
            name: np.empty((n_paths, n_years))
            for name in ("account_value", "benefit_base", "bonus_base", "bonus_end",
                         "adjustment_live", "death_rollup_base", "death_ratchet_base",
                         "pre_account_value", "pre_benefit_base",
                         "pv_claim", "pv_fee", "discount", "persistency",
                         "variance", "zero_10y", "short_rate")
        }
    else:
        recorded = None

    if record_deficiency:
        deficiency_pv = np.zeros((n_paths, n_years))
        in_force_account = np.zeros((n_paths, n_years))
    else:
        deficiency_pv = None
        in_force_account = None

    claims_by_year = np.zeros((n_cohorts, n_years))
    fees_by_year = np.zeros((n_cohorts, n_years))
    exhaustion = np.zeros((n_cohorts, n_years))
    mean_account = np.zeros((n_cohorts, n_years))
    mean_base = np.zeros((n_cohorts, n_years))

    for year in range(n_years):
        alive = year < horizon                       # (n_cohorts, 1), cohort past its horizon
        growth = paths.fund_growth[:, year][None, :]
        discount = paths.discount[:, year][None, :]

        account_before_drag = account * growth
        account = account_before_drag * drag_factor
        base_charge = insurer_share * account_before_drag * collected_per_unit
        if recorded is not None:
            # The state an instant before the anniversary's events: the year's growth and the
            # continuous drag have happened, the charges, the withdrawal and the step-up have
            # not. This is the only state in the recursion whose contract value can sit above
            # its benefit base, because the step-up exists precisely to stop that lasting, and
            # it is the shape of every state a hedge sees between anniversaries. A proxy fitted
            # only on post-event states has no design points there at all and extrapolates.
            recorded["pre_account_value"][:, year] = account[0]
            recorded["pre_benefit_base"][:, year] = benefit_base[0]

        rider_charge = np.minimum(rider_rate * benefit_base, np.maximum(account, 0.0))
        account = account - rider_charge

        death_charge = 0.0
        if has_death:
            death_base = np.maximum(rollup_base, ratchet_base) if np.any(ratchets) else rollup_base
            death_charge = np.minimum(death_charge_rate * death_base, np.maximum(account, 0.0))
            account = account - death_charge

        in_deferral = year < deferral_years
        drawn = np.where(in_deferral, 0.0, utilisation * gawa * benefit_base)
        account_before_withdrawal = np.maximum(account, 0.0)
        from_account = np.minimum(drawn, account_before_withdrawal)
        claim = drawn - from_account
        account = account - from_account

        if has_death:
            # The benefit base is cut for withdrawals: dollar for dollar up to the free amount,
            # then by the proportion the remaining withdrawal takes out of the remaining
            # contract value. The proportional part is what makes a death benefit fall by more
            # than the cash withdrawn, which the prospectus warns about explicitly.
            free_amount = free_withdrawal * rollup_base
            dollar_part = np.minimum(from_account, free_amount)
            excess = from_account - dollar_part
            remaining = np.maximum(account_before_withdrawal - dollar_part, 1e-12)
            proportional = 1.0 - np.clip(excess / remaining, 0.0, 1.0)
            rollup_base = (rollup_base - dollar_part) * proportional
            if np.any(ratchets):
                ratchet_base = ratchet_base * (
                    1.0 - np.clip(from_account / np.maximum(account_before_withdrawal, 1e-12), 0.0, 1.0)
                )

        if np.any(step_up):
            # The Bonus Base moves only when the step-up itself lifts the GWB, which the
            # prospectus is explicit about: "with any step-up (if the GWB increases upon
            # step-up), the Bonus Base is set to the greater of the GWB after, and the Bonus
            # Base before, the step-up". Treating any rise in the GWB as a step-up instead
            # lets the bonus feed its own base, and a 6% simple bonus silently becomes a 6%
            # compound roll-up: over a ten-year deferral that is 179 against 160 on a benefit
            # base of 100.
            lifted = step_up & (account > benefit_base)
            ratcheted = np.where(lifted, account, benefit_base)
            # A step-up that raises the Bonus Base restarts the bonus clock, but only while the
            # owner is 80 or younger.
            raised = lifted & (ratcheted > bonus_base) & (attained_age + year <= BONUS_RESTART_MAX_AGE)
            bonus_base = np.where(raised, ratcheted, bonus_base)
            bonus_end = np.where(raised, year + 1 + BONUS_PERIOD_YEARS, bonus_end)
            benefit_base = ratcheted

        # The bonus is a percentage of the Bonus Base, not of the benefit base, and only in a
        # contract year with no withdrawal at all.
        earns_bonus = in_deferral & (year < bonus_end) & (account > 0.0)
        benefit_base = benefit_base + np.where(earns_bonus, bonus_rate * bonus_base, 0.0)

        # Any withdrawal at or before the adjustment date terminates the provision without
        # value, so it can only fire for a cohort still deferring when the date arrives.
        at_adjustment = adjustment_live & (year == adjustment_year)
        benefit_base = np.where(
            at_adjustment, np.maximum(benefit_base, adjustment_amount), benefit_base
        )
        adjustment_live = adjustment_live & ~at_adjustment & in_deferral

        if has_death:
            # Roll-up compounds until the anniversary immediately preceding the owner's 81st
            # birthday; the quarterly ratchet stops on the same date. Both are modelled
            # annually, so the ratchet is a floor on the quarterly one and the death benefit
            # here understates the highest-anniversary variant by the amount the contract
            # value peaked and fell back inside a year.
            accruing = (attained_age + year) <= DEATH_BENEFIT_FREEZE_AGE
            rollup_base = rollup_base * np.where(accruing, 1.0 + rollup_rate, 1.0)
            if np.any(ratchets):
                ratchet_base = np.where(
                    ratchets & accruing, np.maximum(ratchet_base, account), ratchet_base
                )
            death_base = np.maximum(rollup_base, ratchet_base) if np.any(ratchets) else rollup_base
            death_cost = np.maximum(death_base - np.maximum(account, 0.0), 0.0)

        weight = discount * survival[:, year][:, None] * in_force * alive
        claim_pv += weight * claim
        rider_fee_pv += weight * rider_charge
        fee_pv += weight * (rider_charge + base_charge)

        if has_death:
            # Deaths are weighted by the probability of dying during the year, not by survival
            # to the end of it, and the payment lands at the year end like everything else.
            death_weight = discount * deaths[:, year][:, None] * in_force * alive
            death_claim_pv += death_weight * death_cost
            death_fee_pv += weight * death_charge
            death_claims_by_year[:, year] = (death_weight * death_cost).mean(axis=1)
            mean_death_benefit[:, year] = death_base.mean(axis=1)

        if deficiency_pv is not None:
            # Benefits out less fees in, on the sign convention a reserve uses: positive is a
            # year the block costs the insurer money. Already discounted and already carrying
            # the probability the contract is there to pay, because both are in the weights the
            # present values use, so the accumulated deficiency is a cumulative sum over years
            # with nothing further to apply.
            outgo = weight * claim - weight * (rider_charge + base_charge)
            if has_death:
                outgo = outgo + death_weight * death_cost - weight * death_charge
            deficiency_pv[:, year] = outgo.sum(axis=0)
            # Undiscounted and in force, because the surrender value floor is a floor on the
            # reserve at the balance date rather than a cash flow.
            in_force_account[:, year] = (
                survival[:, year][:, None] * in_force * alive * np.maximum(account, 0.0)
            ).sum(axis=0)

        if recorded is not None:
            # State the proxy regresses on, and the two cash-flow legs it combines. Both legs
            # are discounted to time zero and carry the probability the contract is still there
            # to pay - mortality and lapse together - so a continuation value is the sum of the
            # years after t divided by the discount factor AND by that probability at t. Leaving
            # the probability out values a contract that might already have lapsed, which is a
            # different and much smaller number: at year twenty of a sixty-five-year-old's
            # contract the persistency factor is around a half, and the proxy came out low by
            # that much against the nested standard before the factor was recorded here.
            # Recording the legs separately rather than netted lets one fit serve any
            # attribution percentage.
            recorded["account_value"][:, year] = account[0]
            recorded["benefit_base"][:, year] = benefit_base[0]
            recorded["discount"][:, year] = discount[0]
            recorded["persistency"][:, year] = (
                survival[:, year][:, None] * in_force * alive
            )[0]
            # The rest of the contract's state, which the proxy does not regress on but the
            # nested check has to restart from. Three of these are path-dependent and none of
            # them can be reconstructed from the account value and the benefit base: the bonus
            # clock restarts on a step-up, and the death benefit base is cut by withdrawals on
            # its own schedule and drifts a long way from the living-benefit base. Rebuilding a
            # node with the death base set to the guaranteed withdrawal base instead of the
            # recorded one turned a 0.34 life annuity into a 0.96 one.
            recorded["bonus_base"][:, year] = bonus_base[0]
            recorded["bonus_end"][:, year] = bonus_end[0]
            recorded["adjustment_live"][:, year] = adjustment_live[0]
            if has_death:
                recorded["death_rollup_base"][:, year] = rollup_base[0]
                recorded["death_ratchet_base"][:, year] = ratchet_base[0]
            else:
                recorded["death_rollup_base"][:, year] = 0.0
                recorded["death_ratchet_base"][:, year] = 0.0
            recorded["variance"][:, year] = paths.variance[:, year]
            recorded["zero_10y"][:, year] = paths.zero_10y[:, year]
            recorded["short_rate"][:, year] = paths.short_rate[:, year]
            recorded["pv_claim"][:, year] = (weight * claim)[0] + (
                (death_weight * death_cost)[0] if has_death else 0.0
            )
            recorded["pv_fee"][:, year] = (
                weight * (rider_charge + base_charge + death_charge)
            )[0]

        claims_by_year[:, year] = (weight * claim).mean(axis=1)
        fees_by_year[:, year] = (weight * (rider_charge + base_charge + death_charge)).mean(axis=1)
        exhaustion[:, year] = (account <= 0.0).mean(axis=1)
        mean_account[:, year] = account.mean(axis=1)
        mean_base[:, year] = benefit_base.mean(axis=1)

        # Surrender can only happen while there is a contract value to surrender. With a
        # dynamic rate the decision also depends on how far the guarantee is in the money,
        # which is state the loop already has and a survival curve could never carry.
        period_lapse = (
            behaviour.dynamic_lapse(benefit_base, account, lapse_rate, lapse_beta, lapse_floor)
            if dynamic else lapse_rate
        )
        np.multiply(
            in_force, np.where(account > 0.0, 1.0 - period_lapse, 1.0), out=in_force
        )

    return GmwbProjection(
        pv_claims=claim_pv.mean(axis=1),
        pv_death_claims=death_claim_pv.mean(axis=1),
        pv_rider_fees=rider_fee_pv.mean(axis=1),
        pv_base_fees=(fee_pv - rider_fee_pv).mean(axis=1),
        pv_death_fees=death_fee_pv.mean(axis=1),
        claims_by_year=claims_by_year,
        death_claims_by_year=death_claims_by_year,
        fees_by_year=fees_by_year,
        exhaustion_prob=exhaustion,
        mean_account_value=mean_account,
        mean_benefit_base=mean_base,
        mean_death_benefit=mean_death_benefit,
        claim_paths=claim_pv + death_claim_pv,
        fee_paths=fee_pv + death_fee_pv,
        recorded=recorded,
        deficiency_pv=deficiency_pv,
        in_force_account=in_force_account,
    )
