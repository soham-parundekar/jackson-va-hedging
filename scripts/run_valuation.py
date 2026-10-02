"""Value the representative rider, and measure how much the answer depends on each assumption.

The at-issue valuation, the fee attribution percentage calibration implies, the cash flows
behind the number, and a one-at-a-time robustness table. The robustness table is the part worth
reading: a single-policy valuation of a forty-five-year guarantee means little on its own, and
a great deal alongside how far it moves when the inputs move.

Three things in this valuation have no counterpart in a textbook put and are worth knowing
before reading the table.

*The attribution percentage.* Under ASU 2018-12 the market risk benefit is the present value of
guarantee payments less an attributed share of projected fees, and that share is fixed at
inception. It comes out near enough to one here, which is why the benefit starts at roughly zero
and why every later movement is a movement in a difference between two large numbers.

*The death benefit belongs in the same valuation.* The rider and the death benefit are charged
separately but they sit on the same account, and they move in opposite directions: a market fall
makes the living benefit dearer and the death benefit dearer too, while longer life makes the
living benefit dearer and the death benefit cheaper. Valuing one without the other would report
a hedge target that does not exist.

*Volatility, not rates, is the exposure that is not identified.* The projection runs to age 115
and the option chain this model is calibrated to stops at 3.2 years, so the long-run variance is
an extrapolation. The robustness table prices that extrapolation directly, including at the
level the issuer's own disclosure describes.

Usage:  python -m scripts.run_valuation
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from vahedge import paths
from vahedge.liability import cohorts, mortality
from vahedge.liability import terms as terms_module
from vahedge.market import state as market_state
from vahedge.market.simulate import SubAccountMix
from vahedge.valuation.engine import DEFAULT_PATHS, DEFAULT_SEED, MarketState, Valuer

ISSUE_AGE = 70
# Years from issue until the first withdrawal, and the most consequential single assumption in
# the project after the long-run volatility. The Core option pays a 6% simple bonus on the
# benefit base for each deferred year up to ten, and the rate sheet's guaranteed withdrawal
# percentage is set by the age at the first withdrawal, so deferring five years lifts the
# benefit base by 30% and the rate from 5.75% to 5.95% - together about a third more guaranteed
# income. Nothing in the disclosure identifies the book's deferral behaviour, so this is a stated
# assumption rather than a fitted one: five years is the midpoint of the bonus period, which is
# the only landmark the contract itself provides. The robustness table prices both ends, and the
# direction of the bias is plain - a contract that defers and then draws in full is an upper
# bound on the guarantee.
DEFERRAL_YEARS = 5
DEFERRAL_RANGE = (0, 10)
# Ages the projection is cut off at, to show that cutting it off does not matter.
TRUNCATION_CAPS = (105, 110, 120)
MAX_AGE = 115
PREMIUM = 100_000.0
BASE_CONTRACT_CHARGE = 0.0131
FUND_EXPENSE = 0.0095
# Disclosed range of the fund menu's expense ratios, FY2025 10-K. The base case sits inside it
# rather than at either end, so both ends are priced.
FUND_EXPENSE_RANGE = (0.0052, 0.0238)
# Realised ten-year volatility of the index plus the one point of risk margin the issuer's
# disclosure describes adding to its long-run level. The option chain rejects it - pinning the
# long-run variance here costs four times the volatility error beyond two years and pushes mean
# reversion to its bound - so it is priced as a variation rather than adopted as the calibration.
DISCLOSED_LONG_RUN_VOL = 0.1912
PATH_COUNTS = (2_500, 5_000, 10_000, 20_000, 40_000)


def build(n_paths: int = DEFAULT_PATHS, male_weight: float = 0.5, table: str = "basic",
          cache_size: int = 4):
    """The contract at issue, today's market, and a valuer holding both.

    ``cache_size`` is worth raising for a caller that cycles through a fixed handful of market
    states - the shock table visits the base state and four rate shifts for every moneyness on
    its grid, and a cache that holds fewer than five of them re-simulates all of them at every
    step rather than once.
    """
    calibration = market_state.load()
    state = MarketState.from_calibration(calibration)
    core = terms_module.load()[("flex_gmwb", "single", "core")]
    book = cohorts.single_contract(
        core, issue_age=ISSUE_AGE, base_contract_charge=BASE_CONTRACT_CHARGE,
        fund_expense=FUND_EXPENSE, premium=PREMIUM, deferral_years=DEFERRAL_YEARS,
        max_age=MAX_AGE,
    )
    valuer = Valuer(mortality.load(table), n_paths=n_paths, seed=DEFAULT_SEED,
                    male_weight=male_weight, cache_size=cache_size)
    return {"state": state, "terms": core, "book": book, "valuer": valuer,
            "calibration": calibration}


def base_valuation(setup):
    """The at-issue valuation and the attribution it calibrates to, computed once.

    Three of the four tables below start from the same number, and at twenty thousand paths over
    forty-five years that number costs a simulation and a projection. Keeping it on the setup
    means the robustness table is the only place that pays for a revaluation.
    """
    if "base" not in setup:
        valuer, book, state = setup["valuer"], setup["book"], setup["state"]
        attribution = valuer.calibrate_attribution(book, state)
        setup["base"] = (attribution, valuer.value(book, state, attribution=attribution))
    return setup["base"]


def headline(setup) -> pd.DataFrame:
    valuer, book, state = setup["valuer"], setup["book"], setup["state"]
    _, result = base_valuation(setup)
    projection = result.projection
    n_years = int(book.projection_years.max())
    survival, _ = valuer.mortality_for(book, state.valuation_year, n_years)

    rows = [
        ("premium and initial benefit base", PREMIUM),
        ("PV of living-benefit payments", result.pv_claims),
        ("PV of death benefit above the account", result.pv_death_claims),
        ("PV of the explicit GMWB charge", float(projection.pv_rider_fees[0])),
        ("PV of the insurer's share of the account drag", float(projection.pv_base_fees[0])),
        ("PV of the explicit death-benefit charge", float(projection.pv_death_fees[0])),
        ("PV of total attributable fees", result.pv_fees),
        ("fee attribution percentage", result.attribution),
        ("market risk benefit at issue", result.market_risk_benefit),
        ("Monte Carlo standard error", result.std_error),
        ("gross guarantee value, % of premium", 100 * result.total_claims / PREMIUM),
        ("probability of exhaustion by year 20", float(projection.exhaustion_prob[0, 19])),
        ("expected future lifetime, years", float(survival[0].sum())),
    ]
    return pd.DataFrame(rows, columns=["quantity", "value"])


def cash_flows(setup) -> pd.DataFrame:
    """The profile behind the number, year by year, expected and discounted.

    The discount factor is a mean rather than a given: rates are stochastic here, so what the
    column shows is the average realised factor across paths, which is not the same as the
    factor implied by today's curve once the curve's convexity is in the picture.
    """
    valuer, book, state = setup["valuer"], setup["book"], setup["state"]
    projection = base_valuation(setup)[1].projection
    n_years = int(book.projection_years.max())
    survival, _ = valuer.mortality_for(book, state.valuation_year, n_years)
    simulated = valuer.paths_for(state, n_years)
    return pd.DataFrame({
        "policy_year": np.arange(1, n_years + 1),
        "survival": survival[0],
        "mean_discount_factor": simulated.discount.mean(axis=0),
        "expected_claim": projection.claims_by_year[0],
        "expected_death_claim": projection.death_claims_by_year[0],
        "expected_fee": projection.fees_by_year[0],
        "prob_exhausted": projection.exhaustion_prob[0],
        "mean_account_value": projection.mean_account_value[0],
        "mean_benefit_base": projection.mean_benefit_base[0],
        "mean_death_benefit": projection.mean_death_benefit[0],
    })


def robustness(setup) -> pd.DataFrame:
    """One-at-a-time variations, every one repriced on the same draws.

    The attribution percentage is held at the base-case calibration throughout, which is what
    the accounting does: it is fixed at inception and does not move when a later assumption
    does, so a variation has to be allowed to move the reported value rather than being
    absorbed by a recalibrated percentage.
    """
    valuer, book, state, terms = (setup["valuer"], setup["book"], setup["state"], setup["terms"])
    attribution = base_valuation(setup)[0]

    def contract(**overrides):
        settings = dict(issue_age=ISSUE_AGE, base_contract_charge=BASE_CONTRACT_CHARGE,
                        fund_expense=FUND_EXPENSE, premium=PREMIUM,
                        deferral_years=DEFERRAL_YEARS, max_age=MAX_AGE)
        rider = overrides.pop("terms", terms)
        settings.update(overrides)
        return cohorts.single_contract(rider, **settings)

    def price(label, note="", *, which=None, where=None, valuing=None, path_years=None,
              design=False):
        """One variation, reported on whichever comparison its kind deserves.

        Two kinds of variation, and reading them the same way is a category error that this
        table made until the deferral rows exposed it.

        A *market or assumption* variation - volatility, rates, the mortality basis - happens
        after inception, when the attribution percentage is already fixed. The thing to report is
        how far the benefit moves with that percentage held, which is what the accounting does.

        A *design* variation - the deferral, the issue age, the fund's expense ratio, the
        step-up - is a different contract. At issue it would have been given its own percentage,
        so its benefit would also start at zero and the fixed-percentage figure measures the
        mismatch between one contract's fees and another contract's percentage rather than
        anything about the design. What distinguishes a design is the percentage it needs:
        ``implied_attribution`` near one is a contract whose fees barely cover its guarantee.
        """
        priced = (valuing or valuer).value(which or book, where or state,
                                           attribution=attribution, path_years=path_years)
        return {"variation": label, "note": note,
                "comparison": "own attribution" if design else "fixed attribution",
                "market_risk_benefit": priced.market_risk_benefit,
                "pv_claims": priced.total_claims,
                "implied_attribution": min(
                    1.0, priced.total_claims / priced.pv_fees
                ) if priced.pv_fees > 0 else np.nan,
                "gross_pct_of_premium": 100 * priced.total_claims / PREMIUM,
                "std_error": priced.std_error}

    rows = [price("base case")]

    for bump, label in ((0.02, "+2 points"), (-0.02, "-2 points")):
        long_run = np.sqrt(state.heston.theta) + bump
        rows.append(price(
            f"long-run volatility {label}",
            f"theta at {long_run:.2%} against the calibrated {np.sqrt(state.heston.theta):.2%}",
            where=replace(state, heston=replace(state.heston, theta=long_run ** 2)),
        ))
        current = np.sqrt(state.heston.v0) + bump
        rows.append(price(
            f"current volatility {label}",
            f"v0 at {current:.2%}, which mean reverts out within a year",
            where=replace(state, heston=replace(state.heston, v0=max(current, 1e-4) ** 2)),
        ))

    rows.append(price(
        f"long-run volatility at the disclosed {DISCLOSED_LONG_RUN_VOL:.2%}",
        "realised ten-year level plus the issuer's one-point margin; the chain rejects it",
        where=replace(state, heston=replace(state.heston,
                                            theta=DISCLOSED_LONG_RUN_VOL ** 2)),
    ))

    for shift, label in ((100, "+100bp"), (-100, "-100bp")):
        rows.append(price(f"zero curve {label}", "parallel shift of the fitted level",
                          where=state.with_shocks(rate_shock_bp=shift)))

    # The two sex rows come out exactly symmetric around the base, which is right rather than
    # suspicious: claims and fees are both survival-weighted sums over the same paths, so the
    # valuation is linear in the survival curve and a 50/50 book of lives is the midpoint of the
    # two single-sex books to the cent.
    rows.append(price(
        "mortality on the Period table, with margins", "the reporting basis",
        valuing=Valuer(mortality.load("period"), n_paths=valuer.n_paths, seed=valuer.seed),
    ))
    for weight, label in ((1.0, "all male"), (0.0, "all female")):
        rows.append(price(
            f"mortality, {label}", "the disclosure gives no gender split",
            valuing=Valuer(mortality.load("basic"), n_paths=valuer.n_paths, seed=valuer.seed,
                           male_weight=weight),
        ))

    for expense, end in zip(FUND_EXPENSE_RANGE, ("minimum", "maximum")):
        rows.append(price(f"fund expenses at the disclosed {end}, {expense:.2%}",
                          "a higher drag starves the account and feeds the guarantee",
                          which=contract(fund_expense=expense), design=True))

    rows.append(price("no annual step-up", "the benefit base never ratchets",
                      which=contract(terms=replace(terms, step_up="none")), design=True))

    rows.append(price(
        "sub-account all equity", "against the FY2025 fund mix's 0.83 equity exposure",
        where=replace(state, mix=SubAccountMix(equity=1.0, bond=0.0, balanced=0.0,
                                              money_market=0.0)),
        design=True,
    ))

    for deferral in DEFERRAL_RANGE:
        # Deferral moves two things at once and the label says so, because a reader who sees only
        # the benefit base move will think the bonus is the whole of it.
        varied = contract(deferral_years=deferral)
        rows.append(price(
            f"first withdrawal deferred {deferral} years",
            f"GAWA {float(varied.gawa_pct[0]):.2%} at age {ISSUE_AGE + deferral}, "
            f"bonus {6 * min(deferral, 10)}% on the benefit base",
            which=varied, design=True,
        ))

    for age in (65, 75):
        # The withdrawal rate comes from the rate sheet's own age bands rather than being
        # supplied here, so the variation is the issue age and the GAWA follows it.
        aged = contract(issue_age=age, deferral_years=DEFERRAL_YEARS)
        rows.append(price(
            f"issue age {age}, GAWA {float(aged.gawa_pct[0]):.2%}", "rate sheet band",
            which=aged, design=True,
            valuing=Valuer(mortality.load("basic"), n_paths=valuer.n_paths, seed=valuer.seed),
        ))

    # Truncation, on the draws the longest horizon produced, so the three caps differ only in
    # where the projection stops. The base case is repriced on those same draws as the comparison
    # point, because its own row above was simulated to 45 years and is a different world.
    longest = max(TRUNCATION_CAPS) - ISSUE_AGE
    common = price(f"projection to age {MAX_AGE}", "truncation reference, common draws",
                   path_years=longest)
    rows.append(common)
    for cap in TRUNCATION_CAPS:
        rows.append(price(f"projection to age {cap}", "truncation test, common draws",
                          which=contract(max_age=cap), path_years=longest))

    frame = pd.DataFrame(rows)
    base = frame.iloc[0]
    frame["mrb_change"] = frame["market_risk_benefit"] - float(base["market_risk_benefit"])
    frame["mrb_pct_of_premium"] = 100 * frame["market_risk_benefit"] / PREMIUM
    frame["attribution_change"] = (frame["implied_attribution"]
                                   - float(base["implied_attribution"]))
    return frame[["variation", "comparison", "market_risk_benefit", "mrb_change",
                  "mrb_pct_of_premium", "implied_attribution", "attribution_change",
                  "gross_pct_of_premium", "pv_claims", "std_error", "note"]]


def path_count_convergence() -> pd.DataFrame:
    """Value and standard error against the number of paths.

    Each row is an independent simulation at its own path count rather than a prefix of the
    largest, so the standard errors are what a run at that size would actually report.
    """
    rows = []
    for n_paths in PATH_COUNTS:
        setup = build(n_paths=n_paths)
        priced = base_valuation(setup)[1]
        rows.append({"paths": n_paths, "market_risk_benefit": priced.market_risk_benefit,
                     "pv_claims": priced.total_claims, "std_error": priced.std_error,
                     "attribution": priced.attribution})
    return pd.DataFrame(rows)


def main() -> None:
    paths.ensure_output_dirs()
    setup = build()
    state = setup["state"]
    print(f"Representative Flex GMWB (Single), Core option, issued at {ISSUE_AGE} on "
          f"{setup['calibration'].as_of.date()}")
    print(f"  ten-year zero {state.curve.zero(10.0):.2%}, current volatility "
          f"{np.sqrt(state.heston.v0):.2%}, long-run {np.sqrt(state.heston.theta):.2%}, "
          f"equity exposure of the sub-account {state.mix.equity_weight:.4f}")

    head = headline(setup)
    head.to_csv(paths.TABLES / "valuation_headline.csv", index=False, float_format="%.4f")
    print("\nAt issue")
    for _, row in head.iterrows():
        print(f"  {row['quantity']:<46s} {row['value']:>14,.4f}")

    flows = cash_flows(setup)
    flows.to_csv(paths.TABLES / "cash_flow_profile.csv", index=False, float_format="%.6f")
    peak = int(flows["expected_claim"].idxmax())
    print(f"\nCash flows: the living benefit peaks in policy year "
          f"{int(flows.loc[peak, 'policy_year'])} at "
          f"{flows.loc[peak, 'expected_claim']:,.0f}, and "
          f"{100 * flows['prob_exhausted'].iloc[19]:.1f}% of paths have no contract value left "
          f"by year 20")

    table = robustness(setup)
    table.to_csv(paths.TABLES / "valuation_robustness.csv", index=False, float_format="%.2f")
    print("\nRobustness, one variation at a time, same draws")
    print("  A market or assumption variation is read as the movement in the benefit with the")
    print("  attribution percentage held, which is what the accounting does after inception. A")
    print("  design variation is a different contract, so it is read as the percentage its own")
    print("  fees would have had to attribute at issue.")
    print("\n  market and assumption                                  MRB      change  % premium")
    held = table[table["comparison"] == "fixed attribution"]
    for _, row in held.iterrows():
        print(f"  {row['variation']:<48s} {row['market_risk_benefit']:>10,.0f} "
              f"{row['mrb_change']:>+10,.0f} {row['mrb_pct_of_premium']:>9.2f}")
    ranked = held.iloc[1:].reindex(held.iloc[1:]["mrb_change"].abs().sort_values().index)
    print(f"    most consequential: {ranked.iloc[-1]['variation']} at "
          f"{ranked.iloc[-1]['mrb_change']:+,.0f}; least: {ranked.iloc[0]['variation']} at "
          f"{ranked.iloc[0]['mrb_change']:+,.0f}")

    design = table[table["comparison"] == "own attribution"]
    print("\n  contract design                                  attribution    change  gross %")
    base_alpha = float(table.iloc[0]["implied_attribution"])
    print(f"  {'base case':<48s} {base_alpha:>11.4f} {0.0:>+9.4f} "
          f"{table.iloc[0]['gross_pct_of_premium']:>8.2f}")
    for _, row in design.iterrows():
        print(f"  {row['variation']:<48s} {row['implied_attribution']:>11.4f} "
              f"{row['attribution_change']:>+9.4f} {row['gross_pct_of_premium']:>8.2f}")
    binding = design[design["implied_attribution"] >= 0.9999]
    if not binding.empty:
        print(f"    {len(binding)} design(s) need every attributable fee and still do not cover "
              f"the guarantee, so the benefit starts as a liability rather than at zero: "
              f"{', '.join(binding['variation'])}")

    convergence = path_count_convergence()
    convergence.to_csv(paths.TABLES / "monte_carlo_convergence.csv", index=False,
                       float_format="%.2f")
    print("\nConvergence")
    for _, row in convergence.iterrows():
        print(f"  {int(row['paths']):>7,d} paths  MRB {row['market_risk_benefit']:>10,.0f}  "
              f"standard error {row['std_error']:>8,.2f}  attribution "
              f"{row['attribution']:.4f}")
    print(f"\nwrote four tables to {paths.TABLES}")


if __name__ == "__main__":
    main()
