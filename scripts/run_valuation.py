"""Value the representative rider and test how much the answer depends on assumptions.

Reports the at-issue valuation, the fee attribution percentage that calibration
implies, the cash-flow profile behind the number, and a set of one-at-a-time
robustness runs. The robustness table is the useful part: a single-policy valuation
of a forty-year guarantee is only worth reading alongside how far it moves when the
inputs move.

Usage:  python -m scripts.run_valuation
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from gmwb import figures, market, mortality, session
from gmwb.engine import calibrate_attribution, make_normals, projection_years, value_rider
from gmwb.volatility import VolTermStructure


def headline(s: session.Session) -> pd.DataFrame:
    v = s.at_issue
    alpha = s.fee_attribution
    net = value_rider(
        s.contract, s.state.curve_builder.build(), s.state.vol, s.mortality, s.normals,
        fee_attribution=alpha, max_age=int(s.cfg["simulation"]["max_age"]),
    )
    rows = [
        ("premium and initial benefit base", s.contract.premium),
        ("PV of guarantee payments", v.pv_claims),
        ("PV of the explicit GMWB charge", v.pv_rider_fees),
        ("PV of the base contract charge", v.pv_me_fees),
        ("PV of total attributable fees", v.pv_total_fees),
        ("fee attribution percentage", alpha),
        ("market risk benefit at issue", net.net_value),
        ("Monte Carlo standard error", net.std_error),
        ("gross guarantee value, % of premium", 100 * v.gross_pct_of_av),
        ("probability of exhaustion by year 20", v.exhaustion_prob[19]),
        ("expected future lifetime, years", float(v.survival.sum())),
    ]
    return pd.DataFrame(rows, columns=["quantity", "value"])


def cash_flows(s: session.Session) -> pd.DataFrame:
    v = s.at_issue
    return pd.DataFrame(
        {
            "policy_year": v.times,
            "survival": v.survival,
            "discount_factor": v.discounts,
            "expected_claim": v.claims_by_year,
            "expected_rider_charge": v.rider_fees_by_year,
            "expected_base_charge": v.me_fees_by_year,
            "prob_exhausted": v.exhaustion_prob,
            "mean_account_value": v.mean_account_value,
            "mean_benefit_base": v.mean_benefit_base,
        }
    )


def robustness(s: session.Session) -> pd.DataFrame:
    """One-at-a-time variations, each repriced with the same random draws.

    The attribution percentage is held at the base-case calibration throughout. That
    is what the accounting does: the percentage is fixed at inception and does not move
    when assumptions later change, so a variation has to be allowed to move the
    reported value.
    """
    cfg = s.cfg
    max_age = int(cfg["simulation"]["max_age"])
    base_curve = s.state.curve_builder.build()
    alpha = s.fee_attribution
    common = dict(fee_attribution=alpha, max_age=max_age)

    def price(contract=None, curve=None, vol=None, basis=None, normals=None, **kwargs):
        return value_rider(
            contract or s.contract,
            curve or base_curve,
            vol or s.state.vol,
            basis or s.mortality,
            normals if normals is not None else s.normals,
            **{**common, **kwargs},
        )

    base = price()
    rows = [("base case", base.net_value, base.pv_claims, "")]

    for bump, label in ((0.02, "long-run volatility +2 points"),
                        (-0.02, "long-run volatility -2 points")):
        vol = replace(s.state.vol, long_run_level=s.state.vol.long_run_level + bump)
        result = price(vol=vol)
        rows.append((label, result.net_value, result.pv_claims, ""))

    for bump, label in ((0.02, "short-dated implied volatility +2 points"),
                        (-0.02, "short-dated implied volatility -2 points")):
        result = price(vol=s.state.vol.shift_front(bump))
        rows.append((label, result.net_value, result.pv_claims, ""))

    for shift, label in ((100, "par curve +100bp"), (-100, "par curve -100bp")):
        result = price(curve=s.state.curve_builder.build(shift_bp=shift))
        rows.append((label, result.net_value, result.pv_claims, ""))

    period = mortality.load(s.contract.issue_age, s.valuation_date.year,
                            float(cfg["contract"]["sex_mix"]["male"]), table="period")
    result = price(basis=period)
    rows.append(("mortality on the Period table, with margins", result.net_value,
                 result.pv_claims, "the reporting basis"))

    for weight, label in ((1.0, "all male"), (0.0, "all female")):
        basis = mortality.load(s.contract.issue_age, s.valuation_date.year, weight)
        result = price(basis=basis)
        rows.append((f"mortality, {label}", result.net_value, result.pv_claims, ""))

    for expense, label in ((0.0052, "fund expenses at the disclosed minimum, 0.52%"),
                           (0.0238, "fund expenses at the disclosed maximum, 2.38%")):
        result = price(contract=replace(s.contract, fund_expense=expense))
        rows.append((label, result.net_value, result.pv_claims, ""))

    result = price(contract=replace(s.contract, annual_step_up=False))
    rows.append(("no annual step-up", result.net_value, result.pv_claims, ""))

    result = price(contract=replace(s.contract, fund_equity_beta=0.8334))
    rows.append(("sub-account at the book's equity exposure, 0.83", result.net_value,
                 result.pv_claims, "FY2025 fund mix"))

    for age, label in ((65, "issue age 65"), (75, "issue age 75")):
        gawa = {65: 0.0555, 75: 0.0595}[age]   # the rate sheet's band for that age
        contract = replace(s.contract, issue_age=age, gawa_pct=gawa)
        basis = mortality.load(age, s.valuation_date.year,
                               float(cfg["contract"]["sex_mix"]["male"]))
        n_years = projection_years(contract, max_age)
        normals = s.normals[:, :n_years] if n_years <= s.normals.shape[1] else make_normals(
            s.normals.shape[0], n_years, int(cfg["simulation"]["seed"])
        )
        result = price(contract=contract, basis=basis, normals=normals)
        rows.append((f"{label}, GAWA {100 * gawa:.2f}%", result.net_value, result.pv_claims,
                     "rate sheet band"))

    for cap, label in ((110, "projection capped at age 110"), (120, "projection to age 120")):
        contract = s.contract
        n_years = projection_years(contract, cap)
        normals = s.normals[:, :n_years] if n_years <= s.normals.shape[1] else make_normals(
            s.normals.shape[0], n_years, int(cfg["simulation"]["seed"])
        )
        result = value_rider(contract, base_curve, s.state.vol, s.mortality, normals,
                             fee_attribution=alpha, max_age=cap)
        rows.append((label, result.net_value, result.pv_claims, "truncation test"))

    frame = pd.DataFrame(rows, columns=["variation", "market_risk_benefit", "pv_claims", "note"])
    frame["mrb_change"] = frame["market_risk_benefit"] - base.net_value
    frame["mrb_pct_of_premium"] = 100 * frame["market_risk_benefit"] / s.contract.premium
    return frame


def path_count_convergence(s: session.Session) -> pd.DataFrame:
    """Value and standard error against the number of paths."""
    rows = []
    max_age = int(s.cfg["simulation"]["max_age"])
    for n in (5000, 20000, 50000, 100000, s.normals.shape[0]):
        if n > s.normals.shape[0]:
            continue
        normals = make_normals(n, s.n_years, int(s.cfg["simulation"]["seed"]))
        result = value_rider(s.contract, s.state.curve_builder.build(), s.state.vol,
                             s.mortality, normals, fee_attribution=s.fee_attribution,
                             max_age=max_age)
        rows.append({"paths": n, "market_risk_benefit": result.net_value,
                     "pv_claims": result.pv_claims, "std_error": result.std_error})
    return pd.DataFrame(rows)


def main() -> None:
    s = session.start()
    print(f"Valuation date {s.valuation_date.date()}, "
          f"{s.normals.shape[0]:,} paths, {s.n_years} projection years")
    print(f"S&P 500 {s.state.spot:,.2f}   3-month implied {100 * s.state.implied_vol_3m:.2f}%   "
          f"long-run {100 * s.long_run_vol:.2f}%   10-year zero "
          f"{100 * s.state.curve_builder.build().zero(10):.3f}%")

    print("\nAt issue")
    head = headline(s)
    print(head.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    session.write_table(head, "valuation_headline")

    flows = cash_flows(s)
    session.write_table(flows, "cash_flow_profile")
    print("\nFirst ten policy years")
    print(flows.head(10).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    print("\nRobustness")
    rob = robustness(s)
    print(rob.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    session.write_table(rob, "valuation_robustness", "%.2f")

    print("\nMonte Carlo convergence")
    conv = path_count_convergence(s)
    print(conv.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    session.write_table(conv, "monte_carlo_convergence", "%.2f")

    print("\nFigures")
    print("  " + figures.cash_flow_profile(s.at_issue))
    observed = {s.cfg["volatility"]["short_tenor_years"]: s.state.vix / 100.0,
                s.cfg["volatility"]["mid_tenor_years"]: s.state.implied_vol_3m}
    print("  " + figures.volatility_term_structure(s.state.vol, observed))


if __name__ == "__main__":
    main()
