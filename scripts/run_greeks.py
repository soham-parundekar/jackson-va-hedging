"""Greeks for the representative rider, and how they move with moneyness.

Sensitivities are reported both in dollars and as a share of account value. The share
is the number that travels: it is the only form in which a single $100,000 policy and a
$236bn book can be compared.

Usage:  python -m scripts.run_greeks
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from gmwb import figures, market, mortality, session
from gmwb.engine import projection_years, value_rider
from gmwb.hedging import SubAccountMix
from gmwb.sensitivities import compute_greeks, moneyness_profile


def greeks_table(s: session.Session) -> pd.DataFrame:
    gcfg = s.cfg["greeks"]
    rows = []
    for label, basis, note in (
        ("economic, Basic table", s.mortality, "hedging basis"),
        (
            "reporting, Period table",
            mortality.load(s.contract.issue_age, s.valuation_date.year,
                           float(s.cfg["contract"]["sex_mix"]["male"]), table="period"),
            "margins included",
        ),
    ):
        g = compute_greeks(
            s.contract, s.state.curve_builder, s.state.vol, basis, s.normals,
            s.valuation_state(),
            equity_bump=float(gcfg["equity_bump_pct"]),
            rate_bump_bp=float(gcfg["rate_bump_bp"]),
            vol_bump=float(gcfg["vol_bump"]),
        )
        rows.append(
            {
                "basis": label,
                "value": g.base_value,
                "equity_exposure": g.equity_exposure,
                "equity_exposure_pct_av": 100 * g.equity_exposure_pct_of_av,
                "equity_gamma": g.equity_gamma,
                "rho_per_bp": g.rho_per_bp,
                "rho_per_100bp_pct_av": 100 * g.rho_per_bp * 100 / g.account_value,
                "vega_tradeable": g.vega_per_point,
                "vega_long_run": g.vega_long_run,
                "std_error": g.std_error,
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def hedge_sizing(s: session.Session) -> pd.DataFrame:
    """What the Greeks imply a desk would actually hold against one policy."""
    gcfg = s.cfg["greeks"]
    g = compute_greeks(
        s.contract, s.state.curve_builder, s.state.vol, s.mortality, s.normals,
        s.valuation_state(),
        equity_bump=float(gcfg["equity_bump_pct"]),
        rate_bump_bp=float(gcfg["rate_bump_bp"]),
        vol_bump=float(gcfg["vol_bump"]),
    )
    from gmwb.hedging import swap_annuity

    curve = s.state.curve_builder.build()
    tenor = float(s.cfg["hedge"]["swap_tenor_years"])
    annuity = swap_annuity(curve, tenor, int(s.cfg["curve"]["coupon_frequency"]))
    dv01 = -g.rho_per_bp
    rows = [
        ("short equity index futures, $ notional", g.equity_exposure),
        ("as a multiple of account value", g.equity_exposure / g.account_value),
        (f"receive-fixed {tenor:.0f}-year swap, target DV01 $/bp", dv01),
        (f"implied swap notional at an annuity of {annuity:.4f}", dv01 / (annuity * 1e-4)),
        ("long volatility, $ per vol point", g.vega_per_point),
        ("volatility exposure beyond listed maturities, $ per point", g.vega_long_run),
    ]
    return pd.DataFrame(rows, columns=["position", "size"])


def step_up_proximity(s: session.Session) -> pd.DataFrame:
    """Greeks against time to the next anniversary, for a contract above its benefit base.

    Worth reporting on its own. The Core option steps the benefit base up to the contract
    value on the anniversary, so when the contract value sits above the benefit base the
    index level on that one date fixes the guaranteed income for the rest of the contract's
    life. Approaching it, the insurer stops being short equity and becomes long: a higher
    index means a permanently larger guarantee to fund. Equity exposure changes sign, and a
    hedge that follows it turns from short index to long.

    A book with anniversaries spread across the calendar averages this away. A single policy
    does not, which is one reason single-policy hedge statistics overstate the turnover a
    real programme sees.
    """
    male_weight = float(s.cfg["contract"]["sex_mix"]["male"])
    mix = SubAccountMix.from_config(s.cfg)
    bump = float(s.cfg["greeks"]["equity_bump_pct"])
    span = np.log((1.0 + bump) / (1.0 - bump))
    rows = []

    # Two market states, because the sign turns out to depend on the rate environment. The
    # attribution is held at 1.00 throughout, which is what a contract written in 2016
    # calibrates to and what the hedging backtest carries.
    for label, as_of in (("low rates, Jul 2021", "2021-07-09"),
                         ("current, Dec 2025", s.cfg["valuation_date"])):
        market_state = market.state_at(s.panel, as_of, s.cfg, s.long_run_vol)
        curve = market_state.curve_builder.build()
        for attained in (70, 74):
            contract = replace(s.contract, issue_age=attained,
                               fund_equity_beta=mix.effective_equity_beta)
            basis = mortality.load(attained, pd.Timestamp(as_of).year, male_weight)
            n_years = projection_years(contract, int(s.cfg["simulation"]["max_age"]))
            normals = s.normals[:, :n_years]
            for ratio, tau in ((1.0, 1.0), (1.0, 0.3), (1.25, 1.0), (1.25, 0.3)):
                state = {
                    "account_value": s.contract.premium,
                    "benefit_base": s.contract.premium / ratio,
                    "first_step_years": tau,
                    "fee_attribution": 1.0,
                    "max_age": int(s.cfg["simulation"]["max_age"]),
                }
                base, up, down = (
                    value_rider(contract, curve, market_state.vol, basis, normals,
                                equity_shock=shock, **state)
                    for shock in (0.0, bump, -bump)
                )
                rows.append(
                    {
                        "market": label,
                        "ten_year_zero_pct": 100 * float(curve.zero(10)),
                        "attained_age": attained,
                        "account_value_over_benefit_base": ratio,
                        "years_to_anniversary": tau,
                        "value": base.net_value,
                        "d_claims_dlnS": (up.pv_claims - down.pv_claims) / span,
                        "d_fees_dlnS": (up.pv_total_fees - down.pv_total_fees) / span,
                        "equity_exposure": (up.net_value - down.net_value) / span,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    s = session.start()
    print(f"Greeks at {s.valuation_date.date()} on a ${s.contract.premium:,.0f} policy, "
          f"attribution {s.fee_attribution:.4f}")

    table = greeks_table(s)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    session.write_table(table, "greeks", "%.4f")

    sizing = hedge_sizing(s)
    print("\nHedge the Greeks imply")
    print(sizing.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    session.write_table(sizing, "hedge_sizing", "%.4f")

    print("\nMoneyness profile")
    rows = moneyness_profile(
        s.contract, s.state.curve_builder, s.state.vol, s.mortality, s.normals,
        s.valuation_state(),
        ratios=(0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6),
    )
    frame = pd.DataFrame(rows)
    for column in ("base_value", "equity_up_10pct", "equity_down_10pct",
                   "rates_up_100bp", "rates_down_100bp", "rates_up_50bp", "rates_down_50bp"):
        frame[f"{column}_pct_av"] = 100 * frame[column] / frame["account_value"]
    display = frame[["gwb_over_av", "base_value", "base_value_pct_av",
                     "equity_up_10pct_pct_av", "equity_down_10pct_pct_av",
                     "rates_up_100bp_pct_av", "rates_down_100bp_pct_av"]]
    print(display.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    session.write_table(frame, "moneyness_profile", "%.4f")

    print("\nApproaching a step-up date")
    proximity = step_up_proximity(s)
    print(proximity.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    flips = proximity.query("equity_exposure > 0")
    if flips.empty:
        print("  equity exposure stays short at every state tested")
    else:
        print(f"  equity exposure turns long in {len(flips)} of {len(proximity)} states, "
              f"every one of them with the contract value above the benefit base")
        print("  in those states both legs respond positively to the index and the net is "
              "their difference, so the sign is a small residual of two large numbers")
    at_the_money = proximity.query("account_value_over_benefit_base == 1.0")
    assert (at_the_money["equity_exposure"] < 0).all(), \
        "equity exposure must stay short while the guarantee is at or in the money"
    session.write_table(proximity, "step_up_proximity", "%.4f")

    print("\nFigure")
    print("  " + figures.moneyness_curve(rows))


if __name__ == "__main__":
    main()
