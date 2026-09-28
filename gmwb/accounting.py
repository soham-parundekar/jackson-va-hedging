"""Why a working economic hedge still leaves reported earnings moving.

Jackson states the position plainly in Item 7A of the FY2025 10-K: "We do not
directly use hedging to offset the movement in our U.S. GAAP liabilities as market
conditions change from period to period, which has resulted, and may continue to
result, in U.S. GAAP net income volatility." The hedge targets an internally
modelled economic liability. The reported market risk benefit is a different number
measured on a different basis, and a hedge sized on the first will not neutralise
the second.

Two differences are large enough to measure with public data.

The reporting basis carries margins. Jackson's fair value uses "best estimate
assumptions plus risk margins", and the NAIC-adopted annuity table comes in a Basic
version and a Period version that differs only by the margins the Life Actuarial
Task Force set. Valuing the same contract on both gives two liabilities whose
sensitivities differ, so a delta computed on one leaves a residual against the other.

The reporting basis discounts at a rate that includes Jackson's own non-performance
risk, which the economic model does not. That matters twice. The level difference
moves with credit spreads, and under the market risk benefit rules that piece is
reported in other comprehensive income rather than in net income. But the spread
also changes the sensitivity of the reported liability, and that part does reach net
income: an equity hedge sized on the economic delta is the wrong size for a liability
discounted at a higher rate.

So the reported net income effect is the change in the reporting-basis liability with
the own-credit movement stripped out, hedged with instruments sized on the economic
basis. That is what ``reported_earnings`` builds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "mrb_reporting_basis",
    "mrb_with_own_credit",
    "own_credit_adjustment",
    "liability_pnl",
    "equity_hedge_pnl",
    "rate_hedge_pnl",
)


def reported_earnings(ledger: pd.DataFrame, fee_attribution: float) -> pd.DataFrame:
    """Split reported movement into a net income piece and an OCI piece.

    Expects a ledger from ``hedging.run_backtest(..., with_gaap=True)``. The hedge
    positions in that ledger were sized on the economic basis, which is the point of
    the exercise.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in ledger.columns]
    if missing:
        raise ValueError(f"ledger is missing {missing}; run the backtest with with_gaap=True")

    out = ledger.dropna(subset=["liability_pnl"]).copy()

    # Movement in the reporting-basis liability, split at the own-credit adjustment.
    out["delta_mrb_total"] = out["mrb_with_own_credit"].diff()
    out["oci_pnl"] = -out["own_credit_adjustment"].diff()
    out["delta_mrb_ex_credit"] = out["mrb_reporting_basis"].diff()

    attributed_fees = fee_attribution * (out["me_charge"] + out["rider_charge"])
    financing = out["mrb_reporting_basis"].shift(1) * out["short_rate"].shift(1) * out["dt_years"]
    out["reported_liability_pnl"] = (
        attributed_fees - out["claim"] - out["delta_mrb_ex_credit"] + financing
    )
    out["reported_net_income"] = (
        out["reported_liability_pnl"] + out["equity_hedge_pnl"] + out["rate_hedge_pnl"]
        - out["transaction_cost"]
    )
    out["reported_comprehensive_income"] = out["reported_net_income"] + out["oci_pnl"]
    return out.dropna(subset=["reported_net_income"])


def summarise(reported: pd.DataFrame) -> dict[str, float]:
    """Compare the volatility of economic and reported outcomes on the same hedge."""
    economic = reported["hedged_pnl"].to_numpy(dtype=float)
    unhedged = reported["liability_pnl"].to_numpy(dtype=float)
    net_income = reported["reported_net_income"].to_numpy(dtype=float)
    comprehensive = reported["reported_comprehensive_income"].to_numpy(dtype=float)
    oci = reported["oci_pnl"].to_numpy(dtype=float)

    var_unhedged = float(unhedged.var(ddof=1))
    return {
        "periods": int(reported.shape[0]),
        "unhedged_std": float(unhedged.std(ddof=1)),
        "economic_hedged_std": float(economic.std(ddof=1)),
        "reported_net_income_std": float(net_income.std(ddof=1)),
        "reported_comprehensive_std": float(comprehensive.std(ddof=1)),
        "oci_std": float(oci.std(ddof=1)),
        "economic_variance_ratio": float(economic.var(ddof=1) / var_unhedged),
        "reported_variance_ratio": float(net_income.var(ddof=1) / var_unhedged),
        "reported_over_economic_std": float(
            net_income.std(ddof=1) / economic.std(ddof=1)
        ),
        "mean_own_credit_adjustment": float(reported["own_credit_adjustment"].mean()),
        "max_own_credit_adjustment": float(reported["own_credit_adjustment"].abs().max()),
        "basis_gap_mean": float(
            (reported["mrb_reporting_basis"] - reported["rider_value"]).mean()
        ),
    }


def worst_periods(reported: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """The periods where the economic hedge worked and reported earnings still moved.

    Ranked by how far the reported net income outcome sat from the economic one, which
    is the gap a hedging programme cannot close by trading more.
    """
    frame = reported.copy()
    frame["economic_minus_reported"] = frame["hedged_pnl"] - frame["reported_net_income"]
    columns = [
        "price_return",
        "rate_move_bp",
        "own_credit_spread",
        "liability_pnl",
        "hedged_pnl",
        "reported_net_income",
        "oci_pnl",
        "economic_minus_reported",
    ]
    ranked = frame.reindex(
        frame["economic_minus_reported"].abs().sort_values(ascending=False).index
    )
    return ranked[columns].head(n)


def annualise(period_std: float, periods_per_year: float) -> float:
    return float(period_std * np.sqrt(periods_per_year))
