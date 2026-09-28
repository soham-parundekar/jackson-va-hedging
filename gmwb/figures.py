"""Figures for the report.

Titles say what the figure shows rather than announcing that an analysis happened,
and every axis carries units. Everything renders through Matplotlib's Agg backend so
the scripts run without a display.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import paths

LINE = "#1f4e79"
ACCENT = "#c0504d"
MUTED = "#7f7f7f"
FILL = "#dce6f1"


def _save(fig, name: str) -> str:
    paths.ensure_output_dirs()
    target = paths.FIGURES / f"{name}.png"
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(target)


def cash_flow_profile(valuation, name: str = "cash_flow_profile") -> str:
    """Expected claims and fee income by policy year, with the exhaustion curve."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                      gridspec_kw={"height_ratios": [2, 1]})
    years = valuation.times
    fees = valuation.rider_fees_by_year + valuation.me_fees_by_year

    top.bar(years, fees, color=FILL, edgecolor=LINE, linewidth=0.5,
            label="fee income collected")
    top.bar(years, -valuation.claims_by_year, color=ACCENT, alpha=0.75,
            label="guarantee payments made")
    top.axhline(0, color="black", linewidth=0.8)
    top.set_ylabel("expected cash flow, $ per policy year")
    top.set_title(
        "Fee income arrives first and the guarantee pays later: "
        f"expected cash flows on a ${valuation.account_value:,.0f} policy"
    )
    top.legend(frameon=False)

    bottom.plot(years, valuation.exhaustion_prob, color=LINE, linewidth=1.8)
    bottom.fill_between(years, 0, valuation.exhaustion_prob, color=FILL)
    bottom.plot(years, valuation.survival, color=MUTED, linewidth=1.4, linestyle="--",
                label="still alive")
    bottom.set_ylim(0, 1)
    bottom.set_xlabel("policy year")
    bottom.set_ylabel("probability")
    bottom.set_title("Contract value exhausted (solid) against survival (dashed)")
    bottom.legend(frameon=False, loc="center right")
    return _save(fig, name)


def volatility_term_structure(vol, observed=None,
                              name: str = "volatility_term_structure") -> str:
    """Forward and spot volatility, with the observed implied points the curve was fitted to."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    t = np.linspace(0.02, 30, 800)
    ax.plot(t, 100 * vol.forward_vol(t), color=LINE, linewidth=1.8, label="forward volatility")
    ax.plot(t, 100 * vol.spot_vol(t), color=ACCENT, linewidth=1.6, linestyle="--",
            label="spot volatility to maturity")
    ax.axhline(100 * vol.long_run_level, color=MUTED, linewidth=0.9, linestyle=":")
    ax.annotate(f"long run {100 * vol.long_run_level:.1f}%", (22, 100 * vol.long_run_level + 0.3),
                fontsize=9, color=MUTED)
    if observed:
        ax.scatter(list(observed), [100 * v for v in observed.values()], color=LINE, zorder=5,
                   s=36, label="observed implied indices")
    half_life = np.log(2) / vol.decay
    ax.set_xlabel("maturity, years")
    ax.set_ylabel("volatility, %")
    ax.set_title(
        f"Front level {100 * vol.front_level:.1f}%, mean reverting to "
        f"{100 * vol.long_run_level:.1f}% with a {half_life:.1f}-year half life"
    )
    ax.legend(frameon=False)
    return _save(fig, name)


def moneyness_curve(rows, name: str = "moneyness_curve") -> str:
    """Value and the disclosed shock responses across benefit-base-to-account-value."""
    frame = pd.DataFrame(rows)
    ratios = frame["gwb_over_av"]
    av = frame["account_value"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
    left.plot(ratios, 100 * frame["base_value"] / av, color=LINE, marker="o", linewidth=1.8)
    left.axhline(0, color="black", linewidth=0.8)
    left.set_xlabel("benefit base / account value")
    left.set_ylabel("rider value, % of account value")
    left.set_title("The guarantee is worth nothing until the benefit base catches up")

    for column, colour, label in (
        ("equity_down_10pct", ACCENT, "equity -10%"),
        ("equity_up_10pct", LINE, "equity +10%"),
    ):
        right.plot(ratios, 100 * frame[column] / av, color=colour, marker="o",
                   linewidth=1.6, label=label)
    right.axhline(0, color="black", linewidth=0.8)
    right.set_xlabel("benefit base / account value")
    right.set_ylabel("change in rider value, % of account value")
    right.set_title("Equity sensitivity peaks near the money")
    right.legend(frameon=False)
    return _save(fig, name)


def shock_comparison(model_rows: pd.DataFrame, disclosed: pd.DataFrame,
                     name: str = "shock_comparison") -> str:
    """Model shock responses against the sensitivities Jackson discloses.

    Both sides are scaled by account value, which is the only way a single policy and
    a $236bn book can be put on one axis.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for ax, shock_up, shock_down, title in (
        (left, "equity_up_10pct", "equity_down_10pct", "Equity shock, plus and minus 10%"),
        (right, "rates_up_100bp", "rates_down_100bp", "Parallel rate shock, plus and minus 100bp"),
    ):
        ratios = model_rows["gwb_over_av"]
        ax.plot(ratios, 100 * model_rows[shock_down] / model_rows["account_value"],
                color=ACCENT, marker="o", linewidth=1.6, label="model, down shock")
        ax.plot(ratios, 100 * model_rows[shock_up] / model_rows["account_value"],
                color=LINE, marker="o", linewidth=1.6, label="model, up shock")
        rows = disclosed[disclosed["shock"].isin([shock_up, shock_down])]
        for _, row in rows.iterrows():
            colour = ACCENT if row["shock"] == shock_down else LINE
            ax.axhline(100 * row["impact_pct_of_av"], color=colour, linestyle=":", linewidth=1.4)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("benefit base / account value")
        ax.set_title(title)
    left.set_ylabel("change in value, % of account value")
    left.legend(frameon=False)
    fig.suptitle("Dotted lines are Jackson's disclosed market risk benefit sensitivities")
    return _save(fig, name)


def hedge_performance(ledger: pd.DataFrame, name: str = "hedge_performance") -> str:
    """Cumulative profit hedged and unhedged, with the exposures that drove it."""
    valid = ledger.dropna(subset=["liability_pnl"])
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 1]})

    axes[0].plot(valid.index, valid["liability_pnl"].cumsum(), color=ACCENT, linewidth=1.6,
                 label="unhedged guarantee")
    axes[0].plot(valid.index, valid["hedged_pnl"].cumsum(), color=LINE, linewidth=1.8,
                 label="hedged, after costs")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("cumulative profit, $")
    ratio = valid["hedged_pnl"].var(ddof=1) / valid["liability_pnl"].var(ddof=1)
    axes[0].set_title(
        f"Hedging cut the variance of weekly profit by {100 * (1 - ratio):.0f}% "
        f"over {len(valid)} weeks"
    )
    axes[0].legend(frameon=False)

    axes[1].plot(valid.index, valid["equity_exposure"], color=LINE, linewidth=1.4)
    axes[1].set_ylabel("equity exposure, $")
    axes[1].set_title("Notional the hedge had to carry")

    axes[2].plot(valid.index, valid["rho_per_bp"], color=ACCENT, linewidth=1.4,
                 label="rho per bp")
    axes[2].plot(valid.index, valid["vega_per_point"], color=LINE, linewidth=1.4,
                 label="vega per vol point")
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_ylabel("$ per unit")
    axes[2].set_xlabel("date")
    axes[2].legend(frameon=False)
    return _save(fig, name)


def leg_comparison(results: dict[str, dict], name: str = "leg_comparison") -> str:
    """Variance remaining after each hedge leg is added."""
    labels = list(results)
    remaining = [100 * results[k]["variance_ratio"] for k in labels]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, remaining, color=[FILL] * len(labels), edgecolor=LINE)
    for bar, value in zip(bars, remaining):
        ax.annotate(f"{value:.1f}%", (bar.get_x() + bar.get_width() / 2, value),
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("variance remaining, % of unhedged")
    ax.set_title("Rate risk is second order next to the volatility exposure")
    ax.set_ylim(0, max(remaining) * 1.2)
    return _save(fig, name)


def economic_versus_reported(reported: pd.DataFrame,
                             name: str = "economic_versus_reported") -> str:
    """The same hedge, measured economically and on the reporting basis."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)

    top.plot(reported.index, reported["hedged_pnl"].cumsum(), color=LINE, linewidth=1.8,
             label="economic, hedged")
    top.plot(reported.index, reported["reported_net_income"].cumsum(), color=ACCENT,
             linewidth=1.6, label="reported net income")
    top.plot(reported.index, reported["reported_comprehensive_income"].cumsum(),
             color=MUTED, linewidth=1.3, linestyle="--", label="reported comprehensive income")
    top.axhline(0, color="black", linewidth=0.8)
    top.set_ylabel("cumulative, $")
    economic_std = reported["hedged_pnl"].std(ddof=1)
    reported_std = reported["reported_net_income"].std(ddof=1)
    top.set_title(
        "One hedge, two measurements: reported net income is "
        f"{reported_std / economic_std:.1f} times as volatile as the economic outcome"
    )
    top.legend(frameon=False)

    bottom.plot(reported.index, reported["own_credit_adjustment"], color=LINE, linewidth=1.5)
    bottom.set_ylabel("own-credit adjustment, $")
    bottom.set_xlabel("date")
    bottom.set_title("The non-performance piece, which is reported outside net income")
    return _save(fig, name)
