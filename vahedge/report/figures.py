"""Figures, built from the tables the scripts wrote rather than from the models.

Reading the committed tables instead of taking live model objects is deliberate. It means the
whole figure set redraws in a couple of seconds without a simulation, which is what makes it
cheap enough to redraw every time a number changes; it means a figure can never disagree with
the table beside it in the report; and it means a reader who wants to check a figure has the
exact numbers behind it in a file they can open.

Titles say what the figure shows rather than announcing that an analysis happened, and every
axis carries its units. Everything renders through the Agg backend so the scripts run with no
display attached.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import paths

LINE = "#1f4e79"
ACCENT = "#c0504d"
MUTED = "#7f7f7f"
FILL = "#dce6f1"
# The option chain's longest expiry against the projection's horizon. Both appear on the
# calibration figure because the gap between them is the model's largest extrapolation.
CHAIN_MAX_MATURITY = 3.23
PROJECTION_YEARS = 45


def table(name: str) -> pd.DataFrame:
    return pd.read_csv(paths.TABLES / f"{name}.csv")


def _save(fig, name: str) -> str:
    paths.ensure_output_dirs()
    target = paths.FIGURES / f"{name}.png"
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(target)


def cash_flow_profile(name: str = "cash_flow_profile") -> str:
    """Expected claims and fee income by policy year, with the exhaustion curve under them."""
    flows = table("cash_flow_profile")
    years = flows["policy_year"]
    claims = flows["expected_claim"] + flows["expected_death_claim"]

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                      gridspec_kw={"height_ratios": [2, 1]})
    top.bar(years, flows["expected_fee"], color=FILL, edgecolor=LINE, linewidth=0.5,
            label="attributable fees collected")
    top.bar(years, -claims, color=ACCENT, alpha=0.75, label="guarantee payments made")
    top.axhline(0, color="black", linewidth=0.8)
    top.set_ylabel("expected cash flow, discounted dollars per policy year")
    crossover = int(years[claims > flows["expected_fee"]].min())
    top.axvline(crossover, color=MUTED, linestyle=":", linewidth=1.2)
    top.set_title("Fees arrive first and the guarantee pays later: the crossover is policy "
                  f"year {crossover}")
    top.legend(frameon=False)

    bottom.plot(years, flows["prob_exhausted"], color=LINE, linewidth=1.8)
    bottom.fill_between(years, 0, flows["prob_exhausted"], color=FILL)
    bottom.plot(years, flows["survival"], color=MUTED, linewidth=1.4, linestyle="--",
                label="still alive")
    bottom.set_ylim(0, 1)
    bottom.set_xlabel("policy year")
    bottom.set_ylabel("probability")
    bottom.set_title("Contract value exhausted (solid) against survival (dashed): the guarantee "
                     "bites where the two curves overlap")
    bottom.legend(frameon=False, loc="center right")
    return _save(fig, name)


def volatility_calibration(name: str = "volatility_calibration") -> str:
    """How well the surface is fitted, and how far past it the liability is valued.

    The right-hand panel is the point: the chain stops at 3.2 years and the projection runs 45,
    so the long-run variance that does most of the work on this liability is not in the data the
    model was fitted to. Drawing the two spans on one axis is more honest than a fit statistic.
    """
    import json

    fit = table("surface_fit")
    heston = json.loads((paths.DATA_PROCESSED / "market_calibration.json").read_text())["heston"]
    v0, kappa, theta = heston["v0"], heston["kappa"], heston["theta"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
    left.plot(fit["maturity"], 100 * fit["mean_market_vol"], color=LINE, marker="o",
              linewidth=1.6, label="market, averaged across strikes")
    left.plot(fit["maturity"], 100 * (fit["mean_market_vol"] + fit["bias_vol"]), color=ACCENT,
              marker="s", markersize=4, linewidth=1.4, label="model")
    left.set_xlabel("option maturity, years")
    left.set_ylabel("implied volatility, %")
    left.set_title(f"Fitted to {int(fit['n_quotes'].sum()):,} quotes, worst maturity off by "
                   f"{100 * float(fit['rmse_vol'].max()):.2f} points")
    left.legend(frameon=False)

    # The volatility the liability is actually valued at: the square root of the Heston model's
    # expected integrated variance over each horizon. Drawing it is the only way to show that
    # the number doing the work on a forty-five-year guarantee is a parameter the three-year
    # chain had to pin down on its own.
    horizon = np.linspace(0.05, PROJECTION_YEARS, 600)
    decay = (1.0 - np.exp(-kappa * horizon)) / (kappa * horizon)
    average_vol = 100 * np.sqrt(theta + (v0 - theta) * decay)
    right.plot(horizon, average_vol, color=LINE, linewidth=1.8)
    right.axvspan(0, CHAIN_MAX_MATURITY, color=FILL,
                  label=f"quoted: options out to {CHAIN_MAX_MATURITY:.1f} years")
    right.axhline(100 * np.sqrt(theta), color=ACCENT, linestyle="--", linewidth=1.3,
                  label=f"long-run level, {100 * np.sqrt(theta):.1f}%")
    right.set_xlim(0, PROJECTION_YEARS)
    right.set_xlabel("horizon, years")
    right.set_ylabel("average volatility to the horizon, %")
    right.set_title("Almost all of the liability sits past the last quote")
    right.legend(frameon=False, loc="lower right")

    # The part of that extrapolation free data can check, inset on the same panel. Today's curve
    # is not the comparison - it starts from today's variance and the indices are a decade - so
    # this plots the decade mean at each index tenor against the decade mean of what the model's
    # own mapping produces when it carries one of those quotes to another tenor. Level is held
    # equal on both sides by construction and only the slope is being judged.
    tenors = table("volatility_curve_check")
    observed = pd.concat([
        tenors[["quoted_tenor_years", "mean_quoted_vol_points"]].rename(
            columns={"quoted_tenor_years": "tenor", "mean_quoted_vol_points": "level"}),
        tenors[["target_tenor_years", "mean_target_vol_points"]].rename(
            columns={"target_tenor_years": "tenor", "mean_target_vol_points": "level"}),
    ]).drop_duplicates().sort_values("tenor")

    inset = right.inset_axes([0.30, 0.26, 0.40, 0.46])
    inset.plot(observed["tenor"], observed["level"], color=MUTED, marker="D", markersize=5,
               linewidth=1.4, label="the indices")
    carried = tenors["mean_target_vol_points"] + tenors["mean_error_vol_points"]
    inset.plot(tenors["target_tenor_years"], carried, color=ACCENT, marker="x", markersize=7,
               linestyle="none", label="carried there by the model")
    inset.set_xlim(0, 0.58)
    inset.set_xlabel("tenor, years", fontsize=7)
    inset.tick_params(labelsize=7)
    inset.set_title("decade means: the model's slope is the flatter", fontsize=8)
    inset.legend(frameon=False, fontsize=7, loc="upper left")
    return _save(fig, name)


def moneyness_curve(name: str = "moneyness_curve") -> str:
    """Value and the disclosed shock responses across the benefit-base-to-account ratio."""
    profile = table("moneyness_profile")
    ratios = profile["gwb_over_av"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
    left.plot(ratios, profile["value_pct_av"], color=LINE, marker="o", linewidth=1.8)
    left.axhline(0, color="black", linewidth=0.8)
    left.set_xlabel("benefit base / account value")
    left.set_ylabel("market risk benefit, % of account value")
    left.set_title("The guarantee is an asset until the benefit base catches up")

    for column, colour, label in (("equity_down_10pct_pct_av", ACCENT, "equity -10%"),
                                  ("equity_up_10pct_pct_av", LINE, "equity +10%")):
        right.plot(ratios, profile[column], color=colour, marker="o", linewidth=1.6, label=label)
    right.axhline(0, color="black", linewidth=0.8)
    right.set_xlabel("benefit base / account value")
    right.set_ylabel("change in value, % of account value")
    peak = float(ratios[profile["equity_down_over_up"].idxmax()])
    right.axvline(peak, color=MUTED, linestyle=":", linewidth=1.2)
    right.set_title(f"Asymmetry peaks at a ratio of {peak:.2f}, where the guarantee is "
                    "closest to being at the money")
    right.legend(frameon=False)
    return _save(fig, name)


def shock_comparison(name: str = "shock_comparison") -> str:
    """The model's shock responses across moneyness, against what Jackson discloses.

    Both sides are shares of account value, which is the only axis a single policy and a $236bn
    book can share. Where a dotted line crosses a solid one is the moneyness at which the model
    would reproduce that year's disclosure, which is the statistic the level comparison rests on.
    """
    profile = table("moneyness_profile")
    disclosed = pd.read_csv(paths.DATA_PROCESSED / "disclosed_scaled.csv")
    disclosed = disclosed.query("line_item == 'market_risk_benefits'").drop_duplicates(
        subset=["as_of", "shock"])
    latest = disclosed[disclosed["as_of"] == disclosed["as_of"].max()]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, up, down, title in (
        (left, "equity_up_10pct", "equity_down_10pct", "Equity shock, plus and minus 10%"),
        (right, "rates_up_100bp", "rates_down_100bp", "Parallel rate shock, 100bp either way"),
    ):
        ratios = profile["gwb_over_av"]
        ax.plot(ratios, profile[f"{down}_pct_av"], color=ACCENT, marker="o", linewidth=1.6,
                label="model, down shock")
        ax.plot(ratios, profile[f"{up}_pct_av"], color=LINE, marker="o", linewidth=1.6,
                label="model, up shock")
        for _, row in latest[latest["shock"].isin([up, down])].iterrows():
            colour = ACCENT if row["shock"] == down else LINE
            ax.axhline(100 * row["impact_pct_of_av"], color=colour, linestyle=":", linewidth=1.4)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("benefit base / account value")
        ax.set_title(title)
    left.set_ylabel("change in value, % of account value")
    left.legend(frameon=False)
    fig.suptitle(f"Dotted lines are the disclosed sensitivities at "
                 f"{latest['as_of'].iloc[0]}; the equity pair is crossed inside the model's "
                 f"range and the rate pair is not")
    return _save(fig, name)


def hedge_frontier(name: str = "hedge_frontier") -> str:
    """Residual volatility against what it cost, every strategy and rebalance rule.

    The frontier rather than a single number, because a hedging result quoted at one cost
    assumption is a result about that assumption. Each strategy appears three times, once per
    cost multiple, and a conclusion that survives the spread is worth something.
    """
    frontier = table("hedge_frequency_frontier")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    markers = {"daily": "o", "weekly": "s", "monthly": "^", "band": "D"}
    for strategy, block in frontier.groupby("strategy"):
        for rule, rows in block.groupby("rebalance"):
            ordered = rows.sort_values("cost_multiple")
            ax.plot(100 * ordered["total_cost_pct"], 100 * ordered["pnl_sd_pct"],
                    marker=markers.get(rule, "o"), markersize=5, linewidth=1.0, alpha=0.85,
                    label=f"{strategy}, {rule}")
    ax.set_xlabel("total cost over the decade, % of account value")
    ax.set_ylabel("residual daily standard deviation, % of account value")
    ax.set_title("Each line is one strategy and rebalance rule swept over the cost assumption; "
                 "the useful corner is bottom left")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    return _save(fig, name)


def leg_comparison(name: str = "leg_comparison") -> str:
    """What each instrument class took out of the daily variation, and what it cost.

    The left panel is the residual itself rather than a share of the unhedged book, and the
    unhedged bar is in the picture, so the comparison carries its own denominator. An earlier
    version divided by whichever row happened to be largest when the unhedged run was missing
    from the table, which read as "share of unhedged" and was a share of the delta-only hedge.
    """
    frontier = table("hedge_frequency_frontier")
    base = frontier[frontier["cost_multiple"] == 1.0]
    base = base[base["rebalance"].isin(("daily", "none"))].sort_values("strategy")

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    heights = 100 * base["pnl_sd_pct"]
    bars = left.bar(base["strategy"], heights, color=FILL, edgecolor=LINE)
    for bar, value in zip(bars, heights):
        left.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, value),
                      ha="center", va="bottom", fontsize=9)
    left.set_ylabel("residual daily standard deviation, % of account value")
    left.set_title("what each one removes")

    costs = 100 * base["total_cost_pct"]
    cost_bars = right.bar(base["strategy"], costs, color="white", edgecolor=ACCENT, hatch="//")
    # The option leg is two orders of magnitude above the others, so on a linear axis the first
    # three bars are a flat line. Labelling them is the only way the ratio is readable without
    # a log axis, which the unhedged book's exact zero rules out anyway.
    for bar, value in zip(cost_bars, costs):
        right.annotate(f"{value:.2f}", (bar.get_x() + bar.get_width() / 2, value),
                       ha="center", va="bottom", fontsize=9)
    right.set_ylabel("total cost over the decade, % of account value")
    right.set_title("what each one cost")
    for ax in (left, right):
        ax.set_xlabel("strategy")

    unhedged = base[base["strategy"] == "S0"]["pnl_sd_pct"]
    removed = ("" if unhedged.empty else
               f", removing {100 * (1 - (base['pnl_sd_pct'].min() / float(unhedged.iloc[0])) ** 2):.1f}% "
               f"of the variance between them")
    fig.suptitle(f"Each instrument class takes out less than the one before{removed}, "
                 f"and the option leg is where the cost is")
    fig.tight_layout()
    return _save(fig, name)


def economic_versus_reported(name: str = "economic_versus_reported") -> str:
    """One hedge, measured economically and on the reporting basis, with the own-credit line."""
    daily = table("reporting_lens_daily_s2")
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.set_index("date")

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    top.plot(daily.index, daily["economic_pnl"].cumsum(), color=LINE, linewidth=1.8,
             label="economic, hedged")
    top.plot(daily.index, daily["net_income"].cumsum(), color=ACCENT, linewidth=1.6,
             label="reported net income")
    top.plot(daily.index, daily["comprehensive_income"].cumsum(), color=MUTED, linewidth=1.3,
             linestyle="--", label="reported comprehensive income")
    top.axhline(0, color="black", linewidth=0.8)
    top.set_ylabel("cumulative, dollars")
    economic_sd = float(daily["economic_pnl"].std(ddof=0))
    reported_sd = float(daily["net_income"].std(ddof=0))
    top.set_title("One hedge, two measurements: reported net income is "
                  f"{reported_sd / economic_sd:.1f} times as variable day to day as the "
                  "economic outcome the hedge was sized on")
    top.legend(frameon=False)

    bottom.plot(daily.index, daily["own_credit_adjustment"], color=LINE, linewidth=1.5,
                label="own-credit adjustment")
    twin = bottom.twinx()
    twin.plot(daily.index, 100 * daily["own_credit_spread"], color=ACCENT, linewidth=1.1,
              alpha=0.7, label="spread, right axis")
    bottom.set_ylabel("own-credit adjustment, dollars")
    twin.set_ylabel("spread over Treasury, %")
    bottom.set_xlabel("date")
    bottom.set_title("The non-performance piece, which the rule books outside net income, and "
                     "the spread that drives it")
    bottom.legend(frameon=False, loc="upper left")
    twin.legend(frameon=False, loc="lower right")
    return _save(fig, name)


def disclosed_offset(name: str = "disclosed_offset") -> str:
    """How much of its own guarantee move Jackson's derivative book covered, against a full hedge.

    The reference line at one is where a complete hedge sits, and the band around it is the range
    the model's own delta-and-rho hedge reaches across these dates - not a tolerance, but the
    scatter convexity alone produces over a shock this large. A disclosed bar inside the band is
    indistinguishable from a full hedge; one well below it is not.
    """
    offsets = table("hedge_offset_comparison").sort_values(["risk", "as_of", "shock"])

    fig, panels = plt.subplots(1, 2, figsize=(11.5, 5), sharey=True)
    for ax, (risk, colour) in zip(panels, (("rates", LINE), ("equity", ACCENT))):
        side = offsets[offsets["risk"] == risk]
        model = side["model_offset_S2"]
        position = np.arange(len(side))
        ax.axhspan(100 * model.min(), 100 * model.max(), color=FILL, zorder=0,
                   label=f"a full hedge of the model's liability, "
                         f"{100 * model.min():.0f} to {100 * model.max():.0f}%")
        ax.axhline(100, color=MUTED, linewidth=1.0, linestyle="--", zorder=1)
        heights = 100 * side["offset_of_guarantee"]
        bars = ax.bar(position, heights, color=colour, zorder=2)
        for bar, value in zip(bars, heights):
            ax.annotate(f"{value:.0f}", (bar.get_x() + bar.get_width() / 2, value),
                        ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.8, zorder=3)
        ax.set_xticks(position)
        ax.set_xticklabels([f"{row.as_of[:4]}\n{'down' if 'down' in row.shock else 'up'}"
                            for row in side.itertuples()], fontsize=9)
        # The 2022 bars are the pre-LDTI carrying value and the rest are the market risk benefit.
        # Two measurements, not one series, so the break is drawn rather than left to the reader.
        pre_ldti = int((side["liability_basis"] == "pre-LDTI guarantee liability").sum())
        if 0 < pre_ldti < len(side):
            ax.axvline(pre_ldti - 0.5, color=MUTED, linewidth=0.9, linestyle=":", zorder=4)
            ax.annotate("pre-LDTI basis", (pre_ldti / 2 - 0.5, 0.97), xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=8, color=MUTED)
            ax.annotate("market risk benefit", ((pre_ldti + len(side)) / 2 - 0.5, 0.97),
                        xycoords=("data", "axes fraction"), ha="center", va="top",
                        fontsize=8, color=MUTED)
        ax.set_title(f"{risk}, {'50bp in 2022 and 100bp after' if risk == 'rates' else '10%'}")
        ax.legend(frameon=False, loc="lower left", fontsize=8)
    panels[1].annotate("the 2025 pair is the guarantee alone.\nThe RILA book absorbs 79 to 84% of "
                       "its\nequity move before any derivative does,\nleaving these 44% of what "
                       "remains",
                       (0.74, 0.57), xycoords="axes fraction", ha="center", va="center",
                       fontsize=8, color=MUTED,
                       bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="none"))
    panels[0].set_ylabel("derivative impact over guarantee impact, %")
    # Headroom for the basis labels along the top, which the 2022 equity bar otherwise runs into.
    heights = 100 * offsets["offset_of_guarantee"]
    panels[0].set_ylim(min(heights.min(), 0.0) - 16.0, heights.max() + 18.0)
    fig.suptitle("Jackson's derivative book covers four fifths of its guarantees' rate "
                 "sensitivity and half the equity sensitivity,\nwhere a full hedge of the same "
                 "exposure would cover all of it")
    fig.tight_layout()
    return _save(fig, name)


ALL = (cash_flow_profile, volatility_calibration, moneyness_curve, shock_comparison,
       hedge_frontier, leg_comparison, disclosed_offset, economic_versus_reported)


def draw_all() -> list[str]:
    """Every figure that has the table it needs, and a note for every one that does not."""
    written = []
    for draw in ALL:
        try:
            written.append(draw())
        except FileNotFoundError as missing:
            written.append(f"skipped {draw.__name__}: {missing.filename} has not been written")
    return written
