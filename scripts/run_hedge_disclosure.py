"""The hedge book against Jackson's disclosed derivative sensitivities.

Everything else in the validation workstream tests the liability. The hedging half of the project
had no external check at all, which is the gap this closes. Item 7A publishes two tables under the
same shocks on the same dates: the fair-value impact on the guarantee liabilities, and the
fair-value impact on the derivative portfolio. Their ratio is a disclosed offset ratio, and it is
the one number in the filings that speaks directly to how completely Jackson hedges.

Three things come out of it.

*What the filings say the hedge covers.* Pure arithmetic on two disclosed tables, no model
involved. Pairing is within a filing and a date, because the rate shock is 50bp in the FY2022
10-K and 100bp in the FY2025 one and crossing them would compare a half-shift with a whole one.

*What a full economic hedge of the model's liability would have covered.* An offset ratio below
one is not by itself evidence of a partial hedge: the liability is convex in rates and a swap is
nearly linear, so a book sized to kill rho exactly still under-recovers a 100bp shift. The model
supplies that benchmark, which turns "Jackson offsets 79%" into a statement about how much of the
shortfall is choice and how much is convexity.

*Where the derivative table stops being about the variable annuity.* The same derivatives hedge
the fixed-index and RILA book, and in the FY2025 filing that book's equity sensitivity jumps from
$4m at the end of 2024 to $1,321m at the end of 2025 on a balance that only doubled. Nothing
public explains a move that size, so the equity comparison is run on both bases - the market risk
benefit alone and the two liability lines together - and the 2025 equity figure is reported but not
leaned on.

Usage:  python -m scripts.run_hedge_disclosure
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from vahedge import paths
from vahedge.hedge import instruments as inst
from vahedge.hedge import simulator, sizing, strategies
from vahedge.market import scenarios
from vahedge.market import state as market_state
from vahedge.valuation import greeks as greeks_module

from scripts.run_shock_validation import (
    DIVIDEND_YIELD,
    INCEPTION,
    contract_at,
    market_at,
    rolled_states,
)
from scripts.run_valuation import PREMIUM, build

# The strategies the comparison runs. S3 is in because the disclosed book holds puts - 16.5bn of
# notional at a quarter-year weighted term at the end of 2025 - so a futures-and-swaps benchmark
# would be answering a question about an instrument set Jackson does not use.
COMPARED = ("S1", "S2", "S3")

# The other liability line Item 7A shows next to the derivative table: the fixed-index and RILA
# embedded derivative, which the same hedge book covers and which this project does not model at
# all. Its sensitivity is read alongside the guarantee's so the offset can be reported on both
# bases, because a hedge sized to the pair is not a hedge sized to the variable annuity.
OTHER_LIABILITY_LINE = "fia_rila_embedded_derivatives"


def load_derivatives() -> pd.DataFrame:
    return pd.read_csv(paths.DERIVATIVE_SENSITIVITIES, comment="#")


def load_liabilities() -> pd.DataFrame:
    return pd.read_csv(paths.DISCLOSED_SENSITIVITIES, comment="#")


def transcription_identity(derivatives: pd.DataFrame) -> pd.DataFrame:
    """Instrument lines have to sum to the disclosed Total, block by block.

    The only guard against a transcription slip in a table that was read off a filing by hand.
    It is arithmetic the filing itself asserts, so a failure is unambiguous: either a line was
    mistyped or one was missed.
    """
    rows = []
    keys = ["source_filing_fy", "as_of", "risk", "shock"]
    for key, block in derivatives.groupby(keys, sort=True):
        instruments = block.loc[block["level"] == "instrument", "impact_musd"].sum()
        total = block.loc[block["level"] == "total", "impact_musd"]
        if total.empty:
            raise ValueError(f"no disclosed total for {key}")
        rows.append({
            **dict(zip(keys, key)),
            "instrument_lines": int((block["level"] == "instrument").sum()),
            "sum_of_instruments_musd": float(instruments),
            "disclosed_total_musd": float(total.iloc[0]),
            "difference_musd": float(instruments - total.iloc[0]),
        })
    return pd.DataFrame(rows)


def disclosed_offsets(liabilities: pd.DataFrame, derivatives: pd.DataFrame) -> pd.DataFrame:
    """The offset ratio the filings themselves imply, on whichever liability basis they publish.

    The sign convention is the thing to get right and it is not the obvious one. A liability
    impact is the change in a carrying amount, so positive is a loss; a derivative impact is the
    change in an asset's fair value, so positive is a gain. The earnings effects are therefore
    minus the first and plus the second, and they cancel when the two *raw* figures carry the
    same sign. So the ratio is the derivative impact over the liability impact, with no sign
    flip: one is a complete hedge, zero is none, and a negative number is a position that moved
    the same way as the exposure and made the swing larger.

    Two liability bases appear. The FY2023 filing onward publishes the market risk benefit under
    ASU 2018-12; the FY2022 10-K publishes the pre-LDTI fixed-index and variable-annuity
    guarantee liability, which is a different measurement and cannot be compared in level with
    the other. Both are kept, each paired only with the derivative table in its own filing so
    the shock size matches, and the basis is carried in a column rather than left to be inferred
    from the date.
    """
    bases = {"market_risk_benefits": "market risk benefit",
             "fixed_index_and_va_guarantee_liabilities": "pre-LDTI guarantee liability"}
    totals = derivatives[derivatives["level"] == "total"]
    rows = []
    for (filing, as_of, shock), block in totals.groupby(
            ["source_filing_fy", "as_of", "shock"], sort=True):
        # One derivative risk table per shock: the rate shocks sit in the rates block and the
        # equity shocks in the equity block, so a group here is a single row.
        if len(block) != 1:
            raise ValueError(f"{filing} {as_of} {shock} has {len(block)} disclosed totals")
        hedge = float(block["impact_musd"].iloc[0])
        same = liabilities[(liabilities["source_filing_fy"] == filing)
                           & (liabilities["as_of"] == as_of)
                           & (liabilities["shock"] == shock)]
        lines = {name: float(same.loc[same["line_item"] == name, "impact_musd"].iloc[0])
                 for name in (*bases, OTHER_LIABILITY_LINE)
                 if not same.loc[same["line_item"] == name].empty}
        primary = next((name for name in bases if name in lines), None)
        if primary is None:
            raise ValueError(f"{filing} {as_of} {shock} has a derivative table and no liability")
        guarantee = lines[primary]
        rila = lines.get(OTHER_LIABILITY_LINE, np.nan)
        both = guarantee + (0.0 if np.isnan(rila) else rila)
        rows.append({
            "source_filing_fy": int(filing),
            "as_of": as_of,
            "liability_basis": bases[primary],
            "risk": "equity" if shock.startswith("equity") else "rates",
            "shock": shock,
            "guarantee_impact_musd": guarantee,
            "rila_impact_musd": rila,
            "both_lines_impact_musd": both,
            "derivative_impact_musd": hedge,
            "offset_of_guarantee": hedge / guarantee if guarantee else np.nan,
            "offset_of_both_lines": hedge / both if both else np.nan,
            # The both-lines ratio divides by what is left after the two liability lines cancel,
            # and by the end of 2025 that remainder is a fifth of the market risk benefit's own
            # move on the way down and a sixth on the way up. A ratio built on a sixth of a
            # number is arithmetic rather than a measurement, so the denominator is published
            # next to it instead of the reader being left to notice.
            "both_lines_share_of_guarantee": abs(both / guarantee) if guarantee else np.nan,
            # How much of the guarantee's move the other liability line absorbs on its own,
            # before any derivative is involved. A RILA book that owes more when equity rises is
            # a natural short against a guarantee that gets cheaper, and by the end of 2025 the
            # disclosure says that internal offset is larger than the derivative book's.
            "rila_absorbs_of_guarantee": (
                np.nan if np.isnan(rila) or not guarantee else -rila / guarantee),
        })
    return pd.DataFrame(rows)


def hedge_market(state, index: float, smile) -> inst.HedgeMarket:
    return inst.HedgeMarket(
        index=index, curve=state.curve, volatility=float(np.sqrt(state.heston.v0)),
        smile=smile, dividend_yield=DIVIDEND_YIELD,
        financing_rate=float(state.curve.zero(1.0 / 12.0)),
    )


def shocked_markets(state, index: float, smile, rate_shocks_bp) -> dict:
    """The same shocks the liability is repriced under, applied to the hedge's market.

    Equity is a move in the index, matching the engine's convention: it shocks the account value
    by the equity weight times the move, so the move itself is an index move and the instruments
    see it undiluted. Rates move the fitted curve's level parameter, which is the same object the
    valuation shocks, so neither side is shifting a different curve from the other.
    """
    out = {}
    for move in (0.10, -0.10):
        label = f"equity_{'up' if move > 0 else 'down'}_{abs(int(round(move * 100)))}pct"
        out[label] = hedge_market(state, index * (1.0 + move), smile)
    for basis_points in rate_shocks_bp:
        label = f"rates_{'up' if basis_points > 0 else 'down'}_{abs(int(basis_points))}bp"
        shifted = replace(state.curve, beta0=state.curve.beta0 + basis_points / 10000.0)
        out[label] = hedge_market(replace(state, curve=shifted), index, smile)
    return out


def model_offsets(setup, panel, history, smile) -> pd.DataFrame:
    """Size each strategy to the model's liability at each disclosed date, then shock both sides.

    The Greeks come from ``greeks.compute`` rather than from the regression proxy. The proxy's
    delta is off by a tenth of itself in the middle of its usable range, and a sizing comparison
    that inherited that would be measuring the proxy rather than the hedge. Four dates and one
    contract is cheap enough to do properly.
    """
    valuer, terms, calibration = setup["valuer"], setup["terms"], setup["calibration"]
    equity_weight = setup["state"].mix.equity_weight
    matrix = strategies.matrix()

    inception_state = market_at(panel, history, calibration, INCEPTION)
    alpha = valuer.calibrate_attribution(contract_at(PREMIUM, PREMIUM, 0, terms),
                                         inception_state)
    states = rolled_states(history, terms, equity_weight)

    rows = []
    for target, rolled in states.items():
        state = market_at(panel, history, calibration, rolled["date"])
        duration = int(np.floor(rolled["elapsed_years"]))
        book = contract_at(rolled["account_value"], rolled["benefit_base"], duration, terms)
        account = float(book.account_value[0])

        liability = greeks_module.disclosed_shocks(valuer, book, state, alpha)
        liability_change = {row["shock"]: float(row["change"])
                            for _, row in liability.iterrows() if row["shock"] != "base"}

        position = int(np.searchsorted(history.dates, pd.Timestamp(rolled["date"]),
                                       side="right")) - 1
        index = float(history.index[max(position, 0)])
        base_market = hedge_market(state, index, smile)
        shocked = shocked_markets(state, index, smile, (100, -100, 50, -50))

        greeks = greeks_module.compute(valuer, book, state, alpha)
        exposure = sizing.insurer_exposures(greeks, equity_weight=equity_weight)
        for key in COMPARED:
            changes, solved = simulator.shock_response(matrix[key], base_market, exposure,
                                                       account, shocked)
            for shock, hedge_change in changes.items():
                moved = liability_change[shock]
                rows.append({
                    "as_of": str(pd.Timestamp(rolled["date"]).date()),
                    "strategy": key,
                    "risk": "equity" if shock.startswith("equity") else "rates",
                    "shock": shock,
                    "liability_change": moved,
                    "hedge_change": hedge_change,
                    "liability_change_pct_av": 100 * moved / account,
                    "hedge_change_pct_av": 100 * hedge_change / account,
                    # Same convention as the disclosed ratio: the liability change is a change in
                    # a carrying amount and the hedge change is a change in an asset's mark, so
                    # they cancel in earnings when they carry the same raw sign.
                    "offset_of_liability": hedge_change / moved if moved else np.nan,
                    "residual_pct_av": 100 * (hedge_change - moved) / account,
                    "delta_left_pct_av": 100 * solved.residual.delta / account,
                    "rho_left_per_bp_pct_av": 100 * solved.residual.rho / account,
                })
    return pd.DataFrame(rows)


def comparison(disclosed: pd.DataFrame, model: pd.DataFrame) -> pd.DataFrame:
    """The two offset ratios side by side, on the shocks and years both tables reach.

    Matched on the year rather than the day, for the same reason the in-force comparison is: the
    roll lands on the last trading day of the year and the filings are dated the 31st, so
    2022-12-30 and 2022-12-31 are the same balance-sheet date seen from two sides. The comparison
    is on the ratio and never on the dollars - one stylised contract against a $236bn book cannot
    match a level, which is hypothesis four and is not retested here.
    """
    model_wide = (model.groupby(["as_of", "shock", "strategy"])["offset_of_liability"]
                  .first().unstack("strategy").add_prefix("model_offset_").reset_index())
    # One row per date, basis and shock. Three dates are disclosed by two filings each and the
    # overlapping figures agree exactly, so the duplicate adds nothing here - it stays in
    # disclosed_hedge_offset.csv, where the agreement is the point. What is *not* a duplicate is
    # 2024-12-31 at 50bp and at 100bp: two shock sizes from two filings, and keeping both is how
    # the offset ratio's own stability in the shock size becomes visible.
    once = disclosed.sort_values("source_filing_fy").drop_duplicates(
        subset=["as_of", "liability_basis", "shock"], keep="first")
    for frame in (once, model_wide):
        frame["year"] = pd.to_datetime(frame["as_of"]).dt.year
    merged = once.merge(model_wide.drop(columns="as_of"), on=["year", "shock"], how="inner")
    return (merged.drop(columns="year")
            .sort_values(["risk", "as_of", "shock"]).reset_index(drop=True))


def main() -> None:
    print("Loading the disclosure and the market history", flush=True)
    panel = pd.read_csv(paths.FRED_PANEL, comment="#", parse_dates=["date"]).set_index("date")
    calibration = market_state.load()
    history = scenarios.load_history(panel, calibration.heston, calibration.mix,
                                     dividend_yield=DIVIDEND_YIELD)
    smile = inst.smile_from_heston(calibration.heston, calibration.curve, maturity=1.0)

    derivatives = load_derivatives()
    liabilities = load_liabilities()

    identity = transcription_identity(derivatives)
    identity.to_csv(paths.TABLES / "derivative_transcription_check.csv", index=False,
                    float_format="%.4f")
    worst = float(identity["difference_musd"].abs().max())
    if worst > 0.5:
        raise ValueError(f"the derivative instrument lines miss their disclosed total by {worst}")
    print(f"  {len(identity)} disclosed blocks; instrument lines reproduce every total "
          f"exactly (worst difference ${worst:.0f}m)")

    disclosed = disclosed_offsets(liabilities, derivatives)
    disclosed.to_csv(paths.TABLES / "disclosed_hedge_offset.csv", index=False,
                     float_format="%.4f")
    print("\nWhat the filings say the derivative book offsets")
    for basis, block in disclosed.groupby("liability_basis", sort=False):
        print(f"  on the {basis}")
        for _, row in block.iterrows():
            if np.isnan(row["rila_absorbs_of_guarantee"]):
                both = ""
            else:
                thin = "" if row["both_lines_share_of_guarantee"] > 0.25 else " on a thin base"
                both = (f", {row['offset_of_both_lines']:+7.1%} of both lines{thin}"
                        f"  (RILA absorbs {row['rila_absorbs_of_guarantee']:+.0%} first)")
            print(f"    {row['as_of']} {row['shock']:<17} "
                  f"liability {row['guarantee_impact_musd']:+7.0f}m  "
                  f"derivatives {row['derivative_impact_musd']:+7.0f}m  "
                  f"offset {row['offset_of_guarantee']:+7.1%}{both}")

    print("\nFitting the valuation and rolling the contract to each disclosed date", flush=True)
    setup = build(cache_size=5)
    model = model_offsets(setup, panel, history, smile)
    model.to_csv(paths.TABLES / "model_hedge_offset.csv", index=False, float_format="%.6f")
    print("\nWhat a hedge of the model's liability offsets, sized on the model's own Greeks")
    for key in COMPARED:
        block = model[model["strategy"] == key]
        for risk in ("equity", "rates"):
            side = block[block["risk"] == risk]
            print(f"  {key} {risk:<7} offset {side['offset_of_liability'].min():+6.1%} to "
                  f"{side['offset_of_liability'].max():+6.1%} across "
                  f"{len(side)} shocks and dates, residual up to "
                  f"{side['residual_pct_av'].abs().max():.2f}% of account value")

    together = comparison(disclosed, model)
    together.to_csv(paths.TABLES / "hedge_offset_comparison.csv", index=False,
                    float_format="%.4f")
    print(f"\nThe two side by side, on the {len(together)} shocks both tables reach")
    for _, row in together.iterrows():
        print(f"  {row['as_of']} {row['shock']:<17} {row['liability_basis'][:18]:<18} disclosed "
              f"{row['offset_of_guarantee']:+7.1%}  "
              + "  ".join(f"{key} {row[f'model_offset_{key}']:+7.1%}" for key in COMPARED))

    # 2024-12-31 is disclosed at both shock sizes, which turns the shock size into a control
    # rather than a caveat: if the disclosed ratio moves far less than the model's full hedge
    # does, the shortfall is a sizing choice and not convexity.
    both_sizes = together[together["as_of"].str.startswith("2024-12")
                          & (together["risk"] == "rates")]
    if len(both_sizes) == 4:
        for direction in ("up", "down"):
            pair = both_sizes[both_sizes["shock"].str.contains(direction)]
            spread = pair["offset_of_guarantee"].max() - pair["offset_of_guarantee"].min()
            model_spread = (pair["model_offset_S2"].max() - pair["model_offset_S2"].min())
            print(f"  rates {direction} at 2024-12-31, 50bp against 100bp: the disclosed ratio "
                  f"moves {spread:.1%} and a full hedge of the model's liability {model_spread:.1%}")
    # The headline the table exists to produce. A ratio below one is not evidence of a partial
    # hedge on its own, so the gap is quoted against the model's own full hedge rather than
    # against one: S2 kills delta and rho together, which is what the disclosed programme says
    # it does, so it is the right benchmark for both risks.
    for risk, benchmark in (("rates", "S2"), ("equity", "S2")):
        side = together[together["risk"] == risk]
        if side.empty:
            continue
        print(f"  {risk}: disclosed {side['offset_of_guarantee'].min():.0%} to "
              f"{side['offset_of_guarantee'].max():.0%} against "
              f"{side[f'model_offset_{benchmark}'].min():.0%} to "
              f"{side[f'model_offset_{benchmark}'].max():.0%} for a full hedge of the model's "
              f"own liability")

    print(f"\nwrote four tables to {paths.TABLES}")


if __name__ == "__main__":
    main()
