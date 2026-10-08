"""E5. What the index-linked book nets against the guarantee, and whether the filings agree.

Jackson owes the upside on a registered index-linked annuity and owns it on a withdrawal
guarantee, so the two books carry opposite equity risk and part of the guarantee never needs an
external hedge at all. The disclosure validation measured that netting from the outside: at the
end of 2025 the fixed-index and RILA embedded derivative absorbs 79 to 84% of the market risk
benefit's equity move before any derivative is involved, and a year earlier it absorbs 0.2%, on a
balance that only doubled. Nothing public explains a change that size. This is the experiment that
asks whether an index-linked book *can* behave either way.

The mechanism is the cap, and it reads backwards from the product's own vocabulary. Differentiating
the credited return gives a slope of one between zero and the cap, zero above it, zero inside the
buffer and one again below it - so a buffer does not reduce the insurer's exposure past its own
depth, because the contract holder bears every point of loss beyond it. Only a capped segment has
stopped moving. A book's equity exposure is therefore a statement about where its segments sit
against their caps, not about how large it is, which is why a balance that doubled can move its
sensitivity by much more or much less than twice.

Three things come out.

*The surface.* What a segment's equity exposure is across its term and its index level, per unit
of account value, from the same simulation the guarantee book is valued on.

*The book.* Six annual cohorts rolled along realised index history, which places each one against
its own cap the way the decade actually went, and the exposure that book carries at each year-end.

*The locator.* What share of the disclosed guarantee sensitivity that book absorbs at the
disclosed relative size, and - the other way round - what size would be needed to absorb the
disclosed share. Where the disclosure sits outside anything the model reaches, that is the result.

Usage:  python -m scripts.run_rila_netting
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vahedge import paths
from vahedge.liability import rila
from vahedge.market import scenarios
from vahedge.market import state as market_state
from vahedge.market.simulate import SubAccountMix, simulate

# The 10-K gives 20% as its buffer example and does not publish caps, which move with the term,
# the index and the rate environment. Six years is Jackson's longest standard point-to-point term
# and the one a book of this age would be dominated by.
BUFFER = 0.20
TERM_YEARS = 6

# The cap turned out not to be solvable, and why is a result rather than an obstacle. Funding a
# 20% buffer over six years costs about 0.21 of premium at the September 2026 surface, while the
# six-year Treasury alone frees up 0.20 and a general-account spread adds 0.05 to 0.09 on top. The
# break-even cap therefore does not bind at all: the contract is profitable uncapped, and any cap
# an insurer sets is margin and competition rather than necessity. So the cap is swept, not
# solved - which is also why the product grew as rates rose.
FUNDING_SPREADS = (0.010, 0.015, 0.020)
CAP_GRID = (0.40, 0.60, 0.80, 1.00)
HEADLINE_CAP = 0.60

# From put-call parity on the committed chain, which identifies the forward and the discount factor
# per expiry without assuming a yield. Beyond one year the implied figure sits between 0.77% and
# 0.86% at every expiry out to 3.23 years, so it is carried flat to six. That extrapolation is the
# assumption; the level is not.
CHAIN_DIVIDEND_YIELD = 0.0082

# Where a segment can sit: whole years elapsed, and the index ratio since its own term began.
ELAPSED_GRID = (0, 1, 2, 3, 4, 5)
GROWTH_GRID = (0.70, 0.85, 1.00, 1.15, 1.30, 1.50, 1.75, 2.00)

# One cohort per issue year, weighted by the net new money that went into the book that year -
# which the filings give, because the index-linked balance is disclosed at every year-end. An
# equal weighting would be badly wrong here: the book went from $1.9bn to $20.3bn in three years,
# so most of it is young, and a young segment sits nearer its cap strike and carries more equity
# exposure than an old one that has run away from it.
COHORT_YEARS = 6
VALUATION_DATES = ("2022-12-30", "2023-12-29", "2024-12-31", "2025-12-31")

DIVIDEND_YIELD = 0.015
SIM_PATHS = 100_000
SIM_SEED = 20261004


def market_paths(state, n_years: int = TERM_YEARS):
    """One simulation, reused for every segment on the grid.

    Under Heston the return distribution does not depend on the index level, so a segment that
    has already moved is the same contract on a scaled forward rather than a reason to redraw.
    That is what makes a whole surface affordable.
    """
    return simulate(state.heston, state.hull_white(), state.correlations,
                    SubAccountMix.all_equity(), n_years=n_years, n_paths=SIM_PATHS, seed=SIM_SEED)


def forward(state, years: float = float(TERM_YEARS)) -> float:
    """The index forward per unit of spot, carrying the chain's own dividend yield.

    A RILA credits a price return, so the forward has to be the price-index forward and not the
    total-return one. Getting that wrong overstates the upside the insurer owes by the whole
    dividend yield compounded over six years, which is five per cent of premium here.
    """
    return float(np.exp(-CHAIN_DIVIDEND_YIELD * years)) / float(state.curve.discount(years))


def funding_table(state) -> pd.DataFrame:
    """What the buffer costs against what the general account frees up to pay for it.

    The arithmetic the break-even cap would have hidden. The insurer takes the premium and owes
    the account grown by the credited return, so what it has to spend is one less the funding
    discount factor, and what it has to buy is the present value of the credit. Where the second
    is smaller than the first at any cap, the cap does not bind and the contract is written at a
    margin rather than at break-even.
    """
    discount = float(state.curve.discount(TERM_YEARS))
    zero = -np.log(discount) / TERM_YEARS
    rows = []
    for spread in FUNDING_SPREADS:
        available = 1.0 - float(np.exp(-(zero + spread) * TERM_YEARS))
        for cap in CAP_GRID:
            cost = rila.value_heston(state.heston, forward(state), discount,
                                     rila.RilaTerms(BUFFER, cap, float(TERM_YEARS)))
            rows.append({
                "funding_spread": spread,
                "cap": cap,
                "annualised_cap": (1.0 + cap) ** (1.0 / TERM_YEARS) - 1.0,
                "funding_available_pct_of_premium": 100 * available,
                "credit_cost_pct_of_premium": 100 * cost,
                "margin_pct_of_premium": 100 * (available - cost),
                "cap_binds": cost >= available,
            })
    return pd.DataFrame(rows)


def exposure_surface(paths_, cap: float) -> pd.DataFrame:
    """Equity exposure per unit of account value, across term position and index level."""
    terms = rila.RilaTerms(BUFFER, cap, float(TERM_YEARS))
    rows = []
    for elapsed in ELAPSED_GRID:
        for growth in GROWTH_GRID:
            point = rila.equity_exposure(paths_, terms, realised_growth=growth,
                                         elapsed_years=elapsed)
            rows.append({
                "elapsed_years": elapsed,
                "remaining_years": TERM_YEARS - elapsed,
                "index_since_issue": growth,
                "embedded_derivative_pct_of_account": 100 * point["embedded_derivative"],
                "equity_exposure_pct_of_account": point["equity_exposure_pct_of_account"],
                "equity_up_10pct_pct_of_account": point["equity_up_10pct_pct_of_account"],
                "equity_down_10pct_pct_of_account": point["equity_down_10pct_pct_of_account"],
                "share_capped": point["share_capped"],
                "share_through_buffer": point["share_through_buffer"],
            })
    return pd.DataFrame(rows)


def issuance_weights(book_stats: pd.DataFrame, as_of: str) -> dict:
    """How much of the book was written in each of the last six years, from the filed balances.

    Net new money is the year-on-year change in the index-linked balance, which is issuance less
    what ran off and plus what was credited. Separating those needs data nobody publishes, so the
    change is used as it stands and the first year the series reaches is taken whole, because the
    product launched shortly before it. Years the series does not reach get no weight, which is
    right rather than conservative: the book did not exist.
    """
    funds = (book_stats.set_index(pd.to_datetime(book_stats["as_of"]).dt.year)
             ["rila_contract_holder_funds_musd"].sort_index())
    year = pd.Timestamp(as_of).year
    weights = {}
    for elapsed in range(COHORT_YEARS):
        issued_in = year - elapsed
        if issued_in not in funds.index:
            weights[elapsed] = 0.0
        elif issued_in - 1 not in funds.index:
            weights[elapsed] = float(funds[issued_in])
        else:
            weights[elapsed] = max(float(funds[issued_in] - funds[issued_in - 1]), 0.0)
    total = sum(weights.values())
    return {k: (v / total if total else 0.0) for k, v in weights.items()}


def cohort_book(paths_, cap: float, history, as_of: str, book_stats: pd.DataFrame) -> pd.DataFrame:
    """Cohorts placed against their own caps by what the index actually did, weighted by issuance.

    Each cohort is one year of issuance, so at a given date the cohort issued n years ago has n
    years elapsed and an index ratio equal to the index then against the index now. Rolling them
    on realised history rather than on a model path is the point: where a book sits against its
    caps is a fact about the decade, and in a decade that rose it is most of the reason the
    exposure is where it is.
    """
    terms = rila.RilaTerms(BUFFER, cap, float(TERM_YEARS))
    stamp = pd.Timestamp(as_of)
    position = int(np.searchsorted(history.dates, stamp, side="right")) - 1
    level_now = float(history.index[position])
    weights = issuance_weights(book_stats, as_of)

    rows = []
    for elapsed in range(COHORT_YEARS):
        issued = stamp - pd.DateOffset(years=elapsed)
        at_issue = int(np.searchsorted(history.dates, issued, side="right")) - 1
        if at_issue < 0 or weights[elapsed] == 0.0:
            # Either the free index history cannot place the cohort - it starts in September 2016
            # - or the book had not been written yet. Carried as a zero-weight row rather than
            # dropped, so the table says which cohorts are in the book and which are not.
            rows.append({"as_of": as_of, "elapsed_years": elapsed, "weight": weights[elapsed],
                         "index_since_issue": np.nan,
                         "equity_exposure_pct_of_account": np.nan, "reachable": False})
            continue
        growth = level_now / float(history.index[at_issue])
        point = rila.equity_exposure(paths_, terms, realised_growth=growth,
                                     elapsed_years=elapsed)
        rows.append({
            "as_of": as_of,
            "elapsed_years": elapsed,
            "weight": weights[elapsed],
            "index_since_issue": growth,
            "embedded_derivative_pct_of_account": 100 * point["embedded_derivative"],
            "equity_exposure_pct_of_account": point["equity_exposure_pct_of_account"],
            "equity_up_10pct_pct_of_account": point["equity_up_10pct_pct_of_account"],
            "equity_down_10pct_pct_of_account": point["equity_down_10pct_pct_of_account"],
            "share_capped": point["share_capped"],
            "reachable": True,
        })
    return pd.DataFrame(rows)


def disclosed_guarantee_sensitivity() -> pd.DataFrame:
    """The guarantee's own equity sensitivity per dollar of variable annuity account value.

    Jackson's, not the model's. The netting ratio in the filings is a ratio of two disclosed
    dollar figures, so reproducing it needs the disclosed denominator - the model's own level sits
    an order of magnitude above it, for reasons hypothesis four already covers, and substituting
    it here would compare the model's book with Jackson's RILA.
    """
    scaled = pd.read_csv(paths.DISCLOSED_SCALED)
    guarantee = scaled[(scaled["line_item"] == "market_risk_benefits")
                       & (scaled["shock"].str.startswith("equity"))]
    out = (guarantee.drop_duplicates(subset=["as_of", "shock"])
           .pivot_table(index="as_of", columns="shock", values="impact_pct_of_av"))
    return out.reset_index()


def disclosed_absorption() -> pd.DataFrame:
    """What share of the guarantee's equity move the other liability line already absorbs."""
    offsets = pd.read_csv(paths.TABLES / "disclosed_hedge_offset.csv")
    equity = offsets[(offsets["risk"] == "equity")
                     & (offsets["liability_basis"] == "market risk benefit")]
    return (equity.drop_duplicates(subset=["as_of", "shock"])
            .pivot_table(index="as_of", columns="shock", values="rila_absorbs_of_guarantee")
            .reset_index())


def weighted(cohorts: pd.DataFrame, column: str) -> float:
    live = cohorts[cohorts["reachable"]]
    if live.empty or live["weight"].sum() == 0:
        return float("nan")
    return float(np.average(live[column], weights=live["weight"]))


def level_check(book: pd.DataFrame, book_stats: pd.DataFrame) -> pd.DataFrame:
    """The model's embedded derivative against the filed one, which is what licenses its delta.

    A delta nobody can check is worth little. The level can be checked: the filings publish the
    index-linked embedded derivative and the balance it sits inside at every year-end, and the
    model produces the same ratio from contract terms and realised index history. Agreement here
    does not make the delta right, but disagreement would make it worthless.

    The denominator is "other contract holder funds", which the note says *includes* the embedded
    derivative, so it is larger than the host and is not the account value. The model's figure is
    per unit of account value. The two are therefore close but not the same quantity, and the
    comparison is of magnitude and direction of travel rather than of a number.
    """
    rows = []
    stats = book_stats.copy()
    stats["year"] = pd.to_datetime(stats["as_of"]).dt.year
    for as_of, cohorts in book.groupby("as_of"):
        match = stats[stats["year"] == pd.Timestamp(as_of).year]
        if match.empty:
            continue
        filed = float(match["rila_embedded_derivative_musd"].iloc[0])
        funds = float(match["rila_contract_holder_funds_musd"].iloc[0])
        rows.append({
            "as_of": as_of,
            "model_embedded_derivative_pct": weighted(cohorts, "embedded_derivative_pct_of_account"),
            "disclosed_embedded_derivative_pct": 100 * filed / funds,
            "disclosed_funds_musd": funds,
            "rila_funds_over_va_account": funds / float(match["va_separate_account_musd"].iloc[0]),
        })
    return pd.DataFrame(rows)


def netting(book: pd.DataFrame, guarantee: pd.DataFrame, absorbed: pd.DataFrame,
            book_stats: pd.DataFrame) -> pd.DataFrame:
    """The model's absorption at a given relative size, and the size the disclosure implies.

    Both sides are per dollar of their own account value, so the only thing that turns one into
    the other is the ratio of the two books:

        absorbed = rila exposure per unit * (RILA account / VA account) / guarantee per unit

    The guarantee's per-unit figure is the disclosed one. Running it the other way gives the
    implied size: what the index-linked book would have to be worth, relative to the variable
    annuity book, for the disclosed absorption to be an unhedged index-linked book.
    """
    rows = []
    stats = book_stats.copy()
    stats["year"] = pd.to_datetime(stats["as_of"]).dt.year
    for as_of, cohorts in book.groupby("as_of"):
        exposure = weighted(cohorts, "equity_exposure_pct_of_account")
        year = pd.Timestamp(as_of).year
        gmatch = guarantee[pd.to_datetime(guarantee["as_of"]).dt.year == year]
        amatch = absorbed[pd.to_datetime(absorbed["as_of"]).dt.year == year]
        smatch = stats[stats["year"] == year]
        if gmatch.empty or amatch.empty or smatch.empty or not np.isfinite(exposure):
            continue
        observed = (float(smatch["rila_contract_holder_funds_musd"].iloc[0])
                    / float(smatch["va_separate_account_musd"].iloc[0]))
        for shock in ("equity_down_10pct", "equity_up_10pct"):
            if shock not in amatch or shock not in gmatch:
                continue
            share = float(amatch[shock].iloc[0])
            # Shock for shock. The guarantee's disclosed impact is per 10% move and so is the
            # model's RILA figure, both per unit of their own account value, so the only thing
            # between them is the ratio of the two books.
            guarantee_move = abs(float(gmatch[shock].iloc[0]))
            rila_move = abs(weighted(cohorts, f"{shock}_pct_of_account"))
            rows.append({
                "as_of": as_of,
                "shock": shock,
                "cohorts_in_book": int((cohorts["weight"] > 0).sum()),
                "model_rila_exposure_per_unit": exposure,
                "model_rila_10pct_per_unit": rila_move,
                "disclosed_guarantee_10pct_per_unit": guarantee_move,
                "disclosed_absorbed_share": share,
                "observed_rila_over_va": observed,
                # What the model says a book of the observed size would absorb, and - the other
                # way round - what size the filed absorption implies. The gap between them is
                # this experiment's result.
                "model_absorbed_at_observed_size": rila_move * observed / guarantee_move,
                "implied_rila_over_va": (share * guarantee_move / rila_move
                                         if rila_move else np.nan),
                "filed_over_model": (share / (rila_move * observed / guarantee_move)
                                     if rila_move and observed else np.nan),
            })
    return pd.DataFrame(rows)


def main() -> None:
    print("Calibration, history and one simulation", flush=True)
    calibration = market_state.load()
    state = calibration
    panel = pd.read_csv(paths.FRED_PANEL, comment="#", parse_dates=["date"]).set_index("date")
    history = scenarios.load_history(panel, calibration.heston, calibration.mix,
                                     dividend_yield=DIVIDEND_YIELD)
    from vahedge.valuation.engine import MarketState
    simulated_state = MarketState.from_calibration(calibration)
    market = market_paths(simulated_state)

    funding = funding_table(simulated_state)
    funding.to_csv(paths.TABLES / "rila_funding.csv", index=False, float_format="%.6f")
    print(f"  a {BUFFER:.0%} buffer over {TERM_YEARS} years, priced on the September 2026 surface "
          f"at a forward carrying the chain's {CHAIN_DIVIDEND_YIELD:.2%} dividend yield")
    for spread in FUNDING_SPREADS:
        block = funding[funding["funding_spread"] == spread]
        available = float(block["funding_available_pct_of_premium"].iloc[0])
        binding = block[block["cap_binds"]]
        print(f"    Treasury plus {spread:.2%} frees up {available:.2f}% of premium; the credit "
              f"costs {block['credit_cost_pct_of_premium'].min():.2f} to "
              f"{block['credit_cost_pct_of_premium'].max():.2f}% across the swept caps, so the cap "
              + ("binds somewhere on the grid" if len(binding) else "never binds"))
    print("  The break-even cap does not exist at these rates, so it is swept rather than solved: "
          "the contract is written at a margin and the cap is competition, not necessity.")
    cap = HEADLINE_CAP

    print(f"\nThe segment's own equity exposure at a {cap:.0%} cap, per unit of account value",
          flush=True)
    surface = exposure_surface(market, cap)
    surface.to_csv(paths.TABLES / "rila_exposure_surface.csv", index=False, float_format="%.6f")
    wide = surface.pivot_table(index="elapsed_years", columns="index_since_issue",
                               values="equity_exposure_pct_of_account")
    print("  rows are years elapsed, columns the index since issue")
    print("  " + "        ".join(f"{c:.2f}" for c in wide.columns))
    for elapsed, line in wide.iterrows():
        print(f"  {elapsed}  " + "  ".join(f"{v:+.3f}" for v in line))
    peak = surface.loc[surface["equity_exposure_pct_of_account"].idxmax()]
    trough = surface.loc[surface["equity_exposure_pct_of_account"].idxmin()]
    print(f"  highest {peak['equity_exposure_pct_of_account']:.3f} at {int(peak['elapsed_years'])} "
          f"years elapsed and an index ratio of {peak['index_since_issue']:.2f}; lowest "
          f"{trough['equity_exposure_pct_of_account']:.3f} at "
          f"{int(trough['elapsed_years'])} years and {trough['index_since_issue']:.2f}, "
          f"{trough['share_capped']:.0%} capped")

    print("\nCohorts rolled along realised index history, weighted by filed issuance", flush=True)
    stats = pd.read_csv(paths.BOOK_STATISTICS, comment="#")
    book = pd.concat([cohort_book(market, cap, history, date, stats)
                      for date in VALUATION_DATES], ignore_index=True)
    book.to_csv(paths.TABLES / "rila_cohort_book.csv", index=False, float_format="%.6f")
    for as_of, cohorts in book.groupby("as_of"):
        inside = cohorts[cohorts["weight"] > 0]
        print(f"  {as_of}: {len(inside)} cohorts in the book, exposure "
              f"{weighted(cohorts, 'equity_exposure_pct_of_account'):.3f} per unit of account "
              f"value")

    print("\nDoes the model's level agree with the filed one", flush=True)
    levels = level_check(book, stats)
    levels.to_csv(paths.TABLES / "rila_level_check.csv", index=False, float_format="%.6f")
    for _, row in levels.iterrows():
        print(f"  {row['as_of']}: model {row['model_embedded_derivative_pct']:5.1f}% of account "
              f"value against {row['disclosed_embedded_derivative_pct']:5.1f}% filed, on a book "
              f"{row['rila_funds_over_va_account']:.1%} the size of the variable annuity book")

    print("\nAgainst what the filings disclose", flush=True)
    guarantee, absorbed = disclosed_guarantee_sensitivity(), disclosed_absorption()
    swept = []
    for swept_cap in CAP_GRID:
        at_cap = netting(pd.concat([cohort_book(market, swept_cap, history, date, stats)
                                    for date in VALUATION_DATES], ignore_index=True),
                         guarantee, absorbed, stats)
        swept.append(at_cap.assign(cap=swept_cap))
    sweep = pd.concat(swept, ignore_index=True)
    sweep.to_csv(paths.TABLES / "rila_netting_by_cap.csv", index=False, float_format="%.6f")
    table = netting(book, guarantee, absorbed, stats)
    table.to_csv(paths.TABLES / "rila_netting.csv", index=False, float_format="%.6f")
    for _, row in table.iterrows():
        print(f"  {row['as_of']} {row['shock']:<18} filed absorption "
              f"{row['disclosed_absorbed_share']:7.2%}; a book of the observed "
              f"{row['observed_rila_over_va']:.1%} would absorb "
              f"{row['model_absorbed_at_observed_size']:7.1%}; filed over model "
              f"{row['filed_over_model']:7.3f}")

    span = surface["equity_exposure_pct_of_account"]
    print(f"\n  Across every term position and index level a segment can occupy, and every cap on "
          f"the swept grid, the exposure spans a factor of "
          f"{sweep['model_rila_exposure_per_unit'].max() / sweep['model_rila_exposure_per_unit'].min():.1f} "
          f"as a book and {span.max() / span.min():.1f} as a single segment. The filed line's "
          f"equity sensitivity moved by a factor of 330 between two consecutive year-ends.")

    print(f"\nwrote six tables to {paths.TABLES}")


if __name__ == "__main__":
    main()
