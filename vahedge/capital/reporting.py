"""The reported market risk benefit, and why a working hedge still leaves earnings moving.

Jackson states the position in Item 7A of the FY2025 10-K: it does not use hedging to offset the
movement in its US GAAP liabilities as market conditions change, and that has resulted in net
income volatility. The hedge targets an internally modelled economic liability. The reported
market risk benefit is a different number on a different basis, and a hedge sized on the first
does not neutralise the second. This measures how much of the gap is which.

Three liabilities on the same contract, the same paths and the same day, because the split needs
all three and no two of them answer the question:

1. **Economic.** Best-estimate mortality, Treasury discounting. What the hedge is sized on and
   what every other workstream means by the market risk benefit.
2. **Reporting without own credit.** The margin-loaded mortality table, Treasury discounting.
   The difference from (1) is the risk margin, which is a level difference that moves with the
   market and therefore reaches net income.
3. **Reporting with own credit.** The same, discounted at Treasury plus the insurer's own
   non-performance spread. This is the reported figure. The difference from (2) is the
   own-credit adjustment, and under ASU 2018-12 that piece goes to other comprehensive income
   rather than through net income.

So reported net income moves with (2) and the OCI line moves with (3) minus (2), while the hedge
was sized on (1). Two separate mismatches, and separating them is the point: one is an
assumption difference the insurer chose and the other is an accounting rule it did not.

**Why the margins come from the mortality table rather than from a loading.** Note 6 says the
fair value uses best estimate assumptions plus risk margins, and the NAIC-adopted annuity table
ships in exactly those two versions: the 2012 IAM Basic table is best estimate, and the Period
table is the Basic table with the margins the Life Actuarial Task Force set. Valuing the same
contract on both gives two liabilities whose difference is a margin someone else calibrated,
rather than a percentage this project invented. The margins lengthen life, so the living benefit
gets dearer and the death benefit cheaper, and which way the net goes is a result rather than an
assumption.

**How the spread enters.** Not by shifting the curve the paths are simulated under, which would
move the equity drift as well and price a different contract. The liability's cash flows are
discounted at the risk-free rate times ``exp(-spread * t)`` on the same paths, so the dynamics
stay risk-neutral and only the discounting moves. The spread itself is the Baa index scaled to
the insurance subsidiaries' own credit, which `vahedge/market/scenarios.py` explains, and it moves
daily - 0.8 to 2.6 per cent across the replay window, and 1.3 to 2.6 through covid alone - so the
liability is fitted at a grid of spread levels and read off by interpolation, rather than at one
level and extrapolated.

**Both differences flip sign when the market risk benefit does.** On this contract the benefit
starts as a small net liability and ends the replay window as a large net asset, and discounting
an asset at a higher rate shrinks it in magnitude rather than growing it. So the own-credit
adjustment runs from -7.1 to +1.5 of premium across the decade and the risk margin from +1.7 to
-0.5. Neither is a bug and neither is a sign convention: the margin makes the living benefit
dearer and the death benefit cheaper, and which way the net goes depends on whether the net is a
liability at all.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from ..hedge.simulator import daily_profit
from ..liability import gmwb, mortality
from ..market.simulate import MarketPaths
from ..valuation import lsmc

# Spread levels the reporting-basis liability is fitted at, as decimals. The replay window's own
# BAA10Y spread runs 1.36 to 4.31 per cent and the own-credit share of it is 0.82 to 2.59, so the
# grid brackets that and the zero member doubles as the no-own-credit basis. Three interior points
# would buy very little: the liability is close to linear in the spread over a range this narrow,
# and each extra level is another regression fit over twenty thousand paths.
SPREAD_GRID = (0.0, 0.013, 0.027)


def discounted_at_spread(paths: MarketPaths, spread: float) -> MarketPaths:
    """The same paths with the discount factor loaded by a flat spread.

    Only the discount factor changes. The equity and fund growth, the variance and the short rate
    are the objects the contract's own mechanics run on and they belong to the risk-neutral
    measure; loading them too would value a different contract in a different world and call the
    difference own credit.
    """
    if spread == 0.0:
        return paths
    years = np.arange(1, paths.n_years + 1, dtype=float)
    return replace(paths, discount=paths.discount * np.exp(-spread * years)[None, :])


@dataclass(frozen=True)
class ReportingBasis:
    """The fitted liabilities behind the three marks, and the grid the spread is read off."""

    economic: lsmc.ProxyFit
    reporting: dict                 # spread level -> ProxyFit on the margin-loaded table
    attribution: float

    @property
    def spreads(self) -> np.ndarray:
        return np.sort(np.array(list(self.reporting), dtype=float))

    def value_at(self, fit, **state) -> float:
        return float(np.ravel(fit.greeks_at(attribution=self.attribution, **state)["value"])[0])

    def marks(self, **state) -> dict:
        """The three liabilities at one state, with the spread interpolated across the grid."""
        spread = float(state.pop("own_credit_spread"))
        grid = self.spreads
        at_grid = [self.value_at(self.reporting[level], **state) for level in grid]
        return {
            "economic": self.value_at(self.economic, **state),
            # The zero-spread member is the no-own-credit basis by construction rather than by a
            # second calculation, which is why the grid starts there.
            "reporting_ex_own_credit": at_grid[0],
            "reporting": float(np.interp(spread, grid, at_grid)),
            "own_credit_spread": spread,
        }


def fit_bases(
    book,
    state,
    market_paths: MarketPaths,
    attribution: float = 1.0,
    spread_grid=SPREAD_GRID,
) -> ReportingBasis:
    """Fit the economic basis and the reporting basis at each spread level, on one simulation.

    One simulation for all of them, which is what makes this affordable: the paths do not depend
    on the mortality table or on the discount spread, so what has to be repeated is the
    projection and the regression rather than the Monte Carlo. Four fits instead of one.
    """
    horizon = int(book.projection_years.max())
    fits = {}
    for basis in ("basic", "period"):
        survival, deaths = mortality.load(basis).rates(
            book.attained_age, state.valuation_year, horizon, 0.5
        )
        levels = (0.0,) if basis == "basic" else tuple(spread_grid)
        for spread in levels:
            projection = gmwb.project(
                book, discounted_at_spread(market_paths, spread), survival, deaths,
                equity_weight=state.mix.equity_weight, record=True,
            )
            fits[(basis, spread)] = lsmc.fit(projection)
    return ReportingBasis(
        economic=fits[("basic", 0.0)],
        reporting={spread: fits[("period", spread)] for spread in spread_grid},
        attribution=attribution,
    )


def mark(ledger: pd.DataFrame, path, bases: ReportingBasis) -> pd.DataFrame:
    """Re-mark a hedge run's ledger on all three bases, day by day.

    The hedge positions in the ledger were sized on the economic basis, which is the whole
    exercise. Nothing here changes a trade; it only asks what the same book was worth on the two
    reporting bases while those trades were being put on.
    """
    if path.own_credit_spread is None:
        raise ValueError("the path carries no own-credit spread; load_history needs BAA10Y")
    if path.dates.size != ledger.shape[0]:
        raise ValueError(f"path has {path.dates.size} dates, ledger has {ledger.shape[0]} rows")

    rows = []
    for day, (_, row) in enumerate(ledger.iterrows()):
        rows.append(bases.marks(
            years_since_issue=float(row["policy_year"]),
            account_value=float(row["account_value"]),
            benefit_base=float(row["benefit_base"]),
            variance=float(row["variance"]),
            zero_10y=float(row["zero_10y"]),
            own_credit_spread=float(path.own_credit_spread[day]),
        ))

    out = pd.DataFrame(rows, index=ledger.index)
    out["risk_margin"] = out["reporting_ex_own_credit"] - out["economic"]
    out["own_credit_adjustment"] = out["reporting"] - out["reporting_ex_own_credit"]
    out["hedge_mark"] = ledger["hedge_mark"]
    out["cash"] = ledger["cash"]

    # Net worth on each basis, then the period movements. Reported net income carries the
    # movement in the liability excluding own credit; the own-credit movement is the OCI line.
    out["economic_net_worth"] = out["cash"] + out["hedge_mark"] - out["economic"]
    out["reported_net_worth"] = (
        out["cash"] + out["hedge_mark"] - out["reporting_ex_own_credit"]
    )
    out["economic_pnl"] = daily_profit(out["economic_net_worth"], -out["economic"].iloc[0])
    out["net_income"] = daily_profit(out["reported_net_worth"],
                                     -out["reporting_ex_own_credit"].iloc[0])
    # The own-credit adjustment is not new on the first day - the liability and its spread both
    # existed before the hedge did - so the opening movement in it really is zero.
    out["oci"] = -out["own_credit_adjustment"].diff().fillna(0.0)
    out["comprehensive_income"] = out["net_income"] + out["oci"]
    return out


def summarise(marked: pd.DataFrame, account_value: float) -> dict:
    """Headline figures as shares of the starting account value."""
    def total(column: str) -> float:
        return float(marked[column].sum() / account_value)

    def sd(column: str) -> float:
        return float(marked[column].std(ddof=1) / account_value)

    economic_sd = sd("economic_pnl")
    return {
        "days": float(marked.shape[0]),
        "economic_total_pct": total("economic_pnl"),
        "net_income_total_pct": total("net_income"),
        "oci_total_pct": total("oci"),
        "comprehensive_total_pct": total("comprehensive_income"),
        "economic_sd_pct": economic_sd,
        "net_income_sd_pct": sd("net_income"),
        "oci_sd_pct": sd("oci"),
        "comprehensive_sd_pct": sd("comprehensive_income"),
        # How much worse the reported series is than the one the hedge was aimed at. Above one
        # means the hedge left more earnings volatility than it removed economic volatility.
        "net_income_sd_multiple": (sd("net_income") / economic_sd
                                   if economic_sd > 0 else np.nan),
        "mean_risk_margin_pct": float(marked["risk_margin"].mean() / account_value),
        "mean_own_credit_pct": float(marked["own_credit_adjustment"].mean() / account_value),
        "worst_net_income_day_pct": float(marked["net_income"].min() / account_value),
        "worst_economic_day_pct": float(marked["economic_pnl"].min() / account_value),
    }
