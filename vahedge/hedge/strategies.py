"""The seven hedge programmes being compared, as data rather than as code branches.

A strategy here is a choice of instruments, a rebalancing rule, a hedge ratio and a weighting.
Writing them as one table rather than as seven code paths is what makes the comparison fair:
every strategy runs through the same simulator, so a difference in the result is a difference in
the strategy and not in how it was implemented.

What each one is for, in the order they add things:

S0 is unhedged, and it is the only number the rest are measured against.

S1 is equity futures alone - the hedge most people mean by "delta hedging". It leaves the entire
rate exposure open, which on a forty-year guarantee is not a detail.

S2 adds rate instruments. Jackson's own book is mostly S2 by notional, and the pair of
exposures it closes are the two the disclosed sensitivity table reports.

S3 adds puts, which is the first strategy with any convexity at all. It is also the first with a
premium bill, and the experiments exist to price that trade.

S4 is S3 rebalanced on a band rather than a calendar. The band is the cost-control lever: a
hedge that trades only when it has drifted spends less and carries more risk between trades.

S5 is a deliberately partial hedge. Jackson does not hedge to zero and says so; the interesting
question is what the last ten per cent of coverage costs.

S6 adds a long-dated out-of-the-money put spread on top of S3 - the macro hedge. It is not there
to flatten the Greeks week to week but to cap the statutory and rating-agency damage in a
severe fall, which is a capital question rather than a hedging one, and W5 is where it is judged.

The total return swap is not in any default strategy and is in the library on purpose: E3
swaps it in for futures to show that the ranking of the two depends entirely on rebalancing
frequency, because one costs a round trip and the other costs a running spread.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import instruments as inst
from .sizing import DEFAULT_WEIGHTS

# Strike and tenor of the puts in S3, picked off the sweep in E3 rather than assumed, and a
# trade-off rather than a winner. The first version held a one-year put ten per cent down with
# nothing behind the choice. Against it, six months at the same strike costs 13.7 per cent of
# account value over the ten-year window instead of 21.2 and leaves a slightly smaller average
# residual, 0.137 against 0.142 per cent a day, because a listed option's spread is charged on
# its vega while the hedge is bought for its gamma and that ratio rises as the tenor shortens.
# What it gives up is the tail: through covid the one-year put returned 3.3 per cent of account
# value against the six-month's 1.9. So the short tenor buys a third off the running cost for
# about a point and a half of crisis upside, which is the trade taken here and is reversible by
# changing these two numbers. The whole grid is on the frontier once the crisis outcome is an
# axis, and E3 prints it. Jackson's own equity option book averaged 0.24 years of remaining
# term at 31 December 2025, shorter than either.
PUT_TENOR_YEARS = 0.5
PUT_STRIKE = 0.90
# The macro hedge in S6: a two-year put spread well out of the money. A spread rather than a
# put because the far wing is where the premium goes and the floor it buys is below the level
# at which the capital question has already been decided.
MACRO_TENOR_YEARS = 2.0
MACRO_STRIKES = (0.80, 0.60)
# Notional of the macro spread as a share of account value. A quarter means a twenty per cent
# fall pays nothing, a thirty per cent fall pays about two and a half per cent of the account,
# and the payout caps at five per cent below forty per cent down. That is a capital floor
# rather than an economic hedge, which is the distinction W5 is built to measure.
MACRO_NOTIONAL_SHARE = 0.25
# S4's band, as a share of account value of delta left open. Sized to be a cost-control device
# rather than a disguise for daily rebalancing: the liability's curvature is around a quarter of
# the account value, so a one per cent index move opens about a quarter of a per cent of delta,
# and a band of two per cent takes a sustained move of several days to breach. The first version
# used a quarter of a per cent, which fired most days and made S4 cost twice what weekly
# rebalancing did - the opposite of what the strategy is for. Swept in E3.
BAND_SHARE = 0.02


@dataclass(frozen=True)
class Strategy:
    """One hedge programme."""

    name: str
    description: str
    instruments: tuple = ()
    rebalance: str = "weekly"          # daily, weekly, monthly, or band
    band: float = 0.0                  # fraction of account value of hedge error, if band
    hedge_ratio: float = 1.0
    vega: str = "current"              # which vega to size against; see sizing.insurer_exposures
    weights: np.ndarray = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())
    # Positions held outside the Greek solve, each as a share of account value of notional.
    # The macro hedge belongs here rather than among the solved instruments. Handing a deep
    # put spread to a least-squares fit invites it to exploit the near-collinearity of the two
    # legs: the first version did exactly that, taking fifty units long against seventy-six
    # short to manufacture gamma, at a hundred and thirty times the account value in notional.
    # A tail hedge is bought in a size, for a reason, and held.
    overlay: tuple = ()

    def with_rebalance(self, rebalance: str, band: float = 0.0) -> "Strategy":
        return replace(self, rebalance=rebalance, band=band)


def equity_only() -> tuple:
    return (inst.EquityFuture(),)


def rate_instruments() -> tuple:
    """One, not three, and the reason is the rate model rather than the instruments.

    A note future, a bond forward and a receive-fixed swap are three different trades with
    three different costs and liquidity. Under a one-factor short-rate model they are not three
    different risks: each one's only exposure is to a parallel shift, so their exposure columns
    are exactly proportional and a strategy holding two of them has a singular design matrix.
    The split between them is then decided by the ridge rather than by anything financial, which
    is false precision dressed as optimisation.

    So the strategies hold the swap, whose sensitivity is the annuity of its own fixed leg read
    off the curve rather than a duration assumption, and the other two stay in the library for
    the derivative-book reproduction in V1, where matching Jackson's disclosed categories by
    notional is the point. Telling the three apart as risks would need a second curve factor in
    the proxy's state, and the proxy does not carry one; that is a stated limitation rather than
    something the hedge quietly papers over.
    """
    return (inst.InterestRateSwap(tenor=10.0, receive_fixed=True),)


def put_leg(tenor: float = PUT_TENOR_YEARS, strike: float = PUT_STRIKE) -> tuple:
    """The convexity leg. Both arguments are swept in E3 rather than argued for.

    The tenor is the single largest lever on what convexity costs, because a listed option's
    spread is charged on its vega while the hedge wants its gamma, and short-dated options carry
    far more of the second per unit of the first. The defaults come from the sweep; see the
    constants above.
    """
    return (inst.IndexPut(maturity=tenor, strike_over_spot=strike),)


def macro_leg(notional_share: float = MACRO_NOTIONAL_SHARE,
              strikes: tuple = MACRO_STRIKES,
              tenor: float = MACRO_TENOR_YEARS) -> tuple:
    """Long the far put, short the further one: a spread, bought for the capital floor.

    Returned as (instrument, units as a share of account value) pairs, because this leg is held
    in a size rather than solved for. Long the nearer strike and short the further one in equal
    size, which is what makes it a spread and caps both the payout and the premium.
    """
    return (
        (inst.IndexPut(maturity=tenor, strike_over_spot=strikes[0]), notional_share),
        (inst.IndexPut(maturity=tenor, strike_over_spot=strikes[1]), -notional_share),
    )


def matrix() -> dict:
    """The strategy matrix, keyed by name."""
    delta_and_rho = equity_only() + rate_instruments()
    with_puts = delta_and_rho + put_leg()
    return {
        "S0": Strategy(
            name="S0", description="unhedged baseline", instruments=(), rebalance="weekly",
        ),
        "S1": Strategy(
            name="S1", description="equity futures only, delta",
            instruments=equity_only(),
        ),
        "S2": Strategy(
            name="S2", description="futures and rate instruments, delta and rho",
            instruments=delta_and_rho,
        ),
        "S3": Strategy(
            name="S3",
            description=(f"adds {PUT_TENOR_YEARS:g}-year {1 - PUT_STRIKE:.0%} out-of-the-money "
                         "puts for gamma and vega"),
            instruments=with_puts,
        ),
        "S4": Strategy(
            name="S4", description="S3 rebalanced on a hedge-error band rather than a calendar",
            instruments=with_puts, rebalance="band", band=BAND_SHARE,
        ),
        "S5": Strategy(
            name="S5", description="S3 at a 90% hedge ratio",
            instruments=with_puts, hedge_ratio=0.90,
        ),
        "S6": Strategy(
            name="S6", description="S3 plus a two-year 80/60 macro put spread held outside the solve",
            instruments=with_puts, overlay=macro_leg(),
        ),
    }


def rebalance_dates(dates, rebalance: str) -> np.ndarray:
    """Boolean mask over ``dates`` marking the calendar rebalances.

    Band rebalancing is not a calendar rule and cannot be decided in advance: it depends on how
    far the hedge has drifted, which is only known as the simulation runs. A band strategy
    therefore gets every date marked here and the simulator applies the band itself.
    """
    import pandas as pd

    index = pd.DatetimeIndex(dates)
    if rebalance in ("daily", "band"):
        return np.ones(index.size, dtype=bool)
    if rebalance == "weekly":
        week = index.isocalendar()
        changed = np.ones(index.size, dtype=bool)
        key = list(zip(week["year"], week["week"]))
        changed[1:] = [current != previous for current, previous in zip(key[1:], key[:-1])]
        return changed
    if rebalance == "monthly":
        key = list(zip(index.year, index.month))
        changed = np.ones(index.size, dtype=bool)
        changed[1:] = [current != previous for current, previous in zip(key[1:], key[:-1])]
        return changed
    raise ValueError(f"unknown rebalance rule {rebalance!r}")
