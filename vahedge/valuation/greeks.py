"""Greeks by bump and revalue, on common random numbers, with the noise reported.

A Greek is a difference between two valuations. At twenty thousand paths the standard error on
the level of this book is around a tenth of a percent of account value, while a one basis point
rate bump moves it by a few thousandths. Taking that difference across independent simulations
would produce a number that is almost entirely noise and would look perfectly reasonable. Every
bump here therefore reuses the base valuation's seed, so the two runs share their draws and the
level noise cancels.

Reusing the seed is necessary and not sufficient, so the standard error of each Greek is
computed from the paired path-level differences rather than from the two levels, and reported
next to it. A Greek whose standard error is a third of its value is not a risk number, and the
only way to know is to print it.

Two conventions, because they decide whether a hedge is long or short.

*Equity exposure is the derivative with respect to the log index level*, which is the dollar
notional a hedge has to carry, not a dimensionless sensitivity. For the market risk benefit as
Jackson reports it - a liability when positive - this comes out negative, so the hedge is a
short index position. That is the right answer for something that behaves like a written put.

*Rho is per basis point of parallel shift in the fitted curve* and is also negative, because a
long-dated liability discounts away faster when rates rise.

Vega is split. The current variance is what trades and what moves week to week; the long-run
level is an assumption no listed option reaches. A single vega that moves both at once hides
which one the number came from, and on a forty-year liability the long-run level does most of
the work. Both are reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# How wide a move the reported curvature is averaged over, as a log move in the index. Every
# second difference in the project uses this one number, and the reason it lives here rather
# than in whichever module needed it first is that a gamma is only a hedgeable quantity if the
# liability and the instrument sold against it are measured the same way. They were not: the
# regression proxy averaged over ten per cent while the listed puts used two, which is a three
# to nine per cent difference in the put's gamma depending on strike and tenor, and a solve
# matching one against the other is sizing a position off a unit mismatch. Ten rather than two
# because the liability's own second difference at two per cent is mostly noise, and because a
# convexity hedge on a guarantee is bought for moves of that order rather than for the limit.
#
# The full-repricing route below kept its own two per cent for a while after the proxy and the
# put were moved onto this constant, which left the disclosure comparison solving a liability
# gamma measured one way against instrument gammas measured another - the same unit mismatch, in
# the half of the project that was supposed to have been fixed. Both routes read it now.
EQUITY_CURVATURE_STEP = 0.10


@dataclass(frozen=True)
class Greeks:
    """Sensitivities of the book, in the units a hedge is sized in."""

    value: float
    account_value: float
    equity_exposure: float          # dV / d(ln S), dollars
    equity_gamma: float             # d2V / d(ln S)^2, dollars
    rho_per_bp: float               # dV / d(parallel shift), dollars per basis point
    vega_current: float             # dV / d(current volatility), dollars per point
    vega_long_run: float            # dV / d(long-run volatility), dollars per point
    equity_std_error: float
    rho_std_error: float
    vega_std_error: float
    level_std_error: float

    @property
    def equity_exposure_pct_of_account(self) -> float:
        return self.equity_exposure / self.account_value

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                ("value", self.value, self.level_std_error),
                ("equity exposure per log index", self.equity_exposure, self.equity_std_error),
                ("equity gamma", self.equity_gamma, np.nan),
                ("rho per basis point", self.rho_per_bp, self.rho_std_error),
                ("vega per point, current variance", self.vega_current, self.vega_std_error),
                ("vega per point, long-run level", self.vega_long_run, np.nan),
            ],
            columns=["measure", "value", "std_error"],
        )


def compute(
    valuer,
    book,
    state,
    attribution,
    equity_bump: float = 0.01,
    gamma_bump: float = EQUITY_CURVATURE_STEP,
    rate_bump_bp: float = 10.0,
    vol_bump: float = 0.01,
) -> Greeks:
    """Central differences in the index, the curve and volatility.

    The rate bump defaults to ten basis points rather than one. One basis point is the textbook
    choice and it is the wrong one here: the difference it produces is small enough that the
    paired Monte Carlo error is a material share of it, and the liability's convexity over ten
    basis points is negligible. Ten trades a little bias for a lot of variance and the trade is
    reported, because ``rho_std_error`` makes it checkable rather than a matter of taste.
    """
    base = valuer.value(book, state, attribution=attribution)

    up = valuer.value(book, state, attribution=attribution, equity_shock=equity_bump)
    down = valuer.value(book, state, attribution=attribution, equity_shock=-equity_bump)
    # A proportional bump of h is a move of ln((1+h)/(1-h)) in log space, not 2h. At one percent
    # the two differ by a hundredth of a percent, and using the exact denominator costs nothing.
    log_span = np.log((1.0 + equity_bump) / (1.0 - equity_bump))
    equity_exposure = (up.market_risk_benefit - down.market_risk_benefit) / log_span

    gamma_up = valuer.value(book, state, attribution=attribution, equity_shock=gamma_bump)
    gamma_down = valuer.value(book, state, attribution=attribution, equity_shock=-gamma_bump)
    gamma_h = np.log(1.0 + gamma_bump)
    equity_gamma = (
        gamma_up.market_risk_benefit
        - 2.0 * base.market_risk_benefit
        + gamma_down.market_risk_benefit
    ) / gamma_h**2

    rate_up = valuer.value(book, state.with_shocks(rate_shock_bp=rate_bump_bp),
                           attribution=attribution)
    rate_down = valuer.value(book, state.with_shocks(rate_shock_bp=-rate_bump_bp),
                             attribution=attribution)
    rho = (rate_up.market_risk_benefit - rate_down.market_risk_benefit) / (2.0 * rate_bump_bp)

    vol_up = valuer.value(book, state.with_shocks(vol_shock=vol_bump), attribution=attribution)
    vol_down = valuer.value(book, state.with_shocks(vol_shock=-vol_bump), attribution=attribution)
    vega_both = (vol_up.market_risk_benefit - vol_down.market_risk_benefit) / (2.0 * vol_bump)

    # Splitting the vega: move the current variance alone, then attribute the remainder of the
    # combined move to the long-run level.
    from dataclasses import replace
    current_only = replace(
        state,
        heston=replace(
            state.heston, v0=max((np.sqrt(state.heston.v0) + vol_bump) ** 2, 1e-8)
        ),
    )
    vega_current = (
        valuer.value(book, current_only, attribution=attribution).market_risk_benefit
        - base.market_risk_benefit
    ) / vol_bump
    vega_long_run = vega_both - vega_current

    return Greeks(
        value=base.market_risk_benefit,
        account_value=base.account_value,
        equity_exposure=equity_exposure,
        equity_gamma=equity_gamma,
        rho_per_bp=rho,
        vega_current=vega_current * 0.01,
        vega_long_run=vega_long_run * 0.01,
        equity_std_error=_paired_error(up, down, book, attribution) / log_span,
        rho_std_error=_paired_error(rate_up, rate_down, book, attribution) / (2.0 * rate_bump_bp),
        vega_std_error=_paired_error(vol_up, vol_down, book, attribution) / (2.0 * vol_bump) * 0.01,
        level_std_error=base.std_error,
    )


def _paired_error(up, down, book, attribution) -> float:
    """Standard error of a bumped difference, taken across paired paths.

    The two valuations share their draws, so the error of the difference is the error of the
    per-path difference, which is far smaller than either level's. Computing it from the levels
    instead would overstate it by an order of magnitude and would make every Greek look
    worthless.
    """
    weight = book.weight
    attribution = np.asarray(attribution, dtype=float)
    difference = (
        np.einsum("c,cp->p", weight, up.projection.claim_paths - down.projection.claim_paths)
        - np.einsum(
            "c,cp->p", weight * attribution,
            up.projection.fee_paths - down.projection.fee_paths,
        )
    )
    n = difference.size
    if n % 2 == 0:
        half = n // 2
        pairs = 0.5 * (difference[:half] + difference[half:])
        return float(pairs.std(ddof=1) / np.sqrt(half))
    return float(difference.std(ddof=1) / np.sqrt(n))


def disclosed_shocks(
    valuer,
    book,
    state,
    attribution,
    equity_shocks=(0.10, -0.10),
    rate_shocks_bp=(100, -100, 50, -50),
) -> pd.DataFrame:
    """Reprice under the shocks Item 7A runs, as full repricings rather than delta times shock.

    The disclosed shocks are large enough that convexity shows up in them, and the asymmetry
    between the up and the down case is the scale-free statistic the validation is built on.
    Approximating them with a delta would throw away exactly the thing being tested.
    """
    base = valuer.value(book, state, attribution=attribution)
    rows = [{"shock": "base", "value": base.market_risk_benefit, "change": 0.0,
             "change_pct_of_account": 0.0}]

    for shock in equity_shocks:
        shocked = valuer.value(book, state, attribution=attribution, equity_shock=shock)
        change = shocked.market_risk_benefit - base.market_risk_benefit
        rows.append({
            "shock": f"equity_{'up' if shock > 0 else 'down'}_{abs(int(round(shock * 100)))}pct",
            "value": shocked.market_risk_benefit,
            "change": change,
            "change_pct_of_account": change / base.account_value,
        })

    for basis_points in rate_shocks_bp:
        shocked = valuer.value(book, state.with_shocks(rate_shock_bp=basis_points),
                               attribution=attribution)
        change = shocked.market_risk_benefit - base.market_risk_benefit
        rows.append({
            "shock": f"rates_{'up' if basis_points > 0 else 'down'}_{abs(int(basis_points))}bp",
            "value": shocked.market_risk_benefit,
            "change": change,
            "change_pct_of_account": change / base.account_value,
        })

    return pd.DataFrame(rows)


def asymmetry(frame: pd.DataFrame, risk: str = "equity", size: int = 10) -> float:
    """The down impact over the up impact, which is scale-free and therefore comparable.

    This is the statistic V2 is built on. A model of one stylised book cannot match Jackson's
    dollars, and does not try to. What it can match is the shape: how much more the liability
    moves on the way down than on the way up, and how that ratio changes as the book moves out
    of the money.
    """
    suffix = "pct" if risk == "equity" else "bp"
    up = frame.loc[frame["shock"] == f"{risk}_up_{size}{suffix}", "change"]
    down = frame.loc[frame["shock"] == f"{risk}_down_{size}{suffix}", "change"]
    if up.empty or down.empty:
        raise ValueError(f"no {risk} shock of size {size} in the repricing table")
    return float(abs(down.iloc[0]) / abs(up.iloc[0]))
