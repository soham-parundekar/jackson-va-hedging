"""Daily paths for the hedge backtest: what the market actually did, and what it might do.

A hedging result is only as honest as the path it was run on. Three kinds are built here.

*Replays* are the real thing - the index, the Treasury curve and the volatility surface as they
were, day by day. They are the only paths with real correlation structure in them, including the
ones nobody models: a crash arriving with rates falling, volatility tripling in a week, or 2022,
where equity and rates fell together and every hedge that assumed they would not lost twice.

*Real-world Monte Carlo* paths use the same models with a drift under the real-world measure
instead of the risk-neutral one, which is the right measure for asking what a hedge is likely to
cost. Mixing them up - valuing under the real-world drift, or projecting a hedge programme under
the risk-neutral one - is the most common way to get a hedging study wrong by a wide margin.

*Stationary bootstrap* paths resample blocks of realised history, which keeps the volatility
clustering and the fat tails that a parametric generator smooths away, at the cost of inventing
nothing new. The block length is the one parameter and is set from the autocorrelation of
squared returns rather than chosen.

Mapping the real market onto the model's state is the part that takes care, and two of the three
state variables need work.

*Volatility.* The model's state is an instantaneous variance; what is quoted is a thirty-day or
six-month implied volatility. Under Heston the expected average variance over a window has a
closed form, so the quoted number can be inverted into the instantaneous variance that would
produce it. That inversion is exact given the model; what it assumes is that the quoted index is
the fair variance of a log contract, which is what the volatility index methodology is built to
be, give or take the discretisation of its own strike grid.

*The curve.* Par yields are bootstrapped to continuously compounded zeros on each date, with the
same code the valuation date uses. Nothing is fitted smoothly here: the backtest needs a
ten-year zero, discount factors and a swap annuity, and linear interpolation between quoted
tenors gives all three without the extrapolation risk a six-parameter fit carries.

*The sub-account.* Mirrors the simulator exactly - a rebalanced portfolio of an equity sleeve, a
constant-maturity bond sleeve and cash - because the proxy was fitted on contract values that
grew that way. Building it any other way here would show up as basis risk that is really a
specification mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .curves import ZeroCurve, bootstrap
from .state import PAR_SERIES

TRADING_DAYS = 252.0
VIX_TENOR_YEARS = 30.0 / 365.0
VIX6M_TENOR_YEARS = 0.5
# Share of the Baa index taken as the insurer's own non-performance spread. Jackson National
# Life is rated several notches above Baa, so the index overstates what the market would charge
# on its claims-paying credit, and ASU 2018-12's own-credit adjustment is about that credit
# rather than about the average Baa issuer. No public series prices the subsidiary's own spread
# across this decade, so the share is a stated assumption, not an estimate - the figure it most
# affects is the own-credit line in the reporting lens, and docs/limitations.md says so.
OWN_CREDIT_SHARE = 0.6

# Stress windows, with the reason each one is in the list. Start and end are inclusive. Which
# of these a run can actually use depends on how far back the index series reaches; the loader
# reports the ones it had to drop rather than silently shortening them.
EPISODES = {
    "dot-com unwind": ("2000-03-24", "2002-10-09", "a slow three-year decline rather than a crash"),
    "global financial crisis": ("2007-10-09", "2009-03-09", "equity and rates both collapse"),
    "august 2011": ("2011-07-22", "2011-10-03", "a fast fall with a rates rally into it"),
    "volmageddon": ("2018-01-26", "2018-02-09", "a volatility shock with little index damage"),
    "fourth quarter 2018": ("2018-09-20", "2018-12-24", "a grinding fall into year end"),
    "covid": ("2020-02-19", "2020-03-23", "the fastest fall in the sample, rates to zero"),
    "2022 double": ("2022-01-03", "2022-10-14", "equity and rates fall together, the hard one"),
}


@dataclass(frozen=True)
class DailyPath:
    """One scenario, daily, in the units the hedge simulator and the proxy both want."""

    dates: pd.DatetimeIndex
    index: np.ndarray              # equity total-return index, rebased to 1 at the start
    fund: np.ndarray               # sub-account total return index, rebased to 1
    curves: list                   # one ZeroCurve per date
    zero_10y: np.ndarray
    implied_vol: np.ndarray        # at the reference tenor, decimal
    variance: np.ndarray           # Heston instantaneous variance implied by the quote
    cash_rate: np.ndarray          # overnight financing, decimal
    # Own non-performance spread over the ten-year Treasury, decimal. The economic valuation
    # never touches it; the reporting basis discounts at the Treasury curve plus this, because
    # a market risk benefit is a fair value and a fair value of the insurer's own obligation
    # includes the insurer's own credit. Its movement is what goes to other comprehensive
    # income rather than through net income.
    own_credit_spread: np.ndarray | None = None
    label: str = "path"

    def __len__(self) -> int:
        return int(self.dates.size)

    @property
    def year_fraction(self) -> np.ndarray:
        """Years elapsed since the first date, on an actual/365 basis."""
        return (self.dates - self.dates[0]).days.to_numpy(dtype=float) / 365.0

    def window(self, start, end, label: str | None = None) -> "DailyPath":
        keep = (self.dates >= pd.Timestamp(start)) & (self.dates <= pd.Timestamp(end))
        if keep.sum() < 2:
            raise ValueError(f"{start} to {end} leaves {keep.sum()} dates in {self.label}")
        index = np.flatnonzero(keep)
        return DailyPath(
            dates=self.dates[keep],
            index=self.index[keep] / self.index[keep][0],
            fund=self.fund[keep] / self.fund[keep][0],
            curves=[self.curves[i] for i in index],
            zero_10y=self.zero_10y[keep],
            implied_vol=self.implied_vol[keep],
            variance=self.variance[keep],
            cash_rate=self.cash_rate[keep],
            own_credit_spread=(None if self.own_credit_spread is None
                               else self.own_credit_spread[keep]),
            label=label or f"{self.label} {start} to {end}",
        )


def instantaneous_variance(implied_vol, heston, tenor: float = VIX_TENOR_YEARS) -> np.ndarray:
    """The instantaneous variance whose Heston forward curve averages to a quoted variance.

    Over a window of length T the model's expected average variance is

        w(T) = theta + (v0 - theta) * (1 - exp(-kappa T)) / (kappa T),

    which is linear in v0 and inverts in one line. Reading the quote straight into v0 instead -
    which is the obvious thing and what the first version of this did - is wrong in a direction
    that matters: the model mean-reverts towards a long-run variance well above the quote, so a
    quiet market's thirty-day quote implies an instantaneous variance lower than the quote, and
    a panic's implies one higher. At a volatility index of 12 against a long-run level above 20
    the difference is a third of the variance.

    Floored at a hundredth of the long-run level rather than at zero, because an exactly zero
    variance is an absorbing state the model cannot leave and the proxy was never fitted at.
    """
    quoted = np.asarray(implied_vol, dtype=float) ** 2
    decay = (1.0 - np.exp(-heston.kappa * tenor)) / (heston.kappa * tenor)
    return np.maximum(heston.theta + (quoted - heston.theta) / decay, 0.01 * heston.theta)


def implied_at_tenor(implied_vol, heston, quoted_tenor: float, target_tenor: float) -> np.ndarray:
    """Rescale a quoted implied volatility to another tenor using the model's term structure.

    The hedge buys one- and two-year puts and the longest free volatility series is six months,
    so the level has to be extended. The model supplies the shape and the quote supplies the
    level, which is the same division of labour the smile uses. What it cannot supply is a term
    structure that moves differently from the model's own, and in a volatility spike the front
    moves far more than the back - so this understates how much a long-dated put gains in a
    panic, and every put result inherits that.
    """
    variance = instantaneous_variance(implied_vol, heston, quoted_tenor)

    def average(tenor: float) -> np.ndarray:
        decay = (1.0 - np.exp(-heston.kappa * tenor)) / (heston.kappa * tenor)
        return heston.theta + (variance - heston.theta) * decay

    return np.sqrt(np.maximum(average(target_tenor), 1e-12))


def daily_curves(panel: pd.DataFrame) -> tuple:
    """A bootstrapped zero curve for every date, and the ten-year zero alongside.

    Dates missing any quoted tenor are dropped rather than forward filled. A Treasury holiday
    is a day on which nothing traded, so carrying the previous curve forward would invent a
    mark; dropping the date leaves the hedge unrebalanced over it, which is what happened.
    """
    tenors = list(PAR_SERIES)
    columns = [PAR_SERIES[tenor] for tenor in tenors]
    usable = panel[columns].notna().all(axis=1)
    frame = panel.loc[usable]
    curves = [
        bootstrap(tenors, frame.loc[date, columns].to_numpy(dtype=float) / 100.0)
        for date in frame.index
    ]
    zero_10y = np.array([float(curve.zero(10.0)) for curve in curves])
    return frame.index, curves, zero_10y


def sub_account_path(index: np.ndarray, curves: list, cash_rate: np.ndarray,
                     mix, years: np.ndarray) -> np.ndarray:
    """The contract value's own index, built the way the simulator builds it.

    A rebalanced portfolio: the equity sleeve earns the index, the bond sleeve earns a
    constant-maturity zero held for a day and rolled, cash earns overnight. Weights are
    reapplied every day, which is what a sub-account allocation does and what the simulator
    assumes.
    """
    steps = np.diff(years)
    equity_step = index[1:] / index[:-1]
    bond_step = np.array([
        float(curves[i + 1].discount(mix.bond_duration - steps[i])
              / curves[i].discount(mix.bond_duration))
        for i in range(steps.size)
    ])
    cash_step = np.exp(0.5 * (cash_rate[:-1] + cash_rate[1:]) * steps)
    blended = (mix.equity_weight * equity_step
               + mix.bond_weight * bond_step
               + mix.money_market * cash_step)
    return np.concatenate([[1.0], np.cumprod(np.maximum(blended, 1e-12))])


def load_history(
    panel: pd.DataFrame,
    heston,
    mix,
    index_column: str = "SP500",
    vol_column: str = "VIXCLS",
    vol_tenor: float = VIX_TENOR_YEARS,
    cash_column: str = "DFF",
    # Moody's Baa spread over the ten-year Treasury, which is the own non-performance proxy.
    # BAMLC0A4CBBB is the better instrument and is unusable here: 775 observations from
    # September 2023, so it cannot span the 2020 stress. BAA10Y has the full history and both
    # stay in the panel so the two can be compared where they overlap.
    credit_column: str = "BAA10Y",
    dividend_yield: float = 0.0,
    reference_tenor: float = 1.0,
    label: str = "realised history",
) -> DailyPath:
    """Assemble a replay path from the committed market panel.

    ``dividend_yield`` is added to a price index to approximate a total return. It should be
    zero whenever the index column is already a total-return series, which is the only way to
    avoid the assumption entirely: a constant yield is wrong in both directions over a decade,
    and the contract value compounds the error.
    """
    panel = panel.sort_index()
    dates, curves, zero_10y = daily_curves(panel)
    frame = panel.loc[dates]

    needed = [index_column, vol_column, cash_column]
    usable = frame[needed].notna().all(axis=1)
    frame = frame.loc[usable]
    curves = [curve for curve, keep in zip(curves, usable) if keep]
    zero_10y = zero_10y[usable.to_numpy()]
    if frame.shape[0] < 50:
        raise ValueError(f"only {frame.shape[0]} usable dates; check the panel columns")

    dates = pd.DatetimeIndex(frame.index)
    years = (dates - dates[0]).days.to_numpy(dtype=float) / 365.0
    price = frame[index_column].to_numpy(dtype=float)
    total_return = price / price[0] * np.exp(dividend_yield * years)
    cash_rate = frame[cash_column].to_numpy(dtype=float) / 100.0

    quoted = frame[vol_column].to_numpy(dtype=float) / 100.0
    return DailyPath(
        dates=dates,
        index=total_return,
        fund=sub_account_path(total_return, curves, cash_rate, mix, years),
        curves=curves,
        zero_10y=zero_10y,
        implied_vol=implied_at_tenor(quoted, heston, vol_tenor, reference_tenor),
        variance=instantaneous_variance(quoted, heston, vol_tenor),
        cash_rate=cash_rate,
        own_credit_spread=_own_credit_spread(panel, dates, credit_column),
        label=label,
    )


def _own_credit_spread(panel: pd.DataFrame, dates: pd.DatetimeIndex, column: str):
    """The insurer's own non-performance spread, from the Baa index, on the equity calendar.

    The published series is Moody's Baa over the ten-year Treasury, and Jackson's insurance
    subsidiaries are rated several notches above Baa, so the index is scaled rather than used
    raw. ``OWN_CREDIT_SHARE`` is that scaling, and the factor is applied here rather than at the
    point of use so that the field this fills means what its name says everywhere it is read -
    the figure plotted on the reporting figure and differenced in the disclosure replica is the
    own-credit spread, not a corporate index that stands in for one.

    Deliberately not added to the columns a date has to have. Requiring it drops two of the
    2,491 replay dates, and two dates is immaterial to every conclusion while being enough to
    move every figure in every hedging table - a silent renumbering of results that are quoted
    in commit messages and notes. The credit series publishes on federal business days and the
    equity series on NYSE days, which is the same calendar mismatch the rates already have, so
    this uses the same remedy: carry forward by at most five days and refuse if a gap is longer.
    """
    if not column or column not in panel.columns:
        return None
    series = panel[column].reindex(dates).ffill(limit=5)
    if series.isna().any():
        missing = series.index[series.isna()]
        raise ValueError(
            f"{column} has a gap longer than five days at {missing[0].date()} "
            f"({series.isna().sum()} dates unfilled); check the panel rather than widening the fill"
        )
    return OWN_CREDIT_SHARE * series.to_numpy(dtype=float) / 100.0


def available_episodes(path: DailyPath) -> dict:
    """The stress windows this path can actually cover, and the ones it cannot.

    Returned rather than warned about, because which episodes are reachable depends on how far
    back the free index history goes and that belongs in the write-up rather than in a log line.
    """
    covered, missing = {}, {}
    for name, (start, end, why) in EPISODES.items():
        inside = ((path.dates >= pd.Timestamp(start)) & (path.dates <= pd.Timestamp(end))).sum()
        span = np.busday_count(np.datetime64(start), np.datetime64(end))
        if inside >= 0.5 * span:
            covered[name] = (start, end, why)
        else:
            missing[name] = (start, end, why, int(inside), int(span))
    return {"covered": covered, "missing": missing}


def block_length(returns: np.ndarray) -> float:
    """Expected block length for the stationary bootstrap, from volatility persistence.

    Politis and Romano's stationary bootstrap draws geometric blocks, and the one parameter is
    the mean block length. Choosing it by eye is where a bootstrap scenario set stops being
    evidence, so it is set from the data: the integrated autocorrelation of squared returns,
    which is the horizon over which volatility clustering decays and therefore the length a
    block has to be to carry any of it.
    """
    squared = np.asarray(returns, dtype=float) ** 2
    centred = squared - squared.mean()
    total = float(centred @ centred)
    # Judged against the level of the squared returns rather than against zero. A series whose
    # squared returns barely vary carries no clustering to measure, and the autocorrelation of
    # what is left is rounding noise: on a constant series an exact-zero guard let this through
    # and returned a block of 114 days, longer than most windows it would be drawn from.
    if total <= 1e-12 * max(float(squared @ squared), 1e-300):
        return 1.0
    acf = []
    for lag in range(1, 61):
        value = float(centred[lag:] @ centred[:-lag]) / total
        if value <= 0:
            break
        acf.append(value)
    return 1.0 + 2.0 * float(np.sum(acf))


def stationary_bootstrap(
    path: DailyPath, n_days: int, n_paths: int, seed: int, mean_block: float | None = None,
) -> np.ndarray:
    """Resampled daily log returns of the equity index, shape (n_paths, n_days).

    Returns log returns rather than a whole DailyPath, because the curve and the volatility
    state have to be resampled with them to stay coherent and that is the simulator's job. What
    this provides is the index path and the dates it was drawn from, so everything else can
    follow the same draws.
    """
    log_returns = np.diff(np.log(path.index))
    if log_returns.size < 50:
        raise ValueError("not enough history to bootstrap")
    mean_block = block_length(log_returns) if mean_block is None else mean_block
    probability = 1.0 / max(mean_block, 1.0)

    rng = np.random.default_rng(seed)
    position = rng.integers(0, log_returns.size, size=n_paths)
    drawn = np.empty((n_paths, n_days), dtype=int)
    for day in range(n_days):
        drawn[:, day] = position
        restart = rng.random(n_paths) < probability
        position = np.where(restart, rng.integers(0, log_returns.size, size=n_paths),
                            (position + 1) % log_returns.size)
    return drawn


def resample(
    path: DailyPath,
    drawn: np.ndarray,
    mix,
    label: str = "bootstrap",
    annual_drift: float | None = None,
) -> DailyPath:
    """One bootstrap path: the equity days reordered, the rate path left as it happened.

    ``drawn`` is one row of ``stationary_bootstrap``: indices into the array of daily log
    returns. Index i is the move from day i to day i+1, so the volatility state that belongs
    with it is day i+1's, and that pairing is kept. A day drawn from March 2020 arrives with its
    crash return and its spiked variance together, which is the co-movement a bootstrap exists
    to carry and the one no model here reproduces.

    **What is not resampled, and why.** The zero curve, the overnight rate and the own-credit
    spread stay on the history's own course. They are persistent levels rather than returns, and
    neither way of resampling them survives inspection:

    - Drawing the *level* with the day, as the volatility is drawn, puts a jump at every block
      boundary. At the fitted 11-day mean block that is about 225 boundaries in a ten-year path,
      and two days drawn at random from this decade sit 139bp apart in the ten-year zero on
      average against a realised daily standard deviation of 5.3bp. A ten-year yield that moves
      twenty-six standard deviations two hundred times a decade is not a world, and the rho leg's
      profit and loss would be nothing but those jumps.
    - Accumulating the daily *changes* is supported by the data - the variance ratio is close to
      one at both 11 days (17bp realised against 17.5bp implied by the daily figure) and 252 days
      (90bp against 84bp), so this decade's ten-year yield is close to a random walk. But a
      random walk of 5.3bp a day over 2,492 days has a terminal dispersion of 264bp around a
      starting 1.61 per cent, so a quarter of paths would end at a negative ten-year yield and
      the liability on them would be valued by extrapolating a regression fitted at positive
      rates. That measures extrapolation error, not hedge performance.

    Holding the rate path fixed makes this a controlled experiment in the one thing it is about:
    the order the equity days arrived in. Every path faces the decade of rates that actually
    happened, so the rate hedge is tested against a real rate path rather than a synthetic one.
    The cost is explicit and not small - the equity and rate moves of a drawn day are separated,
    so 2022, the one episode where both fell together, sits at a fixed point in the calendar
    instead of travelling with the equity path. This experiment therefore says nothing about that
    joint tail; the crisis replays do, because they keep every day whole.

    The sub-account is rebuilt rather than resampled, by ``sub_account_path`` on the new equity
    index and the retained curves and rates. So the bond and money-market sleeves earn what they
    earned on those dates, the equity sleeve earns the drawn return, and the blend is the
    contract's own allocation at every step with nothing approximated.

    The dates are the calendar's own first n days rather than the drawn ones. They are not a
    claim about when anything happened; they carry the contract's anniversaries and the spacing
    between rebalances, and a resampled date index would put two anniversaries in one week.

    ``annual_drift`` re-centres the equity log returns on a stated expected return, and without
    it the experiment answers a narrower question than it looks like it does. A bootstrap inherits
    the drift of the window it draws from, and this window is 2016 to 2026: an index that
    compounded at about fifteen per cent a year. Every path drawn from it is a bull market on
    average, so a hedging result averaged over them is a hedging result in a bull market, which
    is the bias the experiment exists to remove rather than to reproduce. Passing a drift keeps
    the window's volatility clustering and its fat tails while putting the average somewhere
    defensible. The shift is the sample's own mean less the target, so the ensemble is centred on
    the target and each path keeps its own sampling variation around it; re-centring each path
    individually would throw away the dispersion in realised outcomes, which is half of what is
    being measured.
    """
    drawn = np.asarray(drawn, dtype=int)
    steps = np.diff(np.log(path.index))
    drawn_steps = steps[drawn]
    if annual_drift is not None:
        per_day = annual_drift * max(path.year_fraction[-1], 1e-9) / steps.size
        drawn_steps = drawn_steps + (per_day - steps.mean())
    state = drawn + 1
    n_days = drawn.size + 1

    index = np.concatenate([[1.0], np.exp(np.cumsum(drawn_steps))])
    curves = path.curves[:n_days]
    cash_rate = path.cash_rate[:n_days]
    years = path.year_fraction[:n_days]

    def drawn_with_the_day(values):
        return None if values is None else np.concatenate([[values[0]], values[state]])

    return DailyPath(
        dates=path.dates[:n_days],
        index=index,
        fund=sub_account_path(index, curves, cash_rate, mix, years),
        curves=curves,
        zero_10y=path.zero_10y[:n_days],
        implied_vol=drawn_with_the_day(path.implied_vol),
        variance=drawn_with_the_day(path.variance),
        cash_rate=cash_rate,
        own_credit_spread=(None if path.own_credit_spread is None
                           else path.own_credit_spread[:n_days]),
        label=label,
    )
