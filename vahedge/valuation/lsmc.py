"""A regression proxy for the liability, so a hedge backtest is possible at all.

A hedge backtest needs the liability's value and its Greeks at every rebalance date on every
path. Valuing each of those by its own risk-neutral simulation is nested simulation, and at five
hundred weekly dates and a dozen seconds a valuation it is a day of compute for one strategy.
The way out is Longstaff and Schwartz's: the conditional expectation that defines the value is a
function of the state, so estimate that function once by regression and evaluate it afterwards
for nothing.

The trick that makes it work is that the regression does not need an accurate value at each
node, only an unbiased one. A single path's realised discounted future cash flow is a terrible
estimate of the value at that node - its error is the size of the payoff - but it is unbiased,
and least squares averages that error away across nodes. So the design points are every path at
every anniversary of one simulation, which costs one simulation rather than tens of thousands.

Four things decide whether the fit is usable, and three of them were learned by watching the
first version fail against a nested gold standard.

*The liability is homogeneous of degree one in the contract value and the benefit base
together.* Double both and every cash flow doubles. So the thing to regress is the value per
unit of benefit base, against the ratio of contract value to benefit base, and a whole dimension
of variation disappears from the problem before any basis function is chosen. Regressing the
level instead makes the fit spend its degrees of freedom on scale.

*The continuation value is conditional on the contract still being there.* The recorded cash
flows carry the discount factor, the survival curve and the persistency curve from time zero, so
all three come out, not just the discount factor. Leaving the last two in values a contract that
might already have lapsed or died, which by year twenty is about half of what the contract
standing at the node is worth. The error is one-directional, grows with the horizon, and shows
up in no fit diagnostic, because the fit reproduces it faithfully.

*The exhausted state is a point on the same axis, not a separate regime.* The obvious state
variable, the log of the benefit base over the contract value, goes to infinity when the account
runs out, and a polynomial in a variable ranging to plus twenty-five produced an R-squared of
0.36 against nested values at year nine. Regressing on the ratio the other way up puts a spent
contract at zero, which is where it belongs: a contract with a hundredth of its benefit base
left and one with nothing left are worth almost exactly the same.

*The exhausted replicates do not get to outvote the live contracts.* They are one state
repeated, not a dense region, and by the last years of the contract they are nine rows in ten.
Left alone, least squares gives up the live region - the only part a hedge ever has to size
against - to shave a residual at a point it has already pinned thirty thousand times over.

*The two cash-flow legs are fitted separately.* Claims and fees are regressed on their own and
the market risk benefit assembled afterwards as claims less attribution times fees, so one fit
serves any attribution percentage. The GAAP lens and the economic lens use different ones and
the break-even solve walks a range.

*Thin support is flagged rather than extrapolated over silently.* Where the paths do not go the
spline continues the slope of its last populated segment, confidently and wrongly, and a hedge
sized off a Greek from there in a crisis is the failure this project is meant to measure rather
than commit. The fit records where its own design points fell and the evaluator marks any state
the paths effectively never reached.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_RIDGE = 1e-8
MONEYNESS_CAP = 3.0      # contract value over benefit base, above which the guarantee is dead
N_KNOTS = 10             # knots of the moneyness spline, boundaries included
# The design range is taken at this quantile from each end rather than at the extremes, because
# a handful of paths at an extreme should not vouch for the whole interval beyond them.
DESIGN_TAIL = 0.001
# A range is not enough on its own, because the moneyness distribution is not merely skewed:
# most of it piles up at a single point. Ninety-six per cent of contracts are exhausted by year
# thirty, so the range runs from zero to the cap while the interval from a third of the benefit
# base up to it holds almost nothing, and a node sitting there comes back unflagged and wrong by
# forty per cent of the account value. So the support is recorded as a histogram and a state is
# flagged when its own bin is too thin to have fitted anything, whether that bin is at an end or
# in the middle.
SUPPORT_BIN = 0.05           # bin width in units of contract value over benefit base
MIN_SUPPORT_SHARE = 0.001    # ...of the fitting paths, below which a bin has nothing to say
MIN_SUPPORT_ROWS = 25
# The moneyness distribution has two point masses, not one, and both of them are a single
# state repeated rather than a dense region. At the bottom, an exhausted contract sits at
# exactly zero, and by year twenty-five that is ninety-two per cent of the fitting rows. At the
# top, the annual step-up resets the benefit base to the contract value, so every contract that
# stepped up ends the year at the same ratio of one less that year's charges. Least squares will
# give up the region in between - where the guarantee is actually decided - to shave a residual
# at a point it has already pinned tens of thousands of times over, and at the top the crowding
# also drags the spline's curvature into a spike. So any repeated value is weighted to count as
# this many rows between them. Two thousand independent realisations estimate a conditional mean
# far more precisely than anything else in the fit, so nothing is lost; a dense but genuinely
# varying region carries information in every row and is left alone.
REPLICATE_EFFECTIVE_ROWS = 2_000
REPLICATE_TOLERANCE = 1e-6       # moneyness values this close together count as the same state
# Smoothing strengths tried for the roughness penalty, and how much better a score has to be
# to prefer a rougher fit. Scored on held-out paths; see _choose_smoothing.
SMOOTHING_GRID = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
# Size of the move the reported gamma is averaged over, in log contract value. See ProxyFit.greeks:
# ten per cent is the scale a convexity hedge is sized for and wide enough that the second
# difference is not dominated by the spline's own wiggle.
GAMMA_STEP = 0.10


def _knots(moneyness: np.ndarray, live: np.ndarray, n_knots: int = N_KNOTS) -> np.ndarray:
    """Knots at quantiles of the live contracts, with exhaustion pinned as its own knot.

    Quantiles rather than an even grid, because the states the paths actually visit are
    concentrated: early on almost every contract sits near a moneyness of one, and only later
    does the distribution spread toward exhaustion. An even grid would put most of its knots
    where there is no data and none where the curvature is.

    Quantiles of the live contracts specifically, because the exhausted ones are a point mass
    that by the late years is nine tenths of the sample: quantiles taken over everything would
    put eight of ten knots at exactly zero and leave the live range with none.
    """
    live = np.asarray(live, dtype=bool)
    sample = moneyness[live] if live.sum() > 50 else moneyness
    quantiles = np.linspace(0.01, 0.99, max(n_knots - 1, 3))
    # Boundary knots at the ends of the design, not past them. A natural spline is straight
    # beyond its outermost knots, and straight is the right behaviour where there is no data:
    # the slope carries on, so the delta stays continuous and signed correctly and only the
    # curvature goes to zero. Putting the top knot a margin further out instead - which was
    # tried, to stop the curvature collapsing where the contract lives - leaves a cubic segment
    # with nothing in it, and the fit used it to send the value back up and the delta positive.
    knots = np.unique(np.round(np.concatenate([[0.0], np.quantile(sample, quantiles)]), 6))
    return _separate(knots)


def _separate(knots: np.ndarray, min_gap_share: float = 0.02) -> np.ndarray:
    """Drop knots that sit on top of their neighbour.

    Exhaustion is not the only point mass in the moneyness distribution: the annual step-up
    puts a ceiling on it, so in the early years a large share of contracts sit at exactly the
    same ratio and the top few quantiles land within a thousandth of each other. The spline
    basis divides by the gap between knots, so a pair that close produces coefficients in the
    thousands and a delta that is nonsense while the value it is the slope of still looks fine.
    """
    span = float(knots[-1] - knots[0])
    if span <= 0:
        return knots
    min_gap = min_gap_share * span
    kept = [float(knots[0])]
    for knot in knots[1:-1]:
        if knot - kept[-1] >= min_gap:
            kept.append(float(knot))
    if len(kept) > 1 and knots[-1] - kept[-1] < min_gap:
        kept.pop()
    kept.append(float(knots[-1]))
    return np.asarray(kept)


def _spline(moneyness: np.ndarray, knots: np.ndarray) -> list:
    """Natural cubic spline basis in moneyness: smooth inside, straight outside.

    Three things were tried before this. A global polynomial of order three to five oscillated
    at the ends where a few extreme paths sit and was flat where the value function turns. A
    linear spline fixed the values - it follows the smoothed-put shape the value per unit of
    benefit base has, and cannot blow up between knots - but its derivative is a step function,
    and the proxy's derivative is the hedge ratio. Against nested bump deltas the linear fit
    came back at minus five hundred where the truth was minus twenty-eight, at states sitting
    just past the top knot in the sparse upper tail. A fit that values correctly and hedges on
    a slope like that is worse than useless, because the error only shows up in the hedge.

    The natural cubic spline has a continuous second derivative everywhere and is constrained
    to be linear beyond the outermost knots, which is exactly where the old basis went wrong:
    the sparse tails can no longer bend the fit, and delta and gamma both come out smooth.
    """
    moneyness = np.asarray(moneyness, dtype=float)
    last = knots[-1]

    def ramp(knot: float) -> np.ndarray:
        return (np.maximum(moneyness - knot, 0.0) ** 3
                - np.maximum(moneyness - last, 0.0) ** 3) / (last - knot)

    tail = ramp(knots[-2])
    return [np.ones(moneyness.size), moneyness] + [
        ramp(knot) - tail for knot in knots[:-2]
    ]


def _spline_slope(moneyness: np.ndarray, knots: np.ndarray) -> list:
    """The same basis differentiated in moneyness, term by term.

    Differentiating the basis rather than bumping the fitted function is not a refinement for
    its own sake. The spline's pieces meet at the knots, so a central difference taken across
    one mixes the slopes on either side and, past the outermost knot, mixes an interior slope
    with the linear extrapolation. In the early years the whole design piles up against the
    step-up ceiling and the top knot sits right in that pile, which is exactly where a bumped
    delta goes wrong.
    """
    moneyness = np.asarray(moneyness, dtype=float)
    last = knots[-1]

    def ramp_slope(knot: float) -> np.ndarray:
        return 3.0 * (np.maximum(moneyness - knot, 0.0) ** 2
                      - np.maximum(moneyness - last, 0.0) ** 2) / (last - knot)

    tail = ramp_slope(knots[-2])
    return [np.zeros(moneyness.size), np.ones(moneyness.size)] + [
        ramp_slope(knot) - tail for knot in knots[:-2]
    ]


def _design(moneyness, variance_vol, rate, knots: np.ndarray) -> np.ndarray:
    """The spline in moneyness, low order in volatility and the rate, plus interactions.

    Volatility and the rate get linear and quadratic terms only. Their effect on the value is
    gentle across the whole range, and spending basis functions there instead of on moneyness
    was what produced an R-squared of 0.22 against nested values.
    """
    moneyness = np.asarray(moneyness, dtype=float)
    columns = _spline(moneyness, knots)
    for extra in (variance_vol, rate):
        extra = np.asarray(extra, dtype=float)
        columns.append(extra)
        columns.append(extra ** 2)
        columns.append(extra * moneyness)
    columns.append(np.asarray(variance_vol, float) * np.asarray(rate, float))
    return np.column_stack(columns)


def _spline_curvature(moneyness: np.ndarray, knots: np.ndarray) -> list:
    """The basis differentiated twice. Zero outside the outermost knots, by construction."""
    moneyness = np.asarray(moneyness, dtype=float)
    last = knots[-1]

    def ramp_curvature(knot: float) -> np.ndarray:
        return 6.0 * (np.maximum(moneyness - knot, 0.0)
                      - np.maximum(moneyness - last, 0.0)) / (last - knot)

    tail = ramp_curvature(knots[-2])
    zero = np.zeros(moneyness.size)
    return [zero, zero] + [ramp_curvature(knot) - tail for knot in knots[:-2]]


def _design_slope(moneyness, variance_vol, rate, knots: np.ndarray) -> np.ndarray:
    """``_design`` differentiated in moneyness. The column order has to match it exactly."""
    moneyness = np.asarray(moneyness, dtype=float)
    zero = np.zeros(moneyness.size)
    columns = _spline_slope(moneyness, knots)
    for extra in (variance_vol, rate):
        extra = np.asarray(extra, dtype=float)
        columns.append(zero)          # the level term does not depend on moneyness
        columns.append(zero)          # nor its square
        columns.append(extra)         # d(extra * m)/dm
    columns.append(zero)              # the volatility-rate cross term
    return np.column_stack(columns)


def _design_curvature(moneyness, variance_vol, rate, knots: np.ndarray) -> np.ndarray:
    """``_design`` differentiated twice in moneyness."""
    moneyness = np.asarray(moneyness, dtype=float)
    zero = np.zeros(moneyness.size)
    columns = _spline_curvature(moneyness, knots)
    for _ in range(2):
        columns.extend([zero, zero, zero])   # all the level and interaction terms are linear in m
    columns.append(zero)
    return np.column_stack(columns)


def _design_in(factor: str, moneyness, variance_vol, rate, knots: np.ndarray) -> np.ndarray:
    """``_design`` differentiated in the volatility or the rate.

    Both enter the basis the same way - a level, a square, a product with moneyness, and one
    cross term between them - so one function covers both and the column order cannot drift
    apart from ``_design``.
    """
    moneyness = np.asarray(moneyness, dtype=float)
    volatility = np.asarray(variance_vol, dtype=float)
    rate = np.asarray(rate, dtype=float)
    zero = np.zeros(moneyness.size)
    one = np.ones(moneyness.size)
    columns = [zero] * len(_spline(moneyness, knots))   # the spline block is flat in both
    if factor == "volatility":
        columns += [one, 2.0 * volatility, moneyness]      # d/dvol of vol, vol^2, vol*m
        columns += [zero, zero, zero]                      # the rate block does not move
        columns.append(rate)                               # d(vol * r)/dvol
    elif factor == "rate":
        columns += [zero, zero, zero]
        columns += [one, 2.0 * rate, moneyness]
        columns.append(volatility)
    else:
        raise ValueError(f"factor must be volatility or rate; got {factor!r}")
    return np.column_stack(columns)


def _state_parts(account_value, benefit_base, variance, zero_10y):
    """The three state variables, in the units the basis expects.

    Moneyness is the contract value over the benefit base, capped. Uncapped it runs to whatever
    a lucky path reached, and a spline fitted to that spends its range on states no hedge will
    ever see. Capped at three the guarantee is already worthless, so nothing is lost.

    A spent contract sits at zero, which is a real point on the same axis rather than the minus
    infinity a log would give it, so live and exhausted contracts share one fit. They should:
    a contract with a hundredth of its benefit base left and one with nothing left are worth
    almost exactly the same.
    """
    base = np.maximum(np.asarray(benefit_base, dtype=float), 1e-12)
    moneyness = np.clip(np.asarray(account_value, dtype=float) / base, 0.0, MONEYNESS_CAP)
    return (
        moneyness,
        np.sqrt(np.maximum(np.asarray(variance, dtype=float), 0.0)),
        np.asarray(zero_10y, dtype=float),
        base,
    )


@dataclass(frozen=True)
class _YearFit:
    """One year's coefficients, for both cash-flow legs, in units of the benefit base."""

    claim: np.ndarray
    fee: np.ndarray
    knots: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    support: np.ndarray        # fitting rows per moneyness bin
    min_support: float
    n_exhausted: int
    smoothing: float = 0.0     # the roughness penalty strength chosen for this year

    def thin(self, moneyness: np.ndarray) -> np.ndarray:
        """True where the fitting paths left this part of the moneyness axis empty."""
        index = np.clip((moneyness / SUPPORT_BIN).astype(int), 0, self.support.size - 1)
        return self.support[index] < self.min_support


@dataclass(frozen=True)
class ProxyFit:
    """Fitted coefficients by year, with the design range they are valid over.

    Two families per year. ``fits`` is the value at an anniversary after that year's charges,
    withdrawal and step-up; ``pre_fits`` is the value an instant before them, of everything
    from that year's cash flows onward. They are not redundant. A hedge rebalancing inside a
    policy year sees a contract whose value has grown away from its benefit base and whose
    year-end events are still ahead of it, which is a pre-event state; the post-event family
    has no design points up there, because the step-up resets the base every anniversary and
    caps the moneyness it records. Bracketing an intra-year date between the post-event fit at
    the anniversary behind it and the pre-event fit at the anniversary ahead gives both
    endpoints a state their own design contains.
    """

    fits: dict
    order: int
    n_paths: int
    diagnostics: pd.DataFrame
    pre_fits: dict = field(default_factory=dict)

    @property
    def years(self):
        return sorted(self.fits)

    def value(
        self,
        year: int,
        account_value,
        benefit_base,
        variance,
        zero_10y,
        attribution: float = 1.0,
        flag_extrapolation: bool = True,
        pre_event: bool = False,
    ):
        """The market risk benefit at a state, per the fitted year.

        Returns (value, outside). The value is still returned for states beyond the design
        range, because refusing outright would stop a backtest on its worst day, but it is
        marked so the caller can report how often it happened rather than discover it in a
        result.
        """
        family = self.pre_fits if pre_event else self.fits
        if year not in family:
            raise KeyError(f"no fit for year {year}; fitted years are {self.years}")
        fit_year = family[year]

        account = np.atleast_1d(np.asarray(account_value, dtype=float))
        base = np.atleast_1d(np.asarray(benefit_base, dtype=float))
        variance = np.broadcast_to(np.atleast_1d(np.asarray(variance, float)), account.shape)
        rate = np.broadcast_to(np.atleast_1d(np.asarray(zero_10y, float)), account.shape)

        moneyness, volatility, rate_level, base = _state_parts(account, base, variance, rate)
        design = _design(moneyness, volatility, rate_level, fit_year.knots)
        per_unit = design @ fit_year.claim - attribution * (design @ fit_year.fee)
        values = per_unit * base
        if flag_extrapolation:
            observed = np.column_stack([moneyness, volatility, rate_level])
            outside = np.any((observed < fit_year.lower) | (observed > fit_year.upper), axis=1)
            outside |= fit_year.thin(moneyness)
        else:
            outside = np.zeros(moneyness.size, dtype=bool)
        return values, outside

    def delta(
        self, year: int, account_value, benefit_base, variance, zero_10y,
        attribution: float = 1.0,
    ):
        """Equity exposure, from the derivative of the fitted basis.

        A move in the funds moves the contract value and leaves the benefit base alone, which is
        the whole point of the guarantee, so the derivative that matters is the one in the
        contract value. The value per unit of benefit base is f(m) with m the contract value
        over that base, so dV/dAV is f'(m) and the exposure to a proportional move is AV f'(m).

        Reported per unit log move in the contract value, not in the index. The step between the
        two is the equity weight of the fund mix and the basis between the funds and the index,
        and both belong in the hedge sizing where they are visible rather than folded into a
        Greek. Above the moneyness cap the guarantee is dead and the derivative is zero, which
        is the right answer rather than an artefact of the clip.
        """
        if year not in self.fits:
            raise KeyError(f"no fit for year {year}; fitted years are {self.years}")
        fit_year = self.fits[year]

        account = np.atleast_1d(np.asarray(account_value, dtype=float))
        base = np.atleast_1d(np.asarray(benefit_base, dtype=float))
        variance = np.broadcast_to(np.atleast_1d(np.asarray(variance, float)), account.shape)
        rate = np.broadcast_to(np.atleast_1d(np.asarray(zero_10y, float)), account.shape)

        moneyness, volatility, rate_level, _ = _state_parts(account, base, variance, rate)
        slope = _design_slope(moneyness, volatility, rate_level, fit_year.knots)
        per_unit = slope @ fit_year.claim - attribution * (slope @ fit_year.fee)
        return account * np.where(moneyness < MONEYNESS_CAP, per_unit, 0.0)

    def greeks(
        self, year: int, account_value, benefit_base, variance, zero_10y,
        attribution: float = 1.0, pre_event: bool = False,
        gamma_step: float = GAMMA_STEP,
    ) -> dict:
        """Value, delta, gamma, vega and rho at a state, all from the fitted basis.

        The four derivatives in one call because a hedge needs them together and they share
        every intermediate. All of them are analytic, which matters more here than it does for
        the value: a bumped second derivative on a spline either straddles a knot or sits inside
        double precision's floor, and a bumped vega has to re-clip the state.

        Gamma is the exception and is deliberately not a point second derivative. A regression
        on single-path realisations estimates a conditional mean well, its slope reasonably, and
        its curvature not at all: against nested bumps the analytic second derivative of this
        fit oscillated between plus twelve hundred and minus six hundred across neighbouring
        states. What is returned instead is the average curvature across a move of
        ``gamma_step`` in the log contract value, which averages that oscillation out and is the
        quantity a convexity hedge actually wants - a hedge is sized for moves of several per
        cent, not for the limit as the move goes to zero. Pass ``gamma_step=0`` for the point
        derivative, which is exact for the fitted function and useless as a risk number.

        Units otherwise match the Greeks module, so a hedge can be sized off either without
        conversion. Delta and gamma are per unit log move in the contract value, vega is per
        volatility point, and rho is per basis point of the ten-year zero rate with the curve's
        shape held. That last one is a one-factor rate exposure and is all a one-factor rate
        state can support; a curve-shape hedge would need a slope state in the basis and does
        not get one.
        """
        family = self.pre_fits if pre_event else self.fits
        if year not in family:
            raise KeyError(f"no fit for year {year}; fitted years are {self.years}")
        fit_year = family[year]

        account = np.atleast_1d(np.asarray(account_value, dtype=float))
        base = np.atleast_1d(np.asarray(benefit_base, dtype=float))
        variance = np.broadcast_to(np.atleast_1d(np.asarray(variance, float)), account.shape)
        rate = np.broadcast_to(np.atleast_1d(np.asarray(zero_10y, float)), account.shape)
        moneyness, volatility, rate_level, unit = _state_parts(account, base, variance, rate)
        alive = moneyness < MONEYNESS_CAP

        def combine(design: np.ndarray) -> np.ndarray:
            return design @ fit_year.claim - attribution * (design @ fit_year.fee)

        per_unit = combine(_design(moneyness, volatility, rate_level, fit_year.knots))
        slope = np.where(alive, combine(
            _design_slope(moneyness, volatility, rate_level, fit_year.knots)), 0.0)
        curvature = np.where(alive, combine(
            _design_curvature(moneyness, volatility, rate_level, fit_year.knots)), 0.0)
        in_vol = combine(_design_in("volatility", moneyness, volatility, rate_level,
                                    fit_year.knots))
        in_rate = combine(_design_in("rate", moneyness, volatility, rate_level, fit_year.knots))

        delta = account * slope
        value = per_unit * unit
        if gamma_step > 0.0:
            up = self.value(year, account * np.exp(gamma_step), base, variance, rate,
                            attribution, flag_extrapolation=False, pre_event=pre_event)[0]
            down = self.value(year, account * np.exp(-gamma_step), base, variance, rate,
                              attribution, flag_extrapolation=False, pre_event=pre_event)[0]
            gamma = (up - 2.0 * value + down) / gamma_step ** 2
        else:
            # d2V/d(ln AV)2 = AV f'(m) + AV m f''(m), the second term being the curvature of
            # the value per unit of benefit base.
            gamma = delta + account * moneyness * curvature
        return {
            "value": value,
            "delta": delta,
            "gamma": gamma,
            "vega": unit * in_vol,
            "rho_per_bp": unit * in_rate * 1e-4,
        }

    def greeks_at(
        self, years_since_issue: float, account_value, benefit_base, variance, zero_10y,
        attribution: float = 1.0,
    ) -> dict:
        """The value and its Greeks at any date, not only on an anniversary.

        The fit keyed ``k`` is the value at ``k + 1`` years since issue. A date inside policy
        year ``k`` is bracketed by the post-event fit at its start and the pre-event fit at its
        end, both evaluated at the state as it stands, and interpolated linearly in time.

        Two reasons for that construction rather than the obvious one. Taking the nearer annual
        fit would step the liability at every anniversary and put a jump in the hedge P&L that
        the contract never made. And bracketing with two post-event fits would ask the one
        ahead about a state it has no design points for: the step-up caps the moneyness a
        post-event state can reach, while a contract halfway through a good year sits above
        that cap. The pre-event family is fitted on exactly those states. Where it is missing -
        the last year, or a year with too few surviving paths - the post-event fit is used and
        the result is flagged by the usual support test rather than quietly extrapolated.
        """
        keys = self.years
        policy_year = int(np.floor(years_since_issue + 1e-9))
        start_key = min(max(policy_year - 1, keys[0]), keys[-1])
        end_key = min(max(policy_year, keys[0]), keys[-1])
        weight = float(np.clip(years_since_issue - policy_year, 0.0, 1.0))

        start = self.greeks(start_key, account_value, benefit_base, variance, zero_10y,
                            attribution)
        if weight <= 0.0:
            return start
        use_pre = end_key in self.pre_fits
        end = self.greeks(end_key, account_value, benefit_base, variance, zero_10y,
                          attribution, pre_event=use_pre)
        return {name: (1.0 - weight) * start[name] + weight * end[name] for name in start}


def _roughness(knots: np.ndarray, width: int, n_grid: int = 400) -> np.ndarray:
    """The integrated squared second derivative of the basis, as a matrix.

    A ridge on the coefficients is the wrong penalty for this fit and the Greeks are what
    showed it. With twenty basis functions and eight knots crowded into the narrow band of
    moneyness the contract actually occupies, the fitted level came out smooth and accurate
    while its second derivative oscillated between plus twelve hundred and minus six hundred
    from one state to the next. A ridge cannot see that: it penalises the size of the
    coefficients, and a wildly wiggly function can have small ones. This penalises wiggliness
    itself, which is the thing that is wrong, and leaves the level free.

    Only the moneyness block is penalised. The volatility and rate terms are already linear and
    quadratic, so there is no roughness in them to control, and penalising them would quietly
    shrink two exposures the hedge needs.
    """
    grid = np.linspace(float(knots[0]), float(knots[-1]), n_grid)
    curvature = np.column_stack(_spline_curvature(grid, knots))
    # Trapezoid weights for the integral over the knot range.
    step = grid[1] - grid[0]
    weight = np.full(n_grid, step)
    weight[0] = weight[-1] = 0.5 * step
    block = curvature.T @ (curvature * weight[:, None])
    penalty = np.zeros((width, width))
    penalty[:block.shape[0], :block.shape[1]] = block
    return penalty


def _penalised_solve(design, target, ridge, weights, penalty, strength) -> np.ndarray:
    weighted = design * weights[:, None]
    gram = weighted.T @ weighted
    scale = np.trace(gram) / max(np.trace(penalty), 1e-300)
    total = gram + strength * scale * penalty
    total = total + ridge * np.trace(gram) / total.shape[0] * np.eye(total.shape[0])
    return np.linalg.solve(total, weighted.T @ (target * weights))


def _choose_smoothing(design, target, weights, penalty, ridge, score_design, score_target,
                      grid=SMOOTHING_GRID) -> tuple:
    """Pick the smoothing strength on held-out paths, by the one-standard-error rule.

    Choosing the strength that minimises held-out error and then using the fit's second
    derivative is a mistake with a name: estimating a derivative from noisy data needs more
    smoothing than estimating the level, so a selection made on the level leaves the curvature
    under-smoothed. Here it left it oscillating between plus twelve hundred and minus six
    hundred across neighbouring states while the level stayed accurate to a fraction of a per
    cent, and the gamma is what the convexity leg of the hedge is sized from.

    The one-standard-error rule was the first thing tried, and measuring it against the nested
    standard is what ruled it out. The residual here is a single path's realised cash flow
    against a conditional mean, so its error bars are enormous, and the rule duly selected the
    strongest smoothing on the grid for every early year - which flattened the value function
    almost to a straight line. Values went from 0.6 to 1.1 per cent of premium at year one and
    from 0.2 to 3.3 at year thirty, deltas roughly doubled in error, and the gamma was no better
    for it, only differently wrong: over-smoothed rather than noisy.

    So the strength is chosen at the held-out minimum, which is the best available answer for
    the level and the slope, and the curvature is not taken as a point second derivative of
    this fit at all. See ``greeks``: it is an average over a move of a stated size, which is
    both the estimable quantity and the one a convexity hedge is sized for.
    """
    scores, errors, betas = [], [], []
    for strength in grid:
        beta = _penalised_solve(design, target, ridge, weights, penalty, strength)
        squared = (score_design @ beta - score_target) ** 2
        scores.append(float(squared.mean()))
        errors.append(float(squared.std(ddof=1) / np.sqrt(squared.size)))
        betas.append(beta)

    return grid[int(np.argmin(scores))], scores


def _replicate_weights(moneyness: np.ndarray) -> np.ndarray:
    """One per row, except where the same moneyness repeats more than it can inform.

    Any value that occurs more than ``REPLICATE_EFFECTIVE_ROWS`` times is a point mass in the
    design rather than a dense region, and its rows share that many rows' worth of weight
    between them. Exhaustion and the step-up ceiling are both caught by the same rule, which is
    better than naming either: whatever produces a repeated state, the statistical argument for
    capping it is the same.
    """
    rounded = np.round(np.asarray(moneyness, dtype=float) / REPLICATE_TOLERANCE)
    values, inverse, counts = np.unique(rounded, return_inverse=True, return_counts=True)
    scale = np.where(counts > REPLICATE_EFFECTIVE_ROWS,
                     REPLICATE_EFFECTIVE_ROWS / np.maximum(counts, 1), 1.0)
    return scale[inverse]


def _ridge_solve(design: np.ndarray, target: np.ndarray, ridge: float,
                 weights: np.ndarray | None = None) -> np.ndarray:
    if weights is not None:
        design = design * weights[:, None]
        target = target * weights
    gram = design.T @ design
    penalty = ridge * np.trace(gram) / gram.shape[0] * np.eye(gram.shape[0])
    return np.linalg.solve(gram + penalty, design.T @ target)


def fit(
    projection,
    order: int = 3,
    ridge: float = DEFAULT_RIDGE,
    holdout: float = 0.3,
    seed: int = 7,
    min_paths: int = 200,
) -> ProxyFit:
    """Fit the proxy from a recorded single-cohort projection.

    ``projection`` must come from ``gmwb.project(..., record=True)``. The continuation value at
    the end of year t on a path is the sum of that path's later cash flows, discounted back to t
    and put on the same conditioning as the state: the recorded flows carry the discount factor,
    the survival curve and the persistency curve from time zero, so all three have to be divided
    out, not just the discount factor. Dividing out only the discount leaves the value of a
    contract that might already have lapsed or died, which is smaller than the value of the
    contract standing at the node by exactly the persistency factor - a few percent in the first
    years and about half by year twenty. That error is one-directional and grows with the
    horizon, and it is invisible in a fit diagnostic because the fit reproduces it faithfully;
    only the nested standard shows it.

    A held-out share of paths is kept out of every fit and scored afterwards. Scoring on the
    paths the coefficients were fitted to would measure the fit's memory rather than its ability
    to value, and with twenty basis functions and forty thousand paths the two differ.

    Neither the held-out score nor any other in-sample diagnostic establishes that the fit is
    right. The residual against a single path's realised cash flow is dominated by payoff noise
    that no conditional mean can remove, and both of the errors this module has actually had
    were reproduced faithfully by the fit and invisible in its own numbers. Only nested
    simulation settles it; scripts/run_proxy_validation.py is that test, and it puts the error
    inside the design range at under one per cent of premium through year fourteen, about two
    per cent at year twenty, and beyond use after that, when nine contracts in ten are spent
    and the live tail is too thin to fit.
    """
    recorded = projection.recorded
    if recorded is None:
        raise ValueError("fit needs a projection run with record=True")

    n_paths, n_years = recorded["pv_claim"].shape
    rng = np.random.default_rng(seed)
    is_holdout = rng.random(n_paths) < holdout
    train, test = ~is_holdout, is_holdout
    if train.sum() < min_paths or test.sum() < min_paths // 2:
        raise ValueError(f"too few paths to fit and score: {train.sum()} train, {test.sum()} test")

    # Discounted cash flow from each year onward, read right to left.
    future_claims = np.cumsum(recorded["pv_claim"][:, ::-1], axis=1)[:, ::-1]
    future_fees = np.cumsum(recorded["pv_fee"][:, ::-1], axis=1)[:, ::-1]

    fits, pre_fits, rows = {}, {}, []
    for year in range(n_years - 1):
        discount = recorded["discount"][:, year]
        persistency = recorded["persistency"][:, year]
        base = recorded["benefit_base"][:, year]
        usable = (discount > 0) & (base > 1e-9) & (persistency > 1e-9)
        if usable.sum() < min_paths:
            continue

        # Per unit of benefit base, which is where the homogeneity is used, and per unit of
        # persistency, which is what makes the target a value at the node rather than a value
        # times the chance of reaching it.
        scale = np.where(usable, discount * persistency * base, 1.0)
        claim_target = future_claims[:, year + 1] / scale
        fee_target = future_fees[:, year + 1] / scale

        account = recorded["account_value"][:, year]
        moneyness, volatility, rate_level, _ = _state_parts(
            account, base, recorded["variance"][:, year], recorded["zero_10y"][:, year]
        )
        fit_rows = usable & train
        if fit_rows.sum() < min_paths:
            continue

        fits[year] = _fit_year(
            moneyness, volatility, rate_level, claim_target, fee_target, fit_rows, ridge,
            live=account > 1e-8, score_rows=usable & test,
        )

        # The same year, seen an instant before its anniversary. Its state is the contract
        # after the year's growth and before the year's charges, withdrawal and step-up, so its
        # contract value can sit above its benefit base - and that is the only kind of state a
        # hedge sees between anniversaries. Its target is everything from this year's cash
        # flows onward, because none of them have happened yet.
        pre_base = recorded["pre_benefit_base"][:, year]
        pre_usable = (discount > 0) & (pre_base > 1e-9) & (persistency > 1e-9)
        pre_scale = np.where(pre_usable, discount * persistency * pre_base, 1.0)
        pre_account = recorded["pre_account_value"][:, year]
        pre_moneyness, _, _, _ = _state_parts(
            pre_account, pre_base, recorded["variance"][:, year], recorded["zero_10y"][:, year]
        )
        pre_rows = pre_usable & train
        if pre_rows.sum() >= min_paths:
            pre_fits[year] = _fit_year(
                pre_moneyness, volatility, rate_level,
                future_claims[:, year] / pre_scale, future_fees[:, year] / pre_scale,
                pre_rows, ridge, live=pre_account > 1e-8, score_rows=pre_usable & test,
            )

        scored = usable & test
        fitted, _ = ProxyFit(fits={year: fits[year]}, order=order, n_paths=n_paths,
                             diagnostics=pd.DataFrame()).value(
            year, account[scored], base[scored], recorded["variance"][:, year][scored],
            recorded["zero_10y"][:, year][scored], attribution=1.0, flag_extrapolation=False,
        )
        actual = (claim_target[scored] - fee_target[scored]) * base[scored]
        finite = np.isfinite(fitted)
        residual = fitted[finite] - actual[finite]
        total = actual[finite] - actual[finite].mean()
        rows.append({
            "year": year,
            "n_train": int(fit_rows.sum()),
            "n_test": int(scored.sum()),
            "n_exhausted": fits[year].n_exhausted,
            "smoothing": fits[year].smoothing,
            "r_squared": float(1.0 - (residual @ residual) / (total @ total))
            if total @ total > 0 else np.nan,
            "rmse": float(np.sqrt((residual @ residual) / max(residual.size, 1))),
            "mean_benefit_base": float(base[scored].mean()),
        })

    if not fits:
        raise ValueError("no year had enough surviving paths to fit")

    return ProxyFit(fits=fits, pre_fits=pre_fits, order=order, n_paths=n_paths,
                    diagnostics=pd.DataFrame(rows))


def _fit_year(moneyness, volatility, rate, claim_target, fee_target, fit_rows, ridge,
              live, score_rows=None) -> _YearFit:
    """One year's coefficients for both legs, plus the record of where its design points fell."""
    knots = _knots(moneyness[fit_rows], live[fit_rows])
    design = _design(moneyness, volatility, rate, knots)
    # Square roots, because the solve multiplies design and target by these and the objective
    # is the square of what it sees.
    weights = np.sqrt(_replicate_weights(moneyness[fit_rows]))
    penalty = _roughness(knots, design.shape[1])
    if score_rows is None or score_rows.sum() < 100:
        score_rows = fit_rows
    strength, _ = _choose_smoothing(
        design[fit_rows], claim_target[fit_rows], weights, penalty, ridge,
        design[score_rows], claim_target[score_rows],
    )
    observed = np.column_stack([moneyness, volatility, rate])
    n_bins = int(np.ceil(MONEYNESS_CAP / SUPPORT_BIN)) + 1
    support = np.bincount(
        np.clip((moneyness[fit_rows] / SUPPORT_BIN).astype(int), 0, n_bins - 1),
        minlength=n_bins,
    )
    return _YearFit(
        claim=_penalised_solve(design[fit_rows], claim_target[fit_rows], ridge, weights,
                               penalty, strength),
        fee=_penalised_solve(design[fit_rows], fee_target[fit_rows], ridge, weights,
                             penalty, strength),
        knots=knots,
        smoothing=float(strength),
        lower=np.quantile(observed[fit_rows], DESIGN_TAIL, axis=0),
        upper=np.quantile(observed[fit_rows], 1.0 - DESIGN_TAIL, axis=0),
        support=support,
        min_support=max(MIN_SUPPORT_ROWS, MIN_SUPPORT_SHARE * fit_rows.sum()),
        n_exhausted=int((live[fit_rows] == False).sum()),  # noqa: E712 - count, not a filter
    )


def accuracy_report(proxy: ProxyFit, account_value: float) -> pd.DataFrame:
    """Fit quality in the unit that decides whether the proxy is good enough.

    An R-squared against single-path realisations is not that unit and never will be: the
    residual there is dominated by payoff noise no conditional mean can remove. This is a
    diagnostic for watching the fit across years, and ``nested.summarise`` is the test.
    """
    frame = proxy.diagnostics.copy()
    frame["rmse_pct_of_account"] = frame["rmse"] / account_value
    frame["exhausted_share"] = frame["n_exhausted"] / proxy.n_paths
    return frame
