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

from dataclasses import dataclass

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
# Exhausted contracts are not a dense region of the design space, they are one state repeated:
# every one of them sits at a moneyness of exactly zero. By year twenty-five that single state
# is ninety-two per cent of the fitting rows, and least squares will happily give up the live
# region - where the guarantee is actually decided - to shave a residual at a point it has
# already pinned thirty thousand times over. So the replicates are given a weight that makes
# them count as this many rows. Two thousand independent realisations estimate a conditional
# mean far more precisely than anything else in the fit, so nothing is lost by the cap, and
# only the exhausted state is capped: a dense but genuinely varying region carries information
# in every row and is left alone.
EXHAUSTED_EFFECTIVE_ROWS = 2_000


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

    def thin(self, moneyness: np.ndarray) -> np.ndarray:
        """True where the fitting paths left this part of the moneyness axis empty."""
        index = np.clip((moneyness / SUPPORT_BIN).astype(int), 0, self.support.size - 1)
        return self.support[index] < self.min_support


@dataclass(frozen=True)
class ProxyFit:
    """Fitted coefficients by year, with the design range they are valid over."""

    fits: dict
    order: int
    n_paths: int
    diagnostics: pd.DataFrame

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
    ):
        """The market risk benefit at a state, per the fitted year.

        Returns (value, outside). The value is still returned for states beyond the design
        range, because refusing outright would stop a backtest on its worst day, but it is
        marked so the caller can report how often it happened rather than discover it in a
        result.
        """
        if year not in self.fits:
            raise KeyError(f"no fit for year {year}; fitted years are {self.years}")
        fit_year = self.fits[year]

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
        attribution: float = 1.0,
    ) -> dict:
        """Value, delta, gamma, vega and rho at a state, all from the fitted basis.

        The four derivatives in one call because a hedge needs them together and they share
        every intermediate. All of them are analytic, which matters more here than it does for
        the value: a bumped second derivative on a spline either straddles a knot or sits inside
        double precision's floor, and a bumped vega has to re-clip the state.

        Units match the Greeks module, so a hedge can be sized off either without conversion.
        Delta and gamma are per unit log move in the contract value, vega is per volatility
        point, and rho is per basis point of the ten-year zero rate with the curve's shape held.
        That last one is a one-factor rate exposure and is all a one-factor rate state can
        support; a curve-shape hedge would need a slope state in the basis and does not get one.
        """
        if year not in self.fits:
            raise KeyError(f"no fit for year {year}; fitted years are {self.years}")
        fit_year = self.fits[year]

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
        return {
            "value": per_unit * unit,
            "delta": delta,
            # d2V/d(ln AV)2 = AV f'(m) + AV m f''(m), the second term being the curvature of
            # the value per unit of benefit base.
            "gamma": delta + account * moneyness * curvature,
            "vega": unit * in_vol,
            "rho_per_bp": unit * in_rate * 1e-4,
        }

    def greeks_at(
        self, years_since_issue: float, account_value, benefit_base, variance, zero_10y,
        attribution: float = 1.0,
    ) -> dict:
        """The same, between anniversaries, by interpolating the two fits that bracket the date.

        Each annual fit is the value at the end of its own policy year, so the fit keyed ``k``
        applies at ``k + 1`` years since issue. A weekly hedge sits between two of them, and
        taking the nearer one would step the liability's value at every anniversary and put a
        jump into the P&L that is an artefact of the fitting grid rather than anything the
        contract does. Interpolating linearly in time removes the jump and makes the slope
        between anniversaries the liability's time decay, which is what the attribution's theta
        term needs.
        """
        keys = self.years
        lower_key = int(np.floor(years_since_issue)) - 1
        lower_key = min(max(lower_key, keys[0]), keys[-1])
        upper_key = min(lower_key + 1, keys[-1])
        if upper_key not in self.fits:
            upper_key = lower_key
        weight = float(np.clip(years_since_issue - (lower_key + 1), 0.0, 1.0))

        first = self.greeks(lower_key, account_value, benefit_base, variance, zero_10y,
                            attribution)
        if upper_key == lower_key or weight == 0.0:
            return first
        second = self.greeks(upper_key, account_value, benefit_base, variance, zero_10y,
                             attribution)
        return {name: (1.0 - weight) * first[name] + weight * second[name] for name in first}


def _replicate_weights(account_value: np.ndarray) -> np.ndarray:
    """One for a live contract; for the exhausted replicates, whatever makes them count as
    ``EXHAUSTED_EFFECTIVE_ROWS`` between them."""
    weights = np.ones(account_value.size)
    spent = account_value <= 1e-8
    n_spent = int(spent.sum())
    if n_spent > EXHAUSTED_EFFECTIVE_ROWS:
        weights[spent] = EXHAUSTED_EFFECTIVE_ROWS / n_spent
    return weights


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

    fits, rows = {}, []
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

        knots = _knots(moneyness[fit_rows], account[fit_rows] > 1e-8)
        design = _design(moneyness, volatility, rate_level, knots)
        # Square roots, because the solve multiplies design and target by these and the
        # objective is the square of what it sees.
        weights = np.sqrt(_replicate_weights(account[fit_rows]))
        claim_beta = _ridge_solve(design[fit_rows], claim_target[fit_rows], ridge, weights)
        fee_beta = _ridge_solve(design[fit_rows], fee_target[fit_rows], ridge, weights)

        observed = np.column_stack([moneyness, volatility, rate_level])
        n_bins = int(np.ceil(MONEYNESS_CAP / SUPPORT_BIN)) + 1
        support = np.bincount(
            np.clip((moneyness[fit_rows] / SUPPORT_BIN).astype(int), 0, n_bins - 1),
            minlength=n_bins,
        )
        fits[year] = _YearFit(
            claim=claim_beta, fee=fee_beta, knots=knots,
            lower=np.quantile(observed[fit_rows], DESIGN_TAIL, axis=0),
            upper=np.quantile(observed[fit_rows], 1.0 - DESIGN_TAIL, axis=0),
            support=support,
            min_support=max(MIN_SUPPORT_ROWS, MIN_SUPPORT_SHARE * fit_rows.sum()),
            n_exhausted=int((usable & (account <= 1e-8)).sum()),
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
            "r_squared": float(1.0 - (residual @ residual) / (total @ total))
            if total @ total > 0 else np.nan,
            "rmse": float(np.sqrt((residual @ residual) / max(residual.size, 1))),
            "mean_benefit_base": float(base[scored].mean()),
        })

    if not fits:
        raise ValueError("no year had enough surviving paths to fit")

    return ProxyFit(fits=fits, order=order, n_paths=n_paths, diagnostics=pd.DataFrame(rows))


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
