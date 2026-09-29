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
N_KNOTS = 8              # interior knots of the moneyness spline
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


def _knots(moneyness: np.ndarray, n_knots: int = N_KNOTS) -> np.ndarray:
    """Interior knots at quantiles of the design distribution.

    Quantiles rather than an even grid, because the states the paths actually visit are
    concentrated: early on almost every contract sits near a moneyness of one, and only later
    does the distribution spread toward exhaustion. An even grid would put most of its knots
    where there is no data and none where the curvature is.
    """
    quantiles = np.linspace(0.0, 1.0, n_knots + 2)[1:-1]
    interior = np.quantile(moneyness, quantiles)
    return np.unique(np.round(interior, 10))


def _design(moneyness, variance_vol, rate, knots: np.ndarray) -> np.ndarray:
    """Linear spline in moneyness, low order in volatility and the rate, plus interactions.

    A high-order polynomial was the first thing tried here and it failed in both directions: it
    was flat where the value function turns, and it oscillated wildly at the ends where a few
    extreme paths sit. The value per unit of benefit base is a smooth monotone function of the
    contract value ratio that bends sharply as the contract approaches exhaustion - the shape of
    a smoothed put payoff - and a linear spline with knots at the data's own quantiles follows
    that shape without any ability to blow up between them.

    Volatility and the rate get linear and quadratic terms only. Their effect on the value is
    gentle across the whole range, and spending basis functions there instead of on moneyness
    was what produced an R-squared of 0.22 against nested values.
    """
    moneyness = np.asarray(moneyness, dtype=float)
    columns = [np.ones(moneyness.size), moneyness]
    for knot in knots:
        columns.append(np.maximum(moneyness - knot, 0.0))
    for extra in (variance_vol, rate):
        extra = np.asarray(extra, dtype=float)
        columns.append(extra)
        columns.append(extra ** 2)
        columns.append(extra * moneyness)
    columns.append(np.asarray(variance_vol, float) * np.asarray(rate, float))
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
        attribution: float = 1.0, bump: float = 0.01,
    ):
        """Equity exposure by bumping the proxy's own state rather than re-simulating.

        A move in the index moves the contract value and leaves the benefit base alone, which is
        the whole point of the guarantee, so a proportional bump to the contract value is
        exactly the right perturbation. The proxy is smooth, so a small central difference on it
        is stable in a way a bumped simulation is not.
        """
        account = np.asarray(account_value, dtype=float)
        up, _ = self.value(year, account * (1.0 + bump), benefit_base, variance, zero_10y,
                           attribution, flag_extrapolation=False)
        down, _ = self.value(year, account * (1.0 - bump), benefit_base, variance, zero_10y,
                             attribution, flag_extrapolation=False)
        return (up - down) / np.log((1.0 + bump) / (1.0 - bump))


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

        knots = _knots(moneyness[fit_rows])
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
