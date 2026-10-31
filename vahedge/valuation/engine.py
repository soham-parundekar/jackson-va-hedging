"""One function that values the book in a given market state, and everything else calls it.

Greeks, the break-even fee, the hedge backtest, the capital lenses and every validation are all
the same operation repeated: put the book in a market state and ask what it is worth. Writing
that once, making it deterministic given a seed, and caching it is what makes the rest of the
project affordable.

What gets valued is the market risk benefit on Jackson's own definition, because that is the
only thing a single-contract model and a $236bn disclosure have in common:

    MRB = PV(projected benefits) - attribution * PV(projected fees)

Note 6: "the fair value of the MRB is measured as the difference between the present value of
projected future guaranteed benefits and the present value of projected attributed fees... At
inception of the contract, the Company attributes a percentage of total projected future fees
expected to be assessed against the policyholder to offset the projected future guaranteed
benefits over the lifetime of the contract." The percentage is fixed at inception and held
static, and it cannot exceed 100%. Getting this right is what makes a variable annuity
guarantee sit on a balance sheet as a $4.2bn net asset rather than a liability, and a model
that prices only the guarantee leg is not comparable to the disclosure at all.

Two things about determinism, because everything downstream depends on them.

*The same seed gives the same draws.* A Greek is the difference between two valuations, and at
twenty thousand paths the standard error on the level is far larger than a one basis point rate
bump moves the value. Common random numbers turn that noise into a shared offset that cancels
in the difference. Every shocked valuation reuses the base valuation's seed.

*An equity shock needs no new simulation.* Under Heston the return distribution does not depend
on the index level, so shocking the index changes the account value the paths start from and
nothing else. Rate and volatility shocks do change the paths and do re-simulate. That asymmetry
is worth a factor of two on a full Greek set.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields, replace

import numpy as np
import pandas as pd

from ..liability import gmwb
from ..liability.mortality import MortalityTable
from ..market.curves import NelsonSiegelSvensson
from ..market.heston_cos import HestonParameters
from ..market.hull_white import HullWhite
from ..market.simulate import Correlations, MarketPaths, SubAccountMix, simulate

DEFAULT_PATHS = 20_000
DEFAULT_SEED = 20251231
# The lowest volatility a shock is allowed to land on, in decimal. Squared it is the 1e-8
# variance floor that has always been the intent here.
VOL_FLOOR = 1e-4


def shocked_variance(volatility: float, shock: float) -> float:
    """A variance that is still a variance after the volatility has been shocked.

    The floor goes on the volatility and not on the variance, which sounds like a detail and is
    not. Clamping after squaring cannot catch anything: a shock that takes a 14.1% volatility
    down 20 points comes back as the square of minus 5.9%, so the clamp passes it and the state
    reports a 5.9% volatility for a shock meant to crush volatility to nothing. Nothing in the
    project reaches that far today - the vega bump is one point - but a stress grid would, and
    a reflected volatility is worse than a refused one because it is signed the wrong way.
    """
    return float(max(volatility + shock, VOL_FLOOR) ** 2)


@dataclass(frozen=True)
class MarketState:
    """Everything about one valuation date that the book is priced off."""

    curve: NelsonSiegelSvensson
    heston: HestonParameters
    correlations: Correlations
    mix: SubAccountMix
    mean_reversion: float          # Hull-White a
    rate_vol: float                # Hull-White sigma
    valuation_year: int
    credit_spread: float = 0.0     # own non-performance risk, zero for the economic value

    @classmethod
    def from_calibration(cls, calibration, credit_spread: float = 0.0) -> "MarketState":
        """The saved calibration, as the engine wants it. See vahedge.market.state."""
        return cls(
            curve=calibration.curve,
            heston=calibration.heston,
            correlations=calibration.correlations,
            mix=calibration.mix,
            mean_reversion=calibration.mean_reversion,
            rate_vol=calibration.rate_vol,
            valuation_year=int(calibration.as_of.year),
            credit_spread=credit_spread,
        )

    def hull_white(self) -> HullWhite:
        return HullWhite(a=self.mean_reversion, sigma=self.rate_vol, curve=self.curve)

    def with_shocks(
        self,
        rate_shock_bp: float = 0.0,
        vol_shock: float = 0.0,
    ) -> "MarketState":
        """A shocked copy.

        The rate shock is a parallel shift of the fitted zero curve, applied by moving the
        level parameter, which shifts both the zero rates and the instantaneous forward by the
        same amount and leaves the shape alone. That is what a parallel shift means, and it is
        what Item 7A describes.

        The volatility shock moves the current variance and the long-run level by the same
        number of volatility points. Moving only the current variance would be a shock the
        market can trade but would decay out within months on a mean reversion of 4.8, leaving
        a forty-year liability almost untouched; moving only the long-run level would be a
        shock nothing trades. Moving both is the honest middle, and greeks.py reports the two
        separately so the split is visible rather than buried.
        """
        state = self
        if rate_shock_bp:
            shifted = replace(self.curve, beta0=self.curve.beta0 + rate_shock_bp / 10000.0)
            state = replace(state, curve=shifted)
        if vol_shock:
            heston = self.heston
            state = replace(state, heston=replace(
                heston,
                v0=shocked_variance(np.sqrt(heston.v0), vol_shock),
                theta=shocked_variance(heston.long_run_vol, vol_shock),
            ))
        return state


@dataclass(frozen=True)
class BookValuation:
    """What the book is worth, and the pieces the reader needs to believe it."""

    market_risk_benefit: float
    pv_claims: float
    pv_death_claims: float
    pv_fees: float
    attribution: float
    std_error: float
    account_value: float
    benefit_base: float
    by_cohort: pd.DataFrame
    projection: gmwb.GmwbProjection
    # The paths this was valued on, carried so a caller differencing two valuations can ask
    # them how their draws pair rather than assuming a pairing. See MarketPaths.
    paths: MarketPaths

    @property
    def mrb_pct_of_account(self) -> float:
        return self.market_risk_benefit / self.account_value

    @property
    def total_claims(self) -> float:
        return self.pv_claims + self.pv_death_claims


class Valuer:
    """Holds the fixed inputs and caches simulated paths across revaluations.

    The cache is keyed on every field of every market object, the path count and the seed. An
    equity shock is deliberately not part of the key, because it does not change the paths.

    Reading the fields off the dataclasses rather than naming them is the second version of this
    key. The first named the fifteen that were thought to matter and missed ``bond_duration``,
    which the rolling bond return is built from: doubling it moves a fund return by half a per
    cent and the cache handed back the base paths and reported no difference. Nothing in the
    project varies it, so no published figure was ever wrong, but a duration sensitivity would
    have come out flat and looked like a finding. A key that cannot fall behind the dataclass is
    worth the handful of fields in it that have no effect - the curve's own fit error is in there
    now, and the cost of that is nothing, because it never moves without the betas moving too.

    It is also bounded. One set of paths for a hundred cohorts over sixty years is tens of
    megabytes, and a Greek set visits half a dozen market states while a hedge backtest visits
    one per rebalance date. An unbounded cache turns that into gigabytes and the process dies
    without saying why, which is how this limit came to exist. Four is enough to keep a bumped
    pair and its base resident.
    """

    def __init__(
        self,
        mortality: MortalityTable,
        n_paths: int = DEFAULT_PATHS,
        seed: int = DEFAULT_SEED,
        steps_per_year: int = 24,
        male_weight: float = 0.5,
        cache_size: int = 4,
    ):
        self.mortality = mortality
        self.n_paths = int(n_paths)
        self.seed = int(seed)
        self.steps_per_year = int(steps_per_year)
        self.male_weight = float(male_weight)
        self._paths: "OrderedDict[tuple, MarketPaths]" = OrderedDict()
        self._mortality_cache: dict[tuple, tuple] = {}
        self.cache_size = int(cache_size)
        self.simulations = 0

    def paths_for(self, state: MarketState, n_years: int) -> MarketPaths:
        key = (
            _field_values(state.heston), _field_values(state.curve),
            _field_values(state.correlations), _field_values(state.mix),
            state.mean_reversion, state.rate_vol,
            n_years, self.n_paths, self.seed, self.steps_per_year,
        )
        if key in self._paths:
            self._paths.move_to_end(key)
            return self._paths[key]
        self._paths[key] = simulate(
            state.heston, state.hull_white(), state.correlations, state.mix,
            n_years=n_years, n_paths=self.n_paths, seed=self.seed,
            steps_per_year=self.steps_per_year,
        )
        self.simulations += 1
        while len(self._paths) > self.cache_size:
            self._paths.popitem(last=False)
        return self._paths[key]

    def mortality_for(self, book, valuation_year: int, n_years: int):
        """Survival and year-of-death probabilities, cached on what produced them.

        The sex mix is part of the key even though nothing in the project changes it mid-run.
        Leaving it out would make a sex-mix sensitivity return the base case's rates and report
        no difference, which is the kind of silent zero a robustness table cannot survive.
        """
        key = (tuple(int(a) for a in book.attained_age), valuation_year, n_years,
               self.male_weight)
        if key not in self._mortality_cache:
            self._mortality_cache[key] = self.mortality.rates(
                book.attained_age, valuation_year, n_years, self.male_weight
            )
        return self._mortality_cache[key]

    def value(
        self,
        book,
        state: MarketState,
        attribution=None,
        equity_shock: float = 0.0,
        with_death_benefit: bool = True,
        path_years: int | None = None,
    ) -> BookValuation:
        """Value the book, optionally under an instantaneous equity shock.

        ``attribution`` is the share of projected fees attributed to the guarantee, one number
        per cohort, fixed at each cohort's inception. Passing ``None`` calibrates it here, which
        is only right for a book valued at issue; ``calibrate_attribution`` is what the rest of
        the project uses.

        ``path_years`` simulates to a longer horizon than the book needs, so that two books with
        different horizons can be compared on the same draws. The simulator draws its normals in
        one array whose width depends on the horizon, so a forty-year run and a fifty-year run at
        the same seed are different worlds, not a prefix and its extension. Without this the
        truncation test compares two independent simulations and reports their Monte Carlo
        difference as a truncation effect: at twenty thousand paths that is around a hundred
        dollars of noise against a truncation effect of a few dollars, so the test would pass for
        the wrong reason.
        """
        n_years = int(book.projection_years.max())
        if path_years is not None:
            if path_years < n_years:
                raise ValueError(f"path_years {path_years} is shorter than the book's {n_years}")
            n_years = int(path_years)
        paths = self.paths_for(state, n_years)
        if state.credit_spread:
            paths = _apply_credit_spread(paths, state.credit_spread)

        survival, deaths = self.mortality_for(book, state.valuation_year, n_years)
        projection = gmwb.project(
            book, paths, survival,
            deaths=deaths if with_death_benefit else None,
            equity_shock=equity_shock,
            equity_weight=state.mix.equity_weight,
        )

        if attribution is None:
            attribution = _implied_attribution(projection)
        attribution = np.asarray(attribution, dtype=float)
        if attribution.shape != (book.size,):
            raise ValueError(f"attribution must have one entry per cohort, got {attribution.shape}")

        weight = book.weight
        shock_factor = 1.0 + state.mix.equity_weight * equity_shock
        per_cohort = projection.pv_total_claims - attribution * projection.pv_attributable_fees
        mrb = float(np.sum(weight * per_cohort))

        # Standard error across paths, on the aggregate rather than cohort by cohort, because
        # the cohorts share their paths and their errors are anything but independent.
        path_level = np.einsum(
            "c,cp->p", weight, projection.claim_paths
        ) - np.einsum("c,cp->p", weight * attribution, projection.fee_paths)
        std_error = _standard_error(path_level, paths)

        by_cohort = book.to_frame()
        by_cohort["pv_claims"] = projection.pv_claims
        by_cohort["pv_death_claims"] = projection.pv_death_claims
        by_cohort["pv_fees"] = projection.pv_attributable_fees
        by_cohort["attribution"] = attribution
        by_cohort["mrb"] = per_cohort
        shocked_account = book.account_value * shock_factor
        by_cohort["mrb_pct_of_account"] = np.where(
            shocked_account > 0.0,
            per_cohort / np.where(shocked_account > 0.0, shocked_account, 1.0),
            np.nan,
        )

        return BookValuation(
            market_risk_benefit=mrb,
            pv_claims=float(np.sum(weight * projection.pv_claims)),
            pv_death_claims=float(np.sum(weight * projection.pv_death_claims)),
            pv_fees=float(np.sum(weight * projection.pv_attributable_fees)),
            attribution=float(np.sum(weight * attribution)),
            std_error=std_error,
            account_value=float(np.sum(weight * book.account_value)) * shock_factor,
            benefit_base=float(np.sum(weight * book.benefit_base)),
            by_cohort=by_cohort,
            projection=projection,
            paths=paths,
        )

    def calibrate_attribution(self, book, state: MarketState, with_death_benefit: bool = True):
        """The attribution percentage each cohort's contract would have been given at issue.

        Jackson fixes the percentage at inception, so a cohort six years in force carries the
        percentage set six years ago under that date's market. This reprices each cohort as a
        contract at issue - account value and benefit base both at premium, the full deferral
        ahead of it - but under today's market, because the market of six years ago is
        recoverable and the behaviour and mortality assumptions in force then are not. The
        direction of the resulting error is knowable: attribution is capped at one, and most
        cohorts sit below the cap, so a cohort issued into a lower-rate market would have been
        given a higher percentage than this produces.
        """
        at_issue = _rewind_to_issue(book)
        n_years = int(at_issue.projection_years.max())
        survival, deaths = self.mortality_for(at_issue, state.valuation_year, n_years)
        projection = gmwb.project(
            at_issue,
            self.paths_for(state, n_years),
            survival,
            deaths=deaths if with_death_benefit else None,
            equity_weight=state.mix.equity_weight,
        )
        return _implied_attribution(projection)


def _field_values(obj) -> tuple:
    """Every field of a market dataclass, in order, for use in the path cache key."""
    return tuple(getattr(obj, field.name) for field in fields(obj))


def _implied_attribution(projection: gmwb.GmwbProjection) -> np.ndarray:
    """min(1, PV(claims) / PV(fees)), which is the definition written out.

    Where attributable fees cover projected claims the benefit starts at a fair value of zero,
    and where they do not the percentage is capped at one and an MRB liability is recognised.
    """
    fees = projection.pv_attributable_fees
    if np.any(fees <= 0):
        raise ValueError("a cohort has no attributable fees; its attribution is undefined")
    return np.minimum(1.0, projection.pv_total_claims / fees)


def _rewind_to_issue(book):
    """The same cohorts as new business: at premium, at full deferral, nothing ratcheted yet."""
    from ..liability.cohorts import CohortBook, gwb_adjustment_index
    from ..liability.terms import BONUS_PERIOD_YEARS

    premium = book.premium_at_issue
    issue_age = book.issue_age
    deferral = book.deferral_years + book.years_since_issue
    adjustment_year = gwb_adjustment_index(issue_age, 0, deferral)

    fields = dict(book.__dict__)
    fields.update(
        years_since_issue=np.zeros_like(book.years_since_issue),
        attained_age=issue_age,
        account_value=premium.copy(),
        benefit_base=premium.copy(),
        bonus_base=premium.copy(),
        deferral_years=deferral,
        bonus_years_remaining=np.full_like(book.bonus_years_remaining, BONUS_PERIOD_YEARS),
        adjustment_year=adjustment_year,
        death_benefit_base=premium.copy(),
        projection_years=book.projection_years + book.years_since_issue,
    )
    return CohortBook(**fields)


def _apply_credit_spread(paths: MarketPaths, spread: float) -> MarketPaths:
    """Discount at the risk-free curve plus the insurer's own non-performance spread.

    Note 6: "Non-performance risk is incorporated into the calculation through the adjustment of
    the risk-free rate curve based on credit spreads for debt and debt-like instruments issued
    by the Company or its insurance operating subsidiaries." A flat spread on top of the
    stochastic discount factor is the simplest thing that does that, and it is enough to
    separate the part of the reported movement that runs through other comprehensive income
    from the part that reaches net income.
    """
    years = np.arange(1, paths.n_years + 1, dtype=float)
    return replace(paths, discount=paths.discount * np.exp(-spread * years)[None, :])


def _standard_error(sample: np.ndarray, paths) -> float:
    """Monte Carlo error, off the antithetic pairs where the paths carry them.

    The pairing comes from the paths rather than from a halving rule here, because which rows
    mirror which is a fact about how they were simulated; see ``MarketPaths.antithetic_pairs``.
    """
    pairs = paths.antithetic_pairs(sample)
    if pairs is None:
        return float(sample.std(ddof=1) / np.sqrt(sample.size))
    return float(pairs.std(ddof=1) / np.sqrt(pairs.size))
