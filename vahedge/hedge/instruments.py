"""The six instrument types Jackson names, priced well enough to hedge with and no better.

Item 7A lists the derivative book by category: equity futures, total return swaps, equity
options, interest-rate futures, bond forwards and interest-rate swaps. Those are what this
module builds, because a hedging study that trades something Jackson does not is answering a
different question. What each one is here for:

*Equity futures* are the delta instrument. Daily mark to market, no premium, almost no cost,
and no convexity at all - which is the point of the experiments, since the liability has plenty.

*Total return swaps* do the same job through a financing leg instead of a futures basis. They
matter because their cost is a running spread rather than a round trip, so the two instruments
rank differently depending on how often the hedge is rebalanced.

*Index puts* are the only listed instrument that carries gamma and vega. They cost premium,
and whether that premium is worth paying is the question E1 and E3 exist to answer.

*Interest-rate futures*, *bond forwards* and *interest-rate swaps* are all rate instruments and
differ only in duration and cost: about six and a half years for the note contract, fifteen for
the long bond, and whatever tenor the swap is struck at. A forty-year liability needs the long
end, and the experiments show what happens when it is hedged at the wrong point on the curve.

Two things about the pricing are approximations and are labelled as such wherever a result
depends on them.

*Historical option prices are model prices, not traded prices.* There is no free history of SPX
option quotes, so a backtest has to price its own hedges. The volatility used is the index
volatility of the day scaled by a skew multiplier read off the fitted Heston surface, which
gives a smile with the right shape anchored to the right level but is not what anyone could have
dealt at. Every result that involves puts carries that caveat.

*Rate instruments are duration proxies.* A note future is modelled as a position with a fixed
effective duration rather than by valuing a deliverable basket with a cheapest-to-deliver
option. For a hedge whose rate exposure is a parallel shift on a single-factor curve that is the
right level of detail; it would not be for a curve-shape hedge, which is why none is claimed.

Sign conventions, written out because a sign error here produces a plausible-looking result that
is exactly wrong. Every exposure is the derivative of the instrument's own value with respect to
the risk factor, from the holder's point of view: a long position in the instrument. Equity
exposure is per unit log move in the index, rho is per basis point of parallel shift, vega is
per volatility point. The liability side of the hedge equation is handled in sizing.py, where
the insurer's position in its own guarantee is the negative of the market risk benefit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from ..market.curves import par_equivalent
from ..market.heston_cos import black76, cos_price, implied_vol
from ..valuation.greeks import EQUITY_CURVATURE_STEP

# Round-trip transaction costs, as a share of traded notional unless noted. These are
# assumptions, not quotes; scripts/run_hedge_experiments.py sweeps half, one and two times.
FUTURES_COST = 0.00005          # half a basis point of notional
TRS_COST = 0.00005              # execution only; the financing spread is the real cost
RATE_FUTURES_COST = 0.000025    # a quarter of a basis point
BOND_FORWARD_COST = 0.00005
SWAP_COST = 0.000025
OPTION_COST_VOL_POINTS = 0.005  # half a volatility point one way, charged against vega

NOTE_FUTURE_DURATION = 6.5      # Jackson discloses an effective duration near this
LONG_BOND_DURATION = 15.0


@dataclass(frozen=True)
class Exposures:
    """What one unit of something is sensitive to, in dollars per unit of risk factor."""

    delta: float = 0.0      # d(value) / d(log index)
    gamma: float = 0.0      # d2(value) / d(log index)^2
    vega: float = 0.0       # d(value) / d(volatility), per point
    rho: float = 0.0        # d(value) / d(parallel rate shift), per basis point

    ORDER = ("delta", "gamma", "vega", "rho")

    def as_array(self) -> np.ndarray:
        return np.array([self.delta, self.gamma, self.vega, self.rho], dtype=float)

    def __add__(self, other: "Exposures") -> "Exposures":
        return Exposures(*(self.as_array() + other.as_array()))

    def __mul__(self, scale: float) -> "Exposures":
        return Exposures(*(self.as_array() * float(scale)))

    __rmul__ = __mul__

    @classmethod
    def from_array(cls, values) -> "Exposures":
        return cls(*np.asarray(values, dtype=float))


@dataclass(frozen=True)
class SmileShape:
    """Implied volatility relative to at-the-money, by strike over spot.

    Built once from the calibrated surface and then carried through history unchanged. That is
    the approximation: the level of volatility moves day to day with the index volatility
    series, and the shape of the smile is held at the shape the current chain implies. A sharper
    version would move the shape with the Cboe skew index, which is an extension rather than a
    correction, because a single skew number does not pin a curve either.
    """

    strike_over_spot: np.ndarray
    multiplier: np.ndarray

    def __call__(self, strike_over_spot) -> np.ndarray:
        return np.interp(np.asarray(strike_over_spot, dtype=float),
                         self.strike_over_spot, self.multiplier)


def smile_from_heston(heston, curve, maturity: float = 1.0,
                      moneyness=np.arange(0.50, 1.301, 0.01)) -> SmileShape:
    """The fitted surface's own skew, normalised to one at the money.

    Puts across the strike range are priced with the COS method, converted to implied
    volatility, and divided by the at-the-money level. What survives is the shape, which is
    what a historical volatility level has to be dressed in to price a put off a strike it was
    never quoted at.
    """
    moneyness = np.asarray(moneyness, dtype=float)
    discount = float(curve.discount(maturity))
    forward = 1.0 / discount           # unit spot, so the forward is the carry factor
    prices = cos_price(heston, forward, moneyness, maturity, discount,
                       np.zeros(moneyness.size, dtype=bool))
    vols = np.array([
        implied_vol(float(price), forward, float(strike), maturity, discount, False)
        for price, strike in zip(prices, moneyness)
    ])
    usable = np.isfinite(vols)
    if usable.sum() < 4:
        raise ValueError("the surface did not produce enough implied volatilities for a smile")
    at_the_money = float(np.interp(1.0, moneyness[usable], vols[usable]))
    return SmileShape(strike_over_spot=moneyness[usable],
                      multiplier=vols[usable] / at_the_money)


@dataclass(frozen=True)
class HedgeMarket:
    """One day as the hedge book sees it.

    Deliberately smaller than the valuation's ``MarketState``: a hedge is marked off the index
    level, the curve, and one volatility number dressed in a fixed smile. The liability is
    valued off the full model, and the gap between the two is part of what the backtest
    measures rather than something to paper over.
    """

    index: float
    curve: object                  # anything with discount() and zero()
    volatility: float              # at-the-money implied, decimal
    smile: SmileShape
    dividend_yield: float = 0.0
    financing_rate: float | None = None   # overnight funding; defaults to the short end

    def short_rate(self) -> float:
        if self.financing_rate is not None:
            return float(self.financing_rate)
        return float(self.curve.zero(1.0 / 12.0))

    def forward(self, maturity: float) -> float:
        """Index forward from the curve and the dividend yield, which is how a future prices."""
        return self.index * math.exp(
            (float(self.curve.zero(maturity)) - self.dividend_yield) * maturity
        )

    def strike_volatility(self, strike_over_spot: float) -> float:
        return self.volatility * float(self.smile(strike_over_spot))


# ---------------------------------------------------------------- equity instruments


@dataclass(frozen=True)
class EquityFuture:
    """One index unit of futures. Value is the mark since the last settlement, so zero on a
    freshly struck position; the P&L is the change in the forward."""

    maturity: float = 0.25
    label: str = "equity future"

    def notional(self, market: HedgeMarket) -> float:
        return market.forward(self.maturity)

    def value(self, market: HedgeMarket, struck_at: float | None = None) -> float:
        forward = market.forward(self.maturity)
        return 0.0 if struck_at is None else forward - struck_at

    def exposures(self, market: HedgeMarket) -> Exposures:
        forward = market.forward(self.maturity)
        # The forward carries at the zero rate to its own maturity, so a parallel shift moves
        # it by F * tau. Small against the rate instruments, and left in rather than dropped
        # because the sizing solve should see it.
        return Exposures(delta=forward, rho=forward * self.maturity * 1e-4)

    def trade_cost(self, units_traded: float, market: HedgeMarket) -> float:
        return abs(units_traded) * self.notional(market) * FUTURES_COST

    def carry(self, units: float, market: HedgeMarket, years: float) -> float:
        return 0.0   # margin is funded from cash, which the simulator accrues separately


@dataclass(frozen=True)
class TotalReturnSwap:
    """Receive the index total return, pay financing. One unit is one index unit of notional."""

    maturity: float = 1.0
    spread: float = 0.0035         # over the funding rate; an assumption, swept in E3
    label: str = "total return swap"

    def notional(self, market: HedgeMarket) -> float:
        return market.index

    def value(self, market: HedgeMarket, struck_at: float | None = None) -> float:
        return 0.0 if struck_at is None else market.index - struck_at

    def exposures(self, market: HedgeMarket) -> Exposures:
        # Full index exposure, and a financing leg whose present value moves with rates over
        # the remaining life of the swap.
        return Exposures(delta=market.index, rho=-market.index * self.maturity * 1e-4)

    def trade_cost(self, units_traded: float, market: HedgeMarket) -> float:
        return abs(units_traded) * self.notional(market) * TRS_COST

    def carry(self, units: float, market: HedgeMarket, years: float) -> float:
        """The spread, which is what makes this different from a future.

        A short equity hedge pays the spread whichever way the position points, because the
        dealer charges it on notional rather than on direction.
        """
        return abs(units) * market.index * self.spread * years


@dataclass(frozen=True)
class IndexPut:
    """A listed put on one index unit, struck as a share of the spot at inception.

    The only instrument here with gamma and vega, and the only one that costs premium. Priced
    with Black-Scholes at the strike's own volatility from the smile, which is what makes it a
    model price: the whole point of a skew is that a put is not priced at the at-the-money
    level, and the whole limitation is that the shape comes from today's chain rather than the
    day being replayed.
    """

    maturity: float
    strike_over_spot: float
    strike: float | None = None     # fixed once struck; None means struck at this market
    label: str = "index put"

    def struck(self, market: HedgeMarket) -> "IndexPut":
        """The tradeable instrument, with its strike fixed at this market.

        An unstruck ``IndexPut`` is a specification - a tenor and a strike as a share of spot -
        and both its value and its exposures are evaluated as if it were struck at whatever
        market it is handed. That is what the sizing solve needs. A position being held has to
        be struck first, or it would silently re-strike itself every time the index moved and
        never show a profit.
        """
        return replace(self, strike=market.index * self.strike_over_spot)

    def _terms(self, market: HedgeMarket, maturity: float | None = None):
        maturity = self.maturity if maturity is None else maturity
        strike = self.strike if self.strike is not None else market.index * self.strike_over_spot
        discount = float(market.curve.discount(max(maturity, 1e-8)))
        forward = market.forward(max(maturity, 1e-8))
        vol = market.strike_volatility(strike / market.index)
        return maturity, strike, discount, forward, vol

    def notional(self, market: HedgeMarket) -> float:
        return market.index

    def value(self, market: HedgeMarket, maturity: float | None = None) -> float:
        maturity, strike, discount, forward, vol = self._terms(market, maturity)
        if maturity <= 1e-8:
            return max(strike - market.index, 0.0)
        return float(black76(forward, strike, maturity, vol, discount, is_call=False))

    def exposures(self, market: HedgeMarket, maturity: float | None = None,
                  delta_step: float = 1e-3, gamma_step: float = EQUITY_CURVATURE_STEP,
                  vol_step: float = 1e-4) -> Exposures:
        """Sensitivities of the price this instrument is actually marked at.

        Delta and gamma come from differences on ``value`` rather than from the textbook
        formulas, and the reason is the smile. The put is priced at its own strike's volatility,
        so a move in the index changes the strike's position on the smile and therefore its
        volatility. At a one-year ten per cent out-of-the-money strike that smile term is worth
        about ten dollars of delta per index unit against a Black-Scholes delta of twenty-three
        - it cuts the hedge ratio almost in half. Using the closed-form delta while marking at
        the smile price would leave the attribution unable to reconcile, and the residual it
        produced would be read as basis risk rather than as a derivative taken off the wrong
        function.

        Both are with respect to the log index, which is the unit the liability's Greeks are in.
        Vega is with respect to a shift in the whole volatility level, which is the only move a
        single quoted number can represent and the move the backtest actually feeds in; because
        a level shift scales the smile, the strike's own volatility moves by the multiplier and
        the vega comes out above the Black-Scholes figure by that factor. Rho stays closed-form,
        because the curve enters only through the forward and the discount factor and the smile
        does not depend on it.

        The two index steps are different sizes on purpose. A first difference is well resolved
        at a tenth of a per cent. A second difference at that step is not: the quantity being
        differenced is a part in ten million of a premium of order one, which is double
        precision's floor, and the first version of this returned a gamma of sixteen hundred
        where the real figure is a tenth of that.

        The width of the second difference is not a free numerical choice, which cost this a
        correction. Any step from about one to fifteen per cent is numerically safe, so the first
        version took two and the liability's regression took ten, and the solve then matched one
        definition of curvature against another. At a one-year ten per cent out-of-the-money
        strike the two differ by five per cent of the gamma, and more at shorter tenors where the
        option's own curvature is sharper. Both now come from EQUITY_CURVATURE_STEP.
        """
        maturity, strike, discount, forward, vol = self._terms(market, maturity)
        if maturity <= 1e-8 or vol <= 0:
            return Exposures()

        fixed = self if self.strike is not None else replace(self, strike=strike)

        def at(log_move: float) -> float:
            return fixed.value(replace(market, index=market.index * math.exp(log_move)),
                               maturity)

        here = fixed.value(market, maturity)
        delta = (at(delta_step) - at(-delta_step)) / (2.0 * delta_step)
        gamma = (at(gamma_step) - 2.0 * here + at(-gamma_step)) / gamma_step ** 2

        vol_up = replace(market, volatility=market.volatility + vol_step)
        vol_down = replace(market, volatility=max(market.volatility - vol_step, 1e-8))
        vega = (fixed.value(vol_up, maturity) - fixed.value(vol_down, maturity)) / (2.0 * vol_step)

        total = vol * math.sqrt(maturity)
        d2 = (math.log(forward / strike) - 0.5 * total ** 2) / total
        return Exposures(
            delta=delta,
            gamma=gamma,
            vega=vega,
            # A put gains when rates fall through the forward and loses through the discount
            # factor; the net is the familiar negative rho of a put.
            rho=-discount * strike * maturity * _normal_cdf(-d2) * 1e-4,
        )

    def trade_cost(self, units_traded: float, market: HedgeMarket) -> float:
        """Half a volatility point, charged against the option's own vega."""
        return abs(units_traded) * self.exposures(market).vega * OPTION_COST_VOL_POINTS

    def carry(self, units: float, market: HedgeMarket, years: float) -> float:
        return 0.0   # the premium is paid up front and shows up as time decay in the mark


def _normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


# ---------------------------------------------------------------- rate instruments


@dataclass(frozen=True)
class RateFuture:
    """A note future as a fixed-duration position. One unit is one dollar of notional."""

    duration: float = NOTE_FUTURE_DURATION
    label: str = "rate future"

    def notional(self, market: HedgeMarket) -> float:
        return 1.0

    def value(self, market: HedgeMarket, struck_at: float | None = None) -> float:
        reference = float(market.curve.zero(self.duration))
        return 0.0 if struck_at is None else -self.duration * (reference - struck_at)

    def exposures(self, market: HedgeMarket) -> Exposures:
        return Exposures(rho=-self.duration * 1e-4)

    def trade_cost(self, units_traded: float, market: HedgeMarket) -> float:
        return abs(units_traded) * RATE_FUTURES_COST

    def carry(self, units: float, market: HedgeMarket, years: float) -> float:
        return 0.0


@dataclass(frozen=True)
class BondForward:
    """The long-bond version of the same thing, and the only rate instrument long enough to
    reach the far end of a forty-year liability."""

    duration: float = LONG_BOND_DURATION
    label: str = "bond forward"

    def notional(self, market: HedgeMarket) -> float:
        return 1.0

    def value(self, market: HedgeMarket, struck_at: float | None = None) -> float:
        reference = float(market.curve.zero(self.duration))
        return 0.0 if struck_at is None else -self.duration * (reference - struck_at)

    def exposures(self, market: HedgeMarket) -> Exposures:
        return Exposures(rho=-self.duration * 1e-4)

    def trade_cost(self, units_traded: float, market: HedgeMarket) -> float:
        return abs(units_traded) * BOND_FORWARD_COST

    def carry(self, units: float, market: HedgeMarket, years: float) -> float:
        return 0.0


@dataclass(frozen=True)
class InterestRateSwap:
    """A par swap, valued off the curve plus a constant spread. One unit is a dollar of notional.

    Receive-fixed is the direction a guarantee writer wants: the liability grows when rates
    fall, and a receive-fixed swap grows with it. Its rate sensitivity is the annuity of the
    fixed leg, which is the honest version of a duration assumption - it comes from the curve
    rather than from a number chosen to look right.
    """

    tenor: float = 10.0
    receive_fixed: bool = True
    spread: float = 0.0            # swap over Treasury; zero by default and swept
    frequency: int = 2
    label: str = "interest rate swap"

    def notional(self, market: HedgeMarket) -> float:
        return 1.0

    def annuity(self, market: HedgeMarket) -> float:
        times = np.arange(1, int(round(self.tenor * self.frequency)) + 1) / self.frequency
        return float(np.sum(market.curve.discount(times)) / self.frequency)

    def par_rate(self, market: HedgeMarket) -> float:
        return par_equivalent(market.curve, self.tenor, self.frequency) + self.spread

    def value(self, market: HedgeMarket, struck_at: float | None = None) -> float:
        """Mark of a swap struck at ``struck_at``, per dollar of notional."""
        if struck_at is None:
            return 0.0
        drift = self.par_rate(market) - struck_at
        sign = 1.0 if self.receive_fixed else -1.0
        return -sign * drift * self.annuity(market)

    def exposures(self, market: HedgeMarket) -> Exposures:
        sign = 1.0 if self.receive_fixed else -1.0
        return Exposures(rho=-sign * self.annuity(market) * 1e-4)

    def trade_cost(self, units_traded: float, market: HedgeMarket) -> float:
        return abs(units_traded) * SWAP_COST

    def carry(self, units: float, market: HedgeMarket, years: float) -> float:
        return 0.0   # the fixed-floating accrual is in the mark


def exposure_matrix(instruments, market: HedgeMarket) -> np.ndarray:
    """Columns are instruments, rows are the four risk factors. The hedge solve's G matrix."""
    return np.column_stack([
        instrument.exposures(market).as_array() for instrument in instruments
    ])
