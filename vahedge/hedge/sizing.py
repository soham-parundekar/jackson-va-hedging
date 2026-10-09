"""How many of each instrument, and the sign conventions that decide whether it hedges at all.

The sizing problem is small and over-determined in the wrong direction: four risk factors,
between one and six instruments, and three of the instruments carry only one of the factors. So
it is a weighted least-squares solve rather than an inversion, and the weights are what decides
which residual the hedge is willing to live with.

    minimise  || W (g + G n) ||^2 + mu sum_j ( n_j || W G_j || )^2

with g the exposure the insurer carries, G the instruments' exposures by column, and n the
number of units. The ridge term is there for the degenerate cases - two rate instruments with
nearly proportional exposures, or a strategy with no gamma instrument at all - where the
unregularised solve is free to take enormous offsetting positions that net to the same hedge
and cost a fortune in spread.

What the penalty is measured in matters more than its size. Each position is penalised by the
weighted exposure it produces, || W G_j || per unit, not by its unit count and not by its
notional. Notional would be the obvious choice and it is wrong here: a dollar of notional in the
receive-fixed swap carries eight basis points of rho against an equity future's hundred dollars
of delta, so penalising per dollar of notional is three orders of magnitude apart from
penalising per dollar of risk, and it crushes the rate leg. See ``_weighted_ridge``, which is
where a version of exactly that mistake was found.

The sign convention is the part worth reading twice, because getting it wrong produces a result
that looks like a hedge and doubles the risk.

Jackson reports the market risk benefit as a liability when positive, and the Greeks module
reports its sensitivities on that basis: the equity exposure of the market risk benefit is
negative, because a rally shrinks the guarantee. The insurer, however, is short that liability.
Its own economic exposure is the negative of the market risk benefit's, so the insurer is long
equity through the guarantee - the value of its position rises when markets rise. A hedge has to
offset the insurer's exposure, which means the hedge portfolio has to be short equity, and the
exposure it has to produce is therefore equal to the market risk benefit's own Greek rather than
opposite to it.

In symbols: g = -(market risk benefit Greeks), and the solve drives g + G n to zero, so
G n = +(market risk benefit Greeks), which is negative in delta. Short index. That is the right
answer for a written put, and the test that asserts it is the most valuable one in the file.

Weights. Delta and rho dominate, because they are the exposures the instruments can actually
reach and the ones that move the book week to week. Gamma and vega are weighted down rather
than out: with only one convexity instrument available, weighting gamma equally with delta lets
a put position be sized to match curvature and in doing so unbalance the delta it also carries.
The weights are an assumption and E3 sweeps them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .instruments import Exposures, exposure_matrix

# Relative weights on [delta, gamma, vega, rho] in the sizing objective. Delta and rho are in
# dollars per log move and per basis point; gamma and vega are second-order and are not what a
# weekly hedge is judged on, so they come in at a tenth.
DEFAULT_WEIGHTS = np.array([1.0, 0.1, 0.1, 1.0])
DEFAULT_RIDGE = 1e-6


@dataclass(frozen=True)
class HedgePositions:
    """Units of each instrument, and what the hedge leaves behind."""

    instruments: tuple
    units: np.ndarray
    target: Exposures          # what the hedge had to produce
    achieved: Exposures        # what it does produce
    # The exposure still open: the insurer's own plus whatever the hedge produces. Not the
    # solve's own error, which is ``achieved`` less ``target`` and is a different number as soon
    # as the hedge ratio is below one - at a ninety per cent hedge of this book the solve lands
    # within four thousandths of a dollar of its target while fourteen hundred and fifty dollars
    # of delta is deliberately left open. The two coincide exactly at a full hedge, which is
    # every strategy that reads this field.
    residual: Exposures

    @property
    def labels(self) -> tuple:
        return tuple(instrument.label for instrument in self.instruments)

    def notional(self, market) -> float:
        return float(np.sum(np.abs(self.units) * np.array(
            [instrument.notional(market) for instrument in self.instruments]
        )))

    def value(self, market, struck) -> float:
        """Mark of the whole hedge book, given where each position was struck."""
        return float(np.sum([
            units * instrument.value(market, strike)
            for instrument, units, strike in zip(self.instruments, self.units, struck)
        ]))


def insurer_exposures(greeks, equity_weight: float = 1.0, vega: str = "current") -> Exposures:
    """The insurer's own exposure through the guarantee, which is what has to be offset.

    ``greeks`` is a ``valuation.greeks.Greeks`` or the dictionary the proxy returns. Its fields
    are the market risk benefit's sensitivities, and the insurer is short that, so every sign
    flips here.

    The two accepted inputs differentiate in different variables, which is the trap this
    function exists to close. The hedge instruments move with the *index*, so that is the
    variable the exposure vector has to be in. The proxy's delta is per unit log move in the
    *contract value*, because its state variable is the account value. ``greeks.compute`` bumps
    the index and divides by the index log span, so its delta is already per unit log move in
    the index. Only the first needs converting, and for a continuously rebalanced sub-account

        d ln(contract) / d ln(index) = equity weight,

    so the index delta is the contract delta times that weight and the index gamma the contract
    gamma times its square - the second derivative of the log contract value in the log index
    being zero for fixed weights. Skipping the conversion on the proxy's delta oversizes the
    hedge by one over the weight, about twenty per cent on Jackson's disclosed fund split, and
    the error is a pure short index position that loses in every rising market. Applying it to
    ``greeks.compute``'s delta is the same mistake pointing the other way and undersizes by the
    same factor, which is why ``equity_weight`` is ignored on that branch rather than trusted.

    What the conversion does not capture is the part of the sub-account no index reaches: the
    funds are managed and their returns are not the index's. That is the sub-account basis, it
    is a scenario parameter rather than a calibrated one, and it is swept rather than hedged.

    The vega choice is not cosmetic either. The current-variance vega is what listed options
    reach; the long-run vega is an assumption about a level no option expires at. Sizing a put
    position against the long-run number would be hedging a model parameter with a traded
    instrument, so the default is the tradeable one and the remainder is reported unhedged.
    """
    if vega not in ("current", "long_run", "none"):
        raise ValueError(f"vega must be current, long_run or none; got {vega}")

    if isinstance(greeks, dict):
        delta = float(np.ravel(greeks["delta"])[0])
        gamma = float(np.ravel(greeks["gamma"])[0])
        rho = float(np.ravel(greeks["rho_per_bp"])[0])
        vega_value = 0.0 if vega == "none" else float(np.ravel(greeks["vega"])[0])
        to_index = float(equity_weight)
    else:
        delta, gamma, rho = greeks.equity_exposure, greeks.equity_gamma, greeks.rho_per_bp
        vega_value = {
            "current": greeks.vega_current,
            "long_run": greeks.vega_long_run,
            "none": 0.0,
        }[vega]
        to_index = 1.0

    return Exposures(
        delta=-delta * to_index,
        gamma=-gamma * to_index ** 2,
        vega=-vega_value,
        rho=-rho,
    )


def solve(
    instruments,
    market,
    exposure: Exposures,
    weights=DEFAULT_WEIGHTS,
    ridge: float = DEFAULT_RIDGE,
    hedge_ratio: float = 1.0,
    limits=None,
) -> HedgePositions:
    """Units of each instrument that best offset ``exposure``.

    ``exposure`` is the insurer's exposure, from ``insurer_exposures``. ``hedge_ratio`` below one
    is the partial hedge of strategy S5: it scales the target rather than the solved units, so a
    ninety per cent hedge leaves ten per cent of every exposure rather than ninety per cent of
    whichever ones the instruments happened to cover.

    ``limits`` is an optional array of maximum absolute units per instrument. It is applied by
    clipping and re-solving the remaining instruments against what is left, which is a projection
    rather than a constrained optimum; with six instruments and four factors the difference is
    immaterial, and a proper quadratic program would be machinery without a question behind it.
    """
    instruments = tuple(instruments)
    if not instruments:
        # The unhedged baseline, S0. Nothing is produced, so the whole exposure is left.
        return HedgePositions(
            instruments=(), units=np.zeros(0),
            target=(exposure * hedge_ratio) * -1.0,
            achieved=Exposures(), residual=exposure,
        )

    design = exposure_matrix(instruments, market)
    target = -(exposure * hedge_ratio).as_array()
    weights = np.asarray(weights, dtype=float)

    units = _weighted_ridge(design, target, weights, ridge)

    if limits is not None:
        units = np.clip(units, -np.abs(limits), np.abs(limits))
        at_limit = np.abs(units) >= np.abs(limits) - 1e-12
        if at_limit.any() and not at_limit.all():
            remaining = target - design[:, at_limit] @ units[at_limit]
            free = ~at_limit
            units[free] = _weighted_ridge(design[:, free], remaining, weights, ridge)
            units = np.clip(units, -np.abs(limits), np.abs(limits))

    achieved = Exposures.from_array(design @ units)
    return HedgePositions(
        instruments=instruments,
        units=units,
        target=Exposures.from_array(target),
        achieved=achieved,
        residual=Exposures.from_array(exposure.as_array() + achieved.as_array()),
    )


def _weighted_ridge(design, target, weights, ridge) -> np.ndarray:
    """Weighted least squares with a ridge, on columns normalised first.

    Normalising matters more than it looks. The instruments are quoted in wildly different
    units: one equity future carries a hundred dollars of delta, one dollar of notional in a
    note future carries six and a half ten-thousandths of a dollar per basis point. A ridge
    applied to the raw columns is scaled by the largest of them, which in the first version of
    this was the equity future - and it crushed the rate positions to a thousandth of their
    right size while leaving the delta hedge untouched. The result was a hedge that looked
    sensible, reported a closed delta, and removed one per cent of the rate exposure.
    """
    weighted = design * weights[:, None]
    norms = np.linalg.norm(weighted, axis=0)
    norms = np.where(norms > 1e-12, norms, 1.0)
    scaled = weighted / norms
    gram = scaled.T @ scaled
    penalty = ridge * np.eye(gram.shape[0])
    solved = np.linalg.solve(gram + penalty, scaled.T @ (target * weights))
    return solved / norms


def effectiveness(exposure: Exposures, positions: HedgePositions,
                  weights=DEFAULT_WEIGHTS) -> dict:
    """How much of each exposure the hedge removed, factor by factor.

    Reported per factor rather than as one number, because the single number a weighted norm
    gives is dominated by whichever factor is largest in dollars and says nothing about the one
    the hedge gave up on. A strategy with no put position removes none of the gamma and all of
    the delta, and that is the trade the experiments are about.
    """
    before = exposure.as_array()
    after = before + positions.achieved.as_array()
    out = {}
    for name, start, end in zip(Exposures.ORDER, before, after):
        out[f"{name}_before"] = float(start)
        out[f"{name}_after"] = float(end)
        out[f"{name}_removed"] = float(1.0 - abs(end) / abs(start)) if abs(start) > 1e-12 else np.nan
    weights = np.asarray(weights, dtype=float)
    out["weighted_norm_before"] = float(np.linalg.norm(weights * before))
    out["weighted_norm_after"] = float(np.linalg.norm(weights * after))
    return out
