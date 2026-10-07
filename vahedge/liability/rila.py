"""The registered index-linked annuity, which is the guarantee book's natural offset.

A RILA credits the index return subject to a cap on the upside and absorbs the first slice of
any loss through a buffer. Written out, the credited return over a term is

    credited(R) = min(max(R, 0), c) + min(R + b, 0)

for a buffer b and a cap c. That is a call spread struck at zero and c, less a put struck at
-b, all on the index return:

    credited = call(K = 1) - call(K = 1 + c) - put(K = 1 - b)

which is worth stating because it makes the hedge obvious and gives an exact closed form to
test the simulation against.

Why it belongs in a project about guarantees. The insurer owes the upside on a RILA and owns
it on a withdrawal guarantee. Jackson says so directly: an investor presentation describes the
variable annuity guarantees and the RILA as carrying offsetting equity risk, which reduces how
much has to be hedged externally. The FY2025 Item 7A table shows the same thing from the other
side - the fixed index and RILA embedded derivatives gain $1,321m on a 10% rally while the
market risk benefit loses $1,574m - so at the end of 2025 the two very nearly cancelled. They
did not in 2024, when the RILA book was half the size and the embedded derivative moved $4m
against the market risk benefit's $1,722m. The netting is a function of how fast the RILA book
has grown, and that is what experiment E5 measures.

Only the current term is modelled. The embedded derivative is the obligation Jackson has
already written, and a cap for a term that has not started yet is not contracted, so projecting
renewals would be projecting a product the company has not sold. That also sidesteps having to
guess how caps will be reset, which is the largest unobservable in a RILA.

The cap is solved for rather than assumed. Jackson publishes buffers - the 10-K gives 20% as
its example - and does not publish caps, which move with the term, the index and the rate
environment. A cap that leaves the contract worth exactly its premium at issue is both the
economically right answer and what an insurer actually does when setting one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from ..market.heston_cos import HestonParameters, black76, cos_price


@dataclass(frozen=True)
class RilaTerms:
    """One index-linked segment: how much loss the insurer absorbs, and where the upside stops."""

    buffer: float
    cap: float
    term_years: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.buffer < 1.0:
            raise ValueError("buffer must be in [0, 1)")
        if self.cap <= 0:
            raise ValueError("cap must be positive")
        if self.term_years <= 0:
            raise ValueError("term_years must be positive")


def credited_return(index_return, buffer: float, cap: float):
    """The return credited to the contract holder's account over the term.

    Boundary behaviour, which the tests pin down: with a 10% buffer and a 12% cap, index
    returns of -30%, -10%, -5%, 0, 5% and 20% credit -20%, 0, 0, 0, 5% and 12%.
    """
    r = np.asarray(index_return, dtype=float)
    return np.minimum(np.maximum(r, 0.0), cap) + np.minimum(r + buffer, 0.0)


def value_black_scholes(forward: float, discount: float, vol: float, terms: RilaTerms) -> float:
    """Present value of the credited return, per unit of account, under flat volatility.

    This exists to test the simulation and the Heston pricer against, not to price the book.
    ``forward`` and ``discount`` are for the term, with the index normalised to one.
    """
    call_low = float(black76(forward, 1.0, terms.term_years, vol, discount, True))
    call_high = float(black76(forward, 1.0 + terms.cap, terms.term_years, vol, discount, True))
    put = float(black76(forward, 1.0 - terms.buffer, terms.term_years, vol, discount, False))
    return call_low - call_high - put


def value_heston(
    params: HestonParameters,
    forward: float,
    discount: float,
    terms: RilaTerms,
    n_terms: int = 512,
) -> float:
    """The same decomposition priced off the calibrated surface.

    The three strikes sit at meaningfully different moneyness - at the forward, above it, and
    well below - so the skew matters here in a way it does not for an at-the-money option. That
    is the whole reason the RILA is priced on the calibrated surface rather than at a single
    volatility.
    """
    strikes = np.array([1.0, 1.0 + terms.cap, 1.0 - terms.buffer])
    is_call = np.array([True, True, False])
    prices = cos_price(params, forward, strikes, terms.term_years, discount, is_call,
                       n_terms=n_terms)
    return float(prices[0] - prices[1] - prices[2])


def breakeven_cap(
    funding_discount: float,
    terms: RilaTerms,
    pricer,
    margin: float = 0.0,
    bounds: tuple = (0.01, 3.0),
) -> float:
    """The cap that leaves the contract worth its premium at issue.

    At issue the insurer takes the premium and owes the account grown by the credited return at
    the end of the term. Writing A for the account, the obligation is worth
    A * (D_funding + PV(credited)), and setting that equal to A gives

        PV(credited) = 1 - D_funding - margin

    Two different rates appear in that sentence and keeping them apart is the whole content of
    this function.

    The left-hand side is what the hedge costs, so it is priced off the risk-free curve and the
    option-implied surface. The right-hand side is what the insurer earns on the premium, and
    the insurer holds its general account, not Treasuries. Discounting the funding leg at the
    risk-free rate says a RILA cannot be written at all, and the arithmetic is worth following
    because it is not obvious.

    Under the risk-neutral measure the index price return grows at r - q, so passing it through
    uncapped and unbuffered is worth exp(-qT) - D, while the Treasury carry is 1 - D. The
    difference is exp(-qT) - 1, which is negative for any positive dividend yield. So an insurer
    holding Treasuries and crediting a price return already keeps the dividend yield, and that
    is its entire margin. On the 28 September 2026 curve with an implied dividend yield near
    0.8%, that margin is about 4.7% of premium over six years, against a 10% buffer that costs
    6.9%. Treasuries alone leave the contract 1.5 points short and no finite cap fixes it.

    What closes the gap is the spread the general account earns over Treasuries. That is not a
    modelling convenience: it is the economics of the product. The buffer is paid for out of
    investment spread, which is why RILA terms move with credit spreads as well as with rates
    and volatility, and why a Treasury-funded version of this calculation says the product is
    impossible.

    ``pricer`` takes a cap and returns PV(credited), so the same solver runs against the
    Black-Scholes closed form and against the calibrated surface.
    """
    target = 1.0 - funding_discount - margin
    if target <= 0:
        raise ValueError(
            f"a funding discount factor of {funding_discount:.4f} and a margin of {margin:.4f} "
            "leave nothing to fund the credited return; no positive cap exists"
        )

    def objective(cap: float) -> float:
        return pricer(cap) - target

    low, high = bounds
    if objective(low) > 0:
        raise ValueError(
            f"a cap of {low:.0%} already costs more than the {target:.4f} of funding available; "
            "the buffer is too generous for this rate, spread and volatility environment"
        )
    if objective(high) < 0:
        raise ValueError(
            f"even a cap of {high:.0%} is worth less than the {target:.4f} of funding available, "
            "so the upside is effectively unlimited and the cap does not bind; the contract is "
            "profitable uncapped at these assumptions"
        )
    return float(brentq(objective, low, high, xtol=1e-10, maxiter=200))


def project(
    paths,
    terms: RilaTerms,
    account_value: float = 1.0,
    steps_per_year: int = 1,
    realised_growth: float = 1.0,
    elapsed_years: int = 0,
    index_shock: float = 0.0,
) -> dict:
    """Value the current term by simulation, on the same paths the guarantee book uses.

    Sharing the paths is the point. The netting between the two books is a statement about
    their joint distribution, and valuing them on independent draws would replace that with
    the product of two marginals.

    ``paths.index_growth`` is annual, so a term is the product of its years and the term has to
    be a whole number of them. A six-year term on annual factors is exact for a point-to-point
    credit, which is what the contract does.

    ``realised_growth`` and ``elapsed_years`` value a segment partway through its term. A
    point-to-point credit looks at the index on one day and on one other, so what has already
    happened enters only as a multiplier: the term return is the growth to date times the growth
    still to come, and the only state a live segment carries is that one number. The paths then
    supply the remaining years.

    ``index_shock`` scales the index now, which scales the whole term return by the same factor,
    because the realised part moves with the index and the future part is a ratio to it. That
    makes a delta three valuations of one simulation rather than three simulations, and it is
    what E5 needs to put the RILA's own equity exposure next to the guarantee's.
    """
    term_years = int(round(terms.term_years))
    if abs(terms.term_years - term_years) > 1e-9:
        raise ValueError("a point-to-point term has to be a whole number of annual factors")
    if not 0 <= elapsed_years <= term_years:
        raise ValueError(f"elapsed_years {elapsed_years} is outside a {term_years}-year term")
    if realised_growth <= 0:
        raise ValueError("realised_growth is an index ratio and has to be positive")
    remaining = term_years - elapsed_years
    if paths.index_growth.shape[1] < remaining:
        raise ValueError(
            f"paths carry {paths.index_growth.shape[1]} years, the term needs {remaining}"
        )

    future = (np.prod(paths.index_growth[:, :remaining], axis=1) if remaining
              else np.ones(paths.index_growth.shape[0]))
    growth = realised_growth * (1.0 + index_shock) * future
    index_return = growth - 1.0
    credited = credited_return(index_return, terms.buffer, terms.cap)
    # Discounted over the remaining term, not the original one. A segment three years into six
    # owes its credit in three years' time, and discounting it over six would value an
    # obligation nobody has.
    discount = (paths.discount[:, remaining - 1] if remaining
                else np.ones(paths.index_growth.shape[0]))

    payoff = discount * credited
    half = payoff.size // 2
    if paths.antithetic and payoff.size % 2 == 0:
        pairs = 0.5 * (payoff[:half] + payoff[half:])
        value, std_error = float(pairs.mean()), float(pairs.std(ddof=1) / np.sqrt(half))
    else:
        value = float(payoff.mean())
        std_error = float(payoff.std(ddof=1) / np.sqrt(payoff.size))

    return {
        "embedded_derivative": account_value * value,
        "std_error": account_value * std_error,
        "mean_discount": float(discount.mean()),
        "mean_credited": float(credited.mean()),
        "share_capped": float((index_return >= terms.cap).mean()),
        "share_through_buffer": float((index_return <= -terms.buffer).mean()),
        "share_protected": float(((index_return < 0) & (index_return > -terms.buffer)).mean()),
    }


def equity_exposure(
    paths,
    terms: RilaTerms,
    account_value: float = 1.0,
    realised_growth: float = 1.0,
    elapsed_years: int = 0,
    bump: float = 0.01,
) -> dict:
    """dV/d(ln S) of the embedded derivative, on common random numbers.

    The sign convention is the guarantee book's, so the two can be added: this is what the
    insurer owes, so a positive number means the obligation grows when the index does. That is
    the opposite of a withdrawal guarantee, which gets cheaper in a rally, and it is the whole
    mechanism behind the netting.

    One simulation, three shocks. Under Heston the return distribution does not depend on the
    index level, so shocking the index is a multiplier on the term return rather than a reason
    to redraw - the same economy that lets the convexity surface walk a node up a ladder.

    **The cap kills the exposure and the buffer does not**, which is the opposite of the
    intuition the word "buffer" invites and is worth stating because the first version of this
    had it backwards. Differentiating the credit in the term return gives a slope of one between
    zero and the cap, zero above the cap, zero inside the buffer, and one again below it: the
    buffer absorbs the *first* slice of a loss and the contract holder bears everything past it,
    so an insurer whose segments are through their buffers is still fully exposed. Only a capped
    segment has stopped moving.

    The number is per unit log move in the index, so it also carries the index ratio itself -
    the chain rule puts a factor of one plus the term return on the slope, and a segment up 20%
    with a year to run reaches about 1.08 per unit of account value rather than stopping at one.
    The shape across a term is what E5 is built on: roughly 0.2 to 0.5 at issue, where six years
    of drift already put a third of the mass above the cap, rising to near one for a segment late
    in its term sitting just under the cap, and collapsing to around 0.1 once it is through.
    """
    def value(shock: float) -> float:
        return project(paths, terms, account_value=account_value,
                       realised_growth=realised_growth, elapsed_years=elapsed_years,
                       index_shock=shock)["embedded_derivative"]

    up, down = value(bump), value(-bump)
    base = project(paths, terms, account_value=account_value,
                   realised_growth=realised_growth, elapsed_years=elapsed_years)
    span = np.log((1.0 + bump) / (1.0 - bump))
    # The ten per cent repricings as well as the derivative, because Item 7A runs a 10% shock and
    # a credit with a cap and a buffer in it is not linear over a move that size. Comparing a
    # per-log-move delta with a per-10%-move disclosure is out by one over ln(1.1), a factor of
    # 10.5, which is large enough to be mistaken for a finding.
    shocked_up = value(0.10)
    shocked_down = value(-0.10)
    return {
        "embedded_derivative": base["embedded_derivative"],
        "equity_exposure": (up - down) / span,
        "equity_exposure_pct_of_account": (up - down) / span / account_value,
        "equity_up_10pct": shocked_up - base["embedded_derivative"],
        "equity_down_10pct": shocked_down - base["embedded_derivative"],
        "equity_up_10pct_pct_of_account": (shocked_up - base["embedded_derivative"]) / account_value,
        "equity_down_10pct_pct_of_account": (shocked_down - base["embedded_derivative"])
                                            / account_value,
        "share_capped": base["share_capped"],
        "share_through_buffer": base["share_through_buffer"],
        "std_error": base["std_error"],
    }
