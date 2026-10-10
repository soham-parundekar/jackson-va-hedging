"""Nested simulation: the expensive answer the proxy has to be measured against.

The regression proxy is fitted on single-path realisations, so its residual against those
realisations is dominated by payoff noise and says almost nothing about whether the fitted
conditional mean is right. An out-of-sample R-squared of 0.07 in the first year is not a bad
fit; it is a reminder that forty years of uncertainty sit between a state and its realised
outcome. The only honest test is to compute the true value at a handful of states and compare.

That is what this does. Take states off the simulated paths, rebuild the market as it stands at
each one, and value the contract from there with a full inner simulation. A few hundred nodes at
twenty thousand inner paths is minutes of compute against days for the full nested calculation
the proxy exists to avoid, and it is enough to answer the question the proxy poses.

Rebuilding the market at a node is exact rather than approximate, which is the part worth
explaining. Hull-White gives the whole curve from the short rate at a point in time:
P(t, t + tau) has a closed form in r(t), so the zero curve seen from that node is available
without simulating anything. Fitting the smooth curve to those points gives a market state the
valuation engine accepts unchanged. The variance carries over directly as Heston's starting
variance, which is what it is.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from ..liability import cohorts as cohorts_module
from ..market.curves import ZeroCurve, fit_curve

CURVE_TENORS = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0])

# Where the convexity surface stops walking a node upward, and which nodes are worth walking.
# The anniversary states the nodes are taken from cannot have an account above the benefit base,
# because the ratchet resets the base to the account whenever it is higher, so a surface built
# only at those states ends at moneyness one. The hedge spends much of its time above that, since
# the account rises between anniversaries while the base sits still.
LADDER_FROM = 0.80
LADDER_TO = 1.45


def curve_at_node(hull_white, time: float, short_rate: float):
    """The zero curve an observer sees at a node, from the Hull-White bond formula."""
    maturities = time + CURVE_TENORS
    prices = hull_white.bond_price(time, maturities, short_rate)
    zeros = -np.log(prices) / CURVE_TENORS
    return fit_curve(ZeroCurve(tenors=CURVE_TENORS.copy(), zero_rates=np.asarray(zeros, float)))


def _advance(book, recorded: dict, path: int, year: int):
    """The same contract as it stands at the end of ``year`` on ``path``.

    Everything the recursion carries per path has to come from the recording, not from a guess
    made off the account value and the benefit base. Three of the state variables are genuinely
    path-dependent: the bonus clock restarts on a step-up, the GWB adjustment dies on the first
    withdrawal, and the death benefit base is cut by withdrawals on a schedule of its own and
    ends up nowhere near the guaranteed withdrawal base. An earlier version rebuilt the node
    with the death base set equal to the benefit base, which on an exhausted contract at year
    twenty turned a life annuity worth 0.34 of the base into one worth 0.96 and made the nested
    figures useless as a standard.

    The deferral countdown, the attained age and the projection horizon move with the calendar,
    which is arithmetic rather than state.
    """
    at = lambda name: np.array([float(recorded[name][path, year])])
    years = year + 1
    fields = dict(book.__dict__)
    fields.update(
        years_since_issue=book.years_since_issue + years,
        attained_age=book.attained_age + years,
        account_value=np.array([max(float(recorded["account_value"][path, year]), 0.0)]),
        benefit_base=at("benefit_base"),
        bonus_base=at("bonus_base"),
        death_benefit_base=at("death_rollup_base"),
        death_ratchet_base=at("death_ratchet_base"),
        deferral_years=np.maximum(book.deferral_years - years, 0),
        bonus_years_remaining=np.maximum(
            np.array([int(round(recorded["bonus_end"][path, year]))]) - years, 0
        ),
        adjustment_year=np.where(
            recorded["adjustment_live"][path, year] > 0.5,
            np.maximum(book.adjustment_year - years, 0), -1,
        ),
        projection_years=book.projection_years - years,
    )
    return cohorts_module.CohortBook(**fields)


def _bumped(book, factor: float):
    """The same contract with its value scaled, which is what a move in the funds does to it.

    The benefit base does not move, and that is the whole point of the guarantee.
    """
    fields = dict(book.__dict__)
    fields["account_value"] = book.account_value * factor
    return cohorts_module.CohortBook(**fields)


def gold_standard(
    valuer,
    book,
    state,
    recorded: dict,
    year: int,
    n_nodes: int = 200,
    attribution: float = 1.0,
    seed: int = 21,
    equity_bump: float = 0.0,
    gamma_bump: float = 0.10,
) -> pd.DataFrame:
    """Value the contract exactly at nodes taken off the simulated paths.

    Nodes are chosen by stratifying on moneyness rather than at random, because the states that
    matter are the ones where the guarantee is near the boundary of biting, and a random sample
    of a lognormal puts almost nothing there.

    A non-zero ``equity_bump`` also returns the equity exposure at each node, from a central
    bump of the contract value. It is nearly free: the path cache is keyed on the market state
    and the bump moves only the contract, so all three valuations share one simulation. Having
    it is what lets the proxy's delta be checked rather than assumed, and a proxy whose level
    is right and whose slope is wrong would hedge badly while valuing correctly.

    The derivative is with respect to the log contract value, not the log index, which is the
    same convention the proxy's own delta uses. The step from one to the other is the equity
    weight of the sub-account mix and the basis between the funds and the index, and that
    belongs in the hedge sizing where both are visible rather than buried in a Greek.
    """
    account = recorded["account_value"][:, year]
    benefit = recorded["benefit_base"][:, year]
    variance = recorded["variance"][:, year]
    short_rate = recorded["short_rate"][:, year] if "short_rate" in recorded else None
    if short_rate is None:
        raise ValueError("the recorded projection must carry the short rate to rebuild a curve")

    alive = account > 1e-8
    if alive.sum() < n_nodes:
        raise ValueError(f"only {alive.sum()} paths still have a contract value at year {year}")

    moneyness = np.log(benefit[alive] / account[alive])
    order = np.argsort(moneyness)
    picked = np.asarray(np.where(alive)[0])[order[
        np.linspace(0, order.size - 1, n_nodes).round().astype(int)
    ]]

    hull_white = state.hull_white()
    rows = []
    for path in picked:
        node_curve = curve_at_node(hull_white, float(year + 1), float(short_rate[path]))
        node_state = replace(
            state,
            curve=node_curve,
            heston=replace(state.heston, v0=float(max(variance[path], 1e-8))),
            valuation_year=state.valuation_year + year + 1,
        )
        node_book = _advance(book, recorded, int(path), year)
        attributed = np.array([attribution])
        valuation = valuer.value(node_book, node_state, attribution=attributed)
        row = {
            "path": int(path),
            "year": year,
            "account_value": float(account[path]),
            "benefit_base": float(benefit[path]),
            "log_moneyness": float(np.log(benefit[path] / account[path])),
            "variance": float(variance[path]),
            "zero_10y": float(node_curve.zero(10.0)),
            "short_rate": float(short_rate[path]),
            "nested_value": valuation.market_risk_benefit,
            "nested_std_error": valuation.std_error,
        }
        if equity_bump:
            # Two step sizes, because the two Greeks need different ones. The slope is a local
            # quantity and a two per cent move keeps it local while the shared draws keep the
            # difference out of the noise. The curvature is reported as an average over a move
            # of the size a convexity hedge is sized for, ten per cent, which is the only form
            # in which it is estimable at all - the proxy's point second derivative is not, and
            # comparing against a point second derivative here would be comparing two numbers
            # that neither of them can produce.
            bumps = {}
            for name, size in (("up", equity_bump), ("down", -equity_bump),
                               ("wide_up", gamma_bump), ("wide_down", -gamma_bump)):
                bumps[name] = valuer.value(
                    _bumped(node_book, np.exp(size)), node_state, attribution=attributed
                ).market_risk_benefit
            row["nested_delta"] = (
                (bumps["up"] - bumps["down"]) / (2.0 * equity_bump)
            )
            row["nested_gamma"] = (
                bumps["wide_up"] - 2.0 * valuation.market_risk_benefit + bumps["wide_down"]
            ) / gamma_bump ** 2
        rows.append(row)
    return pd.DataFrame(rows)


def compare(proxy, truth: pd.DataFrame, attribution: float = 1.0) -> pd.DataFrame:
    """Proxy against nested, node by node."""
    year = int(truth["year"].iloc[0])
    values, outside = proxy.value(
        year,
        truth["account_value"].to_numpy(),
        truth["benefit_base"].to_numpy(),
        truth["variance"].to_numpy(),
        truth["zero_10y"].to_numpy(),
        attribution=attribution,
    )
    out = truth.copy()
    out["proxy_value"] = values
    out["error"] = out["proxy_value"] - out["nested_value"]
    out["outside_design_range"] = outside
    if "nested_delta" in truth.columns:
        greeks = proxy.greeks(
            year,
            truth["account_value"].to_numpy(),
            truth["benefit_base"].to_numpy(),
            truth["variance"].to_numpy(),
            truth["zero_10y"].to_numpy(),
            attribution=attribution,
        )
        out["proxy_delta"] = greeks["delta"]
        out["proxy_gamma"] = greeks["gamma"]
        out["delta_error"] = out["proxy_delta"] - out["nested_delta"]
        out["gamma_error"] = out["proxy_gamma"] - out["nested_gamma"]
    return out


def summarise(comparison: pd.DataFrame, premium: float) -> dict:
    """The numbers that decide whether the proxy is fit to hedge against.

    Reported twice: over every node, and over the nodes the fit has data behind. Both belong in
    the answer. The first is what a backtest would suffer if it wandered anywhere; the second is
    what it will actually suffer, because a state the design distribution never reaches is
    flagged when the proxy is asked for it. Quoting only the first understates the proxy and
    quoting only the second hides the states where it should not be trusted.

    The two scaled columns are shares of premium, not percentages of anything, and are named
    for it. They used to be ``..._pct_of_account`` against a parameter called ``account_value``
    that every caller fed a premium, and the mislabel travelled: it reached three documents as
    an error a hundred times too small before the ratio against the nested delta exposed it.
    The denominator has to be premium rather than each node's own account value, because the
    account has moved a long way from premium by the later years and a per-node denominator
    would make the error look like it shrinks when only the scale did.
    """
    def block(frame: pd.DataFrame, suffix: str) -> dict:
        if frame.empty:
            return {f"nodes{suffix}": 0, f"r_squared{suffix}": np.nan, f"rmse{suffix}": np.nan,
                    f"rmse_share_of_premium{suffix}": np.nan,
                    f"worst_share_of_premium{suffix}": np.nan}
        error = frame["error"].to_numpy()
        truth = frame["nested_value"].to_numpy()
        total = truth - truth.mean()
        rmse = float(np.sqrt((error @ error) / error.size))
        return {
            f"nodes{suffix}": int(frame.shape[0]),
            f"r_squared{suffix}": float(1.0 - (error @ error) / (total @ total))
            if total @ total > 0 else np.nan,
            f"rmse{suffix}": rmse,
            f"rmse_share_of_premium{suffix}": rmse / premium,
            f"worst_share_of_premium{suffix}": float(np.abs(error).max() / premium),
        }

    out = block(comparison, "")
    out.update(block(comparison[~comparison["outside_design_range"]], "_in_range"))
    out["mean_nested_std_error"] = float(comparison["nested_std_error"].mean())
    out["share_outside_design_range"] = float(comparison["outside_design_range"].mean())
    return out


def gamma_surface(
    valuer,
    book,
    state,
    recorded: dict,
    years,
    n_nodes: int = 25,
    attribution: float = 1.0,
    equity_bump: float = 0.10,
) -> pd.DataFrame:
    """Curvature per unit of benefit base against moneyness, one curve per policy year.

    Built by valuing each node three times - at its own contract value and at that value moved
    up and down - and taking the second difference. The three share one simulation, because
    under Heston the return distribution does not depend on the index level, so a node costs
    barely more than a single valuation. Two hundred nodes is a few minutes.

    The step is ten per cent rather than a basis point, and that is a definition rather than a
    numerical convenience. A point second derivative of a Monte Carlo valuation is noise divided
    by the square of a small number; what a convexity hedge is sized for is the average
    curvature across a move of the size that actually happens, and ten per cent is that size.
    The same step is used when the proxy's own curvature is measured against this, so the two
    are comparable.

    Nodes are stratified on moneyness rather than drawn at random, because the curve is wanted
    across the whole range the hedge will visit and a lognormal sample puts almost nothing at
    the ends.

    One part of the range no node can reach on its own. These states are read off anniversaries,
    and an anniversary is where the ratchet has just lifted the benefit base to the account value
    if the account was higher, so the moneyness of a live contract there is at most one. Between
    anniversaries it is routinely more: the account grows and the base does not. A node near the
    base is therefore valued along a ladder of equity shifts rather than at three points, each
    rung spaced by the same bump, so consecutive rungs share valuations and the ladder costs one
    call per rung instead of three. Interior rungs give the second difference, and the top of the
    ladder reaches the moneyness the hedge actually sees in a rising market.

    What the ladder does not fix is which paths supplied those states. The rungs above one all
    come from paths that happened to finish the year close to their base, so the variance and the
    short rate attached to a point at moneyness 1.3 are those of a path at 0.9 rather than of a
    path that genuinely ran that far above its guarantee. The surface is indexed on moneyness
    alone, so this only matters through that correlation, and the size of it is what the spread
    reported by the build script measures.
    """
    rows = []
    hull_white = state.hull_white()
    for year in years:
        account = recorded["account_value"][:, year]
        benefit = recorded["benefit_base"][:, year]
        variance = recorded["variance"][:, year]
        short_rate = recorded["short_rate"][:, year]
        alive = (account > 1e-8) & (benefit > 1e-9)
        if alive.sum() < n_nodes:
            continue
        moneyness = account[alive] / benefit[alive]
        order = np.argsort(moneyness)
        picked = np.asarray(np.where(alive)[0])[order[
            np.linspace(0, order.size - 1, n_nodes).round().astype(int)
        ]]

        for path in picked:
            node_curve = curve_at_node(hull_white, float(year + 1), float(short_rate[path]))
            node_state = replace(
                state, curve=node_curve,
                heston=replace(state.heston, v0=float(max(variance[path], 1e-8))),
                valuation_year=state.valuation_year + year + 1,
            )
            node_book = _advance(book, recorded, int(path), year)
            attributed = np.array([attribution])
            base = float(benefit[path])
            spot = float(account[path]) / base

            top_rung = 0
            if spot > LADDER_FROM:
                top_rung = max(int(np.ceil(np.log(LADDER_TO / spot) / equity_bump)), 0)
            priced = {}
            for rung in range(-1, top_rung + 2):
                shifted = (node_book if rung == 0
                           else _bumped(node_book, float(np.exp(rung * equity_bump))))
                result = valuer.value(shifted, node_state, attribution=attributed)
                priced[rung] = (float(np.ravel(result.market_risk_benefit)[0]),
                                float(np.ravel(result.std_error)[0]))

            for rung in range(0, top_rung + 1):
                second = priced[rung + 1][0] - 2.0 * priced[rung][0] + priced[rung - 1][0]
                rows.append({
                    # The fit keyed ``year`` applies at ``year + 1`` years since issue, and the
                    # surface is indexed the way the simulator asks for it.
                    "policy_year": float(year + 1),
                    "moneyness": spot * float(np.exp(rung * equity_bump)),
                    "gamma": second / equity_bump ** 2,
                    "gamma_per_unit": second / equity_bump ** 2 / base,
                    "value": priced[rung][0],
                    "benefit_base": base,
                    "variance": float(variance[path]),
                    "zero_10y": float(node_curve.zero(10.0)),
                    "std_error": priced[rung][1],
                    # Zero marks a state a path actually produced; above that the account has
                    # been walked up from one, which is the only way past the ratchet's cap.
                    "rung": rung,
                })
    return pd.DataFrame(rows)
