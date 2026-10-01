"""Why the hedged P&L came out where it did, decomposed so the pieces add up exactly.

A hedged result is a small number left over from two large ones, so "the hedge worked" is not a
finding until you can say which risk it removed and which it did not. The decomposition is the
Taylor expansion of the liability, period by period, against the hedge's own mark:

    dL  ~  L_a dlna + 1/2 L_aa (dlna)^2 + L_sigma dsigma + L_r dr + theta dt + ...

with ``a`` the contract value - which is the variable the liability's delta and gamma are in -
and the hedge's mark expanded the same way in the index. Every term is computed from the Greeks
the hedge was actually sized with, and the residual is defined as the total less the sum of the
attributed terms, so the identity holds by construction rather than approximately. A residual
that reconciles to zero proves nothing about the model; a residual that is large relative to the
attributed bars is the finding.

Four things the bars are there to separate, and they behave differently.

*Delta* should be near zero on any strategy that hedges it, and what is left is the drift
between rebalances.

*Gamma* cannot be near zero on a strategy without options, and its sign is not a matter of
luck. The insurer is short convexity through the guarantee, so realised variance costs money
whether markets rise or fall: the term is one half gamma times the squared move, gamma is
negative, and the square is not. This is the bar that pays for a put position, and comparing it
against the premium is the whole of the S2-against-S3 question.

*Basis* is the part of the sub-account's move that the index did not explain. With the funds
modelled as a weighted portfolio of index, bond and cash sleeves this term is mechanical and
small; with a tracking error switched on it is the cost of hedging managed funds with an index,
and E3 sweeps it.

*Rate* is the one the equity-only strategies leave entirely open, and on a forty-year guarantee
it is not a rounding term - in 2022 it was larger than the equity bar.

*The anniversary* is the contract acting rather than the market, and on a withdrawal guarantee
in its income phase it is the largest bar after delta. No hedge reaches it.

Realised against implied volatility is reported alongside, because the gamma bar is the price of
that difference. The expansion's gamma term accumulates to roughly one half gamma times realised
variance over the period, while the premium paid for the puts is the implied variance; whether
the convexity hedge was worth buying is that comparison, and quoting the gamma bar without it
hides the trade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252.0


def attribute(ledger: pd.DataFrame, equity_weight: float) -> pd.DataFrame:
    """Period-by-period decomposition of the hedged P&L, reconciling to the total exactly.

    ``ledger`` is a ``HedgeRun``'s own, so the Greeks in it are the ones the hedge was sized
    with rather than a fresh set, which is what makes the attribution a statement about this
    hedge rather than about the model.

    The Greeks are taken at the start of each period, which is the only choice that corresponds
    to a decision anyone could have made. Using the end-of-period Greeks would attribute the
    period's own move to a sensitivity that only existed because of it, and the delta bar would
    come out flatteringly small on every strategy.
    """
    out = pd.DataFrame(index=ledger.index)
    account = ledger["account_value"].to_numpy()
    index = ledger["index"].to_numpy()
    # The market move within a day ends at the state just before the anniversary's events; the
    # drop from there to the post-event state is the contract acting, not the market. Measuring
    # the move to the post-event state instead - which is what the first version did - asks a
    # Taylor expansion to explain a discontinuity, and it cannot: the three largest unexplained
    # days over 2019 to 2021 were the three policy anniversaries, not anything in the crash.
    pre_account = (ledger["pre_account_value"].to_numpy()
                   if "pre_account_value" in ledger else account)
    moved_to = np.concatenate([[account[0]], pre_account[1:]])

    d_log_account = np.concatenate([[0.0], np.log(
        np.maximum(moved_to[1:], 1e-12) / np.maximum(account[:-1], 1e-12)
    )])
    d_log_index = np.concatenate([[0.0], np.diff(np.log(np.maximum(index, 1e-12)))])
    # Two volatility changes, because the two sides are sensitive to different ones. The
    # liability's vega is with respect to the model's instantaneous volatility, which is what
    # the state carries; a put's is with respect to the implied volatility at its own tenor,
    # which is far smoother. Using the instantaneous change for both - the first version of
    # this - attributed twenty per cent of account value to the hedge's vega over three years
    # and buried the mistake in the residual, because the front of the volatility surface moves
    # several times as much as the one-year point the puts are marked at.
    d_vol = np.concatenate([[0.0], np.diff(np.sqrt(ledger["variance"].to_numpy()))])
    d_implied = np.concatenate([[0.0], np.diff(ledger["implied_vol"].to_numpy())])
    d_rate_bp = np.concatenate([[0.0], np.diff(ledger["zero_10y"].to_numpy())]) * 1e4
    d_time = np.concatenate([[0.0], np.diff(
        (ledger.index - ledger.index[0]).days.to_numpy(dtype=float) / 365.0
    )])

    def lagged(name: str) -> np.ndarray:
        values = ledger[name].to_numpy()
        return np.concatenate([[values[0]], values[:-1]])

    # The insurer is short the liability, so its exposure is the negative of every Greek.
    delta, gamma = -lagged("delta"), -lagged("gamma")
    vega, rho = -lagged("vega"), -lagged("rho_per_bp")
    hedge_delta = lagged("hedge_delta")

    hedge_gamma = lagged("hedge_gamma")
    hedge_vega = lagged("hedge_vega")
    hedge_rho = lagged("hedge_rho_per_bp")

    # Splitting the equity terms needs one substitution, written out because it is what makes
    # the delta bar and the basis bar mean different things. The contract value moves with the
    # index through the equity sleeve and with other things through the rest:
    #
    #     dln(contract) = w dln(index) + (what the index did not drive),
    #
    # so the liability's delta term splits into an index part and a remainder. The index part is
    # what the hedge was sized against and cancels against the hedge's own delta to whatever
    # accuracy the sizing achieved; the remainder is the exposure no index instrument reaches.
    index_part = equity_weight * d_log_index
    not_index = d_log_account - index_part

    out["delta"] = (delta * equity_weight + hedge_delta) * d_log_index
    out["basis"] = delta * not_index
    out["gamma"] = 0.5 * gamma * d_log_account ** 2 + 0.5 * hedge_gamma * d_log_index ** 2
    out["vega"] = vega * d_vol + hedge_vega * d_implied
    out["rate"] = (rho + hedge_rho) * d_rate_bp
    # Everything that moved in cash rather than in a mark: the fee the contract collected, any
    # claim it paid, and interest on the balance.
    out["carry"] = ledger["cash_flow"].to_numpy() + ledger["interest"].to_numpy()
    # The contract's own annual events, which happen together and are left together. The
    # charges come out of the account, the guaranteed withdrawal is taken, the benefit base
    # steps up, and the year's cash flow leaves the projection. Splitting that into a fee part
    # and a state part was tried and abandoned: the valuation function and the state change at
    # the same instant, so any split needs a hinge state chosen arbitrarily, and the two
    # candidate hinges gave answers that differed by more than the bar.
    #
    # What dominates it, once income has started, is the withdrawal. Five and three-quarter per
    # cent of the benefit base leaves the account every year and goes to the policyholder - it
    # is their own money, so the insurer pays nothing - and the guarantee standing behind the
    # smaller account is worth more for it. That is the central economics of a withdrawal
    # guarantee and it shows up here rather than in any market bar: over three years on this
    # contract the anniversaries cost eleven per cent of account value against six per cent of
    # fee collected, and no hedge touches the difference.
    if "liability_pre" in ledger:
        out["anniversary"] = -(
            ledger["liability"].to_numpy() - ledger["liability_pre"].to_numpy()
        )
    else:
        out["anniversary"] = 0.0
    # The liability's own passage of time, at the rate the proxy's interpolation implies, with
    # the sign flipped because the insurer is short it.
    out["theta"] = -lagged("theta") * d_time
    out["cost"] = -(ledger["trade_cost"].to_numpy() + ledger["carry_cost"].to_numpy())

    out["total"] = ledger["pnl"].to_numpy()
    names = ["delta", "basis", "gamma", "vega", "rate", "theta", "anniversary", "carry",
             "cost"]
    out["residual"] = out["total"] - out[names].sum(axis=1)
    return out


def summarise(pieces: pd.DataFrame, account_value: float) -> dict:
    """Each bar's total and its contribution to the variance, as shares of contract value.

    Two columns because they answer different questions. The total says what the term cost over
    the period; the standard deviation says how much of the day-to-day noise it was responsible
    for. A hedge can have a small delta total and still be carrying most of its risk in delta,
    and the other way round.
    """
    names = ["delta", "basis", "gamma", "vega", "rate", "theta", "anniversary", "carry",
             "cost", "residual"]
    out = {}
    for name in names:
        series = pieces[name].to_numpy()
        out[f"{name}_total_pct"] = float(series.sum() / account_value)
        out[f"{name}_sd_pct"] = float(series.std(ddof=1) / account_value)
    out["total_pct"] = float(pieces["total"].sum() / account_value)
    out["reconciliation_error"] = float(np.abs(
        pieces["total"] - pieces[names[:-1]].sum(axis=1) - pieces["residual"]
    ).max())
    out["residual_share_of_sd"] = float(
        pieces["residual"].std(ddof=1) / max(pieces["total"].std(ddof=1), 1e-12)
    )
    return out


def volatility_comparison(ledger: pd.DataFrame) -> dict:
    """Realised against implied, which is what the gamma bar is the price of.

    The gamma term accumulates to about one half gamma times realised variance, and the premium
    paid for the convexity is the implied variance. A short-gamma book makes money on this
    comparison when realised comes in below implied and loses when it does not, and reporting
    the gamma bar without the comparison hides which of those happened.
    """
    returns = np.diff(np.log(np.maximum(ledger["account_value"].to_numpy(), 1e-12)))
    years = (ledger.index[-1] - ledger.index[0]).days / 365.0
    realised = float(np.sqrt(np.sum(returns ** 2) / max(years, 1e-9)))
    implied = float(np.sqrt(np.mean(ledger["variance"].to_numpy())))
    return {
        "realised_vol": realised,
        "implied_vol_mean": implied,
        "realised_less_implied": realised - implied,
        "years": years,
    }


def table(runs, equity_weight: float, account_value: float) -> pd.DataFrame:
    """One row per strategy per path, for the experiment tables."""
    rows = []
    for run in runs:
        pieces = attribute(run.ledger, equity_weight)
        rows.append({
            "path": run.path,
            "strategy": run.strategy,
            **summarise(pieces, account_value),
            **volatility_comparison(run.ledger),
        })
    return pd.DataFrame(rows)
