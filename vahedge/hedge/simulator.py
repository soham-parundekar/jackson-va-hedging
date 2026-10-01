"""Roll the contract and its hedge along a path, day by day, and keep the books.

The accounting is the part to get right, because a hedging study is a statement about a
difference between two large numbers and any leak in the ledger shows up as a result.

What is tracked is the insurer's net worth attributable to the rider:

    W = cash + hedge mark - L

with L the model value of the guarantee - the present value of projected claims less the
projected fees attributed to it - and cash everything that has actually moved. Over a period

    dW = (fees collected - claims paid) - dL + hedge mark to market - costs + interest on cash

and under the risk-neutral measure the expectation of that is zero before costs, because L is
by construction the value of what is left. So the realised dW is the hedging error, and the
continuous-hedge limit test is that it falls towards zero as the rebalancing interval does. Any
term left out of the ledger turns up as a drift in dW that reads as a hedging result.

Attribution is one here rather than the accounting percentage. With the full fee stream against
the full claim stream, L is the economic value of the rider and dW is the economic result of
writing and hedging it. The accounting lens, where a fixed share of fees offsets the benefit and
the rest runs through income, is a different question and belongs in W5.

Three things in the mechanics are decisions rather than implementation.

*The contract is rolled by the same annual recursion the valuation uses*, on a one-path market
built from the realised series. The step-up, the bonus, the guaranteed withdrawal and the
benefit-base adjustment are the contract, and a second implementation of them here would
eventually disagree with the first. Between anniversaries the contract value tracks the
sub-account and nothing else happens, so the daily value is the previous anniversary's value
scaled by the fund's return since - and it drops at each anniversary, when the charges come out
and the withdrawal is taken. That drop is not a loss to the insurer. The withdrawal is the
policyholder spending their own account value, and the only reason it matters is that it brings
exhaustion closer, which is already in the liability's value.

*Mortality and lapse are on in the valuation and off in the roll-forward.* The policy being
followed is by construction one that persisted, so its realised fees and withdrawals are the
full amounts; its value has to account for the chance that it stops persisting from here, which
is what the proxy was fitted with. Charging the roll for a lapse that did not happen would
flatter the hedge.

*Options are held, not re-struck every week.* Futures and swaps are marked daily and adjusting
them costs only the change in position, which is why the code closes and reopens them each
rebalance. An option is not like that: re-striking a one-year put every week would pay the
spread on the whole position fifty-two times a year, which is a modelling artefact rather than
a cost anyone incurs. So a held put is kept, and only the change in quantity trades, until its
strike has drifted out of band or it is close enough to expiry to roll. The sizing solve is
given the option actually held rather than a freshly struck one, so the hedge ratio is computed
off the Greeks of the position that exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..liability import gmwb
from ..market.simulate import MarketPaths
from . import instruments as inst
from . import sizing
from .strategies import Strategy, rebalance_dates

DAYS_PER_YEAR = 365.0
# A held put is rolled when its strike has drifted this far from the target moneyness, or when
# this much of its original life is left. Both are assumptions; E3 sweeps the first.
STRIKE_DRIFT_BAND = 0.05
ROLL_WHEN_LIFE_LEFT = 0.25


@dataclass(frozen=True)
class HedgeRun:
    """One strategy on one path: the daily ledger, the positions, and the headline numbers."""

    strategy: str
    path: str
    ledger: pd.DataFrame
    positions: pd.DataFrame
    summary: dict


@dataclass
class _Position:
    """One line of the hedge book, and where its mark is measured from."""

    instrument: object
    units: float
    reference: object          # a strike level, a par rate, or None for an option
    opened_at: float           # policy time, for the option roll rule
    role: str = "solved"       # "solved" or "overlay"; the overlay is not sized by the solve

    def mark(self, market) -> float:
        if isinstance(self.instrument, inst.IndexPut):
            return self.units * self.instrument.value(market)
        return self.units * self.instrument.value(market, self.reference)


def anniversary_indices(path, step_days: float = DAYS_PER_YEAR) -> np.ndarray:
    """Positions in the daily path closest to each policy anniversary.

    Snapped to a trading date rather than interpolated, because the contract's charges and
    withdrawal happen on a date and the hedge has to be able to trade on it.
    """
    years = path.year_fraction
    wanted = np.arange(0.0, years[-1] + 1e-9, step_days / DAYS_PER_YEAR)
    return np.unique(np.abs(years[:, None] - wanted[None, :]).argmin(axis=0))


def annual_market(path, anniversaries: np.ndarray) -> MarketPaths:
    """A one-path ``MarketPaths`` built from the realised series, for the contract recursion.

    Only what the recursion and the recording use is filled: the fund's return over each policy
    year, the realised money-market discount factor to each anniversary, and the state at each
    anniversary. This is a realised path, so nothing stochastic is left in it.
    """
    years = path.year_fraction
    cash_integral = np.concatenate([[0.0], np.cumsum(
        0.5 * (path.cash_rate[:-1] + path.cash_rate[1:]) * np.diff(years)
    )])
    fund = path.fund[anniversaries]
    equity = path.index[anniversaries]
    return MarketPaths(
        fund_growth=(fund[1:] / fund[:-1]).reshape(1, -1),
        index_growth=(equity[1:] / equity[:-1]).reshape(1, -1),
        discount=np.exp(-cash_integral[anniversaries][1:]).reshape(1, -1),
        short_rate=path.cash_rate[anniversaries][1:].reshape(1, -1),
        variance=path.variance[anniversaries][1:].reshape(1, -1),
        zero_10y=path.zero_10y[anniversaries][1:].reshape(1, -1),
        realised_variance=path.variance[anniversaries][1:].reshape(1, -1),
        seed=0,
        antithetic=False,
    )


def roll_contract(policy_book, path, survival, deaths, equity_weight: float = 1.0) -> dict:
    """The contract's annual state along the realised path, and the cash flows it produced."""
    anniversaries = anniversary_indices(path)
    market = annual_market(path, anniversaries)
    years = market.n_years
    if survival.shape[1] < years:
        raise ValueError(f"survival covers {survival.shape[1]} years, the path needs {years}")
    # The recursion refuses a market shorter than the contract's horizon, and rightly: valuing
    # a thirty-year guarantee on two years of paths would silently drop the rest of it. Here the
    # projection is not a valuation - it is the contract's state along a window of realised
    # history, and the window is all there is - so the horizon is cut to match it explicitly.
    from dataclasses import replace as _replace
    windowed = _replace(policy_book, projection_years=np.full(policy_book.size, years))
    projection = gmwb.project(
        windowed, market, survival[:, :years], deaths[:, :years],
        equity_weight=equity_weight, record=True,
    )
    recorded = projection.recorded
    # Undo the discounting and the persistency weighting the recording applies, to get the
    # amount an in-force policy actually paid and collected in each policy year.
    scale = recorded["discount"][0] * np.maximum(recorded["persistency"][0], 1e-12)
    return {
        "anniversaries": anniversaries,
        "account_value": recorded["account_value"][0],
        "benefit_base": recorded["benefit_base"][0],
        "claim": recorded["pv_claim"][0] / scale,
        "fee": recorded["pv_fee"][0] / scale,
        "projection": projection,
    }


def daily_state(path, rolled: dict, opening_account: float, opening_base: float):
    """Contract value and benefit base on every date, and the cash flows by date.

    Within a policy year the contract value is the previous anniversary's value grown by the
    sub-account; on the anniversary it steps to the recursion's own post-charge, post-withdrawal
    figure. Anchoring on the start of each year rather than the end is what puts the charges at
    the right end of it: the other way round implies the year's fees were taken before the
    year's growth, which on a 3% drag over ten years is not a rounding difference.
    """
    anniversaries = rolled["anniversaries"]
    account = np.empty(path.dates.size)
    base = np.empty(path.dates.size)
    cash_flow = np.zeros(path.dates.size)

    account[:anniversaries[0] + 1] = opening_account
    base[:anniversaries[0] + 1] = opening_base
    anchor_account, anchor_base = opening_account, opening_base

    for position, (begin, end) in enumerate(zip(anniversaries[:-1], anniversaries[1:])):
        span = slice(begin + 1, end)
        account[span] = anchor_account * path.fund[span] / path.fund[begin]
        base[span] = anchor_base
        anchor_account = rolled["account_value"][position]
        anchor_base = rolled["benefit_base"][position]
        account[end] = anchor_account
        base[end] = anchor_base
        cash_flow[end] = rolled["fee"][position] - rolled["claim"][position]

    tail = anniversaries[-1] + 1
    if tail < path.dates.size:
        account[tail:] = anchor_account * path.fund[tail:] / path.fund[anniversaries[-1]]
        base[tail:] = anchor_base
    return account, base, cash_flow


def run(
    path,
    policy_book,
    survival,
    deaths,
    proxy,
    strategy: Strategy,
    smile,
    years_at_start: float,
    equity_weight: float = 1.0,
    dividend_yield: float = 0.0,
    cost_multiple: float = 1.0,
    greeks_override=None,
) -> HedgeRun:
    """One strategy on one path.

    ``years_at_start`` is the contract's age on the first date, which is what lines the proxy's
    annual fits up with the calendar. ``cost_multiple`` scales every transaction cost at once,
    which is the sweep E3 runs: a conclusion that survives at twice the assumed cost is worth
    more than one quoted at a single level. ``greeks_override`` replaces the proxy's Greeks with
    another function of the same state, which is how E4 hedges a Heston world with the wrong
    model's sensitivities.
    """
    rolled = roll_contract(policy_book, path, survival, deaths, equity_weight)
    account, base, cash_flow = daily_state(
        path, rolled, float(policy_book.account_value[0]), float(policy_book.benefit_base[0])
    )
    policy_year = path.year_fraction + years_at_start
    calendar = rebalance_dates(path.dates, strategy.rebalance)
    on_band = strategy.rebalance == "band"
    greek_source = greeks_override or (lambda **kwargs: proxy.greeks_at(**kwargs))

    book: list = []
    rows, position_rows = [], []
    cash = 0.0
    years = path.year_fraction

    for day in range(path.dates.size):
        market = inst.HedgeMarket(
            index=float(path.index[day]), curve=path.curves[day],
            volatility=float(path.implied_vol[day]), smile=smile,
            dividend_yield=dividend_yield, financing_rate=float(path.cash_rate[day]),
        )
        greeks = greek_source(
            years_since_issue=float(policy_year[day]), account_value=float(account[day]),
            benefit_base=float(base[day]), variance=float(path.variance[day]),
            zero_10y=float(path.zero_10y[day]), attribution=1.0,
        )
        liability = float(np.ravel(greeks["value"])[0])
        exposure = sizing.Exposures(
            delta=-float(np.ravel(greeks["delta"])[0]),
            gamma=-float(np.ravel(greeks["gamma"])[0]),
            vega=-float(np.ravel(greeks["vega"])[0]),
            rho=-float(np.ravel(greeks["rho_per_bp"])[0]),
        )

        step = 0.0 if day == 0 else years[day] - years[day - 1]
        interest = cash * (np.exp(float(path.cash_rate[day]) * step) - 1.0)
        carry = cost_multiple * sum(
            line.instrument.carry(line.units, market, step) for line in book
        )

        trade_cost = 0.0
        rebalancing = day == 0 or (calendar[day] and (
            not on_band or _drifted(exposure, book, market, strategy, float(account[day]))
        ))
        if rebalancing:
            book, trade_cost, cash_delta, solved = _rebalance(
                book, strategy, market, exposure, float(account[day]), float(policy_year[day]),
                cost_multiple,
            )
            cash += cash_delta
            position_rows.append({
                "date": path.dates[day],
                **{f"units_{name}": line.units
                   for name, line in zip(_names([l.instrument for l in book]), book)},
                "target_delta": solved.target.delta,
                "achieved_delta": solved.achieved.delta,
                "target_rho": solved.target.rho,
                "achieved_rho": solved.achieved.rho,
                "solved_notional": solved.notional(market),
            })

        hedge_mark = float(sum(line.mark(market) for line in book))
        cash += cash_flow[day] + interest - trade_cost - carry
        rows.append({
            "date": path.dates[day],
            "policy_year": policy_year[day],
            "index": path.index[day],
            "fund": path.fund[day],
            "account_value": account[day],
            "benefit_base": base[day],
            "implied_vol": path.implied_vol[day],
            "variance": path.variance[day],
            "zero_10y": path.zero_10y[day],
            "liability": liability,
            "delta": float(np.ravel(greeks["delta"])[0]),
            "gamma": float(np.ravel(greeks["gamma"])[0]),
            "vega": float(np.ravel(greeks["vega"])[0]),
            "rho_per_bp": float(np.ravel(greeks["rho_per_bp"])[0]),
            "hedge_delta": _book_exposure(book, market).delta,
            "hedge_mark": hedge_mark,
            "cash": cash,
            "cash_flow": cash_flow[day],
            "trade_cost": trade_cost,
            "carry_cost": carry,
            "interest": interest,
            "net_worth": cash + hedge_mark - liability,
            "rebalanced": rebalancing,
        })

    ledger = pd.DataFrame(rows).set_index("date")
    ledger["pnl"] = ledger["net_worth"].diff().fillna(0.0)
    return HedgeRun(
        strategy=strategy.name,
        path=path.label,
        ledger=ledger,
        positions=(pd.DataFrame(position_rows).set_index("date")
                   if position_rows else pd.DataFrame()),
        summary=summarise(ledger, float(policy_book.account_value[0])),
    )


# ---------------------------------------------------------------- the hedge book


def _rebalance(book, strategy: Strategy, market, exposure, account_value: float,
               policy_time: float, cost_multiple: float):
    """Close what has to close, resize the rest, and return the new book.

    Returns the new book, the transaction cost, the change in cash, and the solve, so the
    caller can record what the hedge was asked for alongside what it got.
    """
    kept_options, rolled_options, overlay_lines, linear = [], [], [], []
    for line in book:
        if not isinstance(line.instrument, inst.IndexPut):
            linear.append(line)
        elif line.role == "overlay":
            overlay_lines.append(line)
        elif _needs_roll(line, market, policy_time):
            rolled_options.append(line)
        else:
            kept_options.append(line)

    # The instruments the solve is allowed to use: the linear specs, plus whichever options
    # survive the roll test as they actually stand, plus freshly struck ones for the rest.
    kept_by_spec = {_put_key(line.instrument): line for line in kept_options}
    solve_instruments, matched = [], []
    for spec in strategy.instruments:
        if isinstance(spec, inst.IndexPut):
            existing = kept_by_spec.pop(_put_key_spec(spec), None)
            instrument = existing.instrument if existing else spec.struck(market)
            solve_instruments.append(instrument)
            matched.append(existing)
        else:
            solve_instruments.append(spec)
            matched.append(None)
    # Any kept option whose spec is no longer in the strategy is closed.
    rolled_options.extend(kept_by_spec.values())

    solved = sizing.solve(
        solve_instruments, market, exposure,
        weights=strategy.weights, hedge_ratio=strategy.hedge_ratio,
    )

    cash_delta = 0.0
    cost = 0.0
    # Linear positions are marked daily, so the old book is realised into cash and the new one
    # is struck at today's level. Only the change in units pays a spread.
    for line in linear:
        cash_delta += line.mark(market)
    old_linear = _netted(linear)
    new_book: list = []
    for spec, instrument, existing, units in zip(
        strategy.instruments, solve_instruments, matched, solved.units
    ):
        if isinstance(spec, inst.IndexPut):
            previous = existing.units if existing else 0.0
            if existing is None:
                cost += instrument.trade_cost(units, market)
                cash_delta -= units * instrument.value(market)
            else:
                change = units - previous
                cost += instrument.trade_cost(change, market)
                cash_delta -= change * instrument.value(market)
            new_book.append(_Position(
                instrument=instrument, units=float(units), reference=None,
                opened_at=existing.opened_at if existing else policy_time,
            ))
        else:
            change = float(units) - old_linear.get(spec, 0.0)
            cost += spec.trade_cost(change, market)
            new_book.append(_Position(
                instrument=spec, units=float(units), reference=_reference(spec, market),
                opened_at=policy_time,
            ))

    for line in rolled_options:
        cash_delta += line.mark(market)
        cost += line.instrument.trade_cost(line.units, market)

    for spec, share in strategy.overlay:
        units = share * account_value / market.index
        existing = next(
            (line for line in overlay_lines
             if abs(line.instrument.strike_over_spot - spec.strike_over_spot) < 1e-12
             and abs(line.instrument.maturity - spec.maturity) < 1e-12),
            None,
        )
        if existing is None or _needs_roll(existing, market, policy_time):
            if existing is not None:
                cash_delta += existing.mark(market)
                cost += existing.instrument.trade_cost(existing.units, market)
            struck = spec.struck(market)
            cost += struck.trade_cost(units, market)
            cash_delta -= units * struck.value(market)
            new_book.append(_Position(instrument=struck, units=float(units), reference=None,
                                      opened_at=policy_time, role="overlay"))
        else:
            change = units - existing.units
            cost += existing.instrument.trade_cost(change, market)
            cash_delta -= change * existing.instrument.value(market)
            new_book.append(_Position(instrument=existing.instrument, units=float(units),
                                      reference=None, opened_at=existing.opened_at,
                                      role="overlay"))

    return new_book, cost * cost_multiple, cash_delta, solved


def _needs_roll(line: _Position, market, policy_time: float) -> bool:
    """Whether a held put has drifted too far from its target strike, or run too close to expiry."""
    instrument = line.instrument
    elapsed = policy_time - line.opened_at
    life_left = instrument.maturity - elapsed
    if life_left <= ROLL_WHEN_LIFE_LEFT * instrument.maturity:
        return True
    moneyness = instrument.strike / market.index
    return abs(moneyness - instrument.strike_over_spot) > STRIKE_DRIFT_BAND


def _put_key(instrument) -> tuple:
    return (round(instrument.maturity, 6), round(instrument.strike_over_spot, 6))


def _put_key_spec(spec) -> tuple:
    return (round(spec.maturity, 6), round(spec.strike_over_spot, 6))


def _netted(lines) -> dict:
    out: dict = {}
    for line in lines:
        out[line.instrument] = out.get(line.instrument, 0.0) + line.units
    return out


def _reference(instrument, market):
    if isinstance(instrument, inst.EquityFuture):
        return market.forward(instrument.maturity)
    if isinstance(instrument, inst.TotalReturnSwap):
        return market.index
    if isinstance(instrument, inst.InterestRateSwap):
        return instrument.par_rate(market)
    return float(market.curve.zero(instrument.duration))


def _book_exposure(book, market) -> sizing.Exposures:
    total = sizing.Exposures()
    for line in book:
        total = total + line.instrument.exposures(market) * line.units
    return total


def _drifted(exposure, book, market, strategy: Strategy, account_value: float) -> bool:
    """Whether a band strategy has moved far enough to be worth a trade.

    The band is on the delta left open as a share of the contract value, because that is what
    the rebalance exists to close and what a risk limit would be written on.
    """
    if not book:
        return True
    left = exposure.delta + _book_exposure(book, market).delta
    return bool(abs(left) > strategy.band * account_value)


def _names(instruments) -> list:
    seen: dict = {}
    out = []
    for instrument in instruments:
        label = instrument.label.replace(" ", "_")
        seen[label] = seen.get(label, 0) + 1
        out.append(label if seen[label] == 1 else f"{label}_{seen[label]}")
    return out


def summarise(ledger: pd.DataFrame, account_value: float) -> dict:
    """The numbers a strategy is judged on, as a share of the contract value at the start.

    Scale-free because that is the only form in which a hundred-thousand-dollar policy and a
    two-hundred-billion-dollar book can be compared, which is the point of running a model
    policy at all.
    """
    pnl = ledger["pnl"].to_numpy()
    cumulative = np.cumsum(pnl)
    worst = int(np.argmin(pnl))
    drawdown = cumulative - np.maximum.accumulate(cumulative)
    return {
        "days": int(ledger.shape[0]),
        "rebalances": int(ledger["rebalanced"].sum()),
        "total_pnl_pct": float(pnl.sum() / account_value),
        "pnl_sd_pct": float(pnl.std(ddof=1) / account_value),
        "worst_day_pct": float(pnl[worst] / account_value),
        "worst_day": ledger.index[worst],
        "worst_drawdown_pct": float(drawdown.min() / account_value),
        "total_cost_pct": float(
            (ledger["trade_cost"].sum() + ledger["carry_cost"].sum()) / account_value
        ),
        "liability_start_pct": float(ledger["liability"].iloc[0] / account_value),
        "liability_end_pct": float(ledger["liability"].iloc[-1] / account_value),
        # The insurer carries the negative of the liability's delta, so what is left open is
        # the hedge book's delta less the liability's.
        "mean_abs_delta_left_pct": float(
            (ledger["hedge_delta"] - ledger["delta"]).abs().mean() / account_value
        ),
    }
