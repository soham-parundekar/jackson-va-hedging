"""Delta and rho hedging of the rider, backtested on realised market history.

The experiment is a single policy issued at the start of the window and hedged
weekly until the end of it. Each week the rider is revalued at the market that
actually prevailed, the equity and rate exposures are recomputed, the hedge is
resized, and the next week's realised move is applied to both the liability and the
hedge. Nothing about week i+1 is used to size the hedge held over week i.

Sign conventions follow the liability. The rider value is positive when the insurer
owes, so the insurer's profit is minus the change in it. Equity exposure is negative,
which makes the hedge a short index position - the right answer for something that
behaves like a written put. Rho is negative, so the rate hedge is a receive-fixed
swap, which loses when yields rise, matching a liability that shrinks when yields
rise.

Period profit is put together as

    (attributed fees collected - claims paid)
      - (change in rider value)
      + (risk-free return on the assets backing the rider)
      + (equity futures profit)
      + (receive-fixed swap profit)
      - (transaction costs)

The financing term is there so that a world with no market moves produces zero
profit rather than the accretion of the liability. Futures profit is already an
excess return over financing, so the two are on the same footing.

Three sources of basis risk are left deliberately unhedged, because a real programme
cannot hedge them either. The rate hedge is a single instrument against a parallel
shift, so curve reshaping leaks through. The equity hedge is one index against a
sub-account that in Jackson's book is a mix of equity, bond and balanced funds.
Volatility is not hedged at all.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import market, mortality as mortality_module
from .contract import GmwbContract
from .curves import ZeroCurve
from .engine import projection_years, value_rider
from .sensitivities import compute_greeks

DAYS_PER_YEAR = 365.25


@dataclass
class PolicyState:
    """The representative policy as it rolls along a realised market path."""

    account_value: float
    benefit_base: float
    issue_date: pd.Timestamp
    next_anniversary: pd.Timestamp
    anniversaries_passed: int = 0

    @classmethod
    def at_issue(cls, contract: GmwbContract, issue_date) -> "PolicyState":
        issue_date = pd.Timestamp(issue_date)
        return cls(
            account_value=contract.premium,
            benefit_base=contract.premium,
            issue_date=issue_date,
            next_anniversary=issue_date + pd.DateOffset(years=1),
        )

    def years_to_anniversary(self, date) -> float:
        """Time to the next anniversary in years, clipped into (0, 1]."""
        days = (self.next_anniversary - pd.Timestamp(date)).days
        return float(min(max(days / DAYS_PER_YEAR, 1.0 / DAYS_PER_YEAR), 1.0))

    def attained_age(self, contract: GmwbContract, date) -> int:
        elapsed = (pd.Timestamp(date) - self.issue_date).days / DAYS_PER_YEAR
        return int(contract.issue_age + int(elapsed))

    def step(
        self,
        contract: GmwbContract,
        to_date,
        sub_account_return: float,
        dt: float,
    ) -> dict[str, float]:
        """Advance the policy over one rebalance period and return the cash flows.

        ``sub_account_return`` is the realised total return of the contract holder's
        fund allocation over the period, before charges. Charges that accrue on
        average daily value come out as a continuous drag, and the amount collected
        uses the same identity as the valuation engine, so a realised path and a
        simulated path account for fees the same way.

        An anniversary falling inside the period is applied at the end of it. On a
        weekly grid that misdates the withdrawal by at most a few days.
        """
        gross = 1.0 + sub_account_return
        if gross <= 0:
            raise ValueError(f"implausible sub-account return {gross:.4f} over {dt:.4f}y")

        av_pre_drag = self.account_value * gross
        drag_factor = np.exp(-contract.account_drag * dt)
        self.account_value = av_pre_drag * drag_factor
        insurer_share = contract.base_contract_charge / contract.account_drag
        me_charge = insurer_share * av_pre_drag * (1.0 - drag_factor)

        rider_charge = 0.0
        claim = 0.0
        to_date = pd.Timestamp(to_date)
        if to_date >= self.next_anniversary:
            rider_charge = min(contract.rider_charge_pct * self.benefit_base, self.account_value)
            self.account_value -= rider_charge

            gawa = contract.gawa_pct * self.benefit_base
            from_account = min(gawa, self.account_value)
            claim = gawa - from_account
            self.account_value -= from_account

            if contract.annual_step_up:
                self.benefit_base = max(self.benefit_base, self.account_value)

            self.anniversaries_passed += 1
            self.next_anniversary = self.issue_date + pd.DateOffset(
                years=self.anniversaries_passed + 1
            )

        return {"me_charge": me_charge, "rider_charge": rider_charge, "claim": claim}


@dataclass(frozen=True)
class SubAccountMix:
    """The contract holder's fund allocation, and how its return is reconstructed.

    Jackson discloses the separate account split by fund type: at 31 December 2025,
    $171.0bn equity, $19.7bn bond, $43.3bn balanced and $2.3bn money market out of
    $236.4bn. A guarantee written over that mix is not a guarantee over the S&P 500,
    and a hedge that trades the index alone carries the difference as basis risk.

    Bond fund return over a period is taken as accrual at the prevailing 10-year par
    yield less duration times the yield change, which is the first-order return of a
    portfolio at that duration. Balanced funds are split ``balanced_equity_share``
    equity and the rest bond.
    """

    equity: float
    bond: float
    balanced: float
    money_market: float
    bond_duration: float = 6.0
    balanced_equity_share: float = 0.6

    def __post_init__(self) -> None:
        total = self.equity + self.bond + self.balanced + self.money_market
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"sub-account weights must sum to 1, got {total}")

    @property
    def effective_equity_beta(self) -> float:
        """Equity exposure of the mix, which is what the valuation model's single
        risky sub-account is calibrated to."""
        return self.equity + self.balanced * self.balanced_equity_share

    @classmethod
    def from_config(cls, cfg: dict) -> "SubAccountMix":
        m = cfg["subaccount_mix"]
        return cls(
            equity=float(m["equity"]),
            bond=float(m["bond"]),
            balanced=float(m["balanced"]),
            money_market=float(m["money_market"]),
            bond_duration=float(m["bond_duration"]),
            balanced_equity_share=float(m["balanced_equity_share"]),
        )

    @classmethod
    def all_equity(cls) -> "SubAccountMix":
        return cls(equity=1.0, bond=0.0, balanced=0.0, money_market=0.0)

    def realised_return(
        self,
        index_total_return: float,
        bond_yield: float,
        bond_yield_change: float,
        short_rate: float,
        dt: float,
    ) -> float:
        bond_return = bond_yield * dt - self.bond_duration * bond_yield_change
        equity_weight = self.equity + self.balanced * self.balanced_equity_share
        bond_weight = self.bond + self.balanced * (1.0 - self.balanced_equity_share)
        return (
            equity_weight * index_total_return
            + bond_weight * bond_return
            + self.money_market * short_rate * dt
        )


def swap_annuity(curve: ZeroCurve, tenor_years: float, frequency: int = 2) -> float:
    """Present value of one unit of annual fixed coupon on a par swap of this tenor."""
    times = np.arange(1, int(round(tenor_years * frequency)) + 1) / frequency
    return float(curve.discount(times).sum() / frequency)


def roll_policy(
    contract: GmwbContract,
    cfg: dict,
    frame: pd.DataFrame,
    mix: "SubAccountMix",
    issue_date,
    to_date,
    frequency: str = "weekly",
) -> tuple[PolicyState, pd.DataFrame]:
    """Advance a policy from issue along realised market history, without valuing it.

    Useful on its own: an in-force policy's account value and benefit base are not
    assumptions, they are the arithmetic consequence of the returns that happened and
    the withdrawals that were taken. Rolling a policy issued in 2016 forward to a
    disclosed balance-sheet date gives a state that can be compared with the book
    rather than guessed at.
    """
    dates = rebalance_dates(frame, issue_date, to_date, frequency)
    if len(dates) < 2:
        raise ValueError("need at least two dates to roll a policy")

    dividend_yield = float(cfg["dividend_yield"])
    contract = replace(contract, fund_equity_beta=mix.effective_equity_beta)
    policy = PolicyState.at_issue(contract, dates[0])

    history = []
    prev = None
    for date in dates:
        spot = float(frame.loc[date, "SP500"])
        bond_yield = float(frame.loc[date, "DGS10"]) / 100.0
        short_rate = _short_rate(frame, date)
        if prev is not None:
            dt = (date - prev["date"]).days / DAYS_PER_YEAR
            sub_return = mix.realised_return(
                index_total_return=spot / prev["spot"] - 1.0 + dividend_yield * dt,
                bond_yield=prev["bond_yield"],
                bond_yield_change=bond_yield - prev["bond_yield"],
                short_rate=prev["short_rate"],
                dt=dt,
            )
            policy.step(contract, date, sub_return, dt)
        history.append(
            {
                "date": date,
                "spot": spot,
                "account_value": policy.account_value,
                "benefit_base": policy.benefit_base,
                "attained_age": policy.attained_age(contract, date),
                "years_to_anniversary": policy.years_to_anniversary(date),
            }
        )
        prev = {"date": date, "spot": spot, "bond_yield": bond_yield, "short_rate": short_rate}

    return policy, pd.DataFrame(history).set_index("date")


def rebalance_dates(frame: pd.DataFrame, start, end, frequency: str = "weekly") -> pd.DatetimeIndex:
    """Trading dates on which the hedge is reset.

    Weekly takes the last trading day of each calendar week, which is the Friday
    except in holiday weeks. Monthly takes the last trading day of each month.
    """
    dates = market.equity_dates(frame, start, end)
    if frequency == "daily":
        return dates
    series = pd.Series(dates, index=dates)
    rule = {"weekly": "W", "monthly": "ME"}.get(frequency)
    if rule is None:
        raise ValueError(f"unsupported rebalance frequency {frequency!r}")
    picked = series.groupby(series.index.to_period("W" if frequency == "weekly" else "M")).max()
    return pd.DatetimeIndex(sorted(picked.to_numpy()))


def run_backtest(
    contract: GmwbContract,
    cfg: dict,
    frame: pd.DataFrame,
    normals: np.ndarray,
    fee_attribution: float,
    long_run_vol: float,
    mix: "SubAccountMix | None" = None,
    hedge_equity: bool = True,
    hedge_rates: bool = True,
    hedge_vega: bool = False,
    with_gaap: bool = False,
) -> pd.DataFrame:
    """Hedge the rider through the window in cfg['hedge'] and return the period ledger.

    Set ``hedge_equity`` or ``hedge_rates`` to False to isolate what each leg
    contributes. With both False the ledger is the unhedged liability.

    ``hedge_vega`` adds a stylised long-volatility overlay sized on the tradeable
    vega. Jackson does hold index put options alongside its futures and total return
    swaps, so a vega leg is not a hypothetical instrument for this book. The overlay
    here charges a spread on the vega traded but carries no cost of being long
    volatility over time, and implied volatility exceeds realised volatility on
    average, so treat the variance reduction it produces as informative and its
    cumulative profit as flattering.

    ``mix`` is the contract holder's fund allocation. Passing the disclosed book mix
    makes the policy's realised return differ from the index the hedge trades, which
    is the basis risk a real overlay carries. Passing ``SubAccountMix.all_equity()``
    removes it and isolates what discrete rebalancing and curve reshaping cost on
    their own.

    ``with_gaap`` adds two further valuations per date on the reporting basis: the
    margin-loaded mortality table, once discounted on the Treasury curve and once on
    the curve plus Jackson's own non-performance spread. Those two columns are what
    ``accounting`` needs to separate the part of the reported movement that runs
    through net income from the part that runs through other comprehensive income.
    """
    hcfg = cfg["hedge"]
    dates = rebalance_dates(frame, hcfg["start"], hcfg["end"], hcfg["rebalance"])
    if len(dates) < 30:
        raise ValueError(f"only {len(dates)} rebalance dates in the window")

    dividend_yield = float(cfg["dividend_yield"])
    max_age = int(cfg["simulation"]["max_age"])
    hedge_ratio = float(hcfg["hedge_ratio"])
    swap_tenor = float(hcfg["swap_tenor_years"])
    equity_cost = float(hcfg["equity_cost_bp"]) / 10000.0
    rate_cost = float(hcfg["rate_cost_bp"]) / 10000.0
    vega_cost = float(hcfg.get("vega_cost_pct", 0.02))
    male_weight = float(cfg["contract"]["sex_mix"]["male"])
    gcfg = cfg["greeks"]

    own_credit_scale = float(cfg["accounting"]["own_credit_scale"])
    if mix is None:
        mix = SubAccountMix.all_equity()
    # The valuation model carries one risky sub-account. Calibrating its equity beta to
    # the mix's equity exposure is the closest a single-factor model gets; what the
    # model then cannot see is the bond leg, and that gap is the basis risk.
    contract = replace(contract, fund_equity_beta=mix.effective_equity_beta)
    policy = PolicyState.at_issue(contract, dates[0])
    mortality_cache: dict[tuple[int, int, str], mortality_module.MortalityBasis] = {}

    def basis_for(age: int, year: int, table: str) -> mortality_module.MortalityBasis:
        key = (age, year, table)
        if key not in mortality_cache:
            mortality_cache[key] = mortality_module.load(age, year, male_weight, table=table)
        return mortality_cache[key]

    prev = None
    rows = []
    equity_notional = 0.0
    swap_notional = 0.0
    vega_notional = 0.0

    for i, date in enumerate(dates):
        state = market.state_at(frame, date, cfg, long_run_vol)
        curve = state.curve_builder.build()

        # Roll the policy forward on realised returns before revaluing it. The cash
        # flows returned belong to the period that ends on this date.
        bond_yield_now = float(frame.loc[date, "DGS10"]) / 100.0
        if prev is None:
            dt = 0.0
            price_return = 0.0
            sub_account_return = 0.0
            cash = {"me_charge": 0.0, "rider_charge": 0.0, "claim": 0.0}
        else:
            dt = (date - prev["date"]).days / DAYS_PER_YEAR
            price_return = state.spot / prev["spot"] - 1.0
            sub_account_return = mix.realised_return(
                index_total_return=price_return + dividend_yield * dt,
                bond_yield=prev["bond_yield"],
                bond_yield_change=bond_yield_now - prev["bond_yield"],
                short_rate=prev["short_rate"],
                dt=dt,
            )
            cash = policy.step(contract, date, sub_account_return, dt)

        attained = policy.attained_age(contract, date)
        aged_contract = replace(contract, issue_age=attained)
        basis = basis_for(attained, date.year, "basic")

        valuation_state = {
            "account_value": policy.account_value,
            "benefit_base": policy.benefit_base,
            "fee_attribution": fee_attribution,
            "max_age": max_age,
            "first_step_years": policy.years_to_anniversary(date),
        }
        n_years = projection_years(aged_contract, max_age)
        greeks = compute_greeks(
            aged_contract,
            state.curve_builder,
            state.vol,
            basis,
            normals[:, :n_years],
            valuation_state,
            equity_bump=float(gcfg["equity_bump_pct"]),
            rate_bump_bp=float(gcfg["rate_bump_bp"]),
            vol_bump=float(gcfg["vol_bump"]),
        )

        row = {
            "date": date,
            "spot": state.spot,
            "par_2y": float(frame.loc[date, "DGS2"]),
            "par_10y": float(frame.loc[date, "DGS10"]),
            "par_30y": float(frame.loc[date, "DGS30"]),
            "swap_rate": float(frame.loc[date, market.PAR_YIELD_COLUMNS[int(swap_tenor)]]),
            "implied_vol_3m": state.implied_vol_3m,
            "account_value": policy.account_value,
            "benefit_base": policy.benefit_base,
            "attained_age": attained,
            "first_step_years": valuation_state["first_step_years"],
            "rider_value": greeks.base_value,
            "equity_exposure": greeks.equity_exposure,
            "equity_gamma": greeks.equity_gamma,
            "rho_per_bp": greeks.rho_per_bp,
            "vega_per_point": greeks.vega_per_point,
        }

        if with_gaap:
            reporting_basis = basis_for(attained, date.year, "period")
            spread = state.credit_spread * own_credit_scale
            treasury_only = value_rider(
                aged_contract, curve, state.vol, reporting_basis,
                normals[:, :n_years], **valuation_state
            )
            with_spread = value_rider(
                aged_contract, curve, state.vol, reporting_basis,
                normals[:, :n_years], credit_spread=spread, **valuation_state
            )
            row["own_credit_spread"] = spread
            row["mrb_reporting_basis"] = treasury_only.net_value
            row["mrb_with_own_credit"] = with_spread.net_value
            row["own_credit_adjustment"] = with_spread.net_value - treasury_only.net_value

        short_rate = _short_rate(frame, date)
        row["short_rate"] = short_rate
        row["dt_years"] = dt
        row["price_return"] = price_return
        row["sub_account_return"] = sub_account_return
        row["me_charge"] = cash["me_charge"]
        row["rider_charge"] = cash["rider_charge"]
        row["claim"] = cash["claim"]

        if prev is not None:
            risk_free = prev["short_rate"]
            delta_value = greeks.base_value - prev["rider_value"]
            attributed_fees = fee_attribution * (cash["me_charge"] + cash["rider_charge"])
            financing = prev["rider_value"] * risk_free * dt
            liability_pnl = attributed_fees - cash["claim"] - delta_value + financing

            excess_return = price_return + (dividend_yield - risk_free) * dt
            rate_move_bp = (row["swap_rate"] - prev["swap_rate"]) * 100.0
            row.update(
                {
                    "excess_return": excess_return,
                    "rate_move_bp": rate_move_bp,
                    "vol_move": state.implied_vol_3m - prev["implied_vol_3m"],
                    "financing": financing,
                    "delta_rider_value": delta_value,
                    "liability_pnl": liability_pnl,
                    "equity_hedge_pnl": prev["equity_notional"] * excess_return,
                    "rate_hedge_pnl": -prev["swap_dv01"] * rate_move_bp,
                    "vega_hedge_pnl": prev["vega_notional"]
                    * (state.implied_vol_3m - prev["implied_vol_3m"]) * 100.0,
                }
            )

        # Resize the hedge on information available now. Nothing from the next period
        # enters this decision.
        target_equity = hedge_ratio * greeks.equity_exposure if hedge_equity else 0.0
        target_dv01 = -hedge_ratio * greeks.rho_per_bp if hedge_rates else 0.0
        target_vega = hedge_ratio * greeks.vega_per_point if hedge_vega else 0.0
        annuity = swap_annuity(curve, swap_tenor, int(cfg["curve"]["coupon_frequency"]))
        target_swap_notional = target_dv01 / (annuity * 1e-4) if annuity > 0 else 0.0

        # Costs are recorded per leg so effectiveness can be evaluated for any subset of
        # legs from a single run. Each leg is sized from its own Greek and none of the
        # sizings depend on whether the others are switched on.
        row["equity_cost"] = abs(target_equity - equity_notional) * equity_cost
        row["rate_cost"] = abs(target_swap_notional - swap_notional) * rate_cost
        row["vega_cost"] = abs(target_vega - vega_notional) * vega_cost
        row["transaction_cost"] = row["equity_cost"] + row["rate_cost"] + row["vega_cost"]
        row["equity_notional"] = target_equity
        row["swap_notional"] = target_swap_notional
        row["swap_dv01"] = target_dv01
        row["vega_notional"] = target_vega
        equity_notional, swap_notional, vega_notional = (
            target_equity,
            target_swap_notional,
            target_vega,
        )

        rows.append(row)
        prev = {
            "date": date,
            "spot": state.spot,
            "swap_rate": row["swap_rate"],
            "implied_vol_3m": state.implied_vol_3m,
            "rider_value": greeks.base_value,
            "equity_notional": target_equity,
            "swap_dv01": target_dv01,
            "vega_notional": target_vega,
            "short_rate": short_rate,
            "bond_yield": bond_yield_now,
        }

    ledger = pd.DataFrame(rows).set_index("date")
    ledger["hedged_pnl_gross"] = (
        ledger["liability_pnl"]
        + ledger["equity_hedge_pnl"]
        + ledger["rate_hedge_pnl"]
        + ledger["vega_hedge_pnl"]
    )
    ledger["hedged_pnl"] = ledger["hedged_pnl_gross"] - ledger["transaction_cost"]
    ledger.attrs["legs"] = {
        "equity": hedge_equity,
        "rates": hedge_rates,
        "vega": hedge_vega,
    }
    return ledger


LEG_COLUMNS = {
    "equity": ("equity_hedge_pnl", "equity_cost"),
    "rates": ("rate_hedge_pnl", "rate_cost"),
    "vega": ("vega_hedge_pnl", "vega_cost"),
}


def compose(ledger: pd.DataFrame, legs=("equity", "rates")) -> pd.DataFrame:
    """Rebuild hedged profit from a chosen subset of legs.

    A run with every leg computed contains everything needed to evaluate any subset,
    because each leg is sized from its own Greek independently of the others. Composing
    avoids running the backtest once per combination.
    """
    unknown = set(legs) - set(LEG_COLUMNS)
    if unknown:
        raise ValueError(f"unknown hedge legs {sorted(unknown)}")
    out = ledger.copy()
    gross = out["liability_pnl"].copy()
    costs = pd.Series(0.0, index=out.index)
    for leg in legs:
        pnl_column, cost_column = LEG_COLUMNS[leg]
        gross = gross + out[pnl_column]
        costs = costs + out[cost_column]
    out["hedged_pnl_gross"] = gross
    out["transaction_cost"] = costs
    out["hedged_pnl"] = gross - costs
    out.attrs["legs"] = tuple(legs)
    return out


def effectiveness(ledger: pd.DataFrame) -> dict[str, float]:
    """Hedge effectiveness over a ledger produced by ``run_backtest`` or ``compose``.

    The headline number is the variance ratio: the variance of hedged period profit
    over the variance of unhedged period profit. One minus that is the share of
    liability profit variance the hedge removed.
    """
    valid = ledger.dropna(subset=["liability_pnl"])
    unhedged = valid["liability_pnl"].to_numpy()
    hedged_gross = valid["hedged_pnl_gross"].to_numpy()
    hedged = valid["hedged_pnl"].to_numpy()
    costs = valid["transaction_cost"].to_numpy()

    var_unhedged = float(unhedged.var(ddof=1))
    out = {
        "periods": int(valid.shape[0]),
        "unhedged_std": float(unhedged.std(ddof=1)),
        "hedged_std_gross": float(hedged_gross.std(ddof=1)),
        "hedged_std": float(hedged.std(ddof=1)),
        "variance_ratio_gross": float(hedged_gross.var(ddof=1) / var_unhedged),
        "variance_ratio": float(hedged.var(ddof=1) / var_unhedged),
        "unhedged_worst": float(unhedged.min()),
        "hedged_worst": float(hedged.min()),
        "unhedged_total": float(unhedged.sum()),
        "hedged_total": float(hedged.sum()),
        "total_costs": float(costs.sum()),
        "unhedged_max_drawdown": _max_drawdown(unhedged),
        "hedged_max_drawdown": _max_drawdown(hedged),
    }
    out["variance_reduction"] = 1.0 - out["variance_ratio"]
    return out


def _max_drawdown(pnl: np.ndarray) -> float:
    """Largest peak-to-trough fall in cumulative profit."""
    cumulative = np.cumsum(pnl)
    running_peak = np.maximum.accumulate(cumulative)
    return float((cumulative - running_peak).min())


def residual_attribution(ledger: pd.DataFrame) -> pd.DataFrame:
    """Regress hedged profit on the risk factors the hedge does not cover.

    Squared index return stands in for gamma, the change in implied volatility for
    vega, and the gap between the long and short end for curve reshaping. Fitted with
    ordinary least squares on the normal equations, so there is no SciPy dependency.
    """
    valid = ledger.dropna(subset=["hedged_pnl_gross"]).copy()
    valid["gamma_proxy"] = valid["excess_return"] ** 2
    # A single-tenor swap hedges the level of rates. What leaks through is the change
    # in slope, taken here as the 30-year less the 2-year.
    slope = valid["par_30y"] - valid["par_2y"]
    valid["curve_twist_bp"] = slope.diff().fillna(0.0) * 100.0
    # What the contract holder's funds did that the index did not.
    valid["fund_basis"] = valid["sub_account_return"] - valid["excess_return"]
    columns = ["excess_return", "gamma_proxy", "vol_move", "curve_twist_bp", "fund_basis"]

    design = np.column_stack(
        [np.ones(valid.shape[0])] + [valid[c].to_numpy(dtype=float) for c in columns]
    )
    target = valid["hedged_pnl_gross"].to_numpy(dtype=float)
    coefficients, residuals, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    fitted = design @ coefficients
    total_ss = float(((target - target.mean()) ** 2).sum())
    resid_ss = float(((target - fitted) ** 2).sum())

    return pd.DataFrame(
        {
            "term": ["intercept"] + columns,
            "coefficient": coefficients,
        }
    ).assign(r_squared=1.0 - resid_ss / total_ss if total_ss > 0 else np.nan, rank=rank)


def _short_rate(frame: pd.DataFrame, date) -> float:
    """Financing rate for the futures position. The three-month bill is the closest
    free proxy for the rate embedded in index futures."""
    value = frame.loc[date, "DTB3"]
    if not np.isfinite(value):
        value = frame.loc[date, "DGS1"]
    return float(value) / 100.0
