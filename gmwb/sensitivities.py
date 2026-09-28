"""Greeks by bump and revalue, and repricing under Jackson's own disclosed shocks.

Every bump reuses the same normals as the base valuation. Without that, the
difference between two valuations is dominated by simulation noise rather than by
the sensitivity: at 100,000 paths the standard error on the level is a few tens of
dollars on a $100,000 policy, while a one-basis-point rate bump moves the value by
single dollars. Common random numbers turn the level noise into a shared offset that
cancels in the difference.

Sign conventions, since they decide whether the hedge is long or short. The rider
value returned by the engine is the insurer's liability, positive when the insurer
owes. Equity exposure is the derivative with respect to the log index level, which
is the dollar notional to trade, and it is negative: a higher index makes the
guarantee less likely to bite. Rho is quoted per basis point of parallel par shift
and is also negative, because a long-dated liability discounts away faster when
rates rise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contract import GmwbContract
from .curves import ParCurveBuilder
from .engine import RiderValuation, value_rider
from .mortality import MortalityBasis
from .volatility import VolTermStructure


@dataclass(frozen=True)
class Greeks:
    base_value: float
    account_value: float
    equity_exposure: float   # dV / d(ln S), dollars; the notional a hedge must carry
    equity_gamma: float      # d2V / d(ln S)^2, dollars
    rho_per_bp: float        # dV / d(parallel par shift), dollars per basis point
    vega_per_point: float    # dV / d(implied level), dollars per 1 vol point: tradeable
    vega_long_run: float     # dV / d(long-run level), dollars per 1 vol point: assumption
    std_error: float

    @property
    def equity_exposure_pct_of_av(self) -> float:
        return self.equity_exposure / self.account_value


def _revalue(
    contract, curve, vol, mortality, normals, *, state, **overrides
) -> RiderValuation:
    kwargs = dict(state)
    kwargs.update(overrides)
    return value_rider(contract, curve, vol, mortality, normals, **kwargs)


def compute_greeks(
    contract: GmwbContract,
    curve_builder: ParCurveBuilder,
    vol: VolTermStructure,
    mortality: MortalityBasis,
    normals: np.ndarray,
    state: dict,
    equity_bump: float = 0.01,
    rate_bump_bp: float = 1.0,
    vol_bump: float = 0.01,
) -> Greeks:
    """Central differences in the equity level, a parallel par shift, and volatility.

    ``state`` carries the arguments that describe the policy and the discounting
    basis (account value, benefit base, attribution percentage, credit spread,
    max_age) and is passed through unchanged to every revaluation.
    """
    base_curve = curve_builder.build()
    base = _revalue(contract, base_curve, vol, mortality, normals, state=state)

    up = _revalue(contract, base_curve, vol, mortality, normals, state=state,
                  equity_shock=equity_bump)
    down = _revalue(contract, base_curve, vol, mortality, normals, state=state,
                    equity_shock=-equity_bump)

    # Bumping the index by a proportional amount h gives a central difference in
    # log-space of ln((1+h)/(1-h)), not 2h. At h = 1% the difference is 0.01%, but
    # using the exact denominator costs nothing.
    log_span = np.log((1.0 + equity_bump) / (1.0 - equity_bump))
    equity_exposure = (up.net_value - down.net_value) / log_span
    gamma_h = np.log(1.0 + equity_bump)
    equity_gamma = (up.net_value - 2.0 * base.net_value + down.net_value) / gamma_h**2

    curve_up = curve_builder.build(shift_bp=rate_bump_bp)
    curve_down = curve_builder.build(shift_bp=-rate_bump_bp)
    rate_up = _revalue(contract, curve_up, vol, mortality, normals, state=state)
    rate_down = _revalue(contract, curve_down, vol, mortality, normals, state=state)
    rho_per_bp = (rate_up.net_value - rate_down.net_value) / (2.0 * rate_bump_bp)

    # Two vegas. The first moves only the short-dated level, which is what trades and
    # what moves week to week; because the curve mean reverts, that shift decays with
    # maturity the way a real surface does. The second moves the long-run level, which is
    # an assumption, and is reported to show how much volatility exposure sits beyond any
    # listed option maturity.
    implied_up = _revalue(contract, base_curve, vol.shift_front(vol_bump), mortality,
                          normals, state=state)
    implied_down = _revalue(contract, base_curve, vol.shift_front(-vol_bump), mortality,
                            normals, state=state)
    vega = (implied_up.net_value - implied_down.net_value) / (2.0 * vol_bump) * 0.01

    long_run_up = _revalue(contract, base_curve, vol.shift(vol_bump), mortality,
                           normals, state=state)
    vega_long_run = (long_run_up.net_value - implied_up.net_value) / vol_bump * 0.01

    return Greeks(
        base_value=base.net_value,
        account_value=base.account_value,
        equity_exposure=equity_exposure,
        equity_gamma=equity_gamma,
        rho_per_bp=rho_per_bp,
        vega_per_point=vega,
        vega_long_run=vega_long_run,
        std_error=base.std_error,
    )


def disclosed_shock_repricing(
    contract: GmwbContract,
    curve_builder: ParCurveBuilder,
    vol: VolTermStructure,
    mortality: MortalityBasis,
    normals: np.ndarray,
    state: dict,
    equity_shocks=(0.10, -0.10),
    rate_shocks_bp=(100, -100, 50, -50),
) -> dict[str, float]:
    """Reprice under the shocks Jackson runs in Item 7A and return the value changes.

    These are full repricings, not delta approximations. The disclosed shocks are
    large enough that convexity shows up in them, and the asymmetry between the up
    and down cases is one of the features worth comparing against the filing.
    """
    base_curve = curve_builder.build()
    base = _revalue(contract, base_curve, vol, mortality, normals, state=state)
    out = {"base_value": base.net_value, "account_value": base.account_value}

    for shock in equity_shocks:
        shocked = _revalue(contract, base_curve, vol, mortality, normals, state=state,
                           equity_shock=shock)
        label = f"equity_{'up' if shock > 0 else 'down'}_{abs(int(round(shock * 100)))}pct"
        out[label] = shocked.net_value - base.net_value

    for bp in rate_shocks_bp:
        shocked = _revalue(contract, curve_builder.build(shift_bp=bp), vol, mortality,
                           normals, state=state)
        label = f"rates_{'up' if bp > 0 else 'down'}_{abs(int(bp))}bp"
        out[label] = shocked.net_value - base.net_value

    return out


def moneyness_profile(
    contract: GmwbContract,
    curve_builder: ParCurveBuilder,
    vol: VolTermStructure,
    mortality: MortalityBasis,
    normals: np.ndarray,
    state: dict,
    ratios=(0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0),
) -> list[dict[str, float]]:
    """Value and shock sensitivities across benefit-base-to-account-value ratios.

    A single policy at issue sits at a ratio of one. Jackson's in-force book does
    not: after a decade of rising equity markets most contracts have an account value
    well above their benefit base. Reporting the profile rather than one point is what
    lets the disclosed sensitivity be located on the model's own curve.
    """
    rows = []
    account_value = state.get("account_value") or contract.premium
    for ratio in ratios:
        scoped = dict(state)
        scoped["account_value"] = account_value
        scoped["benefit_base"] = account_value * ratio
        result = disclosed_shock_repricing(
            contract, curve_builder, vol, mortality, normals, scoped
        )
        result["gwb_over_av"] = ratio
        rows.append(result)
    return rows
