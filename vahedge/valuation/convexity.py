"""The guarantee's curvature, taken from nested valuations because the regression cannot give it.

The regression proxy estimates the liability's level to within half a per cent of premium and
its slope to within a few per cent of itself. Its second derivative is not a risk number at any
horizon: against nested bumps the error is about a hundred per cent of the quantity, and three
different attempts to fix that - a roughness penalty chosen by cross-validation, the same
penalty chosen at the held-out minimum, and reporting an average curvature across a ten per cent
move instead of a point derivative - all failed in different directions. The reason is not a bad
basis. A regression on single-path realisations estimates a conditional mean well, its gradient
reasonably, and its curvature not at all, because each additional derivative costs another
factor of the payoff noise.

So the convexity comes from the expensive calculation instead, and it is affordable for the one
reason that matters: a gamma needs three valuations of the same contract in the same market, and
under Heston the return distribution does not depend on the index level, so all three share one
simulation. Two hundred nodes is a few minutes, once, cached to disk and reused by every
strategy and every scenario.

What the surface holds, and what it deliberately does not. Gamma is homogeneous of degree one in
the contract value and the benefit base together, so what is tabulated is gamma per unit of
benefit base against moneyness, one curve per policy year. It is not tabulated against
volatility or the rate. Those do move it, and leaving them out is the approximation here - stated
rather than hidden, and bounded by the spread the build script reports, which splits the nodes
inside a narrow band of moneyness at their median variance and compares the two halves. The
curvature near the money differs by five to twenty per cent of itself between a quiet node and a
busy one at the same moneyness, which is the size of what the indexing throws away. The reason
for the simplification is
that the convexity leg is sized with a weight of a tenth in the hedge solve, so an error in gamma
costs a fraction of itself in the put position, while a surface over three state variables costs
a hundred times the compute and would have to be rebuilt for every market state the backtest
visits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConvexitySurface:
    """Gamma per unit of benefit base, by policy year and moneyness.

    Evaluated by linear interpolation in moneyness within a year and linearly in time between
    years, which is the same bracketing the proxy's own value uses so the two stay in step.
    Beyond the tabulated moneyness range the end value is held rather than extrapolated: the
    curvature of a guarantee goes to zero well out of the money and to the annuity's flat line
    well in, and a linear extrapolation of a bell-shaped curve would send it negative.
    """

    table: pd.DataFrame          # columns: policy_year, moneyness, gamma_per_unit
    equity_bump: float
    inner_paths: int

    @property
    def years(self) -> np.ndarray:
        return np.sort(self.table["policy_year"].unique())

    def _year_curve(self, year: float) -> tuple:
        block = self.table[self.table["policy_year"] == year].sort_values("moneyness")
        return block["moneyness"].to_numpy(), block["gamma_per_unit"].to_numpy()

    def per_unit(self, policy_year, moneyness) -> np.ndarray:
        """Gamma per unit of benefit base at a policy age and moneyness."""
        policy_year = np.atleast_1d(np.asarray(policy_year, dtype=float))
        moneyness = np.broadcast_to(
            np.atleast_1d(np.asarray(moneyness, dtype=float)), policy_year.shape
        )
        grid = self.years
        lower = np.clip(np.searchsorted(grid, policy_year, side="right") - 1, 0, grid.size - 1)
        upper = np.clip(lower + 1, 0, grid.size - 1)
        span = np.where(grid[upper] > grid[lower], grid[upper] - grid[lower], 1.0)
        weight = np.clip((policy_year - grid[lower]) / span, 0.0, 1.0)

        out = np.empty(policy_year.size)
        for index in range(policy_year.size):
            first_x, first_y = self._year_curve(grid[lower[index]])
            second_x, second_y = self._year_curve(grid[upper[index]])
            first = float(np.interp(moneyness[index], first_x, first_y))
            second = float(np.interp(moneyness[index], second_x, second_y))
            out[index] = (1.0 - weight[index]) * first + weight[index] * second
        return out

    def gamma(self, policy_year, account_value, benefit_base) -> np.ndarray:
        base = np.atleast_1d(np.asarray(benefit_base, dtype=float))
        account = np.atleast_1d(np.asarray(account_value, dtype=float))
        return base * self.per_unit(policy_year, account / np.maximum(base, 1e-12))

    def to_csv(self, path) -> None:
        frame = self.table.copy()
        frame.attrs = {}
        frame["equity_bump"] = self.equity_bump
        frame["inner_paths"] = self.inner_paths
        frame.to_csv(path, index=False)

    @classmethod
    def from_csv(cls, path) -> "ConvexitySurface":
        frame = pd.read_csv(path)
        return cls(
            table=frame[["policy_year", "moneyness", "gamma_per_unit"]],
            equity_bump=float(frame["equity_bump"].iloc[0]),
            inner_paths=int(frame["inner_paths"].iloc[0]),
        )


def with_nested_gamma(source, surface: ConvexitySurface):
    """Greeks from ``source`` with its own curvature replaced by the surface's.

    ``source`` is either a fitted proxy or any callable with the simulator's Greek signature,
    which is what lets the misspecification experiment stack the two: its hedges are sized from a
    deliberately wrong model, and the convexity leg of those still has to come from somewhere
    other than a regression second derivative. Everything but the gamma is left alone. The
    regression's level and slope are accurate enough to hedge on and nested valuation is far too
    slow to supply them at every rebalance.
    """
    base = getattr(source, "greeks_at", source)

    def greeks_at(**kwargs):
        out = dict(base(**kwargs))
        out["gamma"] = surface.gamma(
            kwargs["years_since_issue"], kwargs["account_value"], kwargs["benefit_base"]
        )
        return out

    return greeks_at
