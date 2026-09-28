"""Risk-neutral valuation and dynamic hedging of a representative GMWB rider.

The package is organised around the pieces of a hedging desk's problem. A discount curve and
a volatility term structure (``curves``, ``volatility``) and annuitant mortality
(``mortality``) are the inputs. ``contract`` holds the policy, ``engine`` prices it,
``sensitivities`` produces the risk numbers, ``hedging`` sizes and backtests the hedge, and
``accounting`` measures the wedge between the economic and the reported liability.

``market`` turns one date into model inputs, ``session`` assembles what every script needs,
and ``figures``, ``config`` and ``paths`` are plumbing.
"""

__all__ = [
    "accounting",
    "config",
    "contract",
    "curves",
    "engine",
    "figures",
    "hedging",
    "market",
    "mortality",
    "paths",
    "sensitivities",
    "session",
    "volatility",
]
