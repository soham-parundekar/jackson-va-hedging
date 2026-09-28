# Limitations

## Scope

**One policy, not a book.** This is a stylised single-contract illustration. Valuing
Jackson's actual in-force block needs attained age, time since issue, actual benefit base
and actual sub-account allocation at policy level, none of which is ever disclosed. The
vintage portfolio in `scripts/run_portfolio_validation.py` widens it to five contracts so the
weighted attained age and blended withdrawal rate resemble the book, which is still five
contracts standing in for millions.

**Withdrawal phase only.** The portfolio leaves out contracts that have not started
withdrawals. Those accrue a bonus to the benefit base during deferral, and the prospectus
mechanics for it interact with the GWB adjustment provision in ways this model does not
implement. Modelling them without the bonus would understate the guarantee, so they are left
out instead, and their absence biases the portfolio towards more guarantee duration than the
book carries.

**No GMDB.** The Flex GMWB pays nothing on death. Enhanced death benefits are 10% of
account value on Jackson's book and are a separate election with a separate charge. Adding
one would raise the guarantee's value and shorten its duration.

**Nested simulation is avoided rather than solved.** A single policy at a single valuation
date needs one layer of simulation. An insurer computing Greeks on a full in-force block at
every future date along a stochastic path needs an inner valuation at every outer node,
which is why real programmes use least-squares Monte Carlo or replicating portfolios. Nothing
here addresses that problem.

## Behaviour assumptions

The base case draws the full guaranteed amount every year and never surrenders. Both are
deliberately the benchmark case from Bauer, Kling and Russ (2008) and both are upper bounds
on the guarantee's value. Real utilisation is below 100% and real lapse is positive, and
`docs/validation.md` shows the gap to the disclosure closes at 90% utilisation and 4% lapse.

The lapse implementation is static. Real lapse is dynamic: contract holders are much less
likely to surrender when a guarantee is deep in the money, which is exactly when surrender
would help the insurer most. A static rate therefore overstates the benefit of lapse. The
direction of that bias is against the insurer's interest, so the reconciled figures in
validation should be read as slightly optimistic about how much lapse does.

The guaranteed percentage is locked at the band for ages 70 to 74. Contracts may re-band
upward with attained age, to 5.95% at 75 and 6.20% at 81. Not modelling that understates the
guarantee.

## Market model

**No implied volatility beyond three months is available free**, and this is the most binding
data constraint in the project. The forward variance curve mean reverts on a decay pinned to
Jackson's own grading statement rather than fitted to market data, because there is no market
data to fit it to past three months. Everything about long-dated volatility rests on that
one assumption, and long-dated volatility is where most of the liability's vega sits: $569
per point on the long-run level against $333 on the tradeable front.

The curve cannot reproduce a steep one-month to three-month slope. It sits about two
volatility points above the 30-day VIX on average, with errors from -17 points in inverted
markets to +6 in normal ones.

**Geometric Brownian motion with a deterministic volatility term structure.** No stochastic
volatility, no jumps, no volatility skew. For a twenty-year guarantee the term structure of
variance matters more than the skew, which is the argument for the simplification, but it is
an argument rather than a proof. A Heston calibration would let the model price the skew and
would give a second volatility factor to hedge; `docs/methodology.md` sets out where it would
attach.

**One equity factor.** The sub-account is modelled as a single risky asset with the mix's
equity beta. The bond sleeve's own duration risk is not represented inside the valuation,
only in the realised path, which is what creates the fund basis term in the hedge residual.

**Single-factor rates.** Rho is a parallel par shift and the hedge is a single swap tenor.
Curve reshaping is left unhedged. There is no term structure model and no rate volatility, so
the interaction between rate volatility and the guarantee's value is absent.

**A constant 1.5% dividend yield.** The S&P 500 price index is what FRED carries, so the
dividend yield enters as an assumption in both the sub-account's total return and the futures
excess return.

## Data

The FRED S&P 500 series is a rolling ten-year window, which fixes the start of the backtest
at 26 September 2016. Ten years and one severe stress episode is a thin sample for a
statement about hedge effectiveness, and the 64.4% figure rests substantially on how
February and March 2020 happened to go.

The Item 7A market risk benefit figure covers several product lines while sensitivities are
scaled by variable annuity account value from Note 11. Variable annuities dominate, so the
imprecision is small, but the two presentations do not tie exactly and the reconciliation gap
at 31 December 2025 is $125m.

FRED revises series and the equity window rolls, so a fresh pull will not reproduce the
committed panel. The raw data is committed for that reason.

## Hedging backtest

**The liability is revalued with the same model that produced the hedge ratios.** This is the
most important caveat in the project. Any factor the model represents and the hedge covers is
removed nearly completely, limited only by convexity between rebalances. The volatility leg
reaching 96% is a statement about internal consistency, not about hedging a real liability.

What the backtest does capture: gamma between weekly rebalances, a single swap tenor against
a parallel-shift rho, one index against a blended sub-account, and transaction spreads.

What it does not: model error, policyholder behaviour differing from assumption, gap risk
between rebalances, liquidity, margin and collateral calls, counterparty risk, and the
possibility that the valuation model itself is recalibrated. On a forty-year guarantee these
are the larger risks. The reported variance reduction is an upper bound on what a real
programme achieves.

**The volatility overlay charges no carry.** A spread is charged on vega traded but nothing
for holding a long volatility position over time, and implied volatility exceeds realised on
average. Its variance reduction is informative and its cumulative profit is flattering.

**The equity hedge is a frictionless futures overlay.** No contract granularity, no roll
basis, no margin funding.

**A single policy path.** One realised history, one policy. No confidence interval on the
variance reduction, and none is claimed.

## Accounting comparison

Only two differences between the economic and reporting bases are measurable from public
data: the margin loading in the annuity table and the own-credit adjustment. The own-credit
spread is proxied at 0.6 times the Baa corporate spread, which is a judgement about where
Jackson's insurance subsidiaries sit rather than an observation of their debt.

Everything else that drives Jackson's reported volatility is outside reach. Assumption
updates, which Note 12 shows contributed $374m in 2025. The attributed-fee percentage frozen
at each contract's own inception across a book of many vintages. The Brooke Re captive
reinsurance structure and its modified GAAP approach for statutory purposes. The fact that
the hedging programme also targets statutory capital and distributable earnings, not just the
economic liability.

So the 1.05 ratio is a floor on the effect rather than a measurement of it, and the value of
the exercise is establishing that the mechanism is real and signed correctly.

## What this does not claim

Nothing here is a valuation of Jackson Financial or of its liabilities. It is not investment
advice. The disclosed figures are Jackson's; the model's figures are a single hypothetical
contract's, and the two are compared on shape and scale-free magnitude rather than treated as
estimates of each other. Where the model and the disclosure differ, the presumption
throughout is that the model is the thing that is wrong.
