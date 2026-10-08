# Research design

## Question

Jackson Financial runs the largest standalone variable annuity book in the United States
and tells you a surprising amount about how it manages the risk in it. Item 1 of the 10-K
describes a core dynamic hedging programme that offsets equity and interest rate movements
in the *economic* liability associated with guaranteed living benefits. Item 7A gives the
sensitivity of the reported market risk benefit to a 10% equity move and a parallel shift
in risk-free rates. Note 6 describes the valuation method, down to the volatility term
structure and the treatment of the company's own credit. What none of it gives you is the
model.

So: can a risk-neutral Monte Carlo model of a representative GMWB rider, built from free
data and public filings, reproduce the sensitivities Jackson discloses, and what does
dynamic hedging actually do to the volatility of the position?

Two sub-questions follow from that, and they turn out to be where the interesting results
are.

How much of the liability's period-to-period volatility does a delta and rho hedge
remove, and what is left? Jackson's core programme is described in terms of equity and
interest rate exposure. It holds index futures, total return swaps and put options against
the equity leg, and swaps, swaptions and bond forwards against the rate leg.

And why does a hedge that works economically still leave reported earnings moving?
Jackson states in Item 7A that it does not hedge its U.S. GAAP liabilities and that this
has produced, and may continue to produce, net income volatility. That is a measurable
claim, not just a caveat.

## Hypotheses

1. A risk-neutral valuation of a single representative GMWB for Life rider reproduces the
   sign of every disclosed sensitivity, for equity moves in both directions and rate
   moves in both directions, across every year disclosed.
2. It reproduces the convexity in the disclosed table: the ratio of the 100bp impact to
   the 50bp impact is above two on the downside and below two on the upside, because the
   liability is convex in the discount rate.
3. It reproduces the direction and rough magnitude of the year-on-year decline in
   sensitivity per dollar of account value from 2022 to 2025, which is driven by a book
   moving out of the money as equity markets rose.
4. It does **not** reproduce the level. A static full-utilisation assumption is an upper
   bound on the guarantee's value, so the model's sensitivities should sit above the
   disclosed ones, and the gap should be closable by behaviour assumptions inside the
   ranges the literature supports.
5. Delta and rho hedging removes a substantial share of the liability's profit variance,
   and the residual is dominated by implied volatility rather than by gamma or curve
   reshaping.
6. Reported net income under the same hedge is more volatile than the economic outcome,
   because the reporting basis carries margins and an own-credit adjustment the economic
   basis does not.

Hypothesis 4 deserves a note. Predicting that a model will not match is unusual, and the
reason for stating it up front is that a single-policy model that *did* match the level of
a $236bn book would be a coincidence worth distrusting. What is testable is whether the
gap has a single structural explanation.

## What is being modelled

One contract, stated as such throughout: Jackson's Perspective II with the Flex GMWB
(Single) rider on the Core benefit option, elected at issue, single premium of $100,000,
issue age 70. Not Jackson's in-force block, which would need policyholder-level data that
is never disclosed.

Every choice in that specification comes from a filing rather than from convenience.

| Choice | Value | Where it comes from |
|---|---|---|
| Benefit type | GMWB for Life | 72% of variable annuity account value at 31 Dec 2025 carries GMWB for Life, 3% carries term GMWB (FY2025 10-K, Item 1) |
| Issue age | 70 | Weighted-average attained age of the book is 70 (Note 12) |
| Guaranteed withdrawal rate | By age band at the first withdrawal; 5.95% here | Flex GMWB Core rate sheet, read from the committed table rather than fixed (Rate Sheet Prospectus Supplement, 27 Apr 2026) |
| Deferral before the first withdrawal | 5 years | Not disclosed anywhere; the midpoint of the Core option's ten-year bonus period, priced at both ends in the robustness table |
| Rider charge | 1.25% of the benefit base | Same rate sheet, Core option elected at issue |
| Base contract charge | 1.31% of contract value | Perspective II prospectus fee table |
| Fund expenses | 0.95% | Inside the disclosed 0.52% to 2.38% range; tested at both ends |
| Step-up | Annual, to contract value | Prospectus, Contract Anniversary Value method |
| Bonus and GWB adjustment | Both modelled | 6% simple on the bonus base for up to ten deferred years; the adjustment on the later of the age-70 anniversary and the twelfth, and dead once a withdrawal is taken |
| Death benefit | Basic, in the same valuation | It sits on the same account as the living benefit and moves against it with longevity, so valuing one without the other reports a hedge target that does not exist |
| Sub-account mix | 72.4% equity, 8.3% bond, 18.3% balanced, 1.0% money market | FY2025 10-K, Note 11 |

The GMIB is deliberately absent. Jackson stopped offering it in 2009 and reinsures the
legacy block, so building around it would model a business Jackson has exited.

## Why the attributed-fee method matters

The disclosed liability is a market risk benefit, and under the attributed-fee method its
fair value is the present value of projected benefits *less* the present value of
attributed fees. Jackson fixes the attribution percentage at inception as a share of total
projected fees, caps it at 100%, and holds it static for the life of the contract.

Getting this right changes the exercise in three ways. It explains how a variable annuity
guarantee can sit on the balance sheet as a $4.2bn net asset. It means a model that prices
only the guarantee leg is not comparable to the disclosure at all, in sign or in
magnitude. And because the percentage is frozen at inception, a contract written when the
ten-year Treasury zero was 1.6% carries a different attribution than one written at 4.2%,
which is the mechanism that turns a decade of rising rates and rising markets into a
reported net asset.

The model calibrates the percentage the way the standard describes, and the at-issue result
is a check on the whole engine: with the calibrated percentage, the market risk benefit at
inception is zero, which is exactly what Note 6 says happens when projected attributed fees
cover projected claims.

## Method in outline

Account value under the risk-neutral measure, net of fees and withdrawals, with the benefit
base ratcheting annually. The contract recursion steps a policy year at a time, because every
path-dependent event in the contract lands on an anniversary; the market underneath it steps
twenty-four times a year, because a variance process has to be stepped to be simulated.

Equity volatility is Heston, calibrated to 1,468 quotes off the live SPX chain, simulated on the
quadratic-exponential scheme with a martingale correction because the calibrated parameters
violate the Feller condition. Rates are a one-factor Hull-White process built around a
Nelson-Siegel-Svensson fit to the bootstrapped Treasury curve, with its mean reversion and
volatility fitted to a long history that the ten-year panel cannot identify. Annuitant mortality
is the SOA 2012 IAM tables with Projection Scale G2, generational, which is the basis the NAIC
adopted for individual annuity valuation; the Basic table carries the economic valuation and the
Period table, which is the Basic table with the Life Actuarial Task Force's margins, carries the
reporting basis.

The book is a set of cohorts rather than one policy wherever the comparison is to a book: issue
age, duration and moneyness on a grid, each cohort carrying the attribution percentage its own
issue date calibrates to.

Greeks by bump and revalue on common random numbers, each reported with the standard error of its
own paired difference. Disclosed shocks by full repricing rather than by delta approximation,
because the shocks are large enough for convexity to show, and the convexity is what the shape
test is made of.

The hedge is daily along the realised path, sized by weighted least squares across delta, gamma,
vega and rho against a real instrument set - equity futures, total return swaps, listed index
puts, rate futures, bond forwards and interest rate swaps - with costs charged on the change in
position and swept at half, one and two times their base level. The liability is revalued through
a regression proxy, because a full valuation at every rebalance date is not affordable, and the
proxy's own error is measured against nested simulation before any hedging result is read.

The same ledger is then read a second way, as fee income against the funding, the trading cost and
the part of the guarantee the hedge did not recover, so the hedging result can be stated as a
margin on the benefit base rather than only as a variance removed. That comparison is against the
charge the valuation says the rider needed at issue, which is where the two halves of the project
meet.

Validation runs against four balance-sheet dates, 2022 through 2025, on sign, shape and scale
separately. `docs/validation.md` has the results and `docs/methodology.md` the equations.

## Period

Market data starts on 26 September 2016, which is the earliest S&P 500 observation FRED
carries, and runs to 25 September 2026. The window contains the February 2018 volatility
spike, the fourth quarter of 2018, February and March 2020, and 2022. Disclosed
sensitivities cover the four year-ends from 2022 to 2025, drawn from three consecutive
10-K filings so that the overlapping years can be cross-checked against each other.

## What would falsify this

A sign that comes out backwards on any disclosed shock. A convexity ratio on the wrong
side of two. A model sensitivity *below* the disclosed one, which would mean the static
behaviour assumption is not the binding difference and something else is wrong. A delta
and rho hedge that fails to reduce variance at all, or one whose residual is uncorrelated
with implied volatility. Reported net income no more volatile than the economic outcome.
