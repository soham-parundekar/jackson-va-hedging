# Validation

Results below come from a full run of the scripts in `scripts/` at 200,000 paths for the
valuation and Greeks and 40,000 for the backtest. Every table is written to
`reports/tables/` and every figure to `reports/figures/`.

## Numerical checks

These test the machinery rather than the economics. If any of them fails nothing downstream
is worth reading.

| Check | Result |
|---|---|
| Par bonds reprice to par off the bootstrapped zeros, all 2,514 curves in the sample | worst error 2e-16 |
| Discounted expected contract value with no charges or withdrawals returns the premium | within 0.3% at horizons 1, 5, 10, 20 and 39 years |
| Closed-form collection of continuous charges against 20,000-step brute force, 200 random paths | agrees to 2e-4 relative |
| Total variance monotone across implied levels 5% to 100% and long-run levels 5% to 45% | no violations on a 0.02-year grid to 60 years |
| Closed-form total variance against numerical quadrature | agrees to 1e-8 relative |
| Volatility curve reproduces its fitting point, 503 dates | exact |
| Period mortality table implies longer life than Basic at every duration | holds |
| Sex blend is exactly the average of the two survival curves | holds; the rate-blended alternative differs |
| Disclosed figures appearing in two consecutive filings | 6 of 6 agree |
| Hedge profit rebuilt from the position recorded on the previous date | matches to 1e-12 |

The martingale check is the one that matters most. With charges and withdrawals switched
off, the contract value is a traded asset, so its discounted expectation has to come back to
where it started. That is what pins the risk-neutral drift and the centring of the lognormal
step, and it cannot pass by accident.

### Monte Carlo error and truncation

| Paths | Market risk benefit | Standard error |
|---|---|---|
| 5,000 | 138.92 | 105.57 |
| 20,000 | 53.84 | 54.18 |
| 50,000 | -26.72 | 33.97 |
| 100,000 | -28.62 | 23.93 |
| 200,000 | 0.00 | 16.90 |

The error halves as paths quadruple, as it should. Truncating the projection at attained age
110 instead of 115 moves the market risk benefit by $1.62 and extending it to 120 moves it by
$0.53, both an order of magnitude inside the $16.90 standard error.

## The at-issue result, which is itself a check

At 31 December 2025, on a $100,000 single premium at issue age 70:

| | |
|---|---|
| Present value of guarantee payments | 19,303 |
| Present value of the explicit GMWB charge | 13,277 |
| Present value of the base contract charge | 8,912 |
| Total attributable fees | 22,189 |
| Attribution percentage | 0.8699 |
| **Market risk benefit at issue** | **0.00** |
| Monte Carlo standard error | 16.90 |
| Probability the contract value is exhausted by year 20 | 86.1% |
| Expected future lifetime | 19.6 years |

Note 6 says that where projected attributed fees are sufficient to offset projected
guaranteed benefits at issue, the market risk benefit has an initial fair value of zero.
The model produces that, and it is not a free result: it requires the claims leg to come out
below the total fee leg but not far below. Attributable fees cover claims 1.15 times over,
so the attribution calibrates to 87% and the benefit starts at zero. Had the model
overstated claims by 20% the attribution would have capped at 100% and the benefit would have
opened as a liability.

Claims at 19.3% of premium on a risk-neutral basis look large until you see the cash-flow
profile. With a 5.75% lifetime draw against a risk-neutral drift of roughly 3.7% less 2.26%
of charges, the account is being drawn down faster than it grows on nearly every path.
First claims appear around year five, and by year twenty the account is gone on 86% of paths.
The guarantee is not a tail event in this contract; it is the expected outcome, and what the
valuation is pricing is when it happens rather than whether.

## Sign

Eighteen comparisons: four balance-sheet dates, equity up and down, rates up and down, at
whichever shock sizes each filing disclosed.

**All eighteen signs agree**, both for a single policy rolled from a 2016 inception and for
the vintage portfolio. Equity falls, the guarantee becomes more expensive. Rates rise, the
liability shrinks. Neither was put in by hand; both come out of the cash-flow logic.

## Shape

Two features of the disclosed table are testable without matching any level.

**Convexity in the rate shock.** The FY2024 and FY2025 filings between them disclose both
±50bp and ±100bp at 31 December 2024, which gives a ratio the model can be held to.

| | Disclosed | Model, benefit base over account value 0.8 / 1.0 / 1.2 |
|---|---|---|
| 100bp up over 50bp up | 1.855 | 1.894 / 1.895 / 1.902 |
| 100bp down over 50bp down | 2.128 | 2.112 / 2.111 / 2.105 |

Within 2% on both, at every moneyness tested, on a quantity the model was never fitted to.
The asymmetry is the signature of a convex liability: doubling the shock more than doubles
the downside impact and less than doubles the upside one.

**The trend across years.** Jackson's book moved out of the money as equity markets rose, and
the disclosed sensitivity per dollar of account value fell with it.

| As of | Disclosed equity -10%, % of account value | Vintage portfolio |
|---|---|---|
| 2022-12-31 | 1.466 | 3.847 |
| 2023-12-31 | 1.179 | 3.360 |
| 2024-12-31 | 0.956 | 2.461 |
| 2025-12-31 | 0.849 | 2.280 |

Both decline monotonically. Disclosed falls to 0.58 of its 2022 level over the period and the
model to 0.59.

## Scale

Here the model does not match, which hypothesis 4 predicted, and the useful part is the
structure of the miss.

A single policy cannot be made to match, and the reason is instructive: matching the book's
moneyness and matching its weighted-average attained age of 70 pull in opposite directions.
A contract issued in 2016 has ridden the market up and sits near the money, which is right,
but it is also nine years older than the book average, which shortens the guarantee and
understates its duration. At 31 December 2025 that single in-force policy gives -1.49% of
account value per 100bp against a disclosed -1.26%, which looks like a good match and is
partly coincidence.

Five vintages fix the age problem. Issue dates from 2016 to 2024 with issue ages chosen so
attained ages straddle 70, which also gives a blended guaranteed withdrawal rate of 4.89%
rather than the 5.75% a new 70-year-old gets, because the rate sheet bands the percentage by
age.

| As of | Weighted attained age | Blended withdrawal rate | Benefit base over account value |
|---|---|---|---|
| 2022-12-30 | 67.4 | 4.71% | 1.171 |
| 2023-12-29 | 68.4 | 4.71% | 1.054 |
| 2024-12-31 | 69.9 | 4.90% | 0.974 |
| 2025-12-31 | 70.9 | 4.89% | 0.948 |

All eighteen ratios of model to disclosed then fall between **2.14 and 2.85, median 2.39**.
That near-constant multiple across six different shocks and four dates is the interesting
result. A pile of unrelated errors would not produce it. One structural assumption would.

### What closes the gap

Two things scale every sensitivity down without touching the model's shape.

About a quarter of Jackson's variable annuity account value carries no withdrawal
guarantee at all. GMWB for Life is 72% of account value and term GMWB another 3%, and Elite
Access and unrestricted Perspective contracts carry no living benefit. Scaling by 0.75 for
that alone takes the mean ratio from 2.39 to 1.79.

The rest is policyholder behaviour. The base case draws the full guaranteed amount every
year and never surrenders, which Bauer, Kling and Russ (2008) use as their benchmark
precisely because it is the most expensive assumption for the insurer. Jackson's fair value
is built on assumed "benefit utilization by policyholders, lapse, mortality, and withdrawal
rates". Sweeping both:

| Utilisation | Lapse | Mean ratio to disclosed | Worst single shock |
|---|---|---|---|
| 1.00 | 0.00 | 1.79 | 2.01 |
| 1.00 | 0.04 | 1.16 | 1.39 |
| 0.90 | 0.02 | 1.21 | 1.43 |
| 0.90 | 0.04 | 0.94 | within 33% |
| 0.80 | 0.02 | 0.96 | within 26% |

At 90% utilisation and a 4% annual surrender rate every one of the four shocks sits within
33% of the disclosed figure. Variable annuity lapse rates in the low single digits and
utilisation below 100% are both ordinary. So the gap closes inside the ranges the literature
and the filings support, which is what hypothesis 4 asked.

This is a reconciliation, not a calibration. The base case stays at full utilisation and no
lapse, because that is the assumption the academic benchmark uses and because fitting
behaviour parameters to a disclosed number would be reverse-engineering the answer.

Two biases run the other way and are worth holding in mind. A static lapse rate overstates
surrender in exactly the states where the guarantee is valuable, since real lapse falls when
a guarantee is deep in the money. And leaving deferral-phase contracts out of the portfolio
biases it towards more guarantee duration than the book has, because the bonus mechanics
those contracts carry are not modelled.

### The withdrawal rate is the lever

Independently of the behaviour sweep, varying the guaranteed withdrawal percentage shows the
same thing:

| Withdrawal rate | Claims, % of account value | Exhausted by year 20 | Rates +100bp, % of account value |
|---|---|---|---|
| 3.00% | 4.12 | 45.6% | -1.34 |
| 4.00% | 8.41 | 64.4% | -2.41 |
| 5.00% | 14.17 | 78.6% | -3.65 |
| 5.75% | 19.30 | 86.1% | -4.63 |
| 6.50% | 25.00 | 91.3% | -5.61 |

The disclosed -1.26% sits near the 3% row. A contract drawing 5.75% a year exhausts on
nearly every path, which turns the guarantee into a long-dated life annuity and gives it the
duration to match. A book where many contracts draw less, or have not started drawing, has
far less of it.

## Hedging

One policy, issued 26 September 2016 on the disclosed fund mix, hedged weekly for 521 weeks
to September 2026.

| Hedge | Weekly std | Variance ratio | Variance removed | Worst week | Costs over 10 years |
|---|---|---|---|---|---|
| Unhedged | 1,705 | 1.000 | 0.0% | -11,500 | 0 |
| Delta | 1,253 | 0.540 | 46.0% | -8,438 | 32 |
| Delta and rho | 1,018 | 0.356 | **64.4%** | -8,258 | 48 |
| Delta, rho and volatility | 328 | 0.037 | 96.3% | -1,061 | 162 |

Annualised, the unhedged guarantee has a profit standard deviation of $12,305 on a $100,000
policy. Delta and rho hedging takes it to $7,346.

Transaction costs are almost an afterthought: $48 across ten years of weekly rebalancing,
under 0.05% of premium. Index futures and interest rate swaps are cheap, and the hedge
notional moves slowly enough that little is traded. On this evidence the basis-risk against
transaction-cost trade-off that sets rebalance frequency is not much of a trade-off at
weekly frequency for a liability of this duration.

### What is left, and where

The residual after delta and rho regresses on the factors the hedge does not cover with an
R² of 0.85. The two large coefficients are on squared index return, standing in for gamma,
and on the change in implied volatility. The fund basis term also loads, which is the
disclosed sub-account mix behaving differently from the index the hedge trades.

Stress windows make the point more plainly than the regression does.

| Window | Weeks | Index return | Unhedged | Delta and rho | Plus volatility |
|---|---|---|---|---|---|
| Feb 2018 volatility spike | 5 | -4.4% | -346 | -1,808 | -846 |
| Q4 2018 | 14 | -14.7% | -7,807 | -1,530 | +1,434 |
| Feb-Mar 2020 | 7 | -24.8% | -30,563 | -18,470 | -1,457 |
| 2022 | 52 | -17.9% | +4,669 | -4,042 | -2,548 |

In the seven weeks to the end of March 2020 the unhedged guarantee lost 30.6% of premium.
Delta and rho hedging recovered 40% of that and left an 18.5% loss standing. Adding the
volatility leg cut the residual to 1.5%. February and March 2020 was a volatility event for
this liability more than a level event, and a programme built on futures and swaps alone was
never going to catch it.

In February 2018 the delta and rho hedge *lost more* than doing nothing. The index barely
moved over those five weeks while implied volatility doubled, so the hedge paid its costs and
had nothing to offset. A hedge that never underperforms in any window is a hedge that has
been fitted.

### The step-up can make the insurer long equity

One result came out of a test that was asserting the wrong thing. The test required equity
exposure to be negative on every date, on the reasoning that a written guarantee behaves like
a short put. It failed on 17 of 522 weeks, and the failures were not noise.

Under the Core option the benefit base steps up to the contract value on the anniversary. When
the contract value sits above the benefit base, the index level on that one date fixes the
guaranteed income for the rest of the contract's life. Approaching it, a higher index means a
permanently larger guarantee to fund, and the claims leg becomes long equity.

Decomposing at the state where it first showed up, a contract at attained age 74 with an
account value 23% above its benefit base and the anniversary 2.6 months away:

| | per unit log index return |
|---|---|
| Change in PV of claims | +25,234 |
| Change in PV of fees | +23,595 |
| Net equity exposure | **+1,638** |

Both legs respond positively. Claims rise because the ratchet is about to lock in a higher
base; fees rise because the account value and the future benefit base are both larger. The
net is a small residual of two large numbers, which is why the sign is delicate.

The rate environment decides it. At the July 2021 market, with the ten-year zero at 1.39%, the
account depletes on nearly every path, so the ratchet is certain to bite and the claims
derivative reaches +26,663 at an account value 25% above the benefit base. At the December 2025
market, with the ten-year zero at 4.20%, the higher risk-neutral drift keeps some paths solvent
and the claims derivative only reaches +12,652, which is not enough to flip the net.

| Market | Ten-year zero | Account over benefit base | Years to anniversary | Claims leg | Fees leg | Net exposure |
|---|---|---|---|---|---|---|
| Jul 2021 | 1.39% | 1.00 | 0.30 | -10,835 | +19,120 | -29,954 |
| Jul 2021 | 1.39% | 1.25 | 1.00 | +17,201 | +18,869 | -1,669 |
| Jul 2021 | 1.39% | 1.25 | 0.30 | +26,663 | +18,562 | **+8,101** |
| Dec 2025 | 4.20% | 1.25 | 0.30 | +12,652 | +18,617 | -5,965 |

Across the backtest ledger the pattern is entirely systematic. Of 522 weeks, 17 carry long
exposure, every one of them with the contract value above the benefit base and the anniversary
within 0.25 years. All 232 weeks where the benefit base is at or above the contract value are
short without exception. Holding the state fixed and varying only the time to the reset,
exposure rises monotonically as it approaches, and it is uniformly lower at a 6% curve than at
a 2% one. The tests assert those properties rather than the blanket claim that failed.

Two practical consequences. A hedge that follows the model turns from short index to long for
a few weeks before a step-up date, which is real turnover a real desk would face. And because
a book with anniversaries spread across the calendar averages this away while a single policy
concentrates it, single-policy hedge statistics overstate the rebalancing a real programme
does.

A related property fell out of the same investigation: in states where the ratchet binds on
every path, equity gamma comes out at exactly zero. That is not a numerical accident. The
rider value is homogeneous of degree one in the account value and the benefit base together,
and when a certain reset makes a shock scale both, the value is exactly linear in the shock.

### A caution the headline number needs

The liability is revalued with the same model that produced the hedge ratios. Any risk factor
the model represents and the hedge covers gets removed nearly completely, limited only by
convexity between rebalances. That is why the volatility leg reaches 96%: it is hedging a
model-implied vega against a model-implied revaluation.

So what these figures measure is the cost of hedging discretely with imperfect instruments:
gamma between weekly rebalances, a single swap tenor against a parallel-shift rho, one index
against a blended sub-account, and the spread paid to trade. What they do not measure is
model error, and on a forty-year guarantee with assumed policyholder behaviour that is the
larger risk. The 64.4% should be read as an upper bound on what a real programme achieves,
not an estimate of it.

## Economic against reported

Same hedge positions, sized on the economic basis, measured two ways.

| Measure | Weekly std | Annualised | Variance vs unhedged |
|---|---|---|---|
| Unhedged guarantee | 1,706 | 12,305 | 1.000 |
| Economic, hedged | 1,019 | 7,346 | 0.356 |
| Reported net income | 1,070 | 7,716 | 0.393 |
| Reported comprehensive income | 808 | 5,828 | 0.224 |
| The OCI piece alone | 440 | 3,175 | 0.067 |

The hedge removes 64.4% of the variance of the economic liability and 60.7% of the variance
that reaches net income. Reported net income is 1.05 times as volatile as the economic
outcome under identical positions, and the ratio sits between 1.01 and 1.12 in every calendar
year of the sample. The margin loading in the reporting basis adds $1,614 to the liability on
average and changes its sensitivities enough that a hedge sized on the economic delta is
slightly the wrong size.

The more interesting number is the third row. Reported comprehensive income is *less*
volatile than either the economic outcome or net income, and the reason is the own-credit
adjustment. Credit spreads widen when equity markets fall, which reduces the own-credit
adjusted liability at the same moment the market move is increasing it. The adjustment is a
partial natural hedge, averaging -$6,610 and reaching -$20,842 at its widest. But the market
risk benefit rules report that movement in other comprehensive income, so it never reaches
net income, and net income gets the full undamped move while comprehensive income gets the
damped one. An accounting boundary turns a natural offset into a reporting mismatch.

The OCI piece on its own carries an annualised standard deviation of $3,175, a quarter of the
unhedged volatility, and no hedge in this programme touches it.

A 5% excess in net income volatility is smaller than Jackson's own disclosure would lead you
to expect, and the honest reading is that the two differences measurable from public data,
margins and own credit, are not the main drivers of its reported volatility. Assumption
updates, the attributed-fee percentage frozen at inception across a book of many vintages,
and the fact that Jackson's hedging also targets statutory capital and distributable
earnings are all outside what a single-policy model built from filings can reach. What the
exercise does establish is that the direction is right and the mechanism is real.

## Falsification tests, restated against results

| Would have falsified | Result |
|---|---|
| A sign backwards on any disclosed shock | 18 of 18 agree |
| A convexity ratio on the wrong side of two | 1.89 up, 2.11 down, against 1.86 and 2.13 disclosed |
| Model sensitivity below the disclosed one | above on all 18, ratio 2.14 to 2.85 |
| A delta and rho hedge that fails to reduce variance | 64.4% of variance removed over 521 weeks |
| A residual uncorrelated with implied volatility | volatility is the dominant residual term |
| Reported net income no more volatile than the economic outcome | 1.05 times, positive in all 11 calendar years |
