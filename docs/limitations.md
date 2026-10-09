# Limitations

Organised by what the limitation threatens. Everything here is either measured somewhere in
`reports/tables/` or stated as an assumption with its direction of bias; nothing is listed as a
caveat in order to be dismissed.

## The book

**Five contracts standing in for millions.** Valuing Jackson's actual block needs attained age,
duration, benefit base and sub-account allocation at policy level, none of which is disclosed.
The vintage portfolio in `scripts/run_portfolio_validation.py` reaches a weighted attained age of
70.9 against the disclosed 70 and a blended withdrawal rate of 5.73%, and two of its five vintages
are still in deferral at the last disclosed date. It is still five model points.

**A quarter of the account value has no living benefit at all.** GMWB for Life is 72% of Jackson's
variable annuity account value and plain GMWB another 3%, so the disclosed sensitivity is already
divided by a denominator that includes contracts with nothing to be sensitive about. The behaviour
reconciliation scales for that explicitly, which is arithmetic rather than an assumption, but the
remaining quarter is not modelled at all.

**The deferral assumption is not identified by anything.** Five years from issue to the first
withdrawal is the most consequential single choice in the project after the long-run volatility: on
the Core option it accrues 30% of bonus on the benefit base and moves the guaranteed rate from
5.75% to 5.95%, so it is about a third more guaranteed income. Nothing in the disclosure says how
long the book defers. Both ends of the bonus period are priced in the robustness table and the
direction of the bias is plain - a contract that defers and then draws in full is an upper bound.

## Behaviour

**The base case draws the full guaranteed amount every year.** That is the benchmark case in the
literature precisely because it is the most expensive one for the insurer, and the behaviour
reconciliation sweeps it rather than defending it.

What that sweep found is itself a limitation and the honest version of a result this project would
have preferred: **no single behaviour assumption closes the gap to the disclosure.** Across the
grid the equity multiple falls 2.10 to 1.59 while the rate multiple falls 2.09 to 0.47, because
drawing less takes duration out of the guarantee and the rate sensitivity goes with it while the
benefit base is still there whatever the owner draws. By the time utilisation is low enough to
match the rate figure the model is still 2.03 times the disclosed equity figure. The residual is
attributed to moneyness on the evidence of the shock locator, which puts the disclosed book near
0.85 of benefit base to account value against the portfolio's 0.985, but that attribution is an
inference rather than a measurement.

**Lapse is dynamic but its parameters are assumed.** The damping exponent and the floor are stated
numbers, not fitted ones, because no free data identifies how a real book's surrender rate responds
to moneyness. The direction is known - a static rate would overstate how much lapse helps the
insurer - and the sweep quotes the rate at the money so the effective rate is lower than the label.

**The roll keeps benchmark behaviour.** Each vintage's account value today comes from a roll that
draws in full, because what a contract is worth now is a fact about the past rather than an
assumption about the future. A book whose holders had been drawing less would have more account
value today and less moneyness, so the sweep understates how far behaviour alone could go.

## The market model

**The long-run variance is extrapolated by a factor of fourteen.** The option chain reaches 3.23
years and the projection runs 45. Almost all of the liability's volatility exposure is in the
long-run level - 866 per point against 27 for the current variance - and that level is set by a fit
to options none of which mature past three years.

It is not a free parameter: pinning it to the realised ten-year level plus the issuer's own
one-point margin, which is the methodology Note 6 describes, costs four times the volatility error
beyond two years and drives mean reversion to its bound, so the chain rejects it. But "the chain
rejects the alternative" is a weaker statement than "the data identifies the value", and on a
forty-five-year liability the difference matters. The robustness table prices the issuer's level at
-2.3% of premium and E4 prices a 17% long-run level as a misspecification arm.

**The fund menu is three sleeves, not a fund menu.** The sub-account is modelled as a rebalanced
portfolio of an index sleeve, a rolling constant-maturity bond sleeve and cash, in the valuation as
well as along the realised path, so the bond sleeve's duration risk is inside both. What is missing
is the step from that portfolio to an actual menu of managed funds. The simulator carries a
``tracking_error`` parameter for exactly that and it is zero in every result here, so the basis
term in the hedge residual is the three-sleeve-against-index gap alone. A real book hedged with
index instruments would carry more.

**A constant 1.5% dividend yield.** FRED carries the S&P 500 price index, so the dividend yield
enters as an assumption in both the sub-account's total return and the futures excess return.

**The volatility surface is held at its December 2025 shape along the whole replay.** No free
historical option data exists, so what moves with the date in the in-force comparison and the
backtest is the curve and the observable instantaneous variance; the skew, the speed of mean
reversion and the long-run level are held. The in-force comparison is therefore run with the
surface of December 2025 attached to the rate environment of each disclosed year.

**The term structure is too flat, by a measured amount, in the direction the puts need.** Mean
reversion at 4.80 has a half-life of seven weeks, so the model is almost fully reverted to its
long-run level by six months, and carrying a one-month quote out to six months lands 1.22
volatility points low on average against VIX6M - 0.68 low from a three-month quote. The option
leg's level is extended by the same mapping out to one and two years, where nothing free can
check it, so its puts are priced too cheaply and the 21.67% ten-year option cost is a lower bound
rather than an estimate. The effect on the residual is not signed: too low a volatility misstates
the put's own vega and gamma as well as its price.

**The thirty-day skew is two thirds as deep as the market's.** The calibrated parameters generate
a SKEW index of 123.1 against 135.0 observed, a risk-neutral skewness of -2.31 against -3.50, and
they are shallower on 80% of the decade's days. Heston has no jumps and a diffusion reaches that
far into the left tail only with a correlation near minus one, which the chain does not ask for.
The direction cuts against this project's own result rather than for it: too little short-horizon
left tail understates the chance of the sharp declines that put a living benefit in the money, so
it biases the liability down, and the model is already around three times the disclosure. What
cannot be measured is the tenor that matters - no free data gives a one-year or ten-year skew, and
thirty days is the horizon furthest from a forty-five-year guarantee.

## The regression proxy

The hedge cannot run a full valuation at every rebalance date, so it runs a least-squares proxy,
and everything in the hedging workstream inherits the proxy's errors.

Its value is accurate - R-squared above 0.995 in range and a root mean square error of a few
thousandths of a per cent of account value. Its **delta is 36% off in the first policy year** and
21% in the second, 10% by year 5 and 5% by year 14, and then deteriorates again to 23% by year 25
as the delta itself shrinks and the same absolute error becomes a larger share of it. The backtest
covers policy years 3 to 13, which is the best part of that range, but nothing in the project
reads a delta off policy year 1. Its **second derivative is not usable**, which is why the option
leg is sized from a tabulated nested surface instead.

**Extrapolation is the condition on every hedging result.** The realised path asks the proxy for a
state outside its design on 36% of rebalance dates. That is reported with every run rather than
checked once, because a hedging conclusion built on mostly extrapolated Greeks is not a conclusion.

## The hedging backtest

**The liability is revalued with the same model that produced the hedge ratios.** This is the most
important caveat in the project. Any factor the model represents and the hedge covers is removed
nearly completely, limited only by convexity between rebalances. The variance reduction is an upper
bound on what a real programme achieves, not an estimate of it.

What the backtest does capture: gamma between rebalances, a single swap tenor against a
parallel-shift rho, one index against a blended sub-account, transaction spreads swept at half, one
and two times their base level, option carry through the roll, and the cost of sizing from
deliberately wrong Greeks.

What it does not: policyholder behaviour differing from assumption, gap risk between rebalances,
liquidity, margin and collateral calls, counterparty risk, and recalibration of the valuation model
itself. On a forty-five-year guarantee these are the larger risks.

**The equity hedge is a frictionless futures overlay.** No contract granularity, no roll basis, no
margin funding.

**One realised decade, widened but not escaped.** The bootstrap experiment resamples the decade's
equity days into other orderings, which gives the hedging result a distribution rather than a
single number. It does not escape the sample: every path is built from the same 2,493 days, and
the rate path, the curve and the credit spread are held on history's own course because a reordered
level is not a rate scenario. So the experiment says nothing about the joint equity-and-rate tail;
the crisis replays do, because they keep every day whole.

## The rider's economics

**Five cohorts, not a book.** The duration-and-moneyness pairs are set together rather than swept,
because a contract twelve years in has been drawing for seven of them and cannot be at its benefit
base; but they are five chosen points on a surface, not a weighted in-force. Aggregating them
would need the issuance history the filings do not break out by rider.

**Every cohort starts at a benefit base of one.** That is the normalisation, not a claim that a
twelve-year-old contract has had no roll-up: the figures are per unit of benefit base, so the
level of the base divides out. What it does mean is that the cohorts cannot be added together, and
that the account-to-base ratio is carrying all the information about where in its life each
contract is.

**The funding rate is the path's own cash rate with no spread.** Interest accrues at the effective
federal funds rate whether the balance is positive or negative. A hedging desk borrows above that
and lends below it, so the funding line is a lower bound on what carrying the hedge cost, and the
asymmetry would widen the figure rather than narrow it.

**Annualising a crisis window makes a rate.** The covid window is twenty-four trading days, so its
figures are scaled by a factor of eleven. They are comparable with each other and with the
decade's rate; they are not a forecast of a year that looks like February 2020.

**The break-even comparison is against a fee the contract never had.** The valuation solves at
issue, off the curve the backtest starts from. A contract sold in 2016 and still in force in 2026
was priced against a rate sheet set some time earlier still, on a curve this project does not
have, so the gap between the charge and the fee the guarantee needed is a statement about
September 2016 rather than a reconstruction of the issuer's pricing decision.

## Data

**Ten years of equity history.** The FRED S&P 500 series is a rolling ten-year window, which fixes
the start of the replay at 26 September 2016. One severe stress episode is a thin sample, and the
crisis replays reach only the four episodes inside it. The three before 2016 - the dot-com unwind,
the global financial crisis and August 2011 - are listed in the episode table as unreachable, with
the dates each would have needed, rather than quietly dropped.

**The credit spread is a proxy.** The own-credit spread is 0.6 times the Baa corporate spread,
which is a judgement about where Jackson's insurance subsidiaries sit rather than an observation of
their debt. BAA10Y is also missing on 21 of the 2,514 equity trading days and is forward filled up
to five days, with a longer gap raising rather than filling.

**The disclosed figure covers several product lines.** Item 7A's market risk benefit sensitivity
is scaled here by variable annuity account value from Note 11. Variable annuities dominate, so the
imprecision is small, but the two presentations do not tie exactly.

**A fresh pull will not reproduce the committed panel.** FRED revises series and the equity window
rolls, which is why the raw data is committed.

## The in-force comparison

**Whole policy years against calendar year-ends.** The contract recursion steps in whole policy
years while the disclosed dates are year-ends and the inception is in September, so each date is
valued at the anniversary on or before it with the market and the account value of the disclosed
date itself. Every date is also valued at the following anniversary and the spread is reported: it
moves the equity-down figure by 0.18 to 0.23 points of account value against a disclosed figure of
about 1.11, so it is a fifth of the quantity being compared and a ninth to a thirteenth of the gap
being explained.

## The comparison with the disclosed hedge book

**Item 7A's derivative table is the whole company's, not the variable annuity's.** The same
swaps, futures, bond forwards and puts hedge the fixed-index and RILA book, the general account's
own duration and, after December 2023, Brooke Re's statutory position. Attributing all of it to
the guarantee is wrong, which is why the offset ratio is reported against both the market risk
benefit alone and the two disclosed liability lines together, and why neither is treated as the
hedge of the variable annuity by itself.

**One of those two lines moved by more than its balance can explain, and E5 says no book can.**
The fixed-index and RILA embedded derivative's equity sensitivity goes from $4m at the end of 2024
to $1,321m at the end of 2025 in the same table of the same filing, on a balance that grew by three
quarters. Modelling the index-linked book directly puts its whole reachable exposure within a
factor of 2.4 across every cap and composition, and within 9.8 even for a single segment at any
point in its term, against a filed move of 330. A book of the filed size would have shown 3.5%,
10.5%, 27% and 50% of the guarantee's equity move across the four year-ends where the filings show
0.1%, 0.2%, 0.5% and 79%.

So the two bases are not two views of one thing. **The combined-basis figures are not comparable
across 2024 and 2025**, the apparent flatness of that series is an artefact of a presentation
change, and the market-risk-benefit column is the one that carries a consistent measurement. The
likeliest explanation - a line carried net of the derivatives hedging it through FY2024 and gross
from FY2025 - is an inference the filings do not state, and it is not relied on for anything beyond
declining to read the combined series as a series.

**E5's own exposure is a model of a product, not of Jackson's product.** Six-year point-to-point
terms, a 20% buffer because that is the 10-K's example, and caps swept because none is published
and the break-even cap does not bind at these rates. Jackson sells one-, three- and six-year terms
at caps it does not disclose. What makes the exposure worth quoting is that the model's level
tracks the filed embedded derivative - 15.9 to 24.7% of account value against 10.9 to 29.8%, same
direction of travel - on a quantity nothing was fitted to. The denominator is "other contract
holder funds", which the filings say *includes* the embedded derivative and so is not the contract
account value; the account value is not disclosed separately, and the size ratio inherits that.

**The combined ratio divides by a remainder.** Once the two liability lines nearly cancel - which
at the end of 2025 they do, leaving a sixth of the guarantee's own up-move - the combined offset
ratio is dividing by a small number and its -53% is arithmetic rather than a measurement. The
denominator is published next to it for that reason.

**The benchmark is the model's hedge of the model's liability.** The 93 to 109% a full hedge
achieves is an upper bound in the same way the backtest's variance reduction is: the liability is
revalued with the model that sized the hedge. It establishes that convexity over a 50 to 100bp
shift costs about eight points either side of one, which is the only thing it is used for, and it
is not evidence about what a real programme achieves.

**Nothing here reaches before 2021.** Item 7A's derivative tables start with the FY2022 filing,
so the series runs 2021 to 2025 and the two bases split it: pre-LDTI carrying value at 2021 and
2022, market risk benefit from 2022 on. The pivot the equity series shows in 2024 sits beside the
Brooke Re transaction of December 2023, and that is adjacency rather than attribution - five
year-ends cannot identify a cause.

## The accounting comparison

Only two differences between the economic and reporting bases are measurable from public data: the
margin loading in the annuity table and the own-credit adjustment. Everything else that drives
Jackson's reported volatility is out of reach - assumption updates, the attributed-fee percentage
frozen at each vintage's own inception across a book of many vintages, the Brooke Re captive
structure and its modified GAAP approach for statutory purposes, and the fact that the hedging
programme also targets statutory capital rather than only the economic liability.

So the reported-to-economic volatility ratio is a floor on the effect rather than a measurement of
it, and the value of the exercise is establishing that the mechanism is real and signed correctly.

## The statutory lens

**The equity risk premium is a stated sweep, not an estimate.** Ten years of free equity history
says nothing about a long-horizon premium, so the requirement is reported at three levels and the
spread between them is part of the result. Reading the middle figure as an estimate would be
reading more into it than the data supports.

**The scenario set is this project's own.** A real VM-21 calculation runs the Academy generator
under prescribed assumptions with a standard projection and a company-specific set of margins. What
is here is the same shape - a greatest present value of accumulated deficiency, a conditional tail
expectation, a cash surrender value floor - run on this project's own real-world paths. It is a
lens on the economic result rather than a statutory filing.

## What this does not claim

Nothing here is a valuation of Jackson Financial or of its liabilities, and none of it is
investment advice. The disclosed figures are Jackson's; the model's are a stylised book's, and the
two are compared on sign, shape and scale-free magnitude rather than treated as estimates of each
other. Where the model and the disclosure differ, the presumption throughout is that the model is
the thing that is wrong.
