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

**One equity factor.** The sub-account is a single risky asset at the mix's equity weight. The bond
sleeve's own duration risk is not inside the valuation, only in the realised path, which is what
creates the fund basis term in the hedge residual.

**A constant 1.5% dividend yield.** FRED carries the S&P 500 price index, so the dividend yield
enters as an assumption in both the sub-account's total return and the futures excess return.

**The volatility surface is held at its December 2025 shape along the whole replay.** No free
historical option data exists, so what moves with the date in the in-force comparison and the
backtest is the curve and the observable instantaneous variance; the skew, the speed of mean
reversion and the long-run level are held. The in-force comparison is therefore run with the
surface of December 2025 attached to the rate environment of each disclosed year.

## The regression proxy

The hedge cannot run a full valuation at every rebalance date, so it runs a least-squares proxy,
and everything in the hedging workstream inherits the proxy's errors.

Its value is accurate - R-squared above 0.995 in range and a root mean square error of a few
thousandths of a per cent of account value. Its **delta is 36% off in the first policy year**,
falling to 4% by year 5 and 2% by year 9; the backtest starts at duration 3, so it lives in the
usable part, but nothing in the project reads a delta off policy year 1. Its **second derivative is
not usable**, which is why the option leg is sized from a tabulated nested surface instead.

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
single number. It does not escape the sample: every path is built from the same 2,492 days, and
the rate path, the curve and the credit spread are held on history's own course because a reordered
level is not a rate scenario. So the experiment says nothing about the joint equity-and-rate tail;
the crisis replays do, because they keep every day whole.

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
about 1.11, so it is a fifth of the quantity being compared and a fifteenth of the gap being
explained.

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
