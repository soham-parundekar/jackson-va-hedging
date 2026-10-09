# Validation

Every number here is in `reports/tables/`, and every claim is the table read back rather than a
summary of it. Where the model misses, the miss is the finding.

The hypotheses this is testing are in `docs/research_design.md` and were written before the
results. Two of them did not survive in the form they were stated, which is reported here rather
than edited there.

## Numerical checks

These establish that the arithmetic closes before anything is asked of the economics.

| Check | Result | Where |
|---|---|---|
| Par bootstrap reprices its own quotes | worst round trip 2e-16 over 2,495 dates | `build_dataset.py`, `tests/test_market_models.py` |
| Hull-White reproduces the initial curve | exact | `tests/test_market_models.py` |
| Hull-White's bond price seen from a future date | martingale identity holds to 1e-9 at a = 0.05 and 0.27 | `tests/test_market_models.py` |
| COS pricer against Black-Scholes at zero volatility of variance | matches | `tests/test_market_models.py` |
| COS pricer holds put-call parity | under 1e-5 of the forward across 7 strikes and 5 maturities | `tests/test_market_models.py` |
| COS pricer prices nothing impossible over the fit's own parameter bounds | no negative price, no NaN, parity within a basis point of the forward at 164 corners | `tests/test_market_models.py` |
| Simulator against the COS pricer with rates switched off | matches | `tests/test_market_models.py` |
| Discounted index and sub-account are martingales | within Monte Carlo error | `tests/test_market_models.py` |
| Continuous charge collection against 20,000-step sub-stepping | exact to 1e-12 | `tests/test_liability.py` |
| The contract recursion against a scalar reading of the documented anniversary order | agree to 1e-13 on a path that exhausts, steps up and reaches the adjustment | `tests/test_liability.py` |
| The rate leg's annuity rho against repricing the swap under a parallel shift | 0.45% apart at the ten-year tenor held | `tests/test_hedge.py` |
| The hedge solve against an explicitly formed penalised least-squares system | agree to 1e-9 across three instrument sets and two hedge ratios | `tests/test_hedge.py` |
| Antithetic pairs are found inside their own simulation block | the mean is pairing-invariant, the error is not | `tests/test_market_models.py` |
| Every register row points at a file that exists, and names a component | 30 rows | `tests/test_artifacts.py` |
| Every source `docs/literature.md` cites has a register row | all 14 DOIs | `tests/test_artifacts.py` |
| Accession numbers in the docs match the register's | one spelling each | `tests/test_artifacts.py` |
| The register and the annotated bibliography name the same sources | both directions | `tests/test_artifacts.py` |
| Deaths and survivors account for everyone | exact to 1e-12 | `tests/test_mortality.py` |
| The replica's p-value approximation against the exact t survival function | conservative on both filed rows; optimistic only where the coefficient is strong enough that p is zero either way | `tests/test_disclosure_replica.py` |
| A mortality rate missing where the valuation reads the table stops the loader | the check reads the export rather than a filled copy of it | `tests/test_mortality.py` |
| Period table implies longer life than Basic | holds at both valuation years | `build_dataset.py` |
| Overlapping filings agree | 12 liability figures and 3 derivative dates in two filings each, all agree | `build_dataset.py`, `tests/test_disclosures.py` |
| Profit attribution and residual add to the total | exact by construction | `tests/test_hedge.py` |
| The ledger uses nothing from after the rebalance date | rebuilt from the prior position | `tests/test_hedge.py` |
| Risk-neutral skewness off the characteristic function against a simulated sample | agree to 1% at one year | `tests/test_market_models.py` |
| Disclosed derivative lines sum to their disclosed totals | all 32 blocks, exactly | `tests/test_disclosures.py` |
| Both routes to the hedge's exposure vector land on the same index delta | exact | `tests/test_hedge.py` |
| Cash moved by the five things that have ledger columns and nothing else | exact | `tests/test_hedge.py` |
| The economic decomposition adds to the ledger's own profit | exact by construction | `tests/test_hedge.py` |
| The chain's own discount factors against Treasury | 38 to 94bp over, across 18 expiries | `build_dataset.py` |
| Every committed figure carries only the chunks a plot needs | all 10, nothing after IEND | `tests/test_artifacts.py` |

281 tests, no framework required.

Three of these rows exist because an earlier version of this table was weaker than it looked, and
the pattern is worth stating once. A put-call parity check was already here, at the money, at one
parameter set, with a tolerance of half a per cent of the price - about thirty times looser than
what the pricer actually delivers. It could not fail for the reason it existed. The defect it was
meant to find lived where no test had ever priced: the call payoff's cosine coefficient integrates
exp(y) over the upper half of the truncation range, so it comes out enormous and the sum over
terms has to cancel it back to a price of order one, and at a dispersed enough variance
distribution that cancellation is gone. Over the calibration's own parameter bounds, which a
least-squares search is free to walk through, the direct call branch broke parity by more than a
basis point of the forward at a third of the corners, returned a *negative* price at a sixth, and
overflowed to NaN at the far ones - every one of them reaching the objective as a residual it
would steer by. `cos_price` now prices the call both ways and keeps the direct value only where it
is finite, above its own arbitrage floor, and within a basis point of the forward of parity. On
the committed chain the direct value is kept for every quote, so the fitted parameters come out
bit for bit identical to what is committed, and the Hull-White row is there for the same kind of
reason: nothing had ever priced a bond seen from a date inside the projection, which is what the
nested valuation rebuilds its curve from.

The figure row is the only check in the project that looks at a file rather than at a number, and
it is there because something got past every other one. A figure is a binary blob: every viewer
renders it identically whatever else is inside it, and a text search of the repository cannot see
in. Ten of these went to GitHub carrying a 5,758-byte private PNG chunk that the step copying them
out of the build environment had inserted - no effect on a single pixel, and a plain-text
provenance record to anything that reads the bytes. Stripping it reproduced the build's own output
exactly, byte for byte on all ten, which is what says the images themselves were never the
problem.

The parity check is the one of these that is not arithmetic. The regression behind the chain's
forwards is self-validating on fit - the call-minus-put spread is linear in strike by
construction, and the worst residual across eighteen expiries is 0.96 index points on an index at
7,710 - but a perfectly straight line can sit at the wrong level, which is what a mislabelled
strike column or a snapshot taken on the wrong date produces. The level has an external check.
Each expiry's discount factor is a financing rate, and across eighteen expiries from eighteen days
to 3.2 years it comes out 38 to 94 basis points over the matched-maturity Treasury, and within a
few points of 45 everywhere except the two shortest expiries. That is what an SPX box spread looks
like, and nothing in the calibration was told to make it so.

### Monte Carlo error and truncation

At twenty thousand paths the standard error on the market risk benefit is $136 on a $100,000
policy, and it halves as paths quadruple - 2.01, 1.96 and 1.99 across the three quadruplings in
`monte_carlo_convergence.csv`. Greeks are taken on common random numbers and each carries the
standard error of its own paired difference: the equity exposure's is 164 on 14,501, rho's is 19
on 5,887 per 100bp, vega's is 40 on 893. A Greek whose error is a third of its value is not a
risk number, and printing the error is the only way to know.

The forty-thousand-path row of that table used to read 106 rather than 96, which is the one
figure in the project an audit found by reading a trend rather than by checking a formula: a
doubling of the paths should divide the error by 1.41 and that row divided it by 1.27. Paths are
simulated in independently seeded blocks of twenty thousand and each block mirrors its own first
half into its own second half, so above one block the halves of the whole array are not
antithetic pairs. Pairing them anyway leaves every value untouched - averaging pairs and
averaging rows are the same sum - and throws away the variance reduction, so the error came out a
tenth too large. The pairing now comes from the paths rather than from a halving rule, and the
row sits on the trend at 1.415.

Truncation is measured on common draws, which is the only way the comparison means anything: the
simulator draws its normals in one array whose width follows the horizon, so a forty-year run and
a fifty-year run at the same seed are different worlds rather than a prefix and its extension.
Measured properly, cutting the projection at age 115 costs less than a dollar and at 105 costs
$36, against a standard error of $136.

## The calibrated surface against free data it never saw

The Heston parameters come from one afternoon's SPX chain, 28 September 2026. Two Cboe index
histories make that one-day fit testable across a decade, and neither enters the calibration.

**The term structure, carried outward.** The chain reaches 3.23 years and the puts in the hedge
run one and two, so the option leg's volatility level is extended from a quoted tenor by the
model's own variance curve. Six months is as far as free data goes, so the outward direction is
only testable there - and it is the direction that matters, because the inward test the project
ran first says nothing about the extension it actually uses.

| Quoted | Carried to | Direction | Mean error | Within two points |
|---|---|---|---|---|
| three-month | one month | inward | +0.98 points | 60% of 2,943 days |
| one-month | six months | outward | -1.22 points | 55% |
| three-month | six months | outward | -0.68 points | 70% |

Both outward rows are biased low, and the reason is structural rather than incidental: mean
reversion at 4.80 has a half-life of 0.14 years, so by six months the model has already reverted
almost entirely to its long-run 21.68%, giving a term structure slope of about +1.9 points from an
18.3% one-month quote against the +3.37 points the market actually prices. The market's volatility
curve between one and six months is steeper than the model's.

Two consequences follow, and they point in opposite directions on the two things the project
concluded. The option leg's volatility is extended by the same mapping out to one and two years,
so its puts are priced too cheaply and the 21.67% ten-year option cost is a lower bound. And the
fitted long-run level of 21.68% comes out looking too *low* rather than too high - which is the
opposite direction from the issuer's described 19.12% and so independent support, from a source
the calibration never touched, for the decision to let the chain reject that level.

**The skew, which is the one thing no variance quote can see.** Every volatility index is a
variance rate, so the whole set of them is silent on the correlation and the volatility of
variance: those two can trade off against each other without moving the expected average variance
at any tenor. Cboe's SKEW index is not silent on them. It reports `100 - 10 x` the risk-neutral
skewness of the thirty-day return, and the calibrated parameters generate that skewness in closed
form off the characteristic function, with the level supplied by the one-month quote exactly as
the paths supply it.

| | Model | Observed |
|---|---|---|
| Mean SKEW index | 123.1 | 135.0 |
| Thirty-day risk-neutral skewness | -2.31 | -3.50 |
| Days the model is shallower | 80% of 2,943 | |

The model's left tail at thirty days is about two thirds as deep as the market's, which is what a
pure diffusion does: reaching a skewness near -3.5 over a month without jumps needs a correlation
close to -1, and the chain does not ask for one. The direction is worth stating because it cuts
against the project's own headline. Too little short-horizon left tail understates the chance of
the sharp declines that put a living benefit in the money, so it biases the liability *down* - and
the model already comes out around three times the disclosure, so correcting it would widen the
gap rather than close it.

What the comparison cannot do is track the index's variation. The model's skewness falls out of
one state variable, so the model index is a monotone function of the one-month quote and its rank
correlation with the observed index is forced to be minus the quote's own: +0.128 against -0.128,
mirror images by construction. That is not weak agreement, it is no information, and whatever
moves the market's skew day to day is not in this model. Both numbers are published side by side
so the identity is visible rather than inferred.

## The at-issue result, which is itself a check

With the attribution percentage calibrated the way Note 6 describes, the market risk benefit at
inception is zero. That is not a free parameter being fitted; it is the definition closing.

| Quantity | Value |
|---|---|
| PV of living-benefit payments | 20,977 |
| PV of death benefit above the account | 1,207 |
| PV of total attributable fees | 28,233 |
| Attribution percentage | 0.7857 |
| Market risk benefit at issue | 0 |
| Gross guarantee, % of premium | 22.18 |
| Probability of exhaustion by year 20 | 0.777 |
| Expected future lifetime at 70 | 19.64 years |

The gross guarantee reaches a fifth of premium because the contract defers five years, accruing
the Core option's 6% bonus on the benefit base, and then draws 5.95% a year against a
risk-neutral drift near 4.2% less 2.26% of charges. First claims arrive in policy year 6, the flow
peaks at year 17, and the account is gone on 78% of paths by year 20. The guarantee is the
expected outcome for this contract rather than a tail event.

## Sign

**Hypothesis 1 holds: 18 of 18.** Every disclosed shock, across four balance-sheet dates, both
directions of both shocks, equity and rates. No sign was put in; the model is a risk-neutral
projection and the signs fall out of it.

## Shape

**Hypothesis 2 holds, and closely.** Where a filing discloses both 50bp and 100bp, the ratio of
the two is a convexity test the model was never fitted to.

| | up 100/50 | down 100/50 |
|---|---|---|
| Disclosed, FY2024 | 1.8553 | 2.1279 |
| Model, GWB/AV 0.8 | 1.8841 | 2.1277 |
| Model, GWB/AV 1.0 | 1.8822 | 2.1275 |
| Model, GWB/AV 1.2 | 1.8840 | 2.1210 |

Above two on the downside and below two on the upside, as the hypothesis required, and the
downside lands within 0.0002 of the filing. The ratio barely moves across moneyness, which matters
because it means the shape test is not quietly a test of where the book sits.

**Hypothesis 3 holds.** The year-on-year decline in sensitivity per dollar of account value, as
the book moved out of the money:

| | 2022 | 2023 | 2024 | 2025 | 2025 / 2022 |
|---|---|---|---|---|---|
| Model, equity down 10% | 4.56 | 3.91 | 3.02 | 2.64 | 0.5795 |
| Disclosed | 1.47 | 1.18 | 0.96 | 0.85 | 0.5794 |

The levels differ by about three times and the declines agree to one part in ten thousand.
That is the strongest single piece of evidence in the project that the model has the right shape:
the thing driving the decline - a book moving out of the money as markets rose - is reproduced
without being fitted.

## Scale

**Hypothesis 4 holds in its first clause and fails in its second.** It predicted that the model
would sit above the disclosure and that the gap would be closable by behaviour assumptions inside
the ranges the literature supports. The first is right. The second is wrong, and the way it is
wrong is more informative than the prediction would have been.

The vintage portfolio - five issue years valued as one book, each carrying the attribution
percentage its own issue date calibrates to - reaches a weighted attained age of 70.9 against the
disclosed 70 and a blended withdrawal rate of 5.73%. Its multiple over the disclosure:

| Shock | Dates | Model over disclosed, mean | Range |
|---|---|---|---|
| equity up 10% | 4 | 2.65 | 2.28 to 3.11 |
| equity down 10% | 4 | 2.83 | 2.58 to 3.19 |
| rates up 50bp | 3 | 3.17 | 2.79 to 3.48 |
| rates down 50bp | 3 | 3.15 | 2.73 to 3.48 |
| rates up 100bp | 2 | 2.82 | 2.82 to 2.83 |
| rates down 100bp | 2 | 2.75 | 2.72 to 2.77 |

A multiple that near-constant across six shocks and four dates points at one structural assumption
rather than a pile of errors. Net amount at risk comes out at 1.11% of account value against the
2.3% disclosed.

Four of the five vintages calibrate to an attribution percentage of exactly 1.00. A guarantee
written anywhere in the 2016 to 2022 rate environment did not have the fees to cover itself at a
risk-neutral valuation; only the 2024 vintage, written at a 4.33% ten-year rate, has margin. That
is the mechanism behind a disclosed liability that is now an asset, and it falls out of the model
rather than being put in.

### What does not close the gap

The behaviour sweep moves utilisation from 1.0 to 0.6 and the at-the-money lapse rate from zero to
4%, after scaling for the 75% of account value that carries a withdrawal guarantee. The best
combination by widest miss is 0.80 utilisation with 4% lapse, every compared shock within 61% of
the disclosed figure, against 2.00 times the disclosure on the static benchmark.

But the two shock families do not move together, and that is the result:

| | static benchmark | far corner |
|---|---|---|
| Equity multiple | 1.90 | 1.60 |
| Rate multiple | 2.09 | 0.47 |
| Equity over rate | 0.91 | 3.42 |

Drawing less takes duration out of the guarantee - fewer paths exhaust, so it is less of a
long-dated annuity - and the rate sensitivity collapses with it. The benefit base is still there
whatever the owner draws, so the equity sensitivity barely moves. By the time utilisation is low
enough to match the rate figure, the model is still 2.03 times the disclosed equity figure.

So behaviour explains the rate half of the gap and not the equity half, which is a different
conclusion from the one the hypothesis predicted and a more useful one, because it says where to
look next.

### What does close the equity half

Moneyness, on the evidence of the model's own curve. For each disclosed year, the benefit base
over account value at which the model would reproduce that year's figure:

| As of | equity down 10% implies | equity up 10% implies |
|---|---|---|
| 2022 | 0.940 | 1.009 |
| 2023 | 0.907 | 0.947 |
| 2024 | 0.853 | 0.901 |
| 2025 | 0.822 | 0.855 |

Two independent shocks implying the same ratio, year after year, without being made to. The ratio
declines across the four years, which is what a decade of rising markets does to a book. The
vintage portfolio sits at 0.985, which is the size of the remaining gap.

The rate shocks fall **outside** the model's range at every moneyness, which is the same statement
in reverse: moneyness cannot explain the rate half, and behaviour can.

### The withdrawal rate is the lever

The diagnostic that makes the rate half quantitative. Holding everything else, the guaranteed
withdrawal rate moves both sensitivities towards the disclosure, and not by the same factor:

| GAWA | attribution | exhausted by 20y | equity down 10% | rates up 100bp |
|---|---|---|---|---|
| 1.00% | 0.154 | 0.133 | 1.14 | -1.52 |
| 2.00% | 0.151 | 0.258 | 0.99 | -1.35 |
| 3.00% | 0.218 | 0.399 | 1.09 | -1.86 |
| 5.75% | 0.731 | 0.756 | 1.92 | -4.95 |
| 6.50% | 0.949 | 0.824 | 2.18 | -5.97 |

Matching the disclosed rate sensitivity takes an effective rate near 1.7%; matching the disclosed
equity sensitivity takes 2% or below. At a 1.7% draw the rate sensitivity is 0.27 of its
contractual-rate value and the equity sensitivity 0.52 of its own, which is the same divergence
the behaviour sweep found, measured on a different lever.

Below about 1.5% the relationship turns: the account stops exhausting at all, the living benefit
stops being what the sensitivity is made of, and the death benefit and the fee stream take over.
The implied rate is read off the monotone branch only, because reading it off the turn would be
reading the turn.

An effective rate of 2% against a contractual 5.95% is not a claim that holders draw a third of
their entitlement. It is a claim about a book, and a book in which a large share of contracts have
not started withdrawing draws less than any of its rate sheet bands. The vintage portfolio models
that share directly - two of its five vintages are still deferring at the last disclosed date -
which is why it closes part of the gap that the single policy does not.

## The proxy the hedging results rest on

Everything in the next section runs on a regression proxy rather than on full valuations, so the
proxy's own error is measured first.

| Policy year | R² in range | value RMSE, share of premium | delta RMSE, share of premium | mean abs nested delta, $ | relative |
|---|---|---|---|---|---|
| 1 | 0.9952 | 0.0059 | 0.126 | 35.0 | 36% |
| 2 | 0.9978 | 0.0053 | 0.081 | 38.6 | 21% |
| 5 | 0.9990 | 0.0068 | 0.044 | 42.7 | 10% |
| 9 | 0.9997 | 0.0044 | 0.021 | 31.0 | 7% |
| 14 | 0.9998 | 0.0034 | 0.010 | 21.4 | 5% |
| 20 | 0.9995 | 0.0038 | 0.021 | 14.9 | 14% |
| 25 | 0.9985 | 0.0049 | 0.025 | 10.5 | 23% |
| 30 | 0.9992 | 0.0020 | 0.006 | 5.7 | 11% |

Both RMSE columns are shares of a premium of 100, not percentages, which is why the last column
is the one worth reading: 0.126 of a premium of 100 is 12.6 dollars against a mean absolute
nested delta of 35, so 36%. That ratio is the quantity that decides whether a hedge sized off the
proxy is sized off anything. The header used to say "% of account" over the value column, and the
same misreading reached `docs/limitations.md` as "a few thousandths of a per cent" for an error
that is two to seven tenths of one. The denominator is premium rather than each node's own
account value, which also matters at the later years, where the account has moved a long way from
it.

The value is accurate everywhere, at two to seven tenths of a per cent of premium. The delta is not usable in the first two policy years, where the
fitting paths have barely dispersed and the whole design piles into a narrow band. It is best from
year 9 to year 14 and then deteriorates again, not because the fit gets worse - the value RMSE
barely moves - but because the delta itself shrinks as the book runs off, so the same absolute
error is a larger share of it. The backtest covers policy years 3 to 13, which is the best part of
that range, and the step-up table is restricted to years 5 and 9 for the same reason.

The second derivative is not usable at all. `convexity_proxy_comparison.csv` puts the proxy's
gamma against nested valuations at the same nodes and the errors are the size of the quantity, so
the option leg is sized from a tabulated nested surface instead. Substituting it is not free and
`hedge_convexity_source.csv` is where it earns its keep or does not.

**Extrapolation is a condition on every hedging result, not a footnote.** The realised path asks
the proxy for a state outside its design on 36% of rebalance dates. That share is reported with
every run.

## Hedging

**Hypothesis 5 holds on variance and is incomplete on the residual.** A delta and rho hedge
removes a substantial share of the daily variation, and adding convexity removes most of the
rest - which means the residual is not dominated by implied volatility so much as by the absence
of an instrument for it.

Residual daily standard deviation over the ten-year replay, as a share of account value, at daily
rebalancing and base costs:

| Strategy | Residual sd | Trading cost over the decade |
|---|---|---|
| S1, futures | 0.304% | 0.05% |
| S2, plus a receive-fixed swap | 0.188% | 0.08% |
| S3, plus listed puts | 0.139% | 21.67% |

The rate leg costs almost nothing to trade and takes out a third of what the equity leg left. The
option leg takes out another quarter and costs two hundred and seventy times as much to trade.

The cost column is commission, spread and carry. On the first two rows it is not what running the
hedge cost: funding the losses the equity leg realised against a rising market was a hundred to
three hundred times larger, and the section on what the rider earned measures it.

### The hedge against the one external check there is

Everything above is the model marking its own homework: the liability is revalued with the model
that produced the hedge ratios, so a high variance reduction is close to arithmetic. Item 7A is
the only outside evidence available, and it is better evidence than it looks. The same tables
that publish the guarantee's sensitivity to a shock publish the derivative book's sensitivity to
the same shock on the same date, so the ratio of the two is a *disclosed offset ratio* - how much
of its own guarantee move Jackson's hedge actually covered. Nothing in the model is fitted to it.

The convention matters and is not the obvious one. A liability impact is a change in a carrying
amount, so positive is a loss; a derivative impact is a change in an asset's mark, so positive is
a gain. The two therefore cancel in earnings when their *raw* figures carry the same sign, which
makes a complete hedge +1 rather than -1. Writing it the other way round turns every ratio
negative, which is how the error announces itself.

On the market risk benefit, which is one measurement across all four years:

| Date | Shift | Equity down | Equity up | Rates down | Rates up |
|---|---|---|---|---|---|
| 2022-12-31 | 50bp | 104% | 77% | 37% | 43% |
| 2023-12-31 | 50bp | 110% | 81% | 86% | 78% |
| 2024-12-31 | 50bp | 49% | 50% | 81% | 86% |
| 2024-12-31 | 100bp | 49% | 50% | 80% | 88% |
| 2025-12-31 | 100bp | 9% | -9% | 79% | 86% |

The pre-LDTI carrying value gives two more dates, 2021-12-31 at 112%/42% on equity and 24%/27% on
rates and 2022-12-31 at 135%/102% and 52%/62%. That is a different measurement and is not
comparable in level with the rows above, so it is kept in the table and read within itself.

**The two legs moved in opposite directions, and the dates say when.** The rate share goes
37-43% at the end of 2022, to 78-86% a year later, and then sits at 79-88% through 2024 and 2025.
The equity share is at or above a full hedge on the downside through 2023 and four fifths of one
on the upside - 104% and 77%, then 110% and 81% - and halves to 49% and 50% in 2024, the first
full year after Brooke Re was formed in December 2023.
Every one of those readings is on the same basis and the same shock size except where noted.

**2024 is disclosed twice, at two shock sizes, and that is the control.** The FY2024 filing shows
2024-12-31 at 50bp and the FY2025 filing shows the same date at 100bp. The disclosed ratio moves
1.1 points on the down shift and 2.7 on the up; a full hedge of the model's own liability moves
3.8 and 4.3 over the same change. So the disclosed book's ratio is *less* sensitive to the size of
the shift than a complete hedge is, which is the opposite of what a convexity explanation of the
shortfall would predict.

**The 2025 equity figure is the index-linked book, and the combined series is not a series.**
Taken alone, the equity offset falls from 49% to 9% and turns negative on the up move, which reads
as a programme being dismantled. The same filing discloses a fixed-index and RILA embedded
derivative whose equity sensitivity goes from $4m to $1,321m over that year against a market risk
benefit sensitivity of $1,574m - a liability that owes more when equity rises is a natural short
against a guarantee that gets cheaper - so on the combined basis the equity offset reads 49% then
44%, and the obvious conclusion is that the hedge did not shrink but the net exposure did.

**E5 says that conclusion is not available.** Modelling the index-linked book directly - six-year
point-to-point segments, a 20% buffer, caps swept, cohorts rolled along realised index history and
weighted by the issuance the filings imply - puts a book of Jackson's own disclosed size at 3.5%,
10.5%, 27% and 50% of the guarantee's equity move across the four year-ends. The filings report
0.1%, 0.2%, 0.5% and 79%. The first three are under two to three *per cent* of what an unhedged
book of that size would show; the last is 1.6 times it. The model's own exposure spans a factor of
2.4 across every cap and composition it can reach, and 9.8 for a single segment at any point in its
term. The filed line moved by a factor of 330 in one year, which nothing an index-linked book can
do accounts for.

What does account for it is a change in what the line reports - a book carried net of the
derivatives hedging it through the FY2024 filing and gross from FY2025 would look exactly like
this. That is an inference and the filings do not say it. What follows from it either way is that
the 2024 and 2025 combined figures are not the same measurement, so the apparent flatness of the
combined series across those two years is an artefact, and the market-risk-benefit column is the
one to read. `docs/limitations.md` carries this where the comparison is caveated.

The model's level is what licenses its exposure. Its embedded derivative comes out at 15.9%, 20.0%,
22.7% and 24.7% of account value across the four dates against 10.9%, 23.5%, 26.2% and 29.8%
filed - the same magnitude and the same direction of travel, within a fifth on three of the four
dates and 45% above on the first, on a quantity
nothing was fitted to. A delta nobody can check is worth little; agreement on the level does not
make the delta right, but disagreement would have made it worthless.

A second caveat on the combined column is arithmetic: it divides by what is left after the two
lines cancel, which at the end of 2025 is a fifth of the guarantee's down move and a sixth of its
up move. A ratio built on a sixth of a number is not a measurement, so the denominator is published
beside it and the -53% is not read as an offset.

**What a full hedge would have looked like, for comparison.** An offset below one is not by
itself evidence of a partial hedge: the liability is convex in rates and a swap is nearly linear,
so even a book sized to kill rho exactly under-recovers a large shift. The model supplies that
benchmark by sizing its own hedge at each of these dates, off the full Greeks rather than the
regression proxy, and repricing both sides under the same shocks.

| | Disclosed | S1, futures | S2, plus the swap | S3, plus puts |
|---|---|---|---|---|
| Rates | 37% to 88% | 2% to 3% | 93% to 109% | 93% to 109% |
| Equity | -9% to 110% | 90% to 113% | 90% to 113% | 101% to 108% |

The disclosed column is the market risk benefit basis on both rows, so the two are read against
each other. The pre-LDTI dates would stretch the equity range to 135% and the rate range down to
24%, and mixing them in would make the comparison a comparison of accounting bases.

Convexity therefore accounts for about eight points of scatter either side of one, not the twelve
to twenty point rate shortfall or the fifty to a hundred point equity one the disclosure shows
outside 2022-2023. That is the same conclusion the
statutory and reporting lenses reach from the inside - the programme is not targeting the economic
liability alone - arrived at this time from Jackson's own numbers rather than from the model's.

The futures-only row is a sanity check rather than a result: a hedge of equity delta should barely
touch a rate shock, and 2 to 3% is the small rho the futures forward carries through its own
discount factor. A number near one there would have meant the rate shock was leaking into the
equity leg.

**What makes any of this checkable.** Three balance-sheet dates are disclosed by two filings each
and every overlapping figure agrees exactly: 2022-12-31 across the FY2022 and FY2023 filings,
2023-12-31 across FY2023 and FY2024, 2024-12-31 across FY2024 and FY2025. On all thirty-two
disclosed blocks the instrument lines reproduce the disclosed Total exactly. Both are asserted in
`tests/test_disclosures.py` rather than checked once by hand, because a hand-transcribed
sensitivity table is the only unvalidated input this project has.

### The crisis replays

Chosen by what happened rather than by what flatters a hedge, and deliberately different in kind.

| Episode | Days | Index | Rates | Peak vol | Unhedged | S1 | S2 | S3 |
|---|---|---|---|---|---|---|---|---|
| volmageddon | 11 | -8.8% | +18bp | 26.5% | -1.96% | 59.1% | 74.5% | 97.9% |
| Q4 2018 | 64 | -19.5% | -32bp | 26.0% | -9.13% | 84.2% | 96.1% | 98.8% |
| covid | 24 | -33.8% | -79bp | 45.4% | -22.00% | 85.7% | 94.1% | 98.9% |
| 2022 double | 197 | -24.4% | +228bp | 26.2% | +1.16% | 73.7% | 98.5% | 99.1% |

Unhedged is the episode's total profit as a share of account value; the strategy columns are
variance reduction. February and March 2020 is the clearest case: the unhedged guarantee lost 22%
of account value in twenty-four trading days, futures alone recovered two thirds of it, futures
and swaps left 2.7%, and the option leg turned it into a 1.9% gain.

Volmageddon is the opposite case and the one worth dwelling on. An 8.8% index move with volatility
doubling, and the futures-and-swaps hedge removes only 74.5% of the variance against 94% in covid.
That episode was a volatility event rather than a level event, and instruments that carry no vega
were never going to catch it.

### What one decade is worth as evidence

Every hedging number above comes from one path: the 2,493 trading days between September 2016 and
September 2026, in the order they arrived. That is one observation, and it is a decade in which
the index grew at 15.4% a year - 14.4% continuously compounded, which is the basis the drift
arms are set on - and the two bad stretches were short.

So the days are resampled. A stationary bootstrap with an 11.09-day mean block, set from the
integrated autocorrelation of squared returns rather than by eye, reorders the equity days while
keeping each one paired with its own volatility state. The rate path, the curve and the credit
spread stay on history's own course, because a reordered level is not a rate scenario: at that
block length a resampled level would jump about 225 times in a ten-year path, by an average of
139bp in the ten-year zero against a realised daily standard deviation of 5.3bp. Two arms on the
same draws - the ordering changed with the average left alone, and the same orderings re-centred
on the window's mean financing rate plus the 4% equity risk premium the statutory work uses.

**The decade that happened sits at the top of the distribution.**

| | | p10 | median | p90 | realised | percentile |
|---|---|---|---|---|---|---|
| as drawn | S1 | 61.3% | 70.7% | 75.9% | 77.7% | 93rd |
| | S2 | 70.4% | 81.3% | 89.0% | 91.2% | 97th |
| | S3 | 75.3% | 87.2% | 93.1% | 95.6% | 100th |
| re-centred | S1 | 63.3% | 69.6% | 73.7% | 77.7% | 100th |
| | S2 | 79.0% | 85.6% | 87.7% | 91.2% | 100th |
| | S3 | 82.7% | 89.4% | 94.4% | 95.6% | 100th |

Variance removed against that path's own unhedged run, paired, because the paths differ by a
factor of five in how much variance there is to remove. The 91% that the delta-and-rho hedge takes
out of the realised decade is the 97th percentile of what reorderings of the same decade produce,
and the median is 81%. The realised backtest did not measure the hedge; it measured the hedge on a
favourable ordering of a favourable decade.

**The ranking survives, and the put leg's edge is a third smaller than it looked.** S2 beats S1 on
100% of reorderings in both arms. S3 beats S2 on 100% in the as-drawn arm and 90% in the
re-centred one. But S3's median edge over S2 is 0.041 and 0.033 points of residual standard
deviation against 0.058 on the realised path, for 12.6 and 9.2 points of extra cost. The
conclusion that convexity is worth buying holds; the size of what it buys was overstated by the
one path.

**A hedge that removes most of the variance does not reliably improve the worst day.** On 63% to
73% of reorderings the hedged book's single worst day is better than the unhedged book's, which
means that on 27% to 37% of them it is worse. Variance is a whole-sample measure and a guarantee
is a tail problem, and this is where the two part company.

**The two arms are not just a drift difference.** The re-centred arm, at about 6.4% a year against
the window's own 14.4%, produces tighter distributions, lower residuals and - the point that
matters for reading any of this - far less extrapolation: its median share of rebalances outside
the proxy's design is 26% against 43% in the as-drawn arm and 36% on the realised path. A bull
market walks the contract out of the money and out of the region the proxy was fitted in, so the
arm that is more defensible economically is also the one the model can actually value. Restricting
both arms to the paths that extrapolate no more than the realised path does not change the
conclusion: the medians move to 87.0% and 86.1% for S2, still below the realised 91.2%.

### Model risk, and what a single path cannot measure

The world is whatever history did; the hedge is sized from Greeks that are deliberately wrong in a
named way. The extra residual standard deviation against the calibrated arm, averaged across
episodes:

| Misspecification | S1 | S2 | S3 |
|---|---|---|---|
| flat volatility, no skew | -0.0017 | -0.0024 | -0.0016 |
| long-run volatility 17% | +0.0004 | +0.0002 | -0.0003 |
| long-run volatility 26% | -0.0012 | -0.0011 | -0.0004 |

Two of the nine averages are positive, and twelve of the forty-five episode-by-strategy cells
behind them. Sizing from a flat-volatility model, which has no skew and so understates how much a
guarantee moves in a fall, left a **smaller** residual than sizing from the calibrated model on
this decade.

That is not evidence that misspecification helps. It is evidence that one realised path cannot
measure the cost of misspecification: the sign of the effect is set by how the particular decade
happened to go, and a wrong model that happened to be wrong in the direction the market moved
looks better than a right one. The honest reading is that this experiment bounds the magnitude -
the median cell is a fifth of the calibrated residual and the worst is nine tenths of it - and
says nothing reliable about its sign.

That bound does a second job. The surface the hedge is sized from is one afternoon's chain, 28
September 2026, so a rebalance in 2016 is taken on parameters nobody had then; these three arms
are what says how much that is worth. `docs/limitations.md` lists every input dated after the
decisions it drives, and what is contemporaneous, under the hedging backtest.

### What the rider earned, after paying for the hedge

Every result above is about variance. None of them says whether the product made money, and the
rider charge is levied on the benefit base, so that is the basis everything here is quoted on.
Over any window the hedged book's profit splits exactly:

    net  =  rider charge  +  interest on cash  -  trading and carry  -  the guarantee, net of the hedge

with the last term defined as claims paid plus the change in the guarantee's value less the
hedge's own result. It is an identity rather than an attribution - the pieces are a rearrangement
of the ledger's own columns - so there is no residual for an error to hide in, and the four terms
reconcile to the ledger's profit to within 2e-14 on a benefit base of 153 across all seventy-five
runs. The cash balance is spent on a second check rather than on the decomposition: it has to
equal the flows that are supposed to make it up, and a run where it does not is one where
something reached cash without a column.

Five cohorts, replayed through the same decade on each of three strategies. Basis points of
benefit base a year:

| Cohort | | charge | interest | trading | guarantee, net of hedge | net |
|---|---|---|---|---|---|---|
| at issue | unhedged | 223 | +38 | 0.0 | -230 | **+491** |
| | delta and rho | 223 | -47 | 0.5 | +330 | **-154** |
| | plus puts | 223 | -56 | 145.4 | +213 | **-191** |
| deferring | unhedged | 229 | +40 | 0.0 | -260 | **+529** |
| | delta and rho | 229 | -35 | 0.3 | +217 | **-24** |
| | plus puts | 229 | -42 | 103.6 | +136 | **-53** |
| first income | unhedged | 227 | +40 | 0.0 | -204 | **+471** |
| | delta and rho | 227 | -50 | 0.2 | +276 | **-99** |
| | plus puts | 227 | -53 | 59.1 | +228 | **-113** |
| drawing | unhedged | 202 | +33 | 0.0 | -159 | **+393** |
| | delta and rho | 202 | -42 | 0.2 | +232 | **-72** |
| | plus puts | 202 | -43 | 26.8 | +211 | **-79** |
| late | unhedged | 167 | +14 | 0.0 | +6 | **+174** |
| | delta and rho | 167 | -52 | 0.2 | +367 | **-252** |
| | plus puts | 167 | -52 | 9.8 | +364 | **-259** |

**The cost of a futures-and-swaps hedge is not its trading cost.** The frontier table puts S2's
cost over the decade at 0.08% of account value, and that figure is right for what it measures:
commission, spread and carry. But the hedge lost money against a rising market, those losses were
funded, and the funding ran 35 to 52 basis points of benefit base a year - a sixth to a third of
the rider's entire charge, and between a hundred and three hundred times the trading cost. The
interest line also changes sign with the strategy: the unhedged book accumulates fees and earns
14 to 40 basis points, the hedged books borrow and pay. None of that is a cost in a bear decade;
it is the financing of whatever the hedge's mark happens to do, and on the decade that happened
it is the largest single charge against the product after the guarantee itself.

**The unhedged rider earned 174 to 529 basis points a year and the hedge took all of it back.**
That is not a finding about hedging being a bad idea, for the same reason the realised
decade sat at the 93rd to 100th percentile of its own bootstrap: an equity guarantee is short the
market, the market rose 15.4% a year, and a programme that removes that exposure removes the gain
with it. The symmetric statement is the covid column.

**Under stress the ranking is kept and the level is not.** Net on the delta-and-rho hedge, at an
annual rate, against the decade's own figure:

| Cohort | decade | volmageddon | Q4 2018 | covid | 2022 double |
|---|---|---|---|---|---|
| at issue | -154 | -1,257 | -308 | -3,451 | -530 |
| deferring | -24 | -3,394 | -595 | -2,854 | -315 |
| first income | -99 | -972 | -174 | -5,248 | -314 |
| drawing | -72 | -621 | -135 | +432 | -204 |
| late | -252 | -534 | -129 | -959 | -238 |

An annual rate on an eleven-day window is a rate and not an outcome, and the covid column is
eleven weeks of fee income against a month of crash. The sign is the content: a hedge carrying no
convexity under-covers a convex guarantee in a fall, so the crisis columns are negative even
though the hedge itself gained, and the one positive cell belongs to the cohort whose account had
already fallen far enough that the guarantee had little convexity left to miss.

**The loop does not close on a break-even fee, because at these rates there is not one.** The
valuation solves for the charge that makes the guarantee worth zero at issue. Off the September
2016 curve the solve refuses, and the fee curve says why:

| Charge | 5bp | 50bp | 125bp | 200bp | 300bp | 500bp | 800bp |
|---|---|---|---|---|---|---|---|
| Guarantee, % of premium | 17.4 | 15.1 | 11.9 | 9.6 | 7.5 | 6.5 | 9.9 |

The guarantee gets cheaper as the charge rises, because the charge is revenue; it stops getting
cheaper near 5%, because by then the charge is draining the account fast enough to bring the
claim forward and to end the fee stream before the payments do; and it never reaches zero. There
is no root, not two roots, and a solve that trusted monotonicity across that bracket would have
returned one of the bounds as a price.

That failure is not specific to one cell. Across issue ages 65, 70 and 75 and income starting at
70, 75 and 80, six of the eight contracts admit no charge that prices them. The two that do are
both sold at 75, and both need more than they are paid: 214 basis points with income starting at
once, 151 deferring five years, against the 125 charged.

The at-issue valuation, off the December 2025 curve, has the market risk benefit at zero with an
attribution percentage of 0.786 - fees comfortably covering claims. The difference between the
two is 264 basis points of ten-year rate, 1.57% against 4.21%, and nothing else: same contract,
same engine, same mortality. On a guarantee whose payments run forty years past the valuation date
that is the whole of the economics, and it is the plainest statement in the project of why
Jackson's rate hedge is the half of its programme that doubled over 2023.

A last caution on reading the two halves together. The replay says the rider earned money
unhedged on the path that happened; the valuation says it was underpriced at the moment it was
sold. Those do not contradict each other - one is a realised outcome on a single favourable
ordering of a single favourable decade, the other is a risk-neutral expectation across all of
them - and neither is a statement about what Jackson charged or earned, since the contract here
is a stylised reconstruction and the fee a published rate sheet figure.

## Economic against reported

**Hypothesis 6 holds.** Reported net income under identical positions is more variable than the
economic outcome, on every hedged strategy:

| Strategy | Economic sd | Net income sd | Multiple | OCI sd | Comprehensive sd |
|---|---|---|---|---|---|
| S0 unhedged | 0.656% | 0.688% | 1.05 | 0.127% | 0.572% |
| S1 | 0.310% | 0.338% | 1.09 | 0.127% | 0.248% |
| S2 | 0.195% | 0.216% | 1.11 | 0.127% | 0.161% |
| S3 | 0.137% | 0.153% | 1.11 | 0.127% | 0.142% |
| S5 | 0.158% | 0.182% | 1.15 | 0.127% | 0.126% |

Daily standard deviations as a share of the account value at the start, in per cent. These runs
rebalance weekly, which is what the strategies carry by default, so the economic column sits a
little above the daily-rebalancing figures in the hedging table two sections up rather than
matching them. An earlier version printed the raw fractions with a per-cent sign and was out by a
factor of a hundred against that table.

The multiple *rises* as the hedge gets tighter, which is the mechanism rather than a paradox: the
hedge removes economic variation and the reporting basis carries margins that move with the market
and are not in the hedge's target, so what is removed from the numerator is not removed from the
gap.

The own-credit line is the one no hedge targets. Its daily standard deviation is 0.127% of account
value against 0.153% for S3's hedged net income - five sixths of the entire residual the tightest
hedge in the study leaves - and under ASU 2018-12 its movement goes to other comprehensive income
rather than through net income. Credit spreads widen
when equity markets fall, so that piece is a natural offset to the guarantee - and the rule books
it outside net income, which is what Item 7A means when it says the company does not use hedging
to offset movements in its US GAAP liabilities and that this has produced net income volatility.

### Against the filed series

The model's own hedge-against-liability offset is near-tautological and is here as a counterfactual
rather than as a result.

| Basis | Periods | Correlation | p | Offset ratio |
|---|---|---|---|---|
| filed, quarterly | 18 | +0.006 | 0.98 | 1.24 |
| filed, annual | 4 | -0.821 | 0.25 | 1.12 |
| model, quarterly | 41 | -0.986 | 0.00 | 1.25 |
| model, annual | 11 | -0.965 | 0.00 | 1.23 |

Eighteen quarters of filed data show a correlation of +0.006 between the liability movement and
the hedging result. At that sample size the smallest correlation detectable at five per cent is
about 0.47, so the filed series cannot rule out a substantial relationship; what it can say is
that nothing large and negative is visible, which is what the disclosure itself says in words.

## Falsification tests, restated against results

| Test stated in the research design | Outcome |
|---|---|
| A sign that comes out backwards on any disclosed shock | Not observed; 18 of 18 |
| A convexity ratio on the wrong side of two | Not observed; 1.88 up and 2.13 down |
| A model sensitivity *below* the disclosed one | Not observed; the model is above on all 18 |
| A delta and rho hedge that fails to reduce variance | Not observed; 74% to 99% across the episodes |
| A residual uncorrelated with implied volatility | **Partly observed.** Adding convexity removes most of the residual that delta and rho leave, so the residual is better described as the absence of a convexity instrument than as a volatility exposure |
| Reported net income no more volatile than the economic outcome | Not observed; 1.05 to 1.15 |
| The gap closable by behaviour alone | **Observed.** Behaviour closes the rate half and not the equity half; the equity half is moneyness |

Two of the seven came out against the stated prediction. Both are reported above with the
measurement behind them, and neither was discovered by looking for it - the behaviour divergence
came out of a sweep run to confirm the opposite, and the residual finding came out of adding an
instrument class the original design did not have.
