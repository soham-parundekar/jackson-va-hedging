# Limitations

Organised by what the limitation threatens. Everything here is either measured somewhere in
`reports/tables/` or stated as an assumption with its direction of bias; nothing is listed as a
caveat in order to be dismissed.

## The book

**Five contracts standing in for millions.** Valuing Jackson's actual block needs attained age,
duration, benefit base and sub-account allocation at policy level, none of which is disclosed.
The vintage portfolio in `scripts/run_portfolio_validation.py` reaches a weighted attained age of
70.9 against the disclosed 70 and a blended withdrawal rate of 5.82%, and two of its five vintages
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
grid the equity multiple falls 2.09 to 1.59 while the rate multiple falls 2.16 to 0.49, because
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

**The volatility surface is one afternoon's chain, held along the whole replay.** It is the SPX
chain of 28 September 2026, which is the only snapshot free data gives, and there is no free
historical option data to move it with. So what moves with the date in the in-force comparison
and the backtest is the curve and the observable instantaneous variance; the skew, the speed of
mean reversion and the long-run level are held. The in-force comparison is run with that chain's
surface attached to the rate environment of each disclosed year, including years that ended
before the chain was quoted.

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

Its value is accurate - R-squared above 0.995 in range and a root mean square error of two to
seven tenths of a per cent of premium against nested simulation. That figure read as thousandths
of a per cent here and in the notebook until an audit divided the committed column by itself. The
column was then called `rmse_pct_of_account` and held a share of premium; it is now
`rmse_share_of_premium`, which is the same number under a name that cannot be misread.

Scaled columns in this project are shares rather than percentages, and the name says the
denominator. The one table that broke that - the index-linked cohort book, where the embedded
derivative was stored at a hundred times its neighbours under the same suffix - now stores the
share like the rest, and the per cent is made where the figure is reported against a filed one.

The proxy's **delta is 36% off in the first policy year** and
21% in the second, 10% by year 5 and 5% by year 14, and then deteriorates again to 23% by year 25
as the delta itself shrinks and the same absolute error becomes a larger share of it. The backtest
covers policy years 3 to 13, which is the best part of that range, but nothing in the project
reads a delta off policy year 1. Its **second derivative is not usable**, which is why the option
leg is sized from a tabulated nested surface instead.

**Which paths become nodes is not stable across numerical environments.** The nested surface
takes its nodes by sorting the fitting paths on moneyness and picking evenly spaced ranks. Two of
those ranks sit on point masses - an exhausted contract is at exactly zero and a contract that
stepped up is at exactly the post-charge ceiling - so a great many paths are tied there, and which
one lands at a given rank is decided by the last bits of the sort key. Rerun under a different
numpy and scipy, where the calibration itself moves in its twelfth significant figure, 60 of the
548 nodes come back as a different path at the same moneyness: the same rank, a different variance
and ten-year rate, and a value up to 262 of its own Monte Carlo standard errors away.

The curvature survives it, which is the point. A swapped path changes the level of the three
prices in a node's ladder together, and a second difference does not see a level. Across all 548
nodes the gamma moves by at most 1.6 of the error implied for it and by 0.000 at the median, so
`data/processed/gamma_surface.csv` - the only thing downstream reads - is reproducible to well
inside its own precision. What is not reproducible is the node table itself, which means the
variance-spread sensitivity read off those nodes is a figure about one draw of them rather than
about the surface.

**Extrapolation is the condition on every hedging result.** The realised path asks the proxy for a
state outside its design on 36% of rebalance dates. That is reported with every run rather than
checked once, because a hedging conclusion built on mostly extrapolated Greeks is not a conclusion.

**The nested standard checks one of the two fits, over part of its range.** The proxy carries two
families: a post-event fit for the state just after an anniversary's charges, withdrawal and
step-up, and a pre-event fit for the state just before them - which is also the state any
Wednesday in the middle of a year resembles, and therefore the one a hedge is sized off. The
nested comparison runs on recorded post-event states, and because the step-up resets the benefit
base to the contract value at every anniversary, a post-event contract value can never exceed its
base: the moneyness at those nodes runs 0.0 to 1.00 and no higher, by construction rather than by
sampling. So the family the backtest evaluates most of the time, and the whole of the range above
the money, has no independent check behind it.

What it has instead is an identity: the pre-event value less the post-event value has to equal
that year's own net cash flow, and `tests/test_lsmc.py` holds both fits to it on their design
points. That is a real constraint and it is not the same thing. Two fits that are jointly wrong by
the same amount satisfy it exactly.

Closing the gap is not an addition to the existing experiment. A nested valuation starts a
projection from an anniversary, and a pre-event state is an instant before one, so pricing from
there means starting the recursion with that anniversary's events pending - which the projection
cannot express, since it applies a year of growth before any event. The honest statement is that
the pre-event fit inherits whatever confidence the post-event comparison earns, through an
identity that constrains their difference and not their level.

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

**The two tables carrying a drawdown predate the convention the code now uses.** A drawdown is
measured from a high-water mark, and that mark starts at zero: before the first day the programme
has earned nothing, so the cost of striking the book belongs inside the drawdown. Running the
maximum straight over the cumulative profit instead makes the first day the peak whenever it is a
loss, which for every option strategy it is, and that is what `summarise` did while
`hedge_crisis_replays.csv` and `real_world_paths.csv` were written. `worst_day_pct` has always
included the first day, so the two sat in the same row on different conventions.

The code measures from zero now and the two tables do not yet reflect it. The size is bounded and
one-sided - a drawdown can only widen - and on the crisis replays it moves nine of the
twenty-eight rows, by between 0.0008 and 0.127 points of account value; the largest is S6 through
volmageddon, from 0.574% to 0.701%, which is 22 per cent of the figure. The other nineteen rows do
not move at all, because their high-water mark fell later than the first day. Nothing in these
documents draws a conclusion from the column.

Regenerating the two tables is what would close it, and the environment that wrote them is no
longer available to do it in: the machine those tables came from does not get through the
calibration stage any more, so nothing downstream of it runs there either. A rerun anywhere else
moves every column of the table rather than this one, and the two tables differ in how much that
matters. On the crisis replays the rerun shifts `worst_drawdown_pct` on all twenty-eight rows but
by at most 3.1 per cent of the figure, against a convention change worth 22, so there the
correction is the larger effect of the two. On the real-world paths it is the other way round -
the rerun moves 121 rows wider and 123 narrower, which swamps a change that can only widen. So
neither is regenerated: fixing one table would leave the hedge set with two numerical provenances
to explain instead of one stale column. **Reproducibility** in `docs/data_sources.md` carries the
per-stage sizes.

The same convention is now computed in one place rather than two. The macro frontier read its
drawdowns off the capital levels, whose first row is already after the book was struck, and the
reading happens to agree with the corrected one on every row of its grid: the hedge is ahead of
its opening level well before the trough in covid, so both conventions find the same peak. On a
window where it never gets ahead they would not, which is why the duplicate was removed rather
than left alone.

**Four model inputs are dated after the decisions they drive, and the list is the whole of it.**
A desk rebalancing in September 2016 could not have had any of them:

| Input | Dated | Reaches back to |
|---|---|---|
| Heston parameters, and the proxy surface fitted under them | the SPX chain of 28 Sep 2026 | the whole replay, from Sep 2016 |
| Hull-White mean reversion and volatility | fitted on the 3-month bill, Sep 1981 to Sep 2026 | the whole replay |
| Mortality improvement under Scale G2 | projected to 2026 and held | the whole replay |
| Sub-account mix at 0.83 equity, cash surrender value at 98.0% of account value | the FY2025 10-K | the whole replay |

What is contemporaneous: the index level, the zero curve bootstrapped from that day's own par
quotes, the instantaneous variance read off the three-month volatility index, the financing rate,
and the position being rebalanced, which comes from the prior day's ledger and nothing else.
`tests/test_hedge.py` pins that last one rather than asserting it.

Three of the four are unavoidable on free data and the fourth is a choice. Nothing free carries a
historical option surface, the mean reversion is not identified by a decade, and a 2016 fund mix
is not disclosed; holding the mortality basis at 2026 is a decision, worth 0.92 years of expected
lifetime at age 70 across the 2012-to-2025 improvement, which is the smallest of the four.

The size of the first one is measured rather than argued, in **Model risk** in
`docs/validation.md`: re-sizing the same hedge from deliberately wrong Greeks - no skew at all, a
long-run level at 17% and at 26% against the calibrated 21.7% - moves the residual standard
deviation by a fifth of the calibrated residual at the median cell and nine tenths at the worst.
That brackets what a desk holding a different surface would have suffered, and it is the reason
the hedging results are reported as a ranking of instrument sets rather than as a profit a desk
could have earned. Read as a ranking they are not a forecast and the dating does not flatter them;
read as a P&L they would be, and the project does not read them that way anywhere.

Only the Heston parameters are dated 2026 in that first row. The curve the proxy is fitted off
is the Treasury curve of 26 September 2016, replaced deliberately in
`scripts/run_hedge_experiments.py`: a proxy fitted off December 2025's curve has a design
distribution for the ten-year zero of roughly three to six and a half per cent, while the decade
being replayed spent most of its time between half and three, so seventy per cent of the
backtest's days would be valued by a fit that had never simulated a rate near them. What is held
at 2026 is the shape of the surface and the speed of mean reversion, not the level and not the
rate environment.

There is no survivorship question here, and the selection question has one real answer rather
than none. No survivorship: one issuer named in advance, every disclosed shock it published used
rather than a subset, and the replay window set by where the free index history starts rather
than by where the hedge looks good.

The selection that does exist is the option leg's two parameters. S3's put tenor and strike were
picked off the sweep in E3 - that is, off the same replay the strategy is then scored on - and
`strategies.py` says so where they are set. What bounds it is that the whole grid is published
and none of it is hidden: across the nine cells of the three-by-three sweep the residual standard
deviation runs 0.135% to 0.163% of account value and the cost 5.2% to 21.5%, every cell sits on
the frontier once the crisis outcome is an axis, and no cell dominates another. The conclusion the
project draws from this is a ranking of instrument sets, and the ranking does not depend on the
choice: the *worst* cell of the put grid still leaves a tighter residual than the *best* cell of
S2 over any rebalance frequency, 0.163% against 0.188%. A reader who prefers a different tenor or
strike can read their own cell out of `reports/tables/hedge_put_design.csv` and the ordering is
unchanged.

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
moves the equity-down figure by 0.07 to 0.23 points of account value, between a quarter and a
twelfth of the disclosed figure for the same date and between a ninth and a twenty-fifth of the gap
being explained. The narrow end of that range is 2025, where the two bases fall either side of the
anniversary at which the contract picks up the rate sheet's top withdrawal band, so the two
valuations differ in the guaranteed percentage as well as in the duration - the one date where the
spread is not purely a timing effect.

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
