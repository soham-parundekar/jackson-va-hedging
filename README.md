# Hedging the guarantee

Rebuilding a variable-annuity living-benefit valuation and dynamic hedging model from public
filings and free data, and testing it against the sensitivities Jackson Financial discloses.

Jackson runs the largest standalone variable annuity book in the United States and says a lot
about how it manages the risk in it. Item 1 of the 10-K describes a core dynamic hedging
programme that offsets equity and interest rate movements in the *economic* liability associated
with guaranteed living benefits. Item 7A publishes the fair-value impact of a 10% equity move and
a parallel rate shift. Note 6 describes the valuation method down to the volatility term structure
and the treatment of the company's own credit. What it does not publish is the model.

This builds one, prices a representative GMWB for Life rider under the risk-neutral measure, and
asks four questions. Do the sensitivities reproduce what Jackson discloses? What does a daily
delta, rho and convexity hedge actually do to the volatility of the position? What does the hedge
leave behind in statutory capital? And why do reported earnings keep moving when the economic
hedge is working?

## Results

**Sign and shape hold. Scale does not, and the gap has two separate causes rather than one.**

All eighteen disclosed shock sensitivities come out with the right sign, across four
balance-sheet dates and both directions of both shocks. The convexity in Jackson's table comes out
close: where the filings disclose both ±50bp and ±100bp, the ratio of the two is 1.855 up and
2.128 down, against 1.882 and 2.128 from the model. The decline in sensitivity per dollar of
account value from 2022 to 2025, as the book moved out of the money, comes out at 0.5795 of its
starting level against 0.5794 disclosed.

The levels sit about three times the disclosure, which the research design predicted. What it did
not predict is that behaviour would not close the gap. Lowering utilisation takes duration out of
the guarantee and collapses the rate sensitivity with it, while the benefit base is still there
whatever the owner draws, so the equity sensitivity barely moves: across the sweep the rate
multiple falls 2.09 to 0.47 and the equity multiple only 2.10 to 1.59. The equity half is
moneyness instead, and the model says so independently - locating the disclosed figure on the
model's own curve implies a benefit base over account value of 0.94 falling to 0.82 across the
four years, with the up and down shocks implying the same ratio without being made to.

**A delta and rho hedge removes most of the daily variation. Convexity removes most of the rest,
and costs two hundred and seventy times as much to trade.**

One policy, three years in force, rolled along realised history from September 2016 to September
2026, on Jackson's disclosed fund mix. The table is the daily-rebalancing rows of the frequency
sweep; the crisis replays below rebalance weekly, which is what the strategies carry by default.

| Hedge | Residual daily sd | Trading cost over the decade |
|---|---|---|
| Unhedged | 0.656% | - |
| Futures | 0.304% | 0.05% |
| Plus a receive-fixed swap | 0.188% | 0.08% |
| Plus listed puts | 0.139% | 21.67% |

All figures as a share of account value. The rate leg costs almost nothing to trade and takes out
a third of what the equity leg left. The option leg takes out another quarter at two hundred and
seventy times the cost, and the put design sweep says most of that cost is in the tenor rather
than the protection: a quarter-year put at 0.90 of spot costs 7.7% of account value over the
decade against 21.2% for a one-year put at the same strike, for almost the same residual.

Trading cost is commission, spread and carry, and on the first two rows it is not what the hedge
cost. Funding the losses the equity leg realised against a rising market ran a hundred to three
hundred times larger, and the section below on what the rider earned puts a number on it.

February and March 2020 makes the point better than the summary statistics. The unhedged guarantee
lost 22.0% of account value in twenty-four trading days. Futures alone recovered two thirds of it,
futures and swaps left 2.7%, and the option leg turned it into a 1.9% gain. Volmageddon is the
opposite case: an 8.8% index move with volatility doubling, where futures and swaps remove 74.5%
of the variance against 94% in covid, because instruments carrying no vega were never going to
catch a volatility event.

**The decade that happened flattered the hedge, and the bootstrap says by how much.**

Resampling the decade's equity days into thirty other orderings puts the realised result at the
93rd to 100th percentile of the distribution on every strategy. The delta-and-rho hedge removes
91% of the daily variance on the path that happened and a median of 81% across reorderings of the
same days. The ranking survives - the richer hedge is tighter on 90 to 100% of reorderings - but
the put leg's edge over futures-and-swaps is about a third smaller than the single path suggested.
And on 27 to 37% of reorderings the hedged book's worst single day is *worse* than the unhedged
book's, which is where variance and tail risk part company.

**Jackson's own filings say its rate hedge and its equity hedge moved in opposite directions, and
they date when.**

Item 7A publishes the derivative portfolio's response to the same shocks as the guarantee's, on
the same dates, so the offset ratio is observable and nothing in the model is fitted to it. Across
four year-ends on one measurement basis:

| | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|
| rates, down / up | 37% / 43% | 86% / 78% | 81% / 86% | 79% / 86% |
| equity, down / up | 104% / 77% | 110% / 81% | 49% / 50% | 9% / -9% |

The rate share roughly doubled over 2023 and has sat near four fifths since. The equity share was
a full hedge on the downside through 2023 and four fifths of one on the upside, and halved in
2024, the first full year after Brooke Re. A full
hedge of the model's own liability at those same dates covers 93 to 109% of the rate shock and 90
to 113% of the equity shock, so convexity buys about eight points either side of one and not the
shortfall on show.

2024-12-31 is disclosed at both a 50bp and a 100bp shift by two different filings, and the
disclosed ratio moves 1 to 3 points between them where a full hedge moves 4 - so the gap is a
sizing choice, not curvature.

**The one line that cannot be read at face value is the index-linked book's.** Its disclosed equity
sensitivity went from $4m to $1,321m over 2025, a factor of 330, on a balance that grew by three
quarters.
Modelling that book directly - six-year segments, a 20% buffer, caps swept, cohorts rolled along
realised index history and weighted by filed issuance - says a book of Jackson's own disclosed size
should have absorbed 3.5%, 10.5%, 27% and 50% of the guarantee's equity move across the four
year-ends. The filings report 0.1%, 0.2%, 0.5% and 79%. The model's whole reachable range is a
factor of 2.4; nothing an index-linked book can do moves a sensitivity by 330. The model's level
agrees with the filed one throughout (15.9 to 24.7% of account value against 10.9 to 29.8%), which
is what makes its exposure worth quoting. The likeliest reading is a line carried net of its hedges
until FY2025 and gross afterwards - which means the 2024 and 2025 combined-basis figures are not
the same measurement, and the market-risk-benefit column is the one to read.

**What the hedge costs is not what it costs to trade, and the fee the rider needed does not
exist at the rates it was sold into.**

Splitting the hedged book's profit into the rider charge, interest on cash, trading cost and the
part of the guarantee the hedge did not recover - an identity, so there is no residual - says
something the variance tables cannot. The delta-and-rho hedge's trading cost over the decade is
0.08% of account value. Funding its losses against a rising market cost 35 to 52 basis points of
benefit base a year, a sixth to a third of the entire rider charge and a hundred to three hundred
times the trading cost. In a falling decade that line is a credit rather than a charge; the point
is that it is the large one either way and a cost column that reports commission and spread
leaves it out.

Across five cohorts the unhedged rider earned 174 to 529 basis points a year on the decade that
happened and every hedged arm gave it back, which is what hedging a short-equity position through
a 15.4%-a-year market does.

The break-even fee was meant to close the loop and it closes it the other way. Off the September
2016 curve there is no charge that makes the guarantee worth zero: its value falls from 17.4% of
premium at 5bp to 6.5% at 500bp, then turns back up to 9.9% at 800bp as the charge starts
exhausting the account and ending the fee stream before the payments end. Six of eight contracts
across issue ages 65 to 75 admit no break-even charge at all; the two that do are both sold at 75
and need 151 to 214 basis points against the 125 charged. The same engine on the December 2025
curve prices the same contract at a market risk benefit of zero. The difference is 264 basis
points of ten-year rate and nothing else.

**The economic hedge is not the capital hedge, and it is not the earnings hedge.**

The statutory requirement is a real-world tail measure with a cash surrender value floor under it.
CTE(70) runs 13.6% of premium at a 2% equity risk premium down to 7.5% at 6%, and the hedge
removes far less of the statutory variation than of the economic variation, because the floor
binds in the states a delta hedge is busy in.

Reported net income under identical positions is 9 to 15% more variable than the economic outcome,
and the multiple *rises* as the hedge gets tighter. The own-credit adjustment, which no hedge
targets, has a daily standard deviation of 0.127% of account value against 0.153% for the whole of
the best-hedged strategy's net income - and under ASU 2018-12 it is reported outside net income.
Credit spreads widen when equity markets fall, so that piece is a natural offset to the guarantee
that the accounting boundary turns into a mismatch.

Eighteen quarters of Jackson's own filed XBRL show a correlation of +0.006 between the liability
movement and the hedging result. At that sample size the smallest correlation detectable at five
per cent is about 0.47, so the filed series cannot rule much out - which is itself worth knowing
before reading anyone's offset ratio.

## Running it

```
make data        # validate the committed inputs, bootstrap the curve history
make calibrate   # fit the market state, and check the surface against the volatility indices
make valuation   # at-issue valuation, cash flows, robustness, convergence
make greeks      # Greeks on paired paths, and the moneyness profile
make validate    # disclosed shocks, in-force comparison, vintage portfolio, behaviour sweep
make offset      # the hedge book against the disclosed derivative sensitivities
make netting     # what the index-linked book absorbs before any hedge   (~10 minutes)
make proxy       # the regression proxy against nested simulation          (~6 minutes)
make convexity   # the nested curvature surface the option leg is sized from (~8 minutes)
make hedge       # crisis replays, the cost frontier, the put sweep, model risk (~15 minutes)
make macro       # what the tail put spread buys, swept over size and strikes (~10 minutes)
make statutory   # the real-world requirement and the surrender value floor
make economics   # fee income against hedge cost and breakage, by cohort     (~15 minutes)
make reporting   # economic against reported earnings, and the own-credit split (~15 minutes)
make replica     # the model's offset against the filed XBRL series, from reporting's table
make real-world  # the hedge over bootstrap reorderings of the decade, two arms (~1 hour)
make figures     # redraw every figure from the tables, in seconds
make test        # the test suite
make all
```

Or directly, for example `python -m scripts.run_valuation`. Everything runs from the repository
root. Tables land in `reports/tables/` and figures in `reports/figures/`.

Run `make data` first. It fails rather than warns on a par curve that will not bootstrap, a Period
mortality table that implies shorter life than the Basic table, or a disclosed figure that two
filings disagree on.

`make calibrate` and `make convexity` write files that are gitignored, so a fresh clone has to
build them before anything downstream will run.

`reports/` is committed, which is what makes the reproduction checkable rather than promised:
`make clean && make all && git diff --stat reports` regenerates every table and figure from the
committed inputs and shows anything that came back different. `make clean` deletes tracked files
to do it, and `git checkout reports` puts them back.

## Where the assumptions live

Each one is a constant in the module that uses it, with the reasoning next to it, rather than in a
parameter file that would be a second place to keep true.

| Assumption | Where |
|---|---|
| Contract terms, withdrawal bands, charges | `data/raw/jackson_rider_terms.csv`, read by `vahedge/liability/terms.py` |
| Issue age, deferral, fund expense, premium | `scripts/run_valuation.py` |
| Market parameters | `data/processed/market_calibration.json`, written by `scripts/run_calibration.py` |
| Path counts and seeds | `vahedge/valuation/engine.py` |
| Hedge instruments, costs, roll rules | `vahedge/hedge/strategies.py` and `vahedge/hedge/instruments.py` |
| Equity risk premium sweep | `scripts/run_statutory.py` |
| Own-credit spread proxy | `vahedge/capital/reporting.py` |

## What is in here

```
data/raw/                   FRED panel, SPX option chain, SOA mortality, figures read out of the filings
data/processed/             built by make data and make calibrate; gitignored, reproducible from raw
vahedge/
  market/                   curve bootstrap and NSS fit, Heston, Hull-White, the option chain,
                            the simulator, and the realised-history scenarios
  liability/                the rate sheet, the contract recursion, cohorts, mortality, behaviour
  valuation/                the engine, Greeks, the regression proxy, nested valuation, break-even
  hedge/                    instruments, sizing, strategies, the daily simulator, attribution
  capital/                  the statutory lens and the reporting-basis lens
  report/                   figures, drawn from the committed tables
scripts/                    one script per analysis, each runnable on its own
tests/                      run with make test; no test framework required
docs/
  research_design.md        question, hypotheses, what would falsify them
  data_sources.md           every series and every gotcha in it
  methodology.md            the equations, and why each choice was made
  validation.md             all results, including the two hypotheses that did not survive
  limitations.md            what this does not do
  literature.md             sources, separated into retrieved and referenced
references/                 the sources themselves, with a register saying what each one is for
notebooks/                  the results end to end, in reading order
```

## Four things that took a rewrite

Worth flagging because all four were wrong in a way that looked right.

**Two models of the same liability.** The valuation, the Greeks and every disclosure validation
were produced by an earlier single-factor implementation - lognormal equity on a deterministic
volatility curve, deterministic rates, no death benefit - while the hedging, capital and reporting
work ran on the Heston and Hull-White cohort model. Thirty of the committed tables came from the
first and the conclusions came from the second, which made the validation evidence a validation of
a model the conclusions did not use. The earlier package is gone and everything runs on one model.

**A denominator that was not there.** A figure labelled "variance remaining, share of unhedged"
divided by whichever row of the frontier table was largest, because the unhedged run was not in
that table at all. It was a share of the delta-only hedge. The frontier now runs the unhedged book
and the figure plots the residual itself with the unhedged bar in the picture. It was caught by
trying to quote the figure's own numbers in the notebook, which is the argument for building
figures from the committed tables rather than from live model objects.

**A truncation test comparing different worlds.** The simulator draws its normals in one array
whose width follows the horizon, so a forty-year run and a fifty-year run at the same seed are
different simulations rather than a prefix and its extension. The test that was meant to show the
projection horizon does not matter was reporting ninety dollars of Monte Carlo difference as a
truncation effect, and passing for the wrong reason. On common draws, cutting at age 115 costs
less than a dollar and at 105 costs thirty-six.

**A solve that returned its own bracket.** The break-even fee was solved by bisection between 5
and 800 basis points on the stated grounds that the guarantee gets cheaper the more it is charged.
It does, up to about 5%, and then the charge starts exhausting the account faster than it raises
revenue and the curve turns back up. On the 2016 curve it never crosses zero at all, so the solve
had no root to find - and it reported the top of its bracket as the fee, with a convergence flag
beside it that nothing was reading. The first table built on it said the rider was worth 800 basis
points. It now returns a NaN and a reason, and a test holds it to that in both directions.

## Scope

One stylised book, stated as such throughout. Not a valuation of Jackson Financial, not a
valuation of its liabilities, and not investment advice. Where the model and the disclosure differ,
the presumption is that the model is what is wrong.
