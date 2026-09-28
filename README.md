# Hedging the guarantee

Rebuilding a variable-annuity living-benefit valuation and dynamic hedging model from
public filings and free data, and testing it against the sensitivities Jackson Financial
discloses.

Jackson runs the largest standalone variable annuity book in the United States and says a
lot about how it manages the risk in it. Item 1 of the 10-K describes a core dynamic hedging
programme that offsets equity and interest rate movements in the *economic* liability
associated with guaranteed living benefits. Item 7A publishes the fair-value impact of a 10%
equity move and a parallel rate shift. Note 6 describes the valuation method down to the
volatility term structure and the treatment of the company's own credit. What it does not
publish is the model.

This builds one, prices a representative GMWB for Life rider under the risk-neutral measure,
and asks three questions. Do the sensitivities reproduce what Jackson discloses? What does
weekly delta and rho hedging actually do to the volatility of the position? And why do
reported earnings keep moving when the economic hedge is working?

## Results

**Sign and shape hold. Scale does not, for one identifiable reason.**

All eighteen disclosed shock sensitivities come out with the right sign, across four
balance-sheet dates and both directions of both shocks. The convexity in Jackson's table
comes out close: where the filings disclose both ±50bp and ±100bp, the ratio of the two is
1.86 up and 2.13 down, against 1.89 and 2.11 from the model. The decline in sensitivity per
dollar of account value from 2022 to 2025, as the book moved out of the money, comes out at
0.59 of its starting level against 0.58 disclosed.

The levels sit above the disclosure by a factor of 2.14 to 2.85, median 2.39, and that
factor is near-constant across six shocks and four dates. A near-constant multiple points at
one structural assumption rather than a pile of errors. Two things account for it. A quarter
of Jackson's variable annuity account value carries no withdrawal guarantee at all. And the
base case assumes the contract holder draws the full guaranteed amount every year and never
surrenders, which is the benchmark case in Bauer, Kling and Russ (2008) precisely because it
is the most expensive one. At 90% utilisation and 4% annual lapse, every shock lands within
33% of the disclosed figure.

**A delta and rho hedge removes 64% of the variance. Volatility is what is left.**

One policy, issued September 2016, hedged weekly for 521 weeks on Jackson's disclosed fund
mix.

| Hedge | Weekly std | Variance removed | Worst week | Costs over ten years |
|---|---|---|---|---|
| Unhedged | 1,705 | | -11,500 | |
| Delta | 1,253 | 46.0% | -8,438 | 32 |
| Delta and rho | 1,018 | **64.4%** | -8,258 | 48 |
| Delta, rho and volatility | 328 | 96.3% | -1,061 | 162 |

All figures in dollars on a $100,000 policy. Transaction costs across ten years of weekly
rebalancing come to $48, under 0.05% of premium, so the basis-risk against cost trade-off
that usually sets rebalance frequency barely binds here.

February and March 2020 makes the point better than the summary statistics. The unhedged
guarantee lost 30.6% of premium in seven weeks. Delta and rho hedging recovered 40% of that
and left an 18.5% loss standing. Adding a volatility leg cut the residual to 1.5%. For this
liability that episode was a volatility event more than a level event, and futures and swaps
were never going to catch it.

**The annual step-up can make the insurer long equity.**

A test asserting that equity exposure is always short failed on 17 of 522 weeks, and the
failures were the model being right. The benefit base resets to the contract value on the
anniversary, so with the contract value above the benefit base the index level on that one
date fixes the guaranteed income for life. Claims and fees both then respond positively to
the index and the net exposure is their difference: +25,234 against +23,595 at the state
where it first appeared, leaving +1,638. The rate environment decides the sign, because at
low rates the account depletes on every path and the ratchet is certain to bite.

**Reported earnings are more volatile than economic ones under the same hedge, and the
reason is an accounting boundary.**

Sized on the economic basis, the hedge removes 64.4% of the variance of the economic
liability and 60.7% of the variance that reaches net income. Reported net income is 1.05
times as volatile as the economic outcome, positive in all eleven calendar years of the
sample.

The interesting part is that reported *comprehensive* income is less volatile than either.
Credit spreads widen when equity markets fall, which reduces the own-credit adjusted
liability at the same moment the market move is increasing it, so the own-credit adjustment
is a partial natural hedge, averaging -$6,610 and reaching -$20,842. The market risk benefit
rules report that movement in other comprehensive income. Net income therefore gets the full
undamped move while comprehensive income gets the damped one. The offset exists and the
accounting routes it away from the line most people read.

## The thing to distrust

The hedging backtest revalues the liability with the same model that produced the hedge
ratios. Any factor the model represents and the hedge covers is removed nearly completely,
limited only by convexity between rebalances, which is why the volatility leg reaches 96%.
What these numbers measure is the cost of hedging discretely with imperfect instruments:
gamma between rebalances, one swap tenor against a parallel-shift rho, one index against a
blended sub-account, and the spread paid to trade. What they cannot measure is model error,
and on a forty-year guarantee with assumed policyholder behaviour that is the larger risk.
Read 64.4% as an upper bound on what a real programme achieves.

`docs/limitations.md` is the full list and is worth reading before the results.

## Running it

Python 3.10 or later. NumPy, pandas, Matplotlib and PyYAML, nothing else. The test suite
needs nothing beyond those.

```
make data        # validate every input, bootstrap the curve history
make valuation   # at-issue valuation, cash flows, robustness, convergence
make greeks      # Greeks and the moneyness profile
make validate    # disclosed shocks, in-force comparison, vintage portfolio, behaviour sweep
make backtest    # weekly hedging backtest, ten years        (~10 minutes)
make accounting  # economic against reported
make test        # the test suite
make all
```

Or directly, for example `python -m scripts.run_valuation`. Everything runs from the
repository root. Tables land in `reports/tables/` as both CSV and fixed-width text, figures
in `reports/figures/`.

Run `make data` first. It fails rather than warns on a par curve that will not bootstrap, a
mortality table with a hole above age 40, a Period table that implies shorter life than the
Basic table, or a disclosed figure that disagrees between two filings.

Every assumption lives in `config/params.yaml`, with the filing or the reasoning behind each
one written next to it.

## What is in here

```
config/params.yaml          every assumption, with its source
data/raw/                   FRED panel, SOA mortality tables, figures read out of the filings
data/processed/             built by make data; committed outputs are reproducible from raw
gmwb/
  curves.py                 par yields to zero rates, and shocks applied the right way round
  volatility.py             mean-reverting forward variance fitted to the implied index
  mortality.py              2012 IAM with Projection Scale G2, generational
  contract.py               the representative contract, with the prospectus language behind it
  engine.py                 risk-neutral Monte Carlo and the attributed-fee method
  sensitivities.py          Greeks by bump and revalue, and disclosed-shock repricing
  hedging.py                policy roll-forward, hedge sizing, the profit ledger
  accounting.py             economic against reported, and the OCI split
  market.py                 one date to model inputs, with the calendar problems handled
  session.py, figures.py, paths.py, config.py
scripts/                    one script per analysis, each runnable on its own
tests/                      run with make test; no test framework required
docs/
  research_design.md        question, hypotheses, what would falsify them
  data_sources.md           every series and every gotcha in it
  methodology.md            the equations, and why each choice was made
  validation.md             all results, including where the model misses
  limitations.md            what this does not do
  literature.md             sources, separated into retrieved and referenced
notebooks/                  the results end to end, in reading order
```

## Two things that took a rewrite

Worth flagging because both were wrong in a way that looked right.

**The volatility term structure.** The first version held the 3-month implied level flat to
five years, which reads as a fair interpretation of Jackson's "implied volatility for
durations up to 5 years". Under stress it is an arbitrage: in March 2020 the 3-month index
reached roughly 70 against a long-run level near 19, and a linearly declining spot volatility
through the grading window put total variance to seven years below total variance to five.
It also propagated a fifty-point spike in a three-month index undamped across the whole front
of the surface, which produced a vega about twice what it should be and then dominated the
hedging residual. The replacement specifies the shape on forward variance with mean
reversion, which keeps forward variance positive by construction and lets a short-dated move
decay with maturity.

**A citation.** A paper carried in memory as Kling, Ruez and Russ (2011) in *ASTIN Bulletin*
on stochastic volatility and hedge efficiency does not exist under those details. Checking
the bibliography against Crossref turned up the real article: same authors, 2014, *European
Actuarial Journal*, and about policyholder behaviour rather than stochastic volatility, which
makes it more relevant to this project than the version being cited.

## Scope

One stylised contract, stated as such throughout. Not a valuation of Jackson Financial, not
a valuation of its liabilities, and not investment advice. Where the model and the disclosure
differ, the presumption is that the model is what is wrong.
