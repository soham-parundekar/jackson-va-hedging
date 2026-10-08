# Methodology

## Notation

$AV_t$ contract value at policy anniversary $t$, $B_t$ the benefit base (the Guaranteed
Withdrawal Balance in the contract's language), $g$ the guaranteed annual withdrawal
percentage, $\phi$ the rider charge as a percentage of the benefit base, $m$ the continuous
proportional drag on the contract value, $r_t$ the short rate, $v_t$ the instantaneous
variance, ${}_tp_x$ the probability that a life aged $x$ survives $t$ years, $D(t)$ the
stochastic discount factor.

## Contract value

Between anniversaries the contract value grows at the sub-account's total return less a
proportional drag:

$$\frac{dAV_t}{AV_t} = (r_t - m)\,dt + \sqrt{v_t}\,dW_t^{Q}$$

Under the risk-neutral measure any traded portfolio earns the short rate, so the drift is
$r_t$ and not a real-world expected return. That distinction is the whole difference between
a valuation and a financial planning projection, and it is why the model says a 5.95%
lifetime draw exhausts the account on 78% of paths by year twenty. At a real-world equity
drift it would not, which is what the statutory lens measures separately.

$m$ is the base contract charge of 1.31% plus fund expenses of 0.95%. Both are quoted against
average daily value, so both are continuous proportional drags.

### Annual stepping in the contract, sub-annual stepping in the market

Nothing happens to the contract between anniversaries except growth and the drag, and every
path-dependent event in it lands on an anniversary: the bonus, the step-up, the withdrawal,
the point of exhaustion, the switch from fee income to claims. So the contract recursion steps
a policy year at a time, over an array of cohorts by paths.

What it steps over is not a lognormal increment. The market simulation runs twenty-four steps
a policy year, because the variance process has to be stepped to be simulated at all, and
hands the recursion the realised annual growth factor, the realised discount factor and the
state at each anniversary. The split is deliberate: the expensive object is the market
simulation and it is shared by every cohort, while the contract recursion is cheap and runs
once per cohort per year.

### Anniversary mechanics

In order, at each anniversary:

1. **Bonus**, while the contract is still deferring: the benefit base accrues $6\%$ of the
   bonus base a year on the Core option, simple rather than compound, for up to ten contract
   years. A withdrawal ends it.
2. **Guaranteed withdrawal base adjustment**, on the later of the anniversary following age
   seventy and the twelfth contract anniversary, and only if no withdrawal has been taken by
   then. It floors the benefit base at a multiple of premium, which is what makes it valuable
   in a market that has fallen.
3. **Rider charge**, $\min(\phi B_{t-1},\, AV_{t^-})$, deducted from the contract value. It is
   charged on the benefit base, not the contract value, and it stops once the contract value
   reaches zero.
4. **Withdrawal** of $u \cdot g B_{t-1}$, where $u$ is utilisation. The contract value covers
   what it can and the insurer pays the shortfall:
   $$C_t = \max\left(u g B_{t-1} - AV_t^{\text{after charge}},\ 0\right)$$
5. **Step-up**: $B_t = \max(B_{t-1},\, AV_t)$ on the Contract Anniversary Value method, applied
   after the withdrawal. A step-up also restarts the bonus period and lifts the bonus base.
6. **Death benefit**, paid on the probability of dying during the year: the excess of the death
   benefit base over the contract value, floored at zero.

Once the contract value reaches zero the benefit base stops moving, so the guaranteed amount is
frozen and the insurer pays it for as long as the owner lives. No fees are collected after that
point. The combination is what makes the guarantee expensive: income and outgo never overlap.

The guaranteed percentage comes from the rate sheet's age bands at the age of the **first**
withdrawal, not at issue, so a contract that defers moves up a band. At issue age 70 with five
years of deferral the rate is 5.95% rather than 5.75%, and the model reads it from the committed
rate sheet rather than carrying a hardcoded number.

### Measuring continuous charges exactly

Attributing fees requires knowing how much of the drag was the insurer's revenue, which is
awkward when the drag is continuous and the balance is stochastic. There is a clean answer. For
a proportional charge $c$ out of total drag $m$ over an interval of length $\Delta$, the value
at the end of the interval of the charges collected during it is exactly

$$\frac{c}{m}\,AV^{\text{before drag}}\left(1 - e^{-m\Delta}\right)$$

Charges left invested in the sub-account grow at the short rate under $Q$, so discounting that
end-of-interval amount gives the same present value as discounting the continuous stream. No
sub-stepping, no approximation. `tests/test_liability.py` checks the closed form against a
brute-force calculation that splits the year into twenty thousand intervals.

## Valuation

The quantity valued is the market risk benefit on Jackson's own definition: projected benefits
less attributed fees.

$$V = \underbrace{\mathbb{E}^{Q}\left[\sum_t D(t)\,{}_tp_x\,C_t\right]}_{\text{claims}}
      - \alpha \underbrace{\mathbb{E}^{Q}\left[\sum_t D(t)\,{}_tp_x\,F_t\right]}_{\text{attributable fees}}$$

with $F_t$ the rider charge, the insurer's share of the account drag and the explicit death
benefit charge. $\alpha$ is the attribution percentage, fixed at inception as

$$\alpha = \min\left(1,\ \frac{PV(\text{claims at inception})}{PV(\text{attributable fees at inception})}\right)$$

and then held static for the life of the contract. That is what Note 6 describes: a portion of
total projected fees attributed to the benefit to offset projected claims, expressed as a
percentage of total projected fees, capped at 100%. Where the attributable fees cover the claims
the benefit starts at a fair value of zero; where they do not, a liability is recognised at
issue.

Holding $\alpha$ static is not a detail. A contract written in September 2016, with the ten-year
zero at 1.57%, calibrates to $\alpha = 1.00$ and starts as a liability. The same contract
written at 31 December 2025, with the zero at 4.21%, calibrates to $\alpha = 0.79$ and starts at
zero. Four of the five vintages in the portfolio comparison calibrate to the cap, and only the
2024 vintage, written at a 4.33% ten-year rate, has margin. Freeze the percentage and a decade
of rising rates and rising markets turns the first contract into a net asset, which is the
mechanism behind a disclosed liability that is now an asset.

Fund expenses are excluded from attributable fees. They are paid to the funds, not to the
insurer.

Mortality is applied as an expectation rather than simulated. For a representative book the two
are identical in the mean, and the point of the exercise is the mean.

Surrender runs alongside mortality as a second decrement, but only while the contract value is
above zero, because an exhausted contract has no surrender value to take. That makes persistency
path-dependent, so it is carried per path rather than folded into the survival curve. The rate
is dynamic: it damps as the guarantee moves into the money, with a floor, because a contract
holder sitting on a valuable guarantee does not surrender it.

## The curve, and the short rate around it

Treasury par yields at eight tenors are bootstrapped to zero rates by forward substitution:

$$D(t_n) = \frac{1 - \frac{c_n}{2}\sum_{i<n} D(t_i)}{1 + \frac{c_n}{2}}, \qquad
  z(t) = -\frac{\ln D(t)}{t}$$

and every bootstrap is checked by repricing the par bonds off the zeros it produced. Over 2,495
dates the worst round-trip error is $2\times10^{-16}$.

A bootstrapped curve stops at thirty years and the liability runs to forty-five, so the valuation
curve is a Nelson-Siegel-Svensson fit to the bootstrapped zeros. What happens past the last quote
is then decided by the fit, and an unconstrained one decides it badly: letting both decay
parameters roam the same range lets the search put the short hump at fifteen years, which fits the
quoted tenors perfectly well and extrapolates to nonsense. The two decays are searched on separate
ranges and kept apart by a factor of 1.5, which is what keeps the level parameter interpretable as
the rate the curve flattens to - and that rate is what a forty-five-year projection sits on.

The fit is a grid search on the two decays with the four betas solving in closed form at each
point, rather than a gradient optimiser. Determinism matters here: this runs on every date of a
ten-year replay, and a fit that jittered between neighbouring local optima would put a sawtooth
into the simulated bond prices that no market made.

What it costs and what it buys are both in `reports/tables/curve_fit.csv`: the fitted par yields
sit within about three basis points of the quotes at every tenor out to ten years and seven at
twenty, and past the last quote the zero curve settles at 5.04 to 5.07 per cent from thirty-five
years to fifty rather than drifting.

Rates are stochastic. The short rate follows a one-factor Hull-White process built around that
curve:

$$dr_t = \left(\theta(t) - a r_t\right)dt + \sigma_r\,dW_t^{r}$$

with $\theta(t)$ chosen so the model reproduces the initial curve exactly, which the tests
check. $a$ and $\sigma_r$ are fitted to the long history of the three-month and ten-year
constant-maturity yields, because ten years of panel cannot identify a mean reversion speed.

The closed-form bond price is what makes the nested work affordable: a node deep in a simulation
carries a short rate, and the whole curve at that node follows from it, so a nested valuation
does not have to carry a curve per path.

A rate shock is a parallel shift of the fitted level, which moves the zero rates and the
instantaneous forward by the same amount and leaves the shape alone. That is what a parallel
shift means and what Item 7A describes.

## Volatility

Note 6 says implied volatility out to five years, grading to a historical level by year ten,
with an explicit risk margin in the long-run level. What is available is the live SPX chain,
which reaches 3.23 years.

Variance follows a Heston process correlated with the equity diffusion:

$$dv_t = \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^{v}, \qquad
  d\langle W^{Q}, W^{v}\rangle_t = \rho\,dt$$

calibrated to 1,468 cleaned quotes across seventeen expiries, weighted by one over vega so that
a price error reads as a volatility error and the wings count. The objective is evaluated with
the COS method rather than by simulation, and the fit comes out at a worst per-maturity root
mean square error of 0.85 volatility points.

Simulation uses the quadratic-exponential scheme with a martingale correction, which is not a
refinement here but a requirement: the calibrated parameters violate the Feller condition by a
wide margin, so a scheme that can take the variance negative would have to truncate it, and
truncation biases the discounted index away from being a martingale. The tests check the
martingale property directly.

**The long-run level is the parameter that matters and the one the data has least to say
about.** Almost all of the liability's volatility exposure sits in $\theta$ - 866 per point
against 27 for the current variance - and $\theta$ is fitted to options none of which mature past
3.23 years, applied to a liability that runs 45. It is not free: pinning it to the realised
ten-year level plus the issuer's one-point margin, which is the methodology Note 6 describes,
costs four times the volatility error beyond two years and drives mean reversion to its bound,
so the chain rejects that level. But "the chain rejects the alternative" is weaker than "the data
identifies the value", and the robustness table prices the issuer's level at $-2.3\%$ of premium.

Along the replay window the instantaneous variance comes from the three-month implied index
rather than from the calibration, so the level moves with the date; the skew, the mean reversion
speed and the long-run level are held at their December 2025 values, because no free historical
option data exists.

Reading a quote straight into $v_0$ would be wrong in a direction that matters, because a
volatility index is a variance rate over its own window rather than an instantaneous level. Over
a window of length $T$ the model's expected average variance is

$$w(T) = \theta + (v_0 - \theta)\,\frac{1 - e^{-\kappa T}}{\kappa T},$$

which is linear in $v_0$ and inverts in one line; the same expression run forward carries a quote
at one tenor to another, which is how the puts get a volatility level at one and two years from
data that stops at six months. Three index histories then test that mapping on a decade the
calibration never saw: VIX6M in the outward direction, which is the direction the option leg
actually uses, and VIXCLS inward from VXVCLS.

The shape parameters need a different instrument, because $w(T)$ is invariant to a trade-off
between $\rho$ and $\xi$ - every variance quote at every tenor is silent on both. Cboe's SKEW
index is not. It publishes $100 - 10\,\zeta$ where $\zeta$ is the risk-neutral skewness of the
thirty-day return, and $\zeta$ comes out of the characteristic function directly: with
$\psi(u) = \log \mathbb{E}[e^{iu X_T}]$,

$$\zeta = \frac{-i\,\psi'''(0)}{\left(-\psi''(0)\right)^{3/2}},$$

taken by central differences rather than from a published cumulant expression, because the second
cumulant already in the pricer is the truncation approximation the COS literature uses and sits
1.5% from the true value - harmless for setting an integration range, wrong for a third moment.
The derivatives are flat to six figures across four decades of step size and the result agrees
with a simulated sample to about one per cent. `docs/validation.md` reports what both checks find
and which way each one biases the result.

## Sub-account

The valuation carries a blended sub-account whose equity weight is the disclosed fund mix's,
$0.7235 + 0.6 \times 0.1832 = 0.833$, so an index shock $s$ moves the contract value by
$0.833 s$.

Along a realised path the sub-account is rebuilt rather than assumed: the equity sleeve earns the
index, the bond sleeve earns a constant-maturity zero held for a day and rolled, cash earns
overnight, and the weights are reapplied every day. The model sees a single-factor sub-account
and the policy experiences a blend, which is genuine basis risk rather than a modelling
convenience. Treating the non-equity sleeve as riskless inside the valuation understates total
sub-account volatility slightly; the direction is stated.

## Mortality

$$q(x, y) = q^{2012}(x)\,(1 - G2_x)^{\,y - 2012}$$

Generational, so the rate applying to attained age $x + k$ in calendar year $y + k$ carries
$y + k - 2012$ years of improvement. Scale G2 stops at age 105, where the published scale has
already trended to zero, so no improvement is applied above it.

Sexes are blended at the **survival** level, not the rate level. A 50/50 book is two populations,
and averaging mortality rates before compounding gives the wrong expected number of payments.
`tests/test_mortality.py` checks that the blend is exactly the average of the two survival curves
and that the rate-blended alternative differs.

Deaths and survivors come out of one pass and account for everyone, because the death benefit is
paid on the deaths and the withdrawal guarantee on the survivors, and anything falling between
the two would be a cash flow the projection never pays and never charges for.

The projection is truncated at attained age 115. On common draws - which is the only way the
comparison means anything, since the simulator's normals depend on the horizon - extending it to
120 moves the market risk benefit by less than a dollar and cutting it to 105 moves it by $36,
against a Monte Carlo standard error of $136. Truncation costs an order of magnitude less than
simulation noise.

## Simulation

Twenty thousand paths, twenty-four steps a policy year, antithetic, with the standard error taken
across pair averages because the two halves of a pair are dependent. The standard error on the
market risk benefit is $136 on a $100,000 policy and it halves as paths quadruple, which the
tests check.

Paths are cached on what actually changes them - the market parameters, the path count and the
seed - and an equity shock is deliberately not part of that key, because it does not change them.
So a Greek set visits half a dozen market states and pays for a simulation at each, while a
shocked valuation at the same state reuses its base's draws. Without that the difference between
two valuations is dominated by simulation noise rather than by the sensitivity: the level has a
standard error of $136 while a ten-basis-point rate bump moves it by a few hundred.

## Greeks

Central differences on common random numbers. Equity exposure is taken with respect to the log
index level, because that is directly the dollar notional a hedge has to carry:

$$\text{equity exposure} = \frac{V(S(1+h)) - V(S(1-h))}{\ln\frac{1+h}{1-h}}$$

The exact log-space denominator rather than $2h$; at $h = 1\%$ the difference is a hundredth of a
percent and costs nothing to get right.

Rho is per basis point of a parallel shift, taken over ten basis points rather than one. One is
the textbook choice and it is the wrong one here: the difference it produces is small enough that
the paired Monte Carlo error is a material share of it, and the liability's convexity over ten
basis points is negligible. The trade is reported rather than argued, because each Greek carries
the standard error of its own paired difference and that makes it checkable.

Vega is split. The current variance is what trades and what moves week to week; the long-run
level is an assumption no listed option reaches. A single vega that moves both at once hides
which one the number came from, and on a forty-five-year liability the long-run level does almost
all of the work.

Curvature is measured over a ten per cent log move, and the same step is used for the liability
and for the instruments sold against it. They were not: the regression proxy averaged over ten
per cent while the listed puts used two, which is a three to nine per cent difference in the put's
gamma and a solve matching one against the other is sizing a position off a unit mismatch.

Signs, which decide whether the hedge is long or short. Equity exposure is negative: a higher
index makes the guarantee less likely to bite. Rho is negative: a long-dated liability discounts
away faster when rates rise. Vega is positive. Equity gamma is positive, so the loss from a fall
exceeds the gain from a rise.

Equity exposure is not negative unconditionally, and the exception is worth knowing before
reading any hedge output. The benefit base steps up to the contract value on the anniversary, so
when the contract value sits above the benefit base the index level on that date fixes the
guaranteed income for the rest of the contract's life. At 1.20 of the benefit base in policy year
five the exposure runs from $-27{,}492$ with most of the year to go to $+2{,}679$ on its eve, and
changes sign with no market move behind it.

## The proxy the hedge runs on

A hedge cannot run a full valuation at every rebalance date, so it runs a least-squares regression
proxy fitted on the simulated paths: a natural cubic spline in raw moneyness, linear and quadratic
in the rate level and in volatility, fitted year by year. The knots sit at quantiles of the
*live* contracts with exhaustion pinned as its own knot, because by the late years nine tenths of
the sample is exhausted and quantiles taken over everything would put every knot on top of zero,
leaving the part of the axis the hedge actually operates in with no resolution at all.

Two fit families per year, and they are not redundant. The post-event family is the value at an
anniversary after that year's charges, withdrawal and step-up; the pre-event family is the value
an instant before them. A hedge rebalancing inside a policy year sees a contract whose value has
grown away from its benefit base with the year-end events still ahead of it, which is a pre-event
state, and the post-event family has no design points up there because the step-up caps the
moneyness it records. Bracketing an intra-year date between the two gives both endpoints a state
their own design contains, and the interpolation's slope is the theta term, without which a
profit attribution has no term for the passage of time at all.

The penalty is on roughness rather than on coefficient size. A ridge cannot see the problem it was
put there to fix: with twenty basis functions crowded into the narrow band of moneyness the
contract occupies, the fitted level came out smooth while its second derivative oscillated between
plus twelve hundred and minus six hundred from one state to the next, and a wildly wiggly function
can have small coefficients.

Even so the proxy's second derivative is not a risk number, so the option leg is sized from a
tabulated nested surface built by valuing the liability at a ladder of equity levels around each
node. The proxy's value is accurate, its delta is usable from about policy year three, and its
curvature is not usable at all; `docs/validation.md` gives the measured sizes.

## Hedge

Daily, along the realised path. At each date the liability is revalued through the proxy at the
market that prevailed, exposures are recomputed, and positions are resized. Nothing about the
following day enters the sizing, and the tests verify that by rebuilding each period's profit from
the position recorded on the previous date.

Positions are not one instrument per Greek. The book is solved by weighted least squares over four
exposures - delta, gamma, vega and rho - against the instruments available, with a ridge on
normalised columns so that an instrument carrying two exposures is not asked to be exactly right
about one of them at the cost of the other. The instrument set is equity futures, total return
swaps, listed index puts, rate futures, bond forwards and interest rate swaps, each priced off the
same curve and smile the liability is valued on.

**Costs** are charged on the change in position, so a stable hedge is cheap, and every cost
assumption is swept at half, one and two times its base level. A conclusion that survives the
sweep is worth something; one quoted at a single cost level is a statement about that level.

**Rolls.** A position rolls when its remaining life falls below a share of its original maturity,
and a solved option position also rolls when its strike drifts away from its target moneyness. An
overlay does not: a macro put spread bought for a crash should not be closed at the first sign of
the crash, and a strike band that closed it was costing the strategy most of what it was bought
for.

Profit is reconciled rather than asserted. The ledger carries cash, the hedge mark and the
liability, and their change adds to the reported period profit by construction, with the
attribution splitting it into delta, gamma, vega, rho, theta, the anniversary's own cash flows and
a residual.

The period profit is measured against the position before the programme starts, which is the
guarantee with no cash and no book against it. The first day's profit is therefore the cost of
striking the book rather than zero. Starting from the first row's net worth instead - the hedge
already on - drops that cost from the profit column while the cost column keeps it, which leaves
the two halves of a frontier table covering different windows; on the strategy holding one-year
puts it was 0.11% of account value.

### What the rider earned

The same profit cut a second way, into terms a pricing committee rather than a hedging desk would
recognise:

$$\text{net} = \text{fee income} + \text{interest on cash} - \text{trading and carry} - \text{uncovered cost}$$

$$\text{uncovered cost} = \text{claims paid} + \Delta L - \text{hedge result}$$

The hedge result is its closing mark plus every dollar the trading moved - premium paid for
options, the realised result of closing a futures position, the proceeds of a short - and the
simulator records that flow per day rather than leaving it to be backed out of the cash balance.
What the balance is then good for is a check: cash opens at zero and moves by the contract's
flows, its own interest, the trading costs and that recorded flow, so the gap between the closing
balance and those four is zero unless something has reached cash without a column, and every run
reports it. On the unhedged arm there is nothing to recover and the uncovered cost is the whole
cost of the guarantee, which is what makes the two arms comparable.

Unlike the Taylor attribution there is no residual here. Every term is an exact rearrangement of
the cash recursion, which is why it can be quoted as a decomposition of the margin rather than as
an explanation of it, and why the reconciliation tolerance in the test is machine precision.

Everything is reported per unit of benefit base and annualised, because the rider charge is levied
on the benefit base and because the cohorts differ by a factor of three in how large that base has
grown. On a window of a few weeks the annualisation makes a rate, not an outcome.

### The offset ratio, which is how the hedge is compared with a disclosed one

Item 7A publishes the fair-value impact of each shock on the guarantee liabilities and, in a
separate table on the same dates, its impact on the derivative book. For a liability impact
$\Delta L$ and a derivative impact $\Delta D$ the offset ratio is

$$\omega = \frac{\Delta D}{\Delta L},$$

and the sign convention is the part to get right. A liability impact is a change in a carrying
amount, so its effect on earnings is $-\Delta L$; a derivative impact is a change in an asset's
mark, so its effect is $+\Delta D$. The total is $\Delta D - \Delta L$, which is zero when the two
raw figures agree - so a complete hedge is $\omega = 1$, no hedge is zero, and a position that
amplified the exposure is negative. Defining it with a sign flip, which is the intuitive-looking
choice, turns every ratio negative at once.

The model produces the same quantity by striking a book against the liability's exposures at a
past balance-sheet date and marking it under the identical shocks: the equity shock multiplies the
index, and the rate shock moves the fitted curve's level parameter, which is the same object the
valuation shifts, so neither side is shocking a different curve from the other. The Greeks for
that solve come from full repricing rather than from the regression proxy, because four dates and
one contract is cheap and a sizing comparison that inherited the proxy's delta error would be
measuring the proxy.

One trap sits in the exposure vector and is worth recording. The proxy differentiates in the
account value and the full Greeks differentiate in the index, so the equity-weight conversion
between contract space and index space applies to the first and not the second. Running both
through it undersizes an equity hedge by a sixth; skipping it on the proxy oversizes one by a
fifth. The two conventions are now told apart by type and an identity test asserts that both
routes reach the same index delta for the same liability.

## Economic against reported

Three liabilities on the same contract, the same paths and the same day, because the split needs
all three.

1. **Economic.** Best-estimate mortality, Treasury discounting. What the hedge is sized on.
2. **Reporting without own credit.** The margin-loaded mortality table, Treasury discounting. The
   difference from (1) is the risk margin, which moves with the market and therefore reaches net
   income.
3. **Reporting with own credit.** The same, discounted at Treasury plus the insurer's own
   non-performance spread, taken as 0.6 times the Baa spread to reflect the higher ratings of the
   insurance subsidiaries. The difference from (2) is the own-credit adjustment, and under ASU
   2018-12 that piece goes to other comprehensive income rather than through net income.

The margins come from the mortality table rather than from a loading: the NAIC-adopted annuity
table ships in a Basic version and a Period version that is the Basic table with the margins the
Life Actuarial Task Force set, so valuing the same contract on both gives a difference someone
else calibrated rather than a percentage this project invented.

The spread enters through the discount factor rather than by shifting the curve the paths are
simulated under, which would move the equity drift as well and price a different contract. Because
the spread itself moves daily, the reporting liability is fitted at a grid of spread levels and
read off by interpolation rather than at one level and extrapolated.
