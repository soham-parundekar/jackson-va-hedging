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
proportional drag. The sub-account is a rebalanced blend of three sleeves rather than one asset -
the Sub-account section below writes it out - and the equity sleeve is the one carrying the
stochastic volatility:

$$\frac{dS_t}{S_t} = r_t\,dt + \sqrt{v_t}\,dW_t^{Q}, \qquad
\frac{dAV_t}{AV_t} = \frac{dF_t}{F_t} - m\,dt$$

with $F$ the blended fund. Under the risk-neutral measure any traded portfolio earns the short
rate, so the drift is
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

1. **Rider charge**, $\min(\phi B_{t-1},\, AV_{t^-})$, deducted from the contract value. It is
   charged on the benefit base, not the contract value, and it stops once the contract value
   reaches zero.
2. **Death benefit charge**, on the death benefit base, deducted the same way and only for the
   add-on benefits; the basic benefit is free.
3. **Withdrawal** of $u \cdot g B_{t-1}$, where $u$ is utilisation. The contract value covers
   what it can and the insurer pays the shortfall:
   $$C_t = \max\left(u g B_{t-1} - AV_t^{\text{after charges}},\ 0\right)$$
4. **Step-up**: $B_t = \max(B_{t-1},\, AV_t)$ on the Contract Anniversary Value method, applied
   after the withdrawal. A step-up that raises the benefit base also lifts the bonus base and
   restarts the bonus clock, while the owner is 80 or younger.
5. **Bonus**, while the contract is still deferring and the contract value is above zero: the
   benefit base accrues $6\%$ of the *bonus base* a year on the Core option, simple rather than
   compound, for up to ten contract years. A withdrawal in the year precludes it.
6. **Guaranteed withdrawal base adjustment**, on the later of the anniversary following age
   seventy and the twelfth contract anniversary, and only if no withdrawal has been taken by
   then. It floors the benefit base at a multiple of premium, which is what makes it valuable
   in a market that has fallen.
7. **Death benefit**, paid on the probability of dying during the year: the excess of the death
   benefit base over the contract value, floored at zero. The roll-up and the ratchet both stop
   at the anniversary before the owner's 81st birthday.

The order is not a presentational choice and an earlier version of this document had it wrong,
with the bonus and the adjustment ahead of the charge. Two things fix it. The prospectus sets the
bonus base *after* a step-up — "with any step-up (if the GWB increases upon step-up), the Bonus
Base is set to the greater of the GWB after, and the Bonus Base before, the step-up" — so the
step-up has to be resolved before the bonus is credited, or the bonus is paid on a base the
step-up has not yet lifted. And the rider charge is assessed on the benefit base the contract year
began with, not on the base after that year's bonus: charging it the other way round bills the
owner for a credit granted at the end of the year. Running the documented order instead of this
one prices a different contract.

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

with $C_t$ the living-benefit shortfall and the death benefit paid above the contract value, and
$F_t$ the rider charge, the insurer's share of the account drag and the explicit death benefit
charge. $\alpha$ is the attribution percentage, fixed at inception as

$$\alpha = \min\left(1,\ \frac{PV(\text{claims at inception})}{PV(\text{attributable fees at inception})}\right)$$

and then held static for the life of the contract. That is what Note 6 describes: a portion of
total projected fees attributed to the benefit to offset projected claims, expressed as a
percentage of total projected fees, capped at 100%. Where the attributable fees cover the claims
the benefit starts at a fair value of zero; where they do not, a liability is recognised at
issue.

Calibrating it for an in-force contract needs the market of its issue date, and only the vintage
portfolio has one: there, each vintage carries the percentage its own issue date's curve implied,
which is the single largest difference between a 2016 vintage and a 2024 one. Everywhere else a
cohort is rewound to issue and repriced under *today's* market, because the market of six years
ago is recoverable and the behaviour and mortality assumptions in force then are not. The
direction of that error is knowable rather than merely acknowledged: attribution is capped at one
and most cohorts sit below the cap, so a cohort issued into a lower-rate market would have been
given a higher percentage than the shortcut produces.

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

The headline at-issue valuation runs with surrender switched off, and that is deliberate rather
than an oversight. A guarantee valued with no lapse is an upper bound, which is the same posture
the full-utilisation assumption takes, and it keeps the one number the disclosure comparison turns
on free of a parameter nothing in the filings identifies. Lapse is switched on where it is being
measured: the behaviour sweep moves it from zero to 4%, and the hedging backtest and every cohort
in the rider economics carry 4% with a damping exponent of 1.2 and a 1% floor.

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
ranges and kept apart by a factor of 1.5, which keeps the level parameter interpretable as the rate
the curve flattens to.

That alone is not enough. An NSS fit free past thirty years sends the long-run level to minus
twenty-six per cent on this curve while still matching every quoted tenor to three basis points, so
the tail is pinned explicitly: synthetic anchor points at 35, 40, 45, 50 and 60 years, each set to
the last bootstrapped zero and weighted at half a quoted point. The assumption that writes in is
the one the bootstrap already makes when asked past its last knot - the zero rate is flat beyond
the last quote - and the half weight is there because it is an assumption rather than an
observation. A forty-five-year liability sits on that tail, so it is stated rather than left
implicit: the committed fit eases from 5.06 per cent at thirty-five years to 4.97 at fifty.

The fit is a grid search on the two decays with the four betas solving in closed form at each
point, rather than a gradient optimiser. Determinism matters here: this runs on every date of a
ten-year replay, and a fit that jittered between neighbouring local optima would put a sawtooth
into the simulated bond prices that no market made.

What it costs and what it buys are both in `reports/tables/curve_fit.csv`: the fitted par yields
sit within about three basis points of the quotes at every tenor out to ten years and seven at
twenty, and past the last quote the zero curve eases from 5.06 per cent at thirty-five years
to 4.97 at fifty rather than drifting.

Rates are stochastic. The short rate follows a one-factor Hull-White process built around that
curve:

$$dr_t = \left(\theta(t) - a r_t\right)dt + \sigma_r\,dW_t^{r}$$

with $\theta(t)$ chosen so the model reproduces the initial curve exactly, which the tests
check. $a$ and $\sigma_r$ are fitted to the long history of the three-month constant-maturity yield,
because ten years of panel cannot identify a mean reversion speed. The ten-year series in the same
file carries the equity-rate correlation instead; the short-rate parameters come from the short
rate, which is the series the process is written on.

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

The two payoffs the expansion has to price are not equally well conditioned, and over the
parameter bounds the search can walk through, only one of them is usable. The put's cosine
coefficient integrates $e^y$ over the part of the truncation range below zero, where it is
bounded by one; the call's integrates it over the part above zero, so it evaluates $e^b$ at the
range's upper edge and the sum over terms has to cancel coefficients of that size back down to a
price of order one. At a variance distribution dispersed enough, the cancellation is gone: tested
across the fit's own bounds, the direct call broke put-call parity by more than a basis point of
the forward at a third of the corners, came out negative at a sixth, and overflowed at the far
ones. Each of those reaches a least-squares objective as a residual it steers by. So the call is
priced both ways and the direct value kept only where it is finite, above its own arbitrage floor
and within a basis point of the forward of parity. At the calibrated parameters the two agree to
about a hundredth of a basis point, so every quote on this chain is priced by the direct
expansion and parity is only ever the fallback it is meant to be.

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

Along the replay window the instantaneous variance comes from the thirty-day implied index
rather than from the calibration, so the level moves with the date; the skew, the mean reversion
speed and the long-run level are held at their December 2025 values, because no free historical
option data exists.

Reading a quote straight into $v_0$ would be wrong in a direction that matters, because a
volatility index is a variance rate over its own window rather than an instantaneous level. Over
a window of length $T$ the model's expected average variance is

$$w(T) = \theta + (v_0 - \theta)\,\frac{1 - e^{-\kappa T}}{\kappa T},$$

which is linear in $v_0$ and inverts in one line; the same expression run forward carries a quote
at one tenor to another, which is how a volatility level is reached at tenors no free index
publishes. Three index histories then test that mapping on a decade the calibration never saw:
VIX6M in the outward direction, which is the direction the option leg actually uses, and VIXCLS
inward from VXVCLS.

The replay carries one such level, at a one-year reference tenor, and every put in the hedge book
is marked at it whatever its own maturity. For the quarter-year and one-year puts the strategies
hold that is close; for the two-year macro overlay it is a stated approximation, and it biases
that leg's mark rather than its payoff.

The shape parameters need a different instrument, because $w(T)$ is invariant to a trade-off
between $\rho$ and $\xi$ - every variance quote at every tenor is silent on both. Cboe's SKEW
index is not. It publishes $100 - 10\,\zeta$ where $\zeta$ is the risk-neutral skewness of the
thirty-day return, and $\zeta$ comes out of the characteristic function directly: with
$\psi(u) = \log \mathbb{E}[e^{iu X_T}]$,

$$\zeta = \frac{i\,\psi'''(0)}{\left(-\psi''(0)\right)^{3/2}},$$

taken by central differences rather than from a published cumulant expression, because the second
cumulant already in the pricer is the truncation approximation the COS literature uses and sits
5.2% from the true value at thirty days, falling to 0.7% at a year - harmless for setting an
integration range, wrong for a third moment at the tenor the SKEW index is quoted on.
The derivatives are flat to six figures across four decades of step size and the result agrees
with a simulated sample to about one per cent. `docs/validation.md` reports what both checks find
and which way each one biases the result.

## Sub-account

The valuation carries a blended sub-account whose equity weight is the disclosed fund mix's,
$0.7235 + 0.6 \times 0.1832 = 0.833$, so an index shock $s$ moves the contract value by
$0.833 s$.

The same blend is carried inside the valuation, not just along the replay. The simulator steps all
three sleeves and rebalances to the weights every step: the equity sleeve earns the Heston index,
the bond sleeve earns a constant-maturity zero priced off the Hull-White short rate and rolled, and
cash earns the short rate itself. So the bond sleeve is stochastic on both sides and the
correlation between it and equity is the one the state carries, rather than a non-equity sleeve
treated as riskless. Along a realised path the same three sleeves are rebuilt from the index, the
bootstrapped curve and the overnight rate.

What is *not* modelled is the gap between a fund menu and the index sleeve that stands in for it.
That gap has a parameter, ``tracking_error``, and it is zero in every result here - no experiment
sweeps it - so the basis risk in these numbers is the difference between a three-sleeve portfolio
and the instruments sold against it, not between a managed fund and its benchmark. The second kind
is real and unpriced, and `docs/limitations.md` says so.

## Mortality

$$q(x, y) = q^{2012}(x)\,(1 - G2_x)^{\,y - 2012}$$

Generational, so the rate applying to attained age $x + k$ carries $y + k + 1 - 2012$ years of
improvement - the improvement runs to the calendar year the policy year ends in, not the one it
begins in. Scale G2 stops at age 105, where the published scale has
already trended to zero, so no improvement is applied above it.

Sexes are blended at the **survival** level, not the rate level. A 50/50 book is two populations,
and averaging mortality rates before compounding gives the wrong expected number of payments.
`tests/test_mortality.py` checks that the blend is exactly the average of the two survival curves
and that the rate-blended alternative differs.

Deaths and survivors come out of one pass and account for everyone, because the death benefit is
paid on the deaths and the withdrawal guarantee on the survivors, and anything falling between
the two would be a cash flow the projection never pays and never charges for.

The at-issue valuation truncates the projection at attained age 115. On common draws - which is
the only way the comparison means anything, since the simulator's normals depend on the horizon -
extending it to 120 moves the market risk benefit by less than a dollar and cutting it to 105
moves it by $36, against a Monte Carlo standard error of $136. Truncation costs an order of
magnitude less than simulation noise.

Everything that has to fit a regression proxy - the hedging backtest, the proxy validation, the
convexity surface and the statutory lens - truncates at 105 instead. That is what the $36 figure
is there to license: a forty-five-year design matrix costs the proxy accuracy in the region the
hedge actually visits, and the liability it drops is a thirtieth of the Monte Carlo error on the
number it is dropped from.

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
all of the work. The split costs one convention: the combined bump is central, the
current-variance leg is a one-sided forward difference against the base valuation, and the
long-run leg is the difference between them. Only the equity and rate Greeks are central on both
sides.

Curvature is measured over a ten per cent log move, and the same step is used for the liability
and for the instruments sold against it. They were not: the regression proxy averaged over ten
per cent while the listed puts used two, which is a three to nine per cent difference in the put's
gamma, and a solve matching one against the other is sizing a position off a unit mismatch. The
full-repricing route kept the two per cent for longer still, which left the disclosure comparison
making the same mistake after the backtest had stopped; both routes read one constant now.

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
in the rate level and in volatility, with the three pairwise interactions between them, fitted year
by year. The knots sit at quantiles of the
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

The penalty that does the work is on roughness rather than on coefficient size. A ridge cannot see
the problem it was put there to fix: with twenty basis functions crowded into the narrow band of
moneyness the contract occupies, the fitted level came out smooth while its second derivative
oscillated between plus twelve hundred and minus six hundred from one state to the next, and a
wildly wiggly function can have small coefficients. A ridge is still applied on top, scaled to the
design matrix's trace at $10^{-8}$, and it is there to keep the solve conditioned rather than to
shape the fit.

Even so the proxy's second derivative is not a risk number, so the option leg is sized from a
tabulated nested surface built by valuing the liability at a ladder of equity levels around each
node. The proxy's value is accurate, its delta is usable from about policy year three, and its
curvature is not usable at all; `docs/validation.md` gives the measured sizes.

## Hedge

The ledger is daily, along the realised path: at each date the liability is revalued through the
proxy at the market that prevailed and the book is marked. Rebalancing is weekly by default - the
calendar the strategy carries, swept to daily, monthly and a drift band in E3 - so the exposures
are recomputed every day and the positions are resized on the rebalance dates. Nothing about the
following day enters the sizing, and the tests verify that by rebuilding each period's profit from
the position recorded on the previous date.

Positions are not one instrument per Greek. The book is solved by weighted least squares over four
exposures - delta, gamma, vega and rho - against the instruments available, with a ridge on
normalised columns so that an instrument carrying two exposures is not asked to be exactly right
about one of them at the cost of the other. The weights are not equal: delta and rho come in at
one and gamma and vega at a tenth, because the first two are what a weekly hedge is judged on and
the second two are second-order. That ratio is what decides how large the put position comes out,
so it is a parameter of the result rather than a numerical detail.

The library prices equity futures, total return swaps, listed index puts, rate futures, bond
forwards and interest rate swaps. The strategies use three of them - an equity future, a listed
put and a receive-fixed swap - because two instruments with nearly the same rate exposure make the
design matrix singular, and because the point of the ladder is to add one kind of exposure at a
time. The instruments are priced off a smaller market object than the liability is valued on: a
bootstrapped curve rather than the fitted one, and Black-76 against a fixed-shape smile rather
than Heston. That gap is deliberate - a hedger marks a book off quotes, not off the model that
produced the target - and the residual it leaves is part of what the backtest measures.

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
attribution splitting it into delta, the sub-account's basis against the index, gamma, vega, rate,
theta, the anniversary's own cash flows, carry, trading cost and a residual. The rate bar is taken
against the ten-year zero rather than against the parallel shift rho is defined on, because the
ten-year point is what the ledger records daily and a parallel shift is not an observable along a
realised path; the two agree to the extent the curve moves in level.

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

## The fee the guarantee is worth

The break-even fee is the charge $\phi^*$ that makes the rider worth zero at issue:

$$\mathbb{E}^{Q}\!\left[\,\mathrm{PV}(\text{claims}) - \mathrm{PV}(\text{attributable fees at } \phi^*)\,\right] = 0$$

solved on the fee itself rather than on a scaling of the fee's value, because the fee changes the
contract: charging more takes more out of the account every year, which brings exhaustion forward
and raises the claim. The same paths are used at every fee, which is what keeps a Monte Carlo
objective smooth enough to solve at all, and the attribution percentage is held at one, because
this asks what the guarantee costs against the fees it is charged rather than what share of them
accounting attributes to it.

The relationship is not monotone over a wide enough bracket, and that is the part worth stating.
Up to a charge of about five per cent the revenue wins and the guarantee gets cheaper; past that
the charge empties the account fast enough to end the fee stream before the payments end, and the
curve turns back up. So there can be two roots or none, a bracket spanning the turn finds neither,
and the solve checks both ends and refuses rather than returning a bound. Off the September 2016
curve it refuses, because on that curve no charge prices this contract at all.

## The statutory lens

The statutory requirement answers a different question from everything above it. A price is an
expectation under a measure chosen so hedges are self-financing; a reserve is a percentile of a
real-world distribution, and on this book the two differ in sign as well as level - the market
risk benefit is a net asset at issue while the requirement is positive.

What is implemented is VM-21's shape rather than VM-21. The block is projected on real-world
scenarios - the same Heston and Hull-White model with an equity risk premium added to the equity
drift and to nothing else - the deficiency is accumulated, and the *greatest present value* of the
accumulated deficiency over the horizon is taken per scenario:

$$\text{GPVAD}_j = \max_{k \le K} \sum_{i \le k} D_{ij}\,\text{outgo}_{ij}$$

The maximum over the path rather than the value at the end is the measure's whole point: a block
that costs money for fifteen years and earns it back has to be funded through the fifteen years.
The requirement is then the mean of the worst tail of those,

$$\text{CTE}(\alpha) = \mathbb{E}\!\left[\,\text{GPVAD} \;\middle|\; \text{GPVAD} \ge q_\alpha \,\right],$$

computed by sorting and averaging rather than from a fitted quantile, with the tail size rounded
up and counted from the kept side. CTE(70) is VM-21's stochastic reserve level and CTE(90) stands
in for the capital question; the gap between them is what separates a hedge that flattens the
tail's mean from one that flattens its shape.

What is missing is everything prescribed: the Academy generator, a net asset earned rate off a
modelled asset portfolio, the standard projection amount, prescribed reinvestment, revenue
sharing, the deterministic floor, and aggregation across the rest of the company. The output is
comparable with itself across scenarios and assumptions, which is what the experiments need, and
is not a filed number.

**The surrender value floor** is the part that earns the section. Statutory reserves for this
business are floored at the cash surrender value, taken here as 98.0% of account value from the
disclosed ratio, so a block whose guarantees are worth nothing still has to be reserved at roughly
the account value. The floor applies to the total policy reserve against a gross quantity: an
earlier version compared the accumulated deficiency, which is already net of assets held, against
the surrender value, which is gross, and the floor then bound on 99.8% of scenarios at 115% of
premium - a number that says only that two balance-sheet quantities were put on one axis.

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
