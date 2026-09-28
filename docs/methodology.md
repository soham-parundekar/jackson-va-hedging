# Methodology

## Notation

$AV_t$ contract value at policy anniversary $t$, $B_t$ the benefit base (the Guaranteed
Withdrawal Balance in the contract's language), $g$ the guaranteed annual withdrawal
percentage, $\phi$ the rider charge as a percentage of the benefit base, $m$ the continuous
proportional drag on the contract value, $r$ the risk-free rate, $\sigma$ volatility,
${}_tp_x$ the probability that a life aged $x$ survives $t$ years, $D(t)$ the discount
factor.

## Contract value

Between anniversaries the contract value grows at the sub-account's total return less a
proportional drag:

$$\frac{dAV_t}{AV_t} = (r_t - m)\,dt + \sigma_t\,dW_t^{Q}$$

Under the risk-neutral measure any traded portfolio earns the risk-free rate, so the drift
is $r$ and not a real-world expected return. That distinction is the whole difference
between a valuation and a financial planning projection, and it is why the model says a
5.75% lifetime draw exhausts the account on 86% of paths by year twenty. At a real-world
equity drift it would not.

$m$ is the base contract charge of 1.31% plus fund expenses of 0.95%. Both are quoted
against average daily value, so both are continuous proportional drags.

### Annual stepping is exact

Nothing happens to the contract value between anniversaries except growth and the drag, so
over each policy year it is a geometric Brownian motion. Stepping annually with exact
lognormal increments therefore introduces no discretisation error in the contract value:

$$AV_{t^-} = AV_{t-1}\exp\left(\int_{t-1}^{t} r_u\,du - m\,\Delta - \tfrac{1}{2}v_t + \sqrt{v_t}\,Z_t\right)$$

where $v_t$ is the forward variance over the interval and $Z_t$ is standard normal. Every
path-dependent event in the contract lands on an anniversary: the step-up, the withdrawal,
the point of exhaustion, the switch from fee income to claims. Those are exactly the dates
being stepped to. Monthly stepping would cost forty-five times the compute and buy nothing.

The integrated forward rate comes straight from the zero curve, so a non-flat curve is
handled without approximation.

### Anniversary mechanics

In order, at each anniversary:

1. **Rider charge**, $\min(\phi B_{t-1},\, AV_{t^-})$, deducted from the contract value. It
   is charged on the benefit base, not the contract value, and it stops once the contract
   value reaches zero.
2. **Withdrawal** of $u \cdot g B_{t-1}$, where $u$ is utilisation. The contract value
   covers what it can and the insurer pays the shortfall:
   $$C_t = \max\left(u g B_{t-1} - AV_t^{\text{after charge}},\ 0\right)$$
3. **Step-up**: $B_t = \max(B_{t-1},\, AV_t)$ on the Contract Anniversary Value method,
   applied after the withdrawal.

Once the contract value reaches zero the benefit base stops moving, so the guaranteed
amount is frozen and the insurer pays it for as long as the owner lives. No fees are
collected after that point. The combination is what makes the guarantee expensive: income
and outgo never overlap.

The guaranteed percentage is locked at 5.75%, the band for ages 70 to 74. Real contracts may
re-band upward with attained age, to 5.95% at 75 and 6.20% at 81, so the model understates
the guarantee. The direction of that bias is stated rather than corrected, since the
prospectus language on re-banding is conditional.

### Measuring continuous charges exactly

Attributing fees requires knowing how much of the drag was the insurer's revenue, which is
awkward when the drag is continuous and the balance is stochastic. There is a clean answer.
For a proportional charge $c$ out of total drag $m$ over an interval of length $\Delta$, the
value at the end of the interval of the charges collected during it is exactly

$$\frac{c}{m}\,AV^{\text{before drag}}\left(1 - e^{-m\Delta}\right)$$

Charges left invested in the sub-account grow at the risk-free rate under $Q$, so
discounting that end-of-interval amount gives the same present value as discounting the
continuous stream. No sub-stepping, no approximation. `tests/test_engine.py` checks the
closed form against a brute-force calculation that splits the year into twenty thousand
intervals, on two hundred random paths.

## Valuation

The quantity valued is the market risk benefit on Jackson's own definition: projected
benefits less attributed fees.

$$V = \underbrace{\mathbb{E}^{Q}\left[\sum_t D(t)\,{}_tp_x\,C_t\right]}_{\text{claims}}
      - \alpha \underbrace{\mathbb{E}^{Q}\left[\sum_t D(t)\,{}_tp_x\,(F_t^{\text{rider}} + F_t^{\text{M\&E}})\right]}_{\text{attributable fees}}$$

$\alpha$ is the attribution percentage, fixed at inception as

$$\alpha = \min\left(1,\ \frac{PV(\text{claims at inception})}{PV(\text{attributable fees at inception})}\right)$$

and then held static for the life of the contract. That is what Note 6 describes: a portion
of total projected fees attributed to the benefit to offset projected claims, expressed as
a percentage of total projected fees, capped at 100%. Where the attributable fees cover the
claims the benefit starts at a fair value of zero; where they do not, a liability is
recognised at issue.

Holding $\alpha$ static is not a detail. A contract written in September 2016, with the
ten-year zero at 1.61%, calibrates to $\alpha = 1.00$ and starts as a liability. The same
contract written at 31 December 2025, with the zero at 4.20%, calibrates to $\alpha = 0.85$
and starts at zero. Freeze the percentage and a decade of rising rates and rising markets
turns the first contract into a net asset, which is the mechanism behind a disclosed
liability that is now a $4.2bn asset.

Fund expenses are excluded from attributable fees. They are paid to the funds, not to the
insurer.

Mortality is applied as an expectation rather than simulated. For a single representative
policy the two are identical in the mean, and the point of the exercise is the mean.

Surrender runs alongside mortality as a second decrement, but only while the contract value
is above zero, because an exhausted contract has no surrender value to take. That makes
persistency path-dependent, so it is carried per path rather than folded into the survival
curve.

## Discount curve

Treasury par yields at eight tenors, interpolated onto a semiannual grid, then bootstrapped
forward:

$$D(t_n) = \frac{1 - \frac{c_n}{2}\sum_{i<n} D(t_i)}{1 + \frac{c_n}{2}}, \qquad
  z(t) = -\frac{\ln D(t)}{t}$$

Below one year the par curve is held flat at the one-year level, since that is the shortest
tenor in the set used.

Rate shocks are applied to the **par yields** and the curve is bootstrapped again, which is
what a parallel shift in the yield curve means operationally. Shifting the bootstrapped
zeros instead gives different discount factors; the difference is small but there is no
reason to accept it. A 100bp par shift moves the zero curve by 95 to 106bp depending on
tenor, and the tests pin that range.

## Volatility term structure

Note 6 says implied volatility out to five years, grading to a historical level by year
ten, with an explicit risk margin in the long-run level. Free data gives implied volatility
at one month and three months and nothing longer.

The first implementation held the 3-month level flat to five years and graded linearly from
there. It is wrong in a way that only shows up under stress. In March 2020 the 3-month index
reached roughly 70 against a long-run level near 19, and a linearly declining *spot*
volatility through the grading window implied total variance to seven years below total
variance to five, which is negative forward variance and an arbitrage. Worse, the shape
propagated a fifty-point spike in a three-month index undamped across the entire front of
the surface, which no volatility surface does, and produced a vega roughly twice what it
should be. In a hedging backtest that error does not stay put: it dominates the residual.

The replacement specifies the shape on **forward** variance, mean reverting:

$$v(t) = v_{\infty} + (v_0 - v_{\infty})e^{-\kappa t}, \qquad
  V(T) = \int_0^T v(u)\,du = v_{\infty}T + (v_0 - v_{\infty})\frac{1 - e^{-\kappa T}}{\kappa}$$

Forward variance is positive by construction, total variance is monotone, and the integral
is closed form. Spot volatility is $\sqrt{V(T)/T}$.

$\kappa$ is pinned to Jackson's own statement rather than fitted: the front level grades
98% of the way to the long-run level by year ten, giving $\kappa = -\ln(0.02)/10 = 0.391$
and a half life of 1.8 years.

$v_0$ is fitted to the 3-month index alone. It is the longer of the two free observations
and therefore the more informative one for a liability with twenty-year duration, and a
one-parameter curve fitted to both would match neither. The 30-day index is then an
out-of-sample check: `build_dataset.py` compares the one-month spot volatility the fitted
curve implies against the observed VIX over 503 dates. The curve sits about two volatility
points above VIX on average, because a mean-reverting curve cannot reproduce a steep
one-month to three-month slope, with errors running from -17 points in inverted markets to
+6 in normal ones. That bias is documented, not corrected.

$v_{\infty}$ is realised volatility of the S&P 500 over the available history, 18.1%, plus a
one-point risk margin, giving 19.1%.

Two vegas come out of this, and the distinction matters for a hedging desk. Moving the front
level is tradeable and decays with maturity. Moving the long-run level is an assumption
change and, for a forty-year liability, carries more of the exposure: at 31 December 2025 the
front-level vega is $333 per volatility point against $569 for the long-run level. Most of
the volatility exposure in a lifetime guarantee sits beyond any listed option maturity.

## Sub-account

The valuation carries one risky sub-account with equity beta $\beta$, so its volatility is
$\beta\sigma$ and an index shock $s$ moves the contract value by $\beta s$. For the
standalone valuation $\beta = 1$. For the hedging backtest and the book comparison, $\beta$
is the equity exposure of Jackson's disclosed fund mix, $0.7235 + 0.6 \times 0.1832 = 0.833$.

In the backtest the realised sub-account return is built from the full mix rather than from
$\beta$ alone: equity and balanced funds on the index, bond funds at accrual less duration
times yield change with duration six, money market at the short rate. The model sees a
single-factor sub-account and the policy experiences a blend, which is genuine basis risk
rather than a modelling convenience. Treating the non-equity sleeve as riskless inside the
valuation understates total sub-account volatility slightly; the direction is stated.

## Mortality

$$q(x, y) = q^{2012}(x)\,(1 - G2_x)^{\,y - 2012}$$

Generational, so the rate applying to attained age $x + k$ in calendar year $y + k$ carries
$y + k - 2012$ years of improvement. Scale G2 stops at age 105, where the published scale
has already trended to zero, so no improvement is applied above it.

Sexes are blended at the **survival** level, not the rate level. A 50/50 book is two
populations, and averaging mortality rates before compounding gives the wrong expected
number of payments. `tests/test_mortality.py` checks that the blend is exactly the average
of the two survival curves and that the rate-blended alternative differs.

Within a policy year the force of mortality is constant, which makes survival log-linear
between integer durations. That is needed because an in-force policy is almost never valued
on its anniversary.

The projection is truncated at attained age 115. Extending it to 120 moves the market risk
benefit by $0.53 on a $100,000 policy against a Monte Carlo standard error of $16.90, and
cutting it to 110 moves it by $1.62. Truncation costs an order of magnitude less than
simulation noise.

## Simulation

200,000 paths for the valuation and the Greeks, 40,000 for the backtest. Antithetic
sampling, with the standard error taken across pair averages because the two halves of a
pair are dependent. Standard error on the market risk benefit at 200,000 paths is $16.90 on
a $100,000 policy, and it halves as paths quadruple, which the tests check.

One array of normals is drawn per run and reused across every revaluation, base and bumped.
Without that the difference between two valuations is dominated by simulation noise rather
than by the sensitivity: the level has a standard error of tens of dollars while a
one-basis-point rate bump moves the value by single dollars.

## Greeks

Central differences. Equity exposure is taken with respect to the log index level, because
that is directly the dollar notional a hedge has to carry:

$$\text{equity exposure} = \frac{V(S(1+h)) - V(S(1-h))}{\ln\frac{1+h}{1-h}}$$

The exact log-space denominator rather than $2h$; at $h = 1\%$ the difference is a hundredth
of a percent and costs nothing to get right. Rho is per basis point of parallel par shift,
vega per volatility point on the front level.

Signs, which decide whether the hedge is long or short. Equity exposure is negative: a
higher index makes the guarantee less likely to bite, so the liability falls. Rho is
negative: a long-dated liability discounts away faster when rates rise. Vega is positive.
Equity gamma is positive, so the loss from a fall exceeds the gain from a rise.

Equity exposure is not negative unconditionally, and the exception is worth knowing before
reading any hedge output. The benefit base steps up to the contract value on the anniversary,
so when the contract value sits above the benefit base the index level on that date fixes the
guaranteed income for the rest of the contract's life. Approaching the reset, the claims leg
turns long equity. Whether the net exposure turns long depends on the rate environment;
`docs/validation.md` decomposes it. Where the guarantee is at or in the money the exposure
stays short, and the tests assert that.

Rho has two channels and they pull the same way, which is why it is larger than duration
alone would suggest. Higher rates discount the claims away, and they also raise the
risk-neutral drift of the contract value, which delays exhaustion and reduces the claims
themselves. Note 12 lists exactly this: interest rate movements affect "both assumed future
separate account returns and discounting of cash flows". At issue the two channels together
give $-4,627 per 100bp on a $100,000 policy, against roughly $-2,700 from discounting alone.

## Hedge

Weekly, on the last trading day of each week. At each date the rider is revalued at the
market that prevailed, exposures are recomputed, and positions are resized. Nothing about
the following week enters the sizing, and the tests verify that by rebuilding each period's
hedge profit from the position recorded on the previous date.

**Equity leg.** Short index futures with dollar notional equal to the equity exposure.
Fully collateralised futures earn the excess return over financing, so the profit is
$N(\text{price return} + q\Delta - r\Delta)$ with $r$ the three-month bill.

**Rate leg.** Receive-fixed ten-year swap with DV01 equal to minus rho. Notional follows
from the swap annuity computed off the current curve. Profit is $-\text{DV01} \times \Delta y$
in basis points of the ten-year par yield. Rho is a parallel shift and the instrument is a
single tenor, so curve reshaping is left unhedged, which a real programme also carries.

**Volatility leg**, reported separately. Long volatility sized on the front-level vega,
profit $\text{vega} \times \Delta\sigma$ in points. A spread is charged on vega traded. No
carry is charged for holding a long volatility position, and implied volatility exceeds
realised on average, so the variance reduction this leg produces is informative while its
cumulative profit is flattering. Jackson does hold index put options, so the leg is not a
hypothetical instrument for this book.

**Costs.** 0.5bp round trip on traded index futures notional, 0.25bp on traded swap
notional, 2% of vega traded. Charged on the change in position, so a stable hedge is cheap.

Period profit:

$$\Pi = \alpha F - C - \Delta V + V_{t-1}r\Delta + \Pi^{\text{equity}} + \Pi^{\text{rates}} - \text{costs}$$

The financing term is the risk-free return on assets backing the liability. Without it a
week with no market movement shows the accretion of the liability as a loss. Futures profit
is already an excess return over financing, so the two sit on the same footing.

Effectiveness is the ratio of hedged to unhedged variance of period profit. Because each leg
is sized from its own Greek independently, one run with all legs computed gives every
subset, and the ledger records profit and cost per leg so subsets compose by addition.

## Economic against reported

The hedge is sized on the economic basis: Basic mortality, Treasury discounting. The
reported liability is the same contract on the reporting basis: Period mortality with its
margins, discounted on Treasury plus the company's own non-performance spread, taken as
0.6 times the Baa spread to reflect the higher ratings of the insurance subsidiaries.

Under the market risk benefit rules the movement attributable to own non-performance risk is
reported in other comprehensive income. So the reported movement splits: the change in the
own-credit adjustment goes to OCI, and the rest reaches net income. The comparison is
between the volatility of that net income figure and the volatility of the economic
outcome, on identical hedge positions.
