# References

Two kinds of source sit behind this project and they are separated here deliberately.

**Primary sources** were retrieved and read for this work. Every number in the model's
contract specification, every disclosed sensitivity, and every market and mortality
observation comes from one of them.

**Method references** locate the modelling choices in the academic literature. They were
not re-read for this project, and nothing here paraphrases their contents beyond what the
titles and the standard results support. Each has been checked against Crossref so the
bibliographic details are right rather than remembered.

## Primary sources

### Jackson Financial Inc., Form 10-K

Filed with the SEC under Central Index Key 1822993.

- Fiscal year 2025, filed 24 February 2026, accession 0001822993-26-000022.
- Fiscal year 2024, filed 26 February 2025, accession 0001822993-25-000011.
- Fiscal year 2023, filed 28 February 2024, accession 0001822993-24-000009.

Sections used: Item 1 for the product mix and the two-programme hedging structure, Item 7A
for the sensitivity tables and the statement that hedging does not target U.S. GAAP
liabilities, Note 6 for the attributed-fee method and the volatility and non-performance
assumptions, Note 11 for the separate account and its fund split, Note 12 for market risk
benefit balances, the roll-forward and the attained age.

### Jackson National Separate Account - I product filings

Central Index Key 927730.

- Perspective II statutory prospectus, Form 485BPOS filed 21 April 2026, accession
  0000927730-26-000193, document `ck0000927730-20260421.htm`. Contract mechanics and the fee
  table.
- Rate Sheet Prospectus Supplement dated 27 April 2026, Form 497 filed 9 April 2026,
  accession 0000927730-26-000157, document `jnlpiiafter6-24x19rateshee.htm`. Current charges
  and the guaranteed annual withdrawal percentages by age band.

### Society of Actuaries mortality tables

Mortality and Other Rate Tables repository, table identities 2581, 2582, 2583, 2584, 2585
and 2586. The repository records the source as: Life Experience Subcommittee, "2012
Individual Annuity Reserving Table", report from the joint American Academy of Actuaries and
Society of Actuaries Payout Annuity Table Team (2011), Exhibit I.

### Market data

Federal Reserve Bank of St. Louis, FRED. Series SP500, VIXCLS, VXVCLS, DGS1 through DGS30,
DTB3, BAA10Y, AAA10Y, DFF and BAMLC0A4CBBB. `docs/data_sources.md` lists what each one is
for and the constraints each carries.

## Method references

Bauer, D., Kling, A., and Russ, J. (2008). A universal pricing framework for guaranteed
minimum benefits in variable annuities. *ASTIN Bulletin* 38(2), 621-651.
[10.2143/ast.38.2.2033356](https://doi.org/10.2143/ast.38.2.2033356)

> The standard reference for the risk-neutral valuation framework used here, and the source
> of the static-withdrawal benchmark. This project takes full utilisation of the guaranteed
> amount as its base case for the reason that paper gives: it is the assumption that makes
> the guarantee most expensive, so it is a bound rather than a central estimate. The
> reconciliation in `docs/validation.md` is built around that point.

Milevsky, M. A., and Salisbury, T. S. (2006). Financial valuation of guaranteed minimum
withdrawal benefits. *Insurance: Mathematics and Economics* 38(1), 21-38.
[10.1016/j.insmatheco.2005.06.012](https://doi.org/10.1016/j.insmatheco.2005.06.012)

> The early valuation treatment of the GMWB specifically, and the source of the result that
> guarantee fees charged in the market have often sat below the risk-neutral cost of the
> guarantee. The at-issue result here, where attributable fees cover projected claims 1.15
> times over only once the base contract charge is included alongside the explicit rider
> charge, is a version of the same arithmetic.

Dai, M., Kwok, Y. K., and Zong, J. (2008). Guaranteed minimum withdrawal benefit in variable
annuities. *Mathematical Finance* 18(4), 595-611.
[10.1111/j.1467-9965.2008.00349.x](https://doi.org/10.1111/j.1467-9965.2008.00349.x)

> Treats the withdrawal decision as a control problem rather than a fixed schedule. This
> project assumes a fixed schedule, so that paper describes the direction in which the
> assumption errs.

Chen, Z., Vetzal, K., and Forsyth, P. A. (2008). The effect of modelling parameters on the
value of GMWB guarantees. *Insurance: Mathematics and Economics* 43(1), 165-173.
[10.1016/j.insmatheco.2008.04.003](https://doi.org/10.1016/j.insmatheco.2008.04.003)

> On how sensitive a GMWB valuation is to its inputs, which is the reason the robustness
> table in `reports/tables/valuation_robustness` exists and is reported alongside the
> headline number rather than after it.

Kling, A., Ruez, F., and Ruß, J. (2014). The impact of policyholder behavior on pricing,
hedging, and hedge efficiency of withdrawal benefit guarantees in variable annuities.
*European Actuarial Journal* 4(2), 281-314.
[10.1007/s13385-014-0093-0](https://doi.org/10.1007/s13385-014-0093-0)

> The closest paper to this project's central finding. Policyholder behaviour affects not
> only the level of the guarantee but the efficiency of hedging it, which is what the
> utilisation and lapse sweep here runs into: a near-constant factor of 2.4 between the
> model's sensitivities and Jackson's disclosed ones that closes on behaviour assumptions
> inside ordinary ranges.

Moenig, T., and Bauer, D. (2016). Revisiting the risk-neutral approach to optimal
policyholder behavior: a study of withdrawal guarantees in variable annuities. *Review of
Finance* 20(2), 759-794. [10.1093/rof/rfv018](https://doi.org/10.1093/rof/rfv018)
(published online 2015)

> On why observed behaviour departs from the risk-neutral optimum, including tax effects,
> which is relevant to why a static assumption overstates the guarantee in practice rather
> than merely in theory.

Hardy, M. R. (2003). *Investment Guarantees: Modeling and Risk Management for Equity-Linked
Life Insurance*. Wiley. ISBN 0-471-39290-1.

> The standard text for this product class, covering the valuation and the hedging of
> equity-linked guarantees together. Reviewed in *ASTIN Bulletin* 33(2), 439-448 and *North
> American Actuarial Journal* 8(3), 133-136.

## A note on what was verified

Every reference above was checked against Crossref for author list, journal, volume, issue,
pages and year. One error was caught that way: a Kling, Ruez and Russ paper carried in
memory as a 2011 ASTIN Bulletin article on stochastic volatility does not exist under those
details. The paper by those authors on hedge efficiency is the 2014 *European Actuarial
Journal* article listed above, and it is about policyholder behaviour rather than stochastic
volatility. The uncorrected version would have been a plausible-looking citation to nothing.
