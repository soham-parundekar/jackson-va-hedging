# References

What each source is for, and what it does not settle. `source_register.csv` carries the
machine-readable version: identifier, locator, retrieval date, and the file or module that
depends on each row. This file says why the row is there.

Bibliographic details for every journal article and working paper were read off the Crossref
record for the DOI rather than written from memory. Two did not resolve that way and say so
below.

## The issuer

**REF-001, REF-002** Jackson Financial's 10-Ks for 2025 and 2022. Item 7A is the only public
statement of what the guarantee book is worth and how it moves: the market risk benefit, the
sensitivities to a given equity fall and rate shift, and the derivative notionals by
instrument. Everything the project claims to reproduce is in those two tables.

The two filings are not comparable in level and the register says so. 2022 is pre-LDTI and
uses a 50 basis point rate shift; 2025 is post-LDTI at 100 basis points. Reading one shock
size off the other would misstate the rate sensitivity by a factor of two, so the shock size
is stored as a property of the filing rather than inferred from the date.

Two details from the 2025 filing shaped decisions rather than just supplying numbers. The
separate-account fund split sets the equity weight that converts the liability's delta into an
index position, which is an 18 per cent error in the hedge ratio if ignored. And the equity
option book's average remaining term of 0.24 years is what prompted the sweep that moved the
hedge's put leg off a one-year tenor.

**REF-003** The 8-K announcing Brooke Re. It is the one place Jackson states in its own words
that the cash surrender value floor imposes a cost that is not economic, and that moving the
riders to a captive is meant to remove it. The statutory lens exists because of that
statement, so the filing is a premise rather than a citation.

**REF-004, REF-005** The Perspective II prospectus and its rate sheet supplement. The
prospectus gives the benefit mechanics: how the bonus applies, what a step-up resets, how the
guaranteed withdrawal base is adjusted against premium, and the four death benefits. The rate
sheet gives the numbers, and it has to be the rate sheet rather than the prospectus, because
the prospectus describes the rider and the supplement sets the charge and the age-banded
withdrawal rates for contracts issued in a given window.

One mechanic in the prospectus cost a correction. The bonus is simple, not compound: applying
it does not feed the bonus base, and only a step-up that itself lifts the guaranteed
withdrawal base does. The first implementation compounded it, which is 179 against 160 over a
ten-year deferral.

**REF-006** The SEC's XBRL company-facts API. The 10-K extracts give two year-ends; the API
gives the quarterly series, which is what the model book is scaled against and what the
hedging results are checked against between annual filings.

## Market and actuarial data

**REF-007, REF-008** FRED. The daily panel is the whole market history the project runs on:
the index, eight Treasury tenors, two volatility indices, two credit spreads and two short
rates. Its binding limitation is a licence rather than a gap. The S&P 500 series is a rolling
ten-year window beginning 26 September 2016, which is what sets the start of the backtest and
what puts the dot-com unwind, 2008 and August 2011 out of reach. The long rate history is held
separately for one reason: Hull-White mean reversion cannot be identified from ten years, and
45 years of the 3-month bill can. A third FRED pull, SOFR and the 3-month bill, was retrieved
and then dropped: it starts in April 2018 and so cannot cover a replay that starts in 2016, and
where it does overlap it sits a fifth of a basis point from the fed funds rate already in the
panel. The measurement behind that decision is in `docs/data_sources.md` rather than left as a
judgement.

**REF-010, REF-011** Cboe. The SPX chain is a single delayed snapshot, 28 September 2026 at
14:14 Eastern with the index at 7710.31, and the whole volatility model is calibrated to it.
A single snapshot is a real limitation, stated rather than hidden: it means the Heston
parameters are one day's surface, and the crisis replays hold the shape of that surface fixed
while feeding the realised volatility level along the path. The index histories are the
independent check, since VIX6M and SKEW are published daily and the fitted term structure and
skew have to agree with them.

**REF-012, REF-013** The Society of Actuaries tables. Two bases, and which one applies decides
a number: the Basic tables are best estimate and carry the economic valuation, the Period
tables are the Basic tables with the margins the Life Actuarial Task Force set and stand in
for the reporting basis, because Note 6 says the fair value uses best estimate assumptions
plus risk margins. REF-013 is the report that establishes that distinction; it is cited and
not stored, and the table identities in REF-012 are what tie it to the data.

**REF-014** ASU 2018-12. The market risk benefit and the attributed-fee method are definitions
rather than modelling choices, and the own-credit component going to other comprehensive
income is why the reporting lens needs a split the economic lens does not have.

## Models

**REF-015** Heston (1993). The variance process and the closed-form characteristic function.

**REF-016** Hull and White (1990). The one-factor short rate, and more usefully the closed-form
bond price, which is what makes a nested node exact rather than approximate: the whole zero
curve an observer sees at a node follows from that node's short rate with nothing simulated.

**REF-017** Longstaff and Schwartz (2001). The regression proxy. The project departs from the
usual basis and from what the brief specified, and the departure is measured rather than
asserted: a cubic polynomial in log-moneyness fits the level acceptably and hedges badly, and
a natural cubic spline in the ratio the other way up fixed the delta and improved the values
at every horizon. What no basis fixed is the second derivative, which is why the convexity
comes from nested valuation instead.

**REF-018** Fang and Oosterlee. The COS method, used for the calibration objective and as the
benchmark the simulation step size was chosen against. In print it is 2009 and it posted online
in 2008, which is why half the literature cites it as 2008; the register carries both.

**REF-019** Andersen (2008). The quadratic-exponential variance scheme. The calibrated
parameters violate the Feller condition at minus 2.72, so the variance process reaches zero and
an Euler discretisation of it is not merely inaccurate but ill-defined. The martingale
correction is the part that matters in practice here: without it the discounted index drifts up
about three per cent by year forty-two, which on a forty-year guarantee is not a rounding error.

**REF-020** Albrecher, Mayer, Schoutens and Tistaert (2007), on the branch of the complex
logarithm in the Heston characteristic function. No DOI is registered with Crossref for this
one; the locator in the register is the University of Lausanne repository record, which gives
Wilmott 2007 issue 1, pages 83 to 92. The paper is load-bearing rather than decorative: the
other branch is numerically unstable at long maturities, and a forty-year liability is nothing
but long maturities.

**REF-021, REF-022** Nelson and Siegel (1987) and Svensson (1994). The curve fit is Svensson's
extension of the Nelson-Siegel form. Svensson's paper exists in two versions, the NBER working
paper and IMF Working Paper 94/114; both DOIs are in the register. The fit needed anchoring
past thirty years, and that is a consequence of the functional form rather than a bug:
unconstrained it matched every quoted Treasury tenor to three basis points and extrapolated to
a long-run level of minus twenty-six per cent.

**REF-023** Politis and Romano (1994). The stationary bootstrap, used to generate paths that
are not drawn from the calibrated model. Geometric blocks rather than fixed ones, which is what
keeps the resampled series stationary, and the one parameter is the mean block length, read off
the persistence of squared returns at 11.1 days.

## What is cited and what is not

Journal articles are cited, never stored. The filings and prospectuses are public records and
are preserved here as verbatim extracts of the sections relied on, each under a header giving
the accession number, the permanent SEC URL and the retrieval date, so the extract can be
checked against the original. Retrieved data sits under `data/raw/` rather than behind a
download script, because the environment this was built in cannot reach sec.gov, FRED or Cboe
from a script; `docs/data_sources.md` says the same thing from the data side.

A source that informed nobody's judgement is not listed. Two entries were written and then
removed on that basis: a nested-simulation paper that supported the design of the nested
standard in spirit but that no formula or parameter choice in the repository actually uses, and
a SOFR series whose role turned out to be covered by a rate already in the panel.

**REF-024, REF-025 are identified but not fully.** The FY2023 and FY2024 10-Ks each supply four
of the rows the shock validation runs on, so they belong here, and for a long time they were
missing from the register entirely while their figures were in the data. They are listed now
with what can be established from a source that is committed: the company, the form, the period
and the filing date, all four read off the XBRL pull in REF-006, which records the filing date
of every 10-K Jackson has filed. Their accession numbers are not among those facts and have not
been retrieved, so the identifier field says that rather than carrying a number written from
memory, and no extract is stored. Two consequences are worth stating plainly. The FY2024
filing's embedded-derivative sensitivity at 31 December 2023 is in that filing and not in
`data/raw/jackson_disclosed_sensitivities.csv`, because transcribing it would mean writing a
figure into a primary-source file without having read it there - it changes nothing computed,
since no derivative table is disclosed at that date, and `tests/test_disclosures.py` names the
gap so it cannot be mistaken for completeness. And the figures those two filings do supply are
checked against the overlapping filings rather than taken on trust: every date two filings both
disclose agrees exactly, which is what the duplicate rows in that file exist to let anyone
confirm.
