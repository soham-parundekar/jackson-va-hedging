# Data sources

Everything here is free and public. Three providers: SEC EDGAR for the filings, FRED for
market data, and the Society of Actuaries table repository for mortality.

## Jackson Financial filings

Central Index Key 1822993. Five annual reports have been filed since the 2021 spin-off
from Prudential plc; the three most recent supply everything used here.

| Fiscal year | Filed | Accession | Primary document |
|---|---|---|---|
| 2025 | 2026-02-24 | 0001822993-26-000022 | jxn-20251231.htm |
| 2024 | 2025-02-26 | 0001822993-25-000011 | jxn-20241231.htm |
| 2023 | 2024-02-28 | 0001822993-24-000009 | jxn-20231231.htm |

Sections used:

- **Item 1, Business.** Product descriptions and the benefit mix by account value.
  Guarantee fees charged on the benefit base rather than the account value, and the reason
  Jackson gives for it. The statement that GMIBs are no longer offered and the legacy block
  is reinsured. The two-programme structure: a core dynamic hedge against economic
  liabilities and a separate macro hedge for statutory capital.
- **Item 7A, Quantitative and Qualitative Disclosures about Market Risk.** The sensitivity
  tables, which are the validation target. Also the statement that hedging does not target
  U.S. GAAP liabilities.
- **Note 6, Fair Value Measurements.** The attributed-fee method. Discounting on Treasury
  rates in a stochastic projection. The volatility term structure: implied out to five
  years, grading to a historical level by year ten, with an explicit risk margin in the
  long-run level. Non-performance risk incorporated by adjusting the risk-free curve for
  the company's own credit spread.
- **Note 11, Separate Account Assets and Liabilities.** Account value roll-forward and the
  split by fund type, which gives the sub-account mix.
- **Note 12, Market Risk Benefits.** Balances, the roll-forward with its attribution to
  interest rates, fund performance, volatility and time, weighted-average attained age, and
  net amount at risk.

### Contract terms

Terms come from the product filings rather than the annual report, because the 10-K does
not publish charge levels.

- **Perspective II statutory prospectus**, Form 485BPOS filed 2026-04-21 by Jackson
  National Separate Account - I (CIK 927730), accession 0000927730-26-000193, document
  `ck0000927730-20260421.htm`. Fee table, step-up mechanics, the GWB adjustment provision,
  what happens when the contract value reaches zero, and the For Life Guarantee.
- **Rate Sheet Prospectus Supplement dated 27 April 2026**, Form 497 filed 2026-04-09,
  accession 0000927730-26-000157, document `jnlpiiafter6-24x19rateshee.htm`. Current annual
  charges and the guaranteed annual withdrawal percentages by age band. Maximum charges in
  the prospectus are not current charges, and the difference is large: the Flex GMWB
  maximum is 3.00% of the benefit base against a current Core charge of 1.25%.

Two passages settle mechanics that a plausible-looking model would otherwise get wrong.
The charge is paid "through the earlier date that you annuitize the Contract or your
Contract Value is zero", so fee income stops exactly when claims start. And with the For
Life Guarantee in effect, once the contract value is exhausted "the Owner will receive
annual payments of the GAWA until the death of the Designated Life", so the tail is a life
annuity rather than a run-off of the remaining benefit base. The same prospectus contains a
different paragraph describing payments that continue only until the benefit base is
depleted; that one applies where the For Life Guarantee is not in effect, which at age 70
it always is.

### Extracted tables

Two CSV files in `data/raw/` hold figures read out of the filings by hand. Both are checked
in `scripts/build_dataset.py`.

- `jackson_disclosed_sensitivities.csv`: fair value and shock impacts for market risk
  benefits and for fixed index and RILA embedded derivatives, by balance-sheet date, with
  the filing each figure came from. Six figures appear in two consecutive filings, and the
  build script fails if any of them disagree.
- `jackson_book_statistics.csv`: account value, cash surrender value, the fund-type split,
  net market risk benefit balances, weighted-average attained age, net amount at risk, and
  the benefit mix percentages.

A reconciliation gap worth stating. The market risk benefit figure in the Item 7A
sensitivity table is not identical to the variable annuity net balance in Note 12: at 31
December 2025 the sensitivity table shows $(4,238)m while Note 12 shows $(4,265)m for
variable annuities and $(4,113)m in total. Item 7A covers market risk benefits across
product lines while Note 12 splits them, and the presentations do not tie exactly. Since
the sensitivities are only published alongside the Item 7A figure, that is the one used,
and sensitivities are scaled by variable annuity account value from Note 11. Scaling a
multi-product liability by a single-product account value is a known imprecision; variable
annuities dominate the balance, so it is small, but it is not zero.

## Market data, FRED

One daily panel, `data/raw/fred_daily_panel.csv`, 2 January 2015 to 25 September 2026,
2,986 rows after dropping days on which no equity, rate or volatility observation exists.

| Series | What it is | Used for |
|---|---|---|
| SP500 | S&P 500 index | Equity proxy for the sub-account and for the futures hedge |
| VIXCLS | CBOE Volatility Index, 30 day | Out-of-sample check on the variance mapping |
| VXVCLS | CBOE 3-Month Volatility Index | Implies the instantaneous variance along the replay |
| DGS1 … DGS30 | Treasury par yields, 8 tenors | Bootstrapped to the zero curve |
| DTB3 | 3-month Treasury bill | Financing rate for the futures position |
| BAA10Y | Moody's Baa corporate spread over the 10-year Treasury | Own non-performance spread proxy |
| AAA10Y | Moody's Aaa spread over the 10-year Treasury | Cross-check on the spread proxy |
| DFF | Effective federal funds rate | Context |
| BAMLC0A4CBBB | ICE BofA BBB corporate option-adjusted spread | Cross-check, but see below |

Three more files sit beside it because the panel's ten-year window cannot carry them:

| File | What it is | Used for |
|---|---|---|
| `fred_long_rate_history.csv` | DGS3MO and DGS10, full history | Hull-White mean reversion and volatility, which ten years cannot identify |
| `fred_financing_rates.csv` | Overnight and bill rates | The financing leg of the futures and total return swap positions |
| `cboe_spx_option_chain.csv`, `cboe_spx_parity_quotes.csv` | SPX chain and the quotes used for put-call parity | The Heston calibration, and the forward and discount factor per expiry |
| `cboe_vix6m_skew.csv` | VIX6M and SKEW | Independent check on the calibrated term structure and skew |

### Gotchas that matter

**The S&P 500 series on FRED is a rolling ten-year window.** It begins 26 September 2016
and there is no way to extend it from this source, which sets the start of the backtest.

**BAMLC0A4CBBB is also windowed**, and far more tightly: 775 observations from September
2023. It cannot span the 2020 stress, so the own-credit proxy is BAA10Y, which has the full
history. Both are kept in the panel so the two can be compared where they overlap.

**Three different holiday calendars.** Treasury yields publish on federal business days,
the S&P 500 on NYSE trading days, and the volatility indices occasionally lag a day.
Columbus Day and Veterans Day are the reliable offenders: the bond market shuts and the
stock market opens. The loader keys off S&P 500 observations, since those are the days a
hedge is actually rebalanced, and carries rates and volatility forward by at most five
days. A longer gap raises rather than filling. Over 2,514 equity trading days nothing is
missing after that fill.

**Treasury publishes par yields, not zero rates.** A forty-year cash flow discounted
straight off a thirty-year par yield is wrong by enough to matter. Everything discounts off
the bootstrapped zero curve, and `build_dataset.py` reprices the par bonds off every one of
the 2,495 bootstrapped curves as a check; worst round-trip error is 2e-16. The nineteen
equity trading days that do not get a curve are days with an incomplete par curve, and they
are dropped rather than filled: a Treasury holiday is a day on which nothing traded, so
carrying the previous curve forward would invent a mark the hedge could not have traded on.

**The index is a price index.** Sub-account returns are total returns, so a dividend yield
enters explicitly at 1.5%, stated as an assumption and tested. It appears twice: in the
realised sub-account return and in the excess return on the futures position.

**The option chain stops at 3.23 years and the liability runs 45.** Cboe's delayed-quote
endpoint publishes the live SPX chain, which is what the Heston surface is calibrated to, and
its longest standard expiry on the retrieval date was 3.23 years out. Nothing free reaches
further. The parameter that matters most on a forty-five-year guarantee is therefore the one
the data has least to say about, and `docs/limitations.md` gives the measured size of the
problem rather than leaving it as a worry.

**The chain is a single snapshot, not a history.** It was retrieved once, on 28 September
2026, so the surface is calibrated at one date and held at that shape along the whole replay.
What moves with the date along the replay is the curve and the observable instantaneous
variance implied from the three-month index.

## Mortality, SOA

Society of Actuaries Mortality and Other Rate Tables, exported as CSV from the table
repository and stored in `data/raw/soa_2012_iam_g2.csv`.

| Table identity | Name |
|---|---|
| 2581 | 2012 IAM Basic Table, Male, age nearest birthday |
| 2582 | 2012 IAM Basic Table, Female |
| 2583 | Projection Scale G2, Male |
| 2584 | Projection Scale G2, Female |
| 2585 | 2012 IAM Period Table, Male |
| 2586 | 2012 IAM Period Table, Female |

Source reference on the SOA records: Life Experience Subcommittee, "2012 Individual Annuity
Reserving Table", report from the joint American Academy of Actuaries and Society of
Actuaries Payout Annuity Table Team (2011), Exhibit I.

The Basic tables are best estimate and carry the economic valuation. The Period tables are
the Basic tables with the margins the NAIC's Life Actuarial Task Force set, and they stand
in for the reporting basis, since Note 6 says the fair value uses best estimate assumptions
plus risk margins. The export leaves a few young female ages blank; everything above age 40
is complete, which is all this project uses, and the loader refuses to proceed if a gap
appears above 40.

## Reproducibility

`python -m scripts.build_dataset` validates every input and writes the processed panel and
the bootstrapped curve history. It fails rather than warns on a par curve that will not
bootstrap, a mortality table with a hole above age 40, a Period table that implies shorter
life than the Basic table, or a disclosed figure that disagrees between two filings.

The raw files are committed. The FRED series are revised occasionally and the S&P 500
window rolls forward, so a pull on a later date will not reproduce the committed panel
exactly. Committing the data is what makes the results reproducible rather than
approximately repeatable.
