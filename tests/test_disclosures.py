"""Arithmetic the filings themselves assert, re-run against the figures typed out of them.

Every other test in the suite checks code. These check transcription, which is the one failure
mode no amount of model testing reaches: a sensitivity table read off a 10-K by hand is the
project's only unvalidated input, and a single mistyped digit would propagate into every
comparison that uses it while leaving the code correct.

Each one is an identity the filing publishes alongside the numbers, so a failure says a line was
mistyped or missed and cannot say anything else. They run against the committed CSVs rather than
against the extracts in references/, because the CSVs are what the project reads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tests.checks import approx, raises
from vahedge import paths

from scripts.run_hedge_disclosure import (
    disclosed_offsets,
    load_derivatives,
    load_liabilities,
    transcription_identity,
)


def test_the_derivative_instrument_lines_sum_to_every_disclosed_total():
    """The check the FY2025 and FY2022 Item 7A extracts both claim. Eight blocks in each
    filing's two tables, two shock directions each, and the instrument rows have to reproduce
    the Total row exactly - the filings round to whole millions, so exactly is the right word."""
    identity = transcription_identity(load_derivatives())
    assert len(identity) == 16
    assert float(identity["difference_musd"].abs().max()) == 0.0
    # Two in the 2021 equity block, which held no total return swaps yet, and three everywhere
    # else. A block that lost a line would still sum to something, so the count is asserted too.
    assert int(identity["instrument_lines"].min()) == 2
    assert int(identity["instrument_lines"].max()) == 3


def test_the_calls_and_puts_breakdown_sits_inside_the_options_line():
    """The FY2022 equity table shows Options as one instrument line and then breaks it into
    Calls and Puts. Those two are a decomposition, not two more instruments, which is why they
    carry level='component' and are excluded from the sum above. If they were ever relabelled
    the totals would double-count the option book and nothing else would notice."""
    derivatives = load_derivatives()
    components = derivatives[derivatives["level"] == "component"]
    assert set(components["instrument"]) == {"calls", "puts"}
    for (as_of, shock), block in components.groupby(["as_of", "shock"]):
        options = derivatives[(derivatives["as_of"] == as_of)
                              & (derivatives["shock"] == shock)
                              & (derivatives["instrument"] == "options")]
        assert len(options) == 1
        assert approx(float(options["impact_musd"].iloc[0]), abs=0.5) == \
            float(block["impact_musd"].sum())


def test_futures_fair_values_are_recorded_as_nil_rather_than_missing():
    """The filings print a dash, and a dash is nil: a futures position settles daily so its
    carrying fair value is zero. Reading it as missing would drop the row out of the sum."""
    derivatives = load_derivatives()
    futures = derivatives[derivatives["instrument"].str.contains("futures")]
    assert not futures.empty
    assert futures["fair_value_musd"].notna().all()
    assert float(futures["fair_value_musd"].abs().max()) == 0.0


def test_the_two_filings_agree_on_every_date_they_both_disclose():
    """Six liability figures appear in two filings each. They have to match, and the duplicate
    rows are kept precisely so that this can be asserted rather than assumed."""
    liabilities = load_liabilities()
    keys = ["as_of", "line_item", "shock"]
    counts = liabilities.groupby(keys)["impact_musd"].agg(["nunique", "size"])
    repeated = counts[counts["size"] > 1]
    assert len(repeated) >= 6
    assert int(repeated["nunique"].max()) == 1


# The one place the committed table is knowingly short of the filings. The FY2024 10-K shows the
# embedded-derivative line at both of its dates and only the 2024-12-31 column was transcribed;
# the 2023-12-31 column has not been retrieved, and inventing it from the FY2023 filing's figure
# for the same date would be writing a number into a primary-source file that was not read off
# that source. It costs nothing computed - no derivative table is disclosed at 2023-12-31, so
# nothing divides by it - and it is listed here rather than left to make this test lie.
KNOWN_UNRETRIEVED = {(2024, "fia_rila_embedded_derivatives"): {"2023-12-31"}}


def test_every_filing_discloses_each_line_at_both_of_its_dates():
    """Item 7A publishes a current-year column and a prior-year column, so a line a filing shows
    at all it shows twice. Asserting that symmetry is what catches a column transcribed one date
    short: the FY2025 embedded-derivative rows existed at 2024-12-31 in the filing and not in
    this file, which silently reduced a both-lines offset ratio to the guarantee alone."""
    liabilities = load_liabilities()
    for filing, block in liabilities.groupby("source_filing_fy"):
        dates = set(block["as_of"])
        assert len(dates) == 2, f"FY{filing} should disclose two dates, has {sorted(dates)}"
        for line, rows in block.groupby("line_item"):
            expected = dates - KNOWN_UNRETRIEVED.get((int(filing), line), set())
            assert set(rows["as_of"]) == expected, (
                f"FY{filing} shows {line} at {sorted(set(rows['as_of']))} and not at "
                f"{sorted(expected)}")
            # And the same four shocks at each of the dates it does carry.
            per_date = rows.groupby("as_of")["shock"].apply(frozenset)
            assert per_date.nunique() == 1, f"FY{filing} {line} has uneven shocks by date"


def test_the_rate_shock_size_belongs_to_the_filing_and_not_the_date():
    """2024-12-31 is disclosed with a 50bp shift in the FY2024 filing and a 100bp shift in the
    FY2025 one. Any comparison that pairs a liability with a derivative table has to match on
    the filing as well as the date, and this is the fact that makes that necessary."""
    liabilities = load_liabilities()
    shared = liabilities[liabilities["as_of"] == "2024-12-31"]
    by_filing = shared.groupby("source_filing_fy")["shock"].apply(
        lambda column: {s for s in column if s.startswith("rates")})
    assert by_filing[2024] == {"rates_up_50bp", "rates_down_50bp"}
    assert by_filing[2025] == {"rates_up_100bp", "rates_down_100bp"}


def test_the_offset_ratio_is_one_when_the_hedge_exactly_covers_the_liability():
    """The sign convention, pinned on a case with a known answer. A liability impact is a change
    in a carrying amount and a derivative impact is a change in an asset's mark, so the two
    cancel in earnings when their raw signs agree - which makes a complete hedge +1 and not -1.
    This got written the other way round first and every ratio came out negative."""
    liabilities = pd.DataFrame([
        {"source_filing_fy": 2025, "as_of": "2025-12-31",
         "line_item": "market_risk_benefits", "fair_value_musd": -4238.0,
         "shock": "rates_up_100bp", "impact_musd": -1000.0},
    ])
    derivatives = pd.DataFrame([
        {"source_filing_fy": 2025, "as_of": "2025-12-31", "risk": "rates",
         "instrument": "total", "level": "total", "notional_musd": np.nan,
         "avg_term_years": np.nan, "fair_value_musd": 0.0,
         "shock": "rates_up_100bp", "impact_musd": -1000.0},
    ])
    exact = disclosed_offsets(liabilities, derivatives)
    assert approx(1.0, rel=1e-12) == float(exact["offset_of_guarantee"].iloc[0])

    half = derivatives.assign(impact_musd=-500.0)
    assert approx(0.5, rel=1e-12) == float(
        disclosed_offsets(liabilities, half)["offset_of_guarantee"].iloc[0])

    # A derivative book that moved against the liability rather than with it. Positive here
    # would mean a hedge; the ratio has to report it as negative.
    wrong_way = derivatives.assign(impact_musd=+400.0)
    assert float(
        disclosed_offsets(liabilities, wrong_way)["offset_of_guarantee"].iloc[0]) < 0.0


def test_a_derivative_table_with_no_liability_beside_it_is_refused():
    """Silently dropping the block would hide half the derivative data behind a filter. The
    pre-LDTI rows were skipped that way until the two bases were given a column of their own."""
    derivatives = pd.DataFrame([
        {"source_filing_fy": 2025, "as_of": "2025-12-31", "risk": "rates",
         "instrument": "total", "level": "total", "notional_musd": np.nan,
         "avg_term_years": np.nan, "fair_value_musd": 0.0,
         "shock": "rates_up_100bp", "impact_musd": -1000.0},
    ])
    empty = pd.DataFrame(columns=["source_filing_fy", "as_of", "line_item",
                                  "fair_value_musd", "shock", "impact_musd"])
    with raises(ValueError, match="no liability"):
        disclosed_offsets(empty, derivatives)


def test_the_committed_book_statistics_carry_the_dates_the_comparison_needs():
    """The scaling denominator. Separate account value has to exist on every date a disclosed
    sensitivity is scaled by it, or a comparison quietly loses a year."""
    book = pd.read_csv(paths.BOOK_STATISTICS, comment="#")
    liabilities = load_liabilities()
    post_ldti = liabilities[liabilities["line_item"] == "market_risk_benefits"]
    assert set(post_ldti["as_of"]) <= set(book["as_of"])
    assert book["va_separate_account_musd"].notna().all()
    # Cash surrender value sits below the account value on every date, which is what makes the
    # statutory floor a floor rather than a cap.
    assert (book["cash_surrender_value_musd"] < book["va_separate_account_musd"]).all()
