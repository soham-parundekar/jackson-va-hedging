"""Integrity of the committed artefacts, which no other test looks at.

Everything else in the suite checks code, or checks a number typed out of a filing. These check
the files the repository publishes. A figure is a binary blob that nobody reads and every viewer
renders identically whatever else is hiding in it, so a byte that should not be there survives
every other check in the project - including a text search of the repository, which is what let
one through here: ten figures reached GitHub carrying a 5,758-byte private chunk inserted by the
tool that copied them, invisible on screen and plainly visible to ``strings``.

The reference register gets the same treatment and for the same reason. Nothing executes it, so
nothing notices when a row goes missing or an identifier drifts from the document it names. Both
happened: six papers that justify the behaviour assumptions sat in `docs/literature.md` and in no
row, and an accession number for the prospectus was carried one digit out through two documents
while the stored extract and the register had it right.
"""

from __future__ import annotations

import csv
import os
import re
import struct

from tests.checks import approx, raises
from vahedge import paths

# Everything in the PNG specification plus the APNG extensions. Anything else is a private or
# vendor chunk: legal PNG, and not something this project writes.
STANDARD_CHUNKS = {
    b"IHDR", b"PLTE", b"IDAT", b"IEND", b"tRNS", b"cHRM", b"gAMA", b"iCCP", b"sBIT", b"sRGB",
    b"tEXt", b"zTXt", b"iTXt", b"bKGD", b"hIST", b"pHYs", b"sPLT", b"tIME", b"eXIf",
    b"acTL", b"fcTL", b"fdAT",
}

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def chunks(raw: bytes) -> list[tuple[bytes, int]]:
    """Every chunk in a PNG as (type, payload length), in file order.

    Raises rather than returning a partial list on anything malformed, because a figure that
    does not parse is a corrupt commit and not a soft finding.
    """
    if raw[:8] != PNG_SIGNATURE:
        raise ValueError("not a PNG")
    found, position = [], 8
    while position < len(raw):
        if position + 12 > len(raw):
            raise ValueError(f"truncated chunk header at byte {position}")
        length = struct.unpack(">I", raw[position:position + 4])[0]
        kind = raw[position + 4:position + 8]
        if position + 12 + length > len(raw):
            raise ValueError(f"{kind!r} claims {length} bytes and the file ends first")
        found.append((kind, length))
        position += 12 + length
        if kind == b"IEND":
            break
    if not found or found[-1][0] != b"IEND":
        raise ValueError("no IEND")
    if position != len(raw):
        raise ValueError(f"{len(raw) - position} bytes after IEND")
    return found


def test_every_committed_figure_carries_only_the_chunks_a_plot_needs():
    figures = sorted(paths.FIGURES.glob("*.png"))
    assert figures, "no committed figures to check"
    for figure in figures:
        extra = [kind.decode("latin1") for kind, _ in chunks(figure.read_bytes())
                 if kind not in STANDARD_CHUNKS]
        assert not extra, f"{figure.name} carries {extra}"


def test_the_chunk_reader_refuses_a_file_with_something_appended():
    """The failure mode worth pinning: a valid image with a payload stuck on the end."""
    original = sorted(paths.FIGURES.glob("*.png"))[0].read_bytes()
    assert chunks(original)[-1][0] == b"IEND"
    with raises(ValueError):
        chunks(original + b"anything at all")
    with raises(ValueError):
        chunks(original[:-4])


def _register() -> list[dict]:
    """The source register's rows, past its comment header."""
    lines = (paths.REFERENCES / "source_register.csv").read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("ref_id,"))
    return list(csv.DictReader(lines[start:]))


def test_every_register_row_points_at_something_that_exists():
    rows = _register()
    assert len(rows) > 20
    for row in rows:
        stored = row["stored"]
        local = [part.strip() for part in row["local_path"].split(";") if part.strip()]
        where = f"{row['ref_id']} ({stored})"
        for name in local:
            assert (paths.ROOT / name).exists(), f"{where} names {name}, which is not there"
        if stored == "cited":
            assert not local, f"{where} is not redistributed and should carry no path"
        if stored in ("extract", "in_full", "data"):
            assert local, f"{where} claims to be stored and names no file"
        assert row["used_by"].strip(), f"{where} names no component that depends on it"
        assert row["identifier"].strip(), f"{where} carries no identifier"


def test_every_source_the_literature_cites_has_a_register_row():
    """A paper in `docs/literature.md` and in no row is a source nothing can trace.

    The register was complete for the model's own machinery and empty of the product
    literature, which is the worse half to lose: the full-utilisation base case is a modelling
    decision taken on one of those papers, and `docs/limitations.md` states the direction the
    assumption errs in on the strength of two more.
    """
    registered = set()
    for row in _register():
        registered |= set(re.findall(r"10\.\d{4,9}/\S+?(?=[\s;,)\]]|$)",
                                     f"{row['identifier']} {row['locator']}"))
    literature = (paths.ROOT / "docs" / "literature.md").read_text()
    cited = {doi.rstrip(".,)") for doi in re.findall(r"10\.\d{4,9}/[^\s)\]]+", literature)}
    missing = sorted(doi for doi in cited
                     if not any(doi in known or known in doi for known in registered))
    assert not missing, f"cited in literature.md and in no register row: {missing}"


def test_the_filing_identifiers_in_the_docs_match_the_register():
    """One accession number, one spelling of it, wherever it appears.

    The prospectus was carried as ...-000193 in two documents against the ...-000195 that the
    stored extract read off the filing and the register's own URL resolves to. A digit in an
    accession number is not a typo a reader can catch, because every candidate looks equally
    plausible; it has to be checked against the thing it identifies.
    """
    pattern = re.compile(r"\d{10}-\d{2}-\d{6}")
    registered = set()
    for row in _register():
        registered |= set(pattern.findall(f"{row['identifier']} {row['locator']}"))
    assert registered, "the register carries no accession numbers to check against"

    for name in ("docs/literature.md", "docs/data_sources.md", "docs/methodology.md",
                 "docs/validation.md", "docs/limitations.md", "README.md",
                 "references/README.md", "references/references.md"):
        path = paths.ROOT / name
        if not path.exists():
            continue
        for accession in pattern.findall(path.read_text()):
            assert accession in registered, f"{name} cites {accession}, which no row carries"


def test_every_stored_extract_is_the_filing_its_row_names():
    """A path that exists is not the same claim as a path that holds the right document.

    The register says which filing each extract came from and every extract repeats it in its own
    header, read off the filing at the time it was saved. Checking one against the other is the
    only way the repository can tell a correct citation from a file that was replaced, renamed or
    pasted from the wrong accession - a reader comparing an extract to the original would catch
    it, and nothing here would.
    """
    pattern = re.compile(r"\d{10}-\d{2}-\d{6}")
    checked = 0
    for row in _register():
        claimed = set(pattern.findall(f"{row['identifier']} {row['locator']}"))
        if not claimed:
            continue
        for name in [part.strip() for part in row["local_path"].split(";") if part.strip()]:
            path = paths.ROOT / name
            if path.suffix != ".txt":
                continue
            header = path.read_text()[:1500]
            assert claimed & set(pattern.findall(header)), (
                f"{row['ref_id']} names {name}, whose header carries "
                f"{sorted(set(pattern.findall(header)))} against the row's {sorted(claimed)}"
            )
            checked += 1
    assert checked >= 8, f"only {checked} stored extracts checked; the register lists more"


def test_every_book_statistic_appears_in_a_stored_filing_extract():
    """The figures the whole comparison is scaled by, against the filings they were read off.

    `build_dataset.py` checks that a figure appearing in two filings agrees with itself, which
    catches a transcription error made once and not one made twice, and says nothing at all about
    a figure that only one filing carries. This checks the other direction: every number in the
    book statistics has to appear in an extract stored under `references/`.

    It exists because 25 of these 46 figures had no stored source at all. Item 7A was preserved
    from the start and notes 10 to 12 were not, so the separate account value, the fund split and
    the surrender value - the denominators under every result - could not be checked against
    anything. The near-miss is what made it worth a test rather than a note: the one cash
    surrender value the repository did store is 6,330, the general-account figure from note 10,
    against the 231,711 separate-account figure in note 11 that SURRENDER_VALUE_SHARE is built
    on. A reader checking the share against the stored extract would have found it wrong by a
    factor of thirty-seven.
    """
    import pandas as pd

    pool = "".join(path.read_text() for path in
                   sorted((paths.REFERENCES / "sec_filings").glob("*.txt")))
    assert pool, "no stored filing extracts to check against"
    book = pd.read_csv(paths.DATA_RAW / "jackson_book_statistics.csv", comment="#")
    numeric = [c for c in book.columns
               if book[c].dtype.kind in "if" and c != "source_filing_fy"]

    missing = []
    for _, row in book.iterrows():
        for column in numeric:
            value = row[column]
            # Percentages and ages below one are not printed as whole numbers in a filing.
            if pd.isna(value) or abs(float(value)) < 1:
                continue
            whole = abs(int(round(float(value))))
            if not any(form in pool for form in (str(whole), f"{whole:,}")):
                missing.append(f"{row['as_of']} {column}={value:g}")
    assert not missing, f"no stored extract carries: {missing}"


def test_every_withdrawal_rate_matches_the_cell_the_rate_sheet_puts_it_in():
    """Each guaranteed withdrawal percentage against the rate sheet's own grid, cell by cell.

    Checking that a number appears somewhere in the extract is weak here, because the rate sheet
    prints 4.00% in nine different cells. What can go wrong is the pairing: a rate landing under
    the wrong benefit option or the wrong age band reads as perfectly plausible and changes every
    cash flow the contract pays. So this parses the three grids out of the stored supplement and
    compares them against `jackson_rider_terms.csv` by benefit, life basis, option and band.

    The extract keeps its tables as the supplement lays them out, which is what makes the parse
    possible and is a reason to leave that formatting alone.
    """
    import pandas as pd

    sheet = (paths.REFERENCES / "issuer_documents"
             / "2026_JNL_PerspectiveII_RateSheet_497.txt").read_text()
    bands = ["35-59", "60-64", "65-69", "70-74", "75-80", "81+"]

    def grid(heading: str, columns: int) -> dict:
        block = sheet.split(heading, 1)[1]
        rows = {}
        for band in bands:
            line = re.search(rf"^\s*{re.escape(band)}\s+(.+)$", block, re.M)
            assert line, f"no {band} row under {heading!r}"
            found = re.findall(r"(\d+\.\d+)%", line.group(1))
            assert len(found) >= columns, f"{band} under {heading!r} has {len(found)} rates"
            rows[band] = [float(p) / 100 for p in found[:columns]]
        return rows

    expected = {}
    for heading, basis in (("Flex GMWB (Single), benefit options elected at issue", "single"),
                           ("Flex GMWB (Joint), benefit options elected at issue", "joint")):
        for band, rates in grid(heading, 3).items():
            for option, rate in zip(("value", "core", "plus"), rates):
                expected[("flex_gmwb", basis, option, band)] = rate
    # The net grid prints single and joint side by side, two options each, in that order.
    for band, rates in grid("Flex Net GMWB (Single) and (Joint), elected at issue", 4).items():
        for option, rate in zip(("value", "core"), rates[:2]):
            expected[("flex_net_gmwb", "single", option, band)] = rate
        for option, rate in zip(("value", "core"), rates[2:]):
            expected[("flex_net_gmwb", "joint", option, band)] = rate

    terms = pd.read_csv(paths.RIDER_TERMS, comment="#")
    checked = 0
    for _, row in terms.iterrows():
        band = "81+" if row.age_low == 81 else f"{row.age_low}-{row.age_high}"
        key = (row.benefit, row.life_basis, row.option, band)
        assert key in expected, f"{key} is in the committed table and in no cell of the rate sheet"
        assert abs(expected[key] - float(row.gawa_pct)) < 1e-12, (
            f"{key}: the table says {row.gawa_pct:.4f}, the rate sheet says {expected[key]:.4f}")
        checked += 1
    assert checked == 48, f"checked {checked} cells, expected the eight modelled combinations"


def test_the_register_and_the_annotated_bibliography_name_the_same_sources():
    """Both directions, because the two drift apart in both.

    `source_register.csv` is the index and `references.md` is where each source's role is
    argued; a row with no entry is a source nobody explained, and an entry citing an id no row
    carries is a pointer into nothing. Six rows were added with no entries, which is how this
    check came to exist.
    """
    rows = _register()
    bibliography = (paths.REFERENCES / "references.md").read_text()
    unexplained = [row["ref_id"] for row in rows if row["ref_id"] not in bibliography]
    assert not unexplained, f"register rows with no entry in references.md: {unexplained}"

    identifiers = {row["ref_id"] for row in rows}
    cited = set(re.findall(r"REF-\d{3}", bibliography))
    assert not cited - identifiers, f"references.md cites ids no row carries: {cited - identifiers}"


def test_every_public_function_in_the_library_is_called_from_somewhere():
    """The guard for the one defect three audit passes in a row turned up.

    Each of those passes found a module member with no caller, and in three of the four cases a
    docstring cited it as the reason for a convention: the par-curve builder the rate shock was
    said to go through, the effectiveness report that nothing reported, the floor function named
    as measuring a cost. A reader tracing the mechanism from the prose arrives at code the
    project does not run, which is worse than no explanation.

    So this is the sweep, standing. Every public function and method the package defines has to
    be mentioned somewhere other than the line that defines it - a call, a test, a notebook. A
    use in tests counts: a function exercised only by a test is documented behaviour, while one
    exercised by nothing is a claim.

    Constants are deliberately out of scope. A number with its reasoning beside the code that
    uses it is how this project carries its assumptions, and most of them are used in one module
    by design.
    """
    import ast

    library = sorted(paths.ROOT.glob("vahedge/**/*.py"))
    searchable = (library + sorted(paths.ROOT.glob("scripts/*.py"))
                  + sorted(paths.ROOT.glob("tests/*.py"))
                  + sorted(paths.ROOT.glob("notebooks/*.ipynb")))
    corpus = "\n".join(path.read_text(errors="ignore") for path in searchable)

    def public(node) -> bool:
        return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
            not node.name.startswith("_")

    orphans = []
    for path in library:
        tree = ast.parse(path.read_text())
        found = [(node.name, node.lineno) for node in tree.body if public(node)]
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                found += [(sub.name, sub.lineno) for sub in node.body if public(sub)]
        for name, line in found:
            # Every mention of the bare name, less the one in its own def line.
            uses = len(re.findall(rf"\b{re.escape(name)}\b", corpus)) - 1
            if uses <= 0:
                orphans.append(f"{path.relative_to(paths.ROOT)}:{line} {name}")
    assert not orphans, (
        "public functions with no use anywhere, which is how a docstring comes to cite code "
        f"the project does not run: {orphans}"
    )


def test_the_notebook_runs_against_the_committed_tables():
    """The reviewer's first contact, which fifteen audit passes never executed.

    `notebooks/results.ipynb` reads the committed tables and is the document a reader opens
    before any of the prose. Nothing ran it, and a column rename three passes ago left it
    broken: the break-even table's second axis became ``deferral_end_age`` when the age it held
    turned out to be one year short of the first withdrawal, and the notebook kept asking for
    ``income_start_age``.

    Executing it here costs under a second, because every cell reads a committed CSV rather than
    computing anything. The two display helpers a Jupyter kernel supplies are stubbed, which is
    also the check that the notebook does no work in them: if a cell's arithmetic lived inside a
    ``display`` call it would be skipped and the test would pass over it.
    """
    import contextlib
    import io
    import json

    source = (paths.ROOT / "notebooks" / "results.ipynb").read_text()
    cells = json.loads(source)["cells"]
    code = "\n".join("".join(cell["source"]) for cell in cells
                     if cell["cell_type"] == "code")
    code = code.replace("from IPython.display import Image, display", "")

    namespace = {"display": lambda *a, **k: None, "Image": lambda *a, **k: None,
                 "__name__": "results_notebook"}
    cwd = os.getcwd()
    try:
        os.chdir(paths.ROOT)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code, "results.ipynb", "exec"), namespace)
    finally:
        os.chdir(cwd)


def test_the_notebook_carries_no_stored_output():
    """A committed notebook with outputs in it publishes numbers nothing regenerated.

    The tables and figures under `reports/` are the committed results and they are rebuilt by
    the pipeline; a cell output pasted into the notebook is a second copy that no rerun touches
    and no diff makes obvious. Execution counts go with them, because a notebook carrying them
    records the order someone happened to run it in rather than the order it reads in.
    """
    import json

    cells = json.loads((paths.ROOT / "notebooks" / "results.ipynb").read_text())["cells"]
    carrying = [i for i, cell in enumerate(cells)
                if cell.get("outputs") or cell.get("execution_count")]
    assert not carrying, f"notebook cells with stored output or an execution count: {carrying}"


def test_the_source_holds_to_its_own_whitespace_conventions():
    """The sweep for a class of drift that no behavioural test can see.

    Pass seventeen read `vahedge/market/heston_cos.py` end to end for the first time - the one
    module sixteen passes had only ever sampled - and the two things wrong with it were a
    missing blank line between `_call_coefficients` and `_chi_psi` and a run of three where the
    file uses two. Neither moves a number, which is exactly why nothing had caught them. Widened
    to the whole tree it turned up twenty-nine more: fifteen lines past a hundred characters
    against a body of code whose ninety-ninth percentile is ninety-seven, two files ending on a
    blank line where forty do not, six continuation lines each sitting one column short of the
    bracket above them, and a loop variable named `l`.

    The conventions asserted here are the ones the repository already keeps, read off its own
    distribution rather than imported from a style guide. Continuation-line alignment and the
    single-character name are left out: one needs a tokeniser to judge and the other is a single
    past slip, and a test that re-implements half a linter is a liability of its own.
    """
    import ast

    sources = (sorted(paths.ROOT.glob("vahedge/**/*.py"))
               + sorted(paths.ROOT.glob("scripts/*.py"))
               + sorted(paths.ROOT.glob("tests/*.py")))
    assert len(sources) > 40, f"only found {len(sources)} source files, so the sweep is not running"

    problems = []
    for path in sources:
        where = path.relative_to(paths.ROOT)
        raw = path.read_text()
        lines = raw.split("\n")

        for number, line in enumerate(lines, start=1):
            if len(line) > 100:
                problems.append(f"{where}:{number} is {len(line)} characters")

        if raw.endswith("\n\n"):
            problems.append(f"{where} ends on a blank line")
        elif raw and not raw.endswith("\n"):
            problems.append(f"{where} has no final newline")

        run = 0
        for number, line in enumerate(lines, start=1):
            run = run + 1 if line.strip() == "" else 0
            if run == 3:
                problems.append(f"{where}:{number - 2} starts a run of three blank lines")

        # Two blank lines above every top-level definition, measured above any decorator and
        # above a comment block that sits flush against it, since such a block is part of the
        # definition rather than something between it and the code before.
        tree = ast.parse(raw)
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            top = min([node.lineno] + [d.lineno for d in node.decorator_list])
            index = top - 2
            while index >= 0 and lines[index].startswith("#"):
                index -= 1
            blanks = 0
            while index >= 0 and lines[index].strip() == "":
                blanks += 1
                index -= 1
            if index >= 0 and blanks != 2:
                problems.append(f"{where}:{top} has {blanks} blank lines above `{node.name}`")

    assert not problems, "\n".join([f"{len(problems)} whitespace departures:"] + problems)


def test_the_prose_figures_still_round_to_what_the_tables_say():
    """Docstring figures against the committed tables they describe.

    Pass seventeen measured every number stated in one module's prose and found five wrong. Pass
    eighteen did it across the package - 329 figures - and the great majority reproduce exactly:
    `scenarios.py`'s seven rate-path figures, `curves.py`'s three fit residuals, `strategies.py`'s
    six put-design figures, every filed figure taken off a 10-K or the rate sheet. Four did not,
    and two of them mattered.

    `cohorts.py` carried a blended withdrawal rate of 5.73% where both `docs/validation.md` and
    `docs/limitations.md` already said 5.82% - a correction that propagated into the documents
    and stopped at the module docstring. Worse, it said the vintage book's multiple over the
    disclosure sits above the reference contract's, so "representativeness moves the multiple
    up". Measured, the direction depends on the shock: the book takes the equity multiple down
    and the rate multiple up, which is the same split `docs/validation.md` reaches from the other
    side and the only part of the comparison worth knowing.

    `reporting.py` stated the own-credit adjustment's path across the decade as -7.1 to +1.5 of
    premium beside the risk margin's +1.7 to -0.5. The risk-margin half was exact and the
    own-credit half was not, which is the shape of a figure that went stale while its neighbour
    was updated.

    Each assertion below pins the *rounding* the prose uses, so it fails when the underlying
    value moves far enough to change the printed digit and stays quiet otherwise.
    """
    import pandas as pd

    book = pd.read_csv(paths.TABLES / "portfolio_vs_disclosed.csv")
    contract = pd.read_csv(paths.TABLES / "in_force_vs_disclosed.csv")
    latest = book[book["as_of"] == "2025-12-31"].iloc[0]

    assert round(float(latest["weighted_attained_age"]), 2) == 70.85
    assert round(float(latest["weighted_gawa_pct"]), 2) == 5.82
    assert [float(v) for v in (book["deferring_share"].iloc[0],
                               book["deferring_share"].iloc[-1])] == [0.75, 0.40]

    # Only the shocks with a disclosed figure at this date, which is equity both ways and rates
    # at 100bp; the 50bp pair is disclosed in other years and not in this one.
    equity = ("equity_up_10pct", "equity_down_10pct")
    rates = ("rates_up_100bp", "rates_down_100bp")

    def multiples(row, shocks):
        return [float(row[f"model_{s}"]) / float(row[f"disclosed_{s}"]) for s in shocks]

    reference = contract[(contract["as_of"] == "2025-12-31")
                         & (contract["duration_basis"] == "anniversary on or before")].iloc[0]
    book_equity, book_rates = multiples(latest, equity), multiples(latest, rates)
    ref_equity, ref_rates = multiples(reference, equity), multiples(reference, rates)

    together = book_equity + book_rates
    assert [round(min(together), 1), round(max(together), 1)] == [2.3, 2.9], (
        f"the vintage book's 2025 multiple now spans {min(together):.2f} to {max(together):.2f}, "
        "and cohorts.py says 2.3 to 2.9"
    )
    assert [round(min(ref_equity), 1), round(max(ref_equity), 1)] == [3.1, 3.2]
    assert [round(min(book_equity), 1), round(max(book_equity), 1)] == [2.3, 2.8]
    assert round(min(ref_rates), 1) == round(max(ref_rates), 1) == 1.4
    assert [round(min(book_rates), 1), round(max(book_rates), 1)] == [2.8, 2.9]

    assert max(book_equity) < min(ref_equity), (
        "cohorts.py says representativeness takes the equity multiple down; the book now sits at "
        f"{min(book_equity):.2f}-{max(book_equity):.2f} against the contract's "
        f"{min(ref_equity):.2f}-{max(ref_equity):.2f}"
    )
    assert min(book_rates) > max(ref_rates), (
        "cohorts.py says it takes the rate multiple up; the book now sits at "
        f"{min(book_rates):.2f}-{max(book_rates):.2f} against the contract's "
        f"{min(ref_rates):.2f}-{max(ref_rates):.2f}"
    )

    daily = pd.read_csv(paths.TABLES / "reporting_lens_daily_s2.csv", parse_dates=["date"])
    first, last = daily.iloc[0], daily.iloc[-1]
    for column, opening, closing in (("risk_margin", 1.7, -0.5),
                                     ("own_credit_adjustment", -4.8, 0.9)):
        assert round(float(first[column]), 1) == opening, (
            f"reporting.py says {column} opens at {opening}; it opens at {first[column]:.2f}")
        assert round(float(last[column]), 1) == closing, (
            f"reporting.py says {column} closes at {closing}; it closes at {last[column]:.2f}")
    assert round(float(daily["own_credit_adjustment"].min()), 1) == -11.1

    # The own-credit spread through the project's own covid episode, which is the window
    # hedge_episodes.csv defines rather than a month picked by hand.
    episodes = pd.read_csv(paths.TABLES / "hedge_episodes.csv")
    covid = episodes[episodes["episode"] == "covid"].iloc[0]
    window = daily[(daily["date"] >= covid["start"]) & (daily["date"] <= covid["end"])]
    assert len(window) > 15
    spread = 100.0 * window["own_credit_spread"]
    assert [round(float(spread.min()), 1), round(float(spread.max()), 1)] == [1.2, 2.6]


def test_the_hypothesis_three_decline_is_the_one_the_table_produces():
    """The project's loudest claim about shape, pinned to the table it comes from.

    `docs/validation.md` tabulates the model's equity sensitivity at the four disclosure dates
    against the filing's, and the last column is each row's 2025 figure over its 2022 one. The
    disclosed row has always been right, because it is typed out of the filings and never moves.
    The model row was carried over from a superseded run - 4.56 through 2.64 against the
    committed 4.10 through 2.34 - and the ratio taken from those stale values happened to land
    within 0.0001 of the disclosed ratio. That read as agreement to one part in ten thousand, and
    the document called it the strongest single piece of evidence in the project that the model
    has the right shape. It was quoted that way in the README and in the notebook as well.

    From the committed table the two declines are 0.5720 and 0.5794, which agree to 1.3 per
    cent. That is still a result - nothing in the fit is aimed at the pace of the decline - and
    it is three orders of magnitude weaker than the claim it replaced.

    Pass seventeen's README figure sweep and two of this pass's own attempts all passed this row,
    because they tested each figure against the union of every rounded value in every committed
    table. A decoy test put that pool's hit rate at 100 per cent for a one-, two- or
    three-decimal figure, so those checks could not fail. This one names the table, the column
    and the row.
    """
    import pandas as pd

    book = pd.read_csv(paths.TABLES / "portfolio_vs_disclosed.csv")
    assert len(book) == 4, f"the comparison now has {len(book)} dates, not four"
    model = book["model_equity_down_10pct"]
    disclosed = book["disclosed_equity_down_10pct"]

    assert [round(float(v), 2) for v in model] == [4.10, 3.75, 2.47, 2.34]
    assert [round(float(v), 2) for v in disclosed] == [1.47, 1.18, 0.96, 0.85]

    model_decline = float(model.iloc[-1] / model.iloc[0])
    disclosed_decline = float(disclosed.iloc[-1] / disclosed.iloc[0])
    assert round(model_decline, 4) == 0.5720, f"the model decline is now {model_decline:.4f}"
    assert round(disclosed_decline, 4) == 0.5794, (
        f"the disclosed decline is now {disclosed_decline:.4f}, which would mean the filed "
        "figures were retyped")

    # The claim the prose makes about the two: close, and not closer than this. The upper bound
    # is what fails if a stale row ever makes them agree to a part in ten thousand again.
    gap = abs(model_decline - disclosed_decline) / disclosed_decline
    assert 0.005 < gap < 0.02, (
        f"the two declines now agree to {100 * gap:.2f} per cent, and validation.md, the README "
        "and the notebook all say 1.3")

    # And the levels, which the same paragraph calls about three times the disclosure.
    multiples = (model / disclosed).tolist()
    assert all(2.0 < float(m) < 4.0 for m in multiples), multiples


def test_the_put_grid_choice_does_not_decide_the_ranking():
    """The one selection the hedging backtest really makes, and the bound on it.

    S3's put tenor and strike were picked off the E3 sweep - the same replay the strategy is
    then scored on - so they are chosen in sample. `docs/limitations.md` used to answer the
    whole bias question with "there is no survivorship or selection question to answer here",
    which is true of the issuer, the shocks and the window and not true of these two numbers.

    What makes the in-sample choice harmless is not an argument, it is an ordering: the project
    concludes a ranking of instrument sets, and the worst cell of the put grid still leaves a
    tighter residual than the best cell of S2 at any rebalance frequency. That is what this
    pins, along with the grid's own spread, because those are the figures the corrected
    paragraph quotes. If a future change ever makes the ranking depend on which cell was picked,
    the claim has to be withdrawn rather than reworded.
    """
    import pandas as pd

    grid = pd.read_csv(paths.TABLES / "hedge_put_design.csv")
    assert len(grid) == 9, f"the sweep is now {len(grid)} cells, not the three-by-three"
    assert bool(grid["on_frontier"].all()), "a cell has dropped off the frontier"

    # The spreads the prose quotes, to the precision the quantity has rather than the precision
    # it is printed at. Both ends are nested Monte Carlo carried through the option sizing, and
    # a rerun under a different numpy and scipy moves them by up to two per cent - which the
    # first version of this test pinned to one decimal and duly failed on, against a rerun that
    # had changed nothing. A guard that fires on an environment is not a guard.
    spread = [100 * float(grid["pnl_sd_pct"].min()), 100 * float(grid["pnl_sd_pct"].max())]
    assert approx([0.135, 0.163], rel=0.03) == spread, f"the grid's residual now spans {spread}"
    cost = [100 * float(grid["total_cost_pct"].min()), 100 * float(grid["total_cost_pct"].max())]
    assert approx([5.2, 21.5], rel=0.05) == cost, f"the grid's cost now spans {cost}"

    # The claim itself, which is an ordering and reproduces far better than either level: the
    # margin is fifteen per cent of the put cell and moved by four tenths of a point across
    # environments. Asserting the margin rather than the two numbers is what makes this a test
    # of the sentence in docs/limitations.md instead of a test of the container.
    frontier = pd.read_csv(paths.TABLES / "hedge_frequency_frontier.csv")
    best_s2 = float(frontier[frontier["strategy"] == "S2"]["pnl_sd_pct"].min())
    worst_put_cell = float(grid["pnl_sd_pct"].max())
    margin = (best_s2 - worst_put_cell) / worst_put_cell
    assert margin > 0.08, (
        f"the worst cell of the put grid leaves {100 * worst_put_cell:.4f}% against S2's best "
        f"{100 * best_s2:.4f}%, a margin of {100 * margin:.1f}%. docs/limitations.md says the "
        "ranking does not depend on which cell was chosen, and at this margin it is starting to"
    )


def test_every_path_the_prose_names_is_a_path_that_exists():
    """The prose-name sweep pass fifteen ran for function names, run for file paths.

    Pass fifteen found four docstrings citing a module member the project does not run and made
    that a standing check. Paths went unchecked, and `vahedge/market/simulate.py` had been
    telling readers for seventeen audit passes that its step-count comparison was regenerated by
    a market-validation script under `scripts/`. No such script exists or ever did here, so the
    three figures it was said to produce had no generator - and when re-measured they turned out
    not to reproduce either, which is how the missing script and the wrong numbers were the same
    finding. This test catches its own example, so the path is described rather than spelled.

    Only fully qualified paths are checked, the ones beginning with a directory this repository
    has. Bare basenames are how the prose refers to a neighbouring module - "sizing.py explains
    what each one trades off" - and resolving those would mean guessing a directory, which turns
    a sweep with no false positives into one with forty.

    Generated inputs are exempt because a fresh clone does not have them: `data/processed` is
    gitignored and built by `make calibrate` and `make convexity`. The tables and figures under
    `reports/` are not exempt - they are committed, so a figure named in the prose and absent
    from the repository is a broken reference.
    """
    import ast
    import io
    import tokenize

    named = re.compile(r"(?<![\w/.])((?:vahedge|scripts|tests|docs|data|reports|notebooks"
                       r"|references)/[\w./*-]*[\w*])")
    built = ("data/processed", "data/interim")

    def prose(path):
        """Every comment and docstring in a module, or the whole of a markdown document."""
        if path.suffix == ".md":
            return [(1, path.read_text())]
        found, text = [], path.read_text()
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.COMMENT:
                found.append((token.start[0], token.string))
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc:
                    found.append((getattr(node, "lineno", 1), doc))
        return found

    documents = (sorted(paths.ROOT.glob("vahedge/**/*.py"))
                 + sorted(paths.ROOT.glob("scripts/*.py"))
                 + sorted(paths.ROOT.glob("tests/*.py"))
                 + sorted(paths.ROOT.glob("docs/*.md"))
                 + [paths.ROOT / "README.md"])
    assert len(documents) > 50, f"only {len(documents)} documents, so the sweep is not running"

    checked, broken = set(), []
    for document in documents:
        for line, text in prose(document):
            for match in named.finditer(text):
                candidate = match.group(1).rstrip(".")
                # A glob in the prose is a description of a set, not a path to resolve.
                if "*" in candidate or candidate.startswith(built):
                    continue
                checked.add(candidate)
                if not (paths.ROOT / candidate).exists():
                    broken.append(f"{document.relative_to(paths.ROOT)}:{line} names {candidate}")

    assert len(checked) > 30, f"only {len(checked)} qualified paths found; the regex has drifted"
    assert not broken, "prose naming paths that do not exist:\n" + "\n".join(broken)


def test_nothing_is_introduced_and_then_left_unused():
    """Locals and imports the code creates and never reads, which closes a hole in the sweep
    above.

    `test_every_public_function_in_the_library_is_called_from_somewhere` counts mentions of a
    name across the repository, and a mention is cheap. `curves.flat_curve` satisfied it for
    nineteen passes on the strength of one line in a test - `curve = flat_curve(0.04)`, followed
    a few lines later by `del curve`, with the value never used and the delete there to stop
    anything noticing. A dead call is a mention. So the orphan sweep needs this one beside it:
    a call whose result is thrown away shows up here as a dead local, and the two together
    cannot both be satisfied by the same trick.

    What the same sweep found on its first run, besides that one:

    - `_choose_smoothing` accumulated a standard error per grid point and a coefficient vector
      per grid point, returned neither, and had done so since the one-standard-error rule it
      was written for was measured and dropped. Its summary line still said it selected by that
      rule, which is the opposite of what the body does and explains.
    - four tuple unpacks and a loop key left bound to a name the function does not use, where
      the repository's own convention elsewhere is to unpack those into `_`.
    - an alias, `state = calibration`, surviving a rename.
    - two private imports in a test carrying `# noqa: F401  (used below)`, used nowhere below.
      The suppression was load-bearing: it turned off the one check that would have said so.

    A list that is only ever appended to counts as dead, because `x.append(v)` reads `x` and
    would otherwise satisfy any check that looks for a read. Arguments are out of scope - an
    unused parameter is often an interface obligation - and so is anything named with a leading
    underscore, which is how this repository already says "deliberately discarded".
    """
    import ast
    import collections

    nested = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
    sources = (sorted(paths.ROOT.glob("vahedge/**/*.py"))
               + sorted(paths.ROOT.glob("scripts/*.py"))
               + sorted(paths.ROOT.glob("tests/*.py")))
    assert len(sources) > 40, f"only {len(sources)} source files, so the sweep is not running"

    def own_stores(function):
        """Names stored in this function's own scope, skipping nested scopes.

        A class body inside a function assigns attributes, not locals, and a nested function
        assigns its own; counting either as a local of the enclosing function is how a first
        version of this reported four tests' fake projections as dead.
        """
        found = collections.defaultdict(list)

        def walk(node):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, nested):
                    continue
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                    found[child.id].append(child.lineno)
                walk(child)

        for statement in function.body:
            if not isinstance(statement, nested):
                walk(statement)
        return found

    def reads(function):
        """Names read as a value, and names only ever appended to, anywhere in the body.

        Loads are counted inside nested scopes as well as outside, because a closure reading an
        enclosing local is a genuine use of it.
        """
        loads, appends = collections.Counter(), collections.Counter()

        class Visitor(ast.NodeVisitor):
            def visit_Call(self, node):
                target = node.func
                if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                        and target.attr in {"append", "extend"}):
                    appends[target.value.id] += 1
                    for argument in node.args:
                        self.visit(argument)
                    for keyword in node.keywords:
                        self.visit(keyword.value)
                    return
                self.generic_visit(node)

            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Load):
                    loads[node.id] += 1

        visitor = Visitor()
        for statement in function.body:
            visitor.visit(statement)
        return loads, appends

    dead = []
    for source in sources:
        where = source.relative_to(paths.ROOT)
        tree = ast.parse(source.read_text())

        for function in [node for node in ast.walk(tree)
                         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            loads, appends = reads(function)
            parameters = {argument.arg for argument in
                          function.args.args + function.args.kwonlyargs
                          + function.args.posonlyargs}
            for name, lines in own_stores(function).items():
                if name in parameters or name.startswith("_") or loads[name]:
                    continue
                shape = "appended to and never read" if appends[name] else "never read"
                dead.append(f"{where}:{lines[0]} in {function.name}(): {name} is {shape}")

        imported = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported[(alias.asname or alias.name).split(".")[0]] = node.lineno
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name != "*":
                        imported[alias.asname or alias.name] = node.lineno
        mentioned = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        mentioned |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        for name, line in sorted(imported.items(), key=lambda row: row[1]):
            if name not in mentioned and name != "annotations":
                dead.append(f"{where}:{line} imports {name} and never uses it")

    assert not dead, "\n".join([f"{len(dead)} names introduced and left unused:"] + dead)


def test_a_drawdown_is_computed_in_one_place_on_one_convention():
    """Running extrema outside `simulator.worst_drawdown`, which is how the duplicate got in.

    The macro frontier had its own drawdown, taking a running maximum straight over the capital
    series. That series starts on the first day, already after the book has been struck, so the
    cost of striking it fell outside the drawdown while the worst-day column in the same row
    included it. It agreed with the corrected convention on every row of its own grid by luck -
    the hedge is ahead of its opening level long before the trough in covid, so both find the
    same peak - and would not have on a window where the hedge never gets ahead.

    A second implementation of a convention is the thing to prevent rather than the arithmetic,
    so this counts running extrema in the library and the scripts and expects exactly the one.
    A legitimate new use means routing it through `worst_drawdown` or widening the allowance
    here with the reason written down, not deleting the check.
    """
    import ast

    allowed = {("vahedge/hedge/simulator.py", "worst_drawdown")}
    sources = (sorted(paths.ROOT.glob("vahedge/**/*.py"))
               + sorted(paths.ROOT.glob("scripts/*.py")))
    assert len(sources) > 30, f"only {len(sources)} source files, so the sweep is not running"

    def running_extremum(node) -> bool:
        if not isinstance(node, ast.Attribute):
            return False
        if node.attr in ("cummax", "cummin"):
            return True
        return (node.attr == "accumulate" and isinstance(node.value, ast.Attribute)
                and node.value.attr in ("maximum", "minimum"))

    found = []
    for source in sources:
        where = source.relative_to(paths.ROOT).as_posix()
        tree = ast.parse(source.read_text())
        for function in [node for node in ast.walk(tree)
                         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            for node in ast.walk(function):
                if running_extremum(node):
                    found.append((where, function.name, node.lineno))
        for node in ast.walk(tree):
            if running_extremum(node) and not any(
                node.lineno == line for _, _, line in found
            ):
                found.append((where, "<module>", node.lineno))

    stray = [f"{where}:{line} in {name}()" for where, name, line in found
             if (where, name) not in allowed]
    assert not stray, (
        "running extrema outside simulator.worst_drawdown:\n" + "\n".join(stray)
    )
    assert found, "no running extremum anywhere, so worst_drawdown is not the one being checked"


def test_the_methodology_document_figures_match_the_tables_they_came_from():
    """The figures `docs/methodology.md` states that a committed table can settle.

    That document carries about forty-five figures and no tables of its own, so nothing in the
    suite reached them. Tracing them one at a time found one wrong: the par-shift claim said a
    hundred basis points arrived as 98.2 at five years and 104.7 at thirty, where the bootstrap
    gives 98.3 and 103.9 and `curves.py` had said so since pass eighteen. The correction had
    been made in the module and not carried into the document, and five passes went by.

    The two figures that caught it are not in any table, so they are not here - a rerun of the
    bootstrap is what checks those, and the module and the document now agree. What is here is
    every methodology figure a committed table does settle, pinned at the rounding the prose
    prints, so the next table that moves takes the document with it.
    """
    import pandas as pd

    def table(name):
        return pd.read_csv(paths.TABLES / f"{name}.csv")

    # "a 5.95% lifetime draw exhausts the account on 78% of paths by year twenty"
    flows = table("cash_flow_profile")
    assert round(float(flows.loc[flows["policy_year"] == 20, "prob_exhausted"].iloc[0]), 2) == 0.78

    # "866 per point against 27 for the current variance", and the $136 standard error, which
    # the Greeks table and the convergence table state separately and have to agree on.
    greeks = table("greeks")
    economic = greeks[greeks["basis"].str.startswith("economic")].iloc[0]
    assert round(float(economic["vega_long_run"])) == 866
    assert round(float(economic["vega_current"])) == 27
    assert round(float(economic["level_std_error"])) == 136
    convergence = table("monte_carlo_convergence").set_index("paths")
    assert round(float(convergence.loc[20000, "std_error"])) == 136
    # "it halves as paths quadruple" - both quadruplings the table carries
    for small, large in ((2500, 10000), (5000, 20000)):
        ratio = float(convergence.loc[large, "std_error"] / convergence.loc[small, "std_error"])
        assert abs(ratio - 0.5) < 0.02, f"{small}->{large} paths moved the error by {ratio:.3f}"

    # "the robustness table prices the issuer's level at -2.3% of premium", and the base case's
    # alpha, which the same document quotes as 0.79 at 31 December 2025
    robustness = table("valuation_robustness")
    issuer = robustness[robustness["variation"].str.contains("disclosed 19.12")].iloc[0]
    assert round(float(issuer["mrb_pct_of_premium"]), 1) == -2.3
    base = robustness[robustness["variation"] == "base case"].iloc[0]
    assert round(float(base["implied_attribution"]), 2) == 0.79

    # "Four of the five vintages calibrate to the cap, and only the 2024 vintage has margin"
    vintages = table("portfolio_vintage_detail")
    latest = vintages[vintages["as_of"] == "2025-12-31"]
    assert len(latest) == 5
    assert int((latest["attribution"] >= 1.0).sum()) == 4
    assert round(float(latest["attribution"].min()), 3) == 0.733

    # "At 1.20 of the benefit base in policy year five the exposure runs from -27,492 with most
    # of the year to go to +2,679 on its eve"
    proximity = table("step_up_proximity")
    eve = proximity[(proximity["policy_year"] == 5) & (proximity["account_over_base"] == 1.2)]
    eve = eve.sort_values("years_to_anniversary", ascending=False)
    assert [round(float(v)) for v in eve["equity_exposure"]] == [-27492, -12407, 2679]

    # "binds on 77% of in-force paths in the first policy year, dips to 55% around year twelve
    # ... and climbs to essentially every path by year thirty-five"
    reach = table("statutory_floor_reach")
    first = float(reach.loc[reach["policy_year"] == 1, "share_floor_binds"].iloc[0])
    assert round(100 * first) == 77
    trough = reach.loc[reach["share_floor_binds"].idxmin()]
    assert int(trough["policy_year"]) == 12
    assert round(100 * float(trough["share_floor_binds"])) == 55
    assert float(reach.loc[reach["policy_year"] == 35, "share_floor_binds"].iloc[0]) > 0.99

    # "moves its average reach from 67.9% to 68.8% of in-force paths and what it adds to the
    # reserve from 9.80% to 10.42%", across the four disclosed year ends and not the model's
    # rounded 98.0%, which is the fifth row of that table
    sensitivity = table("statutory_floor_share_sensitivity")
    disclosed = sensitivity[sensitivity["surrender_share"].isin([0.9729, 0.9773, 0.9795, 0.9801])]
    assert len(disclosed) == 4
    assert [round(100 * float(v), 1) for v in (disclosed["mean_share_floor_binds"].min(),
                                               disclosed["mean_share_floor_binds"].max())
            ] == [67.9, 68.8]
    addition = disclosed["mean_floor_addition_pct_of_account"]
    assert [round(100 * float(v), 2) for v in (addition.min(), addition.max())] == [9.80, 10.42]

    # "calibrated to 1,468 cleaned quotes across seventeen expiries ... worst per-maturity root
    # mean square error of 0.85 volatility points", and the chain reaching 3.23 years
    surface = table("surface_fit")
    assert len(surface) == 17
    assert int(surface["n_quotes"].sum()) == 1468
    assert round(float(surface["maturity"].max()), 2) == 3.23
    assert round(100 * float(surface["rmse_vol"].max()), 2) == 0.85
