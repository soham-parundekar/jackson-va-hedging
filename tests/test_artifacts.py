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


def test_the_withdrawal_rate_grid_passes_through_the_policy_the_headline_values():
    """The diagnostic row called the contractual rate has to be the contract being valued.

    Two figures in `docs/validation.md` are quoted as fractions of the contract's sensitivity at
    its contractual withdrawal rate, so the row they divide by must be that contract and not a
    neighbouring band. For a while it was not: the grid stopped at 5.75%, the band for a first
    withdrawal between 70 and 74, while the policy defers five years into the 75 to 80 band and
    draws 5.95%. Nothing caught it, because 5.75% reads as a perfectly plausible rate for a
    70-year-old and the two sensitivities it produced were the right sort of size.

    What settles it is that three figures in `valuation_headline.csv` come out of the same
    valuation as the diagnostic's contractual row - the attribution percentage, the gross
    guarantee and the probability of exhaustion - and they are written by a different script.
    Exactly one row can carry all three.
    """
    import pandas as pd

    from scripts.run_shock_validation import GAWA_GRID
    from scripts.run_valuation import DEFERRAL_YEARS, ISSUE_AGE

    terms = pd.read_csv(paths.RIDER_TERMS, comment="#")
    core = terms.query("benefit == 'flex_gmwb' and life_basis == 'single' and option == 'core'")
    # The band is set by the age at the first withdrawal, which a deferral running to the sixth
    # anniversary puts a year past the deferral itself.
    first_withdrawal_age = ISSUE_AGE + DEFERRAL_YEARS + 1
    band = core.query("age_low <= @first_withdrawal_age <= age_high")
    assert len(band) == 1, f"{len(band)} Core bands cover age {first_withdrawal_age}"
    contractual = 100 * float(band["gawa_pct"].iloc[0])

    assert any(abs(100 * rate - contractual) < 1e-9 for rate in GAWA_GRID), (
        f"the diagnostic grid has no point at the contractual {contractual:.2f}%, so the figures "
        "quoted against it come from some other withdrawal rate")

    headline = {row["quantity"]: float(row["value"]) for row in
                csv.DictReader((paths.TABLES / "valuation_headline.csv").open())}
    wanted = {"attribution": headline["fee attribution percentage"],
              "pv_claims_pct_av": headline["gross guarantee value, % of premium"],
              "prob_exhausted_20y": headline["probability of exhaustion by year 20"]}

    grid = list(csv.DictReader((paths.TABLES / "withdrawal_rate_diagnostic.csv").open()))
    matching = [row for row in grid
                if all(abs(float(row[column]) - value) < 5e-4
                       for column, value in wanted.items())]
    assert len(matching) == 1, (
        f"{len(matching)} of {len(grid)} rows in the withdrawal rate diagnostic reproduce the "
        "headline valuation, and the contractual row is the one that has to")
    assert float(matching[0]["gawa_pct"]) == approx(contractual, abs=5e-4), (
        f"the row reproducing the headline draws {float(matching[0]['gawa_pct']):.2f}% while the "
        f"rate sheet gives this contract {contractual:.2f}%")


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


def test_the_validation_document_error_and_parity_figures_match_their_tables():
    """`docs/validation.md`'s Monte Carlo and parity paragraphs, which carried three defects.

    The vega error was quoted as 40 on 893. The error belongs to the two vegas added together -
    the volatility bump moves the current variance and the long-run level at once, and the split
    between them is a second valuation with no error of its own - and that sum is 894, so the
    893 was a truncation where every figure beside it rounds. Quoting the same error against the
    27 of current variance alone, which `Greeks.to_frame` did before it was deleted, reads as a
    Greek worth less than its own noise.

    The parity paragraph said the financing spread sits within a few basis points of 45
    everywhere except the two shortest expiries. The second-shortest sits at 43.1, two points
    off 45, and the expiries that miss are the shortest at 93.9, the three-and-a-half-month at
    65.1 and the five-month at 52.7, with the two-year at 38.5 the only miss further out. The
    claim named the wrong exceptions.

    The index level the residual is quoted against, 7,710, is in the calibration diagnostics
    rather than any committed table, so it is not here; `market_calibration.json` is gitignored
    and a test cannot read it.
    """
    import numpy as np
    import pandas as pd

    def table(name):
        return pd.read_csv(paths.TABLES / f"{name}.csv")

    # "the equity exposure's is 164 on 14,501, rho's is 19 on 5,887 per 100bp, vega's is 40 on
    # 894" - the signs are the model's and the prose quotes magnitudes
    greeks = table("greeks")
    economic = greeks[greeks["basis"].str.startswith("economic")].iloc[0]
    assert round(abs(float(economic["equity_exposure"]))) == 14501
    assert round(float(economic["equity_exposure_std_error"])) == 164
    assert round(abs(100 * float(economic["rho_per_bp"]))) == 5887
    assert round(100 * float(economic["rho_std_error"])) == 19
    both = float(economic["vega_current"]) + float(economic["vega_long_run"])
    assert round(both) == 894
    assert round(float(economic["vega_std_error"])) == 40

    # "2.01, 1.96 and 1.99 across the three quadruplings"
    convergence = table("monte_carlo_convergence").set_index("paths")
    quadruplings = [(2500, 10000), (5000, 20000), (10000, 40000)]
    ratios = [round(float(convergence.loc[a, "std_error"] / convergence.loc[b, "std_error"]), 2)
              for a, b in quadruplings]
    assert ratios == [2.01, 1.96, 1.99]

    # "across eighteen expiries from eighteen days to 3.2 years"
    parity = table("parity_discount_check").sort_values("maturity")
    assert len(parity) == 18
    assert round(365 * float(parity["maturity"].min())) == 18
    assert round(float(parity["maturity"].max()), 1) == 3.2
    assert round(float(parity["parity_max_resid"].max()), 2) == 0.96

    # "38 to 94 basis points over the matched-maturity Treasury"
    spread = parity["spread_bp"]
    assert [round(float(spread.min())), round(float(spread.max()))] == [38, 94]

    # "From the six-month expiry out it sits within three basis points of 45 everywhere but the
    # two-year, which comes in at 38"
    out = parity[parity["maturity"] >= 0.45]
    two_year = out[(out["maturity"] > 2.0) & (out["maturity"] < 2.5)]
    assert len(two_year) == 1
    assert round(float(two_year["spread_bp"].iloc[0])) == 38
    rest = out[~out.index.isin(two_year.index)]
    assert (rest["spread_bp"] - 45.0).abs().max() < 3.0

    # "the three that miss are all inside six months: 94 basis points at eighteen days, 65 at
    # three and a half months and 53 at five"
    front = parity[parity["maturity"] < 0.45].sort_values("maturity")
    missing = front[(front["spread_bp"] - 45.0).abs() >= 3.0]
    assert [round(float(v)) for v in missing["spread_bp"]] == [94, 65, 53]
    assert [round(12 * float(m), 1) for m in missing["maturity"]] == [0.6, 3.6, 4.7]

    # The six rows the withdrawal rate section prints out of the nine. Two cells sit on a
    # rounding boundary - the 1.00% attribution at 0.1535 and the 5.75% at 0.7304 - so this
    # compares each printed figure against half a unit of its own last place rather than against
    # a re-rounding, which would turn a tie into a failure with no defect behind it.
    diagnostic = table("withdrawal_rate_diagnostic").set_index("gawa_pct")
    printed = {1.00: (0.154, 0.133, 1.14, -1.52), 2.00: (0.151, 0.258, 0.99, -1.35),
               3.00: (0.218, 0.399, 1.09, -1.86), 5.75: (0.730, 0.756, 1.92, -4.95),
               5.95: (0.786, 0.777, 1.99, -5.22), 6.50: (0.949, 0.824, 2.18, -5.97)}
    columns = ("attribution", "prob_exhausted_20y", "equity_down_10pct_pct_av",
               "rates_up_100bp_pct_av")
    for rate, cells in printed.items():
        row = diagnostic.loc[rate]
        for column, places, cell in zip(columns, (3, 3, 2, 2), cells):
            assert abs(float(row[column]) - cell) <= 0.5 * 10 ** -places + 1e-12, (
                f"the document prints {cell} for the {rate:.2f}% row's {column}, and the "
                f"diagnostic says {float(row[column]):.4f}")

    # "At a 1.7% draw the rate sensitivity is 0.26 of its contractual-rate value and the equity
    # sensitivity 0.50 of its own" - the denominator is the 5.95% row, the contract itself
    draw = 1.69
    for column, factor in (("rates_up_100bp_pct_av", 0.26), ("equity_down_10pct_pct_av", 0.50)):
        at_draw = float(np.interp(draw, diagnostic.index.to_numpy(),
                                  diagnostic[column].to_numpy()))
        measured = at_draw / float(diagnostic.loc[5.95, column])
        assert round(measured, 2) == factor, (
            f"the document says {factor} of the contractual-rate {column} survives a 1.7% draw, "
            f"and the diagnostic gives {measured:.4f}")

    # The proxy accuracy table, including its last column, which is the one the paragraph under
    # it says is worth reading and the only one not lifted straight out of a stored field: the
    # delta RMSE is a share of a premium of 100, so it is dollars against the mean absolute
    # nested delta at the same nodes. The year 25 delta RMSE reached this table as 0.025 for a
    # stored 0.02447, which is what the tolerance below is sized to catch.
    proxy = table("proxy_accuracy").set_index("year")
    quoted = {1: (0.9952, 0.0059, 0.126, 35.0, 36), 2: (0.9978, 0.0053, 0.081, 38.6, 21),
              5: (0.9990, 0.0068, 0.044, 42.7, 10), 9: (0.9997, 0.0044, 0.021, 31.0, 7),
              14: (0.9998, 0.0034, 0.010, 21.4, 5), 20: (0.9995, 0.0038, 0.021, 14.9, 14),
              25: (0.9985, 0.0049, 0.024, 10.5, 23), 30: (0.9992, 0.0020, 0.006, 5.7, 11)}
    fields = ("r_squared_in_range", "rmse_share_of_premium_in_range",
              "delta_rmse_share_of_premium_in_range", "mean_abs_nested_delta")
    for year, cells in quoted.items():
        row = proxy.loc[year]
        for field, places, cell in zip(fields, (4, 4, 3, 1), cells):
            assert abs(float(row[field]) - cell) <= 0.5 * 10 ** -places + 1e-12, (
                f"the document prints {cell} for policy year {year}'s {field}, and the table "
                f"says {float(row[field]):.5f}")
        relative = (100 * float(row["delta_rmse_share_of_premium_in_range"])
                    / float(row["mean_abs_nested_delta"]))
        assert round(100 * relative) == cells[-1], (
            f"policy year {year}'s delta error is {100 * relative:.1f}% of the nested delta and "
            f"the document says {cells[-1]}%")


def test_the_model_risk_and_rider_economics_figures_match_their_tables():
    """`docs/validation.md`'s last two sections, one of which had drifted.

    The model risk table reports the extra residual standard deviation averaged across the five
    episodes. The script printed only the whole-window episode, so the document's aggregation
    existed nowhere in the repository and three of its nine cells sat a last digit stale - the
    17% arm's S3 at -0.0003 for a mean of -0.00025, and the 26% arm's S1 and S3 at -0.0012 and
    -0.0004 for -0.00115 and -0.00035. The script now prints the average as well, and this holds
    the document to it.

    The rider economics block came through clean: all seventy-five cells of the cohort table and
    all twenty-five of the stress table reproduce. What is pinned here is the summarising claims
    rather than the cells, because those are the sentences a reader acts on, and the covid
    window's length, which the prose had as eleven weeks for a window of five.
    """
    import numpy as np
    import pandas as pd

    def table(name):
        return pd.read_csv(paths.TABLES / f"{name}.csv")

    risk = table("hedge_model_risk")
    off = risk[risk["model"] != "calibrated"]
    across = off.pivot_table(index="model", columns="strategy", values="extra_sd_pct",
                             aggfunc="mean")
    quoted = {"flat volatility": (-0.0017, -0.0024, -0.0016),
              "long-run vol 17%": (+0.0004, +0.0002, -0.0002),
              "long-run vol 26%": (-0.0011, -0.0011, -0.0003)}
    for model, cells in quoted.items():
        for strategy, cell in zip(("S1", "S2", "S3"), cells):
            mean = float(across.loc[model, strategy])
            assert abs(mean - cell) <= 0.5e-4 + 1e-12, (
                f"the document prints {cell:+.4f} for {model} on {strategy}, and the average "
                f"across episodes is {mean:+.6f}")

    # "Two of the nine averages are positive, and twelve of the forty-five episode-by-strategy
    # cells behind them"
    assert int((across.to_numpy() > 0).sum()) == 2
    assert [int((off["extra_sd_pct"] > 0).sum()), len(off)] == [12, 45]

    # "the median cell is a fifth of the calibrated residual and the worst is nine tenths of it"
    calibrated = risk[risk["model"] == "calibrated"].set_index(["episode", "strategy"])
    base = [float(calibrated.loc[(e, s), "pnl_sd_pct"])
            for e, s in zip(off["episode"], off["strategy"])]
    share = off["extra_sd_pct"].abs().to_numpy() / np.asarray(base)
    assert round(float(np.median(share)), 1) == 0.2
    assert round(float(share.max()), 1) == 0.9

    economics = table("rider_economics")
    decade = economics[economics["scenario"] == "the whole replay"]
    assert len(economics) == 75, f"{len(economics)} runs, and the identity is claimed over 75"
    # The identity's own residual is below the precision the table is stored at, so what can be
    # checked here is that the net column and the ledger's own profit are the same number.
    assert (decade["net"] - decade["ledger_pnl"]).abs().max() == 0.0

    unhedged = decade[decade["strategy"] == "S0"]
    futures = decade[decade["strategy"] == "S2"]
    # "the unhedged book accumulates fees and earns 14 to 40 basis points, the hedged books borrow
    # and pay", and "the unhedged rider earned 174 to 529 basis points a year"
    assert [round(float(unhedged["cash_interest_bp"].min())),
            round(float(unhedged["cash_interest_bp"].max()))] == [14, 40]
    assert [round(float(unhedged["net_bp"].min())),
            round(float(unhedged["net_bp"].max()))] == [174, 529]
    assert float(futures["cash_interest_bp"].max()) < 0, "S2 is claimed to borrow in every cohort"

    # "the funding ran 35 to 52 basis points of benefit base a year ... between a hundred and
    # three hundred times the trading cost". The document rounds that ratio to one significant
    # figure at both ends; the figures it rounds are 97 and 333.
    funding = futures["cash_interest_bp"].abs()
    assert [round(float(funding.min())), round(float(funding.max()))] == [35, 52]
    ratio = funding / futures["hedge_cost_bp"]
    assert [round(float(ratio.min())), round(float(ratio.max()))] == [97, 333]

    # "An annual rate on an eleven-day window ... the covid column takes five weeks - 24 trading
    # days - and multiplies them by eleven"
    windows = economics.groupby("scenario")[["days", "years"]].first()
    assert int(windows.loc["volmageddon", "days"]) == 11
    assert int(windows.loc["covid", "days"]) == 24
    assert round(float(windows.loc["covid", "days"]) / 5) == 5
    assert round(1 / float(windows.loc["covid", "years"])) == 11

    # "both need more than they are paid: 214 basis points drawing first at 76, 204 drawing first
    # at 81, against the 125 charged", and six of the eight admit no charge at all
    pricing = table("rider_break_even")
    assert len(pricing) == 8
    assert int(pricing["fair_pct"].isna().sum()) == 6
    solved = pricing.dropna(subset=["fair_pct"]).set_index("first_withdrawal_age")
    assert set(solved["issue_age"]) == {75}
    assert [round(10000 * float(solved.loc[age, "fair_pct"])) for age in (76, 81)] == [214, 204]
    assert {round(10000 * float(v)) for v in pricing["charged_pct"]} == {125}

    # "it stops getting cheaper near 5%" - the fee curve's minimum, not an endpoint
    curve = table("rider_fee_curve").set_index("fee_pct")
    assert round(float(curve["value"].idxmin()), 4) == 0.05


def test_the_value_error_above_the_money_matches_the_band_table():
    """The band table in `docs/limitations.md`, which replaced a claim that nothing checked it.

    That document said the range above the money had no independent check behind it. It has one.
    The convexity surface reaches past the ratchet's ceiling by valuing the near-the-base nodes
    along a ladder of equity shifts, so 198 of its 548 nodes sit above a moneyness of one, each
    with a nested value beside the proxy's - and the proxy misses them by a median 2.05 per 100 of
    premium just past the money and 10.76 further out, against 0.20 to 0.68 in range. The
    measurement was in the committed table all along; nothing read it, because the surface script
    summarised only the nodes inside the design range and reported the rest as a single share.

    Errors are checked against premium rather than each node's own nested value on purpose. Above
    the money the guarantee is an asset and the nested value crosses zero inside both bands, so a
    ratio to it reports where the denominator is small, not where the fit is wrong. The first
    draft of this finding used that ratio and had to be redone.
    """
    import numpy as np
    import pandas as pd

    comparison = pd.read_csv(paths.TABLES / "convexity_proxy_comparison.csv")
    assert len(comparison) == 548
    # At or above, which is the set the band table covers: ten nodes sit exactly on the ratchet's
    # ceiling and the rest are rungs of the ladder above it.
    above = comparison[comparison["moneyness"] >= 1.0]
    assert [len(above), int((comparison["moneyness"] == 1.0).sum())] == [208, 10]

    # The structural claim the whole paragraph rests on: a recorded post-event anniversary cannot
    # be above its own benefit base, so the design range stops at exactly one.
    inside = comparison[~comparison["outside_design"]]
    assert float(inside["moneyness"].max()) == approx(1.0, abs=1e-9)

    bands = {(0.00, 0.85): (293, 0.01, 0.40, 1.02), (0.85, 1.00): (47, 0.23, 0.50, 1.19),
             (1.00, 1.20): (81, 0.88, 2.05, 7.31), (1.20, 1.60): (127, 1.00, 10.76, 24.02)}
    for (low, high), (nodes, outside, median, ninetieth) in bands.items():
        block = comparison[(comparison["moneyness"] >= low) & (comparison["moneyness"] < high)]
        assert len(block) == nodes, f"{len(block)} nodes between {low} and {high}, not {nodes}"
        assert round(float(block["outside_design"].mean()), 2) == outside
        error = block["value_error"].abs()
        assert round(float(error.median()), 2) == median, (
            f"median error between {low} and {high} is {float(error.median()):.3f} and the "
            f"document says {median}")
        assert round(float(error.quantile(0.9)), 2) == ninetieth

    # "On 20 of the 208 the proxy returns a liability of the wrong sign"
    assert int(((above["nested_value"] > 0) & (above["proxy_value"] < 0)).sum()) == 20
    # "the median nested value is -3.3 in the first band and -8.1 in the second"
    for low, high, level in ((1.00, 1.20, -3.3), (1.20, 1.60, -8.1)):
        block = comparison[(comparison["moneyness"] >= low) & (comparison["moneyness"] < high)]
        assert round(float(block["nested_value"].median()), 1) == level

    # The in-range figure both documents quote the bands against, as a share of a premium of 100
    accuracy = pd.read_csv(paths.TABLES / "proxy_accuracy.csv")
    in_range = 100 * accuracy["rmse_share_of_premium_in_range"]
    assert [round(float(in_range.min()), 2), round(float(in_range.max()), 2)] == [0.20, 0.68]
    assert float(np.median(in_range)) < 0.68

    # The index-linked balance behind E5, which the notebook had doubling over 2025. It grew by
    # about three quarters, and the filed sensitivity moved by a factor of 330 on top of it -
    # which is the whole point of that section, so the growth figure carries weight.
    book = pd.read_csv(paths.DATA_RAW / "jackson_book_statistics.csv", comment="#")
    funds = book.set_index("as_of")["rila_contract_holder_funds_musd"]
    growth = float(funds["2025-12-31"]) / float(funds["2024-12-31"]) - 1.0
    assert round(growth, 2) == 0.74, f"the index-linked balance grew {growth:.1%} over 2025"
    # The same balance-sheet date appears once per filing that discloses it, so the duplicates go
    # before the series is indexed by date.
    filed = pd.read_csv(paths.TABLES / "disclosed_hedge_offset.csv")
    filed = filed[filed["liability_basis"] == "market risk benefit"]

    def rila_move(shock: str) -> pd.Series:
        rows = filed[filed["shock"] == shock].drop_duplicates(subset=["as_of"])
        return rows.set_index("as_of")["rila_impact_musd"].abs()

    # The $4m to $1,321m pair the prose quotes is the up-shock side, which is where the factor of
    # 330 comes from; the down shock moves from 10 to 1,594 over the same year.
    up = rila_move("equity_up_10pct")
    assert [round(float(up["2024-12-31"])), round(float(up["2025-12-31"]))] == [4, 1321]
    assert round(float(up["2025-12-31"]) / float(up["2024-12-31"])) == 330
    down = rila_move("equity_down_10pct")
    assert [round(float(down["2024-12-31"])), round(float(down["2025-12-31"]))] == [10, 1594]


def test_the_instrument_library_builds_what_its_docstring_counts():
    """A stale count, which is what the module opened with for several passes.

    `instruments.py` began "The six instrument types Jackson names" and gave a paragraph to the
    total return swap as something the module builds. It builds five: the swap had been removed,
    `docs/methodology.md` and `strategies.py` both recorded the removal and the reason, and the
    docstring of the module it was removed from did not.

    An instrument here is a class implementing the whole protocol the solve and the simulator
    call - exposures, value, notional, trade_cost, carry - which is what distinguishes the five
    from `Exposures`, `HedgeMarket` and `SmileShape` in the same file. The count in the first
    line has to match. Adding a sixth means saying so there, which is the point.
    """
    import ast

    protocol = {"exposures", "value", "notional", "trade_cost", "carry"}
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
    source = (paths.ROOT / "vahedge" / "hedge" / "instruments.py").read_text()
    tree = ast.parse(source)

    built = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        methods = {item.name for item in node.body
                   if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))}
        if protocol <= methods:
            built.append(node.name)

    first_line = ast.get_docstring(tree).splitlines()[0].lower()
    counted = [value for word, value in words.items() if f"{word} " in first_line]
    assert len(counted) == 1, (
        f"the first line has to name exactly one count, or this cannot check it: {first_line!r}"
    )
    assert counted[0] == len(built), (
        f"the docstring counts {counted[0]} and the module builds {len(built)}: {sorted(built)}"
    )


def test_the_source_register_and_the_files_it_names_agree():
    """The mechanical half of the reference audit, which had never been a test.

    Four things that are facts rather than judgements, and four ways a reference layer rots: a
    row names a file that is not there; something cites an identifier no row defines; the
    register and the bibliography drift apart; a file sits under references/ that no row claims.
    The judgement half - whether an extract really is the filing it says it is, whether the
    metadata is right - needs the original, and the original is on sec.gov, which nothing here
    can reach from a script. `references/README.md` says so, and this test stops where that
    begins.

    No assertion that every row is used somewhere, which the first version had. Measured, the
    identifiers appear nowhere outside this directory: all thirty are in `references.md` and
    nothing else cites one, because the documents name Item 7A and Note 6 and the rate sheet in
    prose and the mapping to components runs the other way, through the register's `used_by`
    column. So "cited nowhere" would have meant "absent from the bibliography" while reading
    like a usage check - the shape of a check that promises more than it does.

    One deliberate hole in the numbering: a short-rate series was retrieved, measured against
    the rate already in the panel, and dropped. The gap is left rather than closed by
    renumbering, which would invalidate every identifier already cited. The register's header
    says which identifier that was and why, and this reads the retired ones out of that header
    rather than carrying a list of its own - which also stops the check tripping over its own
    docstring, as the first version of it did.
    """
    import re
    import pandas as pd

    csv_path = paths.ROOT / "references" / "source_register.csv"
    header = [line for line in csv_path.read_text().splitlines() if line.startswith("#")]
    retired = {found for line in header if "absent on purpose" in line
               for found in re.findall(r"REF-\d{3}", line)}
    assert retired, "the register's header names no retired identifier; has it been rewritten?"

    register = pd.read_csv(csv_path, comment="#")
    assert len(register) > 25, f"only {len(register)} rows, so the check is not running"
    defined = set(register["ref_id"])
    assert len(defined) == len(register), "the register has a duplicate identifier"
    assert not register.isna().all(axis=1).any(), "the register has a blank row"

    missing = []
    claimed = set()
    for _, row in register.iterrows():
        if pd.isna(row["local_path"]):
            continue
        for piece in str(row["local_path"]).split(";"):
            path = paths.ROOT / piece.strip()
            claimed.add(path.resolve())
            if not path.exists():
                missing.append(f"{row['ref_id']} names {piece.strip()}")
    assert not missing, "rows naming a file that is not there:\n" + "\n".join(missing)

    index_files = {"README.md", "references.md", "source_register.csv"}
    unclaimed = sorted(
        path.relative_to(paths.ROOT).as_posix()
        for path in (paths.ROOT / "references").rglob("*")
        if path.is_file() and path.name not in index_files
        and path.resolve() not in claimed
    )
    assert not unclaimed, f"files under references/ that no row claims: {unclaimed}"

    cited: dict = {}
    for pattern in ("docs/*.md", "*.md", "vahedge/**/*.py", "scripts/*.py", "tests/*.py",
                    "references/*.md", "data/raw/*.csv"):
        for path in sorted(paths.ROOT.glob(pattern)):
            for found in re.findall(r"REF-\d{3}", path.read_text(errors="replace")):
                cited.setdefault(found, set()).add(path.name)
    undefined = set(cited) - defined - retired
    assert not undefined, f"cited and not in the register: {sorted(undefined)}"

    bibliography = (paths.ROOT / "references" / "references.md").read_text()
    listed = set(re.findall(r"REF-\d{3}", bibliography))
    assert listed == defined, (
        f"the bibliography and the register disagree: only in the register "
        f"{sorted(defined - listed)}, only in the bibliography {sorted(listed - defined)}"
    )


def test_the_sizing_weight_table_in_the_documents_reads_its_vega_column_right():
    """`docs/validation.md`'s weight-sweep table, where one column carried two conventions.

    `sizing.effectiveness` reports `1 - |after| / |before|`, so a hedge that enlarges an exposure
    comes back negative and -77.17 means the exposure ends at 78.17 times the size it started.
    The document tabulates that multiple. Three of its four rows carried the raw column instead -
    81, 81 and 12 against the multiples 82, 82 and 13 - while the first row, 78, was already the
    multiple. One row in four on the other reading, which is the kind of thing a column of four
    numbers hides.

    The whole row is pinned rather than the vega alone, because the weight sweep is the one place
    the sizing objective's single operative degree of freedom is measured and the residual, the
    cost and the gamma closed are read together.
    """
    import pandas as pd

    weights = pd.read_csv(paths.TABLES / "hedge_sizing_weights.csv")
    put = weights[weights["strategy"] == "S3"].set_index("objective")
    expected = {
        "gamma and vega equal, the default": (0.137, 13.73, 95, 78),
        "gamma ten times vega": (0.137, 14.89, 100, 82),
        "gamma a hundred times vega": (0.137, 14.90, 100, 82),
        "vega ten times gamma": (0.153, 3.98, 17, 13),
    }
    assert set(put.index) == set(expected), f"the objectives changed: {sorted(put.index)}"
    for objective, (residual, cost, gamma, vega) in expected.items():
        row = put.loc[objective]
        assert round(100 * float(row["pnl_sd_pct"]), 3) == residual, objective
        assert round(100 * float(row["total_cost_pct"]), 2) == cost, objective
        assert round(100 * float(row["gamma_removed"])) == gamma, objective
        assert round(1.0 - float(row["vega_removed"])) == vega, objective

    # S2 has no put, so every row of it has to be identical - the control the table rests on.
    flat = weights[weights["strategy"] == "S2"]
    for column in ("pnl_sd_pct", "total_cost_pct", "gamma_removed", "vega_removed"):
        assert flat[column].nunique() == 1, f"S2's {column} moved with the weights"

    # "worth eleven points of trading cost" - the spread the heading claims
    spread = 100 * (put["total_cost_pct"].max() - put["total_cost_pct"].min())
    assert round(spread) == 11, f"the cost spread is {spread:.2f} points"
