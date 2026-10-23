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
import re
import struct

from tests.checks import raises
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
