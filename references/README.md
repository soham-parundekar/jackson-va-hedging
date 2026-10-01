# References

Everything the project takes from outside itself, and where it came from.

`source_register.csv` is the index: one row per source, with an identifier, the metadata
needed to find it again, and the component of the project that depends on it. When a result
in `docs/validation.md` cites REF-004, that row says which filing it is, what was taken from
it, and whether the document itself is stored here.

## What is stored and what is not

Preserved in full here: the short issuer and regulatory documents, which are public and
small enough to keep.

Preserved as extracts: the 10-Ks. Each is between one and two megabytes of rendered HTML and
the project uses a few pages of each. What is stored is the verbatim text of the sections
relied on, under a header giving the accession number, the document name, the permanent SEC
URL and the retrieval date, so the original can be pulled again and the extract checked
against it. The extracts are marked as extracts and are not presented as the filing.

Not stored: nothing that is used. Every source in the register is either preserved here or
carries enough retrieval metadata to obtain it in one step.

There is a technical reason the retrieval is not automated. The environment this was built in
cannot reach sec.gov, FRED or Cboe from a script, only through a browser, so the acquisition
step is not reproducible by running a command in this repository. That is why the retrieved
data sits in `data/raw/` under version control rather than behind a download script, and why
the filing text sits here. `docs/data_sources.md` says the same thing from the data side.

## Copyright

The SEC filings and the SEC's XBRL company-facts API are public records of the United States
government and of the registrants filing with it, and are freely redistributable. The FRED
series are published by the Federal Reserve Bank of St. Louis; their terms permit
redistribution of the data. The Cboe index histories and the delayed option chain are
published by Cboe on its public website for non-commercial use; the extracts kept in
`data/raw/` are the fields this project uses, and the register records the endpoint each came
from. The mortality tables are from the Society of Actuaries' public Mortality and Other Rate
Tables repository.

Journal articles are cited, never stored. The register carries the DOI for each, and the
metadata in it was checked against Crossref rather than written from memory.

## Layout

```
references/
  README.md                  this file
  source_register.csv        the index, one row per source
  references.md              the annotated bibliography, with why each source is used
  sec_filings/               extracts from Jackson Financial's filings
  issuer_documents/          product filings: rate sheets and prospectus extracts
```

What each market and mortality series is, what it covers and what it does not, lives in
`docs/data_sources.md` rather than in a directory here. It belongs next to the loader that
reads it, and the register points at it.
