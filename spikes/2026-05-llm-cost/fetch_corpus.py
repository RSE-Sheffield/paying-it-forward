"""Rebuild the spike's paper corpus from papers_manifest.csv.

The paper texts are not committed (mixed licences). This fetches exactly the papers
the spike used — full text from PMC by PMCID, abstract from PubMed by PMID for the
abstract-only records — and writes the three JSON files the classify scripts read:

    papers/papers.json               first 10 (mixed abstract + full text)
    papers/papers_fulltext_10.json   10 full-text papers
    papers/papers_fulltext_30.json   30 full-text papers

PMC may have updated an article since May 2026, so token counts can differ slightly
from the recorded runs.
"""
import csv
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

HERE = Path(__file__).parent
OUT_DIR = HERE / "papers"
EMAIL = "daren.howell@crewcreate.co.uk"
TOOL = "paying-it-forward-spike"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def fetch_pmc_full(pmcid):
    """Full text + metadata for one PMC article (same parsing as fetch_to_30.py)."""
    r = requests.get(EUTILS, params={"db": "pmc", "id": pmcid.removeprefix("PMC"), "rettype": "xml",
                                     "tool": TOOL, "email": EMAIL}, timeout=60)
    if r.status_code != 200 or not r.text.strip().startswith("<"):
        return None
    root = ET.fromstring(r.text)
    abs_parts = [" ".join(p.itertext()).strip() for p in root.findall(".//abstract//p")]
    chunks = []
    for sec in root.findall(".//body//sec"):
        tt_el = sec.find("title")
        tt = (tt_el.text or "").strip() if tt_el is not None else ""
        text = " ".join(" ".join(t for t in sec.itertext() if t).split())
        if text:
            chunks.append(f"## {tt}\n{text}")
    fulltext = "\n\n".join(chunks)
    return {"abstract": "\n\n".join(p for p in abs_parts if p) or None,
            "fulltext": fulltext or None, "fulltext_chars": len(fulltext)}


def fetch_pubmed_abstract(pmid):
    r = requests.get(EUTILS, params={"db": "pubmed", "id": pmid, "rettype": "abstract", "retmode": "xml",
                                     "tool": TOOL, "email": EMAIL}, timeout=60)
    if r.status_code != 200:
        return None
    root = ET.fromstring(r.text)
    parts = [" ".join(a.itertext()).strip() for a in root.findall(".//Abstract/AbstractText")]
    return {"abstract": "\n\n".join(p for p in parts if p) or None, "fulltext": None, "fulltext_chars": 0}


def main():
    OUT_DIR.mkdir(exist_ok=True)
    rows = list(csv.DictReader(open(HERE / "papers_manifest.csv")))
    sets = {"in_first10": [], "in_fulltext10": [], "in_fulltext30": []}
    for i, row in enumerate(rows, 1):
        body = fetch_pmc_full(row["pmcid"]) if row["pmcid"] else fetch_pubmed_abstract(row["pmid"])
        time.sleep(0.5)  # stay under NCBI's 3 requests/second without an API key
        if not body:
            print(f"  {i:2}/{len(rows)} FAILED {row['pmcid'] or row['pmid']}")
            continue
        paper = {"field": row["field"], "pmid": row["pmid"], "pmcid": row["pmcid"] or None,
                 "doi": row["doi"] or None, "title": row["title"], "journal": row["journal"],
                 "year": row["year"], **body}
        for col in sets:
            if row[col] == "Y":
                sets[col].append(paper)
        print(f"  {i:2}/{len(rows)} {row['pmcid'] or row['pmid']:>12}  {paper['fulltext_chars']:>7,} chars  {row['title'][:50]}")
    for col, name in [("in_first10", "papers.json"), ("in_fulltext10", "papers_fulltext_10.json"),
                      ("in_fulltext30", "papers_fulltext_30.json")]:
        (OUT_DIR / name).write_text(json.dumps(sets[col], indent=2))
        print(f"Wrote {len(sets[col])} papers to papers/{name}")


if __name__ == "__main__":
    main()
