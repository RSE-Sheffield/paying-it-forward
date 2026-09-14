"""Fetch 8 additional PMC full-text papers across fields, to bring total to 10."""
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

OUT_DIR = Path(__file__).parent / "papers"
OUT_DIR.mkdir(exist_ok=True)
EXISTING = json.loads((OUT_DIR / "papers.json").read_text())
EXISTING_FULLTEXT = [p for p in EXISTING if p.get("fulltext")]

EMAIL = "daren.howell@crewcreate.co.uk"
TOOL = "paying-it-forward-spike"

# Field queries — searching PMC directly to guarantee full text
QUERIES = [
    ("genomics", '"United Kingdom"[AD] AND "data availability"[Title/Abstract] AND 2023[PDAT] AND "genome"[Title/Abstract]'),
    ("ecology", '"United Kingdom"[AD] AND "ecology"[MH] AND 2023[PDAT]'),
    ("epidemiology", '"United Kingdom"[AD] AND "epidemiology"[MH] AND 2023[PDAT] AND "cohort study"[Title/Abstract]'),
    ("bioinformatics", '"United Kingdom"[AD] AND "bioinformatics"[MH] AND 2023[PDAT]'),
    ("immunology", '"United Kingdom"[AD] AND "immunology"[MH] AND "T cell"[Title/Abstract] AND 2023[PDAT]'),
    ("machine-learning", '"United Kingdom"[AD] AND "machine learning"[Title/Abstract] AND 2023[PDAT]'),
    ("methods-paper", '"United Kingdom"[AD] AND "we developed"[Title/Abstract] AND "algorithm"[Title/Abstract] AND 2023[PDAT]'),
    ("clinical-trial", '"United Kingdom"[AD] AND "randomized controlled trial"[PT] AND 2023[PDAT] AND "trial registration"[Title/Abstract]'),
]


def esearch_pmc(query: str) -> str | None:
    """Return one PMCID for the query (searching PMC directly)."""
    r = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={"db": "pmc", "term": query, "retmax": 5, "sort": "relevance", "tool": TOOL, "email": EMAIL},
        timeout=20,
    )
    r.raise_for_status()
    root = ET.fromstring(r.text)
    ids = [el.text for el in root.findall(".//Id")]
    return ids[0] if ids else None


def fetch_pmc_meta_and_full(pmcid: str) -> dict | None:
    r = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params={"db": "pmc", "id": pmcid, "rettype": "xml", "tool": TOOL, "email": EMAIL},
        timeout=60,
    )
    if r.status_code != 200 or not r.text.strip().startswith("<"):
        return None
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        return None

    # Title
    title_el = root.find(".//article-meta/title-group/article-title")
    title = "".join(title_el.itertext()).strip() if title_el is not None else None
    # PMID
    pmid_el = root.find(".//article-meta/article-id[@pub-id-type='pmid']")
    pmid = pmid_el.text if pmid_el is not None else None
    # DOI
    doi_el = root.find(".//article-meta/article-id[@pub-id-type='doi']")
    doi = doi_el.text if doi_el is not None else None
    # Journal
    j_el = root.find(".//journal-title")
    journal = j_el.text if j_el is not None else None
    # Year
    y_el = root.find(".//pub-date/year")
    year = y_el.text if y_el is not None else None
    # Abstract
    abs_parts = []
    for p in root.findall(".//abstract//p"):
        text = " ".join(p.itertext()).strip()
        if text:
            abs_parts.append(text)
    abstract = "\n\n".join(abs_parts) or None
    # Full text
    chunks = []
    for sec in root.findall(".//body//sec"):
        title_el = sec.find("title")
        tt = (title_el.text or "").strip() if title_el is not None else ""
        text = " ".join(t for t in sec.itertext() if t)
        text = " ".join(text.split())
        if text:
            chunks.append(f"## {tt}\n{text}")
    for sec in root.findall(".//back//sec"):
        title_el = sec.find("title")
        tt = (title_el.text or "").strip() if title_el is not None else ""
        text = " ".join(t for t in sec.itertext() if t)
        text = " ".join(text.split())
        if text:
            chunks.append(f"## {tt}\n{text}")
    fulltext = "\n\n".join(chunks)
    return {
        "pmcid": f"PMC{pmcid}",
        "pmid": pmid,
        "doi": doi,
        "title": title,
        "journal": journal,
        "year": year,
        "abstract": abstract,
        "fulltext": fulltext if fulltext else None,
        "fulltext_chars": len(fulltext),
    }


def main():
    papers = []
    for field, query in QUERIES:
        try:
            pmcid = esearch_pmc(query)
            if not pmcid:
                print(f"  {field}: no PMC result")
                continue
            time.sleep(0.4)
            meta = fetch_pmc_meta_and_full(pmcid)
            time.sleep(0.4)
            if not meta or not meta.get("fulltext"):
                print(f"  {field}: PMC{pmcid} no full text")
                continue
            paper = {"field": field, **meta}
            papers.append(paper)
            print(f"  {field:18} PMC{pmcid:>10}  PMID {meta.get('pmid') or '-':>9}  full {meta['fulltext_chars']:>7,}ch  | {meta['title'][:60] if meta.get('title') else ''}")
        except Exception as e:
            print(f"  {field}: ERROR {e}")

    # Add existing 2 full-text papers
    for p in EXISTING_FULLTEXT:
        papers.append(p)
        print(f"  (existing) {p['field']:8} PMID {p['pmid']:>9}  full {p['fulltext_chars']:>7,}ch")

    (OUT_DIR / "papers_fulltext_10.json").write_text(json.dumps(papers, indent=2))
    print(f"\nSaved {len(papers)} full-text papers to papers_fulltext_10.json")


if __name__ == "__main__":
    main()
