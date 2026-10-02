"""Retry with broader/simpler PMC queries to get 5 more papers."""
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

OUT_DIR = Path(__file__).parent / "papers"
EXISTING = json.loads((OUT_DIR / "papers_fulltext_10.json").read_text())
EXISTING_PMCIDS = {p.get("pmcid") for p in EXISTING}

EMAIL = "daren.howell@crewcreate.co.uk"
TOOL = "paying-it-forward-spike"

QUERIES = [
    ("genomics-uk", 'genomics[Title] AND "United Kingdom"[AD] AND 2023[PDAT] AND open access[Filter]'),
    ("ml-uk", 'machine learning[Title] AND "United Kingdom"[AD] AND 2023[PDAT] AND open access[Filter]'),
    ("epidemiology-uk", 'cohort study[Title] AND "United Kingdom"[AD] AND 2023[PDAT] AND open access[Filter]'),
    ("immunology-uk", 'T cell[Title] AND "United Kingdom"[AD] AND 2023[PDAT] AND open access[Filter]'),
    ("methods-uk", 'novel method[Title] AND "United Kingdom"[AD] AND 2023[PDAT] AND open access[Filter]'),
    ("bioinformatics-uk", 'pipeline[Title] AND "United Kingdom"[AD] AND 2023[PDAT] AND open access[Filter]'),
    ("statistics-uk", 'statistical[Title] AND "United Kingdom"[AD] AND 2023[PDAT] AND open access[Filter]'),
    ("software-uk", 'software[Title] AND "United Kingdom"[AD] AND 2023[PDAT] AND open access[Filter]'),
]


def esearch_pmc(query: str) -> list:
    r = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={"db": "pmc", "term": query, "retmax": 10, "sort": "relevance", "tool": TOOL, "email": EMAIL},
        timeout=30,
    )
    r.raise_for_status()
    root = ET.fromstring(r.text)
    return [el.text for el in root.findall(".//Id")]


def fetch_pmc_full(pmcid: str) -> dict | None:
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
    title_el = root.find(".//article-meta/title-group/article-title")
    title = "".join(title_el.itertext()).strip() if title_el is not None else None
    pmid_el = root.find(".//article-meta/article-id[@pub-id-type='pmid']")
    doi_el = root.find(".//article-meta/article-id[@pub-id-type='doi']")
    j_el = root.find(".//journal-title")
    y_el = root.find(".//pub-date/year")
    abs_parts = [" ".join(p.itertext()).strip() for p in root.findall(".//abstract//p")]
    chunks = []
    for sec in root.findall(".//body//sec"):
        tt_el = sec.find("title")
        tt = (tt_el.text or "").strip() if tt_el is not None else ""
        text = " ".join(t for t in sec.itertext() if t).split()
        text = " ".join(text)
        if text:
            chunks.append(f"## {tt}\n{text}")
    fulltext = "\n\n".join(chunks)
    return {
        "pmcid": f"PMC{pmcid}",
        "pmid": pmid_el.text if pmid_el is not None else None,
        "doi": doi_el.text if doi_el is not None else None,
        "title": title,
        "journal": j_el.text if j_el is not None else None,
        "year": y_el.text if y_el is not None else None,
        "abstract": "\n\n".join(p for p in abs_parts if p) or None,
        "fulltext": fulltext if fulltext else None,
        "fulltext_chars": len(fulltext),
    }


def main():
    papers = list(EXISTING)
    need = 10 - len(papers)
    print(f"Have {len(papers)}, need {need} more.\n")
    for field, query in QUERIES:
        if len(papers) >= 10:
            break
        try:
            ids = esearch_pmc(query)
            time.sleep(1.5)
            for pmcid in ids:
                if f"PMC{pmcid}" in EXISTING_PMCIDS or any(p.get("pmcid") == f"PMC{pmcid}" for p in papers):
                    continue
                meta = fetch_pmc_full(pmcid)
                time.sleep(1.5)
                if not meta or not meta.get("fulltext"):
                    continue
                meta["field"] = field
                papers.append(meta)
                print(f"  {field:18} PMC{pmcid:>10}  full {meta['fulltext_chars']:>7,}ch  | {meta['title'][:55] if meta.get('title') else ''}")
                break
        except Exception as e:
            print(f"  {field}: ERROR {e}")
            time.sleep(2)

    (OUT_DIR / "papers_fulltext_10.json").write_text(json.dumps(papers, indent=2))
    print(f"\nSaved {len(papers)} full-text papers")


if __name__ == "__main__":
    main()
