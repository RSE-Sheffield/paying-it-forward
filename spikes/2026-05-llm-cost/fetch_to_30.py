"""Fetch 20 more PMC full-text papers across diverse fields, bringing total to 30."""
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
    ("biochem-1", '"United Kingdom"[AD] AND protein[Title] AND 2023[PDAT] AND open access[Filter]'),
    ("medicine-trial", '"United Kingdom"[AD] AND clinical trial[Title] AND 2023[PDAT] AND open access[Filter]'),
    ("medicine-cohort", '"United Kingdom"[AD] AND UK Biobank[Title/Abstract] AND 2023[PDAT] AND open access[Filter]'),
    ("psychology-1", '"United Kingdom"[AD] AND psychology[Title] AND 2023[PDAT] AND open access[Filter]'),
    ("environ-1", '"United Kingdom"[AD] AND climate[Title] AND 2023[PDAT] AND open access[Filter]'),
    ("environ-2", '"United Kingdom"[AD] AND biodiversity[Title] AND 2023[PDAT] AND open access[Filter]'),
    ("neuro-empirical", '"United Kingdom"[AD] AND brain imaging[Title] AND 2023[PDAT] AND open access[Filter]'),
    ("immunology-2", '"United Kingdom"[AD] AND vaccine[Title] AND 2023[PDAT] AND open access[Filter]'),
    ("genetics-2", '"United Kingdom"[AD] AND GWAS[Title/Abstract] AND 2023[PDAT] AND open access[Filter]'),
    ("micro-1", '"United Kingdom"[AD] AND microbiome[Title] AND 2023[PDAT] AND open access[Filter]'),
    ("cs-deep", '"United Kingdom"[AD] AND deep learning[Title] AND 2023[PDAT] AND open access[Filter]'),
    ("cs-nlp", '"United Kingdom"[AD] AND natural language[Title] AND 2023[PDAT] AND open access[Filter]'),
    ("health-survey", '"United Kingdom"[AD] AND survey[Title] AND 2023[PDAT] AND open access[Filter]'),
    ("pharma-2", '"United Kingdom"[AD] AND drug[Title] AND 2023[PDAT] AND open access[Filter]'),
    ("cancer-1", '"United Kingdom"[AD] AND cancer[Title] AND 2023[PDAT] AND open access[Filter]'),
    ("cardio-1", '"United Kingdom"[AD] AND cardiovascular[Title] AND 2023[PDAT] AND open access[Filter]'),
    ("public-h-1", '"United Kingdom"[AD] AND public health[Title] AND 2023[PDAT] AND open access[Filter]'),
    ("dental-1", '"United Kingdom"[AD] AND dental[Title] AND 2023[PDAT] AND open access[Filter]'),
    ("nutrition-1", '"United Kingdom"[AD] AND nutrition[Title] AND 2023[PDAT] AND open access[Filter]'),
    ("pediatric-1", '"United Kingdom"[AD] AND children[Title] AND 2023[PDAT] AND open access[Filter]'),
    ("imaging-1", '"United Kingdom"[AD] AND MRI[Title] AND 2023[PDAT] AND open access[Filter]'),
    ("infection-1", '"United Kingdom"[AD] AND infection[Title] AND 2023[PDAT] AND open access[Filter]'),
    ("ageing-1", '"United Kingdom"[AD] AND ageing[Title] AND 2023[PDAT] AND open access[Filter]'),
    ("mental-1", '"United Kingdom"[AD] AND mental health[Title] AND 2023[PDAT] AND open access[Filter]'),
    ("review-1", '"United Kingdom"[AD] AND systematic review[Title] AND 2023[PDAT] AND open access[Filter]'),
]


def esearch_pmc(query):
    r = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={"db": "pmc", "term": query, "retmax": 5, "sort": "relevance", "tool": TOOL, "email": EMAIL},
        timeout=30,
    )
    if r.status_code != 200:
        return []
    root = ET.fromstring(r.text)
    return [el.text for el in root.findall(".//Id")]


def fetch_pmc_full(pmcid):
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
    title = "".join(title_el.itertext()).strip() if title_el is not None else None
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
    target = 30
    print(f"Have {len(papers)}, need {target - len(papers)} more\n")
    used = set()
    for field, query in QUERIES:
        if len(papers) >= target:
            break
        try:
            ids = esearch_pmc(query)
            time.sleep(1.5)
            for pmcid in ids:
                pmcid_full = f"PMC{pmcid}"
                if pmcid_full in EXISTING_PMCIDS or pmcid_full in used:
                    continue
                meta = fetch_pmc_full(pmcid)
                time.sleep(1.5)
                if not meta or not meta.get("fulltext") or not meta.get("abstract"):
                    continue
                if meta["fulltext_chars"] < 5000:  # skip stubs
                    continue
                meta["field"] = field
                papers.append(meta)
                used.add(pmcid_full)
                print(f"  {len(papers):2}/30 {field:18} {pmcid_full:>10}  full {meta['fulltext_chars']:>7,}ch  abs {len(meta['abstract']):>5}ch | {(meta.get('title') or '')[:55]}")
                break
        except Exception as e:
            print(f"  {field}: ERROR {e}")
            time.sleep(2)

    (OUT_DIR / "papers_fulltext_30.json").write_text(json.dumps(papers, indent=2))
    print(f"\nSaved {len(papers)} papers")


if __name__ == "__main__":
    main()
