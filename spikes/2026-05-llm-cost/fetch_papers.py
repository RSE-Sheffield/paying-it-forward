"""Fetch 10 PubMed papers with abstracts + (where available) full-text from PMC.

Strategy: query PubMed for recent UK-affiliated articles with a mix of fields,
fetch abstracts via E-utilities, attempt PMC full-text for open-access ones.
"""
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

OUT_DIR = Path(__file__).parent / "papers"
OUT_DIR.mkdir(exist_ok=True)

EMAIL = "daren.howell@crewcreate.co.uk"
TOOL = "paying-it-forward-spike"

# Mix of fields to mirror the OpenAlex stratification we'll need at scale.
# Use simple PubMed queries per field, UK-affiliated, recent.
QUERIES = [
    ("medicine", '("UK"[Affiliation] OR "United Kingdom"[Affiliation]) AND clinical trial[ptyp] AND 2023:2024[dp]'),
    ("biochem", '("UK"[Affiliation] OR "United Kingdom"[Affiliation]) AND biochemistry[mh] AND 2023:2024[dp]'),
    ("neuroscience", '("UK"[Affiliation] OR "United Kingdom"[Affiliation]) AND neuroscience[mh] AND 2023:2024[dp]'),
    ("immunology", '("UK"[Affiliation] OR "United Kingdom"[Affiliation]) AND immunology[mh] AND 2023:2024[dp]'),
    ("health-prof", '("UK"[Affiliation] OR "United Kingdom"[Affiliation]) AND nursing[mh] AND 2023:2024[dp]'),
    ("pharma", '("UK"[Affiliation] OR "United Kingdom"[Affiliation]) AND pharmacology[mh] AND 2023:2024[dp]'),
    ("genetics", '("UK"[Affiliation] OR "United Kingdom"[Affiliation]) AND genomics[mh] AND 2023:2024[dp]'),
    ("public-health", '("UK"[Affiliation] OR "United Kingdom"[Affiliation]) AND "public health"[mh] AND 2023:2024[dp]'),
    ("psychiatry", '("UK"[Affiliation] OR "United Kingdom"[Affiliation]) AND psychiatry[mh] AND 2023:2024[dp]'),
    ("oncology", '("UK"[Affiliation] OR "United Kingdom"[Affiliation]) AND oncology[mh] AND 2023:2024[dp]'),
]


def esearch_one(query: str) -> str | None:
    """Return one PMID for the query."""
    r = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={
            "db": "pubmed",
            "term": query,
            "retmax": 1,
            "sort": "relevance",
            "tool": TOOL,
            "email": EMAIL,
        },
        timeout=20,
    )
    r.raise_for_status()
    root = ET.fromstring(r.text)
    ids = [el.text for el in root.findall(".//Id")]
    return ids[0] if ids else None


def efetch_abstract(pmid: str) -> dict:
    """Return parsed abstract + metadata for a PMID."""
    r = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params={
            "db": "pubmed",
            "id": pmid,
            "rettype": "abstract",
            "retmode": "xml",
            "tool": TOOL,
            "email": EMAIL,
        },
        timeout=30,
    )
    r.raise_for_status()
    root = ET.fromstring(r.text)
    article = root.find(".//Article")
    if article is None:
        return {"pmid": pmid, "title": None, "abstract": None}
    title_el = article.find(".//ArticleTitle")
    title = "".join(title_el.itertext()).strip() if title_el is not None else None

    # Abstract may have multiple labelled sections
    abs_parts = []
    for ab in article.findall(".//Abstract/AbstractText"):
        label = ab.attrib.get("Label", "")
        text = "".join(ab.itertext()).strip()
        if label:
            abs_parts.append(f"{label}: {text}")
        else:
            abs_parts.append(text)
    abstract = "\n\n".join(abs_parts) if abs_parts else None

    journal_el = article.find(".//Journal/Title")
    journal = journal_el.text if journal_el is not None else None
    year_el = article.find(".//Journal/JournalIssue/PubDate/Year")
    year = year_el.text if year_el is not None else None

    # Try to find PMC ID for full-text access
    pmcid = None
    for art_id in root.findall(".//ArticleId"):
        if art_id.attrib.get("IdType") == "pmc":
            pmcid = art_id.text
            break

    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "journal": journal,
        "year": year,
        "pmcid": pmcid,
    }


def efetch_pmc_fulltext(pmcid: str) -> str | None:
    """Return PMC full-text XML extracted to plain text (methods + data avail prioritised)."""
    r = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params={
            "db": "pmc",
            "id": pmcid.lstrip("PMC"),
            "rettype": "xml",
            "tool": TOOL,
            "email": EMAIL,
        },
        timeout=60,
    )
    if r.status_code != 200 or not r.text.strip().startswith("<"):
        return None
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        return None

    # Collect body text — focus on methods, data availability, code availability
    chunks = []
    for sec in root.findall(".//body//sec"):
        title_el = sec.find("title")
        title = (title_el.text or "").strip() if title_el is not None else ""
        # Get all text in this section
        text = " ".join(t for t in sec.itertext() if t)
        text = " ".join(text.split())
        if text:
            chunks.append(f"## {title}\n{text}")
    # Also pull any back-matter / supplementary / data availability sections
    for sec in root.findall(".//back//sec"):
        title_el = sec.find("title")
        title = (title_el.text or "").strip() if title_el is not None else ""
        text = " ".join(t for t in sec.itertext() if t)
        text = " ".join(text.split())
        if text:
            chunks.append(f"## {title}\n{text}")
    full = "\n\n".join(chunks)
    return full if full else None


def main():
    papers = []
    for field, query in QUERIES:
        try:
            pmid = esearch_one(query)
            if not pmid:
                print(f"  {field}: no result")
                continue
            time.sleep(0.4)
            meta = efetch_abstract(pmid)
            time.sleep(0.4)
            full = None
            if meta.get("pmcid"):
                full = efetch_pmc_fulltext(meta["pmcid"])
                time.sleep(0.4)
            paper = {
                "field": field,
                "pmid": pmid,
                "title": meta["title"],
                "journal": meta["journal"],
                "year": meta["year"],
                "pmcid": meta.get("pmcid"),
                "abstract": meta["abstract"],
                "fulltext": full,
                "fulltext_chars": len(full) if full else 0,
            }
            papers.append(paper)
            print(f"  {field}: PMID {pmid} | PMC {meta.get('pmcid') or '-'} | abs {len(meta['abstract'] or '')}ch | full {len(full or '')}ch")
        except Exception as e:
            print(f"  {field}: ERROR {e}")

    (OUT_DIR / "papers.json").write_text(json.dumps(papers, indent=2))
    print(f"\nSaved {len(papers)} papers to {OUT_DIR / 'papers.json'}")


if __name__ == "__main__":
    main()
