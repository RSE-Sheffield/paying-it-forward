"""Download PDFs for the 30-paper corpus, keyed by PMC ID.

Uses the PMC Article Datasets S3 bucket (the successor to NCBI's retired
oa.fcgi / FTP OA service — see https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/).
The bucket is public and needs no credentials. Only articles in the PMC
Open Access Subset have a PDF available; others are skipped and reported
at the end.
"""
import json
import time
from pathlib import Path

import requests

BASE_DIR = Path(__file__).parent
PAPERS_FILE = BASE_DIR / "papers" / "papers_fulltext_30.json"
OUT_DIR = BASE_DIR / "papers" / "pdfs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

S3_BASE = "https://pmc-oa-opendata.s3.amazonaws.com"


def latest_metadata(pmcid: str) -> dict | None:
    """Find the highest-versioned metadata object for a PMCID and return it."""
    r = requests.get(
        f"{S3_BASE}/",
        params={"list-type": "2", "prefix": f"metadata/{pmcid}."},
        timeout=30,
    )
    r.raise_for_status()
    import xml.etree.ElementTree as ET

    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    root = ET.fromstring(r.text)
    keys = [el.text for el in root.findall(".//s3:Key", ns)]
    if not keys:
        return None
    # Keys look like metadata/PMC10575791.1.json — pick the highest version.
    keys.sort(key=lambda k: int(k.rsplit(".", 2)[-2]))
    meta_r = requests.get(f"{S3_BASE}/{keys[-1]}", timeout=30)
    meta_r.raise_for_status()
    return meta_r.json()


def main() -> None:
    papers = json.loads(PAPERS_FILE.read_text())
    ok, skipped = [], []

    for paper in papers:
        pmcid = paper.get("pmcid")
        if not pmcid:
            skipped.append((paper.get("pmid"), "no pmcid"))
            continue

        out_path = OUT_DIR / f"{pmcid}.pdf"
        if out_path.exists():
            ok.append(pmcid)
            continue

        meta = latest_metadata(pmcid)
        if meta is None:
            skipped.append((pmcid, "not in PMC Article Datasets"))
            continue

        pdf_url = meta.get("pdf_url")
        if not pdf_url:
            skipped.append((pmcid, "no pdf_url (not OA / no PDF permitted)"))
            continue

        https_url = pdf_url.replace("s3://pmc-oa-opendata/", f"{S3_BASE}/")
        r = requests.get(https_url, timeout=60)
        if r.status_code != 200:
            skipped.append((pmcid, f"http {r.status_code} fetching pdf"))
            continue

        out_path.write_bytes(r.content)
        ok.append(pmcid)
        print(f"downloaded {pmcid} ({len(r.content) / 1024:.0f} KB)")
        time.sleep(0.2)  # be polite

    print(f"\n{len(ok)}/{len(papers)} PDFs downloaded to {OUT_DIR}")
    if skipped:
        print(f"{len(skipped)} skipped:")
        for pmcid, reason in skipped:
            print(f"  {pmcid}: {reason}")


if __name__ == "__main__":
    main()
