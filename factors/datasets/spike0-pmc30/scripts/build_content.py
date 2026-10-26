"""Build this dataset's content/ from spike 0's corpus files.

spike 0 fetches the papers (spikes/spike0/fetch_corpus.py, download_pdfs.py);
this converts its output into the shape every dataset uses:
  content/text.jsonl     {paper_id, abstract, text} per paper, PMC body text
  content/pdfs/<pmcid>.pdf

Usage: uv run python factors/datasets/spike0-pmc30/scripts/build_content.py
"""
import json
import shutil
from pathlib import Path

from pif.paths import ROOT

HERE = Path(__file__).resolve().parent.parent
SPIKE = ROOT / "spikes" / "spike0" / "papers"

papers = json.loads((SPIKE / "papers_fulltext_30.json").read_text())
content = HERE / "content"
(content / "pdfs").mkdir(parents=True, exist_ok=True)

with (content / "text.jsonl").open("w") as f:
    for p in papers:
        f.write(json.dumps({"paper_id": p["pmid"], "abstract": p.get("abstract") or "",
                            "text": p.get("fulltext") or ""}, ensure_ascii=False) + "\n")

missing = []
for p in papers:
    src = SPIKE / "pdfs" / f"{p['pmcid']}.pdf"
    if src.exists():
        shutil.copy2(src, content / "pdfs" / src.name)
    else:
        missing.append(p["pmcid"])

print(f"wrote {content / 'text.jsonl'} ({len(papers)} papers)")
print(f"copied {len(papers) - len(missing)} PDFs" + (f"; missing: {' '.join(missing)}" if missing else ""))
