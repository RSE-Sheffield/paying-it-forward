"""Building a dataset's text content from its PDFs."""
from __future__ import annotations

import json
import shutil
import subprocess

from . import registry
from .store import atomic_write_json, now


def pdftotext_version() -> str:
    out = subprocess.run(["pdftotext", "-v"], capture_output=True, text=True)
    return (out.stderr or out.stdout).splitlines()[0].strip()


def extract_text(dataset_id: str, from_source: str = "pdf", to_source: str = "text", log=print) -> None:
    """Write <to_source>'s JSONL by running pdftotext over every paper's PDF.

    A sidecar <file>.meta.json records the tool version and when it ran, since
    a different extractor (or version) gives different text and so should be a
    different content source.
    """
    if not shutil.which("pdftotext"):
        raise SystemExit("pdftotext not found; install poppler-utils")
    ds = registry.dataset(dataset_id)
    target = ds.source(to_source)
    if target.get("format") != "jsonl":
        raise SystemExit(f"{dataset_id}: content source '{to_source}' must have format jsonl")
    out = ds.dir / target["path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    missing, lines = [], []
    for p in ds.papers:
        pdf = ds.pdf_path(from_source, p)
        if not pdf.exists():
            missing.append(p["paper_id"])
            continue
        res = subprocess.run(["pdftotext", "-enc", "UTF-8", str(pdf), "-"], capture_output=True, text=True)
        if res.returncode != 0:
            missing.append(p["paper_id"])
            log(f"{p['paper_id']}: pdftotext failed: {res.stderr.strip()[:200]}")
            continue
        lines.append(json.dumps({"paper_id": p["paper_id"], "abstract": "", "text": res.stdout},
                                ensure_ascii=False))
    out.write_text("\n".join(lines) + "\n")
    atomic_write_json(out.with_suffix(".meta.json"), {
        "built_at": now(), "from": from_source, "tool": pdftotext_version(),
        "command": "pdftotext -enc UTF-8 <pdf> -", "papers": len(lines), "missing": missing})
    log(f"wrote {out} ({len(lines)}/{len(ds.papers)} papers)" + (f"; missing: {' '.join(missing)}" if missing else ""))
