#!/usr/bin/env python3
"""Bundle the Experiments/ runs into a single self-contained dashboard page.

Reads every run directory under Experiments/, normalises the different result
shapes (LLM runs vs the manual human assessment), and injects the whole lot as
JSON into dashboard/template.html to produce dashboard/index.html.

Usage:
    python dashboard/build_dashboard.py [--experiments DIR] [--out FILE]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

CATEGORIES = [f"S{s}D{d}" for s in (1, 2, 3) for d in (1, 2, 3)]
VALID = set(CATEGORIES)

TAXONOMY = {
    "S1": ("Software Generated", "New or substantially extended software that plays a substantial role in achieving the research goals, or is itself a research goal, and is shared or published as part of the research."),
    "S2": ("Software Used", "Existing software that plays a substantial role in achieving the research goals, but is not substantially developed or extended by the research team as part of the research."),
    "S3": ("No Software", "Software does not play a substantial role in achieving the research goals."),
    "D1": ("Data Generated", "New or substantially extended data that plays a substantial role in achieving the research goals, or is itself a research goal, and is shared or published as part of the research."),
    "D2": ("Data Used", "Existing data that plays a substantial role in achieving the research goals, but is not substantially generated or extended by the research team as part of the research."),
    "D3": ("No Data", "Data does not play a substantial role in achieving the research goals."),
}

# A *configuration* is one model reading one kind of input — the thing that
# gets repeated. Runs sharing a configuration are repeats of each other, so
# the dashboard treats the configuration, not the run, as the unit of
# comparison. Derived from run.json rather than hardcoded, so a new model or
# input mode shows up on its own.
CONTENT_LABEL = {"full": "full text", "metadata": "abstract only"}
CONTENT_BLURB = {
    "full": "reads the whole paper",
    "metadata": "reads title + abstract metadata only",
}


def pretty_model(model: str) -> str:
    """gemma4:12b -> Gemma4 12B, claude-sonnet-5 -> Claude Sonnet 5."""
    words = [w for w in re.split(r"[-:_\s]+", model) if w]
    out = []
    for w in words:
        if re.fullmatch(r"\d+[bB]", w):
            out.append(w.upper())
        elif w.isdigit():
            out.append(w)
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out)


def config_of(run: dict) -> str:
    """Stable id for the configuration a run belongs to."""
    if not run.get("model"):
        return f"human-{run.get('provider') or 'reviewer'}"
    model = re.sub(r"[^a-z0-9]+", "-", run["model"].lower()).strip("-")
    return f"{model}-{run.get('content') or 'full'}"


def config_meta(run: dict) -> dict:
    """Display metadata for a configuration, from any one of its runs."""
    if not run.get("model"):
        who = (run.get("provider") or "reviewer").title()
        return {
            "label": f"Human ({who})",
            "model": None,
            "content": run.get("content"),
            "blurb": "manual assessment by a reviewer",
            "human": True,
        }
    content = run.get("content") or "full"
    model = pretty_model(run["model"])
    return {
        "label": f"{model} · {CONTENT_LABEL.get(content, content)}",
        "model": model,
        "content": content,
        "blurb": f"{model} {CONTENT_BLURB.get(content, content)}",
        "human": False,
    }


def config_sort_key(cfg: dict) -> tuple:
    """Models first (grouped, full text before abstract), human last."""
    return (1 if cfg["human"] else 0, cfg["model"] or "", 0 if cfg["content"] == "full" else 1)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def load_run(run_dir: Path, warnings: list[str]) -> dict | None:
    meta_file = run_dir / "run.json"
    if not meta_file.exists():
        warnings.append(f"{run_dir.name}: no run.json, skipped")
        return None
    meta = json.loads(meta_file.read_text())

    # LLM runs write results.json; the manual pass has its own filename.
    candidates = [p for p in sorted(run_dir.glob("*.json")) if p.name != "run.json"]
    if not candidates:
        warnings.append(f"{run_dir.name}: no results file, skipped")
        return None
    results_file = next((p for p in candidates if p.name == "results.json"), candidates[0])
    rows = json.loads(results_file.read_text())

    run_id = meta.get("run_id") or run_dir.name
    judgements = []
    for i, row in enumerate(rows):
        category = (row.get("category") or "").strip()
        reasoning = (row.get("reasoning") or "").strip()
        note = None
        if category not in VALID:
            note = "no valid category recorded in the source file"
            warnings.append(f"{run_id} row {i} ({row.get('pmid')}): {note}")
            category = ""
        j = {
            "run": run_id,
            "pmid": str(row.get("pmid") or ""),
            "category": category,
            "reasoning": reasoning,
        }
        if note:
            j["note"] = note
        # Per-judgement telemetry only the LLM runs carry.
        for key in ("prompt_tokens", "output_tokens", "duration_s", "truncated", "timestamp", "num_ctx"):
            if row.get(key) is not None:
                j[key] = row[key]
        if row.get("reviewer"):
            j["reviewer"] = row["reviewer"]
        judgements.append(j)

    return {
        "run": {
            "run_id": run_id,
            "dir": run_dir.name,
            "name": meta.get("name") or run_id,
            "config": config_of(meta),
            "meta": meta,
            "results_file": results_file.name,
            "criteria_sha": sha(run_dir / "criteria.md") if (run_dir / "criteria.md").exists() else None,
            "task_sha": sha(run_dir / "task.md") if (run_dir / "task.md").exists() else None,
            "fields": sorted({k for row in rows for k in row}),
        },
        "judgements": judgements,
        "rows": rows,
    }


def build(experiments: Path, repo_root: Path) -> dict:
    warnings: list[str] = []
    runs, judgements = [], []
    papers: dict[str, dict] = {}

    configs: dict[str, dict] = {}
    for run_dir in sorted(p for p in experiments.iterdir() if p.is_dir()):
        loaded = load_run(run_dir, warnings)
        if not loaded:
            continue
        runs.append(loaded["run"])
        configs.setdefault(loaded["run"]["config"], config_meta(loaded["run"]["meta"]))
        judgements.extend(loaded["judgements"])
        for row in loaded["rows"]:
            pmid = str(row.get("pmid") or "")
            if not pmid:
                continue
            papers.setdefault(pmid, {
                "pmid": pmid,
                "pmcid": row.get("pmcid"),
                "field": row.get("field"),
                "title": row.get("title"),
                "year": row.get("year"),
            })

    # Configuration order drives the colour slots on the page, so pin it here
    # rather than letting directory order decide.
    config_order = [cid for cid, _ in sorted(configs.items(), key=lambda kv: config_sort_key(kv[1]))]
    rank = {cid: i for i, cid in enumerate(config_order)}
    runs.sort(key=lambda r: (rank.get(r["config"], 99), r["run_id"]))
    for i, cid in enumerate(config_order):
        cfg = configs[cid]
        cfg["id"] = cid
        cfg["slot"] = i + 1  # categorical colour slot, 1-based
        cfg["runs"] = [r["run_id"] for r in runs if r["config"] == cid]

    def read(name: str) -> str:
        p = repo_root / name
        return p.read_text() if p.exists() else ""

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "categories": CATEGORIES,
        "taxonomy": {k: {"label": v[0], "desc": v[1]} for k, v in TAXONOMY.items()},
        "configs": [configs[cid] for cid in config_order],
        "criteria_md": read("criteria.md"),
        "task_md": read("task.md"),
        "papers": sorted(papers.values(), key=lambda p: p["pmid"]),
        "runs": runs,
        "judgements": judgements,
        "warnings": warnings,
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiments", type=Path, default=repo_root / "Experiments")
    ap.add_argument("--template", type=Path, default=here / "template.html")
    ap.add_argument("--out", type=Path, default=here / "index.html")
    args = ap.parse_args()

    data = build(args.experiments, repo_root)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    # The payload lands inside a <script> block, so no sequence in it may end it.
    payload = payload.replace("</", "<\\/")

    template = args.template.read_text()
    if "__DASHBOARD_DATA__" not in template:
        raise SystemExit(f"{args.template}: missing __DASHBOARD_DATA__ placeholder")
    args.out.write_text(template.replace("__DASHBOARD_DATA__", payload))

    print(f"{args.out}  ({args.out.stat().st_size / 1024:.0f} KB)")
    print(f"  {len(data['runs'])} runs, {len(data['papers'])} papers, {len(data['judgements'])} judgements")
    for cfg in data["configs"]:
        print(f"  slot {cfg['slot']}  {cfg['label']}  ({len(cfg['runs'])} run{'s' if len(cfg['runs']) != 1 else ''})")
    for w in data["warnings"]:
        print(f"  warning: {w}")


if __name__ == "__main__":
    main()
