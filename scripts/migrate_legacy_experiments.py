"""One-off: convert the flat Experiments/<timestamp>/ runs (classify_papers.py
and import_manual_assessments.py, Sept-Oct 2026) into experiments/.

Each legacy run's factors are recovered from what it recorded:
  dataset   every legacy run read spikes/spike0 papers_fulltext_30.json -> spike0-pmc30
            (recorded under the spike's old name, 2026-05-llm-cost, before 2 Oct)
  criteria  the run's copy of criteria.md, matched by fingerprint -> v1 or v2
  prompt    the run's copy of task.md matched against prompts/v1; the system and
            user templates were verified identical to classify_papers.py, which
            was committed on 22 Sept -- runs before that predate the commit, so
            for them v1 is the best record rather than a verified one
  input     content "full" -> fulltext, content "metadata" -> abstract-only
            (the human passes recorded "full": the reviewer read the whole paper)
  reviewer  model gemma4:12b -> gemma4-12b, claude-sonnet-5 -> sonnet5,
            a human pass's provider (its reviewer's name) -> human-<name>

The original run.json is kept under "legacy" in the new one, and runs within
an experiment are numbered in the order they started.

Usage: uv run python scripts/migrate_legacy_experiments.py [--legacy Experiments] [--dry-run]
"""
import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

from pif import paths, registry
from pif.store import Experiment, ExperimentKey

MODEL_REVIEWERS = {"gemma4:12b": "gemma4-12b", "claude-sonnet-5": "sonnet5"}
MODEL_FIELDS = ("category", "reasoning", "truncated", "prompt_tokens", "output_tokens", "duration_s", "timestamp")


def sha12(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def classify(run_dir: Path) -> tuple[ExperimentKey, dict, list[dict], Path | None]:
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["dataset_file"].endswith("papers_fulltext_30.json"), meta["dataset_file"]

    crit_fp = sha12(run_dir / "criteria.md")
    crit = next(c for c in registry.ids("criteria") if registry.criteria(c).fingerprint == crit_fp)
    task = (run_dir / "task.md").read_text()
    assert task == registry.prompt("v1").task, f"{run_dir.name}: task.md differs from prompts/v1"

    human = meta.get("model") is None
    inp = {"full": "fulltext", "metadata": "abstract-only"}[meta.get("content") or "full"]
    if human:
        reviewer = f"human-{meta['provider']}"
    else:
        reviewer = MODEL_REVIEWERS[meta["model"]]
        settings = registry.reviewer(reviewer).settings
        if meta["provider"] == "ollama":
            assert meta["num_ctx"] == settings["num_ctx"], run_dir.name
        assert meta["timeout_s"] == settings["timeout_s"], run_dir.name
    assert meta.get("limit") is None, run_dir.name

    results = run_dir / "results.json"
    if not results.exists():
        results = next(p for p in sorted(run_dir.glob("*.json")) if p.name != "run.json")
    rows = json.loads(results.read_text())

    judgements = []
    for row in rows:
        j = {"paper_id": str(row["pmid"])}
        if human:
            j.update(category=row.get("category"), reasoning=row.get("reasoning") or "")
        else:
            j.update({k: row[k] for k in MODEL_FIELDS if k in row})
            if row.get("raw_response") is not None:
                j["raw_response"] = row["raw_response"]
        judgements.append(j)

    # The spreadsheet a human pass was imported from, where we still have it.
    source = None
    if meta.get("source_file"):
        source = paths.ROOT / "csv_files" / Path(meta["source_file"]).name
    elif human:
        source = next(iter(run_dir.glob("*_papers.csv")), None)

    key = ExperimentKey(dataset="spike0-pmc30", criteria=crit, prompt="v1", input=inp, reviewer=reviewer)
    return key, meta, judgements, source


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--legacy", type=Path, default=paths.ROOT / "Experiments")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    groups = defaultdict(list)
    for run_dir in sorted(p for p in args.legacy.iterdir() if (p / "run.json").exists()):
        key, meta, judgements, source = classify(run_dir)
        groups[key].append((meta, judgements, source, run_dir))

    for key, runs in sorted(groups.items(), key=lambda kv: kv[0].name):
        runs.sort(key=lambda r: r[0]["started_at"])
        print(f"{key.name}: {len(runs)} run(s) <- {', '.join(r[3].name for r in runs)}")
        if args.dry_run:
            continue
        exp = Experiment.open(key)
        if exp.runs():
            raise SystemExit(f"{key.name} already has runs; refusing to migrate into it twice")
        for meta, judgements, source, run_dir in runs:
            complete = meta["num_papers_classified"] == meta["num_papers_total"] == len(judgements)
            run = exp.new_run({
                "status": "complete" if complete else "incomplete",
                "started_at": meta["started_at"],
                "finished_at": meta["finished_at"],
                "papers_total": meta["num_papers_total"],
                "papers_judged": len(judgements),
                "failures": [],
                "code": {"commit": None, "dirty": None},
                "host": None,
                "migrated_from": f"Experiments/{run_dir.name}",
                "legacy": meta,
            })
            run.write_judgements(judgements)
            if source is not None and source.exists():
                dest = run.dir / f"source{source.suffix}"
                shutil.copy2(source, dest)
                run.meta["source_file"] = dest.name
                run.save()


if __name__ == "__main__":
    main()
