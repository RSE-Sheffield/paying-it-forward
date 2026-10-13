# /// script
# requires-python = ">=3.10"
# dependencies = ["openpyxl"]
# ///
"""Import manual (human) classification spreadsheets into Experiments/.

Each spreadsheet in csv_files/ (named <Reviewer>_Classifications.xlsx) holds
one reviewer's pass over the dataset: the paper columns (title, pmid, pmcid,
doi, field, year, journal) plus Classification and Reasoning. Each one becomes
its own run folder, laid out like the manual assessment in
Experiments/20260921_170000_edwin so the dashboard and analysis pick it up:
  - <reviewer>_manual_assessment.json -- one row per paper (reviewer,
                                          category, reasoning, pmid, ...)
  - run.json                          -- run metadata; provider is the
                                          reviewer and model is null, which
                                          marks the run as human
  - criteria.md / task.md             -- copies of the files the reviewers
                                          worked from

Rows are written in dataset order when the dataset file is available, and
every pmid is checked against it. Categories are validated against the nine
S*D* codes; the import stops on any error rather than writing a partial run.

Usage:
    uv run import_manual_assessments.py
    uv run import_manual_assessments.py --timestamp 20261012_120000 --dry-run
    uv run import_manual_assessments.py csv_files/Matt_Classifications.xlsx
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import openpyxl

REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_INPUT_DIR = REPO_ROOT / "csv_files"
DEFAULT_EXPERIMENTS_DIR = REPO_ROOT / "Experiments"
DEFAULT_DATASET = REPO_ROOT / "spikes" / "spike0" / "papers" / "papers_fulltext_30.json"
DEFAULT_CRITERIA = REPO_ROOT / "criteria.md"
DEFAULT_TASK = REPO_ROOT / "task.md"
VERSION = "0.0.0"

CATEGORIES = [
    "S1D1", "S2D1", "S3D1",
    "S1D2", "S2D2", "S3D2",
    "S1D3", "S2D3", "S3D3",
]

REQUIRED_COLUMNS = ["title", "pmid", "pmcid", "field", "year", "classification", "reasoning"]


def as_text(value) -> str:
    """Spreadsheet cell -> string; Excel stores pmid/year as floats (38110704.0)."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def reviewer_of(path: Path) -> str:
    """Matt_Classifications.xlsx -> matt."""
    return path.stem.split("_")[0].lower()


def read_sheet(path: Path) -> list[dict]:
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True).worksheets[0]
    rows = ws.iter_rows(values_only=True)
    header = [as_text(h).lower() for h in next(rows)]
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    out = []
    for row in rows:
        record = dict(zip(header, row))
        if all(v is None for v in record.values()):
            continue
        out.append(record)
    return out


def to_results(path: Path, reviewer: str, dataset_order: dict[str, int] | None) -> list[dict]:
    results, errors, seen = [], [], set()
    for i, record in enumerate(read_sheet(path), start=2):
        pmid = as_text(record["pmid"])
        category = as_text(record["classification"]).upper()
        if category not in CATEGORIES:
            errors.append(f"row {i} (pmid {pmid}): invalid category {category!r}")
        if pmid in seen:
            errors.append(f"row {i}: duplicate pmid {pmid}")
        if dataset_order is not None and pmid not in dataset_order:
            errors.append(f"row {i}: pmid {pmid} not in dataset")
        seen.add(pmid)
        results.append({
            "reviewer": reviewer,
            "category": category,
            "reasoning": as_text(record["reasoning"]),
            "pmid": pmid,
            "pmcid": as_text(record["pmcid"]),
            "field": as_text(record["field"]),
            "title": as_text(record["title"]),
            "year": as_text(record["year"]),
        })
    if dataset_order is not None:
        absent = set(dataset_order) - seen
        if absent:
            errors.append(f"{len(absent)} dataset papers not classified: {sorted(absent)}")
        results.sort(key=lambda r: dataset_order[r["pmid"]])
    if errors:
        raise ValueError(f"{path.name}:\n  " + "\n  ".join(errors))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", type=Path,
                        help="spreadsheets to import (default: every .xlsx in csv_files/)")
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"),
                        help="YYYYMMDD_HHMMSS prefix for the run folders (default: now)")
    parser.add_argument("--experiments-dir", type=Path, default=DEFAULT_EXPERIMENTS_DIR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="dataset the reviewers classified; used for ordering and pmid checks")
    parser.add_argument("--criteria", type=Path, default=DEFAULT_CRITERIA)
    parser.add_argument("--task", type=Path, default=DEFAULT_TASK)
    parser.add_argument("--force", action="store_true", help="overwrite run folders that already exist")
    parser.add_argument("--dry-run", action="store_true", help="validate and report without writing")
    args = parser.parse_args()

    stamp = datetime.strptime(args.timestamp, "%Y%m%d_%H%M%S").astimezone(timezone.utc).isoformat()
    files = args.files or sorted(DEFAULT_INPUT_DIR.glob("*.xlsx"))
    if not files:
        print(f"No spreadsheets found in {DEFAULT_INPUT_DIR}", file=sys.stderr)
        return 1

    dataset_order = None
    if args.dataset.exists():
        papers = json.loads(args.dataset.read_text())
        dataset_order = {str(p["pmid"]): i for i, p in enumerate(papers)}
    else:
        print(f"warning: {args.dataset} not found; skipping pmid checks and ordering", file=sys.stderr)

    # Validate everything before writing anything, so a bad sheet can't leave
    # some reviewers imported and others not.
    runs = []
    for path in files:
        reviewer = reviewer_of(path)
        results = to_results(path, reviewer, dataset_order)
        run_dir = args.experiments_dir / f"{args.timestamp}_{reviewer}"
        if run_dir.exists() and not args.force:
            raise FileExistsError(f"{run_dir} already exists (use --force to overwrite)")
        runs.append((path, reviewer, results, run_dir))

    for path, reviewer, results, run_dir in runs:
        print(f"{path.name} -> {run_dir} ({len(results)} papers)")
        if args.dry_run:
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.criteria, run_dir / args.criteria.name)
        shutil.copy2(args.task, run_dir / args.task.name)
        (run_dir / f"{reviewer}_manual_assessment.json").write_text(json.dumps(results, indent=4) + "\n")
        run_meta = {
            "run_id": run_dir.name,
            "name": f"{reviewer}_manual_assessment",
            "started_at": stamp,
            "finished_at": stamp,
            "provider": reviewer,
            "model": None,
            "content": "full",
            "host": None,
            "num_ctx": None,
            "timeout_s": None,
            "limit": None,
            "dataset_file": str(args.dataset.resolve()),
            "criteria_file": str(args.criteria.resolve()),
            "task_file": str(args.task.resolve()),
            "source_file": str(path.resolve()),
            "num_papers_total": len(dataset_order) if dataset_order is not None else len(results),
            "num_papers_run": len(results),
            "num_papers_classified": len(results),
            "version": VERSION,
        }
        (run_dir / "run.json").write_text(json.dumps(run_meta, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileExistsError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
