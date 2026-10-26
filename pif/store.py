"""Experiments and their runs on disk.

    experiments/<dataset>/criteria-<id>__prompt-<id>__<input>__<reviewer>/
        experiment.json     the five factor ids and their fingerprints when it was created
        run-01/
            run.json        what happened: when, which code, how far it got, any failures
            judgements.jsonl  one line per paper judged
            source.xlsx     (human runs) the sheet the judgements were imported from

The folder names are for people browsing; code never parses them. Everything
needed to group or compare runs is in experiment.json and run.json.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import paths, registry
from .registry import KINDS
from .taxonomy import VALID


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def code_version() -> dict:
    """The commit the pipeline ran from, and whether its code had local edits."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=paths.ROOT, capture_output=True, text=True).stdout.strip()
    try:
        commit = git("rev-parse", "HEAD") or None
        dirty = bool(git("status", "--porcelain", "--", "pif", "pyproject.toml"))
    except OSError:
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


class FactorChanged(Exception):
    pass


@dataclass(frozen=True)
class ExperimentKey:
    dataset: str
    criteria: str
    prompt: str
    input: str
    reviewer: str

    @property
    def slug(self) -> str:
        return f"criteria-{self.criteria}__prompt-{self.prompt}__{self.input}__{self.reviewer}"

    @property
    def name(self) -> str:
        """Unique, human-readable id: <dataset>/<slug>."""
        return f"{self.dataset}/{self.slug}"

    def factors(self) -> dict:
        return {k: getattr(self, k) for k in KINDS}

    def path(self, root: Path | None = None) -> Path:
        return (root or paths.EXPERIMENTS) / self.dataset / self.slug


class Run:
    def __init__(self, dir: Path):
        self.dir = dir
        self.meta_path = dir / "run.json"
        self.judgements_path = dir / "judgements.jsonl"
        self.meta = json.loads(self.meta_path.read_text()) if self.meta_path.exists() else {}

    @property
    def id(self) -> str:
        return self.dir.name

    @property
    def status(self) -> str:
        return self.meta.get("status", "unknown")

    def save(self) -> None:
        atomic_write_json(self.meta_path, self.meta)

    def judgements(self) -> list[dict]:
        if not self.judgements_path.exists():
            return []
        with self.judgements_path.open() as f:
            return [json.loads(line) for line in f if line.strip()]

    def judged_ids(self) -> set[str]:
        return {str(j["paper_id"]) for j in self.judgements()}

    def append(self, judgement: dict) -> None:
        """Write one judgement durably, so an interrupted run loses nothing."""
        with self.judgements_path.open("a") as f:
            f.write(json.dumps(judgement, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def write_judgements(self, judgements: list[dict]) -> None:
        tmp = self.judgements_path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(j, ensure_ascii=False) + "\n" for j in judgements))
        os.replace(tmp, self.judgements_path)


class Experiment:
    """One combination of factors. Runs inside it are repeats of each other."""

    def __init__(self, key: ExperimentKey, root: Path | None = None):
        self.key = key
        self.dir = key.path(root)
        self.meta_path = self.dir / "experiment.json"

    @staticmethod
    def current_fingerprints(key: ExperimentKey) -> dict:
        return {kind: registry.load(kind, id).fingerprint for kind, id in key.factors().items()}

    @classmethod
    def open(cls, key: ExperimentKey, root: Path | None = None, create: bool = True) -> "Experiment":
        """Open an experiment, creating it if needed, after checking that none
        of its factors has changed since it was created."""
        exp = cls(key, root)
        current = cls.current_fingerprints(key)
        if exp.meta_path.exists():
            pinned = json.loads(exp.meta_path.read_text())["fingerprints"]
            changed = [k for k in KINDS if pinned.get(k) != current[k]]
            if changed:
                lines = [f"  {k} '{getattr(key, k)}': pinned {pinned.get(k)}, now {current[k]}" for k in changed]
                raise FactorChanged(
                    f"{key.name}: these factors changed after runs were made against them:\n"
                    + "\n".join(lines)
                    + "\nA version that has been used is fixed. Revert the edit, or save it as a new "
                    "version (e.g. factors/criteria/v3/) and run against that.")
            return exp
        if not create:
            raise FileNotFoundError(f"no experiment {key.name}")
        reviewer = registry.reviewer(key.reviewer)
        exp.dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(exp.meta_path, {
            **key.factors(),
            "fingerprints": current,
            "reviewer_spec": reviewer.spec,
            "input_spec": registry.input(key.input).spec,
            "created_at": now(),
        })
        return exp

    def runs(self) -> list[Run]:
        return [Run(d) for d in sorted(self.dir.glob("run-*")) if d.is_dir()]

    def new_run(self, meta: dict) -> Run:
        existing = [int(r.id.split("-")[1]) for r in self.runs()]
        n = max(existing, default=0) + 1
        run = Run(self.dir / f"run-{n:02d}")
        run.dir.mkdir()
        run.meta = {"run": run.id, **meta}
        run.save()
        return run


# ------------------------------------------------------------------ loading
def iter_experiments(root: Path | None = None):
    """(experiment.json dict, experiment dir) for every experiment under root
    (default experiments/). Folders starting with _ or . (e.g. _scratch) are skipped."""
    root = root or paths.EXPERIMENTS
    if not root.exists():
        return
    for meta_path in sorted(root.glob("*/*/experiment.json")):
        rel = meta_path.relative_to(root)
        if any(part.startswith(("_", ".")) for part in rel.parts):
            continue
        yield json.loads(meta_path.read_text()), meta_path.parent


def load(root: Path | None = None):
    """Every experiment, run and judgement under root as three DataFrames.

    experiments: one row per experiment, with its factor ids and reviewer details.
    runs:        one row per run, with the factor columns repeated for grouping.
    judgements:  one row per (run, paper), with the factor columns repeated and
                 `valid` marking rows whose category is one of the nine codes.
    """
    import pandas as pd

    root = root or paths.EXPERIMENTS
    exp_rows, run_rows, j_rows = [], [], []
    for meta, exp_dir in iter_experiments(root):
        rev = meta.get("reviewer_spec") or {}
        key = ExperimentKey(**{k: meta[k] for k in KINDS})
        exp = {
            "experiment": key.name,
            **key.factors(),
            "reviewer_kind": rev.get("kind", "model"),
            "provider": rev.get("provider"),
            "model": rev.get("model"),
            "reviewer_name": rev.get("name") or meta["reviewer"],
            "settings": json.dumps(rev.get("settings") or {}, sort_keys=True),
            "fingerprints": meta.get("fingerprints", {}),
        }
        exp_rows.append(exp)
        factor_cols = {k: exp[k] for k in ("experiment", *KINDS, "reviewer_kind", "model")}
        for run in Experiment(key, root).runs():
            m = run.meta
            run_key = f"{key.name}/{run.id}"
            judgements = run.judgements()
            run_rows.append({
                "run_key": run_key, **factor_cols, "run": run.id,
                "status": m.get("status"), "started_at": m.get("started_at"),
                "finished_at": m.get("finished_at"),
                "papers_total": m.get("papers_total"), "papers_judged": len(judgements),
                "failures": len(m.get("failures") or []),
                "commit": (m.get("code") or {}).get("commit"),
                "meta": m,
            })
            for j in judgements:
                cat = j.get("category") or ""
                j_rows.append({"run_key": run_key, **factor_cols, "run": run.id,
                               **j, "paper_id": str(j["paper_id"]), "category": cat,
                               "valid": cat in VALID})
    experiments = pd.DataFrame(exp_rows)
    runs = pd.DataFrame(run_rows)
    judgements = pd.DataFrame(j_rows)
    return experiments, runs, judgements


def host() -> str:
    return socket.gethostname()
