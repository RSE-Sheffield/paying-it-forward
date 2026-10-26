"""The versioned factors an experiment is built from.

Each factor lives in its own folder under factors/ and is referred to by id:

    factors/datasets/<id>/dataset.yaml + papers.csv   which papers, and where their content is
    factors/criteria/<id>/criteria.md                 the definitions papers are classified against
    factors/prompts/<id>/system.md + task.md          the instructions a reviewer is given
    factors/inputs/<id>.yaml                          what of each paper the reviewer reads
    factors/reviewers/<id>.yaml                       a model with its settings, or a person

Every factor has a fingerprint over everything in it that can change a result.
Free text is left out so it can be edited: `description`/`notes` everywhere,
and a dataset's provenance and its content sources' `build`/`method` notes.
An experiment pins the fingerprints it was created with, which means a version
that has been used cannot be edited in place without that being caught: add a
new version alongside it instead.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from . import paths

KINDS = ("dataset", "criteria", "prompt", "input", "reviewer")
_EDITABLE_KEYS = {"description", "notes"}


class RegistryError(Exception):
    pass


def fingerprint(*parts: bytes) -> str:
    """sha256 of a single part, or of the parts' digests in order; 12 hex chars."""
    if len(parts) == 1:
        return hashlib.sha256(parts[0]).hexdigest()[:12]
    h = hashlib.sha256()
    for p in parts:
        h.update(hashlib.sha256(p).digest())
    return h.hexdigest()[:12]


def _canonical(spec: dict) -> bytes:
    """A spec's result-affecting content, independent of key order and comments."""
    kept = {k: v for k, v in spec.items() if k not in _EDITABLE_KEYS}
    return json.dumps(kept, sort_keys=True, ensure_ascii=False).encode()


def _read_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise RegistryError(f"{path}: expected a mapping at the top level")
    return data


def _require(path: Path, kind: str, id: str) -> Path:
    if not path.exists():
        known = ", ".join(ids(kind)) or "none"
        raise RegistryError(f"no {kind} '{id}' ({path.relative_to(paths.ROOT)} missing). Known: {known}")
    return path


# ---------------------------------------------------------------- datasets
@dataclass
class Dataset:
    """A fixed, ordered list of papers plus where to find each paper's content.

    papers.csv is committed and holds bibliographic metadata, with `paper_id`
    as its first column. Content (abstracts, full text, PDFs) is usually not
    redistributable, so it lives under content/ (gitignored) and dataset.yaml
    says how to rebuild it. Each named content source is either a JSONL file of
    {paper_id, abstract, text} records or a per-paper PDF path pattern.
    """
    id: str
    dir: Path
    spec: dict
    papers: list[dict]
    _text_cache: dict = field(default_factory=dict, repr=False)

    @property
    def description(self) -> str:
        return self.spec.get("description", "").strip()

    @property
    def sources(self) -> dict:
        return self.spec.get("content") or {}

    @property
    def fingerprint(self) -> str:
        """Over the ordered paper ids and where each content source is read from
        (its format and path, relative to the dataset folder) -- not over notes
        on how to rebuild it, and not over the content files themselves, which
        aren't committed (each run records those separately)."""
        ids_part = "\n".join(p["paper_id"] for p in self.papers).encode()
        located = {name: {k: src.get(k) for k in ("format", "path")} for name, src in self.sources.items()}
        return fingerprint(ids_part, json.dumps(located, sort_keys=True).encode())

    def source(self, name: str) -> dict:
        if name not in self.sources:
            known = ", ".join(self.sources) or "none"
            raise RegistryError(f"dataset '{self.id}' has no content source '{name}' (has: {known})")
        return self.sources[name]

    def records(self, name: str) -> dict[str, dict]:
        """A JSONL content source as {paper_id: record}."""
        if name not in self._text_cache:
            src = self.source(name)
            if src.get("format") != "jsonl":
                raise RegistryError(f"dataset '{self.id}' source '{name}' is not a JSONL text source")
            path = self.dir / src["path"]
            if not path.exists():
                raise RegistryError(
                    f"{path.relative_to(paths.ROOT)} is missing. Content isn't committed; "
                    f"rebuild it with: {src.get('build', 'see ' + str(self.dir / 'dataset.yaml'))}")
            recs = {}
            with path.open() as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        recs[str(r["paper_id"])] = r
            self._text_cache[name] = recs
        return self._text_cache[name]

    def pdf_path(self, name: str, paper: dict) -> Path:
        src = self.source(name)
        if src.get("format") != "pdf":
            raise RegistryError(f"dataset '{self.id}' source '{name}' is not a PDF source")
        return self.dir / src["path"].format(**paper)

    def content_fingerprint(self, name: str) -> str | None:
        """Fingerprint of the content actually on disk for one source, for provenance."""
        src = self.source(name)
        if src.get("format") == "jsonl":
            path = self.dir / src["path"]
            return fingerprint(path.read_bytes()) if path.exists() else None
        files = sorted(self.pdf_path(name, p) for p in self.papers)
        present = [f.read_bytes() for f in files if f.exists()]
        return fingerprint(*present, b"") if present else None


def _load_dataset(id: str) -> Dataset:
    d = paths.DATASETS / id
    spec = _read_yaml(_require(d / "dataset.yaml", "dataset", id))
    with _require(d / "papers.csv", "dataset", id).open(newline="", encoding="utf-8") as f:
        papers = list(csv.DictReader(f))
    if not papers or "paper_id" not in papers[0]:
        raise RegistryError(f"{d / 'papers.csv'}: needs a paper_id column and at least one paper")
    seen = set()
    for p in papers:
        if p["paper_id"] in seen:
            raise RegistryError(f"{d / 'papers.csv'}: duplicate paper_id {p['paper_id']}")
        seen.add(p["paper_id"])
    return Dataset(id=id, dir=d, spec=spec, papers=papers)


# ---------------------------------------------------------------- criteria
@dataclass
class Criteria:
    id: str
    dir: Path
    text: str

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.text.encode())


def _load_criteria(id: str) -> Criteria:
    path = _require(paths.CRITERIA / id / "criteria.md", "criteria", id)
    return Criteria(id=id, dir=path.parent, text=path.read_text())


# ----------------------------------------------------------------- prompts
@dataclass
class Prompt:
    """system.md is a template: `{task}` and `{criteria}` are replaced with the
    contents of task.md and the criteria version, verbatim. The file's final
    newline is not part of the prompt."""
    id: str
    dir: Path
    system_template: str
    task: str

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.system_template.encode(), self.task.encode())

    def system(self, criteria: Criteria) -> str:
        return self.system_template.replace("{task}", self.task).replace("{criteria}", criteria.text)


def _load_prompt(id: str) -> Prompt:
    d = paths.PROMPTS / id
    system = _require(d / "system.md", "prompt", id).read_text()
    task = _require(d / "task.md", "prompt", id).read_text()
    return Prompt(id=id, dir=d, system_template=system.removesuffix("\n"), task=task)


# ------------------------------------------------------------------ inputs
@dataclass
class Input:
    """What of each paper the reviewer reads.

    `body` names the dataset content source given alongside the metadata:
    `none` for metadata only, a JSONL source (e.g. `text`) to inline its text,
    or a PDF source (e.g. `pdf`) to attach the file. `template` lays
    out the user message; placeholders are paper columns plus {abstract} and
    {text}.
    """
    id: str
    path: Path
    spec: dict

    @property
    def body(self) -> str:
        return self.spec.get("body", "none")

    @property
    def template(self) -> str | None:
        return self.spec.get("template")

    @property
    def description(self) -> str:
        return self.spec.get("description", "").strip()

    @property
    def fingerprint(self) -> str:
        return fingerprint(_canonical(self.spec))


def _load_input(id: str) -> Input:
    path = _require(paths.INPUTS / f"{id}.yaml", "input", id)
    return Input(id=id, path=path, spec=_read_yaml(path))


# --------------------------------------------------------------- reviewers
@dataclass
class Reviewer:
    """A model (provider + model + settings) or a person.

    Settings are part of the reviewer: the same model at a different
    temperature or effort is a different reviewer (e.g. sonnet5 vs
    sonnet5-effort-low), so it can be compared like any other.
    """
    id: str
    path: Path
    spec: dict

    @property
    def kind(self) -> str:
        return self.spec.get("kind", "model")

    @property
    def is_human(self) -> bool:
        return self.kind == "human"

    @property
    def provider(self) -> str | None:
        return self.spec.get("provider")

    @property
    def model(self) -> str | None:
        return self.spec.get("model")

    @property
    def settings(self) -> dict:
        return self.spec.get("settings") or {}

    @property
    def name(self) -> str:
        return self.spec.get("name") or self.id

    @property
    def fingerprint(self) -> str:
        return fingerprint(_canonical(self.spec))


def _load_reviewer(id: str) -> Reviewer:
    path = _require(paths.REVIEWERS / f"{id}.yaml", "reviewer", id)
    r = Reviewer(id=id, path=path, spec=_read_yaml(path))
    if r.kind not in ("model", "human"):
        raise RegistryError(f"{path}: kind must be 'model' or 'human', not '{r.kind}'")
    if r.kind == "model" and not (r.provider and r.model):
        raise RegistryError(f"{path}: a model reviewer needs provider and model")
    return r


# ------------------------------------------------------------------ access
_LOADERS = {
    "dataset": _load_dataset,
    "criteria": _load_criteria,
    "prompt": _load_prompt,
    "input": _load_input,
    "reviewer": _load_reviewer,
}


@lru_cache(maxsize=None)
def load(kind: str, id: str):
    if kind not in _LOADERS:
        raise RegistryError(f"unknown factor kind '{kind}'")
    return _LOADERS[kind](id)


def dataset(id: str) -> Dataset:
    return load("dataset", id)


def criteria(id: str) -> Criteria:
    return load("criteria", id)


def prompt(id: str) -> Prompt:
    return load("prompt", id)


def input(id: str) -> Input:
    return load("input", id)


def reviewer(id: str) -> Reviewer:
    return load("reviewer", id)


def ids(kind: str) -> list[str]:
    """Every id registered for a factor kind."""
    if kind == "dataset":
        return sorted(p.parent.name for p in paths.DATASETS.glob("*/dataset.yaml"))
    if kind == "criteria":
        return sorted(p.parent.name for p in paths.CRITERIA.glob("*/criteria.md"))
    if kind == "prompt":
        return sorted(p.parent.name for p in paths.PROMPTS.glob("*/system.md"))
    if kind == "input":
        return sorted(p.stem for p in paths.INPUTS.glob("*.yaml"))
    if kind == "reviewer":
        return sorted(p.stem for p in paths.REVIEWERS.glob("*.yaml"))
    raise RegistryError(f"unknown factor kind '{kind}'")
