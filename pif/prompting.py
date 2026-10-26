"""Turning one paper into what a model reads: a system prompt from the prompt
and criteria versions, and a user message laid out by the input."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .registry import Criteria, Dataset, Input, Prompt, RegistryError

TRUNCATION_MARK = "\n[...truncated...]"


def system_prompt(prompt: Prompt, criteria: Criteria) -> str:
    return prompt.system(criteria)


class _Blank(dict):
    """Template fields a paper doesn't have render as empty strings."""
    def __missing__(self, key):
        return ""


@dataclass
class UserMessage:
    template: str
    fields: dict
    text: str | None   # inline body text, if the input has one
    pdf: Path | None   # attached PDF, if the input has one

    def render(self, max_chars: int | None = None) -> tuple[str, bool]:
        """The message, with the body text cut to max_chars if given;
        returns (message, truncated)."""
        text = self.text or ""
        truncated = False
        if max_chars is not None and len(text) > max_chars:
            text = text[:max_chars] + TRUNCATION_MARK
            truncated = True
        return self.template.format_map(_Blank({**self.fields, "text": text})), truncated


def abstract_of(dataset: Dataset, paper: dict) -> str:
    """papers.csv's abstract column if it has one, else the `text` source's."""
    if paper.get("abstract"):
        return paper["abstract"]
    if "text" in dataset.sources:
        rec = dataset.records("text").get(paper["paper_id"])
        return (rec or {}).get("abstract") or ""
    return ""


def sources_read(dataset: Dataset, input: Input) -> list[str]:
    """Content sources a run with this input reads, for provenance."""
    used = []
    if "text" in dataset.sources and "{abstract}" in (input.template or ""):
        used.append("text")
    if input.body != "none" and input.body not in used:
        used.append(input.body)
    return used


def user_message(dataset: Dataset, input: Input, paper: dict) -> UserMessage:
    if not input.template:
        raise RegistryError(f"input '{input.id}' has no template, so a model can't be given it")
    fields = {**paper, "abstract": abstract_of(dataset, paper)}
    text = pdf = None
    if input.body != "none":
        fmt = dataset.source(input.body).get("format")
        if fmt == "jsonl":
            rec = dataset.records(input.body).get(paper["paper_id"])
            if rec is None:
                raise RegistryError(f"{dataset.id}: no '{input.body}' content for paper {paper['paper_id']}")
            text = rec.get("text") or ""
        elif fmt == "pdf":
            pdf = dataset.pdf_path(input.body, paper)
            if not pdf.exists():
                raise RegistryError(f"{dataset.id}: missing PDF {pdf}")
        else:
            raise RegistryError(f"{dataset.id}: content source '{input.body}' has unknown format '{fmt}'")
    return UserMessage(template=input.template, fields=fields, text=text, pdf=pdf)
