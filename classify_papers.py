"""Classify papers into S*D* categories using either a local Ollama model
or the Anthropic (Claude) API.

Each paper is sent as an isolated request built from only two context
documents (criteria.md and task.md) plus that paper's own text -- no other
repository files are exposed to the model, and no conversation history
carries over between papers (each call starts a fresh message list).
Papers are processed strictly one at a time (sequential, not concurrent).

Provider configuration is read from a .secrets file (KEY=VALUE per line) in
the repo root, never from the command line:
  - provider "ollama" reads OLLAMA_MODEL
  - provider "claude" reads ANTHROPIC_API_KEY and ANTHROPIC_MODEL

Each run creates its own timestamped folder under Experiments/, containing:
  - results.json  -- updated after every paper, so a crash loses nothing
  - run.json       -- metadata about the run (provider, model, content mode,
                       and the filepaths of the dataset/criteria/task files
                       it read), written at the start and refreshed with a
                       completion time at the end
  - criteria.md / task.md -- copies of the exact files used for this run

By default each paper is classified using its title, journal, year,
abstract, and full text. Pass --content metadata to omit the full text and
classify using only title, journal, year, and abstract.

By default the classification pass over the dataset runs once. Pass
--repeats N to rerun the full pass N times, each into its own run folder
(useful for checking how consistent a model's classifications are).

Usage:
    python classify_papers.py --provider ollama --dataset spikes/2026-05-llm-cost/papers/papers_fulltext_30.json
    python classify_papers.py --provider claude --dataset spikes/2026-05-llm-cost/papers/papers_fulltext_30.json --limit 3
    python classify_papers.py --provider ollama --dataset ... --content metadata --repeats 5
"""
import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_CRITERIA = REPO_ROOT / "criteria.md"
DEFAULT_TASK = REPO_ROOT / "task.md"
DEFAULT_EXPERIMENTS_DIR = REPO_ROOT / "Experiments"
DEFAULT_SECRETS = REPO_ROOT / ".secrets"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

CATEGORIES = [
    "S1D1", "S2D1", "S3D1",
    "S1D2", "S2D2", "S3D2",
    "S1D3", "S2D3", "S3D3",
]

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": CATEGORIES},
        "reasoning": {"type": "string"},
    },
    "required": ["category", "reasoning"],
}

CHARS_PER_TOKEN = 4  # conservative estimate for English prose
OUTPUT_TOKEN_BUDGET = 1000
CTX_SAFETY_MARGIN = 256


def load_secrets(path: Path) -> dict:
    secrets = {}
    if not path.exists():
        return secrets
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        secrets[key.strip()] = value.strip().strip('"').strip("'")
    return secrets


def build_system_prompt(criteria_md: str, task_md: str) -> str:
    return (
        "You are classifying a single academic paper according to the task "
        "and taxonomy defined below. Use ONLY the paper text given by the "
        "user in this message; do not assume access to anything else.\n\n"
        "=== TASK ===\n"
        f"{task_md}\n\n"
        "=== CRITERIA (criteria.md) ===\n"
        f"{criteria_md}\n\n"
        "Respond with a single JSON object matching the required schema: "
        "one of the nine category codes and a short reasoning."
    )


def build_user_prompt(paper: dict, max_chars: int | None, include_fulltext: bool = True) -> tuple[str, bool]:
    title = paper.get("title") or ""
    journal = paper.get("journal") or ""
    year = paper.get("year") or ""
    abstract = paper.get("abstract") or ""

    body = (
        f"TITLE: {title}\n\n"
        f"JOURNAL: {journal}  YEAR: {year}\n\n"
        f"ABSTRACT:\n{abstract}\n"
    )

    truncated = False
    if include_fulltext:
        fulltext = paper.get("fulltext") or ""
        if max_chars is not None and len(fulltext) > max_chars:
            fulltext = fulltext[:max_chars] + "\n[...truncated...]"
            truncated = True
        body += f"\nFULL TEXT:\n{fulltext}\n"

    return body, truncated


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def fit_to_ctx(paper: dict, num_ctx: int, system_prompt: str, include_fulltext: bool) -> tuple[str, bool]:
    """Build the user prompt, truncating full text if needed so the whole
    request fits inside num_ctx tokens; returns (user_prompt, truncated)."""
    user_prompt, truncated = build_user_prompt(paper, max_chars=None, include_fulltext=include_fulltext)

    total_needed = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
    total_needed += OUTPUT_TOKEN_BUDGET + CTX_SAFETY_MARGIN
    if total_needed <= num_ctx or not include_fulltext:
        return user_prompt, truncated

    # Doesn't fit even at num_ctx: truncate the full text to the remaining budget.
    budget_tokens = num_ctx - estimate_tokens(system_prompt) - OUTPUT_TOKEN_BUDGET - CTX_SAFETY_MARGIN
    budget_chars = max(0, budget_tokens * CHARS_PER_TOKEN)
    user_prompt, truncated = build_user_prompt(paper, max_chars=budget_chars, include_fulltext=include_fulltext)
    return user_prompt, True


def classify_paper_ollama(
    paper: dict,
    system_prompt: str,
    model: str,
    host: str,
    num_ctx: int,
    timeout: int,
    include_fulltext: bool,
) -> dict:
    user_prompt, truncated = fit_to_ctx(paper, num_ctx, system_prompt, include_fulltext)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "format": RESPONSE_SCHEMA,
        "stream": False,
        "options": {"num_ctx": num_ctx},
    }

    t0 = time.time()
    resp = requests.post(f"{host}/api/chat", json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    elapsed = time.time() - t0

    content = data.get("message", {}).get("content", "")
    try:
        parsed = json.loads(content)
        category = parsed.get("category", "PARSE_ERROR")
        reasoning = parsed.get("reasoning", "")
    except (json.JSONDecodeError, AttributeError):
        category = "PARSE_ERROR"
        reasoning = ""

    return {
        "provider": "ollama",
        "model": model,
        "category": category,
        "reasoning": reasoning,
        "num_ctx": num_ctx,
        "truncated": truncated,
        "prompt_tokens": data.get("prompt_eval_count"),
        "output_tokens": data.get("eval_count"),
        "duration_s": round(elapsed, 2),
        "raw_response": content if category == "PARSE_ERROR" else None,
    }


def classify_paper_claude(
    paper: dict,
    system_prompt: str,
    model: str,
    api_key: str,
    timeout: int,
    include_fulltext: bool,
) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    user_prompt, truncated = build_user_prompt(paper, max_chars=None, include_fulltext=include_fulltext)

    tool = {
        "name": "classify_paper",
        "description": "Record the S*D* classification for this paper.",
        "input_schema": RESPONSE_SCHEMA,
    }

    t0 = time.time()
    resp = client.messages.create(
        model=model,
        max_tokens=600,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        tools=[tool],
        tool_choice={"type": "tool", "name": "classify_paper"},
        timeout=timeout,
    )
    elapsed = time.time() - t0

    category = "PARSE_ERROR"
    reasoning = ""
    raw_response = None
    tool_use = next((b for b in resp.content if b.type == "tool_use"), None)
    if tool_use is not None:
        args = tool_use.input
        candidate = args.get("category")
        if candidate in CATEGORIES:
            category = candidate
            reasoning = args.get("reasoning", "")
        else:
            raw_response = json.dumps(args)

    return {
        "provider": "claude",
        "model": model,
        "category": category,
        "reasoning": reasoning,
        "num_ctx": None,
        "truncated": truncated,
        "prompt_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "duration_s": round(elapsed, 2),
        "raw_response": raw_response,
    }


def run_once(args, system_prompt: str, papers: list, todo: list, run_suffix: str = "") -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + run_suffix
    run_dir = args.experiments_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.json"
    run_meta_path = run_dir / "run.json"

    shutil.copy2(args.criteria, run_dir / args.criteria.name)
    shutil.copy2(args.task, run_dir / args.task.name)

    run_meta = {
        "run_id": run_id,
        "name": args.name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "provider": args.provider,
        "model": args.model,
        "content": args.content,
        "host": args.host if args.provider == "ollama" else None,
        "num_ctx": args.num_ctx if args.provider == "ollama" else None,
        "timeout_s": args.timeout,
        "limit": args.limit,
        "dataset_file": str(args.dataset.resolve()),
        "criteria_file": str(args.criteria.resolve()),
        "task_file": str(args.task.resolve()),
        "num_papers_total": len(papers),
        "num_papers_run": len(todo),
        "num_papers_classified": 0,
    }
    atomic_write_json(run_meta_path, run_meta)

    print(f"Run folder: {run_dir}")
    print(f"{len(todo)} papers to classify (provider={args.provider}, model={args.model}, content={args.content})")

    results = []
    for i, paper in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] pmid={paper.get('pmid')} field={paper.get('field')} ...", end=" ", flush=True)
        try:
            result = classify_paper(paper, system_prompt, args)
        except Exception as e:
            print(f"REQUEST FAILED: {e}")
            continue

        results.append(result)
        atomic_write_json(results_path, results)

        run_meta["num_papers_classified"] = len(results)
        atomic_write_json(run_meta_path, run_meta)

        flag = " (truncated)" if result["truncated"] else ""
        print(f"{result['category']}{flag}  [{result['duration_s']}s]")

    run_meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(run_meta_path, run_meta)

    print(f"\nWrote {len(results)} classifications to {results_path}")
    counts = {}
    for r in results:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    for cat in CATEGORIES + ["PARSE_ERROR"]:
        if cat in counts:
            print(f"  {cat}: {counts[cat]}")


def classify_paper(paper: dict, system_prompt: str, args) -> dict:
    include_fulltext = args.content == "full"
    if args.provider == "ollama":
        result = classify_paper_ollama(
            paper, system_prompt, args.model, args.host, args.num_ctx, args.timeout, include_fulltext
        )
    else:
        result = classify_paper_claude(
            paper, system_prompt, args.model, args.api_key, args.timeout, include_fulltext
        )

    result.update({
        "pmid": paper.get("pmid"),
        "pmcid": paper.get("pmcid"),
        "field": paper.get("field"),
        "title": paper.get("title"),
        "year": paper.get("year"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    return result


def atomic_write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def resolve_provider_config(args):
    secrets = load_secrets(args.secrets)

    if args.provider == "ollama":
        model = secrets.get("OLLAMA_MODEL")
        if not model:
            raise SystemExit(f"OLLAMA_MODEL not set in {args.secrets}")
        args.model = model
        args.api_key = None
    else:
        api_key = secrets.get("ANTHROPIC_API_KEY")
        model = secrets.get("ANTHROPIC_MODEL")
        if not api_key or not model:
            raise SystemExit(f"ANTHROPIC_API_KEY and ANTHROPIC_MODEL must both be set in {args.secrets}")
        args.model = model
        args.api_key = api_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["ollama", "claude"], default="ollama")
    parser.add_argument("--dataset", type=Path, required=True,
                         help="Path to a .json file of papers to classify.")
    parser.add_argument("--criteria", type=Path, default=DEFAULT_CRITERIA)
    parser.add_argument("--task", type=Path, default=DEFAULT_TASK)
    parser.add_argument("--secrets", type=Path, default=DEFAULT_SECRETS)
    parser.add_argument("--experiments-dir", type=Path, default=DEFAULT_EXPERIMENTS_DIR,
                         help="Base directory under which a new timestamped run folder is created.")
    parser.add_argument("--host", default=DEFAULT_OLLAMA_HOST, help="Ollama server URL (ollama provider only).")
    parser.add_argument("--num-ctx", type=int, default=262144,
                         help="Context window size in tokens for every request (ollama provider only).")
    parser.add_argument("--timeout", type=int, default=600, help="Per-request timeout in seconds.")
    parser.add_argument("--limit", type=int, default=None, help="Only classify the first N papers.")
    parser.add_argument("--name", default=None, help="Optional label for this run, stored in run.json.")
    parser.add_argument("--content", choices=["full", "metadata"], default="full",
                         help="'full' (default) sends title, journal, year, abstract, and full text. "
                              "'metadata' sends only title, journal, year, and abstract.")
    parser.add_argument("--repeats", type=int, default=1,
                         help="Number of times to repeat the full classification pass over the dataset "
                              "(default 1). Each repeat gets its own timestamped run folder.")
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")

    if args.dataset.suffix != ".json":
        raise SystemExit(f"--dataset must point at a .json file, got: {args.dataset}")

    resolve_provider_config(args)

    criteria_md = args.criteria.read_text()
    task_md = args.task.read_text()
    system_prompt = build_system_prompt(criteria_md, task_md)

    papers = json.loads(args.dataset.read_text())
    todo = papers[: args.limit] if args.limit is not None else papers

    for rep in range(1, args.repeats + 1):
        if args.repeats > 1:
            print(f"\n=== Repeat {rep}/{args.repeats} ===")
        run_suffix = f"_rep{rep}" if args.repeats > 1 else ""
        run_once(args, system_prompt, papers, todo, run_suffix=run_suffix)


if __name__ == "__main__":
    main()
