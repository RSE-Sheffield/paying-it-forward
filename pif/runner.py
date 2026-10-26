"""Running a model reviewer over a dataset.

`repeats` is a target, not an increment: an experiment ends up with that many
runs. Any run that didn't finish (interrupted, or papers whose request failed)
is resumed first, judging only the papers it is missing, and new runs are only
started after that. So rerunning the same command after a crash, or raising
--repeats from 5 to 10, does exactly the outstanding work.

Papers are processed one at a time, each as an isolated request, and every
judgement is written to disk as soon as it arrives.
"""
from __future__ import annotations

import os
from pathlib import Path

from . import paths, prompting, providers, registry
from .store import Experiment, ExperimentKey, Run, code_version, host, now


class RunError(Exception):
    pass


def _lock_holder(run: Run) -> str | None:
    """'pid@host' if another live process on this machine is working on the run."""
    lock = run.dir / ".lock"
    if not lock.exists():
        return None
    holder = lock.read_text().strip()
    pid, _, lock_host = holder.partition("@")
    if lock_host != host():
        return holder
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return None  # stale lock from a process that has gone
    return holder if int(pid) != os.getpid() else None


def _execute(run: Run, papers: list[dict], system: str, dataset, input, provider, log) -> None:
    lock = run.dir / ".lock"
    lock.write_text(f"{os.getpid()}@{host()}")
    done = run.judged_ids()
    todo = [p for p in papers if p["paper_id"] not in done]
    failures = []
    run.meta.update(status="running", papers_total=len(papers))
    if done:
        run.meta.setdefault("resumed_at", []).append(now())
    run.save()
    log(f"{run.dir.relative_to(paths.ROOT) if run.dir.is_relative_to(paths.ROOT) else run.dir}: "
        f"{len(todo)} paper(s) to judge" + (f" ({len(done)} already done)" if done else ""))
    try:
        for i, paper in enumerate(todo, 1):
            pid = paper["paper_id"]
            prefix = f"  [{i}/{len(todo)}] {pid}"
            try:
                result = provider.classify(system, prompting.user_message(dataset, input, paper))
            except Exception as e:  # noqa: BLE001 -- a failed request is recorded and retried on resume
                failures.append({"paper_id": pid, "error": f"{type(e).__name__}: {e}", "at": now()})
                log(f"{prefix} FAILED: {type(e).__name__}: {e}")
                continue
            run.append({"paper_id": pid, **result, "timestamp": now()})
            run.meta["papers_judged"] = len(done) + i - len(failures)
            run.save()
            flag = " (truncated)" if result.get("truncated") else ""
            log(f"{prefix} {result['category'] or result.get('error')}{flag} [{result['duration_s']}s]")
    except KeyboardInterrupt:
        run.meta.update(status="interrupted", failures=failures)
        run.save()
        raise
    finally:
        lock.unlink(missing_ok=True)
    judged = run.judged_ids()
    run.meta.update(
        status="complete" if all(p["paper_id"] in judged for p in papers) else "incomplete",
        papers_judged=len(judged), failures=failures, finished_at=now())
    run.save()
    log(f"  -> {run.meta['status']}: {len(judged)}/{len(papers)} judged"
        + (f", {len(failures)} failed (rerun to retry)" if failures else ""))


def run_experiment(key: ExperimentKey, repeats: int = 1, root: Path | None = None,
                   limit: int | None = None, log=print) -> Experiment:
    dataset = registry.dataset(key.dataset)
    criteria = registry.criteria(key.criteria)
    prompt = registry.prompt(key.prompt)
    input = registry.input(key.input)
    reviewer = registry.reviewer(key.reviewer)
    if reviewer.is_human:
        raise RunError(f"'{reviewer.id}' is a person: export a review sheet with `pif human export` instead")

    papers = dataset.papers[:limit] if limit else dataset.papers
    # Check every paper's content is present before starting, not 200 papers in.
    for paper in papers:
        prompting.user_message(dataset, input, paper)
    provider = providers.make(reviewer)
    system = prompting.system_prompt(prompt, criteria)

    exp = Experiment.open(key, root)
    log(f"== {key.name}")
    runs = exp.runs()
    for run in runs:
        if run.status == "complete":
            continue
        holder = _lock_holder(run)
        if holder:
            log(f"{run.id}: being run by {holder}, skipping")
            continue
        _execute(run, papers, system, dataset, input, provider, log)
    for _ in range(max(0, repeats - len(runs))):
        run = exp.new_run({
            "status": "running",
            "started_at": now(),
            "finished_at": None,
            "papers_total": len(papers),
            "papers_judged": 0,
            "limit": limit,
            "failures": [],
            "code": code_version(),
            "host": host(),
            "provider": provider.describe(),
            "content": {name: dataset.content_fingerprint(name)
                        for name in prompting.sources_read(dataset, input)},
        })
        _execute(run, papers, system, dataset, input, provider, log)
    if len(runs) >= repeats:
        log(f"{len(runs)} run(s) already exist (target {repeats}); nothing new started")
    return exp
