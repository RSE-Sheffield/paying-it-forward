"""Command line: `uv run pif <command>`. Run `uv run pif <command> --help` for details.

  pif run       run model reviewers over a dataset
  pif human     export a review sheet for a person, or import a completed one
  pif list      experiments and how many runs each has
  pif check     verify every experiment against the registries
  pif dataset   build dataset content (extract text from PDFs)
  pif dashboard open the results dashboard
"""
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import yaml

from . import paths, registry
from .registry import KINDS, RegistryError
from .store import ExperimentKey, FactorChanged


def expand(factors: dict) -> list[ExperimentKey]:
    """Every combination of the given factor values."""
    missing = [k for k in KINDS if not factors.get(k)]
    if missing:
        raise SystemExit(f"missing factor(s): {', '.join(missing)}")
    return [ExperimentKey(*combo) for combo in itertools.product(*(factors[k] for k in KINDS))]


DEFAULT_INPUT = "fulltext"


def _add_factor_args(p, multiple: bool, required: bool = False):
    """--dataset --criteria --prompt --input --reviewer. --input is never
    required: reviewers read the full text unless told otherwise."""
    for k in KINDS:
        is_input = k == "input"
        default = ([DEFAULT_INPUT] if multiple else DEFAULT_INPUT) if is_input else None
        note = f" (default {DEFAULT_INPUT})" if is_input else ""
        if multiple:
            p.add_argument(f"--{k}", nargs="+", metavar="ID", default=default,
                           help=f"{k} id(s); several make a grid{note}")
        else:
            p.add_argument(f"--{k}", metavar="ID", default=default, required=required and not is_input,
                           help=f"{k} id{note}" if note else None)


# ---------------------------------------------------------------- commands
def cmd_run(args):
    from .providers import input_problem
    from .runner import RunError, run_experiment
    from .store import Experiment

    keys = expand({k: getattr(args, k) for k in KINDS})
    repeats = args.repeats
    root = args.experiments_dir or (paths.SCRATCH if args.limit else paths.EXPERIMENTS)
    skipped = set()
    try:
        for k in keys:
            reviewer = registry.reviewer(k.reviewer)
            if reviewer.is_human:
                why = (f"a person's pass is imported, not run "
                       f"(pif human export ... --reviewer {k.reviewer})")
            else:
                why = input_problem(reviewer, registry.input(k.input))
            if why:
                print(f"skipping {k.name}: {why}")
                skipped.add(k)
    except RegistryError as e:
        raise SystemExit(f"error: {e}")
    keys = [k for k in keys if k not in skipped]
    if args.dry_run:
        for key in keys:
            runs = Experiment(key, root).runs()
            unfinished = sum(r.status != "complete" for r in runs)
            new = max(0, repeats - len(runs))
            todo = ", ".join(x for x in [f"resume {unfinished}" if unfinished else "",
                                         f"start {new}" if new else ""] if x) or "nothing to do"
            print(f"{key.name}: {len(runs)} run(s) -> {todo}")
        return
    for key in keys:
        try:
            run_experiment(key, repeats=repeats, root=root, limit=args.limit)
        except (RegistryError, FactorChanged, RunError) as e:
            raise SystemExit(f"error: {e}")


def cmd_human_export(args):
    from .human import export_sheet

    key = ExperimentKey(**{k: getattr(args, k) for k in KINDS})
    out = args.output or Path(f"review-{key.dataset}-criteria-{key.criteria}-{key.reviewer}.xlsx")
    export_sheet(key, out)
    print(f"wrote {out}: send it to {registry.reviewer(key.reviewer).name}, then `pif human import {out}`")


def cmd_human_import(args):
    from .human import SheetError, import_sheet

    # --input always has a value (its default), so only the other four decide
    # whether the experiment is named on the command line or read from the sheet.
    given = {k: getattr(args, k) for k in KINDS if k != "input"}
    key = ExperimentKey(**given, input=args.input) if all(given.values()) else None
    if any(given.values()) and not key:
        raise SystemExit("pass all of --dataset --criteria --prompt --reviewer (and optionally --input), or none")
    try:
        key, judgements, run_id = import_sheet(args.sheet, key, allow_incomplete=args.allow_incomplete,
                                               notes=args.notes, reviewed_on=args.reviewed_on,
                                               dry_run=args.dry_run)
    except (SheetError, RegistryError, FactorChanged) as e:
        raise SystemExit(f"error: {e}")
    where = f"{key.name}/{run_id}" if run_id else f"{key.name} (dry run, nothing written)"
    print(f"{len(judgements)} judgements -> {where}")


def cmd_list(args):
    from .store import load

    exps, runs, _ = load(args.experiments_dir or paths.EXPERIMENTS)
    if exps.empty:
        print("no experiments yet")
        return
    if args.dataset:
        exps = exps[exps.dataset == args.dataset]
    for ds, group in exps.groupby("dataset"):
        print(f"{ds}")
        for _, e in group.iterrows():
            r = runs[runs.experiment == e.experiment] if not runs.empty else runs
            complete = int((r.status == "complete").sum()) if len(r) else 0
            other = len(r) - complete
            note = f" (+{other} unfinished)" if other else ""
            print(f"  {e.experiment.split('/', 1)[1]:<62} {complete} run(s){note}")


def cmd_check(args):
    """Every experiment's pinned fingerprints against the registries, and each
    run's judgements against its dataset."""
    import json

    from .store import Experiment, iter_experiments
    from .taxonomy import VALID

    problems = 0
    for meta, exp_dir in iter_experiments(args.experiments_dir or paths.EXPERIMENTS):
        key = ExperimentKey(**{k: meta[k] for k in KINDS})
        try:
            current = Experiment.current_fingerprints(key)
        except RegistryError as e:
            print(f"{key.name}: {e}")
            problems += 1
            continue
        for k in KINDS:
            if meta["fingerprints"][k] != current[k]:
                print(f"{key.name}: {k} '{getattr(key, k)}' has changed since its runs were made")
                problems += 1
        ids = {p["paper_id"] for p in registry.dataset(key.dataset).papers}
        for run in Experiment(key, exp_dir.parent.parent).runs():
            for n, j in enumerate(run.judgements(), 1):
                if str(j["paper_id"]) not in ids:
                    print(f"{key.name}/{run.id} line {n}: paper {j['paper_id']} not in the dataset")
                    problems += 1
                elif j.get("category") and j["category"] not in VALID:
                    print(f"{key.name}/{run.id} line {n}: invalid category {j['category']!r}")
                    problems += 1
            if not run.meta:
                print(f"{key.name}/{run.id}: no run.json")
                problems += 1
    for kind in KINDS:
        for id in registry.ids(kind):
            try:
                registry.load(kind, id)
            except (RegistryError, json.JSONDecodeError, yaml.YAMLError) as e:
                print(f"{kind} {id}: {e}")
                problems += 1
    print("ok" if not problems else f"{problems} problem(s)")
    sys.exit(1 if problems else 0)


def cmd_dataset_extract(args):
    from .content import extract_text

    extract_text(args.dataset, from_source=args.from_source, to_source=args.to_source)


def cmd_dashboard(args):
    app = paths.ROOT / "dashboard" / "app.py"
    # Blue primary (the charts' hue) instead of Streamlit's default red, set per mode
    # so the app still follows the system's light/dark setting.
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app),
                              "--theme.light.primaryColor", "#2a78d6",
                              "--theme.dark.primaryColor", "#3987e5", *args.streamlit_args]))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pif", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run model reviewers", description=(
        "Run every combination of the given factors. --repeats N makes sure each experiment has N runs, resuming unfinished "
        "ones first. --limit N is a trial: it judges the first N papers into experiments/_scratch/."))
    _add_factor_args(p, multiple=True)
    p.add_argument("--repeats", type=int, default=1, help="runs each experiment should have (default 1)")
    p.add_argument("--limit", type=int, help="trial run on the first N papers, kept in experiments/_scratch/")
    p.add_argument("--dry-run", action="store_true", help="list the experiments and what would be run, then stop")
    p.add_argument("--experiments-dir", type=Path, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_run)

    h = sub.add_parser("human", help="review sheets for people").add_subparsers(dest="action", required=True)
    e = h.add_parser("export", help="write a review sheet for one reviewer")
    _add_factor_args(e, multiple=False, required=True)
    e.add_argument("-o", "--output", type=Path)
    e.set_defaults(func=cmd_human_export)
    i = h.add_parser("import", help="store a completed review sheet as a run")
    i.add_argument("sheet", type=Path)
    _add_factor_args(i, multiple=False)
    i.add_argument("--allow-incomplete", action="store_true", help="import even if some papers have no category")
    i.add_argument("--reviewed-on", help="when the review was done (free text, e.g. 2026-10-12)")
    i.add_argument("--notes", help="anything worth recording about how the review was done")
    i.add_argument("--dry-run", action="store_true", help="check the sheet without writing anything")
    i.set_defaults(func=cmd_human_import)

    p = sub.add_parser("list", help="experiments and their runs")
    p.add_argument("--dataset")
    p.add_argument("--experiments-dir", type=Path, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("check", help="verify experiments against the registries")
    p.add_argument("--experiments-dir", type=Path, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_check)

    d = sub.add_parser("dataset", help="dataset content").add_subparsers(dest="action", required=True)
    x = d.add_parser("extract-text", help="build a text source from a PDF source with pdftotext")
    x.add_argument("dataset")
    x.add_argument("--from", dest="from_source", default="pdf")
    x.add_argument("--to", dest="to_source", default="text")
    x.set_defaults(func=cmd_dataset_extract)

    p = sub.add_parser("dashboard", help="open the results dashboard (Streamlit)",
                       description="Any further options are passed to `streamlit run`, e.g. --server.port 8502.")
    p.set_defaults(func=cmd_dashboard)

    args, extra = ap.parse_known_args(argv)
    if extra and args.command != "dashboard":
        ap.error(f"unrecognized arguments: {' '.join(extra)}")
    args.streamlit_args = extra
    args.func(args)


if __name__ == "__main__":
    main()
