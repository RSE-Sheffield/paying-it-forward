"""Write each reviewer's paper order, one column per reviewer, from saved
Google Sheets reads of their Classifications sheet.

Each read is the Sheets get_values response for the sheet's openalex_id and doi
columns (header row included), saved as <name>.json. Every row's doi must be
that id's doi in papers.csv, and each sheet must hold each paper exactly once,
so an id that was mistyped or moved between rows fails instead of being written.

Usage: python3 -I sheet_order.py <papers.csv> <out.csv> <read.json> [<read.json> ...]
"""
import argparse
import csv
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("papers")
ap.add_argument("out")
ap.add_argument("reads", nargs="+")
args = ap.parse_args()

with open(args.papers, newline="", encoding="utf-8") as f:
    doi_of = {p["paper_id"]: p["doi"] for p in csv.DictReader(f)}

columns = {}
for path in args.reads:
    name = os.path.splitext(os.path.basename(path))[0]
    with open(path, encoding="utf-8") as f:
        header, *rows = json.load(f)["values"]
    assert header == ["openalex_id", "doi"], f"{name}: unexpected header {header}"
    for n, (pid, doi) in enumerate(rows, start=2):
        assert pid in doi_of, f"{name} row {n}: {pid} is not in {args.papers}"
        assert doi == doi_of[pid], f"{name} row {n}: {pid} has doi {doi}, papers.csv says {doi_of[pid]}"
    ids = [pid for pid, _ in rows]
    assert sorted(ids) == sorted(doi_of), f"{name}: does not hold each paper exactly once"
    columns[name] = ids

with open(args.out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(columns)
    w.writerows(zip(*columns.values()))
print(f"wrote {args.out} ({len(doi_of)} papers x {len(columns)} reviewers: {', '.join(columns)})")
