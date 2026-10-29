"""List, in draw order, the candidates to replace a paper in the v2 list.

Candidates are run 1 manifest papers in the replaced paper's (field, band) cell
that are open access and not in the v2 list, sorted by id and then
shuffled with an RNG seeded from "<seed>:<replaced id>", so one slot's order
does not depend on another's. The first candidate whose PDF is the publisher's
version of a research article is taken; source/replacements.csv records the
decision for each one looked at.

Usage: python3 -I replacement_candidates.py <run1-manifest-v2.csv> <trial2-papers-v2.csv> ID [ID ...] [--seed N]
"""
import argparse
import csv
import random

ap = argparse.ArgumentParser()
ap.add_argument("manifest")
ap.add_argument("v2_list")
ap.add_argument("replace", nargs="+")
ap.add_argument("--seed", type=int, default=20261027)
args = ap.parse_args()

with open(args.manifest, newline="", encoding="utf-8-sig") as f:
    manifest = list(csv.DictReader(f))
with open(args.v2_list, newline="", encoding="utf-8") as f:
    taken = {p["openalex_id"] for p in csv.DictReader(f)}
by_id = {m["openalex_id"]: m for m in manifest}

for rid in args.replace:
    old = by_id[rid]
    cell = sorted((m for m in manifest
                   if (m["field"], m["band"]) == (old["field"], old["band"])
                   and m["access"] == "open" and m["openalex_id"] not in taken),
                  key=lambda m: m["openalex_id"])
    random.Random(f"{args.seed}:{rid}").shuffle(cell)
    print(f"{rid}  {old['field']} / {old['band']}")
    for m in cell:
        print(f"  {m['openalex_id']}  {m['publisher_pdf_url'] or m['vor_url']}  {m['title'][:80]}")
