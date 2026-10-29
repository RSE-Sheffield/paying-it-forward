"""Write papers.csv from the v2 paper list: paper_id (= openalex_id) first, then
the list's own columns, minus the empty Classification column.

With --replacements, each "accepted" row of that log takes its replaced
paper's place (same position), with change set to "replaces <id>".

Usage: python3 -I build_papers.py <trial2-papers-v2.csv> <papers.csv> [--replacements <replacements.csv>]
"""
import argparse
import csv

ap = argparse.ArgumentParser()
ap.add_argument("v2_list")
ap.add_argument("out")
ap.add_argument("--replacements")
args = ap.parse_args()

with open(args.v2_list, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    cols = [c for c in reader.fieldnames if c != "Classification"]

assert not any(r["Classification"] for r in rows), "Classification is no longer empty; decide what to do with it"

replaced = 0
if args.replacements:
    with open(args.replacements, newline="", encoding="utf-8") as f:
        accepted = {r["replaced_id"]: r for r in csv.DictReader(f) if r["decision"] == "accepted"}
    for i, r in enumerate(rows):
        if r["openalex_id"] in accepted:
            new = accepted.pop(r["openalex_id"])
            rows[i] = {**{c: new.get(c, "") for c in cols},
                       "openalex_id": new["candidate_id"], "change": f"replaces {r['openalex_id']}"}
            replaced += 1
    assert not accepted, f"replaced ids not in the list: {sorted(accepted)}"

with open(args.out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, ["paper_id"] + cols, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({"paper_id": r["openalex_id"], **r})
print(f"wrote {args.out} ({len(rows)} papers, {replaced} replaced)")
