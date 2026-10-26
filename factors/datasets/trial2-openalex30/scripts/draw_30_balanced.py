"""Draw a 30-paper balanced subsample from the PIF run 1 venue-banded sample.

Design:
  - bands 8/8/7/7 (which two bands get 8 is random)
  - every one of the 26 fields appears at least once; 4 random fields get a
    second paper, in a different band from their first
  - within each (field, band) cell, one paper is chosen at random (any year)

With --pdf-check, only papers whose oa_pdf_url returned a real PDF in that
check file (from check_pdf_links.py) are eligible.

Usage: python3 -I draw_30_balanced.py <banded.csv> <out.csv> [--seed N]
                                      [--pdf-check pdf_link_check.csv]
"""
import argparse
import csv
import random
from collections import Counter, defaultdict

N = 30

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("out")
ap.add_argument("--seed", type=int, default=20261023)
ap.add_argument("--pdf-check")
args = ap.parse_args()

with open(args.src, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    columns = reader.fieldnames
    rows = list(reader)

fields = sorted({r["field"] for r in rows})
bands = sorted({r["band"] for r in rows})

pdf_checked_at = ""
if args.pdf_check:
    with open(args.pdf_check, newline="", encoding="utf-8") as f:
        checks = list(csv.DictReader(f))
    verified = {c["openalex_id"] for c in checks if c["is_pdf"] == "True"}
    pdf_checked_at = checks[0]["checked_at"] if checks else ""
    rows = [r for r in rows if r["openalex_id"] in verified]

cells = defaultdict(list)
for r in rows:
    cells[(r["field"], r["band"])].append(r)
for k in cells:
    cells[k].sort(key=lambda r: r["openalex_id"])  # order-independent of the file

rng = random.Random(args.seed)

# Band quotas: 8/8/7/7, random which bands get the extra paper.
base, extra = divmod(N, len(bands))
order = bands[:]
rng.shuffle(order)
quota = {b: base + (1 if i < extra else 0) for i, b in enumerate(order)}

# Pair field slots (every field once, plus N - len(fields) repeats) with band
# labels; redraw until no field repeats a band and every cell has a paper.
band_slots = [b for b in bands for _ in range(quota[b])]
for attempt in range(1, 10_001):
    field_slots = fields + rng.sample(fields, N - len(fields))
    rng.shuffle(band_slots)
    pairs = list(zip(field_slots, band_slots))
    if len(set(pairs)) == N and all(cells[p] for p in pairs):
        break
else:
    raise SystemExit("could not find a valid field/band pairing")

picked = [rng.choice(cells[p]) for p in pairs]
picked.sort(key=lambda r: (bands.index(r["band"]), r["field"], r["year"]))

out_cols = columns + ["draw_seed", "Classification"]
if args.pdf_check:
    out_cols.insert(-1, "pdf_checked_at")
with open(args.out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=out_cols)
    w.writeheader()
    for r in picked:
        extra_cols = {"pdf_checked_at": pdf_checked_at} if args.pdf_check else {}
        w.writerow({**r, "draw_seed": args.seed, **extra_cols, "Classification": ""})

print(f"seed={args.seed} pool={len(rows)} pairing_attempts={attempt}")
print("bands: ", dict(Counter(r["band"] for r in picked)))
print("fields:", len({r["field"] for r in picked}), "distinct")
print("years: ", dict(sorted(Counter(r["year"] for r in picked).items())))
print("has oa_pdf_url:", sum(1 for r in picked if r["oa_pdf_url"]), "/", N)
print("has any oa_url:", sum(1 for r in picked if r["oa_url"]), "/", N)
print(f"wrote {args.out}")
