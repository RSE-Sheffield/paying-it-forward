"""Compute stratified allocation of 300,000 LLM calls across UK fields × years.

Strategy: hybrid stratification.
  - Floor: 200 papers per field × year cell (52,000 calls; ensures statistical
    power within small fields/years).
  - Proportional excess: remaining 248,000 distributed in proportion to each
    field's 10-year output share.

Outputs a per-field summary and writes allocation.csv.
"""
import csv
from pathlib import Path

DATA = Path(__file__).parent / "data" / "2026-05-11-openalex-uk-citable-articles-by-field-2016-2025.csv"

TOTAL_BUDGET = 300_000
FLOOR_PER_CELL = 200
YEARS = list(range(2016, 2026))
N_YEARS = len(YEARS)


def main():
    rows = list(csv.DictReader(open(DATA)))
    # Convert numerics
    for r in rows:
        for y in YEARS:
            r[str(y)] = int(r[str(y)])
        r["Total"] = int(r["Total"])

    n_fields = len(rows)
    floor_total = n_fields * N_YEARS * FLOOR_PER_CELL
    excess_total = TOTAL_BUDGET - floor_total
    corpus_total = sum(r["Total"] for r in rows)

    print(f"Fields              : {n_fields}")
    print(f"Years               : {N_YEARS}")
    print(f"Cells (field × year): {n_fields * N_YEARS}")
    print(f"Total UK corpus     : {corpus_total:,}")
    print(f"Sampling rate       : {TOTAL_BUDGET/corpus_total*100:.2f}%")
    print(f"Floor per cell      : {FLOOR_PER_CELL}")
    print(f"Floor allocation    : {floor_total:,} ({floor_total/TOTAL_BUDGET*100:.0f}% of budget)")
    print(f"Excess to allocate  : {excess_total:,} ({excess_total/TOTAL_BUDGET*100:.0f}% of budget)")
    print()

    # Per-field allocations
    field_alloc = []
    for r in rows:
        field = r["Field"]
        field_total = r["Total"]
        share = field_total / corpus_total
        floor = N_YEARS * FLOOR_PER_CELL
        excess = round(excess_total * share)
        allocation = floor + excess
        sample_rate = allocation / field_total
        field_alloc.append({
            "Field": field,
            "field_total_10y": field_total,
            "share_pct": share * 100,
            "floor": floor,
            "excess": excess,
            "allocation": allocation,
            "sample_rate_pct": sample_rate * 100,
        })

    # Sort by total descending (matches CSV)
    field_alloc.sort(key=lambda x: -x["field_total_10y"])

    # Adjust for rounding error to sum exactly to 300k
    total_allocated = sum(f["allocation"] for f in field_alloc)
    diff = TOTAL_BUDGET - total_allocated
    if diff != 0:
        field_alloc[0]["allocation"] += diff
        field_alloc[0]["excess"] += diff
        field_alloc[0]["sample_rate_pct"] = field_alloc[0]["allocation"] / field_alloc[0]["field_total_10y"] * 100

    # Print
    print(f"{'Field':<40} {'10y total':>10} {'share%':>7} {'floor':>6} {'excess':>7} {'alloc':>7} {'rate%':>6}")
    print("-" * 92)
    for f in field_alloc:
        print(f"{f['Field']:<40} {f['field_total_10y']:>10,} {f['share_pct']:>6.1f}% {f['floor']:>6,} {f['excess']:>7,} {f['allocation']:>7,} {f['sample_rate_pct']:>5.1f}%")

    grand_total = sum(f["allocation"] for f in field_alloc)
    print("-" * 92)
    print(f"{'TOTAL':<40} {corpus_total:>10,}        {sum(f['floor'] for f in field_alloc):>6,} {sum(f['excess'] for f in field_alloc):>7,} {grand_total:>7,}")

    # Write allocation CSV
    out = Path(__file__).parent / "allocation.csv"
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["Field", "field_total_10y", "share_pct", "floor", "excess", "allocation", "sample_rate_pct"])
        writer.writeheader()
        for f in field_alloc:
            writer.writerow(f)
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
