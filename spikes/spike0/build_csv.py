"""Build a clean CSV of the 30-paper spike for Google Sheets import.

Output: spike_30_papers.csv — one row per paper, columns for paper metadata,
schema-extracted fields (data/software/methods used + produced + availability),
classifications across the three scaling methods, and costs/tokens.

Also: spike_30_summary.csv — per-method summary stats + agreement matrix.
"""
import csv
import json
from pathlib import Path

SPIKE_DIR = Path(__file__).parent
runs = json.loads((SPIKE_DIR / "runs_30_scaling.json").read_text())
papers = json.loads((SPIKE_DIR / "papers" / "papers_fulltext_30.json").read_text())

HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"

# Index runs by (pmid, model, method)
idx = {}
for r in runs:
    key = (r.get("pmid"), r.get("model"), r.get("method"))
    idx[key] = r


def join_list(items, sep=" | "):
    if not items: return ""
    return sep.join(str(x).strip().replace("\n", " ") for x in items if x)


def get(d, *path, default=""):
    cur = d
    for p in path:
        if not isinstance(cur, dict): return default
        cur = cur.get(p)
        if cur is None: return default
    return cur


def cell(text):
    """Sanitise a cell value — strip newlines, normalise whitespace."""
    if text is None or text == "":
        return ""
    s = str(text).replace("\n", " ").replace("\r", " ")
    return " ".join(s.split())


def bool_yn(v):
    if v is True: return "Y"
    if v is False: return "N"
    return ""


def main():
    # Main per-paper CSV
    out_main = SPIKE_DIR / "spike_30_papers.csv"
    fieldnames = [
        "#", "PMID", "DOI", "PMCID", "Field", "Year", "Journal", "Title", "Full text chars",
        # Data
        "Data sources used", "Primary data produced", "Primary data description",
        # Software
        "Software used", "Primary software produced", "Primary software description",
        # Methods
        "Methods used", "Primary computational methods produced", "Primary methods description",
        # Availability
        "Data availability", "Software availability", "Methods availability",
        # Novelty
        "Novelty phrases",
        # Classifications
        "Sonnet-FT classification", "Haiku-FT classification", "Haiku-ABS classification",
        # Confidence (Sonnet-FT only — best quality, otherwise too many cols)
        "Sonnet-FT confidence",
        # Costs (per call, GBP)
        "Sonnet-FT cost GBP", "Haiku-FT cost GBP", "Haiku-ABS cost GBP",
        # Tokens
        "Sonnet-FT input tokens", "Sonnet-FT output tokens",
        "Haiku-FT input tokens", "Haiku-FT output tokens",
        "Haiku-ABS input tokens", "Haiku-ABS output tokens",
        # Latency
        "Sonnet-FT latency s", "Haiku-FT latency s", "Haiku-ABS latency s",
        # Inter-method agreement
        "Sonnet-FT vs Haiku-FT agree", "Haiku-FT vs Haiku-ABS agree", "Sonnet-FT vs Haiku-ABS agree",
    ]

    with open(out_main, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        for i, p in enumerate(papers, start=1):
            pmid = p.get("pmid")
            sonnet = idx.get((pmid, SONNET, "FT"), {})
            haiku_ft = idx.get((pmid, HAIKU, "FT"), {})
            haiku_abs = idx.get((pmid, HAIKU, "ABS"), {})

            # Use Sonnet-FT result for data/methods extraction (highest quality)
            sres = sonnet.get("result") if isinstance(sonnet.get("result"), dict) and "_parse_error" not in sonnet.get("result", {}) else {}

            s_cls = sonnet.get("classification", "")
            h_ft_cls = haiku_ft.get("classification", "")
            h_abs_cls = haiku_abs.get("classification", "")

            row = {
                "#": i,
                "PMID": cell(p.get("pmid")),
                "DOI": cell(p.get("doi")),
                "PMCID": cell(p.get("pmcid")),
                "Field": cell(p.get("field")),
                "Year": cell(p.get("year")),
                "Journal": cell(p.get("journal")),
                "Title": cell(p.get("title")),
                "Full text chars": p.get("fulltext_chars", ""),
                # Data
                "Data sources used": cell(join_list(get(sres, "secondary_resources_used", "data_sources", default=[]))),
                "Primary data produced": bool_yn(get(sres, "primary_outputs", "primary_data", "produced")),
                "Primary data description": cell(get(sres, "primary_outputs", "primary_data", "description")),
                # Software
                "Software used": cell(join_list(get(sres, "secondary_resources_used", "software_used", default=[]))),
                "Primary software produced": bool_yn(get(sres, "primary_outputs", "primary_software", "produced")),
                "Primary software description": cell(get(sres, "primary_outputs", "primary_software", "description")),
                # Methods
                "Methods used": cell(join_list(get(sres, "secondary_resources_used", "methods_used", default=[]))),
                "Primary computational methods produced": bool_yn(get(sres, "primary_outputs", "primary_computational_methods", "produced")),
                "Primary methods description": cell(get(sres, "primary_outputs", "primary_computational_methods", "description")),
                # Availability
                "Data availability": cell(get(sres, "availability_signals", "data_availability")),
                "Software availability": cell(get(sres, "availability_signals", "software_availability")),
                "Methods availability": cell(get(sres, "availability_signals", "methods_availability")),
                # Novelty
                "Novelty phrases": cell(join_list(get(sres, "novelty_phrases", default=[]))),
                # Classifications
                "Sonnet-FT classification": s_cls,
                "Haiku-FT classification": h_ft_cls,
                "Haiku-ABS classification": h_abs_cls,
                # Confidence
                "Sonnet-FT confidence": cell(get(sres, "confidence")),
                # Costs
                "Sonnet-FT cost GBP": f"{sonnet.get('cost_gbp', 0):.5f}" if sonnet else "",
                "Haiku-FT cost GBP": f"{haiku_ft.get('cost_gbp', 0):.5f}" if haiku_ft else "",
                "Haiku-ABS cost GBP": f"{haiku_abs.get('cost_gbp', 0):.5f}" if haiku_abs else "",
                # Tokens
                "Sonnet-FT input tokens": sonnet.get("input_tokens", "") if sonnet else "",
                "Sonnet-FT output tokens": sonnet.get("output_tokens", "") if sonnet else "",
                "Haiku-FT input tokens": haiku_ft.get("input_tokens", "") if haiku_ft else "",
                "Haiku-FT output tokens": haiku_ft.get("output_tokens", "") if haiku_ft else "",
                "Haiku-ABS input tokens": haiku_abs.get("input_tokens", "") if haiku_abs else "",
                "Haiku-ABS output tokens": haiku_abs.get("output_tokens", "") if haiku_abs else "",
                # Latency
                "Sonnet-FT latency s": sonnet.get("latency_s", "") if sonnet else "",
                "Haiku-FT latency s": haiku_ft.get("latency_s", "") if haiku_ft else "",
                "Haiku-ABS latency s": haiku_abs.get("latency_s", "") if haiku_abs else "",
                # Agreement
                "Sonnet-FT vs Haiku-FT agree": ("Y" if s_cls and h_ft_cls and s_cls == h_ft_cls else ("N" if s_cls and h_ft_cls else "")),
                "Haiku-FT vs Haiku-ABS agree": ("Y" if h_ft_cls and h_abs_cls and h_ft_cls == h_abs_cls else ("N" if h_ft_cls and h_abs_cls else "")),
                "Sonnet-FT vs Haiku-ABS agree": ("Y" if s_cls and h_abs_cls and s_cls == h_abs_cls else ("N" if s_cls and h_abs_cls else "")),
            }
            writer.writerow(row)

    # Summary CSV — per-method stats + agreement matrix
    out_summary = SPIKE_DIR / "spike_30_summary.csv"
    methods = [
        ("Sonnet 4.6 — full text", SONNET, "FT"),
        ("Haiku 4.5 — full text", HAIKU, "FT"),
        ("Haiku 4.5 — abstract only", HAIKU, "ABS"),
    ]

    method_runs = {label: [r for r in runs if r.get("model") == m and r.get("method") == mt] for label, m, mt in methods}

    with open(out_summary, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["=== Per-method summary stats (N=30) ==="])
        w.writerow(["Method", "n", "Avg input tokens", "Avg output tokens", "Avg cost GBP", "Median cost GBP",
                    "Min cost GBP", "Max cost GBP", "Total cost GBP (30 papers)", "300k projection GBP"])
        for label, _, _ in methods:
            rs = method_runs[label]
            if not rs: continue
            n = len(rs)
            costs = sorted(r["cost_gbp"] for r in rs)
            avg_in = sum(r["input_tokens"] for r in rs) / n
            avg_out = sum(r["output_tokens"] for r in rs) / n
            avg_cost = sum(costs) / n
            med = costs[n // 2]
            total = sum(costs)
            w.writerow([label, n, f"{avg_in:.0f}", f"{avg_out:.0f}",
                        f"{avg_cost:.5f}", f"{med:.5f}", f"{min(costs):.5f}", f"{max(costs):.5f}",
                        f"{total:.4f}", f"{avg_cost * 300_000:.0f}"])

        w.writerow([])
        w.writerow(["=== Inter-method classification agreement (N=30) ==="])
        w.writerow(["Pair", "Agreement count", "Total", "Agreement %", "Cohen's kappa"])
        # Pull from summary_30.json if available, else recompute
        try:
            summary_json = json.loads((SPIKE_DIR / "summary_30.json").read_text())
            pairs = summary_json.get("agreement_pairs", {})
            for pair_name, stats in pairs.items():
                w.writerow([pair_name.replace("_vs_", " vs "), stats["agree"], stats["total"],
                            f"{stats['agree']/stats['total']*100:.0f}%", f"{stats['kappa']:.3f}"])
        except Exception as e:
            w.writerow([f"(error reading summary_30.json: {e})"])

        w.writerow([])
        w.writerow(["=== Total cost for 30-paper, 3-method spike (90 API calls) ==="])
        grand_total = sum(sum(r["cost_gbp"] for r in method_runs[label]) for label, _, _ in methods)
        w.writerow(["Total spike cost", "", "", "", "", "", "", "", f"£{grand_total:.4f}"])

        w.writerow([])
        w.writerow(["=== Cost projections to full UK 300k stratified sample ==="])
        w.writerow(["Method", "Per call (GBP)", "300k all-method cost", "Notes"])
        w.writerow(["Sonnet-FT", f"{sum(r['cost_gbp'] for r in method_runs['Sonnet 4.6 — full text'])/30:.4f}",
                    f"{(sum(r['cost_gbp'] for r in method_runs['Sonnet 4.6 — full text'])/30) * 300_000:.0f}",
                    "Highest quality; expensive at scale"])
        w.writerow(["Haiku-FT", f"{sum(r['cost_gbp'] for r in method_runs['Haiku 4.5 — full text'])/30:.4f}",
                    f"{(sum(r['cost_gbp'] for r in method_runs['Haiku 4.5 — full text'])/30) * 300_000:.0f}",
                    "Recommended Mode 2 workhorse"])
        w.writerow(["Haiku-ABS", f"{sum(r['cost_gbp'] for r in method_runs['Haiku 4.5 — abstract only'])/30:.4f}",
                    f"{(sum(r['cost_gbp'] for r in method_runs['Haiku 4.5 — abstract only'])/30) * 300_000:.0f}",
                    "Scaling baseline; lower quality"])

    print(f"Written: {out_main} (30 rows + header, {len(fieldnames)} columns)")
    print(f"Written: {out_summary} (summary stats + agreement matrix + projections)")
    print(f"\nFile sizes:")
    print(f"  {out_main.name}: {out_main.stat().st_size:,} bytes")
    print(f"  {out_summary.name}: {out_summary.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
