"""Build the appendix table for all 30 papers, using Sonnet-FT outputs for
data/methods extraction (highest quality), and showing classifications and
costs across all three scaling methods.
"""
import json
from pathlib import Path

SPIKE_DIR = Path(__file__).parent
runs = json.loads((SPIKE_DIR / "runs_30_scaling.json").read_text())
papers = json.loads((SPIKE_DIR / "papers" / "papers_fulltext_30.json").read_text())

HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"

# Index runs by (pmid, model, method)
runs_idx = {}
for r in runs:
    pmid = r.get("pmid")
    key = (pmid, r.get("model"), r.get("method"))
    runs_idx[key] = r


def short_list(items, n=2, maxlen=85):
    if not items: return "—"
    items = [str(x).strip() for x in items if x]
    if not items: return "—"
    base = "; ".join(items[:n])
    extra = f"… (+{len(items)-n})" if len(items) > n else ""
    s = base + extra
    if len(s) > maxlen:
        s = s[:maxlen-1] + "…"
    return s.replace("|", "/")


def short_text(s, n=70):
    if not s: return "—"
    s = str(s).strip().replace("\n", " ")
    if len(s) > n:
        s = s[:n] + "…"
    return s.replace("|", "/")


def build_row(i, paper):
    pmid = paper.get("pmid")
    pmcid = paper.get("pmcid")
    doi = paper.get("doi")
    title = paper.get("title")
    field = paper.get("field")

    sonnet_ft = runs_idx.get((pmid, SONNET, "FT"))
    haiku_ft = runs_idx.get((pmid, HAIKU, "FT"))
    haiku_abs = runs_idx.get((pmid, HAIKU, "ABS"))

    # Use Sonnet-FT result for data/methods (highest quality)
    sres = sonnet_ft["result"] if sonnet_ft and isinstance(sonnet_ft.get("result"), dict) and "_parse_error" not in sonnet_ft["result"] else {}
    po = sres.get("primary_outputs", {})
    sec = sres.get("secondary_resources_used", {})

    # Build ID column
    ids = []
    if pmid: ids.append(f"PMID {pmid}")
    if doi: ids.append(doi)
    if not ids and pmcid: ids.append(pmcid)
    id_str = "<br>".join(ids) if ids else "—"

    data_used = short_list(sec.get("data_sources", []))
    methods_used = short_list(sec.get("methods_used", []))

    pd_info = po.get("primary_data", {})
    pd_str = ("Y — " + short_text(pd_info.get("description"), 55)) if pd_info.get("produced") else "N"

    pm_info = po.get("primary_computational_methods", {})
    pm_str = ("Y — " + short_text(pm_info.get("description"), 55)) if pm_info.get("produced") else "N"

    s_cls = (sonnet_ft.get("classification") or "—") if sonnet_ft else "—"
    h_ft_cls = (haiku_ft.get("classification") or "—") if haiku_ft else "—"
    h_abs_cls = (haiku_abs.get("classification") or "—") if haiku_abs else "—"

    s_cost = f"£{sonnet_ft['cost_gbp']:.4f}" if sonnet_ft else "—"
    h_ft_cost = f"£{haiku_ft['cost_gbp']:.4f}" if haiku_ft else "—"
    h_abs_cost = f"£{haiku_abs['cost_gbp']:.4f}" if haiku_abs else "—"

    title_short = short_text(title, 55)
    return f"| {i} | {id_str} | {title_short} | {data_used} | {pd_str} | {methods_used} | {pm_str} | {s_cls} | {h_ft_cls} | {h_abs_cls} | {s_cost} | {h_ft_cost} | {h_abs_cost} |"


def main():
    lines = []
    lines.append("| # | PMID / DOI | Title (truncated) | Data used | Primary data? | Methods used | Primary methods? | Sonnet-FT class | Haiku-FT class | Haiku-ABS class | Sonnet £ | Haiku-FT £ | Haiku-ABS £ |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    total_s, total_hft, total_habs = 0, 0, 0
    s_in_total = s_out_total = hft_in_total = hft_out_total = habs_in_total = habs_out_total = 0
    n_s = n_hft = n_habs = 0
    for i, p in enumerate(papers, start=1):
        lines.append(build_row(i, p))
        pmid = p.get("pmid")
        sr = runs_idx.get((pmid, SONNET, "FT"))
        hr = runs_idx.get((pmid, HAIKU, "FT"))
        ar = runs_idx.get((pmid, HAIKU, "ABS"))
        if sr: total_s += sr["cost_gbp"]; s_in_total += sr["input_tokens"]; s_out_total += sr["output_tokens"]; n_s += 1
        if hr: total_hft += hr["cost_gbp"]; hft_in_total += hr["input_tokens"]; hft_out_total += hr["output_tokens"]; n_hft += 1
        if ar: total_habs += ar["cost_gbp"]; habs_in_total += ar["input_tokens"]; habs_out_total += ar["output_tokens"]; n_habs += 1

    lines.append(f"| **TOTAL** | 30 papers | | | | | | | | | **£{total_s:.4f}** | **£{total_hft:.4f}** | **£{total_habs:.4f}** |")
    table_md = "\n".join(lines)

    summary = []
    summary.append("\n### Summary stats (N=30)\n")
    summary.append("| Method | n | Avg input tokens | Avg output tokens | Avg cost / call | Total (30 papers) |")
    summary.append("|---|---|---|---|---|---|")
    summary.append(f"| Sonnet-FT | {n_s} | {s_in_total/n_s:,.0f} | {s_out_total/n_s:,.0f} | £{total_s/n_s:.4f} | £{total_s:.4f} |")
    summary.append(f"| Haiku-FT | {n_hft} | {hft_in_total/n_hft:,.0f} | {hft_out_total/n_hft:,.0f} | £{total_hft/n_hft:.4f} | £{total_hft:.4f} |")
    summary.append(f"| Haiku-ABS | {n_habs} | {habs_in_total/n_habs:,.0f} | {habs_out_total/n_habs:,.0f} | £{total_habs/n_habs:.4f} | £{total_habs:.4f} |")
    summary.append(f"\nFull spike run cost (90 calls): **£{(total_s + total_hft + total_habs):.4f}**.")

    out = table_md + "\n" + "\n".join(summary)
    (SPIKE_DIR / "appendix_table_30.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
