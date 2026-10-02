"""Classify all 10 full-text papers with methods-aware schema (Haiku + Sonnet)
and emit a markdown table for the proposal appendix.
"""
import json
import os
import subprocess
import time
from pathlib import Path

import anthropic

SPIKE_DIR = Path(__file__).parent
PAPERS = json.loads((SPIKE_DIR / "papers" / "papers_fulltext_10.json").read_text())

KEY = subprocess.run(
    ["security", "find-generic-password", "-a", os.environ["USER"], "-s", "ANTHROPIC_API_KEY", "-w"],
    capture_output=True, text=True, check=True
).stdout.strip()
client = anthropic.Anthropic(api_key=KEY)

PRICING_USD_PER_MTOK = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
}
USD_TO_GBP = 0.80

SYSTEM_PROMPT = """You are a scientometric classifier analysing research papers to measure
production of PRIMARY versus SECONDARY research outputs. "Methods" means
computational methods BROADLY — software, algorithms, statistical methods,
mathematical models, equations, ML/AI architectures, pipelines, analytical workflows.

Output VALID JSON only:
{
  "classification": "primary" | "secondary" | "mixed",
  "primary_outputs": {
    "primary_data": {"produced": bool, "description": "..."},
    "primary_software": {"produced": bool, "description": "..."},
    "primary_computational_methods": {"produced": bool, "description": "..."}
  },
  "secondary_resources_used": {
    "data_sources": [...],
    "software_used": [...],
    "methods_used": [...]
  },
  "availability_signals": {
    "data_availability": "...",
    "software_availability": "...",
    "methods_availability": "..."
  },
  "novelty_phrases": [...],
  "confidence": "high" | "medium" | "low"
}"""


def classify(paper, model):
    title = paper.get("title") or ""
    abstract = paper.get("abstract") or ""
    fulltext = paper.get("fulltext") or ""
    user_msg = f"TITLE: {title}\n\nJOURNAL: {paper.get('journal','')}  YEAR: {paper.get('year','')}\n\nABSTRACT:\n{abstract}\n\nFULL TEXT:\n{fulltext}\n"
    t0 = time.time()
    resp = client.messages.create(
        model=model, max_tokens=1200, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    elapsed = time.time() - t0
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"): text = text[4:]
    try:
        result = json.loads(text.strip())
    except Exception as e:
        result = {"_parse_error": str(e), "_raw": text[:300]}
    usage = resp.usage
    in_price, out_price = PRICING_USD_PER_MTOK[model]
    cost_usd = (usage.input_tokens / 1e6) * in_price + (usage.output_tokens / 1e6) * out_price
    return {
        "model": model,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_gbp": round(cost_usd * USD_TO_GBP, 5),
        "latency_s": round(elapsed, 1),
        "result": result,
    }


def short(items, n=3):
    if not items: return "—"
    items = [str(x).strip() for x in items if x]
    if len(items) <= n:
        return "; ".join(items)
    return "; ".join(items[:n]) + f"; … (+{len(items)-n})"


def short_text(s, n=80):
    if not s: return "—"
    s = str(s).strip()
    return s if len(s) <= n else s[:n] + "…"


def main():
    HAIKU = "claude-haiku-4-5-20251001"
    SONNET = "claude-sonnet-4-6"

    runs = []
    for i, p in enumerate(PAPERS):
        print(f"\n[{i+1}/10] {p.get('field')}: {(p.get('title') or '')[:60]}")
        for model in (HAIKU, SONNET):
            r = classify(p, model)
            r["paper"] = {
                "field": p.get("field"),
                "pmid": p.get("pmid"),
                "pmcid": p.get("pmcid"),
                "doi": p.get("doi"),
                "title": p.get("title"),
                "year": p.get("year"),
                "journal": p.get("journal"),
                "fulltext_chars": p.get("fulltext_chars"),
            }
            runs.append(r)
            cls = r["result"].get("classification", "?") if isinstance(r["result"], dict) and "_parse_error" not in r["result"] else "PARSE_ERR"
            print(f"    {model.split('-2')[0]:25}  in={r['input_tokens']:>6,}  out={r['output_tokens']:>4}  £{r['cost_gbp']:.4f}  {r['latency_s']}s  cls={cls}")

    (SPIKE_DIR / "runs_10_fulltext_methods.json").write_text(json.dumps(runs, indent=2))

    # Build the appendix table — one row per paper, using Sonnet outputs (higher quality)
    sonnet_runs = [r for r in runs if r["model"] == SONNET]
    haiku_runs = [r for r in runs if r["model"] == HAIKU]

    print("\n\n=== APPENDIX TABLE ===\n")
    md = []
    md.append("| # | PMID / DOI | Title (truncated) | Data used | Primary data? | Methods used | Primary methods? | Sonnet £ | Haiku £ |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    total_sonnet, total_haiku = 0, 0
    total_in_sonnet = total_out_sonnet = total_in_haiku = total_out_haiku = 0
    for i, (sr, hr) in enumerate(zip(sonnet_runs, haiku_runs), start=1):
        p = sr["paper"]
        sres = sr["result"] if isinstance(sr["result"], dict) and "_parse_error" not in sr["result"] else {}
        ids = []
        if p.get("pmid"): ids.append(f"PMID {p['pmid']}")
        if p.get("doi"): ids.append(p['doi'])
        id_str = "<br>".join(ids) if ids else (p.get("pmcid") or "—")
        title = short_text(p.get("title"), 70).replace("|", "/")
        sec = sres.get("secondary_resources_used", {})
        po = sres.get("primary_outputs", {})
        data_used = short(sec.get("data_sources", []))[:120].replace("|", "/")
        methods_used = short(sec.get("methods_used", []))[:120].replace("|", "/")
        primary_data = po.get("primary_data", {})
        primary_methods = po.get("primary_computational_methods", {})
        pd_str = ("Y — " + short_text(primary_data.get("description"), 60)) if primary_data.get("produced") else "N"
        pm_str = ("Y — " + short_text(primary_methods.get("description"), 60)) if primary_methods.get("produced") else "N"
        pd_str = pd_str.replace("|", "/")
        pm_str = pm_str.replace("|", "/")
        md.append(f"| {i} | {id_str} | {title} | {data_used} | {pd_str} | {methods_used} | {pm_str} | £{sr['cost_gbp']:.4f} | £{hr['cost_gbp']:.4f} |")
        total_sonnet += sr["cost_gbp"]
        total_haiku += hr["cost_gbp"]
        total_in_sonnet += sr["input_tokens"]
        total_out_sonnet += sr["output_tokens"]
        total_in_haiku += hr["input_tokens"]
        total_out_haiku += hr["output_tokens"]

    md.append(f"| **TOTAL** | 10 papers | | | | | | **£{total_sonnet:.4f}** | **£{total_haiku:.4f}** |")
    md.append("")

    avg_sonnet = total_sonnet / 10
    avg_haiku = total_haiku / 10
    md.append("**Summary stats (full text, methods-aware schema):**")
    md.append("")
    md.append("| Metric | Haiku 4.5 | Sonnet 4.6 |")
    md.append("|---|---|---|")
    md.append(f"| Avg input tokens | {total_in_haiku/10:,.0f} | {total_in_sonnet/10:,.0f} |")
    md.append(f"| Avg output tokens | {total_out_haiku/10:,.0f} | {total_out_sonnet/10:,.0f} |")
    md.append(f"| Avg cost / call (GBP) | £{avg_haiku:.4f} | £{avg_sonnet:.4f} |")
    md.append(f"| 270k Mode-2 Teacher projection (Haiku) | £{avg_haiku * 270_000:,.0f} | — |")
    md.append(f"| 15k Mode-3 Judge projection (Sonnet) | — | £{avg_sonnet * 15_000:,.0f} |")
    md.append("")
    table_md = "\n".join(md)
    (SPIKE_DIR / "appendix_table.md").write_text(table_md)
    print(table_md)
    print(f"\n\nWritten to appendix_table.md")
    print(f"Total cost for this run: £{total_sonnet + total_haiku:.4f}")


if __name__ == "__main__":
    main()
