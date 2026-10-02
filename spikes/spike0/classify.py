"""Classify 10 papers via Anthropic API; record token usage and cost.

Modes tested:
  - Haiku 4.5 (workhorse — primary cost baseline for 300k scale-up)
  - Sonnet 4.6 (judge — for quality comparison on a subset)

Pricing reference (USD per million tokens, public Anthropic pricing 2026):
  claude-haiku-4-5-20251001     $1.00 input  / $5.00  output
  claude-sonnet-4-6             $3.00 input  / $15.00 output

GBP conversion: 1 USD = 0.80 GBP (rough, May 2026).
"""
import json
import os
import subprocess
import time
from pathlib import Path

import anthropic

SPIKE_DIR = Path(__file__).parent
PAPERS = json.loads((SPIKE_DIR / "papers" / "papers.json").read_text())

# API key from macOS keychain
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

SYSTEM_PROMPT = """You are a scientometric classifier analysing biomedical research papers.

Given a paper (title, abstract, and where available full text), produce a JSON object with:
  - classification: "primary" (paper produces NEW primary data and/or NEW software/code), "secondary" (paper analyses pre-existing data/code without producing new primary resources), or "mixed" (both)
  - primary_data_produced: true/false (paper generates new datasets, samples, measurements)
  - primary_code_produced: true/false (paper releases new software, scripts, models)
  - data_references_used: list of strings naming any pre-existing datasets, databases, or data sources used (e.g. "UK Biobank", "NHANES", "GenBank accession X")
  - code_references_used: list of strings naming any pre-existing software/code used
  - data_availability_signal: short string describing what the paper says about data availability (e.g. "deposited in GEO accession XXX", "available on request", "not stated")
  - code_availability_signal: short string describing what the paper says about code availability
  - confidence: "high", "medium", "low" — your confidence in the classification

Output VALID JSON only. No prose."""

def truncate_to_budget(text: str, max_chars: int) -> str:
    if not text or len(text) <= max_chars:
        return text or ""
    return text[:max_chars] + "\n\n[...truncated...]"


def classify(paper: dict, model: str, char_budget: int) -> dict:
    title = paper.get("title") or ""
    abstract = paper.get("abstract") or ""
    fulltext = paper.get("fulltext") or ""

    if fulltext:
        body = truncate_to_budget(fulltext, char_budget)
        body_label = "FULL TEXT (methods/data availability sections)"
    else:
        body = abstract
        body_label = "ABSTRACT ONLY"

    user_msg = f"""TITLE: {title}

JOURNAL: {paper.get('journal', '')}  YEAR: {paper.get('year', '')}

{body_label}:
{body}
"""

    t0 = time.time()
    resp = client.messages.create(
        model=model,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    elapsed = time.time() - t0

    text = "".join(b.text for b in resp.content if b.type == "text")
    usage = resp.usage

    in_price, out_price = PRICING_USD_PER_MTOK[model]
    cost_usd = (usage.input_tokens / 1e6) * in_price + (usage.output_tokens / 1e6) * out_price

    try:
        # Strip code-fence if model wrapped it
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        result = json.loads(cleaned)
    except Exception as e:
        result = {"_parse_error": str(e), "_raw": text}

    return {
        "pmid": paper["pmid"],
        "field": paper["field"],
        "model": model,
        "body_type": body_label,
        "body_chars": len(body),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": round(cost_usd, 6),
        "cost_gbp": round(cost_usd * USD_TO_GBP, 6),
        "latency_s": round(elapsed, 2),
        "result": result,
    }


def main():
    HAIKU = "claude-haiku-4-5-20251001"
    SONNET = "claude-sonnet-4-6"

    runs = []
    print("=== HAIKU 4.5 — all 10 papers (cost baseline) ===")
    for p in PAPERS:
        # ~30k chars ≈ ~7.5k tokens; well within Haiku context, reasonable for cost
        r = classify(p, HAIKU, char_budget=30000)
        runs.append(r)
        cls = r["result"].get("classification", "?") if isinstance(r["result"], dict) else "?"
        print(f"  {p['field']:14} {r['body_type']:42} in={r['input_tokens']:5}  out={r['output_tokens']:4}  ${r['cost_usd']:.5f}  £{r['cost_gbp']:.5f}  cls={cls}")

    print("\n=== SONNET 4.6 — 3 papers (judge comparison) ===")
    sonnet_subset = [PAPERS[2], PAPERS[5], PAPERS[0]]  # neuroscience full-text, pharma full-text, medicine abstract
    for p in sonnet_subset:
        r = classify(p, SONNET, char_budget=30000)
        runs.append(r)
        cls = r["result"].get("classification", "?") if isinstance(r["result"], dict) else "?"
        print(f"  {p['field']:14} {r['body_type']:42} in={r['input_tokens']:5}  out={r['output_tokens']:4}  ${r['cost_usd']:.5f}  £{r['cost_gbp']:.5f}  cls={cls}")

    (SPIKE_DIR / "runs.json").write_text(json.dumps(runs, indent=2))

    # Aggregate
    print("\n=== AGGREGATE STATS ===")
    for model in (HAIKU, SONNET):
        rs = [r for r in runs if r["model"] == model]
        if not rs:
            continue
        n = len(rs)
        avg_in = sum(r["input_tokens"] for r in rs) / n
        avg_out = sum(r["output_tokens"] for r in rs) / n
        avg_cost_usd = sum(r["cost_usd"] for r in rs) / n
        avg_cost_gbp = sum(r["cost_gbp"] for r in rs) / n
        avg_latency = sum(r["latency_s"] for r in rs) / n
        proj_300k_gbp = avg_cost_gbp * 300_000
        print(f"\n{model} (n={n}):")
        print(f"  avg input tokens : {avg_in:,.0f}")
        print(f"  avg output tokens: {avg_out:,.0f}")
        print(f"  avg cost/call    : ${avg_cost_usd:.5f}  =  £{avg_cost_gbp:.5f}")
        print(f"  avg latency      : {avg_latency:.1f}s")
        print(f"  300k projection  : £{proj_300k_gbp:,.0f}")


if __name__ == "__main__":
    main()
