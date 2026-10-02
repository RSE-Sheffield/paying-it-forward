"""Re-run classification on the 2 PMC papers WITHOUT truncation — measure true
full-text cost. Also try a deeper fetch for one of the abstract-only papers
via Crossref/OpenAlex to see if we can broaden full-text coverage.
"""
import json
import os
import subprocess
import time
from pathlib import Path

import anthropic

SPIKE_DIR = Path(__file__).parent
PAPERS = json.loads((SPIKE_DIR / "papers" / "papers.json").read_text())

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

Given a paper (title, abstract, and full text), produce a JSON object with:
  - classification: "primary" / "secondary" / "mixed"
  - primary_data_produced: true/false
  - primary_code_produced: true/false
  - data_references_used: list of strings
  - code_references_used: list of strings
  - data_availability_signal: short string
  - code_availability_signal: short string
  - confidence: "high" / "medium" / "low"

Output VALID JSON only. No prose."""


def classify_full(paper: dict, model: str, max_chars: int | None = None) -> dict:
    title = paper.get("title") or ""
    abstract = paper.get("abstract") or ""
    fulltext = paper.get("fulltext") or ""

    body = fulltext
    if max_chars and len(body) > max_chars:
        body = body[:max_chars] + "\n[...truncated...]"
    body_chars = len(body)
    label = "FULL TEXT (untruncated)" if max_chars is None else f"FULL TEXT (truncated to {max_chars}ch)"

    user_msg = f"""TITLE: {title}\n\nJOURNAL: {paper.get('journal','')}  YEAR: {paper.get('year','')}\n\nABSTRACT:\n{abstract}\n\n{label}:\n{body}\n"""

    t0 = time.time()
    resp = client.messages.create(
        model=model,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    elapsed = time.time() - t0
    usage = resp.usage
    text = "".join(b.text for b in resp.content if b.type == "text")

    in_price, out_price = PRICING_USD_PER_MTOK[model]
    cost_usd = (usage.input_tokens / 1e6) * in_price + (usage.output_tokens / 1e6) * out_price

    try:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"): cleaned = cleaned[4:]
        result = json.loads(cleaned)
        cls = result.get("classification", "?")
    except Exception:
        cls = "PARSE_ERROR"

    return {
        "pmid": paper["pmid"],
        "field": paper["field"],
        "model": model,
        "body_label": label,
        "body_chars": body_chars,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": round(cost_usd, 6),
        "cost_gbp": round(cost_usd * USD_TO_GBP, 6),
        "latency_s": round(elapsed, 2),
        "classification": cls,
    }


def main():
    HAIKU = "claude-haiku-4-5-20251001"
    SONNET = "claude-sonnet-4-6"

    full_text_papers = [p for p in PAPERS if p.get("fulltext")]
    print(f"Papers with PMC full text: {len(full_text_papers)}/{len(PAPERS)}")
    for p in full_text_papers:
        print(f"  {p['field']:14} PMID {p['pmid']}  {len(p['fulltext']):>7,} chars")
    print()

    runs = []
    for p in full_text_papers:
        for model in (HAIKU, SONNET):
            r = classify_full(p, model, max_chars=None)
            runs.append(r)
            print(f"  {p['field']:14} {model.split('-2')[0]:25} {r['body_chars']:>7,}ch  in={r['input_tokens']:>6,}  out={r['output_tokens']:>4}  ${r['cost_usd']:.4f}  £{r['cost_gbp']:.4f}  cls={r['classification']}")

    (SPIKE_DIR / "runs_fulltext.json").write_text(json.dumps(runs, indent=2))

    print("\n=== TRUE FULL-TEXT COST PROJECTIONS ===")
    for model in (HAIKU, SONNET):
        rs = [r for r in runs if r["model"] == model]
        n = len(rs)
        avg_in = sum(r["input_tokens"] for r in rs) / n
        avg_out = sum(r["output_tokens"] for r in rs) / n
        avg_cost_gbp = sum(r["cost_gbp"] for r in rs) / n
        avg_lat = sum(r["latency_s"] for r in rs) / n
        # Projections assuming ~30% of 300k papers have similar-size full text
        cost_300k_all = avg_cost_gbp * 300_000
        cost_30pct = avg_cost_gbp * 90_000  # 30% of 300k on full text
        # Combined: 30% full-text at this cost + 70% abstract-only at spike's ~£0.001
        cost_mixed = (avg_cost_gbp * 0.30 + 0.001 * 0.70) * 300_000
        print(f"\n{model} (n={n}):")
        print(f"  avg input tokens : {avg_in:>10,.0f}")
        print(f"  avg output tokens: {avg_out:>10,.0f}")
        print(f"  avg cost / call  : £{avg_cost_gbp:.5f}")
        print(f"  avg latency      : {avg_lat:.1f}s")
        print(f"  All 300k @ full text: £{cost_300k_all:>9,.0f}")
        print(f"  90k (30%) @ full text + 210k abstract-only @ ~£0.001 = mixed: £{cost_mixed:>9,.0f}")


if __name__ == "__main__":
    main()
