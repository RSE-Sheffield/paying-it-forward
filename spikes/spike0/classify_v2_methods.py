"""Revised classification with explicit computational-methods extraction.

Addresses gap in classify.py: the original schema asked for "code references"
but didn't capture computational methods broadly (algorithms, statistical
methods, mathematical models, equations, pipelines). Re-run on the 2 PMC
full-text papers to test methods coverage and validate cost.
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

SYSTEM_PROMPT = """You are a scientometric classifier analysing research papers to measure
production of PRIMARY versus SECONDARY research outputs. "Methods" here means
computational methods BROADLY — software, algorithms, statistical methods,
mathematical models, equations, simulation models, ML/AI architectures, data
pipelines, analytical workflows. NOT just code.

Given a paper, produce a JSON object:

{
  "classification": "primary" | "secondary" | "mixed",
    // "primary" if paper produces NEW primary data and/or NEW primary methods
    // "secondary" if paper analyses pre-existing data/methods without producing new primary resources
    // "mixed" if both

  "primary_outputs": {
    "primary_data": {"produced": bool, "description": "..."},
       // new datasets, samples, measurements, surveys, observations
    "primary_software": {"produced": bool, "description": "..."},
       // new software package, library, application, model weights
    "primary_computational_methods": {"produced": bool, "description": "..."}
       // new algorithm, novel statistical method, new mathematical model,
       // new equation/formulation, new ML architecture, new pipeline,
       // new analytical workflow. NOT just application of standard methods.
  },

  "secondary_resources_used": {
    "data_sources": [...],         // pre-existing datasets/databases used
    "software_used": [...],        // pre-existing software/tools used
    "methods_used": [...]          // pre-existing computational methods used
                                   // (e.g. "PCA", "BLAST", "linear regression",
                                   //  "Random Forest", "MCMC", "BERT")
  },

  "availability_signals": {
    "data_availability": "...",    // what the paper says (deposited where, DOI, etc.)
    "software_availability": "...",// GitHub repo, package URL, "available on request"
    "methods_availability": "..."  // pseudocode in paper, equations documented,
                                   // supplementary mathematical specifications,
                                   // or "not documented beyond brief description"
  },

  "novelty_phrases": [...],        // verbatim phrases signalling primary-output
                                   // claims: "we developed", "we propose",
                                   // "novel approach", "new algorithm",
                                   // "first to demonstrate", etc.

  "confidence": "high" | "medium" | "low"
}

Output VALID JSON only. No prose."""


def classify(paper: dict, model: str, max_chars: int | None = None) -> dict:
    title = paper.get("title") or ""
    abstract = paper.get("abstract") or ""
    fulltext = paper.get("fulltext") or ""

    body = fulltext or abstract
    label = "FULL TEXT" if fulltext else "ABSTRACT ONLY"
    if max_chars and len(body) > max_chars:
        body = body[:max_chars] + "\n[...truncated...]"

    user_msg = f"""TITLE: {title}\n\nJOURNAL: {paper.get('journal','')}  YEAR: {paper.get('year','')}\n\nABSTRACT:\n{abstract}\n\n{label}:\n{body}\n"""

    t0 = time.time()
    resp = client.messages.create(
        model=model,
        max_tokens=1200,  # larger output budget for richer schema
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
    except Exception as e:
        result = {"_parse_error": str(e), "_raw": text[:500]}

    return {
        "pmid": paper["pmid"],
        "field": paper["field"],
        "model": model,
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

    full_text_papers = [p for p in PAPERS if p.get("fulltext")]
    print(f"Re-running with methods-aware schema on {len(full_text_papers)} PMC full-text papers\n")

    runs = []
    for p in full_text_papers:
        for model in (HAIKU, SONNET):
            r = classify(p, model, max_chars=None)
            runs.append(r)
            res = r["result"]
            cls = res.get("classification", "?") if isinstance(res, dict) and "_parse_error" not in res else "PARSE_ERR"
            print(f"  {p['field']:14} {model.split('-2')[0]:25}  in={r['input_tokens']:>6,}  out={r['output_tokens']:>4}  £{r['cost_gbp']:.4f}  cls={cls}")

    (SPIKE_DIR / "runs_v2_methods.json").write_text(json.dumps(runs, indent=2))

    print("\n=== METHODS EXTRACTION DETAIL ===")
    for r in runs:
        res = r["result"]
        if not isinstance(res, dict) or "_parse_error" in res:
            continue
        po = res.get("primary_outputs", {})
        sec = res.get("secondary_resources_used", {})
        av = res.get("availability_signals", {})
        print(f"\n--- {r['field']} | {r['model'].split('-2')[0]} | PMID {r['pmid']} ---")
        print(f"  classification: {res.get('classification')}")
        print(f"  primary_data           : {po.get('primary_data', {})}")
        print(f"  primary_software       : {po.get('primary_software', {})}")
        print(f"  primary_comp_methods   : {po.get('primary_computational_methods', {})}")
        print(f"  data_sources used      : {sec.get('data_sources', [])[:5]}")
        print(f"  software_used          : {sec.get('software_used', [])[:5]}")
        print(f"  methods_used           : {sec.get('methods_used', [])[:8]}")
        print(f"  methods_availability   : {av.get('methods_availability', '')[:120]}")
        print(f"  novelty_phrases        : {res.get('novelty_phrases', [])[:5]}")
        print(f"  confidence             : {res.get('confidence')}")


if __name__ == "__main__":
    main()
