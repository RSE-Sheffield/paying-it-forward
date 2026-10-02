"""Classify 30 full-text papers across 3 scaling methods:
  - Haiku 4.5 + full text  (Mode 2 candidate)
  - Haiku 4.5 + abstract only  (Mode 2 cheaper scaling alternative)
  - Sonnet 4.6 + full text  (Modes 1, 3, 4)

Then compute inter-method classification agreement (κ).
"""
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

SPIKE_DIR = Path(__file__).parent
PAPERS = json.loads((SPIKE_DIR / "papers" / "papers_fulltext_30.json").read_text())

KEY = subprocess.run(
    ["security", "find-generic-password", "-a", os.environ["USER"], "-s", "ANTHROPIC_API_KEY", "-w"],
    capture_output=True, text=True, check=True
).stdout.strip()

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


def make_client():
    return anthropic.Anthropic(api_key=KEY)


def classify_one(paper, model, use_fulltext):
    """One classification call."""
    client = make_client()
    title = paper.get("title") or ""
    abstract = paper.get("abstract") or ""
    fulltext = paper.get("fulltext") or ""

    if use_fulltext and fulltext:
        body = f"ABSTRACT:\n{abstract}\n\nFULL TEXT:\n{fulltext}"
    else:
        body = f"ABSTRACT:\n{abstract}"

    user_msg = f"TITLE: {title}\n\nJOURNAL: {paper.get('journal','')}  YEAR: {paper.get('year','')}\n\n{body}\n"

    t0 = time.time()
    try:
        resp = client.messages.create(
            model=model, max_tokens=1200, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as e:
        return {"model": model, "method": "FT" if use_fulltext else "ABS", "pmid": paper.get("pmid"), "pmcid": paper.get("pmcid"), "error": str(e)}
    elapsed = time.time() - t0
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"): text = text[4:]
    try:
        result = json.loads(text.strip())
    except Exception as e:
        result = {"_parse_error": str(e), "_raw": text[:200]}

    usage = resp.usage
    in_price, out_price = PRICING_USD_PER_MTOK[model]
    cost_usd = (usage.input_tokens / 1e6) * in_price + (usage.output_tokens / 1e6) * out_price
    return {
        "pmid": paper.get("pmid"),
        "pmcid": paper.get("pmcid"),
        "field": paper.get("field"),
        "model": model,
        "method": "FT" if use_fulltext else "ABS",
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_gbp": round(cost_usd * USD_TO_GBP, 5),
        "latency_s": round(elapsed, 1),
        "classification": result.get("classification", "PARSE_ERR") if isinstance(result, dict) else "PARSE_ERR",
        "result": result,
    }


def cohen_kappa(a, b):
    """Cohen's κ for two label lists."""
    assert len(a) == len(b)
    n = len(a)
    if n == 0: return 0.0
    labels = sorted(set(a) | set(b))
    obs_agree = sum(1 for x, y in zip(a, b) if x == y) / n
    # Expected agreement
    exp_agree = 0
    for L in labels:
        pa = a.count(L) / n
        pb = b.count(L) / n
        exp_agree += pa * pb
    if exp_agree >= 1.0: return 1.0
    return (obs_agree - exp_agree) / (1 - exp_agree)


def main():
    HAIKU = "claude-haiku-4-5-20251001"
    SONNET = "claude-sonnet-4-6"

    # Build task list: 3 methods × 30 papers
    tasks = []
    for p in PAPERS:
        tasks.append((p, HAIKU, True, "Haiku-FT"))
        tasks.append((p, HAIKU, False, "Haiku-ABS"))
        tasks.append((p, SONNET, True, "Sonnet-FT"))

    print(f"Running {len(tasks)} classifications (30 papers × 3 methods) with 6-way concurrency...\n")
    results = []
    completed = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(classify_one, p, m, ft): label for p, m, ft, label in tasks}
        for fut in as_completed(futures):
            label = futures[fut]
            try:
                r = fut.result()
                if "error" in r:
                    print(f"  ERR {label} {r.get('pmid')}: {r['error'][:80]}")
                else:
                    results.append(r)
                    completed += 1
                    if completed % 10 == 0:
                        print(f"  [{completed:>3}/{len(tasks)}] running cost so far: £{sum(x['cost_gbp'] for x in results):.3f}")
            except Exception as e:
                print(f"  EXCEPTION {label}: {e}")

    (SPIKE_DIR / "runs_30_scaling.json").write_text(json.dumps(results, indent=2))

    # Aggregate by method
    print("\n\n=== PER-METHOD STATS (N=30 papers) ===\n")
    methods = ["Haiku-FT", "Haiku-ABS", "Sonnet-FT"]
    method_results = {m: [] for m in methods}
    for r in results:
        if r["model"] == HAIKU and r["method"] == "FT": method_results["Haiku-FT"].append(r)
        if r["model"] == HAIKU and r["method"] == "ABS": method_results["Haiku-ABS"].append(r)
        if r["model"] == SONNET and r["method"] == "FT": method_results["Sonnet-FT"].append(r)

    print(f"{'Method':<12} {'n':>3}  {'avg in':>8} {'avg out':>8}  {'£/call':>8}  {'med £':>7}  {'min £':>7}  {'max £':>7}  {'300k @ £':>10}")
    print("-" * 90)
    summary = {}
    for m in methods:
        rs = method_results[m]
        if not rs: continue
        n = len(rs)
        avg_in = sum(r["input_tokens"] for r in rs) / n
        avg_out = sum(r["output_tokens"] for r in rs) / n
        avg_cost = sum(r["cost_gbp"] for r in rs) / n
        costs_sorted = sorted(r["cost_gbp"] for r in rs)
        med = costs_sorted[len(costs_sorted)//2]
        mn = min(r["cost_gbp"] for r in rs)
        mx = max(r["cost_gbp"] for r in rs)
        proj = avg_cost * 300_000
        summary[m] = {"n": n, "avg_in": avg_in, "avg_out": avg_out, "avg_cost": avg_cost, "median": med, "min": mn, "max": mx, "proj_300k": proj}
        print(f"{m:<12} {n:>3}  {avg_in:>8,.0f} {avg_out:>8,.0f}  £{avg_cost:>7.4f}  £{med:>6.4f}  £{mn:>6.4f}  £{mx:>6.4f}  £{proj:>9,.0f}")

    # Inter-method classification agreement
    print("\n\n=== INTER-METHOD CLASSIFICATION AGREEMENT (N=30) ===\n")
    # Build classification arrays indexed by pmid for each method
    by_pmid = {m: {} for m in methods}
    for r in results:
        key = (r["model"] == HAIKU, r["method"])
        label = {(True, "FT"): "Haiku-FT", (True, "ABS"): "Haiku-ABS", (False, "FT"): "Sonnet-FT"}.get(key)
        if label:
            by_pmid[label][r["pmid"]] = r["classification"]

    common_pmids = set(by_pmid["Haiku-FT"].keys()) & set(by_pmid["Haiku-ABS"].keys()) & set(by_pmid["Sonnet-FT"].keys())
    print(f"Common papers across all 3 methods: {len(common_pmids)}")
    for m1 in methods:
        for m2 in methods:
            if m1 >= m2: continue
            a = [by_pmid[m1][pmid] for pmid in common_pmids]
            b = [by_pmid[m2][pmid] for pmid in common_pmids]
            agree = sum(1 for x, y in zip(a, b) if x == y)
            kappa = cohen_kappa(a, b)
            print(f"  {m1:<12} vs {m2:<12}: agree {agree}/{len(common_pmids)} ({agree/len(common_pmids)*100:.0f}%)  κ={kappa:.3f}")

    # Distribution of classifications by method
    print("\n=== CLASSIFICATION DISTRIBUTION ===\n")
    for m in methods:
        labels = list(by_pmid[m].values())
        if not labels: continue
        counts = {l: labels.count(l) for l in set(labels)}
        print(f"  {m:<12}: {counts}")

    # Disagreements detail
    print("\n=== DISAGREEMENTS — Haiku-FT vs Haiku-ABS (the scaling-method comparison) ===\n")
    disagreements = []
    for pmid in sorted(common_pmids):
        h_ft = by_pmid["Haiku-FT"][pmid]
        h_abs = by_pmid["Haiku-ABS"][pmid]
        if h_ft != h_abs:
            disagreements.append((pmid, h_ft, h_abs))
            paper = next((p for p in PAPERS if p.get("pmid") == pmid), None)
            field = paper.get("field") if paper else "?"
            title = (paper.get("title") or "")[:60] if paper else ""
            print(f"  PMID {pmid:>10}  {field:>18}  Haiku-FT={h_ft:<10} Haiku-ABS={h_abs:<10}  | {title}")
    print(f"\n  Total Haiku-FT/ABS disagreements: {len(disagreements)}/{len(common_pmids)}")

    # Save summary
    (SPIKE_DIR / "summary_30.json").write_text(json.dumps({
        "summary": summary,
        "agreement_pairs": {
            f"{m1}_vs_{m2}": {
                "agree": sum(1 for pmid in common_pmids if by_pmid[m1][pmid] == by_pmid[m2][pmid]),
                "total": len(common_pmids),
                "kappa": cohen_kappa([by_pmid[m1][pmid] for pmid in common_pmids], [by_pmid[m2][pmid] for pmid in common_pmids])
            } for m1 in methods for m2 in methods if m1 < m2
        },
        "disagreements_haiku_ft_vs_abs": [{"pmid": pmid, "haiku_ft": ft, "haiku_abs": abs_} for pmid, ft, abs_ in disagreements],
    }, indent=2))


if __name__ == "__main__":
    main()
