# Spike 0 — LLM cost and schema on 30 UK papers (May 2026)

A pre-award spike to find out what it costs to classify a full paper with frontier-class models, and whether a methods-aware schema extracts what WP1 needs. It is **calibration run 0** in the plan: the 1k → 3k → 10k calibration runs that come next are described in the [WP1 high-level design](../../docs/design/wp1/pif-wp1-hld.pdf).

**Author:** Daren Howell (Crew Create) · **Run:** 11 May 2026 · **Spend:** £2.71 (90 API calls)

## What was run

| Round | Papers | What it tested |
|---|---|---|
| 1 | 10 PubMed papers (8 abstract-only, 2 PMC full text) | First cost read. Full text was cut off at 30k characters, so these costs are too low |
| 2 | The same 2 full-text papers, untruncated | Real full-text cost; a methods-aware schema that separates new data, new software and new computational methods |
| 3 | 30 open-access UK papers from 2023, one per field | 3 methods × 30 papers = 90 calls: Sonnet 4.6 full text, Haiku 4.5 full text, Haiku 4.5 abstract only |

## What it found

| Method | Avg input tokens | £ per paper | Projected for 300k papers |
|---|---:|---:|---:|
| Sonnet 4.6, full text | 23,195 | £0.065 | £19.4k |
| Haiku 4.5, full text | 23,194 | £0.022 | £6.7k |
| Haiku 4.5, abstract only | 761 | £0.003 | £1.0k |

- **Choice of model moves the answer.** Haiku and Sonnet on full text agree on 18 of 30 papers (60%, κ 0.40). Sonnet calls far more papers primary or mixed.
- **Abstract vs full text matters less than the model.** Leaving out parse failures, Haiku on the abstract agrees with Haiku on the full text 89% of the time (κ 0.78), and on every paper for the simple primary-vs-secondary split.
- **Long inputs break Haiku's JSON.** 3 of 30 parse failures, all on papers over 100k characters.
- **The methods-aware schema works** across fields for 1–10% extra cost per call.

The full write-up, with budget scenarios and the adopter-scale argument, is [`proposal-llm-budget.md`](proposal-llm-budget.md). Read its errata box first.

## Files

| File | What it is |
|---|---|
| `proposal-llm-budget.md` | Write-up: results, budget scenarios, stratification, adopter cost envelope |
| `spike_30_papers.csv`, `spike_30_summary.csv`, `summary_30.json`, `appendix_table_30.md` | Round 3 results per paper and per method |
| `runs*.json` | Raw run records: model, tokens, cost, latency, parsed output |
| `papers_manifest.csv` | The 38 papers used — IDs, field, licence, size, and which round used them |
| `fetch_corpus.py` | Rebuilds the exact corpus from the manifest (tested: all 30 full texts match) |
| `fetch_papers.py`, `fetch_more_pmc*.py`, `fetch_to_30.py` | The original discovery scripts, kept for the record; re-running them may pick different papers |
| `classify*.py`, `build_*.py` | The classification runs and table builders |
| `allocate.py`, `data/` | Stratified allocation of 300k calls across 26 OpenAlex fields × 10 years (OpenAlex snapshot, 11 May 2026, CC0) |
| `sample_size_analysis.py` | Power calculations for detecting year-on-year shifts per cell |

## What is not in the repo, and why

The paper texts (`papers/*.json`) are not committed. Their licences are mixed: 27 CC BY, 1 CC BY-NC, 1 CC BY-NC-ND, 1 with no licence stated, and 8 records that are publisher-copyright abstracts. Run `fetch_corpus.py` to rebuild them locally. The per-paper results do quote short excerpts — availability statements and novelty phrases — as evidence for each classification.

## Reproduce

```bash
cd spikes/2026-05-llm-cost
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python fetch_corpus.py        # rebuild papers/ (~1 minute)
.venv/bin/python classify_30_scaling.py # 90 calls, ~15 minutes, ~£2.71 at May 2026 prices
.venv/bin/python build_appendix_30.py
.venv/bin/python allocate.py
```

The classify scripts read an Anthropic API key from the macOS keychain (`ANTHROPIC_API_KEY`); adapt that line for other platforms.
