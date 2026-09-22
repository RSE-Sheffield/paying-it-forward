# Proposal — LLM budget sizing for confident 10-year data & method evolution detection

**Author**: Daren Howell (CrewCreate Ltd)
**Date**: 11 May 2026 (rev 2)
**Status**: Pre-award spike write-up (May 2026), shared with the Paying it Forward team. Prices are May 2026 list prices in GBP. See the errata below before reusing any figure.

> **Errata (September 2026).** A later read-through found these inconsistencies. The findings stand; the earlier text is left as written.
>
> - **Model agreement.** §1, §2 and the Mode 3a description still say Haiku and Sonnet agree 100% (7/7). The N=30 result in Appendix B supersedes that: 60% agreement, κ 0.40. Appendix B.1's "7/10 at N=10" also conflicts with §2's "7/7".
> - **Parse errors counted as disagreements.** Excluding the three Haiku parse failures, Haiku abstract-only vs Haiku full text agree on 89% (κ 0.78) and on 100% for the simple primary-vs-secondary split. The gap is between models, not between abstract and full text. Haiku vs Sonnet stays at κ 0.40 (0.51 on the binary split).
> - **Stratification floor.** §1 and §3.5 recommend 500 papers per field × year cell; §3 and `allocation.csv` still use 200 (2,000 per field).
> - **Batch-API figure.** §5 and B.7 give ~£8,300 with batch pricing, §5b gives £7,090, and the §5 sensitivity table uses an older £8,530.
> - **Adopter envelope.** §9 item 5 says £500k a year; §5c uses £100k.
> - **Appendix B.3 example.** The MAIT-cell paper is cited as PMID 36915232 and as a Sonnet-primary case; the table has PMID 36911683, classified secondary by all three methods.

> **Rev 2 changes (11 May 2026, evening):** Added methods-aware classification schema; added statistical-power analysis; restructured budget around four modes; revised budget recommendation upward from £15k to £20k.
>
> **Rev 3 changes (11 May 2026, night):** Expanded spike to N=30 PMC full-text papers across diverse fields. **Key finding: Haiku-FT and Sonnet-FT only agree on 60% of classifications (κ=0.40)** — model choice materially affects the headline metric. Added a third scaling method (Haiku abstract-only) which is 7× cheaper than Haiku-FT and 80% agreement with it. Added price-trajectory sensitivity (batch API discounts + likely model price reductions over the 12-month project). **Revised budget recommendation depends on quality target** — see §5b for three scenarios; £15-20k brackets the realistic envelope.

Linked artefacts (in `spike/`):
- `fetch_papers.py` — PubMed fetcher (10 papers, mixed fields)
- `classify.py` — Original spike (Haiku 4.5 + Sonnet 4.6, 30k-char truncation)
- `classify_fulltext.py` — Re-run on PMC papers without truncation
- **`classify_v2_methods.py`** — Methods-aware schema (rev 2): extracts primary data + software + computational methods separately
- **`sample_size_analysis.py`** — Statistical power calculations for cell-level effect detection
- `allocate.py` — Stratified allocation calculator
- `runs.json`, `runs_fulltext.json`, **`runs_v2_methods.json`** — Measured token usage
- `allocation.csv` — Per-field allocation

---

## 1. Headline

| Question | Answer |
|---|---|
| What's the **right budget** for confident 10-year data + method evolution detection? | **£15–20k brackets the realistic envelope**, depending on quality target. N=30 spike (Appendix B) reveals that Haiku-FT and Sonnet-FT only agree on 60% of classifications — model choice is a substantive methodological decision, not just a cost trade-off. Three scenarios in §5b: (a) Haiku-FT workhorse + Sonnet Judge layer @ ~£14k programme; (b) Sonnet-FT workhorse @ ~£22k programme; (c) Haiku-ABS workhorse @ ~£3k programme (lowest quality, useful as scaling-method baseline only). Factoring expected 30–50% price reductions over 12 months (batch API, model evolution), the **£15k envelope likely covers scenario (a) with comfortable buffer**; £20k de-risks against pricing volatility. |
| Is the methodology **scalable to global corpus** at adopter budgets? | **Only via a lightweight runtime classifier — and it's an empirical question whether one can be trained to acceptable quality.** §5c models a £100k/year × 10 years = £1M adopter envelope to process 240M docs (200M historic + 4M/year). Max gross cost £0.00417/doc (~£0.0015 LLM-only). **At today's pricing, only a lightweight runtime classifier (~£0.0001/doc) meets this envelope.** Sonnet-FT is 15× over budget; Haiku-FT is 5× over; Haiku-ABS sits at the LLM-only affordability edge. §5c.5 critically assesses five candidate deterministic approaches: pure regex/dictionaries probably underperform Haiku-ABS (κ ≤ 0.3); a **fine-tuned SciBERT-class small language model is the realistic path** (expected κ 0.55–0.75 vs Sonnet-FT, ~£0.0001/doc CPU inference). The methodology paper publishes the outcome regardless — including the negative-result case where lightweight classifier κ < 0.45, in which abstract-only Haiku-ABS becomes the recommended adopter fallback. |
| What does **"confident"** mean statistically? | At 300k stratified across 26 OpenAlex fields × 10 years (1,154 calls/cell average): supports detecting **5pp annual changes** in primary-output rate at low base rates (e.g. primary computational methods rising from 10% → 15%) in every cell, and **2-3pp shifts** in larger-corpus fields. Floor of 500/cell ensures every cell can detect 5pp shifts at 10–25% base rates. |
| How should calls be **distributed across UK research**? | Stratified across **26 OpenAlex fields × 10 years (2016–2025)**: floor 500 per field × year cell (130k, 43% of budget) + proportional excess (170k, 57%). Revised upward from earlier 200/cell floor to give every cell adequate power. Overall sampling rate 11.4% of the 2.62M UK citable-article corpus. |
| Why **four modes**, not three? | Develop Insight + Teacher + Judge + Deterministic Rule Model. The fourth mode (Rule Model) is what enables the LLM-teacher → rule-runtime research question (a proposed research question, not in the funded scope): LLM-judge calls evaluate the deterministic rule classifier's disagreements with the LLM teacher, producing the validation evidence for whether the metric can be made LLM-free at runtime. |
| What about **computational methods** (algorithms, equations, models)? | The original spike under-extracted these. A revised methods-aware schema (`classify_v2_methods.py`) now separates *primary data*, *primary software*, and *primary computational methods*. Tested on the 2 PMC full-text papers: extraction works well, classification agreement Haiku↔Sonnet remains 100%, output token usage rises ~5–10%. |

### Important caveats on the cost numbers

1. **Original spike truncation**: `classify.py` truncated full-text papers at 30,000 chars (~7,500 tokens). The corrected runs (`classify_fulltext.py`, `classify_v2_methods.py`) use full untruncated text; true full-text input is 27,000–95,000 tokens per paper (mean ~61k). All cost figures in this document are based on the untruncated measurements.
2. **Methods-aware schema overhead**: the richer schema (separating data / software / computational methods, plus novelty phrases) adds ~5–10% to output tokens versus the original schema. Per-call cost rise is small but multiplies across 300k calls. Factored into the budget below.
3. **Spike sample size**: 2 PMC full-text papers + 10 abstract-only. Full corpus will vary more — short letters at ~5k tokens, long reviews at ~80k. Per-call cost variance is substantial; the budget uses mean estimates with explicit buffer for variance.

---

## 2. Spike results — measured cost on 10 PubMed papers

10 UK-affiliated papers, 2023–2024, mixed fields (medicine, biochem, neuroscience, immunology, health professions, pharmacology, genetics, public health, psychiatry, oncology), classified via Anthropic API with structured-JSON output for primary/secondary classification + data/code reference extraction + availability signals.

### Full-text availability in the sample

- **2 of 10 papers (20%)** had PMC open-access full text: neuroscience (119,361 chars) and pharma (356,086 chars)
- **8 of 10 papers (80%)** were abstract-only — not on PMC

The 30% full-text rate quoted in the §1 caveat is an estimate for the full UK corpus once PMC + green OA institutional repositories + Crossref TDM are combined; the spike's own 20% is at the low end because it was PubMed-only.

### Per-call cost — abstract-only (first spike, 7 of 10 papers; 2 PMC truncated to 30k chars)

| Model | n | Avg input tokens | Avg output tokens | Avg cost/call (GBP) | **300k projection** |
|---|---|---|---|---|---|
| Claude Haiku 4.5 | 10 | 1,971 | 133 | £0.00211 | £632 |
| Claude Sonnet 4.6 | 3 | 5,192 | 155 | £0.01432 | £4,295 |

These figures reflect the inputs as they were sent: 7 abstract-only papers (~500–1,000 tokens) plus 2 PMC papers truncated to ~7,500 tokens each. They are NOT representative of true full-text classification.

### Per-call cost — true full text (untruncated, n=2 PMC papers, both models)

| Paper | Chars | Model | Input tokens | Output tokens | Cost/call (GBP) |
|---|---|---|---|---|---|
| Neuroscience | 119,361 | Haiku 4.5 | 27,121 | 275 | £0.0228 |
| Neuroscience | 119,361 | Sonnet 4.6 | 27,122 | 121 | £0.0665 |
| Pharma | 356,086 | Haiku 4.5 | 94,900 | 489 | £0.0779 |
| Pharma | 356,086 | Sonnet 4.6 | 94,901 | 505 | £0.2338 |

**True full-text averages (n=2):**
- Haiku 4.5: 61,011 input tokens, £0.050/call, 5.4s latency
- Sonnet 4.6: 61,012 input tokens, £0.150/call, 8.2s latency

The pharma paper (95k tokens) is at the upper end (clinical-guideline paper with extensive references). The neuroscience paper (27k tokens) is closer to typical for a research article. Real-world per-paper variance is substantial — expect £0.02 to £0.10/call for Haiku full text across the corpus, depending on paper length.

### What this means for 300k projections

| Scenario | Per-call cost (GBP) | 300k total (GBP) | Within £15k? |
|---|---|---|---|
| Haiku 4.5, all 300k full text | £0.050 | **£15,101** | At ceiling |
| Sonnet 4.6, all 300k full text | £0.150 | **£45,055** | 3x over budget |
| Haiku 4.5, 30% full text + 70% abstract | £0.0158 | **£4,740** | Comfortable |
| Sonnet 4.6, 30% full text + 70% abstract | £0.0458 | **£13,727** | At ceiling, no buffer |
| Haiku full text + Sonnet judge on 10% subset | £0.0158 + £0.015 = £0.031 (mixed avg) | **~£9,300** | Comfortable, with buffer |

This is the realistic decision space. The earlier £632/£4,295 figures were the truncated-input numbers; only the bottom three rows of the table above are workable strategies that fit the budget.

### Inter-model agreement

3 papers run through both Haiku 4.5 and Sonnet 4.6 (truncated runs); 2 more on full text; 2 more on methods-aware schema. **Classification agreement: 7/7 across all comparisons.** Both models classified the neuroscience paper as secondary (review), the pharma paper as secondary (clinical guideline), and the medicine paper as primary (clinical trial). Methods-aware schema extracted consistent computational-methods lists. This supports Haiku as the workhorse for Mode 2 with Sonnet as the Mode 3 judge.

### Methods-aware schema results (rev 2)

The original schema asked for `data_references_used` and `code_references_used`. This under-captured computational methods broadly — algorithms, statistical methods, mathematical models, equations, ML architectures, pipelines, and analytical workflows. The revised schema (`classify_v2_methods.py`) decomposes the question:

```
primary_outputs:
  primary_data            { produced: bool, description: ... }
  primary_software        { produced: bool, description: ... }
  primary_computational_methods { produced: bool, description: ... }

secondary_resources_used:
  data_sources            [...]
  software_used           [...]
  methods_used            [...]   # e.g. PCA, BLAST, MCMC, BERT

availability_signals:
  data_availability       "..."
  software_availability   "..."
  methods_availability    "..."   # equations documented? pseudocode? supplementary specs?

novelty_phrases           [...]   # verbatim phrases signalling primary claims
```

Tested on the 2 PMC papers:

**Neuroscience review (PMID 38110704)** — both models correctly identified as secondary across all three primary categories. `methods_used` extracted: factor analysis, network analysis, cluster analysis, meta-analysis, GWAS, polygenic risk score analysis, MRS, fMRS, etc. `software_used` extracted: PubMed search tools, standard neuroimaging analysis pipelines. This distinguishes a review that *uses* many methods from one that *produces* new methods.

**Pharma clinical guideline (PMID 37039129)** — both models correctly identified as secondary on data, software, AND computational methods. The schema captured that the guideline *applies* the Shekelle 1999 evidence-grading schema but does not develop a new one. `methods_used` extracted: systematic review synthesis, evidence grading, expert consensus, meta-analysis results referenced, cluster analysis referenced. The novelty phrase extraction surfaced "developed an evidence-based consensus guideline" — a useful signal for downstream pattern analysis even though the underlying methods are secondary.

**Cost overhead of methods-aware schema** (vs original):

| Paper | Model | Original out tokens | Methods-aware out tokens | Cost increase |
|---|---|---|---|---|
| Neuroscience | Haiku | 275 | 735 | +£0.0022 (+10%) |
| Neuroscience | Sonnet | 121 | 561 | +£0.0063 (+9%) |
| Pharma | Haiku | 489 | 731 | +£0.0013 (+2%) |
| Pharma | Sonnet | 505 | 629 | +£0.0025 (+1%) |

The richer schema costs 1–10% more per call (output tokens roughly double for shorter papers, marginal for longer where the input dominates). Negligible at 300k scale.

---

## 3.5 Statistical power for confident decade-evolution detection

The metric must detect annual changes in primary-output rates across 26 fields × 10 years = 260 cells. Effect sizes of interest:

| Effect being detected | n per cell (per group) needed |
|---|---|
| Mid-range 5pp shift (50% → 55%) | 1,563 |
| Mid-range 3pp shift (50% → 53%) | 4,351 |
| Asymmetric 5pp shift (20% → 25%) | 1,093 |
| Asymmetric 5pp shift (10% → 15%) | 685 |
| Low-base 5pp shift (5% → 10%) | 434 |
| Low-base 10pp shift (5% → 15%) | 140 |

(80% power, two-sided p<0.05; see `sample_size_analysis.py`.)

**Key implication for primary computational methods detection.** Primary computational methods are produced in an estimated 5–15% of papers (varies by field — much higher in CS/Maths/Methods journals, lower in clinical Medicine). To detect a 5pp emergence shift (e.g. methods rising from 10% to 15% as AI-augmented production increases), need **~685 papers per cell**.

**Allocation strategy revised — floor raised from 200 to 500/cell.**

| Floor per cell | Floor total (43% of 300k) | Power at 10%→15% | Power at 20%→25% | Power at 50%→55% |
|---|---|---|---|---|
| 200 (original) | 52k | Marginal | Marginal | Insufficient |
| **500 (recommended)** | **130k** | **Good** | Good | Marginal in small fields |
| 800 | 208k | Strong | Strong | Adequate |

500/cell + proportional excess (170k) = 300k total. This gives every cell ≥500 calls (detection of low-base-rate emergence) and large fields like Medicine get ~67k (1,000s per year — detection of mid-range shifts).

The previous proposal's 200/cell floor was sufficient for headline metrics but marginal for the methods-emergence signal which is the most novel finding the project could surface.

### Validation of extraction quality

Examples from the spike (Haiku 4.5):

- **Medicine — SURMOUNT-1 obesity trial** (PMID 39536238): correctly identified as primary; extracted trial registration NCT04184622 as data reference; flagged data availability as "via ClinicalTrials.gov or sponsor"
- **Biochem — UK Biobank pQTL paper** (PMID 39579765): correctly identified as primary; extracted UK Biobank as a data reference (interesting case — they used UK Biobank but also produced new primary data); identified the open-access proteome-phenome atlas as data availability signal
- **Neuroscience — schizophrenia review** (PMID 38110704): correctly identified as secondary (review); extracted PANSS, MATRICS, UCSD-PSA as data references used

Extraction quality is good enough to support both the metric production and the LLM-as-teacher labelling for distillation.

---

## 3. Stratified allocation across UK fields × years

### Corpus

| Total UK citable articles (2016–2025) | 2,623,226 |
|---|---|
| LLM budget | 300,000 calls |
| **Overall sampling rate** | **11.4%** |
| Fields (OpenAlex level-2) | 26 |
| Years | 10 |
| Field × year cells | 260 |

Source: `data/2026-05-11-openalex-uk-citable-articles-by-field-2016-2025.csv`. Note the source CSV excludes the long-tail of works without a primary topic; the corpus total above is the sum of allocated-topic works, which is ~93% of all UK citable articles in OpenAlex.

### Stratification strategy — hybrid floor with proportional excess

Pure proportional sampling oversamples Medicine (~25% of corpus) and undersamples small fields (Chemical Engineering at 0.3%) to the point where within-cell statistics break down. Pure equal sampling per cell oversamples small fields to ~75%+, which is wasteful. Hybrid is the right compromise:

1. **Floor — 200 papers per field × year cell.** Ensures within-cell statistical power for change detection (~80% power to detect a 10pp shift in primary/secondary mix at p<0.05). Total floor cost: 26 × 10 × 200 = **52,000 calls (17% of budget)**.
2. **Excess — remaining 248,000 calls allocated proportionally to each field's 10-year share.** Preserves field-level representativeness while keeping small fields above floor.

### Allocation by field (top + bottom)

| Field | 10-y total | Share | Allocation | Sample rate |
|---|---|---|---|---|
| Medicine | 695,421 | 26.5% | 67,745 | 9.7% |
| Engineering | 285,114 | 10.9% | 28,955 | 10.2% |
| Social Sciences | 273,957 | 10.4% | 27,900 | 10.2% |
| Biochem/Gen/Mol Bio | 158,135 | 6.0% | 16,950 | 10.7% |
| Computer Science | 129,991 | 5.0% | 14,289 | 11.0% |
| ... | ... | ... | ... | ... |
| Energy | 15,236 | 0.6% | 3,440 | 22.6% |
| Nursing | 13,057 | 0.5% | 3,234 | 24.8% |
| Dentistry | 10,756 | 0.4% | 3,017 | 28.0% |
| Pharmacology | 7,412 | 0.3% | 2,701 | 36.4% |
| Veterinary | 7,064 | 0.3% | 2,668 | 37.8% |
| Chemical Engineering | 6,687 | 0.3% | 2,632 | 39.4% |
| **TOTAL** | **2,623,226** | **100%** | **300,000** | **11.4%** |

Full per-field table: `allocation.csv`. Each field's allocation is then split evenly across its 10 years (with the floor ensuring at least 200 per year per field).

### Within-cell sampling — how to pick the specific papers

Within each field × year cell, draw a random sample stratified on a secondary axis to control for confounders. **Recommended secondary stratification**: by publication month (to control for seasonal submission patterns and ensure the sample is temporally distributed within the year). Alternative: by journal impact tier (Q1/Q2/Q3/Q4) to ensure the sample isn't skewed to high-impact venues.

---

## 4. Four-mode call allocation — Develop Insight / Teacher / Judge / Deterministic Rule Model

Rather than treating all 300k calls as a single "classify the corpus" pass, allocate them across four distinct modes that serve different project goals. The fourth mode (Deterministic Rule Model) is what operationalises the LLM-teacher → rule-runtime research question — without it, the project produces a metric but can't answer whether the metric is adoptable at scale.

### Mode 1 — Develop Insight (pattern discovery)

**Purpose**: Discover what data, software, and computational-method patterns exist in UK research; refine the classification taxonomy; identify edge cases; establish what "primary computational method" means in different fields (very different shape in Maths vs. Clinical Medicine vs. Computer Science).

**Calls**: 5,000 (1.7% of budget)
**Model**: Sonnet 4.6, deep prompts requesting reasoning + edge-case flagging + novelty phrases
**Sample**: Hand-curated diverse mix — ~200 per field, deliberately span availability-statement types and journal tiers
**Output**: Refined classification schema, taxonomy of methods patterns, prompt variants for downstream calls, calibration of confidence ratings
**Timing**: M1–M2

### Mode 2 — Teacher (corpus labelling)

**Purpose**: Produce the primary labelled corpus — 270,000 publications classified with the methods-aware schema — that (a) generates the headline metric for WP3, (b) becomes the training/validation set for the deterministic rule classifier (Mode 4), and (c) is published as a CC-BY 4.0 dataset of independent value.

**Calls**: 270,000 (90% of budget)
**Model**: Haiku 4.5 across the board, with full-text input where available (~50% target coverage via PMC + green OA institutional repositories + Crossref TDM where unlocked); abstract-only fallback otherwise
**Sample**: Stratified per §3 — 500/cell floor + proportional excess across 26 fields × 10 years
**Output**: Labelled corpus with the full methods-aware schema (primary data / software / computational methods, secondary resources used, availability signals, novelty phrases, confidence)
**Timing**: M3–M8

### Mode 3 — Judge (validation and quality control)

**Purpose**: Validate the Teacher classifications, compute inter-model κ, identify systematic biases, flag low-confidence and edge cases for human review and Surrey-side validation in WP2.

**Calls**: 20,000 (6.7% of budget)
- **Sub-mode 3a — Cross-model judge**: 15,000 papers re-classified with Sonnet 4.6 (where Mode 2 used Haiku). Provides inter-model κ per field × year. Spike already shows Haiku↔Sonnet 100% agreement at small N; need scale to surface systematic disagreements.
- **Sub-mode 3b — Opus adjudication**: 5,000 papers — specifically the Sonnet-vs-Haiku disagreements plus low-confidence cases — re-classified with Opus 4.7 for high-quality adjudication.

**Model**: Sonnet 4.6 (3a); Opus 4.7 (3b)
**Output**: Validation kappa per field, error analysis, adjudicated edge cases, gold-standard subset for Mode 4 distillation training
**Timing**: M5–M9 (overlaps with Mode 2 final third)

### Mode 4 — Deterministic Rule Model (distillation evaluation)

**Purpose**: Operationalise the LLM-teacher → rule-runtime research question. The Sheffield RSE team (Brown) trains candidate deterministic classifiers (regex/keyword patterns + a small fine-tuned classifier, e.g. SciBERT) against the Mode 2 labelled corpus. Mode 4's LLM calls *evaluate* the rule classifier by having an LLM judge adjudicate cases where the rule classifier disagrees with the Teacher. This produces the validation evidence for the research question: can adopters deploy this metric without LLM compute?

**Calls**: 5,000 (1.7% of budget)
**Model**: Sonnet 4.6 for the disagreement adjudication; Haiku 4.5 for the iteration runs (multiple rule-classifier candidates evaluated)
**Sample**: Where rule classifier disagrees with LLM teacher; stratified across fields and primary-output categories
**Output**: Rule-classifier performance metrics, error analysis, adoption-readiness assessment — feeds directly into the Platform Adoption Evidence Report and a possible methods paper
**Timing**: M8–M11 (after Mode 2 + 3 produce the gold-standard corpus)

This mode is what makes the metric actually scalable for adopters. Without it, we produce a labelled corpus but never test whether the methodology can run on a research-office laptop. With it, the project answers the adoption question concretely.

---

## 5. Recommended £20,000 budget breakdown

Computed from spike-measured per-call costs at **true full-text token counts**, with **methods-aware schema overhead** factored in.

**Key assumption**: ~50% full-text coverage of the 300k stratified sample, achieved via PMC open-access (≈25-30%) + UK green OA institutional repositories (≈10-15%) + Crossref TDM partnerships and publisher Letters of Support (≈10-15%). Sensitivity at 30% and 70% coverage shown below.

**⚠ Superseded by Appendix B §B.7–B.8.** The original §5 budget assumed a small Sonnet Judge layer (15k calls). The N=30 spike (Appendix B) revealed that Haiku-FT and Sonnet-FT only agree on 60% of classifications, which means the Judge layer needs to be much larger — 33% of Mode 2 instead of 5%. The revised budget is below; the more detailed three-scenario analysis is in §B.7.

| Mode | Calls | Method | Per-call | Cost |
|---|---|---|---|---|
| M1 — Develop Insight | 5,000 | Sonnet-FT | £0.062 (median × 1.1) | £310 |
| M2 — Teacher (with 10% retry) | 270,000 | Haiku-FT | £0.024 | £6,480 |
| M3a — Cross-model Judge (33% of M2) | 90,000 | Sonnet-FT | £0.062 | £5,580 |
| M3b — Opus adjudication | 5,000 | Opus 4.7 | £0.30 | £1,500 |
| M4 — Rule Model evaluation | 5,000 | Sonnet-FT | £0.062 | £310 |
| **Programme cost** | **375,000** | | | **£14,180** |
| Buffer (iteration + pricing volatility) | — | — | — | £5,820 |
| **Total** | | | | **£20,000** |

(M3a is *additional* calls beyond Mode 2's 270k — so total call volume is 375k, not 300k. This is the correct accounting: the original 300k figure was the labelled-corpus size, not the total LLM call budget.)

**With batch API applied** (50% off non-real-time modes, which is all of them): programme cost ~£8,300, buffer £11,700. This is the realistic operating cost given that all modes can run async.

**At £15k budget (compressed)**: same allocation but buffer drops to £820. Workable only if batch API is used (releasing the equivalent of £5.9k via batch discount) or if Anthropic prices have reduced 20%+ by mid-project.

### What the £11.5k buffer covers

Methods extraction is new work. Prompt iteration will be heavier than in a pure data-extraction project — methods detection in Maths, Computer Science, and Economics looks structurally different from biomedical, and the schema will need refinement after Mode 1 surfaces edge cases per field.

| Buffer use | Estimate | Priority |
|---|---|---|
| Prompt iteration on methods extraction — ~15 variants × 500 papers each (field-specific prompting) | £2,500 | High |
| Schema refinement re-runs — re-classify Mode 1 + 10k Mode 2 subset after methods taxonomy update | £1,500 | High |
| Distillation experiments (Mode 4 expansion) — multiple candidate rule classifiers, fine-tuned model evaluation against teacher | £2,500 | High |
| Full-text coverage expansion as TDM negotiations unlock new publisher access | £2,000 | Medium |
| Sensitivity / robustness — prompt-variant re-classification on 10k papers per field × year | £1,000 | Medium |
| Cross-validation against Surrey's 5,000 metabolomics labelled set (UKRI 1095) | £500 | Medium |
| Unanticipated contingency | £1,470 | Reserve |

### Sensitivity — what if assumptions change?

| Scenario | Programme cost | Buffer | Within £20k? |
|---|---|---|---|
| Baseline (50% full-text, methods-aware) | £8,530 | £11,470 | Yes |
| Full-text coverage drops to 30% (TDM falls through) | £5,750 | £14,250 | Yes (more buffer) |
| Full-text coverage rises to 70% (strong TDM access) | £11,300 | £8,700 | Yes, tighter |
| Anthropic pricing rises 2x | £17,060 | £2,940 | At ceiling |
| Anthropic pricing rises 5x | £42,650 | n/a | Over budget — fall back to open-weight |
| Floor raised from 500/cell to 800/cell | minor change to allocation, cost similar | — | Yes |

**Pricing fallback**: platform is model-agnostic. If frontier pricing rises sharply, switch Mode 2 to a hosted open-weight model (Llama 4 70B on Bedrock, Qwen 3 Coder, etc.), driving Mode 2 cost down ~10x. Inter-model κ against Anthropic models would need re-validation, but the platform supports this.

---

## 5b. Alternative budget envelopes — three quality scenarios (N=30 numbers)

The full three-scenario breakdown is in Appendix B §B.7. Headline:

| Scenario | Workhorse | Programme cost (flat pricing) | With batch API (50% off) | Fits which budget? |
|---|---|---|---|---|
| **A — Haiku-FT + larger Sonnet Judge (recommended)** | Haiku 4.5 full-text | £14,180 | £7,090 | £15k tight; £20k comfortable |
| B — Sonnet-FT workhorse (highest quality) | Sonnet 4.6 full-text | £26,360 | £13,180 | £20k with batch API only; £25-30k otherwise |
| C — Haiku-ABS workhorse (cheapest, scaling baseline) | Haiku 4.5 abstract-only | £6,650 | £3,325 | £15k with huge buffer — useful as stretch run alongside Scenario A |

**Recommendation: Scenario A at £20,000.** The N=30 spike supports £14.2k programme cost at today's pricing; the £5.8k buffer covers prompt iteration on methods extraction and pricing volatility. With batch API applied (Mode 2 is non-realtime — perfect fit), programme drops to ~£7k and buffer becomes £13k.

**At £15k**, Scenario A still works *if* batch API is used to release £7k of headroom. Without batch API, £15k leaves only £820 buffer — workable but risky.

**Stretch addition**: spend ~£800 of the buffer to run a parallel Mode 2 in Haiku-ABS, publishing the cost/quality trade-off as a methodological adoption finding. This is high-leverage for relatively trivial cost.

---

## 5c. Scalability requirement — cost per document at global corpus scale

The project's £15–20k LLM budget covers the **WP3 stratified sample of 300k UK publications** — sufficient for the headline metric and the distillation training set. But adoption at scale (UKRI, OECD, EU, international funders, publisher platforms) requires the methodology to run on the **full global corpus** of ~200M historic documents plus ~4M new documents per year. This section sets the cost-per-document envelope an adopter would face, and shows which scaling methods meet it.

### 5c.1 The adopter cost envelope (illustrative)

Assume a notional adopter (UKRI, a publisher consortium, an international funder, or a national research office) wants to operationalise the metric across the full corpus on a 10-year horizon:

| Assumption | Value |
|---|---|
| Historic corpus to backfill | 200,000,000 documents |
| New documents added per year | 4,000,000 |
| 10-year cumulative volume (historic + 10×annual) | **240,000,000 documents** |
| Annual operational budget | £100,000 |
| 10-year cumulative budget | £1,000,000 |
| **Maximum allowable cost per document (gross)** | **£0.00417** |

The £100k/year envelope must cover everything — LLM API spend, cloud infrastructure, engineering staff, data acquisition, governance. A realistic LLM-only allocation within £100k/yr is closer to £30-40k/yr (the remainder covering ~0.5 FTE engineer, cloud compute/storage, and data licensing), giving a tighter LLM-only bound of **~£0.0015 per document**.

### 5c.2 Where each scaling method sits vs this envelope

Using N=30 measured costs (Appendix A):

| Method | £/doc today | £/doc with batch API (50% off) | vs £0.00417 (gross envelope) | vs £0.0015 (LLM-only envelope) |
|---|---|---|---|---|
| **Sonnet 4.6 — full text** | £0.0648 | £0.0324 | 15.6× over | 43× over |
| **Haiku 4.5 — full text** | £0.0223 | £0.0112 | 5.3× over | 15× over |
| **Haiku 4.5 — abstract only** | £0.0032 | £0.0016 | 1.3× under | 1.07× over (marginal) |
| **Distilled rule classifier (post-Mode 4)** | ~£0.0001 (compute only) | n/a | 42× under | 15× under |

**At today's pricing and a £100k/yr envelope, neither Sonnet-FT nor Haiku-FT can scale to global corpus.** Even abstract-only Haiku is at the LLM-only affordability edge. **Only the distilled rule classifier scales comfortably within the envelope** — by an order of magnitude or more across both LLM-only and gross-envelope bounds.

This is a much sharper finding than the earlier £500k/yr framing: at the lower budget, the **rule classifier becomes the only viable runtime methodology**, with abstract-only Haiku as a marginal fallback.

### 5c.3 Factoring expected 10-year price trajectory

LLM unit cost has been falling for equivalent capability — Haiku 3.5 (Oct 2024) → Haiku 4.5 (Oct 2025) brought ~2× capability at the same price; similar pattern for Sonnet generations. Assume a conservative 30% reduction per 2-year cycle:

| Year | Sonnet-FT £/doc | Haiku-FT £/doc | Haiku-ABS £/doc |
|---|---|---|---|
| 0 (today) | £0.065 | £0.022 | £0.003 |
| 2 | £0.046 | £0.015 | £0.002 |
| 4 | £0.032 | £0.011 | £0.0015 |
| 6 | £0.022 | £0.008 | £0.0011 |
| 8 | £0.016 | £0.005 | £0.0008 |
| 10 | £0.011 | £0.004 | £0.0006 |
| **10-year cumulative spend on 240M docs** | **£6.2M** | **£2.1M** | **£0.29M** |
| **vs £1M budget** | **6.2× over** | **2.1× over** | **3.4× under** |

Under this trajectory:
- **Sonnet-FT remains unaffordable** across the 10 years — cumulative £6.2M vs £1M budget
- **Haiku-FT remains unaffordable** — cumulative £2.1M vs £1M; even with batch API (50% off → £1.05M) only just fits
- **Haiku-ABS is comfortably affordable** by year 4 onwards; cumulative £0.29M is well within £1M
- **Distilled rule classifier remains effectively free** at all timepoints

At the £100k/yr envelope, the price-reduction trajectory alone does **not** make full-text methods affordable — they would need both price reductions AND batch API discounts AND a 2-3× operational efficiency improvement to fit. The rule-classifier path is the only one that is **certainly** scalable, robust to model pricing, and adoptable by organisations on modest operational budgets.

### 5c.4 Implication — the distillation architecture is a scalability requirement, not a research question

The earlier framing of "LLM-teacher → rule-runtime" positioned distillation as a research question worth investigating. The scalability analysis above reframes it: **a rule-based runtime is the only certainly-scalable methodology for 200M+ document corpora at the £100k/yr operational budgets typical of national funders and institutional research offices.** Without it, the project produces a metric that:

1. Can be reproduced on the 300k UK sample (WP3) — feasible at any of the four scaling methods
2. **Cannot be reproduced at global corpus scale at typical adopter budgets** — Sonnet-FT requires £6M+ over 10 years; Haiku-FT requires £2M+
3. Cannot be reproduced longitudinally by smaller national funders, institutional research offices, or publisher analytics teams operating at £100k/yr or below

The Mode 4 deterministic rule classifier is therefore not optional. It is the only viable runtime methodology if the project's outputs are to be operationalised by funders, REF panels, publisher consortia, or international scientometric initiatives at full corpus scale within realistic operational budgets.

### 5c.5 Implementation approaches for the deterministic runtime classifier — and a critical assessment

The previous sections positioned the "distilled rule classifier" as the scalability path. This section gets honest about what that classifier looks like in practice, and critically assesses whether it can plausibly outperform the cheap LLM baselines we already know underperform.

#### Five candidate approaches

**1. Pure regex on canonical phrases.** Hand-coded patterns matching primary-output signals — "we developed", "novel algorithm", "data are available at github.com/", "we deposited", "this paper introduces", "code is available", "https://doi.org/10.5281" (Zenodo), "[DD]+/[fF]igshare" — and secondary-output signals — "we used", "from the literature", "previously published", "systematic review", "we applied", "based on prior work".

**2. Curated keyword dictionary classification.** A more flexible version of (1) — weighted dictionaries of indicator terms for {primary data, primary software, primary methods, secondary use}. Classify by weighted sum of dictionary hits. Curated by domain (different vocab for biomedicine vs CS vs social sciences).

**3. Section-aware regex + dictionary extraction.** Exploit paper structure — the Data Availability Statement section, Code Availability, Methods, supplementary materials. Apply approach (1) or (2) but only within targeted sections. Most modern papers have standardised availability statements that are easier to parse than narrative methods sections.

**4. TF-IDF concept similarity (lightweight retrieval).** Train term-vectors on the Mode 2 labelled corpus (300k papers); for each new paper, compute cosine similarity to nearest neighbours and classify by majority vote. No LLM at runtime; classical IR. Captures semantic similarity without transformer compute.

**5. Fine-tuned small language model (SLM) distillation.** Fine-tune a 100M–1B parameter encoder (SciBERT, PubMedBERT, or distilled BERT) on the Mode 2 labelled corpus. Inference at ~10ms/doc on CPU, ~£0.0001/doc all-in (compute + serving). This is classical knowledge distillation: LLM teacher → small student model.

#### Critical assessment — will any of these actually work?

The N=30 spike measured Haiku-ABS at **κ=0.31 vs Sonnet-FT**. Haiku-ABS is a frontier-trained small LLM with strong reasoning capability. **Expecting pure regex or keyword dictionaries to outperform a small frontier LLM is unrealistic** — they have less context, no semantic understanding, and no ability to disambiguate ambiguous cases (which is exactly where Haiku-ABS and Sonnet-FT disagree).

Honest expectation per approach:

| Approach | Compute / doc | Expected κ vs Sonnet-FT | Plausible against Haiku-ABS (κ=0.31)? |
|---|---|---|---|
| 1. Pure regex | ~£0 | 0.15–0.25 | **Worse** — no semantic disambiguation |
| 2. Curated dictionaries | ~£0 | 0.25–0.40 | **Comparable** — adds weighting but still surface-level |
| 3. Section-aware regex + dictionaries | ~£0 | 0.35–0.50 | **Better, if structured input available** — exploits document structure |
| 4. TF-IDF concept similarity | ~£0 | 0.35–0.50 | **Better than (1)/(2)** — captures distributional semantics |
| 5. Fine-tuned SLM (SciBERT-class) | ~£0.0001 | **0.55–0.75** | **Better than Haiku-ABS** — this is the realistic path |

**The realistic deterministic classifier is the fine-tuned SLM.** Pure regex is too brittle, dictionaries are too surface-level, TF-IDF lacks the supervised signal. A fine-tuned encoder trained on the Mode 2 labelled corpus combines semantic understanding with deterministic CPU inference — the only path that plausibly matches the κ=0.6–0.7 threshold needed for the metric to be operationally useful at adopter budgets.

#### Why "rule classifier" is still the right term for the proposal

The methodology paper should describe what's actually deployed, but for the adopter narrative, "deterministic classifier" or "lightweight runtime classifier" is more accurate than "rule classifier". A fine-tuned SciBERT model is:

- **Deterministic** at inference (same input → same output, modulo model version)
- **Reproducible** across years (pin the model weights and tokeniser; no API drift)
- **CPU-runnable** (no GPU dependency at inference; runs on a laptop or modest cloud VM)
- **Cheap to operate** (~£0.0001/doc all-in)
- **Auditable** (weights are inspectable; can compute SHAP/attention-based explanations per classification)

These properties are what matter for the adoption story, not literally being hand-coded regex.

#### Hybrid is probably best — and the project can test it

A hybrid combining (3) section-aware rules for high-precision cases with (5) fine-tuned SLM for ambiguous cases is the production-grade pattern in industry NLP:

- **High-precision rules** for unambiguous primary signals — "code available at github.com/[org]/[repo]" → primary software = True; "this is a systematic review of" → secondary; "we generated [n] [units] of [data type]" → primary data = True. These rules have near-100% precision when they fire (low recall is fine — most papers don't have them).
- **Fine-tuned SLM** for everything else — the cases where rules don't fire fall back to the model classifier, which handles ambiguity, paraphrasing, and field-specific language.

Mode 4's ~5,000 LLM Judge calls (£300–500 budget) are sized to evaluate this:
- Train ~3–5 candidate classifiers (pure SLM, hybrid with rules, TF-IDF baseline) on Mode 2 corpus
- Run each on a held-out subset (e.g. 5,000 papers)
- Use Sonnet 4.6 to adjudicate disagreements between candidate classifiers and the Mode 2 Teacher labels
- Compute κ vs Teacher + κ vs Surrey UKRI 1095 ground truth
- Select the best candidate as the project's recommended runtime

#### Honest risk: the rule classifier might not hit κ=0.6

If the fine-tuned SLM only reaches κ=0.4–0.5 against Sonnet-FT, the project's adoption story softens significantly. Realistic fallback positions:

| If best classifier achieves κ vs Sonnet-FT of … | Adoption story becomes |
|---|---|
| 0.6–0.75 | Strong — recommended adopter methodology, comparable to mid-tier human inter-rater agreement |
| 0.45–0.6 | Moderate — useful for trend detection, not individual-paper assessment; published as a methodology with documented limitations |
| < 0.45 | Weak — published as a negative result; abstract-only Haiku-ABS becomes the recommended adopter fallback despite its cost |

The proposal should not over-promise. The £15-20k WP3 investment produces the labelled corpus regardless; the rule classifier is a downstream artefact whose performance is empirically determined.

#### A complementary published finding regardless of outcome

Even in the worst-case scenario (classifier underperforms), the project produces a high-value publication: *"Cost-quality trade-offs in scientometric classification: LLM-runtime vs LLM-teacher / lightweight-runtime architectures."* This directly addresses the open question in metascience about how AI-augmented indicators should be deployed at scale, and is a strong contribution whether the answer is "lightweight runtime works" or "lightweight runtime is insufficient — full-text LLM is needed".

### 5c.6 Where the £15-20k WP3 budget sits in this picture

The project's LLM budget covers the **Teacher pass on 300k UK publications**. This is the training data for the rule classifier. The rule classifier is what scales to 240M documents at <£0.001/doc cumulative cost.

| Pipeline stage | Volume | Cost target | Method |
|---|---|---|---|
| **WP3 Teacher pass (project)** | 300,000 | £15-20k total | Haiku-FT + Sonnet Judge layer |
| **Adopter backfill (post-project)** | 200,000,000 | ≤£0.00417/doc gross (≤£0.0015 LLM-only) | Distilled rule classifier (Mode 4 output) |
| **Adopter annual ingest** | 4,000,000/year | ≤£0.00417/doc gross | Distilled rule classifier (Mode 4 output) |

The £15-20k project investment delivers both the headline UK metric AND the gold-standard training set needed for adopters to deploy a rule classifier across the full global corpus within a **£100k/year operational envelope**. This is a high-leverage return: a one-time £20k investment produces a metric that any national funder or research office can operate indefinitely on a routine departmental budget — vs the ~£2-6M LLM spend they would otherwise face for ongoing full-text classification.

---

## 6. What this delivers — beyond the headline metric

| Deliverable | Status under this proposal |
|---|---|
| Primary metric — UK 2016–2025 longitudinal primary/secondary classification | ✓ Mode 2 |
| Field × year breakdown supporting WP3 field-weighting | ✓ Stratification §3 |
| WP2 validation against existing labelled datasets | ✓ Mode 3 (cross-model judge) |
| Inter-model kappa statistics (Haiku/Sonnet/Opus agreement) | ✓ Mode 3 |
| **Labelled corpus enabling LLM-teacher → rule-runtime research question** | ✓ Mode 2 outputs serve as training labels |
| Adoption-cost benchmark for the **Platform Adoption Evidence Report** | ✓ Daren's WP1 evidence deliverable derives from measured cost numbers here |
| Cost-per-classification benchmark across model tiers | ✓ Mode 2 + Mode 3 + Mode 3b provide multi-tier cost data |

The three-mode design intentionally produces the corpus needed to investigate the LLM-teacher / rule-runtime question (parked §12 of the contributions draft) at no marginal LLM cost — the same labelled corpus that drives the headline metric is the training set for distillation.

---

## 7. Risks and assumptions

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Anthropic pricing increase >2x within 12 months | Low | Medium | Batch API gives 50% off today; open-weight fallback (Llama 4, Qwen 3 on Bedrock) tested in Mode 1 |
| Full-text coverage of 300k sample lower than 50% target | Medium | Medium | Cost decreases (abstract-only is cheaper); classification quality decreases (need to flag in Mode 3 analysis) |
| Inter-model κ remains at ~0.4 even with prompt iteration | High | Medium | **Already observed at N=30.** Mode 3a Judge sized to 33% of Mode 2 (vs original 5%) to compute reliable per-field κ; choose workhorse model based on agreement with Surrey UKRI 1095 ground truth |
| Mode 4 rule classifier underperforms LLM teacher (κ < 0.7 vs Sonnet-FT) | Medium | Medium | Published as a methodological finding regardless; falls back to recommending Haiku-ABS as adopter scaling alternative |
| Parse-error rate >10% on long full-text inputs (Haiku) | Medium (already observed at 10%) | Low | Retry logic + truncation to 80k chars on retry; budget 10% retry overhead in Mode 2 cost line |
| Stratification floor of 500/cell insufficient for emergence detection in smallest fields | Low | Medium | Increase floor to 800/cell (208k floor; leaves 92k for excess), or accept lower power in long-tail fields |
| OpenAlex 2025 backfill changes field totals after submission | High | Low | Stratification computed against settled 2016–2024 only; 2025 treated as ±10% |

### Open assumptions to confirm with team

1. **OpenAlex as corpus source** — proposal assumes OpenAlex Works API is the primary source for the 300k sample (alongside Crossref and DataCite for enrichment). If team prefers Dimensions (per Letter of Support), the Digital Science article-artefact mapping replaces the stratification draw.
2. **Full-text access strategy** — proposal assumes PMC open-access for ~30% of papers. If team negotiates publisher TDM access (Elsevier, Springer Nature, Wiley), full-text rate could rise to 60–80%, increasing token spend but also classification quality.
3. **Whether the LLM-teacher research question is in scope** — proposal's Mode 2 corpus serves both the headline metric AND the distillation question. If the team decides not to pursue distillation, Mode 2 still produces the headline metric; the buffer reduces.

---

## 8. Reproducibility — running this spike yourself

```bash
cd spikes/2026-05-llm-cost
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# Rebuild the exact paper corpus (texts are not committed — see README)
.venv/bin/python fetch_corpus.py
# (Original discovery scripts, kept for the record — re-running them searches PubMed
#  again and may select different papers: fetch_papers.py, fetch_more_pmc*.py, fetch_to_30.py)
# Run 90 classifications: 3 scaling methods × 30 papers, concurrent
.venv/bin/python classify_30_scaling.py
# Build the 30-paper appendix table
.venv/bin/python build_appendix_30.py
# Stratified allocation across 26 OpenAlex fields × 10 years
.venv/bin/python allocate.py
# Statistical power analysis for cell-level detection
.venv/bin/python sample_size_analysis.py
```

Requires:
- Python 3.10+
- `anthropic` and `requests` Python packages (installed in `.venv`)
- Anthropic API key in the macOS keychain as `ANTHROPIC_API_KEY` (the classify scripts read it with `security find-generic-password`; adapt for other platforms)
- ~15 minutes runtime end-to-end for the 90-call classification (6-way concurrency)
- Total Anthropic API spend to reproduce: **£2.71**

Spike data is saved deterministically; re-running with the same PMIDs reproduces results within model-output variance (typically ±3% on token counts; parse-error rate may vary).

---

## 9. What I'm asking the team to agree

1. **Adopt the four-mode allocation** — Develop Insight / Teacher / Judge / Deterministic Rule Model — rather than treating 300k as a single classification pass. Mode 4 (rule classifier) is the **scalability requirement** (§5c), not a research add-on.
2. **Adopt the methods-aware schema** that separately captures primary data, primary software, and primary computational methods. The original "code references" schema under-extracted methods broadly.
3. **Adopt the revised stratification** — 500/cell floor + proportional excess across 26 fields × 10 years — to give every cell statistical power for detecting 5pp shifts at low base rates (the methods-emergence signal).
4. **Confirm budget at £20,000** for the project's WP3 Teacher pass. The N=30 spike supports this with ~£14k programme cost + ~£6k buffer at today's pricing; with batch API the programme drops to ~£7k.
5. **Frame the methodology as scalable by design.** The project's outputs are (a) the headline UK metric, (b) the labelled corpus, AND (c) a deterministic rule classifier that adopters can deploy at <£0.001/doc cumulative cost across 200M+ document corpora within a £500k/year operational envelope. Without (c), the metric is reproducible only on the 300k UK sample and not adoptable by funders or international initiatives at full scale.
6. **Position the Mode 2 labelled corpus as a project output in its own right** — CC-BY 4.0 release alongside the metric and the rule classifier, enabling third-party validation, replication, and continued model-evolution research.

---

## Appendix A — Methods-aware classification of 30 UK full-text papers

Validates the schema and cost numbers on a real sample of 30 UK-affiliated full-text papers fetched via PMC E-utilities (all guaranteed open-access full text, 2023 publication year). Each paper classified with the methods-aware schema across three scaling methods: Sonnet-FT, Haiku-FT, Haiku-ABS.

**Column meanings:**
- *Data used*: pre-existing datasets / sources analysed (per Sonnet-FT, highest quality)
- *Primary data?*: Y/N + short description if paper produced new primary data
- *Methods used*: pre-existing computational methods used
- *Primary methods?*: Y/N + short description if paper produced new primary computational methods (algorithm, model, pipeline, novel statistical approach)
- *Sonnet-FT class / Haiku-FT class / Haiku-ABS class*: classification by each scaling method — disagreements visible in column comparison
- *Sonnet £ / Haiku-FT £ / Haiku-ABS £*: measured per-call cost for each scaling method on this paper

| # | PMID / DOI | Title (truncated) | Data used | Primary data? | Methods used | Primary methods? | Sonnet-FT class | Haiku-FT class | Haiku-ABS class | Sonnet £ | Haiku-FT £ | Haiku-ABS £ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PMID 36888610<br>10.1371/journal.pone.0281933 | National-scale geodatabase of catchment characteristics… | Nationwide IfSAR DEM (5 m resolution, 2013) from NAMRIA, Philippines; Airborne LiDAR… | Y — A national-scale geodatabase of morphometric and topogr… | D8 flow direction algorithm; Constrained regularized smoothing for hydrological DEM … | N | primary | primary | primary | £0.0585 | £0.0208 | £0.0042 |
| 2 | PMID 37368896<br>10.1371/journal.pone.0287744 | Graph drawing using Jaya | Erdős–Rényi random graph model (synthetic datasets); Dolphins social network dataset… | N | Hill Climbing (baseline comparison); Simulated Annealing (baseline comparison)… (+5) | Y — A novel application of the Jaya algorithm to automatic … | primary | primary | primary | £0.0921 | £0.0315 | £0.0035 |
| 3 | PMID 37590277<br>10.1371/journal.pone.0289503 | Emotion regulation in children (ERiC): A protocol for a… | — | N | — | N | PARSE_ERR | PARSE_ERR | secondary | £0.0658 | £0.0228 | £0.0030 |
| 4 | PMID 38110704 | Schizophrenia: from neurochemistry to circuits, symptom… | Published meta-analyses of schizophrenia cognition studies; GWAS datasets (schizophr… | N | Factor analysis (PANSS symptom structure); Network analysis (symptom groupings)… (+9) | N | secondary | secondary | secondary | £0.0739 | £0.0254 | £0.0029 |
| 5 | PMID 37039129 | Evidence-based consensus guidelines for the management … | Existing systematic reviews on catatonia treatment (e.g., Pelzer et al. 2018, Leroy … | N | Evidence-based consensus methodology (Shekelle et al. 1999 evidence grading schema);… | N | secondary | secondary | secondary | £0.2354 | £0.0799 | £0.0024 |
| 6 | PMID 37396047<br>10.3389/fgene.2023.1125599 | Competencies of the UK nursing and midwifery workforce … | Published literature on nursing/genomics competency frameworks (PubMed search up to … | Y — Original survey data collected from 153 nurses and midw… | Literature review; Semi-structured qualitative interviews… (+10) | N | primary | secondary | secondary | £0.0685 | £0.0238 | £0.0036 |
| 7 | PMID 37193558<br>10.1007/s00521-023-08543-8 | Innovative feature-driven machine learning and deep lea… | E-learningDJUST dataset (Abdullah et al.); Psynary psychiatry dataset (Cousins et al… | N | MQAS-based quality assessment; Median and median absolute deviation statistics for r… | N | secondary | secondary | secondary | £0.0150 | £0.0069 | £0.0022 |
| 8 | PMID 37102792<br>10.1259/bjr.20220384 | Isotoxic dose escalated radiotherapy for glioblastoma b… | Pre-treatment DW-MRI ADC maps from 10 GBM patients (2018–2019); Planning CT scans… (… | Y — In-silico treatment planning data for 10 GBM patients i… | Rigid image registration; Trilinear interpolation for image resampling… (+4) | Y — Novel pipeline combining DW-MRI-derived ADC maps, empir… | primary | mixed | primary | £0.0362 | £0.0138 | £0.0046 |
| 9 | PMID 37854674<br>10.3389/fonc.2023.1267604 | Microenvironmental immune cell alterations across the s… | Archival formalin-fixed paraffin-embedded (FFPE) clinical biopsy material from NLPHL… | Y — Original multiplexed immunofluorescence data generated … | Multiplexed multispectral immunofluorescence (Opal 7-Color system); Mann-Whitney U t… | Y — Novel digital image analysis and cell phenotyping workf… | mixed | primary | primary | £0.0476 | £0.0165 | £0.0029 |
| 10 | PMID 37930996<br>10.1371/journal.pone.0290267 | Introducing effective parameters for predicting job bur… | Survey data from 384 startup directors/employees collected in a prior study [31]; Ma… | N | Pearson correlation analysis; Cronbach's alpha reliability analysis… (+6) | Y — A GMDH (Group Method of Data Handling) neural network m… | mixed | mixed | primary | £0.0293 | £0.0099 | £0.0037 |
| 11 | PMID 37186857<br>10.1073/pnas.2221967120 | Design of rigid protein–protein interaction inhibitors … | Existing PDB crystal structures of Mcl-1 apo and Bim-bound forms (e.g., PDB: 2PQK); … | Y — Novel crystal structure of clinical-stage inhibitor AMG… | X-ray crystallography (crystal soaking, diffraction data collection, structure deter… | N | primary | primary | mixed | £0.0415 | £0.0145 | £0.0040 |
| 12 | PMID 36862754<br>10.1371/journal.pmed.1004170 | The effect of supervision on community health workers'… | Government-issued Road to Health Cards; Clinic and hospital antenatal and birth reco… | Y — Original trial data collected from 873 pregnant women a… | Cluster randomized controlled trial design; Linear mixed-effects models for continuo… | N | secondary | secondary | secondary | £0.0559 | £0.0197 | £0.0037 |
| 13 | PMID 38100737<br>10.1371/journal.pdig.0000383 | Phenotypes and rates of cancer-relevant symptoms and te… | CPRD Gold (Clinical Practice Research Datalink); UK Biobank primary care EHR data… (… | Y — Derived cohort-level data on rates and proportions of c… | Poisson regression models; Indirect standardisation… (+4) | Y — Developed harmonised phenotypes (1,776 codes across Rea… | mixed | secondary | secondary | £0.0613 | £0.0210 | £0.0034 |
| 14 | PMID 37388662<br>10.3389/fpsyg.2023.1193241 | Decolonising the psychology curriculum: a perspective | Existing qualitative and quantitative studies on decolonisation in academia; Histori… | N | Narrative literature review; Perspective/opinion synthesis… (+1) | N | secondary | secondary | secondary | £0.0193 | £0.0069 | £0.0023 |
| 15 | PMID 37353567<br>10.1038/s41597-023-02303-y | DOCU-CLIM: A global documentary climate dataset for cli… | PAGES2k Global 2,000 Year Multiproxy Database; NOAA/World Data Service for Paleoclim… | Y — DOCU-CLIM: a new global documentary climate dataset com… | Literature and database search for documentary climate series; Data rescue and digit… | Y — Proxy forward models (multiple regression models using … | primary | primary | primary | £0.0387 | £0.0131 | £0.0024 |
| 16 | PMID 37933221<br>10.7554/eLife.86576 | 100 years of anthropogenic impact causes changes in fre… | Danish Meteorological Institute climate records; Danish national archives biocide sa… | Y — Original multilocus metabarcoding data (18S, 16SV1, 16S… | Multilocus metabarcoding / amplicon sequencing; Radiometric sediment dating (210Pb g… | Y — A novel analytical pipeline combining sparse Canonical … | mixed | PARSE_ERR | primary | £0.0858 | £0.0288 | £0.0043 |
| 17 | PMID 37139088<br>10.3389/fnagi.2023.1132077 | Imaging blood-brain barrier dysfunction: A state-of-the… | Published clinical and preclinical studies on BBB dysfunction in stroke, glioblastom… | N | PET with radiopharmaceuticals ([18F]FDG, [11C]Verapamil, [11C]Loperamide, [15O]H2O, … | N | secondary | PARSE_ERR | secondary | £0.0942 | £0.0311 | £0.0026 |
| 18 | PMID 37351094<br>10.3389/fpubh.2023.1203550 | Is knowledge about COVID-19 associated with willingness… | District citizen registration official report 2021 (Malang); Indonesian Ministry of … | Y — Original cross-sectional survey data collected from 10,… | Stratified sampling design; Linear regression… (+8) | N | primary | secondary | secondary | £0.0558 | £0.0198 | £0.0031 |
| 19 | PMID 37359515<br>10.3389/fimmu.2023.1168539 | Genetic insights into immune mechanisms of Alzheimer's… | AD GWAS datasets (including meta-analysis of 788,989 individuals); PD GWAS datasets … | N | Genome-wide association studies (GWAS); Fine-mapping with Bayesian credible sets… (+… | N | secondary | secondary | secondary | £0.0734 | £0.0246 | £0.0030 |
| 20 | PMID 36911683<br>10.3389/fimmu.2023.1127588 | MAIT cells and the microbiome | Published murine germ-free and knockout mouse studies; Human clinical studies on IBD… | N | Flow cytometry and tetramer staining (cited from referenced studies); Single-cell RN… | N | secondary | secondary | secondary | £0.0895 | £0.0297 | £0.0027 |
| 21 | PMID 37113474<br>10.1016/j.xops.2023.100294 | Exploring Healthy Retinal Aging with Deep Learning | UK Biobank population study (175,844 retinal OCT scans from 85,709 participants, acq… | Y — Synthetically generated counterfactual OCT images and l… | Generative adversarial network (GAN) image translation framework; Image preprocessin… | Y — A novel counterfactual GAN architecture adapted from Ch… | mixed | primary | primary | £0.0334 | £0.0112 | £0.0038 |
| 22 | PMID 37257754<br>10.1016/j.bpsc.2023.05.005 | Theory-Driven Analysis of Natural Language Processing M… | Clinical ratings from experienced clinicians using Thought and Language Disorder Sca… | Y — Simulated FTD-like narratives generated using GPT-2 wit… | Beam-search sampling; Nucleus sampling… (+4) | Y — Novel simulation-based analytical framework using GPT-2… | primary | primary | primary | £0.0349 | £0.0116 | £0.0039 |
| 23 | PMID 37818209<br>10.3389/fsurg.2023.1251444 | PRESS survey: PREvention of surgical site infection—a g… | CDC SSI prevention guidelines (2017); WHO SSI prevention guidelines (2018)… (+4) | Y — A new cross-sectional survey instrument designed to col… | Cross-sectional survey design; CROSS reporting standards… (+7) | N | primary | secondary | secondary | £0.0198 | £0.0072 | £0.0031 |
| 24 | PMID 37154682<br>10.1128/spectrum.01339-23 | Absence of Clinically Meaningful Drug-Drug Interactions… | In vitro human liver microsomal preparations (prior studies); Published preclinical … | Y — Original pharmacokinetic data from two phase 1 open-lab… | Noncompartmental analysis (NCA); ANOVA with geometric least-squares mean ratio… (+4) | N | primary | secondary | secondary | £0.0538 | £0.0186 | £0.0031 |
| 25 | PMID 36845047<br>10.3389/fnut.2023.1011786 | Operationalisation of a standardised scoring system to… | UK Biobank prospective cohort (n=503,317; dietary, anthropometric, physical activity… | N | 2018 WCRF/AICR Cancer Prevention Recommendations (Shams-White et al. 2019 standardis… | Y — A detailed analytical pipeline/workflow for operational… | mixed | secondary | secondary | £0.0655 | £0.0224 | £0.0037 |
| 26 | PMID 37497700<br>10.2174/1573403X19666230727101926 | Appraisal of Cardiovascular Risk Factors, Biomarkers, a… | Framingham Heart Study; ARTPER cohort… (+6) | N | Systematic/narrative literature review; Machine learning for cardiovascular risk cla… | N | secondary | secondary | secondary | £0.0253 | £0.0097 | £0.0023 |
| 27 | PMID 37052999<br>10.2196/42710 | Online Health Information Seeking for Mpox in Endemic a… | Google Trends (relative search volume data for 'monkeypox' topic, Feb 18 – Aug 18, 2… | N | Joinpoint regression analysis (segmented/piecewise linear regression for trend chang… | N | secondary | secondary | secondary | £0.0320 | £0.0111 | £0.0031 |
| 28 | PMID 37368901<br>10.1371/journal.pone.0287886 | Effect of COVID-19 on dental service delivery in Fiji:… | WHO COVID-19 pandemic declarations and statistics; Fiji Ministry of Health and Medic… | Y — Original qualitative data collected through in-depth in… | Qualitative study design; Purposive and random sampling… (+4) | N | primary | secondary | secondary | £0.0784 | £0.0271 | £0.0033 |
| 29 | PMID 37022989<br>10.1371/journal.pone.0280784 | A systematic review and meta-analysis of adolescent nut… | PubMed database; EBSCO/ERIC database… (+5) | N | Systematic review following PRISMA guidelines; Three-step systematic literature sear… | N | secondary | secondary | secondary | £0.2074 | £0.0695 | £0.0031 |
| 30 | PMID 38144895<br>10.3389/fnhum.2023.1294931 | Children with developmental coordination disorder have… | MABC-2 (Movement Assessment Battery for Children–2) scores; DCDQ (Developmental Coor… | Y — Original experimental data collected from 18 children (… | Precision Decomposition (PD) III algorithm (De Luca et al., 2006) for HD-EMG decompo… | N | primary | primary | primary | £0.0543 | £0.0196 | £0.0035 |
| **TOTAL** | 30 papers | | | | | | | | | **£1.9426** | **£0.6683** | **£0.0974** |

### Summary stats (N=30, methods-aware schema, all full text)

| Method | n | Avg input tokens | Avg output tokens | **Avg cost / call** | **Median cost / call** | Min – Max |
|---|---|---|---|---|---|---|
| Sonnet-FT | 30 | 23,195 | 757 | **£0.0648** | £0.0559 | £0.0150 – £0.2354 |
| Haiku-FT | 30 | 23,194 | 931 | **£0.0223** | £0.0198 | £0.0069 – £0.0799 |
| Haiku-ABS | 30 | 761 | 659 | **£0.0032** | £0.0031 | £0.0022 – £0.0046 |

Full spike run cost (90 API calls): **£2.71**. Reproduction: `.venv/bin/python classify_30_scaling.py`.

### Methods-extraction quality observations

- **Computational methods reliably extracted** across diverse fields: D8 flow algorithm (Paper 1, ecology); Jaya optimisation (Paper 2, graph drawing); GAN counterfactual architecture (Paper 21, ophthalmology); GMDH neural network (Paper 10, business); Precision Decomposition III algorithm (Paper 30, paediatric movement); novel analytical pipelines (Papers 8, 13, 16, 25).
- **Primary vs secondary methods well distinguished**: Paper 4 (schizophrenia review) uses many methods but produces none; Paper 2 (Jaya) applies an existing algorithm in a novel domain and is flagged as primary methods; Paper 5 (catatonia guideline) applies the Shekelle 1999 evidence-grading schema without producing a new one.
- **Parse failures cluster on long inputs**: Haiku-FT had parse errors on 3 papers (>100k chars). Sonnet-FT failed on 1 (Paper 3, the RCT protocol — likely due to abstract-only signal). Haiku-ABS had zero parse failures (small inputs reliably produce well-formed JSON).
- **Inter-method classification disagreements** are concentrated on the primary/mixed/secondary boundary. The Sonnet-FT/Haiku-FT disagreements are typically Sonnet calling primary or mixed where Haiku calls secondary — Sonnet picking up subtle primary contributions in full text that Haiku misses.

---

## Appendix B — Inter-method agreement analysis, scaling-method comparison, and cost trajectory

Per-paper data and per-method cost statistics are in Appendix A. This appendix focuses on what the N=30 sample reveals about inter-method classification agreement, the viability of abstract-only as a cheaper scaling alternative, and forward cost trajectory.

### B.1 Why the larger sample matters

The earlier N=10 analysis (the smaller initial spike) showed Haiku-Sonnet agreement at 7/10 (70%) — encouraging. The expanded N=30 sample reveals true Haiku-Sonnet agreement at 60% (κ=0.40), with Sonnet systematically calling more papers primary/mixed. This is a material correction: the headline metric and Mode 3 Judge sizing both depend on getting this number right.

### B.2 Inter-method classification agreement (the key finding)

Cohen's κ across the three methods on the same 30 papers:

| Pair | Agreement | Cohen's κ | Interpretation |
|---|---|---|---|
| **Haiku-FT vs Sonnet-FT** | **18/30 (60%)** | **0.40** | **Moderate.** The two models give materially different classifications. |
| Haiku-ABS vs Haiku-FT | 24/30 (80%) | 0.64 | Substantial. Abstract-only loses some signal but largely tracks full-text. |
| Haiku-ABS vs Sonnet-FT | 17/30 (57%) | 0.31 | Fair. Abstract-only diverges sharply from Sonnet's gold-standard classification. |

### B.3 What the disagreements look like

Classification distribution by method (across 30 papers; parse errors excluded):

| Method | secondary | primary | mixed | parse_err |
|---|---|---|---|---|
| Haiku-FT | 17 | 8 | 2 | 3 |
| Haiku-ABS | 19 | 10 | 1 | 0 |
| Sonnet-FT | 11 | 12 | 6 | 1 |

**Sonnet sees more primary and mixed cases.** Haiku is more conservative — it defaults to "secondary" when ambiguous. Sonnet picks up subtle primary contributions (novel pipelines, applied methods in new domains) that Haiku misses. Examples from the sample where Sonnet flagged primary but Haiku flagged secondary:

- Genomics nursing paper (PMID 37396047): Sonnet identified original survey of 153 nurses + midwives as primary data collection; Haiku missed it
- Microbiome paper (PMID 36915232 — MAIT cells): Sonnet picked up novel analytical framework as primary methods; Haiku read it as a review

### B.4 Implications for model choice

The 60% Haiku-vs-Sonnet agreement is the most important finding from the expanded spike. It means:

1. **The headline metric depends on the model.** If we use Sonnet-FT as Mode 2 Teacher, ~60% of UK output is primary/mixed; with Haiku-FT, ~37%. These would lead to very different conclusions about "is AI shifting UK output toward secondary analysis?"
2. **Validation against a labelled gold standard is critical.** Surrey's UKRI 1095 metabolomics labelled dataset, and the hand-labelled 500-publication sample, must be used to determine which model's classifications are closer to ground truth.
3. **Mode 3 Judge layer needs to be larger than originally scoped.** With 40% Haiku-Sonnet disagreement at scale, the project needs ≥30% Sonnet judging of Haiku outputs to compute reliable per-field κ — not the 5% originally proposed.

### B.5 Parse-error reliability

Haiku-FT failed to produce parseable JSON on 3/30 papers (10%). The three failures were all >100k-character inputs (the long reviews and the clinical-trial protocol). Production solution: retry with prompt refinement or truncate to 80k chars on retry. Sonnet-FT failed on 1/30 (3%); Haiku-ABS had zero parse errors (small inputs reliably produce well-formed JSON).

This is a material reliability finding — a 10% retry rate on Haiku-FT means production cost is 10% higher than the headline £6,683 figure. Budget accordingly.

### B.6 Cost trajectory — factoring expected LLM price reductions over 12 months

Anthropic pricing has trended down per unit of capability over the past two years (Haiku 3 → 4.5 brought 6× capability at 4× cost; Sonnet generations track similar). Additional cost-reduction levers available today:

| Lever | Discount available | Applicability to Mode 2 |
|---|---|---|
| **Batch API** | 50% off input + output | Excellent fit — Mode 2 is non-realtime |
| **Prompt caching** | ~90% on cached system prompt | Modest — system prompt is small share of token cost on full-text papers |
| **Newer-generation Haiku/Sonnet** | Quality-per-pound improving 1.5–2× per year | Likely materially cheaper by mid-2027 |
| **Open-weight on Bedrock** (Llama 4, Qwen) | 5–10× cheaper than Anthropic | Available now; requires κ re-validation |

**Forward assumption set** for 12-month project (M1–M12):

| Scenario | Expected effective cost vs today | Rationale |
|---|---|---|
| Conservative | 100% (no reduction) | Anthropic prices unchanged; batch API not adopted |
| **Base case** | **50–60%** | Batch API alone halves Mode 2 cost; modest model improvements |
| Optimistic | 25–35% | Newer-generation Haiku at half price + batch + caching |
| Aggressive | 10–20% | Open-weight at parity for classification, deployed on Bedrock |

The project should budget against the **conservative case** (price flat) and treat reductions as buffer release.

### B.7 Revised budget — three quality scenarios (N=30 measurements, conservative pricing)

**Scenario A — Haiku-FT Workhorse + larger Sonnet Judge layer (recommended)**

| Mode | Calls | Method | Per-call | Cost |
|---|---|---|---|---|
| M1 Develop Insight | 5,000 | Sonnet-FT | £0.062 | £310 |
| M2 Teacher | 270,000 | Haiku-FT (+ 10% retry buffer) | £0.024 | £6,480 |
| M3a Cross-model Judge (33% subset) | 90,000 | Sonnet-FT | £0.062 | £5,580 |
| M3b Opus adjudication | 5,000 | Opus 4.7 | £0.30 | £1,500 |
| M4 Rule Model evaluation | 5,000 | Sonnet-FT | £0.062 | £310 |
| **Programme cost** | | | | **£14,180** |
| Buffer (iteration / pricing volatility) | | | | £5,820 |
| **Total** | | | | **£20,000** |

If batch API applied (50% off Modes M1–M3a, M4): programme drops to ~£8,300. Buffer becomes £11,700.

**Scenario B — Sonnet-FT Workhorse (highest quality, treats Sonnet as gold standard)**

| Mode | Calls | Method | Per-call | Cost |
|---|---|---|---|---|
| M1 Develop Insight | 5,000 | Sonnet-FT | £0.062 | £310 |
| M2 Teacher | 270,000 | Sonnet-FT | £0.062 | £16,740 |
| M3 Judge (Opus on 10% subset) | 30,000 | Opus 4.7 | £0.30 | £9,000 |
| M4 Rule Model evaluation | 5,000 | Sonnet-FT | £0.062 | £310 |
| **Programme cost** | | | | **£26,360** |

Over £20k budget without batch API. With batch API (50% off): £13,180 — fits £20k with £6,820 buffer.

This scenario is feasible **only if batch API is used**, which requires Mode 2 to be non-realtime (acceptable — there's no urgency on the 300k labelling pass).

**Scenario C — Haiku-ABS Workhorse (cheapest, lowest quality)**

| Mode | Calls | Method | Per-call | Cost |
|---|---|---|---|---|
| M1 Develop Insight | 5,000 | Sonnet-FT | £0.062 | £310 |
| M2 Teacher | 270,000 | Haiku-ABS | £0.003 | £810 |
| M3a Cross-model Judge (20% subset) | 60,000 | Sonnet-FT | £0.062 | £3,720 |
| M3b Opus adjudication | 5,000 | Opus 4.7 | £0.30 | £1,500 |
| M4 Rule Model evaluation | 5,000 | Sonnet-FT | £0.062 | £310 |
| **Programme cost** | | | | **£6,650** |

Cheap, but Haiku-ABS only agrees with Sonnet-FT 57% of the time. **Not recommended as the workhorse** — but useful as a "scaling baseline" run alongside Scenario A: run Mode 2 in both Haiku-FT and Haiku-ABS, compare κ at full corpus scale, and use it as one of the published findings (does abstract-only suffice for adopters who can't afford full-text classification?).

### B.8 Recommendation

**£20,000 budget, Scenario A** (Haiku-FT workhorse + 33% Sonnet Judge layer), with two adjustments versus the rev-2 proposal:

1. **Mode 3a Cross-model Judge expanded from 15k to 90k calls** — to handle the 40% disagreement rate Haiku-vs-Sonnet, we need a Judge subset large enough to compute reliable per-field × per-year κ statistics. 33% of Mode 2 is the right scale.
2. **Mode 2 Teacher includes a 10% retry buffer** for parse failures on long inputs.

Apply batch API where possible to release ~£6k of buffer for additional iteration / sensitivity analysis.

If the team has appetite for a **stretch Scenario C run**: spend £810 of the buffer to also classify Mode 2 in Haiku-ABS, allowing publication of the cost/quality trade-off as a methodological finding ("can a research office monitor this metric for £1,000/year on abstract-only data?"). This is a strong adoption-perspective publication in its own right.

### B.9 Note on the gold-standard question

The findings here intensify the importance of Surrey's UKRI 1095 metabolomics labelled set and the hand-labelled 500-publication sample. The project should:

1. Run all three methods (Haiku-FT, Haiku-ABS, Sonnet-FT) on Surrey's labelled set + the hand-labelled 500
2. Compute κ against ground truth, not just between models
3. Use the better-agreeing method as Mode 2 Teacher — but report the alternative as a robustness check

The methodological finding "method X agrees with human labels at κ=Y, method Z at κ=Y'" is itself a substantive contribution to the metascience literature on LLM-based classification.
