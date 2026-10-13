# Results dashboard

A single self-contained HTML page for exploring everything under
`Experiments/`. No server, no network, no dependencies — open
`dashboard/index.html` in a browser.

```
xdg-open dashboard/index.html
```

## Rebuilding after a new run

`index.html` is generated. Whenever you add a run to `Experiments/`, rebuild:

```
python3 dashboard/build_dashboard.py
```

The script reads every subdirectory of `Experiments/`, normalises the rows,
and injects the whole dataset into `template.html` as JSON. Edit
`template.html` for anything visual or behavioural; never edit `index.html`
directly, it will be overwritten.

## Configurations

The unit of comparison is the **configuration** — one model reading one kind
of input — not the individual run. Runs sharing a configuration are repeats of
each other. Configurations are derived from `run.json`, not hardcoded, so a new
model or input mode appears on its own:

| Derived from | Configuration |
|---|---|
| `model` + `content: "full"` | *model* · full text |
| `model` + `content: "metadata"` | *model* · abstract only |
| `model` is null | Human (*provider*) |

At the time of writing that gives seven: Claude Sonnet 5 and Gemma4 12B each
over full text and abstract-only, plus a manual pass each from Edwin, Matt and
Michael. Each is assigned one of eight colour slots in a fixed order and keeps
it across every tab.

**Comparing against humans.** The Model consensus vs human tab never pools
the reviewers into a single answer. Each model configuration is reduced to its
consensus (the category its repeats chose most often per paper) and compared
with one reviewer at a time. The Reasoning tab still uses the reviewers'
majority category.

The selector at the top of the page toggles whole configurations on and off,
and scopes every tab below it. Individual runs can still be added or dropped
one at a time on the **Runs** tab; when a configuration is only partly
selected its toggle says so.

**Current criteria** selects the runs (model and human) whose copy of
`criteria.md` matches the repo's current one, compared by hash.

The selection is saved in the browser's `localStorage`, so it survives
reloading the page and rebuilding it. What's saved is the list of runs switched
*off*, so a run added by a later rebuild appears switched on. **All runs**
resets it. Because it lives in the browser, it isn't shared across browsers or
machines.

## Tabs

- **Overview** — one software×data matrix per configuration. Each
  configuration's repeats are reduced to the category chosen most often for
  each paper, so every matrix totals the paper count rather than the judgement
  count, and all matrices share one colour scale so they can be read against
  each other.
- **Papers** — papers scored by how many of the nine categories at least one
  selected run put them in: 1 means every run agreed, higher means the runs
  spread across more categories. An ordered dot plot runs from agreement on the
  left to widest disagreement on the right; click a dot to open that paper's
  distribution below. The list beside it is sortable by most categories, fewest
  categories, or title, and any run's row expands to its full reasoning.
- **Model variation** — how much a model configuration disagrees with *itself*. Mean
  agreement per configuration, how often k of n repeats land on the same
  category, a paper-by-paper strip showing every repeat's vote (majority in
  colour, dissent in grey with its own label), and a matrix of where the
  dissenting votes go when it wavers.
- **Human variation** — how much the human reviewers disagree with each
  other, one vote per selected human run. Headline agreement, every pair of
  reviewers scored on exact category, and each axis, how often each
  reviewer is outvoted, a paper-by-paper strip (majority cells in each
  reviewer's colour, dissent in grey), and where the outvoted reviewers go.
  Tied papers have no majority, so they are left uncoloured and out of the
  outvoted and split counts. If a reviewer has more than one run in view, each
  run votes separately and the tab says so.
- **Model consensus vs human** — one sub-tab per selected reviewer plus
  **Combined**. A reviewer's sub-tab simply scores each model configuration's
  consensus against that reviewer's category, paper by paper, with the other
  reviewers listed as rows for comparison. **Combined** compares the consensus
  with every reviewer separately and pools the comparisons, so matching two of
  three reviewers counts as 2 of 3 and no tie between reviewers ever needs
  breaking; each reviewer is scored against the others the same way, as the
  benchmark. Every card on the tab (the summary table with a per-reviewer
  breakdown, one confusion panel per configuration, what the models pick
  instead, each axis on its own, and the per-paper leaderboard) uses the same
  consensus-against-reviewer comparison. Ties among a configuration's own
  repeats are resolved in taxonomy order and counted in the Tied column.
- **Reasoning** — the free-text justifications rather than the labels. Shows
  the papers where no model run reached the human's category, each expanding to
  each human's reasoning beside one quote per model configuration. The quotes come
  straight from the run files; the commentary between them is written by Claude
  and is labelled as such, because it is an LLM's interpretation of unstructured
  text rather than something derived arithmetically from the data.
- **Runs** — full `run.json` metadata for every run, plus derived totals
  (tokens, model time, category mix) and the shared task/criteria text.

Ties are resolved in taxonomy order wherever a single category is needed, and
flagged (`*`, or a "tie" marker) wherever that happens.

## Data shapes

The LLM runs and the manual pass do not carry the same fields, and the loader
handles the difference rather than assuming one shape:

- Per-judgement telemetry (`prompt_tokens`, `output_tokens`, `duration_s`,
  `num_ctx`, `truncated`, `timestamp`) is carried through only when present.
- The manual pass carries `reviewer` instead, and its results file is not
  called `results.json`; any non-`run.json` JSON file in the directory is used.
- Rows whose `category` is not one of the nine valid codes are flagged: the
  judgement is shown as uncategorised and excluded from consensus and
  agreement figures.

Every flagged row is printed by the build script and listed under **Data
notes** on the Overview tab, so nothing is silently corrected.
