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

At the time of writing that gives five: Claude Sonnet 5 and Gemma4 12B each
over full text and abstract-only, plus the manual human pass. Each is assigned
a colour slot in a fixed order and keeps it across every tab.

The selector at the top of the page toggles whole configurations on and off,
and scopes every tab below it. Individual runs can still be added or dropped
one at a time on the **Runs** tab; when a configuration is only partly
selected its toggle says so.

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
- **Variation** — how much a configuration disagrees with *itself*. Mean
  agreement per configuration, how often k of n repeats land on the same
  category, a paper-by-paper strip showing every repeat's vote (majority in
  colour, dissent in grey with its own label), and a matrix of where the
  dissenting votes go when it wavers.
- **Consensus vs human** — a confusion-matrix panel per configuration (rows
  the human's category, columns the configuration's most-chosen one, all
  sharing one colour scale; every matrix on this tab puts the human on the
  rows, since the human is the reference);
  a summary of stability and exact/per-axis agreement; which of the human's
  categories the models agree with most, and what they pick instead when they
  don't (both pooled across every model run, row-normalised); and a leaderboard
  ranking every paper by how many model runs matched the human's call, each
  row expandable to the per-configuration split.
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
