# Results dashboard

```
uv run pif dashboard                     # opens in the browser
uv run pif dashboard --server.port 8502  # further options go to `streamlit run`
```

A Streamlit app over `experiments/`. It reads the run files on every page load
(cached until something under `experiments/` changes), so a finished run shows
up on refresh. There is no build step and no generated HTML.

## How it compares things

- **The sidebar scopes every page.** It filters on the five factors: dataset,
  criteria, prompt, input and reviewer. Unfinished runs are hidden unless you
  switch them on, so every run in view covers its whole dataset.
- **Save as default** in the sidebar keeps the current filters in
  `dashboard/.saved_filters.json` (gitignored, so it stays on your computer),
  and the dashboard opens with them from then on. A value added after you
  saved, such as a new reviewer, starts selected rather than hidden. Reset
  filters goes back to the saved default; Forget saved default deletes it.
- **Experiments are labelled by the factors that differ** between those in view.
  Each also gets a number (`#3`) that stays the same across pages, for axes and
  columns too narrow for the full label.
- **A consensus** reduces each experiment to one category per paper: the one
  its runs chose most often. Ties are broken in taxonomy order and flagged.
- **Comparisons between experiments** use their consensus on the papers both
  have judged. They are only made within one dataset.
- **Agreement with humans** pools every comparison between an experiment and
  each human experiment in view on the same dataset, never with itself. No tie
  between reviewers ever needs breaking.

The statistics come from `pif/metrics.py`, the same code the analysis scripts
use.

## Pages

- **Overview.** On the left, one row per experiment with its runs and papers.
  On the right, what the papers were classified as: every reviewer in view has
  one vote per paper, and a paper counts towards the category most of them
  chose. A reviewer with several experiments in view votes for the category
  most of them chose. Papers where two or more categories share the most votes
  are counted as tied.

  Below it, the category mix. First a dot plot by axis: each experiment's
  share of papers at every software level (S1–S3) and data level (D1–D3), with
  models, humans and humans + AI review (reviewer ids ending
  `-plus-ai-review`) side by side and a bar spanning each group's spread. Under
  it, a bar per experiment split into all nine categories by share of papers.
  Colour is the data axis and shade the software axis. Data notes list
  anything left out, such as unfinished runs, invalid judgements, or factors
  edited since an experiment was made.
- **Compare.**
  - An agreement matrix for every experiment in view, by exact category,
    software axis or data axis.
  - Where they disagree, among everyone in view or one group (models, humans,
    humans + AI review). A bar per boundary between levels (S2 / S3, D1 / D3
    and so on) counts the papers that split on it. Below it, a grid of papers
    by experiments for each axis shows the level each chose, most split papers
    first; a divided cell is an experiment whose runs chose more than one level.
  - A drill-down into any two experiments: summary figures, a confusion matrix,
    and every paper they disagree on with both sides' reasoning. Use it for
    criteria v1 vs v2, model vs human, full text vs abstract, and so on.
- **Consistency.** For experiments with two or more runs: run-to-run agreement,
  Fleiss' κ, how many runs agree per paper, where dissenting votes go, and the
  papers the runs split on. A person's repeat passes appear here the same way
  as a model's repeats.
- **Papers.** Every paper with each experiment's consensus side by side, sorted
  by disagreement. Select one for a bar per category showing which runs chose
  it, and every judgement with its reasoning.
- **Runs.** Every run's `run.json` and raw judgements.
- **Factors.** Every version of every factor, which experiments used it, and a
  diff between any two criteria or prompt versions.

Every chart has a "Show as table" twin, and light and dark mode each have their
own colour ramp.

## Code

- `app.py`: page setup and navigation.
- `views/common.py`: loading, the sidebar filters, experiment labels, and the
  chart helpers and palette.
- `views/<page>.py`: one module per page, each with a `render(ctx)` function.

To add a page, write `views/<name>.py` with `render(ctx)`, register it in
`app.py`, and add it to `check_pages.py`. To check that every page renders
without errors against the real results:

```
uv run python dashboard/check_pages.py
```
