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

- **Overview.** One row per experiment:
  - runs, run-to-run agreement and agreement with humans
  - share of papers rated software-generated (S1) and data-generated (D1)
  - seconds or minutes per paper, and input tokens

  Below it, the category mix: the share of papers in each of the nine categories,
  per experiment. Data notes list anything left out, such as unfinished runs,
  invalid judgements, or factors edited since an experiment was made.
- **Compare.**
  - An agreement matrix for every experiment in view, by exact category,
    software axis, data axis or Cohen's κ.
  - A table against each human reviewer.
  - A drill-down into any two experiments: summary figures, a confusion matrix,
    and every paper they disagree on with both sides' reasoning. Use it for
    criteria v1 vs v2, model vs human, full text vs abstract, and so on.
- **Consistency.** For experiments with two or more runs: run-to-run agreement,
  Fleiss' κ, how many runs agree per paper, where dissenting votes go, and the
  papers the runs split on. A person's repeat passes appear here the same way
  as a model's repeats.
- **Papers.** Every paper with each experiment's consensus side by side, sorted
  by disagreement. Select one to read every judgement and its reasoning.
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
