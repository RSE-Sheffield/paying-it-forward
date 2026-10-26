"""Every run's metadata and raw judgements."""
from __future__ import annotations

import streamlit as st

from .common import Context, need


def render(ctx: Context | None) -> None:
    st.title("Runs")
    if not need(ctx):
        return
    st.caption("Runs of the experiments in view, including unfinished ones. Select a row for its run.json "
               "and judgements.")
    runs = ctx.all_runs[ctx.all_runs.experiment.isin(ctx.experiments.experiment)].copy()
    runs["Experiment"] = runs.experiment.map(ctx.label)
    runs["Commit"] = runs.commit.str[:8]
    table = runs[["Experiment", "run", "status", "papers_judged", "papers_total", "failures",
                  "started_at", "finished_at", "Commit", "run_key"]].rename(columns={
                      "run": "Run", "status": "Status", "papers_judged": "Judged", "papers_total": "Of",
                      "failures": "Failed", "started_at": "Started", "finished_at": "Finished"})
    event = st.dataframe(table.drop(columns="run_key"), hide_index=True, width="stretch",
                         on_select="rerun", selection_mode="single-row", key="runs_table")
    selected = event.selection.rows if event and event.selection else []
    if not selected:
        return
    row = runs.iloc[selected[0]]
    st.subheader(f"{row.experiment}/{row.run}")
    left, right = st.columns([1, 2])
    with left:
        st.json(row.meta, expanded=1)
    with right:
        j = ctx.judgements[ctx.judgements.run_key == row.run_key]
        if j.empty:  # unfinished runs are filtered out of ctx.judgements unless the sidebar includes them
            st.info("Judgements of unfinished runs are shown when the sidebar includes unfinished runs.")
            return
        cols = [c for c in ["paper_id", "category", "reasoning", "minutes", "prompt_tokens", "output_tokens",
                            "duration_s", "truncated", "error"] if c in j and j[c].notna().any()]
        st.dataframe(j[cols], hide_index=True, width="stretch",
                     column_config={"reasoning": st.column_config.TextColumn(width="large")})
