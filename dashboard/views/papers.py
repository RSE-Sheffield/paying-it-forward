"""Paper by paper: every experiment's consensus side by side, and every
judgement with its reasoning for the paper you pick."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from pif import registry

from .common import MISSING, Context, need


def render(ctx: Context | None) -> None:
    st.title("Papers")
    if not need(ctx):
        return
    datasets = list(dict.fromkeys(ctx.experiments.dataset))
    ds = st.selectbox("Dataset", datasets, key="papers_ds") if len(datasets) > 1 else datasets[0]
    exps = ctx.experiments[ctx.experiments.dataset == ds]
    cons = ctx.consensus()
    cons = cons[cons.experiment.isin(exps.experiment)]
    papers = registry.dataset(ds).papers

    cell = {(r.experiment, r.paper_id): r.category + ("*" if r.tied else "") for r in cons.itertuples()}
    rows = []
    for p in papers:
        cats = [cell.get((e, p["paper_id"]), MISSING) for e in exps.experiment]
        distinct = {c.rstrip("*") for c in cats if c != MISSING}
        rows.append({"paper_id": p["paper_id"], "Title": p.get("title", ""), "Field": p.get("field", ""),
                     "Categories": len(distinct), **dict(zip(exps.tag, cats))})
    table = pd.DataFrame(rows)

    st.caption("One column per experiment (#n as on Overview; hover a header for its name), showing its "
               "consensus (* = tied between categories). "
               "**Categories** counts how many different consensus categories the paper received: 1 means every "
               "experiment in view agrees. Select a row to see every judgement and its reasoning.")
    c1, c2 = st.columns([3, 1])
    query = c1.text_input("Search titles", key="papers_q", placeholder="e.g. geodatabase")
    order = c2.selectbox("Sort", ["Most disagreement", "Least disagreement", "Dataset order"], key="papers_sort")
    if query:
        table = table[table.Title.str.contains(query, case=False, regex=False)]
    if order != "Dataset order":
        table = table.sort_values("Categories", ascending=order == "Least disagreement", kind="stable")
    event = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun",
                         selection_mode="single-row", key="papers_table",
                         column_config={"paper_id": st.column_config.TextColumn("Paper"),
                                        "Title": st.column_config.TextColumn(width="large"),
                                        **{t: st.column_config.TextColumn(t, help=n, width="small")
                                           for t, n in zip(exps.tag, exps.label)}})
    selected = event.selection.rows if event and event.selection else []
    if not selected:
        return

    pid = table.iloc[selected[0]].paper_id
    paper = next(p for p in papers if p["paper_id"] == pid)
    st.subheader(paper.get("title") or pid)
    meta = " · ".join(str(x) for x in [paper.get("journal"), paper.get("year"), paper.get("field")] if x)
    doi = paper.get("doi")
    st.caption(meta + (f" · [doi:{doi}](https://doi.org/{doi})" if doi else "") + f" · id {pid}")
    j = ctx.judgements[(ctx.judgements.paper_id == pid) & ctx.judgements.experiment.isin(exps.experiment)]
    for _, e in exps.iterrows():
        mine = j[j.experiment == e.experiment]
        if mine.empty:
            continue
        verdict = cell.get((e.experiment, pid), MISSING)
        agree = (mine.category == verdict.rstrip("*")).sum()
        with st.expander(f"{e['tagged']} — {verdict} ({agree} of {len(mine)} run(s))", expanded=len(exps) <= 4):
            for r in mine.itertuples():
                extra = []
                if getattr(r, "minutes", None) and not pd.isna(r.minutes):
                    extra.append(f"{r.minutes:g} min")
                if getattr(r, "truncated", None) is True:
                    extra.append("full text truncated")
                st.markdown(f"**{r.run}** `{r.category or 'invalid'}`"
                            + (f" _({', '.join(extra)})_" if extra else "")
                            + f" — {r.reasoning or '_no reasoning recorded_'}")
