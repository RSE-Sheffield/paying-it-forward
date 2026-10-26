"""Paper by paper: every experiment's consensus side by side and, for the
paper you pick, which runs chose each category and every judgement with its
reasoning."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pif import registry
from pif.taxonomy import CATEGORIES, label

from .common import MISSING, Context, experiment_colors, ink_on, need, show, style, theme


def votes(exps: pd.DataFrame, j: pd.DataFrame) -> tuple[go.Figure, pd.DataFrame]:
    """A bar per category counting the runs that chose it, one segment per
    experiment, coloured by experiment and labelled with its #n."""
    t = theme()
    j = j[j.valid]
    colors = experiment_colors(exps)
    fig, rows = go.Figure(), []
    for _, e in exps.iterrows():
        mine = j[j.experiment == e.experiment]
        if mine.empty:
            continue
        color = colors[e.experiment]
        runs = {k: sorted(mine.run[mine.category == k]) for k in CATEGORIES}
        fig.add_bar(y=CATEGORIES, x=[len(runs[k]) for k in CATEGORIES], orientation="h", width=0.6,
                    name=e["tagged"], legendgroup=e.reviewer, legendgrouptitle_text=e.reviewer,
                    marker=dict(color=color, line=dict(color=t["page"], width=2)),
                    text=[e.tag if runs[k] else "" for k in CATEGORIES], textposition="inside",
                    insidetextanchor="middle", textangle=0, textfont=dict(color=ink_on(color)),
                    customdata=[[f"{k} · {label(k)}", ", ".join(runs[k])] for k in CATEGORIES],
                    hovertemplate=f"%{{customdata[0]}}<br>{e['tagged']}<br>"
                                  f"%{{x}} of {len(mine)} run(s): %{{customdata[1]}}<extra></extra>")
        rows += [{"Category": k, "Experiment": e["tagged"], "Runs": len(runs[k]), "Which": ", ".join(runs[k])}
                 for k in CATEGORIES if runs[k]]
    totals = j.category.value_counts()
    for k in CATEGORIES:  # the total at each bar's end
        if totals.get(k, 0):
            fig.add_annotation(x=totals[k], y=k, text=str(totals[k]), showarrow=False, xanchor="left", xshift=6,
                               font=dict(color=t["ink2"]))
    most = max(totals, default=1)
    fig.update_xaxes(title="runs", range=[0, most * 1.08], dtick=1 if most <= 10 else 5, showgrid=True)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    # Tall enough for the nine bars or the legend (~19px an entry, ~30px a reviewer), whichever is more.
    shown = exps[exps.experiment.isin(j.experiment)]
    style(fig, height=max(9 * 34 + 70, 19 * len(shown) + 30 * shown.reviewer.nunique() + 80))
    fig.update_layout(barmode="stack", barcornerradius=4, uniformtext=dict(mode="hide", minsize=10),
                      legend=dict(orientation="v", x=1.02, xanchor="left", y=1, yanchor="top",
                                  traceorder="grouped", groupclick="toggleitem"))
    return fig, pd.DataFrame(rows)


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
    st.markdown("**Which runs chose each category** — each segment is one experiment's runs (#n as in the "
                "table above). Each reviewer has its own hue, shaded when it has several experiments. Hover a "
                "segment to see which runs; click a legend entry to hide that experiment.")
    show(*votes(exps, j), key="paper_votes")
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
