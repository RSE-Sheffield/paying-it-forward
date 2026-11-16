"""Paper by paper: how much the reviewers disagree across the dataset, every
reviewer's consensus side by side and, for the paper you pick, which runs chose
each category and every judgement with its reasoning."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pif import registry
from pif.registry import KINDS
from pif.taxonomy import CATEGORIES, label

from .common import MISSING, Context, experiment_colors, ink_on, need, show, style, theme

_PREFIX = {"criteria": "criteria ", "prompt": "prompt "}


def reviewer_names(exps: pd.DataFrame) -> dict[str, str]:
    """{experiment: its reviewer's name}. A reviewer with several experiments
    in view gets the factors that tell them apart, e.g. "Daren (criteria v2)"."""
    names = {}
    for name, g in exps.groupby("reviewer_name", sort=False):
        differ = [k for k in KINDS if g[k].nunique() > 1]
        for _, e in g.iterrows():
            tell_apart = ", ".join(_PREFIX.get(k, "") + str(e[k]) for k in differ)
            names[e.experiment] = name if len(g) == 1 else f"{name} ({tell_apart})"
    return names


def distinct_codes(counts: pd.Series, reviewers: int) -> tuple[go.Figure, pd.DataFrame]:
    """A bar per number of distinct codes (1 = every reviewer agrees), its
    height the number of papers. Every possible number gets a bar, empty or not."""
    t = theme()
    most = min(reviewers, len(CATEGORIES))
    ks = list(range(min(1, int(counts.min())), max(most, int(counts.max())) + 1))
    n = counts.value_counts().reindex(ks, fill_value=0)
    share = n / max(len(counts), 1)
    fig = go.Figure(go.Bar(
        x=[str(k) for k in ks], y=n.values, width=0.55, marker=dict(color=t["seq"][3]),
        text=[str(v) for v in n.values], textposition="outside", cliponaxis=False,
        textfont=dict(color=t["ink2"]), customdata=[f"{s:.0%}" for s in share],
        hovertemplate="%{x} distinct code(s): %{y} paper(s), %{customdata}<extra></extra>"))
    fig.update_xaxes(title="distinct codes", type="category", showgrid=False)
    fig.update_yaxes(title="papers", range=[0, max(int(n.max()), 1) * 1.18], showgrid=True)
    style(fig, height=260)
    fig.update_layout(barcornerradius=4, showlegend=False)
    table = pd.DataFrame({"Distinct codes": ks, "Papers": n.values, "Share": [f"{s:.0%}" for s in share]})
    return fig, table


def votes(exps: pd.DataFrame, j: pd.DataFrame, names: dict[str, str]) -> tuple[go.Figure, pd.DataFrame]:
    """A bar per category counting the runs that chose it, one segment per
    experiment, coloured by experiment and labelled with its reviewer's name."""
    t = theme()
    j = j[j.valid]
    colors = experiment_colors(exps)
    fig, rows = go.Figure(), []
    for _, e in exps.iterrows():
        mine = j[j.experiment == e.experiment]
        if mine.empty:
            continue
        color, name = colors[e.experiment], names[e.experiment]
        runs = {k: sorted(mine.run[mine.category == k]) for k in CATEGORIES}
        fig.add_bar(y=CATEGORIES, x=[len(runs[k]) for k in CATEGORIES], orientation="h", width=0.6,
                    name=name, legendgroup=e.reviewer,
                    marker=dict(color=color, line=dict(color=t["page"], width=2)),
                    text=[name if runs[k] else "" for k in CATEGORIES], textposition="inside",
                    insidetextanchor="middle", textangle=0, textfont=dict(color=ink_on(color)),
                    customdata=[[f"{k} · {label(k)}", ", ".join(runs[k])] for k in CATEGORIES],
                    hovertemplate=f"%{{customdata[0]}}<br>{name}<br>"
                                  f"%{{x}} of {len(mine)} run(s): %{{customdata[1]}}<extra></extra>")
        rows += [{"Category": k, "Reviewer": name, "Runs": len(runs[k]), "Which": ", ".join(runs[k])}
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
    names = reviewer_names(exps)
    columns = [names[e] for e in exps.experiment]

    cell = {(r.experiment, r.paper_id): r.category + ("*" if r.tied else "") for r in cons.itertuples()}
    rows = []
    for p in papers:
        cats = [cell.get((e, p["paper_id"]), MISSING) for e in exps.experiment]
        distinct = {c.rstrip("*") for c in cats if c != MISSING}
        rows.append({"Paper": p["paper_id"], "Title": p.get("title", ""),
                     "Distinct codes": len(distinct), **dict(zip(columns, cats))})
    table = pd.DataFrame(rows)

    if not table.empty:
        st.subheader("How much the reviewers disagree")
        st.caption(f"How many papers received 1, 2, 3… different codes from the {len(exps)} reviewer column(s) "
                   "below (1 = every reviewer agrees). Covers the whole dataset, whatever the search below. Hover "
                   "a bar for its share of papers.")
        show(*distinct_codes(table["Distinct codes"], len(exps)), key="papers_distinct")

    st.subheader("Every paper")
    st.caption("One column per reviewer (hover a header for its experiment and its #n on other pages), showing "
               "the consensus of its runs (* = tied between categories, — = not judged). "
               "**Distinct codes** counts how many different codes the reviewers in view gave the paper: 1 means "
               "they all agree. Select a row to see every judgement and its reasoning.")
    c1, c2, c3 = st.columns([3, 1, 1], vertical_alignment="bottom")
    query = c1.text_input("Search titles", key="papers_q", placeholder="e.g. geodatabase")
    order = c2.selectbox("Sort", ["Most disagreement", "Least disagreement", "Dataset order"], key="papers_sort")
    if query:
        table = table[table.Title.str.contains(query, case=False, regex=False)]
    if order != "Dataset order":
        table = table.sort_values("Distinct codes", ascending=order == "Least disagreement", kind="stable")
    # utf-8-sig so Excel reads the dashes and accented titles correctly.
    c3.download_button("Download CSV", table.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"papers-{ds}.csv", mime="text/csv", icon=":material/download:",
                       help="The table as shown: the current search, sort and sidebar filters.", width="stretch")
    event = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun",
                         selection_mode="single-row", key="papers_table",
                         column_config={"Title": st.column_config.TextColumn(width="large"),
                                        "Distinct codes": st.column_config.NumberColumn(
                                            help="How many different codes the reviewers in view gave this "
                                                 "paper; 1 = all agree."),
                                        **{c: st.column_config.TextColumn(c, help=f"{t} · {n}", width="small")
                                           for c, t, n in zip(columns, exps.tag, exps.experiment)}})
    selected = event.selection.rows if event and event.selection else []
    if not selected:
        return

    pid = table.iloc[selected[0]].Paper
    paper = next(p for p in papers if p["paper_id"] == pid)
    st.subheader(paper.get("title") or pid)
    meta = " · ".join(str(x) for x in [paper.get("journal"), paper.get("year"), paper.get("field")] if x)
    doi = paper.get("doi")
    st.caption(meta + (f" · [doi:{doi}](https://doi.org/{doi})" if doi else "") + f" · id {pid}")
    j = ctx.judgements[(ctx.judgements.paper_id == pid) & ctx.judgements.experiment.isin(exps.experiment)]
    st.markdown("**Which runs chose each category** — each segment is one reviewer's runs, named as in the "
                "table above. Each reviewer has its own hue, shaded when it has several experiments. Hover a "
                "segment to see which runs; click a legend entry to hide that reviewer.")
    show(*votes(exps, j, names), key="paper_votes")
    for _, e in exps.iterrows():
        mine = j[j.experiment == e.experiment]
        if mine.empty:
            continue
        verdict = cell.get((e.experiment, pid), MISSING)
        agree = (mine.category == verdict.rstrip("*")).sum()
        with st.expander(f"{names[e.experiment]} — {verdict} ({agree} of {len(mine)} run(s))",
                         expanded=len(exps) <= 4):
            for r in mine.itertuples():
                extra = []
                if getattr(r, "minutes", None) and not pd.isna(r.minutes):
                    extra.append(f"{r.minutes:g} min")
                if getattr(r, "truncated", None) is True:
                    extra.append("full text truncated")
                st.markdown(f"**{r.run}** `{r.category or 'invalid'}`"
                            + (f" _({', '.join(extra)})_" if extra else "")
                            + f" — {r.reasoning or '_no reasoning recorded_'}")
