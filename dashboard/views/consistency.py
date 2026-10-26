"""How much an experiment agrees with itself across repeat runs. The same
measure covers a model's repeats and a person's repeat passes."""
from __future__ import annotations

from collections import Counter

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pif import metrics, registry
from pif.taxonomy import CATEGORIES

from .common import Context, heatmap, need, pct, show, style, theme


def render(ctx: Context | None) -> None:
    st.title("Consistency")
    st.caption("Experiments with two or more runs. A model's repeats show how stable its answers are; a person's "
               "repeat passes show the same for a human.")
    if not need(ctx):
        return
    rows = []
    for _, e in ctx.experiments.iterrows():
        j = ctx.judgements[ctx.judgements.experiment == e.experiment]
        sc = metrics.self_consistency(j)
        if sc["runs"] >= 2:
            rows.append({"experiment": e.experiment, "Experiment": e["tagged"], "Human": e.reviewer_kind == "human",
                         "Runs": sc["runs"], "Papers": sc["papers"], "Pairwise agreement": sc["pairwise"],
                         "Unanimous papers": sc["unanimous"], "Fleiss κ": sc["fleiss"],
                         "Fleiss κ software": sc["fleiss_s"], "Fleiss κ data": sc["fleiss_d"]})
    if not rows:
        st.info("No experiment in view has more than one run.")
        return
    table = pd.DataFrame(rows)

    t = theme()
    fig = go.Figure()
    for human, name, color in [(False, "Model", t["model"]), (True, "Human", t["human"])]:
        sub = table[table.Human == human]
        if sub.empty:
            continue
        fig.add_bar(y=sub.Experiment, x=sub["Pairwise agreement"], orientation="h", name=name,
                    marker=dict(color=color, cornerradius=4), width=0.5,
                    text=[pct(v) for v in sub["Pairwise agreement"]], textposition="outside",
                    textfont=dict(color=t["ink2"]), cliponaxis=False,
                    customdata=sub[["Runs", "Fleiss κ"]].values,
                    hovertemplate="%{y}<br>pairwise agreement %{x:.0%}<br>%{customdata[0]} runs, "
                                  "Fleiss κ %{customdata[1]:.2f}<extra></extra>")
    fig.update_xaxes(range=[0, 1.08], tickformat=".0%", showgrid=True)
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=list(table.Experiment))
    style(fig, height=max(180, 34 * len(table) + 80))
    fig.update_layout(showlegend=table.Human.nunique() > 1, bargap=0.4)
    st.subheader("Run-to-run agreement")
    st.caption("Mean, over every pair of runs, of the share of papers on which both chose the same category.")
    show(fig, table.drop(columns=["experiment", "Human"]), key="consistency_bars")

    st.subheader("One experiment in detail")
    pick = st.selectbox("Experiment", list(table.Experiment), key="cons_pick")
    exp = table.set_index("Experiment").experiment[pick]
    j = ctx.judgements[(ctx.judgements.experiment == exp) & ctx.judgements.valid]
    n_runs = j.run.nunique()
    votes = {p: list(g.category) for p, g in j.groupby("paper_id", sort=False)}

    agree = Counter(Counter(v).most_common(1)[0][1] for v in votes.values() if len(v) == n_runs)
    left, right = st.columns(2)
    with left:
        st.markdown("**How many runs agree, per paper**")
        ks = list(range(n_runs, 0, -1))
        bar = go.Figure(go.Bar(x=[f"{k} of {n_runs}" for k in ks], y=[agree.get(k, 0) for k in ks],
                               marker=dict(color=t["model"], cornerradius=4), width=0.5,
                               text=[agree.get(k, 0) for k in ks], textposition="outside",
                               textfont=dict(color=t["ink2"]), cliponaxis=False,
                               hovertemplate="%{x} runs chose the modal category: %{y} papers<extra></extra>"))
        bar.update_yaxes(title="papers", showgrid=True)
        show(style(bar, 260), pd.DataFrame({"Runs agreeing": [f"{k} of {n_runs}" for k in ks],
                                            "Papers": [agree.get(k, 0) for k in ks]}), key="cons_k")
    with right:
        st.markdown("**Where dissenting votes go** — rows are the consensus, columns the dissenting vote.")
        sub = pd.DataFrame(0, index=CATEGORIES, columns=CATEGORIES)
        for v in votes.values():
            modal = metrics.modal(v)
            for x in v:
                if x != modal:
                    sub.loc[modal, x] += 1
        hover = [[f"consensus {a}, dissent {b}: {sub.loc[a, b]} vote(s)" for b in CATEGORIES] for a in CATEGORIES]
        fig2 = heatmap(sub.values.tolist(), CATEGORIES, CATEGORIES, zmin=0, zmax=max(1, sub.values.max()),
                       fmt=lambda v: str(int(v)) if v else "", hover=hover, cell=26, scale_title="votes")
        show(fig2, sub.reset_index().rename(columns={"index": "consensus"}), key="cons_dissent")

    papers = {p["paper_id"]: p for p in registry.dataset(ctx.experiments.set_index("experiment").dataset[exp]).papers}
    split = [(p, v) for p, v in votes.items() if len(set(v)) > 1]
    st.markdown(f"**The {len(split)} paper(s) its runs split on**")
    if split:
        st.dataframe(pd.DataFrame([{
            "Paper": p, "Title": papers.get(p, {}).get("title", ""),
            "Consensus": metrics.modal(v) + (" (tie)" if metrics.is_tied(v) else ""),
            "Votes": ", ".join(f"{c} ×{n}" for c, n in sorted(Counter(v).items(), key=lambda kv: -kv[1])),
        } for p, v in split]), hide_index=True, width="stretch")
