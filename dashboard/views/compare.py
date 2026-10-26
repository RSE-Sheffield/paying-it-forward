"""Compare experiments with each other: an agreement matrix for everything in
view, and a drill-down into any pair -- criteria v1 vs v2, model vs human,
full text vs abstract, or anything else."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from pif import metrics, registry
from pif.taxonomy import CATEGORIES

from .common import Context, data_range, heatmap, need, pct, show, theme

METRICS = {"Exact category": "exact", "Software axis": "s", "Data axis": "d"}


def matrix(ctx: Context, dataset: str, metric: str):
    key = METRICS[metric]
    exps = ctx.experiments[ctx.experiments.dataset == dataset]
    cons = ctx.consensus_map()
    z, hover, rows = [], [], []
    for _, a in exps.iterrows():
        zr, hr = [], []
        for _, b in exps.iterrows():
            if a.experiment == b.experiment:
                zr.append(None)
                hr.append(f"{a.label}<br>(itself — see Consistency)")
                continue
            r = metrics.agreement(cons.get(a.experiment, {}), cons.get(b.experiment, {}))
            v = r[key]
            zr.append(None if pd.isna(v) else v)
            hr.append(f"{a.label}<br>vs {b.label}<br>{metric}: {pct(v)} over {r['n']} papers")
            rows.append({"A": a.label, "B": b.label, "Papers": r["n"], "Exact": r["exact"],
                         "Software axis": r["s"], "Data axis": r["d"]})
        z.append(zr)
        hover.append(hr)
    zmin, zmax = data_range([v for r in z for v in r], 0, 1)
    names = list(exps["tagged"])
    fig = heatmap(z, names, names, zmin=zmin, zmax=zmax, fmt=pct, hover=hover, scale_title="agreement")
    # Names along the top too, angled so long ones don't collide; the extra height
    # (about 5px per character at 45°) keeps the cells from shrinking to make room.
    fig.update_xaxes(tickangle=-45)
    fig.update_layout(height=fig.layout.height + 5 * max(map(len, names)))
    return fig, pd.DataFrame(rows)


def judgement_lines(ctx: Context, experiment: str, paper_id: str) -> str:
    j = ctx.judgements[(ctx.judgements.experiment == experiment) & (ctx.judgements.paper_id == paper_id)]
    return "\n".join(f"- **{r.run}** `{r.category or 'invalid'}` — {r.reasoning or '_no reasoning recorded_'}"
                     for r in j.itertuples())


def pair(ctx: Context, dataset: str) -> None:
    exps = ctx.experiments[ctx.experiments.dataset == dataset]
    by_label = dict(zip(exps.label, exps.experiment))
    labels = list(by_label)
    humans = [l for l in labels if ctx.is_human(by_label[l])]
    c1, c2 = st.columns(2)
    a_label = c1.selectbox("A", labels, index=0, key=f"pair_a_{dataset}")
    default_b = next((l for l in humans if l != a_label), next(l for l in labels if l != a_label))
    b_label = c2.selectbox("B", labels, index=labels.index(default_b), key=f"pair_b_{dataset}")
    if a_label == b_label:
        st.info("Pick two different experiments.")
        return
    A, B = by_label[a_label], by_label[b_label]
    cons = ctx.consensus_map()
    ca, cb = cons.get(A, {}), cons.get(B, {})
    r = metrics.agreement(ca, cb)

    cols = st.columns(4)
    cols[0].metric("Papers compared", r["n"])
    cols[1].metric("Exact agreement", pct(r["exact"]))
    cols[2].metric("Software axis", pct(r["s"]))
    cols[3].metric("Data axis", pct(r["d"]))

    shared = [p for p in ca if p in cb]
    counts = pd.crosstab(pd.Series([ca[p] for p in shared], name="A"),
                         pd.Series([cb[p] for p in shared], name="B")).reindex(
        index=CATEGORIES, columns=CATEGORIES, fill_value=0)
    z = counts.values.tolist()
    hover = [[f"A {a}, B {b}: {counts.loc[a, b]} paper(s)" for b in CATEGORIES] for a in CATEGORIES]
    fig = heatmap(z, CATEGORIES, CATEGORIES, zmin=0, zmax=max(1, counts.values.max()),
                  fmt=lambda v: str(int(v)) if v else "", hover=hover, cell=34,
                  x_title=f"B: {b_label}", y_title=f"A: {a_label}", scale_title="papers")
    for i in range(len(CATEGORIES)):  # outline the agreement diagonal
        fig.add_shape(type="rect", x0=i - 0.5, x1=i + 0.5, y0=i - 0.5, y1=i + 0.5,
                      line=dict(color=theme()["ink"], width=2.5), layer="above")
    st.markdown("**Where they land** — rows are A's consensus, columns B's; the outlined diagonal is agreement.")
    show(fig, counts.reset_index().rename(columns={"index": "A \\ B"}), key=f"confusion_{dataset}")

    papers = {p["paper_id"]: p for p in registry.dataset(dataset).papers}
    differ = [p for p in shared if ca[p] != cb[p]]
    st.markdown(f"**The {len(differ)} paper(s) they disagree on**")
    if not differ:
        return
    st.dataframe(pd.DataFrame([{"Paper": p, "Title": papers.get(p, {}).get("title", ""),
                                "A": ca[p], "B": cb[p]} for p in differ]),
                 hide_index=True, width="stretch")
    for p in differ[:60]:
        with st.expander(f"{ca[p]} vs {cb[p]} — {papers.get(p, {}).get('title', p)}"):
            left, right = st.columns(2)
            left.markdown(f"**A: {a_label}**\n\n" + judgement_lines(ctx, A, p))
            right.markdown(f"**B: {b_label}**\n\n" + judgement_lines(ctx, B, p))


def render(ctx: Context | None) -> None:
    st.title("Compare")
    st.caption("Experiments are compared through their consensus, on the papers both have judged. "
               "Only experiments on the same dataset can be compared paper by paper.")
    if not need(ctx, 2):
        return
    metric = st.segmented_control("Measure", list(METRICS), default="Exact category", key="cmp_metric") \
        or "Exact category"
    datasets = [d for d in dict.fromkeys(ctx.experiments.dataset)
                if (ctx.experiments.dataset == d).sum() >= 2]
    if not datasets:
        st.info("No dataset has two experiments in view.")
        return
    tabs = st.tabs(datasets) if len(datasets) > 1 else [st.container()]
    for ds, tab in zip(datasets, tabs):
        with tab:
            st.subheader("Agreement between experiments")
            fig, table = matrix(ctx, ds, metric)
            show(fig, table, key=f"matrix_{ds}_{metric}")

            st.subheader("Any two experiments")
            pair(ctx, ds)
