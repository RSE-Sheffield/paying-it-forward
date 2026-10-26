"""One row per experiment: what it is, how consistent it is, how it compares
with the humans, what it costs, and how it spreads papers across categories."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from pif import metrics
from pif.taxonomy import CATEGORIES, data, software

from .common import Context, heatmap, need, pct, share_column, show


def human_agreement(ctx: Context) -> dict[str, float]:
    """Each experiment's consensus against every human experiment in view on the
    same dataset (never itself), pooled: matches / comparisons."""
    cons = ctx.consensus_map()
    exps = ctx.experiments.set_index("experiment")
    out = {}
    for e, row in exps.iterrows():
        humans = [h for h in exps.index if h != e and exps.reviewer_kind[h] == "human"
                  and exps.dataset[h] == row.dataset]
        matches = total = 0
        for h in humans:
            a = metrics.agreement(cons.get(e, {}), cons.get(h, {}))
            if a["n"]:
                matches += a["exact"] * a["n"]
                total += a["n"]
        out[e] = matches / total if total else None
    return out


def summary(ctx: Context) -> pd.DataFrame:
    cons = ctx.consensus()
    vs_humans = human_agreement(ctx)
    rows = []
    for _, e in ctx.experiments.iterrows():
        j = ctx.judgements[ctx.judgements.experiment == e.experiment]
        c = cons[cons.experiment == e.experiment]
        sc = metrics.self_consistency(j)
        rows.append({
            "#": e.tag, "Experiment": e.label,
            "Runs": sc["runs"], "Papers": len(c),
            "Run-to-run agreement": sc["pairwise"],
            "Agreement with humans": vs_humans.get(e.experiment),
            "Software generated (S1)": (c.category.map(software) == "S1").mean() if len(c) else None,
            "Data generated (D1)": (c.category.map(data) == "D1").mean() if len(c) else None,
            "Seconds / paper": j.duration_s.mean() if "duration_s" in j and j.duration_s.notna().any() else None,
            "Input tokens / paper": j.prompt_tokens.mean() if "prompt_tokens" in j and j.prompt_tokens.notna().any() else None,
            "Minutes / paper (human)": j.minutes.mean() if "minutes" in j and j.minutes.notna().any() else None,
        })
    return pd.DataFrame(rows)


def render(ctx: Context | None) -> None:
    st.title("Overview")
    st.caption("An **experiment** is one combination of dataset, criteria version, prompt version, input and "
               "reviewer; its **runs** are repeats. Each experiment is reduced to a **consensus** per paper — the "
               "category its runs chose most often, ties broken in taxonomy order.")
    if not need(ctx):
        return

    cols = st.columns(4)
    cols[0].metric("Experiments", len(ctx.experiments))
    cols[1].metric("Runs", len(ctx.runs))
    cols[2].metric("Judgements", f"{len(ctx.judgements):,}")
    cols[3].metric("Datasets", ctx.experiments.dataset.nunique())
    if ctx.notes:
        with st.expander(f"Data notes ({len(ctx.notes)})", icon=":material/info:"):
            for n in ctx.notes:
                st.markdown(f"- {n}")

    st.subheader("Experiments")
    st.caption("Each experiment is labelled by the factors that differ between the experiments in view; the "
               "Factors page has every version in full. Run-to-run agreement is the mean, over every pair of runs, of the share of papers they agree on. "
               "Agreement with humans pools every comparison between this experiment's consensus and each human "
               "experiment in view on the same dataset, never with itself (so a person's pass under one criteria "
               "version does count against their pass under another).")
    table = summary(ctx)
    shares = ["Run-to-run agreement", "Agreement with humans", "Software generated (S1)", "Data generated (D1)"]
    table[shares] = table[shares].astype(float) * 100
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "Run-to-run agreement": share_column(),
        "Agreement with humans": share_column(),
        "Software generated (S1)": share_column(bar=False),
        "Data generated (D1)": share_column(bar=False),
        "Seconds / paper": st.column_config.NumberColumn(format="%.1f"),
        "Input tokens / paper": st.column_config.NumberColumn(format="%d"),
        "Minutes / paper (human)": st.column_config.NumberColumn(format="%.1f"),
    })

    st.subheader("Category mix")
    st.caption("Share of each experiment's papers whose consensus falls in each category. Shares, not counts, "
               "so datasets of different sizes line up.")
    cons = ctx.consensus()
    labels = list(ctx.experiments["tagged"])
    z, hover, rows = [], [], []
    for _, e in ctx.experiments.iterrows():
        c = cons[cons.experiment == e.experiment]
        counts = c.category.value_counts()
        n = len(c)
        z.append([counts.get(k, 0) / n if n else None for k in CATEGORIES])
        hover.append([f"{e.label}<br>{k}: {counts.get(k, 0)} of {n} papers" for k in CATEGORIES])
        rows.append({"Experiment": e.label, **{k: counts.get(k, 0) for k in CATEGORIES}, "Papers": n})
    zmax = max((v for r in z for v in r if v is not None), default=1)
    fig = heatmap(z, CATEGORIES, labels, zmin=0, zmax=zmax,
                  fmt=lambda v: pct(v) if v else "", hover=hover, cell=36, scale_title="share")
    show(fig, pd.DataFrame(rows), key="mix")
