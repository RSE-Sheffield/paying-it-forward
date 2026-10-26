"""One row per experiment: what it is, how consistent it is, how it compares
with the humans, what it costs, and how it spreads papers across categories."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pif import metrics
from pif.taxonomy import CATEGORIES, DATA, SOFTWARE, data, label, software

from .common import Context, ink_on, need, pct, share_column, show, style, theme


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


def category_mix(ctx: Context) -> tuple[go.Figure, pd.DataFrame]:
    """A 100% bar per experiment, its segments in taxonomy order, so each data
    level is one block of hue and the software levels shade it."""
    t = theme()
    cons = ctx.consensus()
    exps = ctx.experiments
    counts = pd.crosstab(cons.experiment, cons.category).reindex(
        index=exps.experiment, columns=CATEGORIES, fill_value=0)
    totals = counts.sum(axis=1)
    fig = go.Figure()
    for k, color in zip(CATEGORIES, t["mix"]):
        share = (counts[k] / totals.where(totals > 0)).fillna(0)
        fig.add_bar(y=list(exps.tagged), x=list(share), orientation="h", width=0.55,
                    name=f"{k} {SOFTWARE[software(k)].lower()}",
                    legendgroup=data(k), legendgrouptitle_text=DATA[data(k)],
                    marker=dict(color=color, line=dict(color=t["page"], width=2)),
                    text=[pct(v) if v >= 0.06 else "" for v in share], textposition="inside",
                    insidetextanchor="middle", textangle=0, textfont=dict(color=ink_on(color)),
                    customdata=list(zip(counts[k], totals)),
                    hovertemplate=f"%{{y}}<br>{k} · {label(k)}<br>"
                                  "%{customdata[0]} of %{customdata[1]} papers (%{x:.0%})<extra></extra>")
    fig.update_xaxes(range=[0, 1], tickformat=".0%", dtick=0.25, showgrid=True)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    style(fig, height=max(300, 40 * len(exps) + 60))
    fig.update_layout(barmode="stack", barcornerradius=4, uniformtext=dict(mode="hide", minsize=11),
                      legend=dict(orientation="v", x=1.02, xanchor="left", y=1, yanchor="top",
                                  traceorder="grouped", groupclick="toggleitem"))
    table = counts.rename(index=dict(zip(exps.experiment, exps.label))).rename_axis(index="Experiment",
                                                                                     columns=None)
    return fig, table.assign(Papers=totals.values).reset_index()


HUMAN_AGREEMENT_HELP = (
    "This experiment's consensus is compared with each human experiment in view on the same dataset, on the "
    "papers both have judged. The matches are added up across all of those humans and divided by the total "
    "number of papers compared, so a human who judged more papers counts for more.\n\n"
    "An experiment is never compared with itself, but a person's pass under one criteria version does count "
    "against their pass under another. Blank when no other human experiment is in view on this dataset.")


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
            "Seconds / paper": j.duration_s.mean() if "duration_s" in j and j.duration_s.notna().any() else None,
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
               "Factors page has every version in full. Run-to-run agreement is the mean, over every pair of runs, of the share of papers they agree on.")
    table = summary(ctx)
    shares = ["Run-to-run agreement", "Agreement with humans"]
    table[shares] = table[shares].astype(float) * 100
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "Run-to-run agreement": share_column(),
        "Agreement with humans": share_column("Agreement with humans ⓘ", help=HUMAN_AGREEMENT_HELP),
        "Seconds / paper": st.column_config.NumberColumn(format="%.1f"),
    })

    st.subheader("Category mix")
    st.caption("Each bar splits one experiment's papers by their consensus category: colour is the data axis, "
               "shade the software axis. Shares, not counts, so datasets of different sizes line up. Click a "
               "legend entry to hide that category.")
    show(*category_mix(ctx), key="mix")
