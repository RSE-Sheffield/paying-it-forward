"""The experiments in view, what the reviewers together classified the papers
as, and how each experiment spreads papers across categories."""
from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from pif import metrics
from pif.taxonomy import CATEGORIES, DATA, SOFTWARE, data, label, software

from .common import Context, ink_on, need, pct, show, style, theme


TIED = "Tied"


def reviewer_votes(ctx: Context) -> pd.DataFrame:
    """One row per (dataset, paper, reviewer): that reviewer's single vote. A
    reviewer with several experiments in view on a dataset votes for the
    category most of those experiments' consensuses chose, ties broken in
    taxonomy order."""
    exps = ctx.experiments[["experiment", "dataset", "reviewer"]]
    cons = ctx.consensus().merge(exps, on="experiment")
    votes = cons.groupby(["dataset", "paper_id", "reviewer"], sort=False).category.agg(metrics.modal)
    return votes.reset_index()


def paper_categories(ctx: Context) -> tuple[go.Figure, pd.DataFrame, dict]:
    """How many papers the reviewers in view put in each category, one vote per
    reviewer: a paper goes to the category most reviewers chose, or to "Tied"
    when two or more categories share the most votes."""
    t = theme()
    votes = reviewer_votes(ctx)
    papers = votes.groupby(["dataset", "paper_id"], sort=False).category.agg(
        winner=lambda v: TIED if metrics.is_tied(v) else metrics.modal(v),
        voters="size", unanimous=lambda v: v.nunique() == 1)
    keys = [*CATEGORIES, TIED]
    counts = papers.winner.value_counts().reindex(keys, fill_value=0)
    if not counts[TIED]:
        keys, counts = CATEGORIES, counts[CATEGORIES]
    total = int(counts.sum())
    names = [TIED if k == TIED else label(k) for k in keys]
    colors = [t["muted"] if k == TIED else c for k, c in zip(keys, [*t["mix"], None])]
    fig = go.Figure(go.Bar(
        y=keys, x=list(counts), orientation="h", width=0.7, marker=dict(color=colors),
        text=[str(n) if n else "" for n in counts], textposition="outside", cliponaxis=False,
        textfont=dict(color=t["ink"]), customdata=[[n, pct(c / total)] for n, c in zip(names, counts)],
        hovertemplate=f"%{{y}} · %{{customdata[0]}}<br>%{{x}} of {total} papers (%{{customdata[1]}})"
                      "<extra></extra>"))
    fig.update_xaxes(title="Papers", range=[0, max(int(counts.max()), 1) * 1.15], showgrid=True, fixedrange=True,
                     tickformat="d")
    fig.update_yaxes(autorange="reversed", showgrid=False, fixedrange=True, type="category")
    style(fig, height=30 * len(keys) + 70)
    fig.update_layout(barcornerradius=4, showlegend=False)
    table = pd.DataFrame({"Category": keys, "Meaning": names, "Papers": list(counts),
                          "Share": [pct(c / total) for c in counts]})
    info = dict(papers=total, reviewers=votes.reviewer.nunique(), tied=int((papers.winner == TIED).sum()),
                unanimous=int(papers.unanimous.sum()), min_voters=int(papers.voters.min()),
                max_voters=int(papers.voters.max()))
    return fig, table, info


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


def _dodge(values: list[float], eps: float, step: float) -> list[float]:
    """Horizontal offsets so dots closer than `eps` in value don't sit on top of
    each other: each takes the nearest free slot 0, +step, -step, +2·step, ..."""
    placed: list[tuple[float, float]] = []
    out = [0.0] * len(values)
    for i in sorted(range(len(values)), key=lambda i: values[i]):
        n = 0
        while True:
            slot = (n + 1) // 2 * step * (1 if n % 2 else -1)
            if all(abs(values[i] - v) >= eps or abs(slot - s) > 1e-9 for v, s in placed):
                break
            n += 1
        placed.append((values[i], slot))
        out[i] = slot
    return out


# Reviewer groups in the axis dot plot, in their left-to-right order: (key,
# legend name, plural, theme colour). A human reviewer whose id ends in
# AI_REVIEW_SUFFIX is a person's codes revised by an AI review of them.
AI_REVIEW_SUFFIX = "-plus-ai-review"
GROUPS = [("model", "Model", "Models", "model"), ("human", "Human", "Humans", "human"),
          ("human-ai", "Human + AI review", "Human + AI reviews", "human_ai")]


def reviewer_group(reviewer: str, kind: str) -> str:
    if kind != "human":
        return "model"
    return "human-ai" if reviewer.endswith(AI_REVIEW_SUFFIX) else "human"


def axis_mix(ctx: Context) -> tuple[go.Figure, pd.DataFrame]:
    """A dot plot per axis: each experiment's share of papers at every software
    level, and beside it every data level. Within a level, models, humans and
    humans with an AI review sit side by side, and a faint bar behind each group
    spans its lowest to highest dot, so the bar's height is how far reviewers in
    that group disagree on how often the level applies."""
    t = theme()
    cons = ctx.consensus()
    exps = ctx.experiments.reset_index(drop=True)
    axes = [("Software", SOFTWARE, cons.category.map(software)), ("Data", DATA, cons.category.map(data))]
    counts = [pd.crosstab(cons.experiment, levels).reindex(index=exps.experiment, columns=list(names),
                                                           fill_value=0)
              for _, names, levels in axes]
    totals = counts[0].sum(axis=1)
    shares = [c.div(totals.where(totals > 0), axis=0) for c in counts]
    top = min(1.0, math.ceil(max(float(s.max().max()) for s in shares) * 4 + 0.2) / 4) or 1.0

    group = pd.Series([reviewer_group(r, k) for r, k in zip(exps.reviewer, exps.reviewer_kind)], index=exps.index)
    groups = [g for g in GROUPS if (group == g[0]).any()]
    # Groups spread evenly around each level's tick; narrower with three.
    spacing = {1: 0.0, 2: 0.38, 3: 0.29}[len(groups)]
    offsets = {g[0]: (i - (len(groups) - 1) / 2) * spacing for i, g in enumerate(groups)}
    bar_width, step, dot = (0.3, 0.065, 10) if len(groups) < 3 else (0.24, 0.056, 9)

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.05,
                        subplot_titles=[title for title, _, _ in axes])
    for col, ((title, names, _), c, s) in enumerate(zip(axes, counts, shares), start=1):
        for kind, kind_name, plural, color_key in groups:
            color = t[color_key]
            mine = exps[group == kind]
            xs, ys, hover = [], [], []
            for i, (level, level_name) in enumerate(names.items()):
                vals = s.loc[mine.experiment, level]
                vals = vals[vals.notna()]
                if vals.empty:
                    continue
                x = i + offsets[kind]
                if len(vals) > 1:
                    lo, hi = float(vals.min()), float(vals.max())
                    fig.add_bar(x=[x], y=[max(hi - lo, 0.004)], base=[lo], width=bar_width, row=1, col=col,
                                marker=dict(color=color, opacity=0.18), showlegend=False,
                                hovertemplate=f"{plural} · {level} {level_name.lower()}<br>"
                                              f"{pct(lo)} to {pct(hi)} across {len(vals)} experiments"
                                              "<extra></extra>")
                for e, dx in zip(vals.index, _dodge(list(vals), eps=0.03 * top, step=step)):
                    row = mine.set_index("experiment").loc[e]
                    xs.append(x + dx)
                    ys.append(float(vals[e]))
                    hover.append(f"{row.tagged}<br>{level} · {level_name}<br>"
                                 f"{c.at[e, level]} of {totals[e]} papers ({pct(vals[e])})")
            fig.add_scatter(x=xs, y=ys, mode="markers", cliponaxis=False, row=1, col=col, name=kind_name,
                            legendgroup=kind, showlegend=col == 1, customdata=hover,
                            hovertemplate="%{customdata}<extra></extra>",
                            marker=dict(size=dot, color=color, line=dict(color=t["page"], width=1.5)))
        fig.update_xaxes(row=1, col=col, range=[-0.55, len(names) - 0.45], tickvals=list(range(len(names))),
                         ticktext=[f"<b>{k}</b><br>{v}" for k, v in names.items()])

    style(fig, height=420)
    fig.update_xaxes(showgrid=False, zeroline=False, fixedrange=True)
    fig.update_yaxes(range=[0, top], dtick=0.25, tickformat=".0%", showgrid=True, fixedrange=True)
    fig.update_yaxes(title="Share of papers", row=1, col=1)
    fig.update_annotations(font=dict(size=13, color=t["ink"]))
    fig.update_layout(barmode="overlay", barcornerradius=4, margin=dict(t=32, b=8),
                      legend=dict(orientation="v", x=1.01, xanchor="left", y=1, yanchor="top"))

    table = pd.concat(counts, axis=1).rename(index=dict(zip(exps.experiment, exps.label)))
    table = table.rename_axis(index="Experiment", columns=None).assign(Papers=totals.values)
    return fig, table.reset_index()


def summary(ctx: Context) -> pd.DataFrame:
    cons = ctx.consensus()
    papers = cons.groupby("experiment").size()
    runs = ctx.runs.groupby("experiment").size()
    return pd.DataFrame([{"#": e.tag, "Experiment": e.label, "Runs": int(runs.get(e.experiment, 0)),
                          "Papers": int(papers.get(e.experiment, 0))}
                         for e in ctx.experiments.itertuples()])


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

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Experiments")
        st.caption("Each experiment is labelled by the factors that differ between the experiments in view; the "
                   "Factors page has every version in full.")
        st.dataframe(summary(ctx), hide_index=True, width="stretch")
    with right:
        st.subheader("What the papers were classified as")
        fig, table, info = paper_categories(ctx)
        voters = (f"{info['max_voters']}" if info["min_voters"] == info["max_voters"]
                  else f"{info['min_voters']}–{info['max_voters']}")
        caption = (f"Each of the {info['reviewers']} reviewers in view has one vote per paper, and a paper counts "
                   f"towards the category most of them chose. {info['papers']} papers, {voters} votes each; "
                   f"all reviewers agree on {info['unanimous']}.")
        if info["tied"]:
            caption += (f" On {info['tied']}, two or more categories share the most votes; those are counted as "
                        "**Tied** rather than forced into one.")
        st.caption(caption)
        show(fig, table, key="paper_categories")

    st.subheader("Category mix")
    st.markdown("**Software and data, separately**")
    st.caption("Each experiment's consensus categories split by axis. Each dot is one experiment's share of papers "
               "at that level; within a level, models are on the left, then humans, then humans whose codes were "
               "revised by an AI review. The faint bar behind each group runs from its lowest to its highest dot, "
               "so a tall bar means reviewers in that group disagree on how often the level applies. Hover a "
               "dot for the experiment and its counts. Filter to one dataset in the sidebar to compare like with "
               "like.")
    show(*axis_mix(ctx), key="axis_mix")

    st.markdown("**All nine categories**")
    st.caption("Each bar splits one experiment's papers by their consensus category: colour is the data axis, "
               "shade the software axis. Shares, not counts, so datasets of different sizes line up. Click a "
               "legend entry to hide that category.")
    show(*category_mix(ctx), key="mix")
