"""Where reviewers disagree, for the Compare page: how many papers split on each
boundary between levels (S2 against S3, D1 against D3, ...), and a paper by
experiment grid of the level each one chose, on each axis."""
from __future__ import annotations

from collections import Counter

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from pif import registry
from pif.taxonomy import DATA, SOFTWARE, data, software

from .common import Context, ink_on, show, style, theme
from .overview import GROUPS, reviewer_group

# (title, {level: name}, category -> level) for each axis.
AXES = [("Software", SOFTWARE, software), ("Data", DATA, data)]
EVERYONE = "Everyone in view"
MAX_ROWS = 80          # papers drawn in the grid; the most split come first
TITLE_CHARS = 46


def _short(name: str) -> str:
    """'Software generated' -> 'generated', 'No data' -> 'none'."""
    return "none" if name.startswith("No ") else name.split()[-1]


def _boundary(levels: tuple[str, ...], names: dict[str, str]) -> str:
    if len(levels) == len(names):
        return "<b>All three</b>"
    return f"<b>{' / '.join(levels)}</b><br>{' / '.join(_short(names[l]) for l in levels)}"


def members(ctx: Context, dataset: str, key: str) -> pd.DataFrame:
    """The experiments to look for disagreement among: everyone in view on the
    dataset, or one group of reviewers, chosen above the charts."""
    exps = ctx.experiments[ctx.experiments.dataset == dataset]
    group = pd.Series([reviewer_group(r, k) for r, k in zip(exps.reviewer, exps.reviewer_kind)], index=exps.index)
    options = {EVERYONE: exps}
    options.update({plural: exps[group == g] for g, _, plural, _ in GROUPS if (group == g).sum() >= 2})
    if len(options) == 1:
        return exps
    choice = st.segmented_control("Disagreement among", list(options), default=EVERYONE, key=key) or EVERYONE
    return options[choice]


def paper_splits(ctx: Context, exps: pd.DataFrame) -> pd.DataFrame:
    """One row per paper: for each axis, the levels the experiments' consensus
    lands on (`<axis>_levels`, a tuple; more than one means a split) and the
    share of experiments outside the largest camp (`<axis>_dissent`)."""
    cons = ctx.consensus()
    cons = cons[cons.experiment.isin(exps.experiment)]
    rows = []
    for pid, g in cons.groupby("paper_id", sort=False):
        row = {"paper_id": pid, "experiments": len(g)}
        for title, names, level in AXES:
            c = Counter(g.category.map(level))
            row[f"{title}_levels"] = tuple(l for l in names if l in c)
            row[f"{title}_dissent"] = 1 - max(c.values()) / len(g)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out[out.experiments >= 2]
    out["dissent"] = out[[f"{t}_dissent" for t, _, _ in AXES]].sum(axis=1)
    out["axes_split"] = sum(out[f"{t}_levels"].map(len).gt(1) for t, _, _ in AXES)
    return out.sort_values(["axes_split", "dissent"], ascending=False, kind="stable").reset_index(drop=True)


def boundaries(splits: pd.DataFrame, titles: dict[str, str]) -> tuple[go.Figure, pd.DataFrame]:
    """A bar per boundary on each axis: the number of papers whose experiments
    land on exactly those levels. Every boundary is drawn, most common first, so
    one nobody splits on shows as an empty row."""
    t = theme()
    color = t["cat"][0]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.14, shared_xaxes=True,
                        subplot_titles=[title for title, _, _ in AXES])
    table, top = [], 1
    for col, (title, names, _) in enumerate(AXES, start=1):
        keys = list(names)
        sets = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]] + [tuple(keys)]
        found = splits[f"{title}_levels"]
        papers = {s: list(splits.paper_id[found == s]) for s in sets}
        sets.sort(key=lambda s: -len(papers[s]))
        counts = [len(papers[s]) for s in sets]
        top = max(top, *counts)
        hover = []
        for s in sets:
            shown = [f"· {titles.get(p, p)[:70]}" for p in papers[s][:8]]
            more = [f"· and {len(papers[s]) - 8} more"] if len(papers[s]) > 8 else []
            hover.append(f"{' / '.join(s)}: {len(papers[s])} paper(s)<br>" + "<br>".join(shown + more))
            table.append({"Axis": title, "Boundary": " / ".join(s), "Papers": len(papers[s]),
                          "Paper ids": ", ".join(papers[s])})
        fig.add_bar(y=[_boundary(s, names) for s in sets], x=counts, orientation="h", width=0.55,
                    row=1, col=col, marker=dict(color=color), showlegend=False,
                    text=[str(c) for c in counts], textposition="outside", cliponaxis=False,
                    textfont=dict(color=t["ink"], size=12), customdata=hover,
                    hovertemplate="%{customdata}<extra></extra>")
        split = int((found.map(len) > 1).sum())
        fig.layout.annotations[col - 1].update(
            text=f"<b>{title} axis</b> · {split} of {len(splits)} papers split", x=fig.layout[
                "xaxis" if col == 1 else f"xaxis{col}"].domain[0], xanchor="left",
            font=dict(size=13, color=t["ink"]))
    style(fig, height=250)
    fig.update_xaxes(range=[0, top * 1.12], dtick=max(1, round(top / 6)), showgrid=True, fixedrange=True,
                     title="Papers")
    fig.update_yaxes(autorange="reversed", showgrid=False, fixedrange=True)
    fig.update_layout(barcornerradius=4, margin=dict(t=34, b=8))
    return fig, pd.DataFrame(table)


def grid(ctx: Context, exps: pd.DataFrame, splits: pd.DataFrame,
         titles: dict[str, str]) -> tuple[go.Figure, pd.DataFrame]:
    """Papers down, experiments across, once per axis. A cell is filled with the
    level the experiment chose; when its runs chose more than one, the cell is
    divided by the share of runs choosing each. A blank cell is a paper the
    experiment has no valid judgement for."""
    t = theme()
    colors = t["mix"][:3]  # the validated three-step ramp: level 1 furthest from the surface
    exps = exps.reset_index(drop=True)
    column = {e: i for i, e in enumerate(exps.experiment)}
    row_of = {p: i for i, p in enumerate(splits.paper_id)}
    j = ctx.judgements
    j = j[j.valid & j.experiment.isin(column) & j.paper_id.isin(row_of)]
    pad, span = 0.04, 0.92

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.03,
                        subplot_titles=[f"<b>{title} axis</b>" for title, _, _ in AXES])
    for col, (title, names, level) in enumerate(AXES, start=1):
        votes = j.assign(level=j.category.map(level)).groupby(["experiment", "paper_id", "level"]).size()
        totals = votes.groupby(level=[0, 1]).sum()
        marks = {l: dict(y=[], x=[], base=[], text=[], hover=[]) for l in names}
        for (e, p), n in totals.items():
            left = column[e] + pad
            for l, name in names.items():
                k = int(votes.get((e, p, l), 0))
                if not k:
                    continue
                m = marks[l]
                m["y"].append(row_of[p])
                m["x"].append(span * k / n)
                m["base"].append(left)
                m["text"].append(l if k / n >= 0.5 else "")
                m["hover"].append(f"{exps.tagged[column[e]]}<br>{titles.get(p, p)[:90]}<br>{l} · {name}"
                                  + (f": {k} of {n} runs" if n > 1 else ""))
                left += span * k / n
        for (l, name), color in zip(names.items(), colors):
            m = marks[l]
            fig.add_bar(y=m["y"], x=m["x"], base=m["base"], orientation="h", width=0.84, row=1, col=col,
                        name=f"{l[1]} · {_short(name)}", legendgroup=l[1], showlegend=col == 1,
                        marker=dict(color=color, line=dict(color=t["page"], width=1)),
                        text=m["text"], textposition="inside", insidetextanchor="middle", textangle=0,
                        textfont=dict(color=ink_on(color), size=10), customdata=m["hover"],
                        hovertemplate="%{customdata}<extra></extra>")
        fig.update_xaxes(row=1, col=col, range=[0, len(exps)], tickvals=[i + 0.5 for i in range(len(exps))],
                         ticktext=list(exps.tag), side="top", showgrid=False, fixedrange=True)

    labels = [f"{titles.get(p, p)[:TITLE_CHARS]}{'…' if len(titles.get(p, p)) > TITLE_CHARS else ''}"
              for p in splits.paper_id]
    style(fig, height=22 * len(splits) + 110)
    fig.update_yaxes(range=[len(splits) - 0.5, -0.5], tickvals=list(range(len(splits))), ticktext=labels,
                     showgrid=False, fixedrange=True)
    fig.update_annotations(font=dict(size=13, color=t["ink"]), yshift=22)
    fig.update_layout(barmode="overlay", uniformtext=dict(mode="hide", minsize=9), margin=dict(t=64, b=8),
                      legend=dict(orientation="v", x=1.01, xanchor="left", y=1, yanchor="top",
                                  title=dict(text="Level")))

    cons = ctx.consensus()
    wide = cons[cons.experiment.isin(column) & cons.paper_id.isin(row_of)].pivot(
        index="paper_id", columns="experiment", values="category").reindex(
        index=splits.paper_id, columns=exps.experiment)
    wide.columns = list(exps.label)
    wide.insert(0, "Title", [titles.get(p, "") for p in wide.index])
    return fig, wide.rename_axis(index="Paper").reset_index()


def render(ctx: Context, dataset: str) -> None:
    exps = members(ctx, dataset, key=f"split_group_{dataset}")
    splits = paper_splits(ctx, exps)
    if splits.empty:
        st.info("No paper here has been judged by two of these experiments.")
        return
    titles = {p["paper_id"]: p.get("title") or p["paper_id"] for p in registry.dataset(dataset).papers}

    st.markdown("**Which boundaries they split on** — a paper counts once per axis, under the set of levels "
                "the experiments' consensus lands on.")
    fig, table = boundaries(splits, titles)
    show(fig, table, key=f"boundaries_{dataset}")

    agreed = int((splits.axes_split == 0).sum())
    st.markdown("**Every paper, by experiment** — the level each experiment chose on each axis, most split "
                "papers first. A divided cell is an experiment whose runs chose more than one level.")
    keep_agreed = st.toggle(f"Include the {agreed} paper(s) with no disagreement", value=len(splits) <= 40,
                            key=f"split_agreed_{dataset}") if agreed else False
    rows = splits if keep_agreed else splits[splits.axes_split > 0]
    if rows.empty:
        st.info("These experiments agree on every paper.")
        return
    if len(rows) > MAX_ROWS:
        st.caption(f"Showing the {MAX_ROWS} most split of {len(rows)} papers.")
        rows = rows.head(MAX_ROWS)
    fig, table = grid(ctx, exps, rows.reset_index(drop=True), titles)
    show(fig, table, key=f"grid_{dataset}")
    st.caption("Columns: " + " · ".join(exps.tagged))
