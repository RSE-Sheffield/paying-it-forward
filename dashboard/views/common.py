"""Data loading, the sidebar filters, and chart helpers shared by every page."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pif import metrics, paths, registry, store
from pif.registry import KINDS

# ------------------------------------------------------------------ palette
# The dataviz reference palette. Sequential is one hue (blue), light -> dark
# on the light surface and the same steps reversed on the dark one, so "more"
# is always "further from the surface". Model/human are categorical slots 1-2,
# validated for both modes.
_SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", seq=_SEQ, model="#2a78d6", human="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", seq=_SEQ[::-1], model="#3987e5", human="#d95926"),
}
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
MISSING = "—"


def theme() -> dict:
    try:
        mode = st.context.theme.type or "light"
    except AttributeError:
        mode = "light"
    return THEMES.get(mode, THEMES["light"])


def _hex_rgb(h: str) -> tuple[int, int, int]:
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


def _ramp_at(ramp: list[str], t: float) -> tuple[int, int, int]:
    t = min(max(t, 0.0), 1.0) * (len(ramp) - 1)
    i = min(int(t), len(ramp) - 2)
    a, b = _hex_rgb(ramp[i]), _hex_rgb(ramp[i + 1])
    f = t - i
    return tuple(round(x + (y - x) * f) for x, y in zip(a, b))


def _ink_on(rgb: tuple[int, int, int]) -> str:
    """White or near-black text, whichever reads on this fill."""
    def lin(c):
        c /= 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    lum = 0.2126 * lin(rgb[0]) + 0.7152 * lin(rgb[1]) + 0.0722 * lin(rgb[2])
    return "#0b0b0b" if lum > 0.36 else "#ffffff"


def style(fig: go.Figure, height: int) -> go.Figure:
    t = theme()
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=t["ink2"]),
        hoverlabel=dict(font=dict(family=FONT)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(gridcolor=t["grid"], linecolor=t["axis"], zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor=t["grid"], linecolor=t["axis"], zeroline=False, automargin=True)
    return fig


def heatmap(z: list[list], x: list[str], y: list[str], *, zmin: float, zmax: float,
            fmt, hover: list[list[str]], x_title: str = "", y_title: str = "",
            cell: int = 44, scale_title: str = "") -> go.Figure:
    """A sequential heatmap with readable in-cell labels. None cells are left blank."""
    t = theme()
    span = (zmax - zmin) or 1
    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=y, zmin=zmin, zmax=zmax, xgap=2, ygap=2,
        colorscale=[[i / (len(t["seq"]) - 1), c] for i, c in enumerate(t["seq"])],
        customdata=hover, hovertemplate="%{customdata}<extra></extra>",
        colorbar=dict(title=dict(text=scale_title, side="right"), thickness=10, outlinewidth=0),
    ))
    for i, row in enumerate(z):
        for j, v in enumerate(row):
            if v is None or fmt(v) == "":
                continue
            fig.add_annotation(x=x[j], y=y[i], text=fmt(v), showarrow=False,
                               font=dict(size=11, color=_ink_on(_ramp_at(t["seq"], (v - zmin) / span))))
    fig.update_xaxes(title=x_title, side="top", tickangle=0, showgrid=False, type="category")
    fig.update_yaxes(title=y_title, autorange="reversed", showgrid=False, type="category")
    return style(fig, height=max(160, cell * len(y) + 120))


def show(fig: go.Figure, table: pd.DataFrame | None = None, key: str | None = None) -> None:
    """A chart and its table twin."""
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False}, key=key)
    if table is not None:
        with st.expander("Show as table"):
            st.dataframe(table, hide_index=True, width="stretch")


def data_range(values, lo: float, hi: float) -> tuple[float, float]:
    """The colour range for a heatmap: the observed values widened to tenths,
    within [lo, hi], so small differences stay visible."""
    vals = [v for v in values if v is not None and not pd.isna(v)]
    if not vals:
        return lo, hi
    a, b = max(lo, math.floor(min(vals) * 10) / 10), min(hi, math.ceil(max(vals) * 10) / 10)
    return (a, b) if b > a else (lo, hi)


def share_column(label: str | None = None, bar: bool = True):
    """A 0-1 share shown as a whole percentage (expects the value x100)."""
    if bar:
        return st.column_config.ProgressColumn(label, format="%.0f%%", min_value=0, max_value=100)
    return st.column_config.NumberColumn(label, format="%.0f%%")


def pct(v) -> str:
    return MISSING if v is None or pd.isna(v) else f"{v:.0%}"


def num(v, digits=2) -> str:
    return MISSING if v is None or pd.isna(v) else f"{v:.{digits}f}"


# -------------------------------------------------------------------- data
def _stamp() -> float:
    """Latest modification time under experiments/, so new runs invalidate the cache."""
    latest = 0.0
    for dirpath, dirnames, filenames in os.walk(paths.EXPERIMENTS):
        dirnames[:] = [d for d in dirnames if not d.startswith(("_", "."))]
        for f in filenames:
            latest = max(latest, os.stat(os.path.join(dirpath, f)).st_mtime)
    return latest


@st.cache_data(show_spinner="Loading experiments…")
def _load(stamp: float):
    return store.load()


@st.cache_data(show_spinner=False)
def _consensus(stamp: float, run_keys: tuple[str, ...]) -> pd.DataFrame:
    _, _, j = _load(stamp)
    return metrics.consensus(j[j.run_key.isin(run_keys)])


def experiment_labels(exps: pd.DataFrame) -> pd.Series:
    """Short labels naming only the factors that differ between the experiments shown."""
    varying = [k for k in KINDS if exps[k].nunique() > 1] or ["reviewer"]
    prefix = {"criteria": "criteria ", "prompt": "prompt "}

    def lab(row):
        return " · ".join(f"{prefix.get(k, '')}{row[k]}" for k in varying)
    return exps.apply(lab, axis=1)


@dataclass
class Context:
    """Everything in view after the sidebar filters."""
    stamp: float
    experiments: pd.DataFrame        # one row per experiment, with `label`, `tag` (#n) and `tagged` (both)
    runs: pd.DataFrame
    judgements: pd.DataFrame
    all_runs: pd.DataFrame
    notes: list[str] = field(default_factory=list)

    def label(self, experiment: str) -> str:
        return self.experiments.set_index("experiment").label.get(experiment, experiment)

    def consensus(self) -> pd.DataFrame:
        return _consensus(self.stamp, tuple(sorted(self.runs.run_key)))

    def consensus_map(self) -> dict[str, dict[str, str]]:
        """{experiment: {paper_id: consensus category}}."""
        out: dict[str, dict[str, str]] = {}
        for exp, g in self.consensus().groupby("experiment"):
            out[exp] = dict(zip(g.paper_id, g.category))
        return out

    def is_human(self, experiment: str) -> bool:
        return bool(self.experiments.set_index("experiment").reviewer_kind.get(experiment) == "human")


def _sort_key(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(_h=df.reviewer_kind.eq("human")).sort_values(
        ["dataset", "criteria", "prompt", "_h", "input", "reviewer"]).drop(columns="_h")


def build_context() -> Context | None:
    registry.load.cache_clear()  # pick up registry edits without restarting
    stamp = _stamp()
    exps, runs, judgements = _load(stamp)
    if exps.empty:
        st.info("No experiments yet. Run one with `uv run pif run …` and refresh.")
        return None
    exps = _sort_key(exps)

    sb = st.sidebar
    sb.markdown("### In view")
    chosen = {}
    for k, title in [("dataset", "Datasets"), ("criteria", "Criteria versions"), ("prompt", "Prompt versions"),
                     ("input", "Inputs"), ("reviewer", "Reviewers")]:
        options = list(dict.fromkeys(exps[k]))
        chosen[k] = sb.multiselect(title, options, default=options, key=f"f_{k}")
    unfinished = sb.toggle("Include unfinished runs", value=False, key="f_unfinished",
                           help="Runs that were interrupted or had failed requests. Off by default so every "
                                "run in view covers the whole dataset.")
    if sb.button("Reset filters"):
        for k in [*KINDS, "unfinished"]:
            st.session_state.pop(f"f_{k}", None)
        st.rerun()

    mask = pd.Series(True, index=exps.index)
    for k in KINDS:
        mask &= exps[k].isin(chosen[k])
    r = runs[runs.experiment.isin(exps[mask].experiment)]
    if not unfinished:
        r = r[r.status == "complete"]
    j = judgements[judgements.run_key.isin(r.run_key)] if not judgements.empty else judgements
    in_view = exps[mask & exps.experiment.isin(r.experiment)].copy()
    if not in_view.empty:
        in_view["label"] = experiment_labels(in_view)
        # A short tag, the same on every page, for axes and columns too narrow for the label.
        in_view["tag"] = [f"#{i}" for i in range(1, len(in_view) + 1)]
        in_view["tagged"] = in_view.tag + "  " + in_view.label

    notes = []
    for _, e in exps.iterrows():
        try:
            current = store.Experiment.current_fingerprints(store.ExperimentKey(**{k: e[k] for k in KINDS}))
        except registry.RegistryError as err:
            notes.append(f"{e.experiment}: {err}")
            continue
        changed = [k for k in KINDS if e.fingerprints.get(k) != current[k]]
        if changed:
            notes.append(f"{e.experiment}: {', '.join(changed)} edited since its runs were made "
                         "(`uv run pif check` for details)")
    hidden = runs[runs.experiment.isin(exps[mask].experiment) & (runs.status != "complete")]
    if len(hidden) and not unfinished:
        notes.append(f"{len(hidden)} unfinished run(s) hidden: "
                     + ", ".join(f"{x.experiment}/{x.run} ({x.status})" for x in hidden.itertuples()))
    if not j.empty and (~j.valid).any():
        bad = j[~j.valid]
        notes.append(f"{len(bad)} judgement(s) without a valid category are left out of every statistic: "
                     + ", ".join(f"{x.run_key} paper {x.paper_id}" for x in bad.head(10).itertuples()))

    sb.caption(f"{len(in_view)} experiment(s) · {len(r)} run(s) · {len(j):,} judgements")
    return Context(stamp=stamp, experiments=in_view, runs=r, judgements=j, all_runs=runs, notes=notes)


def need(ctx: Context | None, n: int = 1) -> bool:
    """True if there is enough in view to draw a page; otherwise says why."""
    if ctx is None:
        return False
    if len(ctx.experiments) < n:
        st.info("Nothing in view — widen the filters in the sidebar." if n == 1
                else f"This page needs at least {n} experiments in view.")
        return False
    return True

