"""Data loading, the sidebar filters, and chart helpers shared by every page."""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pif import metrics, paths, registry, store
from pif.registry import KINDS

# ------------------------------------------------------------------ palette
# The dataviz reference palette. Sequential is one hue (blue), light -> dark
# on the light surface and the same steps reversed on the dark one, so "more"
# is always "further from the surface". Model/human/human_ai (a person's codes
# revised by an AI review) are categorical slots 1-3, validated together for
# both modes; slot 3 is a darker step in light mode (#0ca06c), since the slot's
# own #1baf7a is under 3:1 against the surface.
#
# The nine categories are too many for distinct hues, so `mix` (in taxonomy
# order) encodes both axes: hue is the data axis (categorical slots 1-3: blue,
# orange, aqua) and shade the software axis, S1 furthest from the surface. Blue
# is the documented ramp; orange and aqua hold their slot's hue at the same
# lightness steps. Each three-shade ramp passes the ordinal checks in its mode,
# and every pair of touching segments clears CVD dE 12.
#
# `cat` is the eight categorical slots in their fixed order; experiments take
# their reviewer's slot, in shades between the `shades` lightness bounds (OKLCH
# L, first experiment furthest from the surface) when a reviewer has several.
# `page` is Streamlit's own background, for the gaps between touching marks.
_SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
THEMES = {
    "light": dict(surface="#fcfcfb", page="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", seq=_SEQ, model="#2a78d6", human="#eb6834",
                  human_ai="#0ca06c",
                  mix=["#184f95", "#2a78d6", "#6da7ec", "#892e01", "#c94908", "#fb7744",
                       "#056041", "#058f62", "#35be88"],
                  cat=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
                  shades=(0.42, 0.72)),
    "dark": dict(surface="#1a1a19", page="#0e1117", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", seq=_SEQ[::-1], model="#3987e5", human="#d95926",
                 human_ai="#199e70",
                 mix=["#9ec5f4", "#3987e5", "#184f95", "#fea98c", "#d95926", "#8a2d01",
                      "#68dbaa", "#1a9e70", "#036042"],
                 cat=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
                 shades=(0.82, 0.45)),
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


def _linear(c: int) -> float:
    c /= 255
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def _ink_on(rgb: tuple[int, int, int]) -> str:
    """White or near-black text, whichever reads on this fill."""
    lum = 0.2126 * _linear(rgb[0]) + 0.7152 * _linear(rgb[1]) + 0.0722 * _linear(rgb[2])
    return "#0b0b0b" if lum > 0.36 else "#ffffff"


def ink_on(fill: str) -> str:
    return _ink_on(_hex_rgb(fill))


def _oklch(h: str) -> tuple[float, float, float]:
    """sRGB hex -> OKLCH (lightness, chroma, hue in degrees)."""
    r, g, b = map(_linear, _hex_rgb(h))
    l, m, s = (x ** (1 / 3) for x in (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b,
                                       0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b,
                                       0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b))
    L = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s
    a = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s
    bb = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s
    return L, math.hypot(a, bb), math.degrees(math.atan2(bb, a)) % 360


def _oklch_hex(L: float, C: float, H: float) -> str:
    """OKLCH -> sRGB hex, giving up chroma until the colour fits the gamut."""
    while True:
        a, b = C * math.cos(math.radians(H)), C * math.sin(math.radians(H))
        l, m, s = ((L + 0.3963377774 * a + 0.2158037573 * b) ** 3, (L - 0.1055613458 * a - 0.0638541728 * b) ** 3,
                   (L - 0.0894841775 * a - 1.2914855480 * b) ** 3)
        rgb = (4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
               -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
               -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s)
        if C <= 0 or all(-1e-4 <= x <= 1 + 1e-4 for x in rgb):
            break
        C = max(C - 0.005, 0)

    def srgb(x):
        x = min(max(x, 0.0), 1.0)
        return 12.92 * x if x <= 0.0031308 else 1.055 * x ** (1 / 2.4) - 0.055
    return "#" + "".join(f"{round(255 * srgb(x)):02x}" for x in rgb)


def experiment_colors(exps: pd.DataFrame) -> dict[str, str]:
    """A colour per experiment: its reviewer's categorical slot (in the reviewer
    registry's order, so filters never repaint it), in shades when the reviewer
    has several experiments here. Reviewers past the eighth slot fold into gray."""
    t = theme()
    order = registry.ids("reviewer")
    out = {}
    for reviewer, g in exps.groupby("reviewer", sort=False):
        i = order.index(reviewer) if reviewer in order else len(order)
        base = t["cat"][i] if i < len(t["cat"]) else t["muted"]
        if len(g) == 1:
            out[g.experiment.iloc[0]] = base
            continue
        (lo, hi), (_, c, h) = t["shades"], _oklch(base)
        for n, e in enumerate(g.experiment):
            out[e] = _oklch_hex(lo + (hi - lo) * n / (len(g) - 1), c, h)
    return out


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


def pct(v) -> str:
    return MISSING if v is None or pd.isna(v) else f"{v:.0%}"


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


# The sidebar filters a person saved as their default: local to this checkout,
# not committed. For each factor it keeps what was selected and what could have
# been, so a value that appears later (a new reviewer, say) starts selected
# rather than silently hidden.
FILTERS_FILE = Path(__file__).resolve().parent.parent / ".saved_filters.json"


def _saved_filters() -> dict:
    try:
        saved = json.loads(FILTERS_FILE.read_text())
    except (OSError, ValueError):
        return {}
    return saved if isinstance(saved, dict) else {}


def _saved_default(saved: dict, kind: str, options: list[str]) -> list[str]:
    s = saved.get(kind)
    if not isinstance(s, dict):
        return options
    selected, seen = set(s.get("selected", [])), set(s.get("options", []))
    return [o for o in options if o in selected or o not in seen]


def _save_filters(chosen: dict[str, list[str]], options: dict[str, list[str]], unfinished: bool) -> None:
    saved = {k: {"selected": chosen[k], "options": options[k]} for k in chosen}
    FILTERS_FILE.write_text(json.dumps({**saved, "unfinished": unfinished}, indent=2) + "\n")


def build_context() -> Context | None:
    registry.load.cache_clear()  # pick up registry edits without restarting
    stamp = _stamp()
    exps, runs, judgements = _load(stamp)
    if exps.empty:
        st.info("No experiments yet. Run one with `uv run pif run …` and refresh.")
        return None
    exps = _sort_key(exps)

    sb = st.sidebar
    sb.markdown("### Factors")
    saved = _saved_filters()
    chosen, options = {}, {}
    for k, title in [("dataset", "Datasets"), ("criteria", "Criteria versions"), ("prompt", "Prompt versions"),
                     ("input", "Inputs"), ("reviewer", "Reviewers")]:
        options[k] = list(dict.fromkeys(exps[k]))
        chosen[k] = sb.multiselect(title, options[k], default=_saved_default(saved, k, options[k]), key=f"f_{k}")
    unfinished = sb.toggle("Include unfinished runs", value=bool(saved.get("unfinished", False)), key="f_unfinished",
                           help="Runs that were interrupted or had failed requests. Off by default so every "
                                "run in view covers the whole dataset.")
    save_col, reset_col = sb.columns(2)
    if save_col.button("Save as default", width="stretch",
                       help="Open the dashboard with these filters from now on. Saved on this computer only."):
        _save_filters(chosen, options, unfinished)
        st.toast("Saved these filters as your default.", icon=":material/bookmark:")
        saved = _saved_filters()
    if reset_col.button("Reset filters", width="stretch",
                        help="Back to your saved default, or to everything if none is saved."):
        for k in [*KINDS, "unfinished"]:
            st.session_state.pop(f"f_{k}", None)
        st.rerun()
    if saved:
        sb.caption("Opening with your saved filters. Values added since you saved them start selected.")
        if sb.button("Forget saved default", type="tertiary",
                     help="Delete the saved filters, so the dashboard opens showing everything."):
            FILTERS_FILE.unlink(missing_ok=True)
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

