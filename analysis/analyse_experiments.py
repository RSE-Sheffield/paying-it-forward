"""Reproduce every number and figure in `classification-reliability.md`.

Reads every run under Experiments/, treats the manual pass as the reference,
and writes figures to analysis/figures/ plus a machine-readable dump of the
statistics to analysis/statistics.json.

Depends only on numpy, pandas and matplotlib (no scipy -- the handful of
tests used here are implemented below):

    python3 -m venv .venv && .venv/bin/pip install numpy pandas matplotlib
    .venv/bin/python analysis/analyse_experiments.py
"""
import glob
import itertools
import json
import math
import os
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "analysis", "figures")
CATS = ["S1D1", "S2D1", "S3D1", "S1D2", "S2D2", "S3D2", "S1D3", "S2D3", "S3D3"]
CFG = ["Gemma4 12B · full", "Gemma4 12B · abstract",
       "Sonnet 5 · full", "Sonnet 5 · abstract"]

# --- palette (validated with the dataviz skill's validate_palette.js) -------
SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8a85"
GRID = "#e6e5e1"
GEMMA, SONNET = "#eb6834", "#2a78d6"      # identity: model
SEQ = ["#e8f1fd", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]
ORD3 = ["#86b6ef", "#2a78d6", "#104281"]  # ordinal: 5-0, 4-1, 3-2
COLOR = {c: (GEMMA if c.startswith("Gemma") else SONNET) for c in CFG}
FULL = {c: c.endswith("full") for c in CFG}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9.5,
    "font.family": "DejaVu Sans", "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.grid": False, "legend.frameon": False,
})


def style(ax, xgrid=False, ygrid=False):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# --------------------------------------------------------------- statistics
def fleiss_kappa(table):
    table = np.asarray(table, float)
    n, N = table.sum(1)[0], table.shape[0]
    p_j = table.sum(0) / (N * n)
    P_i = (np.square(table).sum(1) - n) / (n * (n - 1))
    Pe = np.square(p_j).sum()
    return (P_i.mean() - Pe) / (1 - Pe)


def counts_table(votes, cats):
    return np.array([[Counter(v)[c] for c in cats] for v in votes])


def cohen_kappa(a, b, cats):
    idx = {c: i for i, c in enumerate(cats)}
    m = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    n = m.sum()
    po, pe = np.trace(m) / n, (m.sum(0) * m.sum(1)).sum() / n ** 2
    return (po - pe) / (1 - pe)


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def binom_two_sided(k, n, p=0.5):
    if n == 0:
        return 1.0
    pr = [math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    return min(1.0, sum(x for x in pr if x <= pr[k] + 1e-12))


def entropy(v):
    c = Counter(v)
    n = sum(c.values())
    return -sum((x / n) * math.log2(x / n) for x in c.values())


def modal(v):
    """Most-voted category; ties broken in taxonomy order (as the dashboard does)."""
    c = Counter(v)
    mx = max(c.values())
    return sorted([k for k, n in c.items() if n == mx], key=CATS.index)[0]


# --------------------------------------------------------------------- load
def load():
    rows = []
    for d in sorted(glob.glob(os.path.join(ROOT, "Experiments", "*"))):
        run = json.load(open(os.path.join(d, "run.json")))
        rf = [f for f in glob.glob(os.path.join(d, "*.json"))
              if not f.endswith("run.json")]
        for r in json.load(open(rf[0])):
            rows.append(dict(
                run_id=run["run_id"], provider=run["provider"], model=run.get("model"),
                content=run.get("content"), pmid=r.get("pmid"), field=r.get("field"),
                title=r.get("title"), category=r.get("category"),
                reasoning=r.get("reasoning") or "", prompt_tokens=r.get("prompt_tokens"),
                output_tokens=r.get("output_tokens"), duration_s=r.get("duration_s"),
                truncated=r.get("truncated")))
    df = pd.DataFrame(rows)
    df["config"] = df.apply(
        lambda r: "Human" if r.model is None else
        f"{ {'gemma4:12b': 'Gemma4 12B', 'claude-sonnet-5': 'Sonnet 5'}[r.model] } · "
        f"{'full' if r.content == 'full' else 'abstract'}", axis=1)
    bad = set(df.category) - set(CATS)
    assert not bad, f"unrecognised categories: {bad}"
    return df


df = load()
papers = sorted(df.pmid.unique())
H = df[df.config == "Human"].set_index("pmid")
human = H.category.to_dict()
llm = df[df.config != "Human"]
votes = {c: {p: list(llm[(llm.config == c) & (llm.pmid == p)].category) for p in papers}
         for c in CFG}
maj = {c: {p: modal(votes[c][p]) for p in papers} for c in CFG}
S = dict(configs={}, papers={}, overall={})

# ------------------------------------------------- 1. consistency & accuracy
for c in CFG:
    v = [votes[c][p] for p in papers]
    s = llm[llm.config == c]
    per_run = [np.mean([g.set_index("pmid").category[p] == human[p] for p in papers])
               for _, g in s.groupby("run_id")]
    S["configs"][c] = dict(
        fleiss=fleiss_kappa(counts_table(v, CATS)),
        fleiss_S=fleiss_kappa(counts_table([[x[:2] for x in a] for a in v], ["S1", "S2", "S3"])),
        fleiss_D=fleiss_kappa(counts_table([[x[2:] for x in a] for a in v], ["D1", "D2", "D3"])),
        pairwise=float(np.mean([np.mean([a[i] == a[j] for a in v])
                                for i in range(5) for j in range(5) if i < j])),
        unanimous=float(np.mean([len(set(a)) == 1 for a in v])),
        entropy=float(np.mean([entropy(a) for a in v])),
        run_acc_mean=float(np.mean(per_run)), run_acc_min=float(min(per_run)),
        run_acc_max=float(max(per_run)),
        maj_acc=float(np.mean([maj[c][p] == human[p] for p in papers])),
        maj_ci=wilson(sum(maj[c][p] == human[p] for p in papers), 30),
        kappa_human=cohen_kappa([maj[c][p] for p in papers], [human[p] for p in papers], CATS),
        acc_S=float(np.mean([maj[c][p][:2] == human[p][:2] for p in papers])),
        acc_D=float(np.mean([maj[c][p][2:] == human[p][2:] for p in papers])),
        kappa_S=cohen_kappa([maj[c][p][:2] for p in papers], [human[p][:2] for p in papers], ["S1", "S2", "S3"]),
        kappa_D=cohen_kappa([maj[c][p][2:] for p in papers], [human[p][2:] for p in papers], ["D1", "D2", "D3"]),
        in_tok=float(s.prompt_tokens.mean()), out_tok=float(s.output_tokens.mean()),
        sec=float(s.duration_s.mean()), run_min=float(s.duration_s.sum() / 5 / 60),
        s1_rate=float(np.mean([maj[c][p][:2] == "S1" for p in papers])),
        d1_rate=float(np.mean([maj[c][p][2:] == "D1" for p in papers])),
        words=float(s.reasoning.str.split().str.len().mean()),
        names_tool=float(s.reasoning.str.contains(
            r"\b(?:R|Python|SPSS|MATLAB|Stata|ArcGIS|ImageJ|SAS|Qualtrics|REDCap|"
            r"NiftyReg|scikit|PyTorch|TensorFlow)\b", regex=True).mean()),
    )
S["overall"]["human_s1_rate"] = float(np.mean([v[:2] == "S1" for v in human.values()]))
S["overall"]["human_d1_rate"] = float(np.mean([v[2:] == "D1" for v in human.values()]))
S["overall"]["human_s1_ci"] = wilson(sum(v[:2] == "S1" for v in human.values()), 30)
S["overall"]["human_d1_ci"] = wilson(sum(v[2:] == "D1" for v in human.values()), 30)

# --------------------------------------------------------- 2. per paper view
prec = []
for p in papers:
    v = list(llm[llm.pmid == p].category)
    prec.append(dict(pmid=p, field=df[df.pmid == p].field.iloc[0],
                     title=df[df.pmid == p].title.iloc[0], human=human[p],
                     match=sum(x == human[p] for x in v), ncat=len(set(v)),
                     modal=Counter(v).most_common(1)[0][0],
                     modal_n=Counter(v).most_common(1)[0][1]))
P = pd.DataFrame(prec).sort_values(["match", "pmid"]).reset_index(drop=True)
S["papers"] = P.to_dict("records")

# ------------------------------------------------------ 3. confusion by axis
conf = {}
for axis, lv, sl in [("S", ["S1", "S2", "S3"], slice(0, 2)), ("D", ["D1", "D2", "D3"], slice(2, 4))]:
    m = pd.DataFrame(0, index=lv, columns=lv)
    for _, r in llm.iterrows():
        m.loc[human[r.pmid][sl], r.category[sl]] += 1
    conf[axis] = m
    S["overall"][f"confusion_{axis}"] = m.to_dict()

# --------------------------------------------- 4. full vs abstract (McNemar)
S["overall"]["mcnemar"] = {}
for m in ["Gemma4 12B", "Sonnet 5"]:
    F, A = maj[f"{m} · full"], maj[f"{m} · abstract"]
    b = sum(F[p] == human[p] and A[p] != human[p] for p in papers)
    c = sum(F[p] != human[p] and A[p] == human[p] for p in papers)
    S["overall"]["mcnemar"][m] = dict(
        full_only=b, abstract_only=c, p=binom_two_sided(b, b + c),
        agreement=float(np.mean([F[p] == A[p] for p in papers])),
        s_shift=dict(Counter(f"{F[p][:2]}->{A[p][:2]}" for p in papers if F[p][:2] != A[p][:2])),
        d_shift=dict(Counter(f"{F[p][2:]}->{A[p][2:]}" for p in papers if F[p][2:] != A[p][2:])))

# ---------------------------------------------------- 5. ensembles & repeats
S["overall"]["repeat_curve"] = {}
for c in CFG:
    runs = sorted(llm[llm.config == c].run_id.unique())
    cur = {}
    for k in (1, 3, 5):
        accs = []
        for combo in itertools.combinations(runs, k):
            sub = llm[(llm.config == c) & (llm.run_id.isin(combo))]
            mm = {p: modal(list(sub[sub.pmid == p].category)) for p in papers}
            accs.append(np.mean([mm[p] == human[p] for p in papers]))
        cur[k] = float(np.mean(accs))
    S["overall"]["repeat_curve"][c] = cur
ens = {"all 20 runs": float(np.mean([modal(list(llm[llm.pmid == p].category)) == human[p]
                                     for p in papers]))}
for a, b in itertools.combinations(CFG, 2):
    e = {p: modal([maj[a][p], maj[b][p]]) for p in papers}
    ens[f"{a} + {b}"] = float(np.mean([e[p] == human[p] for p in papers]))
S["overall"]["ensembles"] = ens

# ------------------------------------------- 6. triage signals + agreement mx
u = uc = sp = spc = 0
for c in CFG:
    for p in papers:
        v = votes[c][p]
        if len(set(v)) == 1:
            u += 1
            uc += v[0] == human[p]
        else:
            sp += 1
            spc += modal(v) == human[p]
a_, b_, c_, d_ = uc, u - uc, spc, sp - spc
n = a_ + b_ + c_ + d_
S["overall"]["triage"] = dict(
    unanimous_n=u, unanimous_acc=uc / u, unanimous_ci=wilson(uc, u),
    split_n=sp, split_acc=spc / sp, split_ci=wilson(spc, sp),
    chi2=n * (a_ * d_ - b_ * c_) ** 2 / ((a_ + b_) * (c_ + d_) * (a_ + c_) * (b_ + d_)))

HEDGE = re.compile(r"difficult|i think|not clear|unclear|arguably|although|"
                   r"ultimately|hard to|not sure|debatab", re.I)
acc_p = np.array([np.mean([x == human[p] for x in llm[llm.pmid == p].category]) for p in papers])
hed = np.array([bool(HEDGE.search(H.loc[p, "reasoning"])) for p in papers])
obs = acc_p[~hed].mean() - acc_p[hed].mean()
rng = np.random.default_rng(0)
perm = sum((acc_p[~s].mean() - acc_p[s].mean()) >= obs
           for s in (rng.permutation(hed) for _ in range(100_000)))
S["overall"]["hedging"] = dict(
    n_hedged=int(hed.sum()), agr_hedged=float(acc_p[hed].mean()),
    agr_plain=float(acc_p[~hed].mean()), diff=float(obs), p=(perm + 1) / 100_001)

names = CFG + ["Human"]
allm = dict(maj, Human=human)
agr_mx = pd.DataFrame([[np.mean([allm[a][p] == allm[b][p] for p in papers]) for b in names]
                       for a in names], index=names, columns=names)
kap_mx = pd.DataFrame([[cohen_kappa([allm[a][p] for p in papers], [allm[b][p] for p in papers], CATS)
                        for b in names] for a in names], index=names, columns=names)
S["overall"]["between_config_agreement"] = agr_mx.to_dict()
S["overall"]["between_config_kappa"] = kap_mx.to_dict()

# cluster bootstrap over papers -- the 600 judgements are not independent
_pp = {q: [x == human[q] for x in llm[llm.pmid == q].category] for q in papers}
_rng = np.random.default_rng(0)
_b = [np.mean([np.mean(_pp[papers[i]]) for i in _rng.integers(0, 30, 30)]) for _ in range(20_000)]
S["overall"]["pooled_cluster_ci"] = (float(np.percentile(_b, 2.5)), float(np.percentile(_b, 97.5)))
_cells = {q: [(len(set(votes[c][q])) == 1, maj[c][q] == human[q]) for c in CFG] for q in papers}
_d = []
for _ in range(20_000):
    _i = _rng.integers(0, 30, 30)
    _u = [x for i in _i for x in _cells[papers[i]] if x[0]]
    _s = [x for i in _i for x in _cells[papers[i]] if not x[0]]
    if _u and _s:
        _d.append(np.mean([x[1] for x in _u]) - np.mean([x[1] for x in _s]))
S["overall"]["triage"]["gap_cluster_ci"] = (float(np.percentile(_d, 2.5)),
                                            float(np.percentile(_d, 97.5)))

n_agree = int(sum(r.category == human[r.pmid] for _, r in llm.iterrows()))
S["overall"]["pooled_agreement"] = n_agree / len(llm)
S["overall"]["pooled_ci"] = wilson(n_agree, len(llm))
S["overall"]["axis_errors"] = dict(
    S=int(sum(human[r.pmid][:2] != r.category[:2] for _, r in llm.iterrows())),
    D=int(sum(human[r.pmid][2:] != r.category[2:] for _, r in llm.iterrows())),
    both=int(sum(human[r.pmid][:2] != r.category[:2] and human[r.pmid][2:] != r.category[2:]
                 for _, r in llm.iterrows())))

# =============================================================== FIGURES ====
def label(c):
    return c.replace(" · ", "\n")


# Fig 1 - reliability vs validity
fig, ax = plt.subplots(figsize=(6.6, 4.8))
style(ax, xgrid=True, ygrid=True)
ax.plot([0.58, 1.0], [0.58, 1.0], color=INK3, lw=1, zorder=1)
ax.text(0.845, 0.862, "reliability = validity", color=INK3, fontsize=8.5,
        rotation=41, rotation_mode="anchor", ha="center", va="bottom")
OFF = {"Gemma4 12B · full": (0, -20, "center"), "Gemma4 12B · abstract": (13, -4, "left"),
       "Sonnet 5 · full": (0, -20, "center"), "Sonnet 5 · abstract": (0, -20, "center")}
for c in CFG:
    x, y = S["configs"][c]["fleiss"], S["configs"][c]["kappa_human"]
    ax.plot([x, x], [y, x], color=INK3, lw=0.8, zorder=2)
    ax.scatter([x], [y], s=110, zorder=3, color=COLOR[c] if FULL[c] else SURFACE,
               edgecolor=COLOR[c], linewidth=2.0)
    dx, dy, ha = OFF[c]
    ax.annotate(c.replace(" · ", "  ·  "), (x, y), xytext=(dx, dy),
                textcoords="offset points", ha=ha, color=INK, fontsize=8.5)
    side = -7 if x > 0.8 else 7
    ax.annotate(f"gap {x - y:.2f}", (x, (x + y) / 2), xytext=(side, 0),
                textcoords="offset points", va="center",
                ha="right" if side < 0 else "left", color=INK3, fontsize=8)
ax.set_xlim(0.6, 1.0)
ax.set_ylim(0.25, 1.0)
ax.set_xlabel("Self-consistency across 5 repeats  (Fleiss κ)")
ax.set_ylabel("Agreement with the human pass  (Cohen κ)")
ax.set_title("Every configuration is far more consistent than it is correct",
             loc="left", fontsize=11, color=INK, pad=12)
ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=9, mfc=GEMMA, mec=GEMMA, label="Gemma4 12B"),
                   Line2D([], [], marker="o", ls="", ms=9, mfc=SONNET, mec=SONNET, label="Sonnet 5"),
                   Line2D([], [], marker="o", ls="", ms=9, mfc=INK2, mec=INK2, label="full text"),
                   Line2D([], [], marker="o", ls="", ms=9, mfc=SURFACE, mec=INK2, mew=2, label="abstract only")],
          loc="upper left", fontsize=8.5, ncol=1, labelcolor=INK2,
          handletextpad=0.5, labelspacing=0.45)
save(fig, "fig1-reliability-vs-validity.png")

# Fig 2 - per paper agreement
fig, ax = plt.subplots(figsize=(7.6, 7.2))
style(ax, xgrid=True)
y = np.arange(len(P))
frac = P["match"] / 20
ax.hlines(y, 0, frac, color=GRID, lw=1.4, zorder=1)
ax.scatter(frac, y, s=52, color=SONNET, zorder=3)
ax.set_yticks(y)
ax.set_yticklabels([f"{r.human}  {r.title[:44]}" for r in P.itertuples()], fontsize=7.6)
ax.set_ylim(-0.8, len(P) - 0.2)
ax.set_xlim(-0.02, 1.40)
ax.set_xticks([0, .25, .5, .75, 1])
ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
ax.set_xlabel("Share of the 20 model runs matching the human category")
ax.set_title("57% of the corpus is settled at one end or the other",
             loc="left", fontsize=11, color=INK, pad=10)
ax.axhspan(-0.8, 5.5, color="#f1f0ec", zorder=0)
ax.axhspan(24.5, len(P) - 0.2, color="#f1f0ec", zorder=0)
ax.text(1.06, 2.6, "6 papers: not one\nrun of 20 agrees\nwith the human", color=INK2, fontsize=8.4,
        va="center", ha="left")
ax.text(1.06, 27.0, "5 papers:\nall 20 agree", color=INK2, fontsize=8.4,
        va="center", ha="left")
save(fig, "fig2-per-paper-agreement.png")

# Fig 3 - axis confusion
fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
for ax, axis in zip(axes, ["S", "D"]):
    m = conf[axis]
    rn = m.div(m.sum(1), axis=0) * 100
    ax.imshow(rn.values, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", SEQ),
              vmin=0, vmax=100)
    for i in range(3):
        for j in range(3):
            v = rn.values[i, j]
            ax.text(j, i, f"{v:.0f}%\n{int(m.values[i, j])}", ha="center", va="center",
                    fontsize=9, color="#ffffff" if v > 55 else INK)
    ax.set_xticks(range(3), m.columns, fontsize=9.5)
    ax.set_yticks(range(3), m.index, fontsize=9.5)
    ax.set_xlabel("model judgement")
    ax.set_ylabel("human reference")
    ax.set_title({"S": "Software axis — 195/600 disagree",
                  "D": "Data axis — 148/600 disagree"}[axis],
                 loc="left", fontsize=10.5, color=INK, pad=8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
fig.suptitle("Where the disagreement lives: rows sum to 100%, all 600 model judgements",
             x=0.02, ha="left", fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94))
save(fig, "fig3-axis-confusion.png")

# Fig 4 - prevalence bias
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
for ax, (key, human_key, ttl) in zip(axes, [
        ("s1_rate", "human_s1_rate", "S1 — “this paper generated software”"),
        ("d1_rate", "human_d1_rate", "D1 — “this paper generated data”")]):
    style(ax, ygrid=True)
    xs = np.arange(4)
    vals = [S["configs"][c][key] * 100 for c in CFG]
    err = np.array([[v * 100 - wilson(round(v * 30), 30)[0] * 100,
                     wilson(round(v * 30), 30)[1] * 100 - v * 100]
                    for v, c in zip([S["configs"][c][key] for c in CFG], CFG)]).T
    for i, c in enumerate(CFG):
        ax.bar(i, vals[i], width=0.62, color=COLOR[c] if FULL[c] else SURFACE,
               edgecolor=COLOR[c], linewidth=1.8,
               hatch=None if FULL[c] else "///", zorder=2)
    ax.errorbar(xs, vals, yerr=err, fmt="none", ecolor=INK3, elinewidth=1, capsize=3, zorder=3)
    hv = S["overall"][human_key] * 100
    lo, hi = [x * 100 for x in S["overall"]["human_s1_ci" if key == "s1_rate" else "human_d1_ci"]]
    ax.axhspan(lo, hi, color=GRID, alpha=0.55, zorder=0)
    ax.axhline(hv, color=INK2, lw=1.4, zorder=1)
    ax.text(-1.12, hv, f"human\n{hv:.0f}%", color=INK2, fontsize=8.5,
            va="center", ha="left")
    for i, v in enumerate(vals):
        top = wilson(round(v / 100 * 30), 30)[1] * 100
        ax.text(i, top + 2.0, f"{v:.0f}%", ha="center", fontsize=9.5, color=INK)
    ax.set_xticks(xs, [label(c) for c in CFG], fontsize=8.2)
    ax.set_xlim(-1.2, 3.6)
    ax.set_title(ttl, loc="left", fontsize=10.5, color=INK, pad=8)
    ax.set_ylim(0, 86)
axes[0].set_ylabel("% of the 30 papers")
fig.suptitle("The headline rates are biased in opposite directions — software down, data up",
             x=0.02, ha="left", fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.92))
save(fig, "fig4-prevalence-bias.png")

# Fig 5 - cost vs agreement
fig, ax = plt.subplots(figsize=(6.6, 4.4))
style(ax, xgrid=True, ygrid=True)
for m in ["Gemma4 12B", "Sonnet 5"]:
    a, f = f"{m} · abstract", f"{m} · full"
    xa, ya = S["configs"][a]["in_tok"], S["configs"][a]["maj_acc"] * 100
    xf, yf = S["configs"][f]["in_tok"], S["configs"][f]["maj_acc"] * 100
    col = GEMMA if m.startswith("Gemma") else SONNET
    ax.annotate("", xy=(xf, yf), xytext=(xa, ya),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6, alpha=0.65,
                                shrinkA=9, shrinkB=9))
    ax.scatter([xa], [ya], s=120, color=SURFACE, edgecolor=col, lw=2.2, zorder=3)
    ax.scatter([xf], [yf], s=120, color=col, zorder=3)
    up = 1 if m.startswith("Sonnet") else -1
    ax.annotate(f"{m}\nabstract only", (xa, ya), xytext=(0, 16 if up > 0 else -32),
                textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax.annotate(f"{m}\nfull text", (xf, yf), xytext=(0, -32 if up > 0 else 16),
                textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax.annotate(f"×{xf / xa:.0f} input tokens,  {yf - ya:+.0f} pp",
                ((xa * xf) ** 0.5 * (0.78 if up > 0 else 1.0), (ya + yf) / 2),
                xytext=(0, 8 if up > 0 else -14),
                textcoords="offset points", ha="center", fontsize=8.5, color=col)
ax.set_xscale("log")
ax.set_xlim(1500, 60000)
ax.set_ylim(36, 68)
ax.set_xticks([2000, 5000, 10000, 20000, 50000])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
ax.set_xlabel("Mean input tokens per paper  (log scale)")
ax.set_ylabel("Agreement with human, majority of 5 runs (%)")
ax.set_title("Reading the full paper costs an order of magnitude and buys nothing",
             loc="left", fontsize=11, color=INK, pad=12)
save(fig, "fig5-cost-vs-agreement.png")

# Fig 6 - vote split distribution
fig, ax = plt.subplots(figsize=(7.0, 3.4))
style(ax, xgrid=True)
shapes = ["5–0", "4–1", "3–2 or wider"]
for i, c in enumerate(CFG[::-1]):
    v = [votes[c][p] for p in papers]
    tops = [max(Counter(a).values()) for a in v]
    vals = [sum(t == 5 for t in tops), sum(t == 4 for t in tops), sum(t <= 3 for t in tops)]
    left = 0
    for j, val in enumerate(vals):
        ax.barh(i, val, left=left, height=0.6, color=ORD3[j], zorder=2,
                edgecolor=SURFACE, linewidth=2)
        if val:
            ax.text(left + val / 2, i, str(val), ha="center", va="center", fontsize=9,
                    color="#ffffff" if j >= 1 else INK)
        left += val
ax.set_yticks(range(4), [c.replace(" · ", "  ·  ") for c in CFG[::-1]], fontsize=9)
ax.set_xlim(0, 30)
ax.set_xlabel("papers (of 30)")
ax.legend(handles=[Patch(facecolor=ORD3[j], label=f"{shapes[j]} vote split") for j in range(3)],
          loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=8.5, labelcolor=INK2)
ax.set_title("How often five repeats of the same configuration land on the same category",
             loc="left", fontsize=11, color=INK, pad=10)
save(fig, "fig6-vote-splits.png")

# Fig 7 - triage signals
fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9), sharey=True)
panels = [("Do the 5 repeats agree with each other?\n(120 configuration × paper cells)",
           ["all 5 repeats\nagree", "repeats\nsplit"],
           [S["overall"]["triage"]["unanimous_acc"], S["overall"]["triage"]["split_acc"]],
           [S["overall"]["triage"]["unanimous_ci"], S["overall"]["triage"]["split_ci"]],
           [S["overall"]["triage"]["unanimous_n"], S["overall"]["triage"]["split_n"]],
           f"χ²={S['overall']['triage']['chi2']:.1f}, p≈0.002"),
          ("Did the human hedge in their own note?\n(30 papers, all 20 runs pooled)",
           ["confident\nnote", "hedged\nnote"],
           [S["overall"]["hedging"]["agr_plain"], S["overall"]["hedging"]["agr_hedged"]],
           [None, None], [20, 10],
           f"permutation p={S['overall']['hedging']['p']:.4f}")]
for ax, (ttl, labels, vals, cis, ns, note) in zip(axes, panels):
    style(ax, ygrid=True)
    for i, v in enumerate(vals):
        ax.bar(i, v * 100, width=0.5, color=SONNET if i == 0 else "#b9b8b2", zorder=2)
        top = cis[i][1] * 100 if cis[i] else v * 100
        ax.text(i, top + 3.0, f"{v * 100:.0f}%", ha="center", fontsize=10.5, color=INK)
        if cis[i]:
            ax.errorbar(i, v * 100, yerr=[[v * 100 - cis[i][0] * 100], [cis[i][1] * 100 - v * 100]],
                        fmt="none", ecolor=INK3, elinewidth=1, capsize=3, zorder=3)
    ax.set_xticks(range(2), [f"{l}\n(n={n})" for l, n in zip(labels, ns)], fontsize=8.5)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 85)
    ax.set_title(ttl, loc="left", fontsize=9.5, color=INK, pad=8)
    ax.text(1.58, 76, note, ha="right", fontsize=8.5, color=INK3)
axes[0].set_ylabel("agreement with human (%)")
fig.suptitle("Two cheap signals that flag which judgements not to trust",
             x=0.02, ha="left", fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.88))
save(fig, "fig7-triage-signals.png")


# ======================================================= REASONING TEXT =====
# Everything below reads the free-text `reasoning` field rather than the label.
llm = llm.copy()
llm["agree"] = [r.category == human[r.pmid] for _, r in llm.iterrows()]
llm["words"] = llm.reasoning.str.split().str.len()

FRAMES = {
    "novelty (new / novel)": r"\bnovel\b|\bnew(?:ly)?\b",
    "pre-existing": r"\bexisting\b|pre-existing|previously published|already (?:available|published)",
    "sharing / availability": r"\bshar(?:e|es|ed|ing)\b|publicly available|made available|\brepositor"
                              r"|github|zenodo|figshare|\bosf\b|dryad|deposit"
                              r"|\bprovide[sd]? (?:the )?(?:code|software|script|data)",
    "genericness": r"\bgeneric\b|\bstandard\b|\broutine\b|off-the-shelf|commercial|established"
                   r"|widely[- ]used|conventional|general-purpose",
    "study's purpose / output": r"primary (?:goal|aim|output|contribution)|core (?:of|contribution)"
                                r"|main (?:output|focus|goal)|itself a|central to|research goal"
                                r"|focus of the study",
}
# "acknowledge the artefact, then dismiss it as not substantial enough"
DISMISS = (r"rather than|instead of|does not (?:constitute|rise|amount|qualify)|is more (?:a|of)"
           r"|more (?:a|of a) (?:demonstration|tutorial|application)|not (?:a )?(?:substantial|novel|new)"
           r"|but (?:this|it|these) (?:is|are)|standard (?:statistical |analytical )?(?:methods?|practice|tools?|software)"
           r"|merely|only (?:a|an|used)|routine|generic|off-the-shelf|as-is"
           r"|without (?:developing|modifying|extending|substantial)")
TOOLNAME = (r"\b(?:R|Python|SPSS|MATLAB|Stata|SAS|ArcGIS|ImageJ|Qualtrics|REDCap|NiftyReg|scikit-?learn"
            r"|PyTorch|TensorFlow|GPT-2|Word2Vec|GloVe|fastText|Eclipse|TopoToolbox|InForm|Prism|Excel"
            r"|NVivo|RevMan)\b")
HEDGE_M = (r"\bappears?\b|\bseems?\b|\blikely\b|presumably|\bsuggests?\b|\bmay\b|\bmight\b|unclear"
           r"|not (?:explicitly|clear|specified|described|detailed)|does not specify|insufficient"
           r"|without (?:more|further|access|the full)|difficult to|ambiguous|arguably|it is possible"
           r"|cannot (?:be )?determin|limited (?:detail|information)")
DEFICIT = (r"abstract (?:does not|alone|only|provided|is)|without (?:the )?full text"
           r"|not (?:enough|sufficient) (?:detail|information)|limited information"
           r"|based (?:only )?on the abstract|no methods section")

llm["dismiss"] = llm.reasoning.str.contains(DISMISS, case=False, regex=True)
llm["toolname"] = llm.reasoning.str.contains(TOOLNAME, regex=True)
llm["hedged"] = llm.reasoning.str.contains(HEDGE_M, case=False, regex=True)
llm["deficit"] = llm.reasoning.str.contains(DEFICIT, case=False, regex=True)

T = {}
T["frames_per_100_words"] = {}
for nm, pat in FRAMES.items():
    T["frames_per_100_words"][nm] = {
        c: float(100 * df[df.config == c].reasoning.str.count(pat, flags=re.I).sum()
                 / df[df.config == c].reasoning.str.split().str.len().sum())
        for c in CFG + ["Human"]}
T["novelty_sharing_ratio"] = {
    c: T["frames_per_100_words"]["novelty (new / novel)"][c]
       / T["frames_per_100_words"]["sharing / availability"][c] for c in CFG + ["Human"]}

d_, nd_ = llm[llm.dismiss], llm[~llm.dismiss]
T["dismissal"] = dict(
    rate={c: float(llm[llm.config == c].dismiss.mean()) for c in CFG},
    with_move=dict(n=len(d_), S1=float((d_.category.str[:2] == "S1").mean()),
                   S2=float((d_.category.str[:2] == "S2").mean()),
                   S3=float((d_.category.str[:2] == "S3").mean()), agree=float(d_.agree.mean())),
    without=dict(n=len(nd_), S1=float((nd_.category.str[:2] == "S1").mean()),
                 S2=float((nd_.category.str[:2] == "S2").mean()),
                 S3=float((nd_.category.str[:2] == "S3").mean()), agree=float(nd_.agree.mean())))

# cluster bootstrap for the dismissal agreement gap
_c = {q: list(zip(llm[llm.pmid == q].dismiss, llm[llm.pmid == q].agree)) for q in papers}
_g = []
for _ in range(20_000):
    _i = _rng.integers(0, 30, 30)
    _x = [t for k in _i for t in _c[papers[k]] if t[0]]
    _y = [t for k in _i for t in _c[papers[k]] if not t[0]]
    if _x and _y:
        _g.append(np.mean([t[1] for t in _y]) - np.mean([t[1] for t in _x]))
T["dismissal"]["agree_gap"] = float(nd_.agree.mean() - d_.agree.mean())
T["dismissal"]["agree_gap_ci"] = (float(np.percentile(_g, 2.5)), float(np.percentile(_g, 97.5)))

# within-paper: does the dismissal move go with a lower P(S1) for the same paper?
_lo = _hi = 0
for q in papers:
    g = llm[llm.pmid == q]
    if not 0 < g.dismiss.sum() < len(g):
        continue
    x = (g[g.dismiss].category.str[:2] == "S1").mean()
    y = (g[~g.dismiss].category.str[:2] == "S1").mean()
    _lo += x < y
    _hi += x > y
T["dismissal"]["within_paper"] = dict(lower=_lo, higher=_hi, p=binom_two_sided(_lo, _lo + _hi))

t_, nt_ = llm[llm.toolname], llm[~llm.toolname]
T["toolname"] = dict(
    rate={c: float(llm[llm.config == c].toolname.mean()) for c in CFG},
    with_tool=dict(n=len(t_), S2=float((t_.category.str[:2] == "S2").mean()), agree=float(t_.agree.mean())),
    without=dict(n=len(nt_), S2=float((nt_.category.str[:2] == "S2").mean()), agree=float(nt_.agree.mean())))
_a = _b = 0
for q in papers:
    g = llm[llm.pmid == q]
    if not 0 < g.toolname.sum() < len(g):
        continue
    x = (g[g.toolname].category.str[:2] == "S2").mean()
    y = (g[~g.toolname].category.str[:2] == "S2").mean()
    _a += x > y
    _b += x < y
T["toolname"]["within_paper"] = dict(higher=_a, lower=_b, n_discordant=_a + _b,
                                     p=binom_two_sided(_a, _a + _b))

# reasoning length: a paper-level signal, not a judgement-level one
_W = llm.groupby("pmid").words.mean()
_A = llm.groupby("pmid").agree.mean()
_r = [np.corrcoef(_W.values[i], _A.values[i])[0, 1]
      for i in (_rng.integers(0, 30, 30) for _ in range(20_000))]
llm["words_c"] = llm.groupby(["config", "pmid"]).words.transform(lambda x: x - x.mean())
_var = llm[llm.groupby(["config", "pmid"]).category.transform("nunique") > 1]
T["length"] = dict(
    paper_level_r=float(np.corrcoef(_W, _A)[0, 1]),
    paper_level_ci=(float(np.percentile(_r, 2.5)), float(np.percentile(_r, 97.5))),
    within_cell_r=float(np.corrcoef(_var.words_c, _var.agree.astype(float))[0, 1]),
    by_config_raw={c: float(np.corrcoef(llm[llm.config == c].words,
                                        llm[llm.config == c].agree.astype(float))[0, 1]) for c in CFG},
    by_config_centred={c: float(np.corrcoef(llm[llm.config == c].words_c,
                                            llm[llm.config == c].agree.astype(float))[0, 1]) for c in CFG})

# do repeats that disagree on the label also disagree on the argument?
_STOP = set("a an the and or but of to in for on with by is are was were be been being it its this that "
            "these those they them their there as at from into than then so such not no nor does do did "
            "done has have had also which who what when where while if both each all can could would "
            "should may might will we you i".split())
_TOK = re.compile(r"[a-z][a-z\-']{2,}")


def _bag(t):
    return set(w for w in _TOK.findall(t.lower()) if w not in _STOP)


def _jac(x, y):
    a, b = _bag(x), _bag(y)
    return len(a & b) / len(a | b) if a | b else 0.0


_same, _diff, _near = [], [], 0
for c in CFG:
    for q in papers:
        g = llm[(llm.config == c) & (llm.pmid == q)]
        rs, cs = list(g.reasoning), list(g.category)
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                v = _jac(rs[i], rs[j])
                (_same if cs[i] == cs[j] else _diff).append(v)
                if cs[i] != cs[j] and v > 0.45:
                    _near += 1
T["argument_overlap"] = dict(same_label=float(np.mean(_same)), diff_label=float(np.mean(_diff)),
                             n_diff_pairs=len(_diff), near_identical_but_different=_near)

T["hedging_model"] = dict(
    rate={c: float(llm[llm.config == c].hedged.mean()) for c in CFG},
    agree_hedged=float(llm[llm.hedged].agree.mean()), n_hedged=int(llm.hedged.sum()),
    agree_plain=float(llm[~llm.hedged].agree.mean()))
T["deficit_statements"] = dict(
    abstract_runs=int(llm[llm.content == "metadata"].deficit.sum()) if "content" in llm else None,
    total=int(llm.deficit.sum()))
T["repository_mentions"] = {k: int(llm.reasoning.str.contains(v, case=False, regex=True).sum())
                            for k, v in {"github": "github", "zenodo": "zenodo", "figshare": "figshare",
                                         "osf": r"\bosf\b", "dryad": "dryad",
                                         "'repository'": "repositor"}.items()}
S["reasoning"] = T

# Fig 8 - framing fingerprint
fig, ax = plt.subplots(figsize=(7.4, 4.1))
style(ax, xgrid=True)
raters = CFG + ["Human"]
yy = np.arange(len(raters))[::-1]
nov = [T["frames_per_100_words"]["novelty (new / novel)"][c] for c in raters]
shr = [T["frames_per_100_words"]["sharing / availability"][c] for c in raters]
ax.barh(yy + 0.19, nov, height=0.34, color=SONNET, zorder=2, label="novelty — “new”, “novel”")
ax.barh(yy - 0.19, shr, height=0.34, color=GEMMA, zorder=2,
        label="artefact exists / is shared — “shared”, “publicly available”, “provides the code”")
for y, n, s_ in zip(yy, nov, shr):
    ax.text(n + 0.06, y + 0.19, f"{n:.2f}", va="center", fontsize=8.5, color=INK)
    ax.text(s_ + 0.06, y - 0.19, f"{s_:.2f}", va="center", fontsize=8.5, color=INK)
ax.set_yticks(yy, [c.replace(" · ", "  ·  ") for c in raters], fontsize=9)
ax.set_xlim(0, 3.35)
ax.set_xlabel("mentions per 100 words of reasoning")
ax.axhspan(-0.5, 0.5, color="#f1f0ec", zorder=0)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, fontsize=8.3,
          labelcolor=INK2, handletextpad=0.6)
ax.set_title("The models argue about novelty; the human argues about what exists",
             loc="left", fontsize=11, color=INK, pad=10)
save(fig, "fig8-framing.png")

# Fig 9 - the dismissal move
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
ax = axes[0]
style(ax, ygrid=True)
for i, c in enumerate(CFG):
    ax.bar(i, T["dismissal"]["rate"][c] * 100, width=0.6,
           color=COLOR[c] if FULL[c] else SURFACE, edgecolor=COLOR[c], linewidth=1.8,
           hatch=None if FULL[c] else "///", zorder=2)
    ax.text(i, T["dismissal"]["rate"][c] * 100 + 2, f"{T['dismissal']['rate'][c] * 100:.0f}%",
            ha="center", fontsize=9.5, color=INK)
ax.set_xticks(range(4), [label(c) for c in CFG], fontsize=8.2)
ax.set_ylim(0, 72)
ax.set_ylabel("% of judgements")
ax.set_title("How often the reasoning names an artefact\nthen argues it doesn't count",
             loc="left", fontsize=10, color=INK, pad=8)

ax = axes[1]
style(ax, xgrid=True)
rows_ = [("without the move", T["dismissal"]["without"]), ("with the move", T["dismissal"]["with_move"])]
for i, (nm, v) in enumerate(rows_):
    left = 0
    for j, k in enumerate(["S1", "S2", "S3"]):
        ax.barh(i, v[k] * 100, left=left, height=0.5, color=ORD3[j], edgecolor=SURFACE,
                linewidth=2, zorder=2)
        if v[k] > 0.07:
            ax.text(left + v[k] * 50, i, f"{v[k] * 100:.0f}%", ha="center", va="center",
                    fontsize=9, color="#ffffff" if j >= 1 else INK)
        left += v[k] * 100
    ax.text(101, i, f"agrees with human {v['agree'] * 100:.0f}%   (n={v['n']})",
            va="center", fontsize=8.6, color=INK2)
ax.set_yticks(range(2), [r[0] for r in rows_], fontsize=9)
ax.set_xlim(0, 168)
ax.set_xticks([0, 25, 50, 75, 100], ["0", "25%", "50%", "75%", "100%"])
ax.set_ylim(-0.6, 1.6)
ax.legend(handles=[Patch(facecolor=ORD3[j], label=k) for j, k in enumerate(["S1 generated", "S2 used", "S3 none"])],
          loc="upper center", bbox_to_anchor=(0.36, -0.16), ncol=3, fontsize=8.5, labelcolor=INK2)
ax.set_title("Software verdict, with and without the move", loc="left", fontsize=10, color=INK, pad=8)
fig.tight_layout()
save(fig, "fig9-dismissal-move.png")

# Fig 10 - reasoning length as a paper-difficulty detector
fig, ax = plt.subplots(figsize=(6.6, 4.3))
style(ax, xgrid=True, ygrid=True)
ax.scatter(_W.values, _A.values * 100, s=62, color=SONNET, zorder=3, alpha=0.9)
_m, _b2 = np.polyfit(_W.values, _A.values * 100, 1)
_xs = np.linspace(_W.min() - 2, _W.max() + 2, 10)
ax.plot(_xs, _m * _xs + _b2, color=INK3, lw=1.3, zorder=2)
ANNOT = {"37388662": ("Decolonising the psychology curriculum", 6, 9, "left"),
         "37186857": ("Design of rigid PPI inhibitors", -8, 4, "right"),
         "36888610": ("National-scale geodatabase", 0, -14, "center"),
         "37102792": ("Isotoxic dose escalated radiotherapy", 6, 6, "left"),
         "38100737": ("Phenotypes and rates of cancer-relevant symptoms", -8, 6, "right")}
for q, (nm, dx, dy, ha_) in ANNOT.items():
    ax.annotate(nm, (_W[q], _A[q] * 100), xytext=(dx, dy), textcoords="offset points",
                fontsize=7.6, color=INK2, ha=ha_)
ax.set_xlabel("mean reasoning length across all 20 runs (words)")
ax.set_ylabel("share of runs agreeing with the human (%)")
ax.set_title(f"Papers the models write more about are papers they get wrong  (r = "
             f"{T['length']['paper_level_r']:.2f}, n=30)", loc="left", fontsize=10.5, color=INK, pad=12)
ax.text(0.98, 0.985, "within a single paper, the longer\nrepeat is no less likely to be right\n(r = "
        f"{T['length']['within_cell_r']:+.2f})", transform=ax.transAxes, ha="right", va="top",
        fontsize=8.4, color=INK3)
ax.set_ylim(-20, 112)
save(fig, "fig10-length-difficulty.png")

with open(os.path.join(ROOT, "analysis", "statistics.json"), "w") as fh:
    json.dump(S, fh, indent=1, default=float)
print("wrote analysis/statistics.json")
