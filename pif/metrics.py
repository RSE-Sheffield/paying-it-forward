"""Agreement statistics, shared by the dashboard and the analysis scripts so
there is one implementation of each.

Conventions used throughout:
- A *consensus* reduces an experiment's runs to one category per paper: the
  one chosen most often, ties broken in taxonomy order and flagged.
- Comparisons between two experiments use the papers both have a consensus for.
- Axis agreement compares the software (S) or data (D) half of the code alone.
"""
from __future__ import annotations

import itertools
import math
from collections import Counter

import numpy as np

from .taxonomy import CATEGORIES, DATA, SOFTWARE, data, software

S_LEVELS, D_LEVELS = list(SOFTWARE), list(DATA)


def modal(votes) -> str:
    """Most-voted category; ties broken in taxonomy order."""
    c = Counter(votes)
    mx = max(c.values())
    return sorted([k for k, n in c.items() if n == mx], key=CATEGORIES.index)[0]


def is_tied(votes) -> bool:
    c = Counter(votes)
    mx = max(c.values())
    return sum(1 for n in c.values() if n == mx) > 1


def counts_table(votes, cats):
    return np.array([[Counter(v)[c] for c in cats] for v in votes])


def fleiss_kappa(table):
    """Fleiss' kappa over an items x categories table of vote counts (equal raters per item)."""
    table = np.asarray(table, float)
    n, N = table.sum(1)[0], table.shape[0]
    p_j = table.sum(0) / (N * n)
    P_i = (np.square(table).sum(1) - n) / (n * (n - 1))
    Pe = np.square(p_j).sum()
    if Pe == 1:
        return float("nan")
    return (P_i.mean() - Pe) / (1 - Pe)


def cohen_kappa(a, b, cats):
    idx = {c: i for i, c in enumerate(cats)}
    m = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    n = m.sum()
    po, pe = np.trace(m) / n, (m.sum(0) * m.sum(1)).sum() / n ** 2
    if pe == 1:
        return float("nan")
    return (po - pe) / (1 - pe)


def wilson(k, n, z=1.96):
    """Wilson score interval for k successes out of n."""
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# --------------------------------------------------------- over DataFrames
def consensus(judgements):
    """One row per (experiment, paper_id): the consensus category, how many
    valid votes it was drawn from, how many agreed with it, and whether it was
    a tie. Invalid judgements are left out."""
    import pandas as pd

    rows = []
    valid = judgements[judgements["valid"]]
    for (exp, pid), g in valid.groupby(["experiment", "paper_id"], sort=False):
        votes = list(g["category"])
        cat = modal(votes)
        rows.append({"experiment": exp, "paper_id": pid, "category": cat, "votes": len(votes),
                     "agree": Counter(votes)[cat], "tied": is_tied(votes)})
    return pd.DataFrame(rows, columns=["experiment", "paper_id", "category", "votes", "agree", "tied"])


def agreement(a: dict, b: dict) -> dict:
    """Agreement between two {paper_id: category} maps over the papers in both."""
    shared = [p for p in a if p in b]
    n = len(shared)
    if not n:
        return {"n": 0, "exact": float("nan"), "s": float("nan"), "d": float("nan"),
                "kappa": float("nan"), "kappa_s": float("nan"), "kappa_d": float("nan")}
    xa, xb = [a[p] for p in shared], [b[p] for p in shared]
    return {
        "n": n,
        "exact": float(np.mean([x == y for x, y in zip(xa, xb)])),
        "s": float(np.mean([software(x) == software(y) for x, y in zip(xa, xb)])),
        "d": float(np.mean([data(x) == data(y) for x, y in zip(xa, xb)])),
        "kappa": float(cohen_kappa(xa, xb, CATEGORIES)),
        "kappa_s": float(cohen_kappa([software(x) for x in xa], [software(y) for y in xb], S_LEVELS)),
        "kappa_d": float(cohen_kappa([data(x) for x in xa], [data(y) for y in xb], D_LEVELS)),
    }


def self_consistency(judgements) -> dict:
    """How much one experiment's runs agree with each other.

    pairwise:  mean, over every pair of runs, of the share of papers they agree on
    unanimous: share of papers on which every run chose the same category
    fleiss:    Fleiss' kappa over the papers every run judged (overall and per axis)
    """
    valid = judgements[judgements["valid"]]
    by_run = {r: dict(zip(g["paper_id"], g["category"])) for r, g in valid.groupby("run")}
    runs = sorted(by_run)
    out = {"runs": len(runs), "papers": int(valid["paper_id"].nunique())}
    if len(runs) < 2:
        return {**out, "pairwise": float("nan"), "unanimous": float("nan"),
                "fleiss": float("nan"), "fleiss_s": float("nan"), "fleiss_d": float("nan")}
    pair_scores = []
    for r1, r2 in itertools.combinations(runs, 2):
        shared = [p for p in by_run[r1] if p in by_run[r2]]
        if shared:
            pair_scores.append(np.mean([by_run[r1][p] == by_run[r2][p] for p in shared]))
    common = [p for p in by_run[runs[0]] if all(p in by_run[r] for r in runs)]
    votes = [[by_run[r][p] for r in runs] for p in common]
    return {
        **out,
        "pairwise": float(np.mean(pair_scores)) if pair_scores else float("nan"),
        "unanimous": float(np.mean([len(set(v)) == 1 for v in votes])) if votes else float("nan"),
        "fleiss": float(fleiss_kappa(counts_table(votes, CATEGORIES))) if votes else float("nan"),
        "fleiss_s": float(fleiss_kappa(counts_table([[software(x) for x in v] for v in votes], S_LEVELS))) if votes else float("nan"),
        "fleiss_d": float(fleiss_kappa(counts_table([[data(x) for x in v] for v in votes], D_LEVELS))) if votes else float("nan"),
    }
