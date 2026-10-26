"""The registries: every version of every factor, what it contains, which
experiments used it, and a diff between any two criteria or prompt versions."""
from __future__ import annotations

import difflib

import pandas as pd
import streamlit as st
import yaml

from pif import registry

from .common import Context


def used_by(ctx: Context | None, kind: str) -> dict[str, int]:
    if ctx is None or ctx.all_runs.empty:
        return {}
    return ctx.all_runs.groupby(kind).experiment.nunique().to_dict()


def diff(a: str, b: str, a_name: str, b_name: str) -> None:
    lines = list(difflib.unified_diff(a.splitlines(), b.splitlines(), a_name, b_name, lineterm=""))
    if lines:
        st.code("\n".join(lines), language="diff", wrap_lines=True)
    else:
        st.info("Identical.")


def versioned_text(ctx, kind: str, texts: dict[str, str]) -> None:
    ids = list(texts)
    usage = used_by(ctx, kind)
    st.dataframe(pd.DataFrame([{"Version": i, "Fingerprint": registry.load(kind, i).fingerprint,
                                "Experiments using it": usage.get(i, 0)} for i in ids]),
                 hide_index=True, width="stretch")
    view, compare = st.tabs(["Read", "Diff"])
    with view:
        pick = st.selectbox("Version", ids, index=len(ids) - 1, key=f"{kind}_read")
        st.code(texts[pick], language="markdown", wrap_lines=True)
    with compare:
        if len(ids) < 2:
            st.info("Only one version so far.")
            return
        c1, c2 = st.columns(2)
        a = c1.selectbox("From", ids, index=len(ids) - 2, key=f"{kind}_a")
        b = c2.selectbox("To", ids, index=len(ids) - 1, key=f"{kind}_b")
        diff(texts[a], texts[b], f"{kind} {a}", f"{kind} {b}")


def render(ctx: Context | None) -> None:
    st.title("Factors")
    st.caption("The versioned inputs every experiment is built from. A version is fixed once an experiment has "
               "used it; changes go in a new version.")
    tabs = st.tabs(["Criteria", "Prompts", "Datasets", "Inputs", "Reviewers"])
    with tabs[0]:
        versioned_text(ctx, "criteria", {i: registry.criteria(i).text for i in registry.ids("criteria")})
    with tabs[1]:
        versioned_text(ctx, "prompt", {i: f"# system.md\n{registry.prompt(i).system_template}\n\n"
                                          f"# task.md\n{registry.prompt(i).task}" for i in registry.ids("prompt")})
    with tabs[2]:
        usage = used_by(ctx, "dataset")
        for i in registry.ids("dataset"):
            ds = registry.dataset(i)
            with st.expander(f"{i} — {ds.spec.get('title', '')} ({len(ds.papers)} papers, "
                             f"{usage.get(i, 0)} experiment(s))"):
                st.markdown(ds.description)
                sources = []
                for name, src in ds.sources.items():
                    if src.get("format") == "jsonl":
                        present = "yes" if (ds.dir / src["path"]).exists() else "no"
                    else:
                        present = sum(ds.pdf_path(name, p).exists() for p in ds.papers)
                        present = f"{present}/{len(ds.papers)}"
                    sources.append({"Source": name, "Format": src.get("format"), "Path": src.get("path"),
                                    "On this machine": present, "Method": src.get("method", "")})
                st.dataframe(pd.DataFrame(sources), hide_index=True, width="stretch")
                st.code(yaml.safe_dump(ds.spec.get("provenance", {}), sort_keys=False, allow_unicode=True),
                        language="yaml")
                st.dataframe(pd.DataFrame(ds.papers), hide_index=True, width="stretch")
    for tab, kind in [(tabs[3], "input"), (tabs[4], "reviewer")]:
        with tab:
            usage = used_by(ctx, kind)
            for i in registry.ids(kind):
                f = registry.load(kind, i)
                with st.expander(f"{i} ({usage.get(i, 0)} experiment(s))"):
                    st.code(f.path.read_text(), language="yaml")
