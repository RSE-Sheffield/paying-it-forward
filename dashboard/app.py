"""Results dashboard.

    uv run pif dashboard

Reads experiments/ directly, so there is nothing to rebuild: refresh the page
after a run finishes and it appears. The sidebar filters scope every page.
"""
import streamlit as st

from views import common, compare, consistency, factors, overview, papers, runs

st.set_page_config(page_title="PIF experiments", page_icon=":material/lab_panel:", layout="wide")

ctx = common.build_context()


def overview_page():
    overview.render(ctx)


def compare_page():
    compare.render(ctx)


def consistency_page():
    consistency.render(ctx)


def papers_page():
    papers.render(ctx)


def runs_page():
    runs.render(ctx)


def factors_page():
    factors.render(ctx)


st.navigation([
    st.Page(overview_page, title="Overview", icon=":material/dashboard:", url_path="overview", default=True),
    st.Page(compare_page, title="Compare", icon=":material/compare_arrows:", url_path="compare"),
    st.Page(consistency_page, title="Consistency", icon=":material/repeat:", url_path="consistency"),
    st.Page(papers_page, title="Papers", icon=":material/article:", url_path="papers"),
    st.Page(runs_page, title="Runs", icon=":material/list:", url_path="runs"),
    st.Page(factors_page, title="Factors", icon=":material/tune:", url_path="factors"),
]).run()
