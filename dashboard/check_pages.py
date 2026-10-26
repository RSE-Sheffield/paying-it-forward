"""Render every dashboard page headlessly against the real experiments/ and
report any exception. Run after changing the dashboard:

    uv run python dashboard/check_pages.py
"""
import sys

from streamlit.testing.v1 import AppTest

PAGES = ["overview", "compare", "consistency", "papers", "runs", "factors"]


def page_script(page):
    import importlib
    import sys

    sys.path.insert(0, "dashboard")
    from views import common

    ctx = common.build_context()
    importlib.import_module(f"views.{page}").render(ctx)


failed = 0
for page in PAGES:
    at = AppTest.from_function(page_script, args=(page,), default_timeout=180)
    at.run()
    errors = [e.value for e in at.exception]
    print(f"{page:<12} {'ok' if not errors else 'FAILED: ' + errors[0].splitlines()[0]}")
    failed += bool(errors)
sys.exit(1 if failed else 0)
