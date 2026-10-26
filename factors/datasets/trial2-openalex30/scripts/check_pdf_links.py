"""Check which oa_pdf_url links in the PIF run 1 banded sample return a real PDF.

Fetches the first few KB of each link with plain curl (default user agent,
redirects followed, cookies kept) and records whether the body starts with
%PDF. Rate-limited or server-error responses are retried once. Writes one row
per paper with an oa_pdf_url.

Usage: python3 -I check_pdf_links.py <banded.csv> <out.csv> [--workers N]
"""
import argparse
import csv
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("out")
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--timeout", type=float, default=30)
args = ap.parse_args()


def fetch(url):
    with tempfile.TemporaryDirectory() as d:
        body, jar = os.path.join(d, "body"), os.path.join(d, "jar")
        p = subprocess.run(
            ["curl", "-sSL", "--max-time", str(args.timeout), "-r", "0-4095",
             "-c", jar, "-b", jar, "-o", body,
             "-w", "%{http_code}\t%{content_type}\t%{url_effective}", url],
            capture_output=True, text=True,
        )
        code, ctype, final = (p.stdout.split("\t") + ["", "", ""])[:3]
        head = open(body, "rb").read(4096) if os.path.exists(body) else b""
    return {"http_status": code, "content_type": ctype, "final_url": final,
            "is_pdf": str(head.lstrip().startswith(b"%PDF")),
            "error": p.stderr.strip()[:200]}


def check(row):
    url = row["oa_pdf_url"].strip()
    result = fetch(url)
    if result["is_pdf"] != "True" and (result["http_status"] in ("429", "000")
                                       or result["http_status"].startswith("5")):
        time.sleep(10)
        result = fetch(url)
    return {"openalex_id": row["openalex_id"], "oa_pdf_url": url, **result}


with open(args.src, newline="", encoding="utf-8") as f:
    rows = [r for r in csv.DictReader(f) if r["oa_pdf_url"].strip()]

checked_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
with ThreadPoolExecutor(args.workers) as pool:
    results = list(pool.map(check, rows))

cols = ["openalex_id", "oa_pdf_url", "is_pdf", "http_status", "content_type",
        "final_url", "error", "checked_at"]
with open(args.out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in results:
        w.writerow({**r, "checked_at": checked_at})

ok = sum(r["is_pdf"] == "True" for r in results)
print(f"checked {len(results)} links at {checked_at}: {ok} returned a PDF")
