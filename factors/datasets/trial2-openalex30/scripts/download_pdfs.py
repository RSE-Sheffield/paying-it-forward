"""Download the PDFs for a drawn subsample into <out_dir>/<openalex_id>.pdf.

Tries the request style that passed check_pdf_links.py first (plain curl, or a
browser user agent), then the other. A file is kept only if it starts with %PDF
and ends with %%EOF. Existing files are skipped.

Usage: python3 -I download_pdfs.py <subsample.csv> <pdf_link_check.csv> <out_dir>
"""
import argparse
import csv
import http.cookiejar
import os
import subprocess
import tempfile
import time
import urllib.request

UA = "Mozilla/5.0 (X11; Linux x86_64; rv:131.0) Gecko/20100101 Firefox/131.0"

ap = argparse.ArgumentParser()
ap.add_argument("subsample")
ap.add_argument("link_check")
ap.add_argument("out_dir")
ap.add_argument("--timeout", type=float, default=120)
args = ap.parse_args()


def via_curl(url, dest):
    jar = dest + ".jar"
    subprocess.run(["curl", "-sSL", "--max-time", str(args.timeout),
                    "-c", jar, "-b", jar, "-o", dest, url], capture_output=True)
    if os.path.exists(jar):
        os.remove(jar)


def via_browser_ua(url, dest):
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    req = urllib.request.Request(
        url, headers={"User-Agent": UA, "Accept": "application/pdf,*/*;q=0.8"})
    try:
        with opener.open(req, timeout=args.timeout) as resp, open(dest, "wb") as f:
            f.write(resp.read())
    except Exception:
        pass


def is_complete_pdf(path):
    if not os.path.exists(path) or os.path.getsize(path) < 1024:
        return False
    with open(path, "rb") as f:
        head = f.read(1024)
        f.seek(-2048, os.SEEK_END)
        tail = f.read()
    return head.lstrip().startswith(b"%PDF") and b"%%EOF" in tail


with open(args.link_check, newline="", encoding="utf-8") as f:
    method = {c["openalex_id"]: c["method"] for c in csv.DictReader(f)}
with open(args.subsample, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

os.makedirs(args.out_dir, exist_ok=True)
fetchers = {"curl": via_curl, "browser-ua": via_browser_ua}
failed = []
for r in rows:
    oid, url = r["openalex_id"], r["oa_pdf_url"].strip()
    out = os.path.join(args.out_dir, f"{oid}.pdf")
    if is_complete_pdf(out):
        print(f"{oid}  exists")
        continue
    order = ["browser-ua", "curl"] if method.get(oid) == "browser-ua" else ["curl", "browser-ua"]
    used = None
    for name in order:
        fd, tmp = tempfile.mkstemp(suffix=".pdf", dir=args.out_dir)
        os.close(fd)
        fetchers[name](url, tmp)
        if is_complete_pdf(tmp):
            os.replace(tmp, out)
            used = name
            break
        os.remove(tmp)
    if used:
        print(f"{oid}  ok  {used:<10} {os.path.getsize(out) // 1024:>6} KB")
    else:
        print(f"{oid}  FAILED  {url}")
        failed.append(oid)
    time.sleep(1)

print(f"\n{len(rows) - len(failed)}/{len(rows)} downloaded to {args.out_dir}")
if failed:
    print("failed:", " ".join(failed))
