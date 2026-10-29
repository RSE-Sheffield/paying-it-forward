"""Download each paper's PDF from publisher_pdf_url into <out_dir>/<paper_id>.pdf.

Uses plain curl. A file is kept only if it starts
with %PDF and ends with %%EOF, and is then checked against the paper's title:
the share of title words found in pdftotext's first two pages. If the download
fails and the v2 list says plain "kept", the legacy trial 2 PDF is copied
instead (--legacy), since that copy was judged fine. Papers with no
publisher_pdf_url, those named in --skip (a listed URL known to be the wrong
version), or that still fail, are left for a manual download.
Existing files are not re-downloaded, only re-checked.

Writes a log row per paper to <log.csv>, keeping an earlier run's row for a
file that is unchanged since.

Usage: python3 -I download_pdfs.py <papers.csv> <out_dir> <log.csv> [--legacy <dir>] [--skip <id> ...]
"""
import argparse
import csv
import datetime
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata

ap = argparse.ArgumentParser()
ap.add_argument("papers")
ap.add_argument("out_dir")
ap.add_argument("log")
ap.add_argument("--legacy", help="folder of legacy trial 2 PDFs, <openalex_id>.pdf")
ap.add_argument("--skip", nargs="*", default=[], help="paper ids whose publisher_pdf_url is not to be used")
ap.add_argument("--timeout", type=float, default=120)
args = ap.parse_args()


def via_curl(url, dest):
    jar = dest + ".jar"
    subprocess.run(["curl", "-sSL", "--max-time", str(args.timeout),
                    "-c", jar, "-b", jar, "-o", dest, url], capture_output=True)
    if os.path.exists(jar):
        os.remove(jar)


def is_pdf(path):
    if not os.path.exists(path) or os.path.getsize(path) < 1024:
        return False
    with open(path, "rb") as f:
        head = f.read(5)
        f.seek(-1024, os.SEEK_END)
        tail = f.read()
    return head == b"%PDF-" and b"%%EOF" in tail


def words(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.findall(r"[a-z0-9]{3,}", s)


def title_match(path, title):
    res = subprocess.run(["pdftotext", "-l", "2", "-enc", "UTF-8", path, "-"],
                         capture_output=True, text=True)
    found = set(words(res.stdout))
    want = words(title)
    return round(sum(w in found for w in want) / len(want), 2) if want else 0.0


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


with open(args.papers, newline="", encoding="utf-8") as f:
    papers = list(csv.DictReader(f))
os.makedirs(args.out_dir, exist_ok=True)
previous = {}
if os.path.exists(args.log):
    with open(args.log, newline="", encoding="utf-8") as f:
        previous = {r["paper_id"]: r for r in csv.DictReader(f)}

log = []
for p in papers:
    pid = p["paper_id"]
    url = "" if pid in args.skip else p["publisher_pdf_url"]
    dest = os.path.join(args.out_dir, f"{pid}.pdf")
    how = "existing" if is_pdf(dest) else ""
    if not how and url:
        fd, tmp = tempfile.mkstemp(suffix=".pdf", dir=args.out_dir)
        os.close(fd)
        via_curl(url, tmp)
        if is_pdf(tmp):
            os.chmod(tmp, 0o644)  # mkstemp creates it 0600
            os.replace(tmp, dest)
            how = "curl"
        else:
            os.remove(tmp)
    if not how and args.legacy and p["change"] == "kept":
        src = os.path.join(args.legacy, f"{pid}.pdf")
        if is_pdf(src):
            shutil.copyfile(src, dest)
            how = "legacy-copy"
    row = {"paper_id": pid, "access": p["access"], "how": how or "MANUAL",
           "url": url, "title_match": "", "bytes": "", "sha256": "",
           "at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if how:
        row.update(title_match=title_match(dest, p["title"]),
                   bytes=os.path.getsize(dest), sha256=sha256(dest))
        old = previous.get(pid)
        if how == "existing" and old and old["sha256"] == row["sha256"]:
            row = old
        elif how == "existing" and old and old["how"] == "MANUAL":
            row["how"] = "manual"
    log.append(row)
    print(f"{pid}  {row['how']:<12} title_match={row['title_match']}")

with open(args.log, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, list(log[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(log)
manual = [r["paper_id"] for r in log if r["how"] == "MANUAL"]
print(f"{len(log) - len(manual)}/{len(log)} PDFs present" + (f"; manual: {' '.join(manual)}" if manual else ""))
