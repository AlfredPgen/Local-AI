r"""Download arXiv papers that match the keyword list and convert them to Markdown
for data_prep.py, from their LaTeX source (equations stay LaTeX).

1. harvest  arXiv's metadata (OAI-PMH: title, abstract, categories, licence) for
            the subject areas in --sets, 1,000+ papers per request, cached in
            <meta>\metadata\<set>.jsonl.gz; a rerun continues where it stopped.
2. select   papers whose title and abstract match the keywords (default: at
            least 3 matches from 3 different terms; a title match counts 3
            times), ranked by group and then by keyword density: biology
            (q-bio, physics.bio-ph) first, then statistics, then AI and
            mathematics together. Saved in <meta>\list.tsv.
3. fetch    each paper's source (e-print) from export.arxiv.org, one request at a
            time and at most one every 3 seconds, as arXiv asks of automated
            tools (about 28,000 papers a day). --max-papers limits the number.
4. convert  LaTeX to Markdown with pandoc: headings, paragraphs, lists,
            equations as LaTeX, tables, footnotes; title and abstract from the
            metadata; citations numbered [1], [2, 3] as in the PubMed Central
            papers; figures reduced to their captions; references,
            acknowledgements, funding and similar sections left out; e-mail
            addresses masked. Papers without usable source (PDF-only
            submissions, or LaTeX pandoc cannot read) are converted from their
            PDF with convert_to_markdown.py's PDF converter.

Outputs: <out>\<arXiv id>__<title>.md and <meta>\report.tsv (one row per paper:
status, characters, method, group, year, categories, licence). Sources are not
kept. Existing outputs are skipped, so the script can be stopped (Ctrl+C) and
rerun; the ranked order means the most relevant papers come first.

Safety: archives are read in memory (only .tex and .bbl members, size-limited,
never written out); pandoc parses the LaTeX as data (it does not run TeX) in its
sandbox mode, which lets it read nothing but the one file it is given, and it
has a time limit.

Licences: most arXiv papers carry arXiv's own non-exclusive distribution
licence (reading, not reuse); some are Creative Commons. Each paper's licence
is recorded in report.tsv.

Needs pandoc (pypandoc_binary, or pandoc on PATH) and, for the PDF fallback,
pymupdf4llm.

Example (everything the default subject areas offer, best matches first):
    python download_arxiv.py --out %USERPROFILE%\ai_training_data\arxiv
"""

import argparse
import concurrent.futures as cf
import gzip
import http.client
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import types
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
OAI_URL = "https://oaipmh.arxiv.org/oai"
EPRINT_URL = "https://export.arxiv.org/e-print/{}"
PDF_URL = "https://export.arxiv.org/pdf/{}"
USER_AGENT = "tinyGPT-arxiv-downloader/1.0 (personal research; python urllib)"
MIN_REQUEST_GAP = 3.0  # seconds between request starts: arXiv's limit for automated access
MAX_DOWNLOAD = 60 << 20  # e-print or PDF bytes; a larger source falls back to the PDF
MAX_UNPACKED = 300 << 20
MAX_TEX_FILE = 5 << 20
PANDOC_SECONDS = 90
# subject areas (OAI-PMH sets) and their priority groups
DEFAULT_SETS = ["q-bio", "physics:physics:bio-ph", "stat", "cs:cs:AI", "cs:cs:LG", "cs:cs:CL", "cs:cs:CV",
                "cs:cs:NE", "cs:cs:MA", "cs:cs:IR", "math"]
GROUPS = [("biology", ("q-bio", "physics.bio-ph")), ("statistics", ("stat",)),
          ("AI", ("cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.NE", "cs.MA", "cs.IR")), ("mathematics", ("math",))]
GROUP_RANK = {"biology": 0, "statistics": 1, "AI": 2, "mathematics": 2}  # AI and mathematics share a rank
FINAL = {"ok", "too_short", "no_text"}  # statuses not retried on a rerun
REPORT_COLUMNS = ["id", "status", "chars", "method", "group", "year", "categories", "licence", "file"]
LIST_COLUMNS = ["id", "group", "score", "year", "categories", "licence", "title"]
ARXIV_NS = "{http://arxiv.org/OAI/arXiv/}"
OAI_NS = "{http://www.openarchives.org/OAI/2.0/}"


# ---------------------------------------------------------------------------
# HTTP: one connection at a time, at most one request start every 3 seconds
# ---------------------------------------------------------------------------
_NET_LOCK = threading.Lock()
_LAST_START = [0.0]


def http_get(url, limit=MAX_DOWNLOAD, tries=6):
    """Bytes of url; None on 404 (or when the body exceeds `limit`, as
    'too_large'). Server busy (503 with Retry-After), rate limits and network
    failures are retried with backoff."""
    delay = 5.0
    for attempt in range(tries):
        with _NET_LOCK:
            pause = _LAST_START[0] + MIN_REQUEST_GAP - time.monotonic()
            if pause > 0:
                time.sleep(pause)
            _LAST_START[0] = time.monotonic()
            retry_after = None
            try:
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=300) as response:
                    chunks, size = [], 0
                    while True:
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > limit:
                            return "too_large"
                        chunks.append(chunk)
                    return b"".join(chunks)
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    return None
                if exc.code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                    raise
                try:
                    retry_after = float(exc.headers.get("Retry-After") or 0)
                except ValueError:
                    retry_after = None
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException, OSError):
                if attempt == tries - 1:
                    raise
        time.sleep(min(retry_after, 600) if retry_after else delay + random.random())
        delay = min(delay * 2, 300)
    return None


# ---------------------------------------------------------------------------
# 1. harvest (OAI-PMH)
# ---------------------------------------------------------------------------
def _flat(value):
    return " ".join(str(value or "").replace("\t", " ").split())


def _set_file(set_spec):
    return re.sub(r"[^A-Za-z0-9.-]+", "_", set_spec)


def parse_oai_page(xml_bytes):
    """([record dicts], resumption token or None, error code or None)."""
    root = ET.fromstring(xml_bytes)
    error = root.find(f"{OAI_NS}error")
    if error is not None:
        return [], None, error.get("code")
    records = []
    for rec in root.iter(f"{OAI_NS}record"):
        header = rec.find(f"{OAI_NS}header")
        if header is not None and header.get("status") == "deleted":
            continue
        meta = rec.find(f"{OAI_NS}metadata/{ARXIV_NS}arXiv")
        if meta is None:
            continue
        get = lambda tag: _flat(meta.findtext(f"{ARXIV_NS}{tag}"))  # noqa: E731
        records.append({"id": get("id"), "created": get("created"), "title": get("title"),
                        "abstract": get("abstract"), "categories": get("categories"), "license": get("license"),
                        "doi": get("doi")})
    token_el = root.find(f"{OAI_NS}ListRecords/{OAI_NS}resumptionToken")
    token = (token_el.text or "").strip() if token_el is not None else ""
    return records, token or None, None


def harvest(set_spec, cache_dir):
    """Append the set's records to <cache>/<set>.jsonl.gz, page by page; a
    progress file lets a rerun continue (or restart the set if arXiv no longer
    accepts the saved position). Returns the number of records written."""
    base = os.path.join(cache_dir, _set_file(set_spec))
    state_path = base + ".state.json"
    state = {"token": None, "pages": 0, "records": 0, "complete": False}
    if os.path.isfile(state_path):
        with open(state_path, encoding="utf-8") as handle:
            state.update(json.load(handle))
    if state["complete"]:
        print(f"  [{set_spec}] cached: {state['records']:,} records")
        return state["records"]
    if not state["token"] and os.path.exists(base + ".jsonl.gz"):
        os.remove(base + ".jsonl.gz")  # an interrupted first page: start clean
    t0 = time.time()
    while True:
        if state["token"]:
            url = OAI_URL + "?" + urllib.parse.urlencode({"verb": "ListRecords", "resumptionToken": state["token"]})
        else:
            url = OAI_URL + "?" + urllib.parse.urlencode({"verb": "ListRecords", "set": set_spec,
                                                          "metadataPrefix": "arXiv"})
        body = http_get(url, limit=200 << 20)
        if not isinstance(body, bytes):
            raise RuntimeError(f"{set_spec}: no answer from the OAI-PMH server")
        records, token, error = parse_oai_page(body)
        if error == "badResumptionToken":  # expired position: restart this set (duplicates are removed later)
            print(f"  [{set_spec}] saved position expired; restarting the set", flush=True)
            state.update(token=None, pages=0, records=0)
            if os.path.exists(base + ".jsonl.gz"):
                os.remove(base + ".jsonl.gz")
            continue
        if error and error != "noRecordsMatch":
            raise RuntimeError(f"{set_spec}: OAI-PMH error {error}")
        with gzip.open(base + ".jsonl.gz", "at", encoding="utf-8") as out:  # gzip members append cleanly
            for r in records:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
        state["pages"] += 1
        state["records"] += len(records)
        state["token"] = token
        state["complete"] = token is None
        with open(state_path + ".part", "w", encoding="utf-8") as handle:
            json.dump(state, handle)
        os.replace(state_path + ".part", state_path)
        if state["pages"] % 20 == 0 or state["complete"]:
            print(f"  [{set_spec}] {state['records']:,} records, {state['pages']:,} pages "
                  f"({time.time() - t0:,.0f} s)", flush=True)
        if state["complete"]:
            return state["records"]


def read_cache(cache_dir, sets):
    """Records of the given sets, each paper once."""
    seen = set()
    for set_spec in sets:
        path = os.path.join(cache_dir, _set_file(set_spec) + ".jsonl.gz")
        if not os.path.isfile(path):
            continue
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:  # a line cut by an interruption
                    continue
                if rec["id"] and rec["id"] not in seen:
                    seen.add(rec["id"])
                    yield rec


# ---------------------------------------------------------------------------
# 2. select
# ---------------------------------------------------------------------------
def paper_group(categories):
    """The highest-priority group any of the paper's categories belongs to."""
    cats = categories.split()
    best = None
    for name, prefixes in GROUPS:
        if any(c == p or c.startswith(p + ".") for c in cats for p in prefixes):
            if best is None or GROUP_RANK[name] < GROUP_RANK[best]:
                best = name
    return best


def build_list(cache_dir, sets, keyword_path, min_hits, min_distinct, path):
    kw = dp.KeywordFilter(dp.read_terms(keyword_path), [], min_hits, min_distinct, title_weight=3)
    rows, counts = [], {"records": 0, "no_group": 0, "keywords": 0}
    t0 = time.time()
    for rec in read_cache(cache_dir, sets):
        counts["records"] += 1
        group = paper_group(rec["categories"])
        if group is None:
            counts["no_group"] += 1
            continue
        keep, _, hits, _ = kw.evaluate(rec["title"], rec["abstract"])
        if not keep:
            counts["keywords"] += 1
            continue
        words = len(rec["title"].split()) + len(rec["abstract"].split())
        score = hits / math.sqrt(max(words, 50))
        rows.append((GROUP_RANK[group], -score, rec["id"], group, rec["created"][:4], rec["categories"],
                     rec["license"], rec["title"]))
        if counts["records"] % 200_000 == 0:
            print(f"  checked {counts['records']:,} papers ({time.time() - t0:,.0f} s)", flush=True)
    rows.sort()
    with open(path + ".part", "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(LIST_COLUMNS) + "\n")
        for _, neg, pid, group, year, cats, lic, title in rows:
            handle.write("\t".join([pid, group, f"{-neg:.3f}", year, cats, lic, _flat(title)]) + "\n")
    os.replace(path + ".part", path)
    by_group = {}
    for r in rows:
        by_group[r[3]] = by_group.get(r[3], 0) + 1
    print(f"Paper list: {len(rows):,} of {counts['records']:,} papers match the keywords {by_group} "
          f"(no matching category {counts['no_group']:,}, too few keyword matches {counts['keywords']:,}); {path}")
    return [dict(zip(LIST_COLUMNS, [r[2], r[3], f"{-r[1]:.3f}", r[4], r[5], r[6], r[7]])) for r in rows]


def read_list(path):
    with open(path, encoding="utf-8") as handle:
        head = handle.readline().rstrip("\n").split("\t")
        return [dict(zip(head, line.rstrip("\n").split("\t"))) for line in handle if line.strip()]


# ---------------------------------------------------------------------------
# 3. LaTeX source -> Markdown
# ---------------------------------------------------------------------------
def _decode(data):
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _gunzip(data, limit=MAX_UNPACKED):
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as handle:
        out = handle.read(limit + 1)
    if len(out) > limit:
        raise ValueError("source archive too large when unpacked")
    return out


def tex_files(blob):
    """{relative name: text} of the .tex and .bbl files of an e-print (a tar.gz
    archive, a gzipped single file or a plain file); None for a PDF-only
    submission. Members are read in memory; names that leave the archive
    folder are ignored."""
    if blob[:2] == b"\x1f\x8b":
        blob = _gunzip(blob)
    if blob[:5] == b"%PDF-":
        return None
    try:
        with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
            out, total = {}, 0
            for member in tar.getmembers():
                if not member.isfile() or member.size > MAX_TEX_FILE:
                    continue
                name = member.name.replace("\\", "/")
                if not name.lower().endswith((".tex", ".bbl")):
                    continue
                norm = os.path.normpath(name).replace("\\", "/")
                if norm.startswith("../") or norm == ".." or os.path.isabs(norm) or re.match(r"^[A-Za-z]:", norm):
                    continue
                total += member.size
                if total > 4 * MAX_TEX_FILE:
                    break
                out[norm] = _decode(tar.extractfile(member).read())
            return out
    except tarfile.ReadError:
        text = _decode(blob)
        return {"main.tex": text} if "\\" in text[:100_000] else {}


def main_tex(files):
    """The file with \\begin{document} (the longest if several)."""
    cands = [n for n, t in files.items() if n.endswith(".tex") and re.search(r"\\begin\s*\{document\}", t)]
    return max(cands, key=lambda n: len(files[n])) if cands else None


_COMMENT_RE = re.compile(r"(?<!\\)%.*")
_INPUT_RE = re.compile(r"\\(?:input|include|subfile)\s*\{([^{}]+)\}|\\input\s+([^\s{}\\]+)")


def flatten(files, name, depth=0):
    """The main file with \\input/\\include replaced by the included files
    (inside the archive only; anything else is dropped), comments removed."""
    text = _COMMENT_RE.sub("", files[name])
    base = os.path.dirname(name)

    def include(m):
        target = (m.group(1) or m.group(2) or "").strip()
        if depth >= 10 or not target:
            return ""
        for cand in (target, target + ".tex"):
            for key in (os.path.normpath(os.path.join(base, cand)).replace("\\", "/"),
                        os.path.normpath(cand).replace("\\", "/")):
                if key in files and key != name:
                    return "\n" + flatten(files, key, depth + 1) + "\n"
        return ""

    return _INPUT_RE.sub(include, text)


_FIGURE_RE = re.compile(r"\\begin\s*\{(figure\*?|wrapfigure|SCfigure|sidewaysfigure)\}(.*?)\\end\s*\{\1\}", re.S)


def _braced(text, start):
    """Content of the {...} group starting at text[start] == '{' (nesting
    allowed) and the index after it; (None, start) if unbalanced."""
    depth = 0
    for i in range(start, min(len(text), start + 20_000)):
        ch = text[i]
        if ch == "\\":
            continue
        if ch == "{" and text[i - 1] != "\\":
            depth += 1
        elif ch == "}" and text[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
    return None, start


def figures_to_captions(tex):
    """Each figure environment becomes its main caption as a paragraph
    ('Figure: ...'): the images carry no text, and pandoc's Markdown drops the
    caption of a figure with several panels."""
    def caption(m):
        body, captions, pos = m.group(2), [], 0
        while True:
            k = body.find("\\caption", pos)
            if k < 0:
                break
            j = k + len("\\caption")
            if body.startswith("of", j):  # \captionof: skip its {type} argument
                j += 2
                _, j = _braced(body, body.find("{", j)) if "{" in body[j:j + 5] else (None, j)
            while j < len(body) and body[j] in " \t\n*":
                j += 1
            if j < len(body) and body[j] == "[":  # optional short caption
                close = body.find("]", j)
                j = close + 1 if close > 0 else j
            while j < len(body) and body[j] in " \t\n":
                j += 1
            text, end = _braced(body, j) if j < len(body) and body[j] == "{" else (None, j + 1)
            if text:
                captions.append(text)
            pos = max(end, k + 1)
        return f"\n\n\\par Figure: {max(captions, key=len)}\\par\n\n" if captions else "\n\n"
    return _FIGURE_RE.sub(caption, tex)


def pandoc_path():
    try:
        import pypandoc
        return pypandoc.get_pandoc_path()
    except Exception:  # noqa: BLE001 - pypandoc missing or without its binary
        return shutil.which("pandoc")


PANDOC_TO = ("markdown-raw_html-raw_tex-raw_attribute-fenced_divs-bracketed_spans-native_divs-native_spans"
             "-link_attributes-header_attributes-simple_tables-multiline_tables-grid_tables-subscript-superscript-smart"
             "+pipe_tables")


def run_pandoc(tex, pandoc):
    """Markdown of a flattened LaTeX document (headings shifted one level, so
    the title is the only top-level heading)."""
    with tempfile.TemporaryDirectory(prefix="arxiv_") as tmp:
        src = os.path.join(tmp, "paper.tex")
        with open(src, "w", encoding="utf-8") as handle:
            handle.write(tex)
        result = subprocess.run([pandoc, src, "-f", "latex", "-t", PANDOC_TO, "--wrap=none", "--sandbox",
                                 "--shift-heading-level-by=1"], capture_output=True, timeout=PANDOC_SECONDS,
                                cwd=tmp)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", "replace").strip()[:200] or "pandoc failed")
    return result.stdout.decode("utf-8", "replace")


sys.path.insert(0, HERE)
import convert_to_markdown as ctm  # noqa: E402 - at start, so a running job survives files being moved
import data_prep as dp  # noqa: E402
from download_pmc import EMAIL_RE, SKIP_SECTION_RE, safe_name  # noqa: E402

_CITE_KEY = r"@[\w][\w:.#$%&+?<>~/-]*[\w]|@[\w]"
_CITE_GROUP_RE = re.compile(r"\[(?=[^\[\]]*@)([^\[\]]*)\]")
_CITE_BARE_RE = re.compile(r"(?<![\w@\[`])(" + _CITE_KEY + r")")
_IMAGE_RE = re.compile(r"!\[((?:[^\[\]]|\[[^\[\]]*\])*)\]\([^()]*(?:\([^()]*\)[^()]*)*\)")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def number_citations(md):
    """[@a; @b] -> [1, 2] and bare @a -> [1], numbered in order of first use."""
    numbers = {}

    def num(key):
        return numbers.setdefault(key, len(numbers) + 1)

    def group(m):
        keys = re.findall(_CITE_KEY, m.group(1))
        return "[" + ", ".join(str(num(k)) for k in dict.fromkeys(keys)) + "]" if keys else m.group(0)

    md = _CITE_GROUP_RE.sub(group, md)
    return _CITE_BARE_RE.sub(lambda m: f"[{num(m.group(1))}]", md)


def drop_sections(md):
    """Leave out back-matter sections (references, acknowledgements, funding...)."""
    out, skip_level = [], None
    for line in md.split("\n"):
        m = _HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            if skip_level is not None and level <= skip_level:
                skip_level = None
            if skip_level is None and SKIP_SECTION_RE.match(re.sub(r"^[\d.\s]+", "", m.group(2))):
                skip_level = level
        if skip_level is None:
            out.append(line)
    return "\n".join(out)


def clean_markdown(body, title, abstract):
    body = body.replace("\r\n", "\n").replace("\r", "\n")
    body = EMAIL_RE.sub("[email]", body)
    body = _IMAGE_RE.sub(lambda m: f"Figure: {m.group(1).strip()}"  # stray images: pandoc's default text is "image"
                         if m.group(1).strip() not in ("", "image") else "", body)
    body = number_citations(body)
    body = re.sub(r"\\label\{[^{}]*\}", "", body)  # equation and section labels: noise for a language model
    body = drop_sections(body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    head = f"# {_flat(title)}\n\n" + (f"## Abstract\n\n{EMAIL_RE.sub('[email]', _flat(abstract))}\n\n" if abstract else "")
    return head + body + "\n"


def pdf_markdown(pdf_bytes):
    with tempfile.TemporaryDirectory(prefix="arxiv_pdf_") as tmp:
        path = os.path.join(tmp, "paper.pdf")
        with open(path, "wb") as handle:
            handle.write(pdf_bytes)
        text = ctm.convert_pdf(path, types.SimpleNamespace(extract_images=False, out=None))
    text = EMAIL_RE.sub("[email]", text.replace("\r\n", "\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def latex_markdown(blob, title, abstract, pandoc):
    """(markdown or None, reason) from an e-print's bytes."""
    files = tex_files(blob)
    if files is None:
        return None, "pdf_only"
    name = main_tex(files)
    if name is None:
        return None, "no_main_tex"
    body = run_pandoc(figures_to_captions(flatten(files, name)), pandoc)
    return clean_markdown(body, title, abstract), "latex"


def convert_paper(row, blob, out_dir, min_body, pandoc):
    """Convert one downloaded paper (falling back to its PDF) and write it."""
    base = {k: row.get(k, "") for k in ("id", "group", "year", "categories")}
    base.update(licence=row.get("licence", ""), chars=0, method="", file="")
    md, method = None, ""
    if isinstance(blob, bytes):
        try:
            md, method = latex_markdown(blob, row["title"], row.get("abstract", ""), pandoc)
        except (subprocess.TimeoutExpired, RuntimeError, ValueError, tarfile.TarError, EOFError, OSError) as exc:
            md, method = None, f"latex failed: {type(exc).__name__}"
        if md is not None and len(md) - len(row["title"]) - len(row.get("abstract", "")) < min_body:
            md, method = None, "latex too short"
    if md is None:  # PDF-only submission, unreadable LaTeX or no source: the PDF
        pdf = blob if isinstance(blob, bytes) and blob[:5] == b"%PDF-" else http_get(PDF_URL.format(row["id"]))
        if not isinstance(pdf, bytes) or pdf[:5] != b"%PDF-":
            return {**base, "status": "no_text", "method": method or "no source"}
        try:
            md = pdf_markdown(pdf)
            method = "pdf" if not method or method == "pdf_only" else f"pdf ({method})"
        except Exception as exc:  # noqa: BLE001 - a damaged PDF: recorded, not fatal
            return {**base, "status": "no_text", "method": f"pdf failed: {type(exc).__name__}"}
    if len(md) < min_body:
        return {**base, "status": "too_short", "chars": len(md), "method": method}
    name = f"{row['id'].replace('/', '_')}__{safe_name(row['title'])}.md"
    path = os.path.join(out_dir, name)
    with open(path + ".part", "w", encoding="utf-8", newline="\n") as handle:
        handle.write(md)
    os.replace(path + ".part", path)
    return {**base, "status": "ok", "chars": len(md), "method": method, "file": name}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def read_report(path):
    done = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            head = handle.readline().rstrip("\n").split("\t")
            for line in handle:
                row = dict(zip(head, line.rstrip("\n").split("\t")))
                done[row.get("id")] = row.get("status")
    return done


def _lower_priority():
    try:
        import psutil
        proc = psutil.Process()
        if sys.platform == "win32":
            if proc.nice() != psutil.IDLE_PRIORITY_CLASS:
                proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        elif proc.nice() < 10:
            proc.nice(10)
    except Exception:  # noqa: BLE001 - priority is a courtesy
        pass


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="folder for the Markdown files (e.g. ai_training_data\\arxiv)")
    p.add_argument("--meta", help="folder for the metadata cache, list.tsv and report.tsv (default: _<out name>_meta "
                                  "next to --out, which data_prep --text-root skips)")
    p.add_argument("--sets", default=",".join(DEFAULT_SETS), help="OAI-PMH sets (subject areas), comma-separated")
    p.add_argument("--keywords", default=os.path.join(HERE, "keywords_biology.txt"))
    p.add_argument("--min-hits", type=int, default=3, help="keyword matches in title + abstract (title counts 3x)")
    p.add_argument("--min-distinct", type=int, default=3, help="different keyword terms needed")
    p.add_argument("--max-papers", type=int, default=0, help="download at most this many (0 = all in the list)")
    p.add_argument("--min-body-chars", type=int, default=3000, help="skip papers whose text is shorter")
    p.add_argument("--rebuild-list", action="store_true", help="select again from the cached metadata (after "
                                                                "changing the keywords or thresholds)")
    p.add_argument("--list-only", action="store_true", help="harvest and select, then stop")
    args = p.parse_args(argv)
    args.sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    if args.max_papers < 0:
        p.error("--max-papers must be >= 0")
    args.out = os.path.abspath(args.out)
    args.meta = os.path.abspath(args.meta or os.path.join(os.path.dirname(args.out),
                                                          f"_{os.path.basename(args.out)}_meta"))
    return args


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    args = parse_args(argv)
    _lower_priority()
    pandoc = pandoc_path()
    if not pandoc:
        raise SystemExit("pandoc is needed: python -m pip install pypandoc_binary")
    cache = os.path.join(args.meta, "metadata")
    os.makedirs(cache, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    list_path = os.path.join(args.meta, "list.tsv")
    fresh = False
    print(f"Harvesting arXiv metadata for {len(args.sets)} subject areas into {cache} ...", flush=True)
    for set_spec in args.sets:
        state = os.path.join(cache, _set_file(set_spec) + ".state.json")
        was_complete = os.path.isfile(state) and json.load(open(state, encoding="utf-8")).get("complete")
        harvest(set_spec, cache)
        fresh |= not was_complete
    if fresh or args.rebuild_list or not os.path.isfile(list_path):
        rows = build_list(cache, args.sets, args.keywords, args.min_hits, args.min_distinct, list_path)
    else:
        rows = read_list(list_path)
        print(f"Paper list: {len(rows):,} papers from {list_path} (--rebuild-list selects again)")
    if args.list_only:
        return
    abstracts = {}
    for rec in read_cache(cache, args.sets):  # the abstract goes into each paper's Markdown
        abstracts[rec["id"]] = rec["abstract"]
    for row in rows:
        row["abstract"] = abstracts.get(row["id"], "")
    del abstracts
    report_path = os.path.join(args.meta, "report.tsv")
    done = read_report(report_path)
    have = {name.split("__", 1)[0] for name in os.listdir(args.out) if name.endswith(".md")}
    todo = [r for r in rows if r["id"].replace("/", "_") not in have and done.get(r["id"]) not in FINAL]
    if args.max_papers:
        todo = todo[:max(0, args.max_papers - len([1 for r in rows if done.get(r["id"]) == "ok"]))]
    print(f"{len(rows) - len(todo):,} papers done or not wanted; fetching {len(todo):,} into {args.out} "
          f"(one request every {MIN_REQUEST_GAP:.0f} s: about {len(todo) * MIN_REQUEST_GAP / 86400:.1f} days)",
          flush=True)
    new_report = not os.path.isfile(report_path)
    counts, t0, n = {}, time.time(), 0
    with open(report_path, "a", encoding="utf-8", newline="\n") as report, \
            cf.ThreadPoolExecutor(max_workers=2) as pool:
        if new_report:
            report.write("\t".join(REPORT_COLUMNS) + "\n")

        def record(future, row):
            nonlocal n
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - retried on the next run
                result = {"id": row["id"], "status": f"error: {type(exc).__name__}: {str(exc)[:80]}", "chars": 0,
                          "method": "", "group": row["group"], "year": row["year"],
                          "categories": row["categories"], "licence": row.get("licence", ""), "file": ""}
            report.write("\t".join(_flat(result[c]) for c in REPORT_COLUMNS) + "\n")
            report.flush()
            key = result["status"].split(":")[0]
            counts[key] = counts.get(key, 0) + 1
            n += 1
            if n % 50 == 0:
                rate = n / max(time.time() - t0, 1e-9)
                print(f"  {n:,}/{len(todo):,} papers ({rate * 3600:,.0f}/hour, about "
                      f"{(len(todo) - n) / max(rate, 1e-9) / 3600:,.1f} h left) {counts}", flush=True)

        pending = {}
        try:
            for row in todo:
                blob = None
                try:
                    blob = http_get(EPRINT_URL.format(row["id"]))
                except Exception as exc:  # noqa: BLE001 - network trouble: the PDF route may still work
                    blob = None
                    print(f"  {row['id']}: source download failed ({type(exc).__name__})", flush=True)
                pending[pool.submit(convert_paper, row, blob, args.out, args.min_body_chars, pandoc)] = row
                while len(pending) >= 4:
                    finished, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                    for future in finished:
                        record(future, pending.pop(future))
            for future in cf.as_completed(list(pending)):
                record(future, pending.pop(future))
        except KeyboardInterrupt:
            for future in pending:
                future.cancel()
            print("\nStopped; rerun the same command to continue.")
            raise
    print(f"Done: {counts.get('ok', 0):,} papers written to {args.out}; statuses {counts}; report {report_path}")


if __name__ == "__main__":
    main()
