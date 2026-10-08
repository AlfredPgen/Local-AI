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

Pacing: a random pause of --wait seconds (default 5-60) before each paper, on
top of the 3-second minimum; about 110 papers an hour with the default.

Example (everything the default subject areas offer, best matches first):
    python download_arxiv.py --out %USERPROFILE%\ai_training_data\arxiv
"""

import argparse
import datetime
import concurrent.futures as cf
import email.utils
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
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zlib

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
PDF_SECONDS = 180  # default --pdf-seconds: a PDF conversion that takes longer is stopped and the paper left
_PDF_CHILD = (
    "import sys, types; import convert_to_markdown as c; "
    "t = c.convert_pdf(sys.argv[1], types.SimpleNamespace(extract_images=False, out=None)); "
    "open(sys.argv[2], 'w', encoding='utf-8').write(t)")
# subject areas (OAI-PMH sets) and their priority groups
DEFAULT_SETS = ["q-bio", "physics:physics:bio-ph", "stat", "cs:cs:AI", "cs:cs:LG", "cs:cs:CL", "cs:cs:CV",
                "cs:cs:NE", "cs:cs:MA", "cs:cs:IR", "math"]
GROUPS = [("biology", ("q-bio", "physics.bio-ph")), ("statistics", ("stat",)),
          ("AI", ("cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.NE", "cs.MA", "cs.IR")), ("mathematics", ("math",))]
GROUP_RANK = {"biology": 0, "statistics": 1, "AI": 2, "mathematics": 2}  # AI and mathematics share a rank
FINAL = {"ok", "too_short", "no_text"}  # statuses not retried on a rerun
# "pdf_timeout" is retried only by a run with a larger --pdf-seconds; "error: ..." always
_TIMEOUT_RE = re.compile(r"pdf took over ([\d.]+) s")
# a PDF converter failure caused by the environment, not by the paper: retried on the next run (a MemoryError
# is not one: it comes from a huge or pathological PDF, which would be fetched and converted again on every run)
_NOT_THE_PAPER_RE = re.compile(r"\b(ImportError|ModuleNotFoundError|DLL load failed|No module named)\b")
REPORT_COLUMNS = ["id", "status", "chars", "method", "group", "year", "categories", "licence", "file"]
LIST_COLUMNS = ["id", "group", "score", "year", "categories", "licence", "title", "abstract"]
ARXIV_NS = "{http://arxiv.org/OAI/arXiv/}"
OAI_NS = "{http://www.openarchives.org/OAI/2.0/}"


# ---------------------------------------------------------------------------
# HTTP: one connection at a time, at most one request start every 3 seconds
# ---------------------------------------------------------------------------
_NET_LOCK = threading.Lock()
_LAST_START = [0.0]
_STOP = threading.Event()  # set on Ctrl+C: requests that are waiting give up at once
MAX_SLOW_DOWN_PAUSE = 7200.0  # seconds of server-requested pauses one request may sit through in total


class Stopped(Exception):
    """The run was stopped (Ctrl+C) while a request was waiting."""


def _pause(seconds):
    """Sleep, but give up (Stopped) as soon as the run is stopped; short steps
    keep the main thread responsive to Ctrl+C on Windows too."""
    end = time.monotonic() + seconds
    while True:
        left = end - time.monotonic()
        if left <= 0:
            return
        if _STOP.wait(min(left, 0.5)):
            raise Stopped("stopped")


def _acquire_net():
    while not _NET_LOCK.acquire(timeout=0.5):
        if _STOP.is_set():
            raise Stopped("stopped")


def _retry_after(headers):
    """Seconds asked for by a Retry-After header (a number or an HTTP date); None if absent or unreadable."""
    value = (headers.get("Retry-After") if headers is not None else None) or ""
    value = value.strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
        if when.tzinfo is None:
            when = when.replace(tzinfo=datetime.timezone.utc)
        return max(0.0, (when - datetime.datetime.now(datetime.timezone.utc)).total_seconds())
    except (TypeError, ValueError, IndexError, OverflowError):
        return None


def http_get(url, limit=MAX_DOWNLOAD, tries=4, timeout=60):
    """Bytes of url; None on 404 (or when the body exceeds `limit`, as
    'too_large'). Network failures and server errors are retried with backoff.
    "Too many requests" (429) and "busy" (503) pause ALL requests, not just
    this one: for as long as the server's Retry-After asks (at most an hour at a
    time; after 8 pauses at least 1, 2, 4 ... 10 minutes, so an outage is waited
    out; two hours in all), or without it for 1, 2, 4 ... 10 minutes, up to 8
    times. Raises Stopped when the run is stopped during a pause."""
    delay, slow_down, attempt, paused = 5.0, 0, 0, 0.0
    while True:
        _acquire_net()
        try:
            pause = _LAST_START[0] + MIN_REQUEST_GAP - time.monotonic()
            if pause > 0:
                _pause(pause)
            _LAST_START[0] = time.monotonic()
            try:
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=timeout) as response:
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
                if exc.code in (429, 503):
                    asked = _retry_after(exc.headers)
                    if asked is not None:  # the server says how long: honour it, short or long
                        wait = min(max(asked, MIN_REQUEST_GAP), 3600.0)
                        if slow_down >= 8:  # still busy after 8 short pauses: an outage, so wait longer
                            wait = max(wait, min(60.0 * 2 ** (slow_down - 8), 600.0))
                        go_on = paused + wait <= MAX_SLOW_DOWN_PAUSE
                    else:
                        wait = min(60.0 * 2 ** slow_down, 600.0)
                        go_on = slow_down < 8
                    if go_on:
                        slow_down += 1
                        paused += wait
                        # every thread waits: the next request may start only after the pause
                        _LAST_START[0] = time.monotonic() + wait - MIN_REQUEST_GAP
                        print(f"  arXiv asks to slow down ({exc.code}); pausing all requests for {wait:.0f} s",
                              flush=True)
                        continue
                attempt += 1
                if exc.code not in (500, 502, 504) or attempt >= tries:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException, OSError):
                attempt += 1
                if attempt >= tries:
                    raise
        finally:
            _NET_LOCK.release()
        _pause(delay + random.random())
        delay = min(delay * 2, 300)


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


def _window_end(start, days=30):
    """Last day (inclusive) of the date window that begins at `start`, never after today."""
    end = datetime.date.fromisoformat(start) + datetime.timedelta(days=days - 1)
    return min(end, datetime.date.today()).isoformat()


def harvest(set_spec, cache_dir, update=False):
    """Append the set's records to <cache>/<set>.jsonl.gz, page by page; a
    progress file lets a rerun continue (or restart the set if arXiv no longer
    accepts the saved position). With update=True a finished set is extended
    with the papers added or changed since its last harvest (from 3 days before
    it, so nothing at the boundary is missed; papers seen twice are removed
    later). Returns the number of records written."""
    base = os.path.join(cache_dir, _set_file(set_spec))
    state_path = base + ".state.json"
    cache_path = base + ".jsonl.gz"
    state = {"token": None, "pages": 0, "records": 0, "complete": False}
    if os.path.isfile(state_path):
        with open(state_path, encoding="utf-8") as handle:
            state.update(json.load(handle))
    # "bytes": the cache size when the progress file was last written. Anything after it is a page whose
    # progress was never saved (possibly a gzip member cut off by a crash): cut it off, it is fetched again.
    # Progress files from before this field have none and are left as they are.
    if state.get("bytes") is not None and os.path.isfile(cache_path) and os.path.getsize(cache_path) > state["bytes"]:
        with open(cache_path, "r+b") as handle:
            handle.truncate(state["bytes"])
    if state["complete"] and update:
        # sets finished before this field existed: the state file was last written when the harvest ended
        since = state.get("harvested_until") or datetime.date.fromtimestamp(os.path.getmtime(state_path)).isoformat()
        start = (datetime.date.fromisoformat(since) - datetime.timedelta(days=3)).isoformat()
        print(f"  [{set_spec}] cached: {state['records']:,} records; adding papers new or changed since {start}",
              flush=True)
        state.update(token=None, complete=False, **{"from": start})
    elif state["complete"]:
        print(f"  [{set_spec}] cached: {state['records']:,} records")
        return state["records"]
    if not state["token"] and not state["pages"] and os.path.exists(cache_path):
        os.remove(cache_path)  # an interrupted first page: start clean
    t0 = time.time()
    while True:
        if state["token"]:
            url = OAI_URL + "?" + urllib.parse.urlencode({"verb": "ListRecords", "resumptionToken": state["token"]})
        else:
            query = {"verb": "ListRecords", "set": set_spec, "metadataPrefix": "arXiv"}
            if state.get("from"):  # date windows: the server answers these fast, open-ended ranges slowly
                query["from"], query["until"] = state["from"], _window_end(state["from"])
            url = OAI_URL + "?" + urllib.parse.urlencode(query)
        try:
            body = http_get(url, limit=200 << 20, tries=3, timeout=120)
        except Exception as exc:  # noqa: BLE001
            # the server sometimes stops answering one saved position; the position carries the date
            # it has reached, so ask afresh from that date (the few papers seen twice are removed later)
            m = re.search(r"from%3D(\d{4}-\d{2}-\d{2})", state["token"] or "")
            if not m:
                raise
            print(f"  [{set_spec}] the saved position does not load ({type(exc).__name__}); asking again for "
                  f"everything from {m.group(1)}", flush=True)
            state.update(token=None, **{"from": m.group(1)})
            continue
        if not isinstance(body, bytes):
            raise RuntimeError(f"{set_spec}: no answer from the OAI-PMH server")
        records, token, error = parse_oai_page(body)
        if error == "badResumptionToken":
            if state.get("from"):  # a date window (an update, or a harvest in windows): ask for the window again
                print(f"  [{set_spec}] saved position expired; asking again for everything from {state['from']}",
                      flush=True)
                state["token"] = None  # the records kept so far stay; the few seen twice are removed later
                continue
            # an open-ended first harvest has no date to go back to: restart the set
            print(f"  [{set_spec}] saved position expired; restarting the set", flush=True)
            state.update(token=None, pages=0, records=0, bytes=0, **{"from": None})
            if os.path.exists(cache_path):
                os.remove(cache_path)
            continue
        if error and error != "noRecordsMatch":
            raise RuntimeError(f"{set_spec}: OAI-PMH error {error}")
        with gzip.open(cache_path, "at", encoding="utf-8") as out:  # gzip members append cleanly
            for r in records:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
        state["bytes"] = os.path.getsize(cache_path)
        state["pages"] += 1
        state["records"] += len(records)
        state["token"] = token
        state["complete"] = token is None
        if token is None and state.get("from"):  # one date window done: the next, until today
            end = _window_end(state["from"])
            if end < datetime.date.today().isoformat():
                state["from"] = (datetime.date.fromisoformat(end) + datetime.timedelta(days=1)).isoformat()
                state["complete"] = False
        if state["complete"]:
            state["harvested_until"] = datetime.date.today().isoformat()
        with open(state_path + ".part", "w", encoding="utf-8") as handle:
            json.dump(state, handle)
        os.replace(state_path + ".part", state_path)
        if state["pages"] % 20 == 0 or state["complete"]:
            print(f"  [{set_spec}] {state['records']:,} records, {state['pages']:,} pages "
                  f"({time.time() - t0:,.0f} s)", flush=True)
        if state["complete"]:
            return state["records"]


_ID_PREFIX = '{"id": "'


def _cache_lines(path, warn=True):
    """(line number, line) of one metadata cache file. A damaged file (a page
    cut off by a crash before the cache was guarded against that) is read up
    to the damage, with a warning, instead of stopping the run."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for number, line in enumerate(handle):
                yield number, line
    except (OSError, EOFError, zlib.error) as exc:  # gzip.BadGzipFile is an OSError
        if warn:
                print(f"  warning: {path} is damaged ({type(exc).__name__}: {exc}); using the records before the "
                  "damage. Delete it and its .state.json to harvest that set again.", flush=True)


def _line_id(line):
    """The paper id of a complete cache line (each starts with {"id": "...), else None."""
    if line.startswith(_ID_PREFIX) and line.endswith("}\n") and line.count(_ID_PREFIX) == 1:
        return line[len(_ID_PREFIX):line.find('"', len(_ID_PREFIX))] or None
    return None


def read_cache(cache_dir, sets):
    """Records of the given sets, each paper once: its newest copy, i.e. the
    last one in its set's file (an --update appends the papers that changed),
    from the first set it appears in. Two passes keep memory small: the first
    only notes where each paper's newest copy is."""
    paths = [os.path.join(cache_dir, _set_file(s) + ".jsonl.gz") for s in sets]
    paths = [p for p in paths if os.path.isfile(p)]
    newest = {}  # id -> set index << 40 | line number of the copy to keep
    for k, path in enumerate(paths):
        for number, line in _cache_lines(path):
            pid = _line_id(line)
            if pid is None:
                continue
            here = newest.get(pid)
            if here is None or here >> 40 == k:  # a later copy in the same set replaces the earlier one
                newest[pid] = k << 40 | number
    for k, path in enumerate(paths):
        for number, line in _cache_lines(path, warn=False):
            pid = _line_id(line)
            if pid is None or newest.get(pid) != (k << 40 | number):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:  # a line cut by an interruption
                continue
            if rec.get("id") == pid:
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
                     rec["license"], rec["title"], rec["abstract"]))
        if counts["records"] % 200_000 == 0:
            print(f"  checked {counts['records']:,} papers ({time.time() - t0:,.0f} s)", flush=True)
    rows.sort()
    with open(path + ".part", "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(LIST_COLUMNS) + "\n")
        for _, neg, pid, group, year, cats, lic, title, abstract in rows:
            handle.write("\t".join([pid, group, f"{-neg:.3f}", year, cats, lic, _flat(title), _flat(abstract)])
                         + "\n")
    os.replace(path + ".part", path)
    by_group = {}
    for r in rows:
        by_group[r[3]] = by_group.get(r[3], 0) + 1
    print(f"Paper list: {len(rows):,} of {counts['records']:,} papers match the keywords {by_group} "
          f"(no matching category {counts['no_group']:,}, too few keyword matches {counts['keywords']:,}); {path}")
    return [dict(zip(LIST_COLUMNS, [r[2], r[3], f"{-r[1]:.3f}", r[4], r[5], r[6], r[7], r[8]])) for r in rows]


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


_COMMENT_RE = re.compile(r"(?<!\\)%.*")
_INPUT_RE = re.compile(r"\\(?:input|include|subfile)\s*\{([^{}]+)\}|\\input\s+([^\s{}\\]+)")
_CLASS_RE = re.compile(r"\\document(?:class|style)\s*(?:\[[^\]]*\])?\s*\{([^{}]*)\}")


def _include_keys(name, target):
    """Archive names an \\input{target} in file `name` may refer to."""
    base = os.path.dirname(name)
    return [key for cand in (target, target + ".tex")
            for key in (os.path.normpath(os.path.join(base, cand)).replace("\\", "/"),
                        os.path.normpath(cand).replace("\\", "/"))]


def main_tex(files):
    """The main file: a .tex file with \\begin{document} outside comments that
    is not a part of another one (a subfiles section, a standalone figure, a
    file another candidate \\input's). If several remain, one with a
    \\documentclass is preferred, then the longest once its \\input files are
    filled in (a short main.tex that only includes its sections beats a long
    supplement or template)."""
    texts = {n: _COMMENT_RE.sub("", t) for n, t in files.items() if n.endswith(".tex")}
    cands = [n for n, t in texts.items() if re.search(r"\\begin\s*\{document\}", t)]
    if not cands:
        return None
    parts = set()
    for n in cands:
        m = _CLASS_RE.search(texts[n])
        if m and m.group(1).strip() in ("subfiles", "standalone"):
            parts.add(n)
        for inc in _INPUT_RE.finditer(texts[n]):
            target = (inc.group(1) or inc.group(2) or "").strip()
            if target:
                parts.update(k for k in _include_keys(n, target) if k != n)
    cands = [n for n in cands if n not in parts] or cands
    return max(cands, key=lambda n: (bool(_CLASS_RE.search(texts[n])), len(flatten(files, n)), n))


def flatten(files, name, depth=0):
    """The main file with \\input/\\include replaced by the included files
    (inside the archive only; anything else is dropped), comments removed."""
    text = _COMMENT_RE.sub("", files[name])

    def include(m):
        target = (m.group(1) or m.group(2) or "").strip()
        if depth >= 10 or not target:
            return ""
        for key in _include_keys(name, target):
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


def pdf_markdown(pdf_bytes, seconds=PDF_SECONDS):
    """Markdown of a PDF, converted in a child process that is stopped after
    `seconds` (raises subprocess.TimeoutExpired): one pathological PDF must not
    hold up the whole download."""
    with tempfile.TemporaryDirectory(prefix="arxiv_pdf_") as tmp:
        path, out = os.path.join(tmp, "paper.pdf"), os.path.join(tmp, "paper.md")
        with open(path, "wb") as handle:
            handle.write(pdf_bytes)
        done = subprocess.run([sys.executable, "-c", _PDF_CHILD, path, out], cwd=HERE, capture_output=True,
                              timeout=seconds)
        if done.returncode != 0 or not os.path.isfile(out):
            err = done.stderr.decode("utf-8", "replace").strip().splitlines()
            raise RuntimeError(err[-1][:120] if err else f"PDF converter exit code {done.returncode}")
        with open(out, encoding="utf-8") as handle:
            text = handle.read()
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


def check_pdf_route():
    """None if the PDF converter can run, else the reason (checked once in a
    child process, as the conversions are)."""
    import importlib.util
    if importlib.util.find_spec("pymupdf4llm") is None:
        return "pymupdf4llm is not installed (python -m pip install pymupdf4llm)"
    try:
        done = subprocess.run([sys.executable, "-c", "import convert_to_markdown"], cwd=HERE, capture_output=True,
                              timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"the converter could not be started ({type(exc).__name__})"
    if done.returncode != 0:
        err = done.stderr.decode("utf-8", "replace").strip().splitlines()
        return "convert_to_markdown.py does not import: " + (err[-1][:160] if err else f"exit code {done.returncode}")
    return None


def convert_paper(row, blob, out_dir, min_body, pandoc, pdf_seconds=PDF_SECONDS, pdf_problem=None):
    """Convert one downloaded paper (falling back to its PDF) and write it.
    pdf_problem: why the PDF converter cannot run (from check_pdf_route); such
    papers get a status that a later run retries."""
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
        if pdf_problem:  # nothing to do with this paper: left for a run where the converter works
            return {**base, "status": "error: no PDF converter", "method": method or "no source"}
        pdf = blob if isinstance(blob, bytes) and blob[:5] == b"%PDF-" else http_get(PDF_URL.format(row["id"]))
        if not isinstance(pdf, bytes) or pdf[:5] != b"%PDF-":
            return {**base, "status": "no_text", "method": method or "no source"}
        try:
            md = pdf_markdown(pdf, pdf_seconds)
            method = "pdf" if not method or method == "pdf_only" else f"pdf ({method})"
        except subprocess.TimeoutExpired:  # retried by a run with a larger --pdf-seconds
            return {**base, "status": "pdf_timeout", "method": f"pdf took over {pdf_seconds:g} s"}
        except Exception as exc:  # noqa: BLE001 - a damaged PDF: recorded, not fatal
            reason = f"{type(exc).__name__}: {str(exc)[:80]}"
            if isinstance(exc, OSError) or _NOT_THE_PAPER_RE.search(reason):
                return {**base, "status": f"error: pdf failed: {reason}", "method": method}
            return {**base, "status": "no_text", "method": f"pdf failed: {reason}"}
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
def read_report(path, pdf_limits=None):
    """{id: last status}. pdf_limits, if given, is filled with {id: seconds}
    for papers whose PDF conversion ran out of time (status pdf_timeout, or
    no_text from before that status existed)."""
    done = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            head = handle.readline().rstrip("\n").split("\t")
            for line in handle:
                row = dict(zip(head, line.rstrip("\n").split("\t")))
                done[row.get("id")] = row.get("status")
                if pdf_limits is not None:
                    m = _TIMEOUT_RE.fullmatch(row.get("method") or "")
                    if m and row.get("status") in ("pdf_timeout", "no_text"):
                        pdf_limits[row.get("id")] = float(m.group(1))
                    else:
                        pdf_limits.pop(row.get("id"), None)
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
    p.add_argument("--keywords", default=os.path.join(HERE, "keywords.txt"))
    p.add_argument("--min-hits", type=int, default=3, help="keyword matches in title + abstract (title counts 3x)")
    p.add_argument("--min-distinct", type=int, default=3, help="different keyword terms needed")
    p.add_argument("--max-papers", type=int, default=0, help="download at most this many (0 = all in the list)")
    p.add_argument("--wait", default="5-60", help="random pause in seconds before each paper, MIN-MAX (default "
                                                  "5-60; 0 = only arXiv's 3 s minimum between requests)")
    p.add_argument("--min-body-chars", type=int, default=3000, help="skip papers whose text is shorter")
    p.add_argument("--pdf-seconds", type=float, default=PDF_SECONDS,
                   help=f"time limit of one PDF conversion (default {PDF_SECONDS}); papers that ran out of time are "
                        "tried again by a run with a larger limit")
    p.add_argument("--rebuild-list", action="store_true", help="select again from the cached metadata (after "
                                                                "changing the keywords or thresholds)")
    p.add_argument("--update", action="store_true",
                   help="also fetch the papers added to arXiv since the last harvest, then select again")
    p.add_argument("--list-only", action="store_true", help="harvest and select, then stop")
    args = p.parse_args(argv)
    args.sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    if args.max_papers < 0:
        p.error("--max-papers must be >= 0")
    if args.pdf_seconds <= 0:
        p.error("--pdf-seconds must be > 0")
    try:
        low, _, high = args.wait.partition("-")
        args.wait = (float(low), float(high or low))
    except ValueError:
        p.error("--wait must be MIN-MAX seconds, e.g. 5-60")
    if not 0 <= args.wait[0] <= args.wait[1] <= 3600:
        p.error("--wait must be MIN-MAX with 0 <= MIN <= MAX <= 3600")
    # %USERPROFILE% / $HOME / ~ are expanded here too: PowerShell leaves %VAR% as it is
    args.out = os.path.abspath(os.path.expanduser(os.path.expandvars(args.out)))
    if args.meta:
        args.meta = os.path.expanduser(os.path.expandvars(args.meta))
    args.meta = os.path.abspath(args.meta or os.path.join(os.path.dirname(args.out),
                                                          f"_{os.path.basename(args.out)}_meta"))
    return args


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    args = parse_args(argv)
    _STOP.clear()  # an earlier run in this process may have been stopped (Ctrl+C)
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
        harvest(set_spec, cache, update=args.update)
        fresh |= not was_complete or args.update
    if fresh or args.rebuild_list or not os.path.isfile(list_path):
        rows = build_list(cache, args.sets, args.keywords, args.min_hits, args.min_distinct, list_path)
    else:
        rows = read_list(list_path)
        print(f"Paper list: {len(rows):,} papers from {list_path} (--rebuild-list selects again)")
    if args.list_only:
        return
    missing = {r["id"] for r in rows if "abstract" not in r}  # a list.tsv written before it held the abstracts
    if missing:
        abstracts = {}
        for rec in read_cache(cache, args.sets):  # the abstract goes into each paper's Markdown
            if rec["id"] in missing:  # only the selected papers: the whole cache would take GBs
                abstracts[rec["id"]] = rec["abstract"]
        for row in rows:
            row.setdefault("abstract", abstracts.get(row["id"], ""))
        del abstracts
    report_path = os.path.join(args.meta, "report.tsv")
    pdf_limits = {}
    done = read_report(report_path, pdf_limits)
    have = {name.split("__", 1)[0] for name in os.listdir(args.out) if name.endswith(".md")}

    def wanted(row):
        if row["id"].replace("/", "_") in have:
            return False
        if row["id"] in pdf_limits:  # its PDF ran out of time: tried again only with a larger --pdf-seconds
            return args.pdf_seconds > pdf_limits[row["id"]]
        return done.get(row["id"]) not in FINAL

    todo = [r for r in rows if wanted(r)]
    pdf_problem = check_pdf_route()
    if pdf_problem:
        print(f"Warning: the PDF converter cannot run: {pdf_problem}. Papers that need it (PDF-only, or LaTeX "
              "pandoc cannot read) are left for a later run.", flush=True)
    if args.max_papers:
        todo = todo[:max(0, args.max_papers - len([1 for r in rows if done.get(r["id"]) == "ok"]))]
    per_paper = max(MIN_REQUEST_GAP, sum(args.wait) / 2)
    print(f"{len(rows) - len(todo):,} papers done or not wanted; fetching {len(todo):,} into {args.out} "
          f"(a random {args.wait[0]:g}-{args.wait[1]:g} s pause before each paper: about {3600 / per_paper:,.0f} "
          f"papers an hour, {len(todo) * per_paper / 86400:,.0f} days for all)", flush=True)
    pause = random.Random()  # not seeded: the pauses should not repeat from run to run
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
                if args.wait[1] > 0:
                    time.sleep(pause.uniform(*args.wait))
                blob = None
                try:
                    blob = http_get(EPRINT_URL.format(row["id"]))
                except Exception as exc:  # noqa: BLE001 - network trouble: the PDF route may still work
                    blob = None
                    print(f"  {row['id']}: source download failed ({type(exc).__name__})", flush=True)
                pending[pool.submit(convert_paper, row, blob, args.out, args.min_body_chars, pandoc,
                                    args.pdf_seconds, pdf_problem)] = row
                while len(pending) >= 4:
                    finished, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                    for future in finished:
                        record(future, pending.pop(future))
            for future in cf.as_completed(list(pending)):
                record(future, pending.pop(future))
        except KeyboardInterrupt:
            _STOP.set()  # a worker sitting in a slow-down pause gives up at once
            for future in pending:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            print("\nStopped; rerun the same command to continue.")
            raise
    print(f"Done: {counts.get('ok', 0):,} papers written to {args.out}; statuses {counts}; report {report_path}")


if __name__ == "__main__":
    main()
