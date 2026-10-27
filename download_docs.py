r"""Download software documentation listed in a sources table and store it as
Markdown training text for data_prep.py.

The sources table (default doc_sources.tsv next to this script) is UTF-8,
tab-separated, '#' lines are comments, and its header is exactly

    tool    kind    url    licence    notes

  tool     output folder name: lowercase letters, digits, '-' and '_' (plink2, bolt-lmm, ldak);
           several rows may share one tool
  kind     github | site | rtd | pdf | page | cran | bioc (below)
  url      what to fetch (below)
  licence  licence of the docs or software if stated (GPL-3.0, MIT, CC BY 4.0), else unknown
  notes    a short note for people ('PLINK 2.0 manual'); not used by the script

What each kind fetches:
  github  https://github.com/OWNER/REPO (optionally .../tree/REF/SUBFOLDER): the repository
          archive from codeload.github.com in one request. If it is larger than --max-mb, the
          file list comes from the GitHub API and only the documentation files are fetched from
          raw.githubusercontent.com (at most --max-pages of them). Kept: README files, Markdown,
          reStructuredText (converted by pandoc), R Markdown (.Rmd, kept as it is), notebooks
          (Markdown cells as text, code cells as fenced code, outputs dropped), Rd help pages
          (converted to Markdown), documentation .txt files (prose only, not data tables), HTML,
          LaTeX and PDF manuals inside doc folders (docs/, doc/, man/, vignettes/, examples/,
          tutorials/ ...) and example scripts (.sh .R .py .wdl .nf .smk) in example/tutorial
          folders or named like examples. Skipped: binaries, data files, files over 2 MB (PDF
          manuals: 20 MB), LICENSE/CHANGELOG-style files, tests, website theme templates and
          vendored third-party folders.
  site    a documentation site, crawled breadth-first from the URL: same host, under the URL's
          path prefix only (give the folder that holds the manual, e.g.
          https://www.cog-genomics.org/plink/2.0/), robots.txt obeyed, at most --max-pages
          pages. Linked PDFs inside the prefix are converted too and plain-text pages kept as
          .txt; other files are skipped. A start page that redirects to a folder above its own
          (a removed manual sent to the home page) is an error, not a crawl of the whole site.
  rtd     a Read the Docs project (https://PROJECT.readthedocs.io/en/latest/): the project's HTML
          zip (/_/downloads/<lang>/<version>/htmlzip/), at most --max-pages pages of it; if the
          project offers none, the site is crawled as above.
  pdf     a PDF manual, converted with convert_to_markdown.py's PDF converter.
  page    one HTML or plain-text page (a one-page manual, a README); no links are followed.
  cran    https://cran.r-project.org/package=NAME: the reference manual (the HTML version CRAN
          now publishes when the package page links it, else NAME.pdf), the vignettes (HTML, or
          PDF when there is no HTML) and the README.
  bioc    https://bioconductor.org/packages/NAME: the vignettes (HTML, else PDF) and the
          reference manual PDF.

HTML pages are reduced to their main content (main / article / [role=main] / .rst-content /
.md-content / .document / #content ...; else the body without navigation, header, footer and
sidebars) with BeautifulSoup, then converted by pandoc to GitHub Markdown without raw HTML:
headings, lists, tables (one line per cell), code blocks with their language, equations as
$...$; images are dropped and links keep only their text. Pages under 200 characters of text,
pages that are nearly all links (site maps, indexes) and exact duplicates are left out. E-mail
addresses become [email]; non-breaking spaces become plain spaces.

Outputs (all under --out, default %USERPROFILE%\ai_training_data\software_doc):
  <tool>\<row>\<name>.md   one file per page or document (.Rmd and .txt files keep their
                           type); <row> names the source row (github_OWNER_REPO, cran_NAME,
                           site_HOST_PATH ...) and <name> comes from the page or file path.
                           The files hold only documentation text, no source lines.
  <tool>\_sources.tsv      file -> page URL (provenance)
  _manifest.tsv            one row per sources row: tool, kind, url, licence, status, files,
                           chars, date, label (the row's folder, kept for good so that adding
                           or moving rows never hands one row's folder to another). Rows with
                           status ok or empty are skipped by later runs (--refresh redoes
                           them); errors are tried again, and so are partial rows (files kept,
                           but cut short by a server or network that kept failing, or a
                           converter that could not run on this machine). A row that finds no
                           files keeps the files of its earlier run.
data_prep.py reads *.md, *.txt and *.Rmd from these folders; nothing else is written there
(raw HTML, archives and PDFs are never saved).

Resumable: each row is built in a hidden folder (<tool>\.<row>.part, which data_prep skips)
and moved into place when it is complete; files are written to .part and renamed. Ctrl+C
leaves no half-written file; rerun the same command to continue. A failing row is recorded
with its error and the run goes on. One run at a time per --out folder: a second run on the
same folder stops with a message.

Pacing and safety: one request at a time, a random --wait pause (default 1-2 s) between two
requests to the same host (longer if robots.txt asks), and every host paused when a server
answers 429 or 503 (as download_arxiv.py does). That back-off is spent once per host: a host
that still refuses after it gets no more pauses, its refused URLs fail at once until it
answers normally, and a crawl cut short that way is retried by the next run. pandoc runs
with a 1.5 GB heap limit. The User-Agent says who is asking
('tinyGPT-docs/1.0 (personal research)'). Downloads are size-limited while they stream
(--max-mb per row). Nothing downloaded is executed, imported or evaluated: archives are read in
memory member by member (never extracted, unsafe member names ignored), HTML and JSON are
parsed as data, pandoc runs in its sandbox mode with a time limit, and PDFs are converted in a
separate, isolated (python -I) process with a time limit.

Examples:
    python download_docs.py --list
    python download_docs.py
    python download_docs.py --only plink2,regenie --refresh
    python download_docs.py --sources my_docs.tsv --out D:\docs_test --max-pages 15
"""

import argparse
import collections
import datetime
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.parse
import urllib.robotparser
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import convert_to_markdown as ctm  # noqa: E402  (code fences)
import download_arxiv as ax  # noqa: E402  (Retry-After, pandoc path, priority)
from download_pmc import EMAIL_RE  # noqa: E402

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError as _exc:  # pragma: no cover - both are installed on this machine
    raise SystemExit(f"download_docs.py needs requests and beautifulsoup4 ({_exc})")

USER_AGENT = "tinyGPT-docs/1.0 (personal research)"
ROBOTS_AGENT = "tinyGPT-docs"
KINDS = ("github", "site", "rtd", "pdf", "page", "cran", "bioc")
SOURCE_COLUMNS = ["tool", "kind", "url", "licence", "notes"]
MANIFEST_COLUMNS = ["tool", "kind", "url", "licence", "status", "files", "chars", "date", "label"]
TOOL_SOURCES_COLUMNS = ["file", "url", "source", "chars"]
FINISHED = ("ok", "empty")  # manifest statuses a rerun skips (without --refresh)
TOOL_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")
MAX_FILE = 2 << 20  # one repository file
MAX_REPO_PDF = 20 << 20  # a PDF manual inside a repository's doc folder
MAX_PAGE = 10 << 20  # one HTML page
MIN_CHARS = 200  # documents with less text are left out
NAV_SHARE = 0.8  # pages whose main text is more link text than this are navigation (site maps, indexes)
PANDOC_SECONDS = 120
PANDOC_HEAP = "1500m"  # pandoc's heap limit (+RTS -M): a huge page or a TeX macro bomb fails alone
PDF_SECONDS = 600  # default --pdf-seconds: manuals run to hundreds of pages
MAX_PATH_CHARS = 240  # longest file path written (Windows' limit is 260 unless long paths are enabled)
# converter failures caused by this machine, not by the document (pandoc crashed, no PDF converter):
# a row that met one is not finished, so the next run fetches it again
ENV_FAILURES = collections.Counter()
MAX_SLOW_DOWN_PAUSE = ax.MAX_SLOW_DOWN_PAUSE
WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
_flat = ax._flat
_PDF_CHILD = (
    "import sys, types; sys.path.insert(0, sys.argv[1]); import convert_to_markdown as c; "
    "t = c.convert_pdf(sys.argv[2], types.SimpleNamespace(extract_images=False, out=None)); "
    "open(sys.argv[3], 'w', encoding='utf-8').write(t)")


class RowError(Exception):
    """A sources row that cannot be done (recorded in the manifest; the run goes on)."""


class TooLarge(Exception):
    """A download that would pass its size limit."""


class NavigationPage(RuntimeError):
    """A page that is nearly all links (a site map or an index): no documentation text."""


class HttpError(Exception):
    def __init__(self, code, url):
        super().__init__(f"HTTP {code}")
        self.code, self.url = code, url


# ---------------------------------------------------------------------------
# HTTP: one request at a time, --wait between requests to one host, 429/503 pause all hosts
# ---------------------------------------------------------------------------
_ROBOTS_DOUBLE_SLASH = re.compile(r"^(\s*(?:dis)?allow\s*:\s*)//\s*(#.*)?$", re.I)
_ROBOTS_RULE = re.compile(r"^(\s*(?:dis)?allow\s*:\s*)([^#]*?)\s*(#.*)?$", re.I)
_ROBOTS_DELAY = re.compile(r"^(\s*crawl-delay\s*:\s*)(\d*\.\d+)\s*(#.*)?$", re.I)


def parse_robots(rp, text):
    """robotparser.parse with three fixes: a rule 'Disallow: //' (paths starting
    with '//') is not read as 'disallow everything'; '*' and '$' in rules work as
    wildcards (as Google and Bing read them) instead of literally, matched against
    the rule as written ('Disallow: /*?' keeps its '?', which robotparser's own
    normalising drops); and a fractional Crawl-delay (2.5) counts, rounded up."""
    lines, wild = [], {}
    for line in text.splitlines():
        line = _ROBOTS_DOUBLE_SLASH.sub(r"\1//*", line)
        m = _ROBOTS_RULE.match(line)
        if m and ("*" in m.group(2) or m.group(2).endswith("$")):
            token = f"/tinygpt-docs-wildcard-{len(wild)}"  # robotparser keeps this path as it is
            wild[token] = m.group(2)
            line = m.group(1) + token
        m = _ROBOTS_DELAY.match(line)
        if m:
            line = m.group(1) + str(max(1, math.ceil(float(m.group(2)))))
        lines.append(line)
    rp.parse(lines)
    for entry in rp.entries + ([rp.default_entry] if rp.default_entry else []):
        for rule in entry.rulelines:
            raw = wild.get(rule.path)
            if raw is not None:
                raw = urllib.parse.unquote(raw)
                anchored = raw.endswith("$")
                rx = re.compile("^" + ".*".join(re.escape(p) for p in raw.rstrip("$").split("*"))
                                + ("$" if anchored else ""))
                rule.applies_to = lambda filename, rx=rx: bool(rx.match(urllib.parse.unquote(filename)))


class Fetched:
    def __init__(self, url, ctype, kind, body):
        self.url, self.ctype, self.kind, self.body = url, ctype, kind, body


def content_kind(ctype, url, head=b""):
    """'html', 'pdf', 'zip', 'text' or 'other' from the Content-Type, the URL and the first bytes."""
    if head[:5] == b"%PDF-":
        return "pdf"
    if head[:4] == b"PK\x03\x04":
        return "zip"
    path = urllib.parse.urlsplit(url).path.lower()
    if ctype in ("text/html", "application/xhtml+xml"):
        return "html"
    if ctype in ("application/pdf", "application/x-pdf"):
        return "pdf"
    if ctype in ("application/zip", "application/x-zip-compressed"):
        return "zip"
    if ctype in ("", "application/octet-stream", "binary/octet-stream", "application/force-download"):
        if path.endswith(".pdf"):
            return "pdf"
        if path.endswith(".zip") or "/htmlzip" in path or "/zip/" in path:
            return "zip"
    return "text" if ctype.startswith("text/") else "other"


class Http:
    def __init__(self, wait):
        self.wait = wait
        self.last = {}  # host -> time.monotonic() of its last request
        self.delay = {}  # host -> Crawl-delay from robots.txt
        self.slow_until = 0.0  # after a 429/503 no request (to any host) starts before this
        # the 429/503 back-off is counted per host until the host answers normally again
        self.strikes = {}  # host -> pauses taken (or a lone 503 given up on)
        self.paused = {}  # host -> seconds paused
        self.spent = set()  # hosts still refusing after the whole back-off: a refusal now fails at once
        self.bytes = 0  # bytes of completed downloads (the per-row --max-mb budget)
        self.robots = {}
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def _turn(self, host):
        gap = max(random.uniform(*self.wait), self.delay.get(host, 0.0))
        start = max(self.last.get(host, -1e9) + gap, self.slow_until)
        left = start - time.monotonic()
        if left > 0:
            time.sleep(left)
        self.last[host] = time.monotonic()

    def get(self, url, limit, want=None, to_file=None, headers=None, tries=4, timeout=60):
        """Fetched(final url, content type, kind, body) of url, or None for 404/410.
        want: the kinds to download ('html', 'pdf', 'zip', 'text'); any other kind
        is not read (body None). to_file: stream the body into this file (body =
        its path) instead of memory. Raises TooLarge past `limit` bytes and
        HttpError for other refusals. Network failures and 500/502/504 are retried
        with backoff; 429 and 503 pause every host, as long as Retry-After asks (at
        most an hour at a time), else 1, 2, 4 ... 10 minutes, 8 times at most. That
        back-off belongs to the host, not to one URL: it is spent once until the
        host answers normally again, and while it is spent a refusal fails the URL
        at once (HttpError, no pause). A lone 503 without Retry-After from a host
        that was answering (one broken page) is retried like a 500."""
        delay, attempt = 5.0, 0
        host = urllib.parse.urlsplit(url).hostname or ""
        while True:
            self._turn(host)
            try:
                with self.session.get(url, stream=True, timeout=(20, timeout), headers=headers) as r:
                    code = r.status_code
                    if code not in (429, 503):
                        self.strikes.pop(host, None)
                        self.paused.pop(host, None)
                        self.spent.discard(host)
                    if code in (404, 410):
                        return None
                    if code in (429, 503):
                        asked = ax._retry_after(r.headers)
                        strikes, paused = self.strikes.get(host, 0), self.paused.get(host, 0.0)
                        if asked is None and code == 503 and not strikes:
                            if attempt + 1 >= tries:  # this URL fails; the next 503 from the host starts the pauses
                                self.strikes[host] = 1
                                raise HttpError(code, url)
                            attempt += 1
                        else:
                            if asked is not None:  # the server says how long: honour it
                                wait = min(max(asked, 1.0), 3600.0)
                                if strikes >= 8:  # still busy after 8 pauses: an outage, so wait longer
                                    wait = max(wait, min(60.0 * 2 ** (strikes - 8), 600.0))
                                go_on = paused + wait <= MAX_SLOW_DOWN_PAUSE
                            else:
                                wait = min(60.0 * 2 ** strikes, 600.0)
                                go_on = strikes < 8
                            if not go_on:
                                if host not in self.spent:
                                    self.spent.add(host)
                                    print(f"  {host} still answers {code}; its refusals now fail at once", flush=True)
                                raise HttpError(code, url)
                            self.strikes[host], self.paused[host] = strikes + 1, paused + wait
                            self.slow_until = time.monotonic() + wait
                            print(f"  {host} asks to slow down ({code}); pausing all requests for {wait:.0f} s",
                                  flush=True)
                            continue
                    elif code in (500, 502, 504) and attempt + 1 < tries:
                        attempt += 1
                    elif code >= 400:
                        raise HttpError(code, url)
                    else:
                        return self._read(r, url, limit, want, to_file)
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ContentDecodingError):
                attempt += 1
                if attempt >= tries:
                    raise
            time.sleep(delay + random.random())
            delay = min(delay * 2, 300)

    def _read(self, r, url, limit, want, to_file):
        ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        kind = content_kind(ctype, r.url)
        if want and kind not in want and kind not in ("other", "text"):
            return Fetched(r.url, ctype, kind, None)
        try:
            if int(r.headers.get("Content-Length") or 0) > limit:
                raise TooLarge(f"{url} is larger than {limit / (1 << 20):.1f} MB")
        except ValueError:
            pass
        size, chunks, out = 0, [], open(to_file, "wb") if to_file else None
        try:
            for chunk in r.iter_content(1 << 16):  # decompressed: the limit counts what is kept
                if size == 0:
                    kind = content_kind(ctype, r.url, chunk[:8])
                    if want and kind not in want:  # sniffed: e.g. a PDF served as octet-stream is wanted
                        return Fetched(r.url, ctype, kind, None)
                size += len(chunk)
                if size > limit:
                    raise TooLarge(f"{url} is larger than {limit / (1 << 20):.1f} MB")
                if out:
                    out.write(chunk)
                else:
                    chunks.append(chunk)
        finally:
            if out:
                out.close()
        self.bytes += size
        return Fetched(r.url, ctype, kind, to_file if to_file else b"".join(chunks))

    def allowed(self, url):
        """True if robots.txt lets this script fetch url (fetched once per host:
        401/403 forbid everything and other 4xx allow everything, as
        urllib.robotparser does; Crawl-delay is honoured, up to 60 s)."""
        parts = urllib.parse.urlsplit(url)
        key = f"{parts.scheme}://{parts.netloc}"
        rp = self.robots.get(key)
        if rp is None:
            rp = urllib.robotparser.RobotFileParser(key + "/robots.txt")
            try:
                got = self.get(key + "/robots.txt", 1 << 20)
            except HttpError as exc:
                if exc.code in (401, 403):
                    rp.disallow_all = True
                    rp.refused = exc.code
                elif exc.code < 500:
                    rp.allow_all = True
                else:
                    raise
            except TooLarge:
                rp.allow_all = True
            else:
                # none, or a web page instead (told apart by the text: some servers send robots.txt as text/html)
                text = decode_text(got.body) if got is not None and got.body else ""
                if not text.strip() or text.lstrip().startswith("<") or "<html" in text[:5000].lower():
                    rp.allow_all = True
                else:
                    parse_robots(rp, text)
                    delay = rp.crawl_delay(ROBOTS_AGENT)
                    if delay:
                        self.delay[parts.hostname] = min(float(delay), 60.0)
            self.robots[key] = rp
        return rp.can_fetch(ROBOTS_AGENT, url)

    def refusal(self, url):
        """Why robots.txt forbids url, for the manifest."""
        parts = urllib.parse.urlsplit(url)
        rp = self.robots.get(f"{parts.scheme}://{parts.netloc}")
        code = getattr(rp, "refused", None)
        if code:
            return (f"{parts.netloc} answers HTTP {code} to this User-Agent (robots.txt refused, so the site is off "
                    "limits)")
        return f"robots.txt of {parts.netloc} disallows {url}"


# ---------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------
def _cp1252_table():
    table = {}
    for byte in range(0x80, 0xA0):
        try:
            table[byte] = bytes([byte]).decode("cp1252")
        except UnicodeDecodeError:  # the five bytes cp1252 leaves undefined stay as Latin-1 reads them
            pass
    return table


_CP1252 = _cp1252_table()  # Latin-1's C1 controls (U+0080-U+009F) -> what Windows-1252 has there (’ – “ ...)


def decode_text(data):
    """Text of a file: UTF-8 (with or without a BOM), UTF-16 with a BOM, else Windows-1252."""
    if data[:3] == b"\xef\xbb\xbf":
        data = data[3:]
    elif data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16", "replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1").translate(_CP1252)


def _email_sub(m):
    found, after = m.group(0), m.string[m.end():m.end() + 2]
    if (found.lower().startswith("git@") or re.match(r":[^\s:]", after)  # git@host:owner/repo, scp's user@host:path
            or not re.fullmatch(r"[A-Za-z]{2,}", found.rsplit(".", 1)[-1])):  # pkg@1.12.1 (spack, npm)
        return found
    return "[email]"


def mask_emails(text):
    """E-mail addresses as [email]; git and scp addresses and package@version kept."""
    return EMAIL_RE.sub(_email_sub, text)


_FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})\s*([^`\s]*)\s*$")
_ESCAPED_RE = re.compile(r"\\([<>\[\]@~^'\"])")
# &nbsp; and its kin as plain spaces; zero-width spaces, byte-order marks and soft hyphens removed
_SPACES = str.maketrans({"\u00a0": " ", "\u2007": " ", "\u202f": " ", "\u200b": None, "\ufeff": None,
                         "\u00ad": None})


def tidy(md):
    """Pandoc's Markdown made plainer: ```lang fences (```nolang -> ```), no
    needless escapes (\\< \\> \\[ ...) or pilcrows outside code, e-mail
    addresses masked, non-breaking spaces made plain, trailing spaces and runs
    of blank lines removed."""
    out, fence = [], None
    for line in md.translate(_SPACES).replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        m = _FENCE_RE.match(line)
        if fence is None and m:
            fence = m.group(2)
            out.append(f"{m.group(1)}{fence}{'' if m.group(3) == 'nolang' else m.group(3)}")
            continue
        if fence is not None:
            if m and not m.group(3) and m.group(2)[0] == fence[0] and len(m.group(2)) >= len(fence):
                fence = None
                out.append(m.group(1) + m.group(2))
            else:
                out.append(line.rstrip())
            continue
        parts = re.split(r"(`[^`]*`)", line)  # inline code stays as it is
        parts = [p if k % 2 else _ESCAPED_RE.sub(r"\1", p).replace("¶", "") for k, p in enumerate(parts)]
        line = "".join(parts).rstrip()
        if line.endswith("\\") and not line.endswith("\\\\"):  # pandoc's hard line break (<br>): a plain newline
            line = line[:-1].rstrip()
        out.append(line)
    md = mask_emails("\n".join(out))
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    md = re.sub(r"\A(?:(?:-{3,}|\*{3,}|_{3,})\s*\n+)+", "", md)  # a rule left at the top by a dropped logo
    return md + "\n"


def tidy_raw(text):
    """A text kept as it is (R Markdown, .txt, scripts): line endings, masked
    e-mail addresses, plain spaces, trailing spaces and runs of blank lines only."""
    text = text.translate(_SPACES).replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", mask_emails(text)).strip() + "\n"


_MD_IMAGE_RE = re.compile(r"!\[[^\]\n]*\]\([^)\n]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]\n]+)\]\((?:[^()\n]|\([^()\n]*\))*\)")
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9-]*(?:\s[^<>]*)?/?>")
_CODE_SPAN_RE = re.compile(r"(`+)(?!`).*?(?<!`)\1(?!`)")


def _outside_code(line, change):
    """change(line) with the line's inline code (`plink --bfile <prefix>`, `x[1](2)`) set aside and put back."""
    spans = []

    def hide(m):
        spans.append(m.group(0))
        return f"\ue000{len(spans) - 1}\ue001"
    return re.sub("\ue000(\\d+)\ue001", lambda m: spans[int(m.group(1))], change(_CODE_SPAN_RE.sub(hide, line)))


def _strip_markup(line):
    return _HTML_TAG_RE.sub("", _MD_LINK_RE.sub(r"\1", _MD_IMAGE_RE.sub("", line)))


def clean_rmd(text):
    """R Markdown kept as it is, but outside code chunks and inline code without
    HTML tags and comments, images and link targets (links keep their text)."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    out, fence = [], None
    for line in text.replace("\r\n", "\n").split("\n"):
        m = re.match(r"^\s*(`{3,}|~{3,})", line)
        if m:
            if fence is None:
                fence = m.group(1)
            elif m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence):
                fence = None
            out.append(line)
            continue
        if fence is None:
            line = _outside_code(line, _strip_markup)
        out.append(line)
    return tidy_raw("\n".join(out))


def visible_chars(md):
    return len(" ".join(md.split()))


_COMMON_WORDS = {"the", "and", "of", "to", "a", "in", "is", "for", "with", "that", "this", "be", "are", "on", "as",
                 "by", "or", "an", "it", "from", "can", "if", "use", "file", "files", "will", "not", "which", "you"}


def looks_like_prose(text):
    """True for a text file that reads as documentation, not a data table or a list of IDs."""
    sample = text[:20000]
    nonspace = sum(not c.isspace() for c in sample)
    if not nonspace or sum(c.isalpha() for c in sample) / nonspace < 0.6:
        return False
    words = re.findall(r"[A-Za-z]{1,}", sample)
    if len(words) < 30 or sum(w.lower() in _COMMON_WORDS for w in words) / len(words) < 0.05:
        return False
    lines = [line for line in sample.splitlines() if line.strip()]
    return not lines or sum("\t" in line for line in lines) / len(lines) < 0.5


# ---------------------------------------------------------------------------
# pandoc: any input -> plain GitHub Markdown (Lua filter below), in its sandbox, with a time limit
# ---------------------------------------------------------------------------
PANDOC_TO = "gfm-raw_html-tex_math_gfm+tex_math_dollars"
HTML_FROM = "html+tex_math_dollars+tex_math_single_backslash"
LUA_FILTER = r"""
-- download_docs.py: documentation -> plain GitHub Markdown for a text-only language model
local function inlines_of(blocks)
  local out = pandoc.Inlines{}
  local function add(inl)
    if #inl == 0 then return end
    if #out > 0 then out:insert(pandoc.Space()) end
    out:extend(inl)
  end
  for _, b in ipairs(blocks) do
    if b.t == 'Plain' or b.t == 'Para' or b.t == 'Header' then add(b.content)
    elseif b.t == 'LineBlock' then for _, l in ipairs(b.content) do add(l) end
    elseif b.t == 'BulletList' or b.t == 'OrderedList' then
      for _, item in ipairs(b.content) do add(inlines_of(item)) end
    elseif b.t == 'Div' or b.t == 'BlockQuote' or b.t == 'Figure' then add(inlines_of(b.content))
    elseif b.t == 'CodeBlock' then add({pandoc.Code((b.text:gsub('%s+', ' ')))})
    elseif b.t == 'DefinitionList' then
      for _, item in ipairs(b.content) do
        add(item[1])
        for _, d in ipairs(item[2]) do add(inlines_of(d)) end
      end
    else
      local s = pandoc.utils.stringify(b)
      if s ~= '' then add({pandoc.Str(s)}) end
    end
  end
  return out:walk{LineBreak = function() return pandoc.Space() end,
                  SoftBreak = function() return pandoc.Space() end}
end

local inline_filter = {
  Image = function() return {} end,
  Link = function(el) return el.content end,
  RawInline = function(el)
    if el.text:match('^<[Bb][Rr]') then return pandoc.Space() end
    return {}
  end,
}

-- a '|' in a table cell's code would end the cell; GitHub reads '\|' there as '|'
local function escape_pipes(el)
  el.text = el.text:gsub('|', '\\|')
  return el
end

-- Sphinx directives pandoc does not know: a toctree or autosummary is a list of names; an autodoc
-- directive (automodule, autoclass ...) starts with the object's name, then any text written for it
local AUTODOC = {automodule = true, autoclass = true, autofunction = true, automethod = true,
                 autoattribute = true, autodata = true, autoexception = true, autoproperty = true,
                 autodecorator = true, autoprogram = true}

local block_filter
block_filter = {
  -- a pipe table needs one line per cell: cells are flattened, spans dropped, rows padded
  Table = function(tbl)
    local s = pandoc.utils.to_simple_table(tbl)
    local width = #s.headers
    for _, row in ipairs(s.rows) do if #row > width then width = #row end end
    if width == 0 then return {} end
    local function cells(row)
      local out = {}
      for k = 1, width do out[k] = {pandoc.Plain(inlines_of(row[k] or {}):walk{Code = escape_pipes})} end
      return out
    end
    local rows = {}
    for i, row in ipairs(s.rows) do rows[i] = cells(row) end
    s.headers, s.rows, s.aligns, s.widths = cells(s.headers), rows, {}, {}
    for k = 1, width do s.aligns[k] = pandoc.AlignDefault; s.widths[k] = 0 end
    s.caption = {}
    return pandoc.utils.from_simple_table(s)
  end,
  CodeBlock = function(el)
    local lang = 'nolang'
    for _, c in ipairs(el.classes) do
      if c ~= 'sourceCode' and c ~= 'numberSource' and c ~= 'numberLines' and c ~= 'highlight' then
        lang = c; break
      end
    end
    el.classes, el.attributes, el.identifier = {lang}, {}, ''
    return el
  end,
  -- **term**: definition, one paragraph per item (gfm has no definition lists)
  DefinitionList = function(el)
    local out = {}
    for _, item in ipairs(el.content) do
      local term, defs = item[1], item[2]
      local bold = true
      for _, x in ipairs(term) do if x.t == 'Strong' or x.t == 'Code' then bold = false end end
      local head = bold and pandoc.Inlines{pandoc.Strong(term)} or pandoc.Inlines(term)
      local first = defs[1] or {}
      if #defs == 1 and #first == 1 and (first[1].t == 'Para' or first[1].t == 'Plain') then
        head:extend{pandoc.Str(':'), pandoc.Space()}
        head:extend(first[1].content)
        table.insert(out, pandoc.Para(head))
      else
        table.insert(out, pandoc.Para(head))
        for _, d in ipairs(defs) do for _, b in ipairs(d) do table.insert(out, b) end end
      end
    end
    return out
  end,
  Div = function(el)
    for _, c in ipairs(el.classes) do
      if c == 'toctree' or c == 'autosummary' then return {} end
      if AUTODOC[c] then
        local first = el.content[1]
        if first and (first.t == 'Para' or first.t == 'Plain') then el.content:remove(1) end
        return el.content
      end
    end
  end,
  -- HTML blocks inside Markdown (centred titles, <details>): their text is kept, the tags dropped
  RawBlock = function(el)
    if el.format:match('html') then
      local ok, doc = pcall(pandoc.read, el.text, 'html')
      if ok then return doc.blocks:walk(inline_filter):walk(block_filter) end
    end
    return {}
  end,
}

return {inline_filter, block_filter}
"""
PANDOC = {"path": None, "lua": None}


def pandoc_md(text, fmt):
    """Markdown of `text` (pandoc input format `fmt`): GitHub flavour without raw
    HTML, images dropped, links reduced to their text, tables one line per cell.
    A pandoc that crashed (it could not start: 0xC0000142 when Windows is short
    of resources) is run once more. Its heap is limited (PANDOC_HEAP): a document
    that needs more fails on its own ('pandoc: Heap exhausted')."""
    for attempt in (1, 2):
        with tempfile.TemporaryDirectory(prefix="docs_pandoc_") as tmp:
            done = subprocess.run([PANDOC["path"], "+RTS", f"-M{PANDOC_HEAP}", "-RTS", "-f", fmt, "-t", PANDOC_TO,
                                   "--wrap=none", "--sandbox", "--lua-filter", PANDOC["lua"]],
                                  input=text.encode("utf-8", "replace"), capture_output=True,
                                  timeout=PANDOC_SECONDS, cwd=tmp)
        crashed = done.returncode < 0 or done.returncode >= 0xC0000000  # a signal or a Windows NTSTATUS
        if not crashed or attempt == 2:
            break
        time.sleep(5)
    if crashed:
        ENV_FAILURES["pandoc crashed"] += 1
        raise RuntimeError(f"pandoc crashed (exit code {done.returncode:#x})")
    if b"Heap exhausted" in done.stderr:
        raise RuntimeError(f"pandoc: over its {PANDOC_HEAP} heap limit")
    if done.returncode != 0:
        err = done.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError("pandoc: " + (err[-1][:150] if err else f"exit code {done.returncode}"))
    return tidy(done.stdout.decode("utf-8", "replace"))


# ---------------------------------------------------------------------------
# PDF: convert_to_markdown.convert_pdf in an isolated child process with a time limit
# ---------------------------------------------------------------------------
def check_pdf_route():
    """None if the PDF converter can run (checked as the conversions run), else the reason."""
    if importlib.util.find_spec("pymupdf4llm") is None:
        return "pymupdf4llm is not installed"
    try:
        done = subprocess.run([sys.executable, "-I", "-c", "import sys; sys.path.insert(0, sys.argv[1]); "
                               "import convert_to_markdown, pymupdf4llm", HERE], cwd=HERE, capture_output=True,
                              timeout=180)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"the converter could not be started ({type(exc).__name__})"
    if done.returncode != 0:
        err = done.stderr.decode("utf-8", "replace").strip().splitlines()
        return "convert_to_markdown.py does not import: " + (err[-1][:160] if err else f"exit {done.returncode}")
    return None


def _script(mark):
    def sub(m):
        inner = " ".join(re.sub(r"<[^>]*>", "", m.group(1)).split())
        return f"{mark}{inner}" if len(inner) <= 1 else f"{mark}{{{inner}}}" if inner else ""
    return sub


def pdf_tags(text):
    """HTML tags pymupdf4llm writes (<sup>, <sub>, <br>, <span>...) as plain text: x<sup>2</sup> -> x^2."""
    text = re.sub(r"<sup>(.*?)</sup>", _script("^"), text, flags=re.S | re.I)
    text = re.sub(r"<sub>(.*?)</sub>", _script("_"), text, flags=re.S | re.I)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</?(?:span|b|i|em|strong|u|s|small|mark|font|sup|sub)\b[^>]*>", "", text, flags=re.I)
    # table-of-contents lines ('Topic . . . . . 12') and the table rule left without its rows
    out = []
    for line in text.split("\n"):
        if re.search(r"(?:\.\s?){8,}", line):
            continue
        rule = re.fullmatch(r"\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-*:?\s*", line)
        if rule and not (out and out[-1].lstrip().startswith("|")):
            continue
        out.append(line)
    return "\n".join(out)


def pdf_markdown(ctx, data):
    """Markdown of a PDF (bytes). The PDF goes into a new temporary folder and is
    read by a `python -I` child process, stopped after --pdf-seconds."""
    if ctx.pdf_problem:
        ENV_FAILURES["no PDF converter"] += 1
        raise RuntimeError(f"no PDF converter ({ctx.pdf_problem})")
    if data[:5] != b"%PDF-":
        raise RuntimeError("not a PDF")
    with tempfile.TemporaryDirectory(prefix="docs_pdf_") as tmp:
        src, out = os.path.join(tmp, "doc.pdf"), os.path.join(tmp, "doc.md")
        with open(src, "wb") as handle:
            handle.write(data)
        try:
            done = subprocess.run([sys.executable, "-I", "-c", _PDF_CHILD, HERE, src, out], cwd=HERE,
                                  capture_output=True, timeout=ctx.args.pdf_seconds)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"PDF conversion took over {ctx.args.pdf_seconds:g} s (--pdf-seconds)") from None
        if done.returncode != 0 or not os.path.isfile(out):
            err = done.stderr.decode("utf-8", "replace").strip().splitlines()
            raise RuntimeError("PDF: " + (err[-1][:120] if err else f"converter exit code {done.returncode}"))
        with open(out, encoding="utf-8") as handle:
            return tidy_raw(pdf_tags(handle.read()))


# ---------------------------------------------------------------------------
# HTML -> main content -> Markdown
# ---------------------------------------------------------------------------
HTML_PARSER = "lxml" if importlib.util.find_spec("lxml") else "html.parser"

MAIN_SELECTORS = ["[itemprop=articleBody]", "[role=main]", ".rst-content", ".md-content", ".markdown-body",
                  ".entry-content", "article", "main", "#main-content", ".main-content", ".document", "#doc-content",
                  "#content", ".post-content", ".page-content", "#main", "#primary"]
# never content: dropped from the whole page before the main content is chosen
STRIP_TAGS = ["script", "style", "noscript", "template", "svg", "canvas", "iframe", "frame", "frameset", "object",
              "embed", "button", "input", "select", "textarea", "link", "meta", "img", "picture", "video", "audio",
              "source", "map", "nav"]
# <aside> is a sidebar, except these docutils/Sphinx content blocks
ASIDE_KEEP = {"footnote", "footnote-list", "topic", "admonition", "sidebar", "pull-quote", "epigraph", "highlights"}
# navigation, sidebars, anchors and widgets inside the main content
DROP_SELECTORS = ", ".join([
    "footer", "a.headerlink", ".headerlink", ".viewcode-link", ".viewcode-back", ".breadcrumb", ".breadcrumbs",
    ".wy-breadcrumbs", ".rst-footer-buttons", ".wy-nav-side", ".md-sidebar", ".md-source-file", ".md-footer",
    ".md-header", ".md-content__button", ".sphinxsidebar", ".related", ".bd-sidebar", ".bd-toc", ".toc-drawer",
    ".sidebar-drawer", ".prev-next-area", ".prev-next-footer", ".sphx-glr-download-link-note", ".sphx-glr-footer",
    ".copybutton", ".edit-this-page", "#searchbox", ".skip-link", ".screen-reader-text", ".sr-only",
    ".visually-hidden", ".sharedaddy", "#comments", ".comments-area", ".post-navigation", ".nav-links",
    ".entry-meta", ".entry-footer", "#TOC", ".author", ".date",
    "[hidden]", "[aria-hidden=true]", ".linenos", ".lineno", ".linenodiv",
    # MediaWiki: [edit] links, 'From ...' and 'Jump to navigation' lines, translation bars, navboxes,
    # 'Retrieved from ...' and the category list
    ".mw-editsection", "#siteSub", "#contentSub", "#contentSub2", ".mw-jump-link", "#jump-to-nav",
    ".mw-pt-languages", ".mw-pt-translate-header", ".navbox", ".printfooter", "#catlinks", ".catlinks",
    ".noprint",
    # Jupyter notebooks rendered by nbsphinx / myst-nb: '[1]:' prompts and stderr warnings
    ".nbinput .prompt", ".nboutput .prompt", ".output_area.stderr", ".nboutput .stderr", ".cell_output .stderr"])
# tables of contents inside the content: dropped when they are mostly links (a singlehtml build puts
# whole chapters inside .toctree-wrapper; MediaWiki's is #toc)
TOC_SELECTORS = ".toctree-wrapper, div.topic.contents, nav.contents, .contents.local, #toc, .toc"
# only when no main element was found: the page's own header, menus and sidebars
BODY_DROP_SELECTORS = "header, #header, .header, .site-header, #masthead, .navbar, #navbar, .menu, #menu, " \
                      "#sidebar, .sidebar, #toc, .toc, #footer, .footer"
KNOWN_LANGS = {"r", "python", "bash", "sh", "shell", "console", "c", "cpp", "c++", "java", "julia", "perl", "sql",
               "yaml", "json", "toml", "ini", "makefile", "dockerfile", "javascript", "js", "html", "xml", "latex",
               "tex", "matlab", "fortran", "rust", "go", "scala", "groovy", "nextflow", "wdl", "snakemake",
               "powershell", "bat", "awk", "diff", "ipython", "ipython3", "pycon", "rout", "text", "none"}
LANG_ALIASES = {"py": "python", "python3": "python", "ipython": "python", "ipython3": "python", "pycon": "python",
                "sh": "bash", "shell": "bash", "zsh": "bash", "shell-session": "console", "rout": "r", "splus": "r",
                "default": "", "none": "", "text": "", "plain": "", "plaintext": "", "nohighlight": "", "output": "",
                "c++": "cpp", "js": "javascript"}


def parse_html(data):
    return BeautifulSoup(data, HTML_PARSER)


def page_title(soup):
    title = " ".join((soup.title.get_text() if soup.title else "").split())
    first = re.split(r"\s+[|\u2014\u2013-]\s+", title)[0].strip()
    return first if len(first) >= 3 else title


def page_links(soup, base):
    """Absolute URLs of the page's links (read before navigation is removed: it holds the table of contents)."""
    tag = soup.find("base", href=True)
    if tag:
        try:
            base = urllib.parse.urljoin(base, tag["href"])
        except ValueError:
            pass
    out = []
    for el in soup.find_all(["a", "area", "frame", "iframe"]):
        href = el.get("href") or el.get("src")
        if href and not href.lstrip().lower().startswith(("javascript:", "mailto:", "data:", "tel:")):
            try:
                out.append(urllib.parse.urljoin(base, href.strip()))
            except ValueError:  # a malformed address: http://[your-server]:8080/, a bad port
                pass
    return out


def link_share(el):
    """Share of an element's text that is link text (navigation is nearly all links)."""
    total = visible_chars(el.get_text(" "))
    return sum(visible_chars(a.get_text(" ")) for a in el.find_all("a")) / total if total else 1.0


def robots_meta(soup):
    """The page's robots directives as a set ('none' stands for noindex and nofollow)."""
    tag = soup.find("meta", attrs={"name": re.compile(r"^(robots|tinygpt-docs)$", re.I)})
    flags = set(re.split(r"[\s,]+", (tag.get("content") or "").lower())) if tag else set()
    if "none" in flags:
        flags |= {"noindex", "nofollow"}
    return flags


def _code_lang(pre):
    classes = list(pre.get("class") or [])
    code = pre.find("code")
    if code is not None:
        classes += code.get("class") or []
    for k, anc in enumerate(pre.parents):
        if k >= 3 or anc is None or anc.name in ("body", "html", "[document]"):
            break
        classes += anc.get("class") or []
    for c in classes:
        c = c.lower()
        for prefix in ("language-", "lang-", "highlight-", "sourcecode-"):
            if c.startswith(prefix) and len(c) > len(prefix):
                c = c[len(prefix):]
                return LANG_ALIASES.get(c, c)
        if c in KNOWN_LANGS:
            return LANG_ALIASES.get(c, c)
    return ""


def _math_to_tex(soup):
    """MathJax / KaTeX markup -> \\( \\) and \\[ \\] text that pandoc reads as TeX math."""
    for el in soup.find_all("script", type=re.compile(r"^math/tex", re.I)):
        tex = el.get_text()
        display = "display" in (el.get("type") or "")
        el.replace_with(soup.new_string(f"\\[{tex}\\]" if display else f"\\({tex}\\)"))
    for el in soup.select(".katex"):
        ann = el.find("annotation", encoding="application/x-tex")
        if ann is not None:
            el.replace_with(soup.new_string(f"\\({ann.get_text()}\\)"))
    for el in soup.select(".MathJax, .MathJax_Preview, .MathJax_Display, mjx-container"):
        el.decompose()


def _fix_inline_code(main, soup):
    """Inline code as one text run: Sphinx writes each word as its own <span>,
    which pandoc would turn into `a`` ``b`. Inline code whose lines are set
    apart by <br> (an R session, one command per line) becomes a code block,
    except in a table cell (there the lines are joined by a space)."""
    for el in main.find_all(["code", "tt", "kbd", "samp"]):
        if el.find_parent("pre") is not None:
            continue
        breaks = el.find_all("br")
        block = bool(breaks) and el.find_parent(["td", "th"]) is None
        for br in breaks:
            br.replace_with("\ue000" if block else " ")
        if block:
            pre = soup.new_tag("pre")
            pre.string = "\n".join(" ".join(line.split()) for line in el.get_text().split("\ue000")).strip("\n")
            el.replace_with(pre)
        elif el.find(True) is not None:
            el.string = el.get_text()


def _fix_code(main, soup):
    """Each <pre> becomes <pre><code class="language-X">text</code></pre>: spans of
    syntax highlighting flattened (<br> kept as a line break), the language found
    from the classes around it."""
    for pre in main.find_all("pre"):
        if pre.find_parent("pre") is not None:
            continue
        for br in pre.find_all("br"):
            br.replace_with("\n")
        lang = _code_lang(pre)
        code = soup.new_tag("code")
        if lang:
            code["class"] = ["language-" + lang]
        code.string = pre.get_text()
        new = soup.new_tag("pre")
        new.append(code)
        pre.replace_with(new)


def _own_cells(table):
    return [cell for tr in table.find_all("tr") if tr.find_parent("table") is table
            for cell in tr.find_all(["td", "th"], recursive=False)]


def _fix_tables(main, soup):
    """Tables used for layout (holding other tables, headings or code, or long
    text) are unwrapped into their cells' content; Sphinx line-number tables
    become their code."""
    for table in main.find_all("table"):
        if table.parent is None:
            continue
        if "highlighttable" in (table.get("class") or []):
            pre = table.select_one("td.code pre") or table.find("pre")
            if pre is not None:
                table.replace_with(pre.extract())
            continue
        cells = _own_cells(table)
        layout = (any(c.find(["table", "h1", "h2", "h3", "h4", "h5", "h6", "pre"]) is not None for c in cells)
                  or any(len(c.get_text(" ", strip=True)) > 1500 for c in cells))
        if layout:
            div = soup.new_tag("div")
            for cell in cells:
                part = soup.new_tag("div")
                part.extend(list(cell.contents))
                div.append(part)
            table.replace_with(div)


def html_markdown(soup):
    """(title, Markdown) of a documentation page: its main content without
    navigation, converted by pandoc. The soup is changed."""
    title = page_title(soup)
    _math_to_tex(soup)
    for el in soup.find_all(STRIP_TAGS):
        if not el.decomposed:
            el.decompose()
    for el in soup.find_all("aside"):
        if not el.decomposed and not ASIDE_KEEP & set(el.get("class") or []):
            el.decompose()
    for el in soup.find_all("form"):  # search boxes; a form that wraps the whole page (ASP.NET) stays
        if not el.decomposed and visible_chars(el.get_text(" ")) < MIN_CHARS:
            el.decompose()
    main = None
    for selector in MAIN_SELECTORS:
        el = soup.select_one(selector)
        if el is not None and visible_chars(el.get_text(" ")) >= MIN_CHARS:
            main = el
            break
    if main is None:
        main = soup.body or soup
        for el in main.select(BODY_DROP_SELECTORS):
            if not el.decomposed:
                el.decompose()
    for el in main.select(DROP_SELECTORS):
        if not el.decomposed:
            el.decompose()
    for el in main.select(TOC_SELECTORS):
        if not el.decomposed and link_share(el) > 0.6:
            el.decompose()
    for el in main.find_all(["ul", "ol"]):  # a hand-made table of contents: a long list of links only
        if (not el.decomposed and el.find_parent(["ul", "ol"]) is None and len(el.find_all("li")) >= 5
                and link_share(el) > 0.8):
            head = el.find_previous_sibling()
            if head is not None and head.name in ("h1", "h2", "h3", "h4", "h5", "h6", "p", "b", "strong") and \
                    re.fullmatch(r"\W*(table of )?contents|on this page\W*", head.get_text(" ", strip=True), re.I):
                head.decompose()
            el.decompose()
    share = link_share(main)
    if share > NAV_SHARE and visible_chars(main.get_text(" ")) >= MIN_CHARS:
        raise NavigationPage(f"navigation page ({share:.0%} link text)")
    _fix_inline_code(main, soup)
    _fix_code(main, soup)
    _fix_tables(main, soup)
    md = pandoc_md(str(main), HTML_FROM)
    first = re.search(r"^(#{1,6}) +(.+)$", md, re.M)
    if title and not (first and (len(first.group(1)) == 1 or " ".join(first.group(2).split()).lower() ==
                                 title.lower())):
        md = f"# {title}\n\n{md}"
    return title, md


# ---------------------------------------------------------------------------
# Rd (R help pages) -> Markdown
# ---------------------------------------------------------------------------
_RD_PLAIN = re.compile(r"[^\\%{}]+")
_RD_NAME = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_RD_RAW = {"eqn", "deqn", "Sexpr", "newcommand", "renewcommand", "out"}  # arguments kept as source text
_RD_DROP = {"name", "alias", "docType", "keyword", "concept", "encoding", "Rdversion", "RdOpts", "Sexpr",
            "newcommand", "renewcommand", "dontshow", "testonly", "figure", "author", "references", "title",
            "concept", "keyword", "md"}
_RD_SECTIONS = {"description": "Description", "usage": "Usage", "format": "Format", "arguments": "Arguments",
                "details": "Details", "value": "Value", "note": "Note", "source": "Source", "seealso": "See also",
                "examples": "Examples", "slot": "Slots"}
_RD_CODE_SECTIONS = {"usage", "examples"}


def _rd_raw(text, i):
    """Source text of the brace group whose '{' is at text[i - 1], and the index after its '}'."""
    depth, j = 1, i
    while j < len(text):
        c = text[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[i:j], j + 1
        j += 1
    return text[i:], len(text)


def _rd_parse(text, i=0, nested=False):
    """Nodes of Rd source from text[i] to the closing '}' (nested) or the end:
    strings and (macro, [arguments], [option]) tuples; % comments dropped."""
    nodes, buf = [], []

    def flush():
        if buf:
            nodes.append("".join(buf))
            buf.clear()

    while i < len(text):
        m = _RD_PLAIN.match(text, i)
        if m:
            buf.append(m.group(0))
            i = m.end()
            continue
        c = text[i]
        if c == "\\":
            nxt = text[i + 1:i + 2]
            if nxt in ("\\", "%", "{", "}"):
                buf.append(nxt)
                i += 2
                continue
            m = _RD_NAME.match(text, i + 1)
            if not m:
                buf.append("\\")
                i += 1
                continue
            name, i, opt = m.group(0), m.end(), None
            if text.startswith("[", i):
                j = text.find("]", i)
                if 0 < j - i < 200:
                    opt, i = text[i + 1:j], j + 1
            args = []
            while text.startswith("{", i):
                if name in _RD_RAW:
                    arg, i = _rd_raw(text, i + 1)
                else:
                    arg, i = _rd_parse(text, i + 1, nested=True)
                args.append(arg)
            flush()
            nodes.append((name, args, opt))
        elif c == "%":
            j = text.find("\n", i)
            i = len(text) if j < 0 else j
        elif c == "{":
            arg, i = _rd_parse(text, i + 1, nested=True)
            flush()
            nodes.append(("", [arg], None))
        else:  # '}'
            if nested:
                flush()
                return nodes, i + 1
            i += 1  # a stray '}'
    flush()
    return nodes, i


def _rd_list(nodes, kind, mode):
    """\\itemize / \\enumerate / \\describe (and \\arguments, kind 'arguments') as a Markdown list."""
    items, cur = [], None
    for node in nodes:
        if isinstance(node, tuple) and node[0] == "item":
            if cur is not None:
                items.append(cur)
            args = node[1]
            label = _rd_render(args[0], "text").strip() if args and isinstance(args[0], list) else None
            cur = [label, list(args[1]) if len(args) > 1 and isinstance(args[1], list) else []]
        elif cur is not None:
            cur[1].append(node)
    if cur is not None:
        items.append(cur)
    lines = []
    for k, (label, body) in enumerate(items, 1):
        text = " ".join(_rd_render(body, "text").split())
        if label is not None and kind in ("describe", "arguments"):
            label = " ".join(label.split())
            label = f"`{label.strip('`')}`" if kind == "arguments" else f"**{label}**"
            lines.append(f"- {label}: {text}" if text else f"- {label}")
        elif text or label:
            text = " ".join(filter(None, [label, text]))
            lines.append(f"{k}. {text}" if kind == "enumerate" else f"- {text}")
    return "\n\n" + "\n".join(lines) + "\n\n" if lines else ""


def _rd_table(nodes):
    text = _rd_render(nodes, "table")
    rows = [[" ".join(cell.split()).replace("|", "\\|") for cell in row.split("\x1f")] for row in text.split("\x1e")]
    rows = [r for r in rows if any(r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width] + ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n\n" + "\n".join(out) + "\n\n"


def _rd_render(nodes, mode="text"):
    """Text of Rd nodes. mode: 'text' (Markdown), 'code' (R code: no markup) or
    'table' (\\tab and \\cr become cell and row markers)."""
    if isinstance(nodes, str):
        return nodes
    out = []
    for node in nodes:
        if isinstance(node, str):
            out.append(node)
            continue
        name, args, _ = node

        def arg(k, m=mode):
            if k >= len(args):
                return ""
            return args[k] if isinstance(args[k], str) else _rd_render(args[k], m)

        if name == "":  # a bare {...}: grouping in text, literal braces in R code (function bodies, loops)
            out.append("{" + arg(0) + "}" if mode == "code" else arg(0))
        elif name in ("code", "verb", "samp", "kbd", "env", "option", "command", "file"):
            inner = arg(0, "code")
            out.append(inner if mode == "code" else f"`{' '.join(inner.split())}`")
        elif name in ("emph", "bold", "strong"):
            mark = "*" if name == "emph" else "**"
            inner = " ".join(arg(0).split())
            out.append(inner if mode == "code" or not inner else f"{mark}{inner}{mark}")
        elif name == "sQuote":
            out.append(f"'{arg(0)}'")
        elif name == "dQuote":
            out.append(f'"{arg(0)}"')
        elif name == "href":
            out.append(arg(1))
        elif name == "email":
            out.append("[email]")
        elif name == "eqn":
            tex = " ".join(arg(0).split())
            out.append(tex if mode == "code" else f"${tex}$")
        elif name == "deqn":
            tex = " ".join(arg(0).split())
            out.append(tex if mode == "code" else f"\n\n$${tex}$$\n\n")
        elif name in ("dots", "ldots"):
            out.append("...")
        elif name == "R":
            out.append("R")
        elif name == "cr":
            out.append("\x1e" if mode == "table" else "\n")
        elif name == "tab":
            out.append("\x1f" if mode == "table" else "  ")
        elif name in ("itemize", "enumerate", "describe"):
            out.append(_rd_list(args[0] if args and isinstance(args[0], list) else [], name, mode))
        elif name == "tabular":
            out.append(_rd_table(args[1]) if len(args) > 1 and isinstance(args[1], list) else "")
        elif name == "preformatted":
            out.append("\n\n```\n" + textwrap.dedent(arg(0, "code")).strip("\n") + "\n```\n\n")
        elif name == "item":  # outside a list
            out.append("\n- " + ": ".join(filter(None, (" ".join(arg(k).split()) for k in range(len(args))))))
        elif name == "if":
            out.append("" if "html" in arg(0) else arg(1))
        elif name == "ifelse":
            out.append(arg(2))
        elif name in ("dontrun", "donttest"):
            out.append(arg(0))
        elif name in _RD_DROP:
            continue
        else:  # \link, \pkg, \var, \acronym, \method, \S4method, \enc, \url, unknown macros: their first argument
            out.append(arg(0))
    return "".join(out)


def _rd_flow(text):
    """Text-section lines without their indentation (four spaces would make a code block); fences kept."""
    out, fence = [], False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            fence = not fence
            out.append(line.strip())
        else:
            out.append(line.rstrip() if fence else line.strip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def rd_markdown(text):
    """Markdown of one Rd help page: title, description, usage and examples as R
    code, arguments as a list, the other sections as text (authors and
    references left out)."""
    nodes, _ = _rd_parse(text)
    title, parts = "", []
    for node in nodes:
        if isinstance(node, str):
            continue
        name, args, _ = node
        if name == "title" and args:
            title = " ".join(_rd_render(args[0]).split())
        elif name in _RD_SECTIONS and args:
            body = args[0] if isinstance(args[0], list) else [args[0]]
            if name in _RD_CODE_SECTIONS:
                code = textwrap.dedent(_rd_render(body, "code")).strip("\n")
                if code.strip():
                    parts.append(f"## {_RD_SECTIONS[name]}\n\n" + ctm._fence(code, "r"))
            elif name == "arguments":
                text_ = _rd_list(body, "arguments", "text").strip()
                if text_:
                    parts.append(f"## Arguments\n\n{text_}")
            else:
                text_ = _rd_flow(_rd_render(body))
                if text_:
                    parts.append(f"## {_RD_SECTIONS[name]}\n\n{text_}")
        elif name in ("section", "subsection") and len(args) >= 2:
            head = " ".join(_rd_render(args[0]).split())
            text_ = _rd_flow(_rd_render(args[1]))
            if text_:
                parts.append(f"{'##' if name == 'section' else '###'} {head}\n\n{text_}")
    if not parts:
        return ""
    return tidy_raw((f"# {title}\n\n" if title else "") + "\n\n".join(parts))


def notebook_markdown(data):
    """A Jupyter notebook as Markdown: Markdown cells as text, code cells as fenced code, outputs dropped."""
    nb = json.loads(decode_text(data))
    if not isinstance(nb, dict):
        raise ValueError("not a notebook")
    meta = nb.get("metadata") or {}
    lang = ((meta.get("kernelspec") or {}).get("language") or (meta.get("language_info") or {}).get("name")
            or "python")
    cells = nb.get("cells")
    if cells is None:  # nbformat 3
        cells = [c for ws in nb.get("worksheets") or [] for c in (ws.get("cells") or [])]
    parts = []
    for cell in cells or []:
        if not isinstance(cell, dict):
            continue
        source = cell.get("source", cell.get("input", ""))
        source = "".join(source) if isinstance(source, list) else str(source)
        if not source.strip():
            continue
        if cell.get("cell_type") in ("markdown", "heading"):
            parts.append(source.strip())
        elif cell.get("cell_type") == "code":
            parts.append(ctm._fence(source, str(lang).lower()))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# output: one hidden folder per row, moved into place when the row is complete
# ---------------------------------------------------------------------------
_DOC_EXT_RE = re.compile(r"\.(html?|xhtml|php|aspx?|shtml|jsp|md|markdown|rst|txt|rmd|ipynb|rd|pdf|tex)$", re.I)


def file_stem(rel):
    """A short, safe file name (no extension) from a page or file path:
    'docs/guide/options.md' -> 'docs__guide__options'."""
    rel = urllib.parse.unquote(rel.split("?")[0].split("#")[0])
    rel = _DOC_EXT_RE.sub("", rel)
    parts = [p for p in re.split(r"[\\/]+", rel) if p not in ("", ".", "..")]
    parts = [re.sub(r"[^A-Za-z0-9._-]+", "_", p).strip("._-") for p in parts]
    stem = "__".join(p for p in parts if p) or "index"
    if len(stem) > 100:
        stem = stem[:90].rstrip("._-") + "_" + hashlib.sha1(rel.encode("utf-8")).hexdigest()[:8]
    if stem.split(".")[0].upper() in WINDOWS_RESERVED:
        stem += "_"
    return stem


class RowWriter:
    """The documents of one sources row, written to <out>/<tool>/.<label>.part
    (hidden: data_prep skips it) and moved to <out>/<tool>/<label> by commit()."""

    def __init__(self, out, tool, label, source_url):
        self.tool_dir = os.path.join(out, tool)
        self.label, self.source = label, source_url
        self.final = os.path.join(self.tool_dir, label)
        self.stage = os.path.join(self.tool_dir, f".{label}.part")
        self.old = os.path.join(self.tool_dir, f".{label}.old")  # the earlier folder while commit() swaps
        shutil.rmtree(self.stage, ignore_errors=True)
        os.makedirs(self.stage)
        self.names, self.hashes, self.files = set(), set(), []
        self.dropped = collections.Counter()
        self.incomplete = []  # why the row is not complete (a server or the network kept failing): retried

    def add(self, rel, url, text, ext=".md"):
        """Write one document (if it has MIN_CHARS characters and is not a duplicate); True if written."""
        if not text or visible_chars(text) < MIN_CHARS:
            self.dropped["short"] += 1
            return False
        digest = hashlib.sha1(" ".join(text.split()).encode("utf-8")).hexdigest()
        if digest in self.hashes:
            self.dropped["duplicate"] += 1
            return False
        self.hashes.add(digest)
        stem, n = file_stem(rel), 1
        room = MAX_PATH_CHARS - len(os.path.join(self.stage, "")) - len(ext) - len("_999.part")
        if len(stem) > room:  # the whole path stays under Windows' 260 characters
            stem = stem[:max(room - 9, 1)].rstrip("._-") + "_" + hashlib.sha1(stem.encode("utf-8")).hexdigest()[:8]
        name = stem + ext
        while name.lower() in self.names:
            n += 1
            name = f"{stem}_{n}{ext}"
        self.names.add(name.lower())
        path = os.path.join(self.stage, name)
        with open(path + ".part", "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(path + ".part", path)
        self.files.append((name, url, len(text)))
        return True

    def discard(self):
        shutil.rmtree(self.stage, ignore_errors=True)
        try:
            os.rmdir(self.tool_dir)  # only if nothing else is in it
        except OSError:
            pass

    def commit(self):
        """Replace the row's folder with the new one and update <tool>/_sources.tsv; (files, characters).
        A run that found no files leaves an earlier run's folder alone (RowError). The earlier folder
        is set aside until the new one is in place, and put back if that fails."""
        had = os.path.isdir(self.final)
        if not self.files:
            shutil.rmtree(self.stage, ignore_errors=True)
            if had:
                raise RowError("no files this time; the files of the earlier run are kept")
        else:
            shutil.rmtree(self.old, ignore_errors=True)
            if had:
                os.replace(self.final, self.old)
            try:
                os.replace(self.stage, self.final)
            except BaseException:
                if had and not os.path.exists(self.final):
                    os.replace(self.old, self.final)
                raise
        index = os.path.join(self.tool_dir, "_sources.tsv")
        rows = []
        if os.path.isfile(index):
            with open(index, encoding="utf-8") as handle:
                head = handle.readline().rstrip("\n").split("\t")
                rows = [dict(zip(head, line.rstrip("\n").split("\t"))) for line in handle if line.strip()]
        rows = [r for r in rows if not (r.get("file") or "").startswith(self.label + "/")]
        rows += [{"file": f"{self.label}/{name}", "url": url, "source": self.source, "chars": str(chars)}
                 for name, url, chars in self.files]
        if rows:
            with open(index + ".part", "w", encoding="utf-8", newline="\n") as handle:
                handle.write("\t".join(TOOL_SOURCES_COLUMNS) + "\n")
                for r in rows:
                    handle.write("\t".join(_flat(r.get(c, "")) for c in TOOL_SOURCES_COLUMNS) + "\n")
            os.replace(index + ".part", index)
        elif os.path.isfile(index):
            os.remove(index)
        shutil.rmtree(self.old, ignore_errors=True)
        return len(self.files), sum(c for _, _, c in self.files)


# ---------------------------------------------------------------------------
# site / page: breadth-first crawl under the URL's path prefix
# ---------------------------------------------------------------------------
_PAGE_EXT = {"", "html", "htm", "xhtml", "shtml", "php", "asp", "aspx", "jsp", "cgi", "pdf"}
_SKIP_URL_RE = re.compile(
    r"/(_sources|_static|_images|_downloads|_modules|_raw|genindex|search|py-modindex|modindex|wp-admin|wp-json|"
    r"wp-login|wp-includes|wp-content|feed|xmlrpc\.php|tag|category|author|page/\d+)(/|\.html?$|$)|"
    r"[?&](replytocom|share|print|format|action|do|rev|diff|oldid|redirect|utm_[a-z]+)=|"
    # licence, author and release-note pages (pkgdown's LICENSE-text.html, authors.html, news/), as for github rows
    r"/(licen[cs]e(-text)?|authors|news|change_?log|changes|history|release[-_]?notes|whatsnew)"
    r"(/|/index\.html?|\.html?)?$", re.I)


def normalise_url(url, scheme_for=None):
    """URL without fragment, with a lower-case host and '/' for an empty path;
    None for anything but http(s). scheme_for=(host, scheme): links to that host
    use that scheme (http and https copies of a site are one site)."""
    try:
        parts = urllib.parse.urlsplit(url.strip())
        port = parts.port
    except ValueError:  # a bad port (http://localhost:PORT/) or IPv6 address
        return None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return None
    host = parts.hostname.lower()
    netloc = host if port in (None, 80, 443) else f"{host}:{port}"
    scheme = scheme_for[1] if scheme_for and host == scheme_for[0] else parts.scheme
    return urllib.parse.urlunsplit((scheme, netloc, parts.path or "/", parts.query, ""))


def _page_ext(path):
    m = re.search(r"\.([A-Za-z][A-Za-z0-9]{0,5})$", path.rsplit("/", 1)[-1])
    return m.group(1).lower() if m else ""


class Scope:
    """Same host and port, under one path prefix (URLs from normalise_url)."""

    def __init__(self, url):
        parts = urllib.parse.urlsplit(url)
        self.host, self.port, self.scheme = parts.hostname, parts.port, parts.scheme
        path = parts.path or "/"
        self.prefix = path if path.endswith("/") else path.rsplit("/", 1)[0] + "/"

    def __contains__(self, url):
        parts = urllib.parse.urlsplit(url)
        path = parts.path or "/"
        return (parts.hostname == self.host and parts.port == self.port
                and (path.startswith(self.prefix) or path + "/" == self.prefix))

    def encloses(self, other):
        """True if this scope is a folder above `other` on the same site (www. ignored)."""
        def site(s):
            return (s.host or "").removeprefix("www."), s.port
        return site(self) == site(other) and other.prefix.startswith(self.prefix) and other.prefix != self.prefix

    def rel(self, url):
        parts = urllib.parse.urlsplit(url)
        rel = (parts.path or "/")[len(self.prefix):] if (parts.path or "/").startswith(self.prefix) else parts.path
        if parts.query:
            rel += "_" + parts.query
        return rel.strip("/") or "index"


def transient(exc):
    """True for a failure worth trying again later: the network, or a server that is busy or down (429, 5xx)."""
    return isinstance(exc, requests.RequestException) or (isinstance(exc, HttpError) and
                                                           (exc.code == 429 or exc.code >= 500))


_HTML_START_RE = re.compile(r"\s*(?:<\?xml[^>]*>\s*)?(?:<!--.*?-->\s*)*<(?:!doctype\s+html|html|head|body)\b",
                            re.I | re.S)


def crawl(ctx, start, writer, single=False):
    """Fetch pages breadth-first from `start` (one page if single) and write
    their Markdown (plain-text pages as they are); returns notes for the
    manifest. A crawl that writes nothing raises RowError; one cut short by a
    server or a network that kept failing marks the row incomplete."""
    http, notes, failures = ctx.http, [], collections.Counter()
    url = normalise_url(start)
    if not url:
        raise RowError(f"not an http(s) URL: {start}")
    start = url
    if not http.allowed(start):
        raise RowError(http.refusal(start))
    scope = Scope(start)
    queue, seen = collections.deque([start]), {start}
    pages = requests_made = over = bad = streak = 0  # over: downloads past --max-mb; bad, streak: transient failures
    max_pages = 1 if single else ctx.args.max_pages
    while queue:
        if pages >= max_pages:
            if not single:
                notes.append(f"stopped at --max-pages {max_pages}")
            break
        if ctx.left() <= 0:
            notes.append(f"stopped at --max-mb {ctx.args.max_mb:g}")
            break
        if requests_made >= 3 * max_pages + 20:
            notes.append("stopped: too many non-page links")
            break
        if streak >= 5:
            writer.incomplete.append(f"stopped after {streak} failed requests in a row")
            break
        url = queue.popleft()
        try:  # anything that goes wrong with one URL is counted, and the crawl goes on
            if not http.allowed(url):
                failures["robots.txt"] += 1
                continue
            requests_made += 1
            limit = min(ctx.left() if url.lower().endswith(".pdf") else MAX_PAGE, ctx.left())
            try:
                got = http.get(url, limit, want={"html", "pdf", "text"})
            except TooLarge:
                if limit < ctx.left():  # one page over MAX_PAGE
                    failures["TooLarge"] += 1
                    continue
                over += 1  # more than the row may still download
                if over >= 3 or ctx.left() < MAX_PAGE:
                    notes.append(f"stopped at --max-mb {ctx.args.max_mb:g}")
                    break
                failures["over --max-mb"] += 1
                continue
            streak = 0
            if got is None:
                failures["404"] += 1
                continue
            final = normalise_url(got.url, (scope.host, scope.scheme)) or url
            if final != url:
                if requests_made == 1:  # the start page moved (/docs -> /docs/en/latest/, another host)
                    moved = Scope(final)
                    if final not in scope and moved.encloses(scope):  # /manual/ -> / : not a crawl of the whole site
                        raise RowError(f"the start page redirects to {final}, a folder above the row's URL (the "
                                       "manual moved or removed?); check the row")
                    if not http.allowed(final):
                        raise RowError(http.refusal(final))
                    if final not in scope or moved.prefix.startswith(scope.prefix):  # /docs/ -> /docs keeps /docs/
                        scope = moved
                elif final not in scope or final in seen:
                    continue
                seen.add(final)
            if got.body is None:
                failures[f"not a page ({got.ctype or 'no type'})"] += 1
                continue
            if got.kind == "pdf":
                pages += 1
                try:
                    writer.add(scope.rel(final), final, pdf_markdown(ctx, got.body))
                except RuntimeError as exc:
                    failures[str(exc)[:60]] += 1
                continue
            if got.kind == "text":
                text = decode_text(got.body)
                if not (_HTML_START_RE.match(text[:5000]) or
                        _page_ext(urllib.parse.urlsplit(final).path) in ("html", "htm", "xhtml")):
                    pages += 1  # plain text (a README): kept as it is if it is the row's page or reads as prose
                    if requests_made == 1 or looks_like_prose(text):
                        writer.add(scope.rel(final), final, tidy_raw(text), ".txt")
                    else:
                        writer.dropped["not prose"] += 1
                    continue
            soup = parse_html(got.body)  # HTML, also when served as text/plain
            flags = robots_meta(soup)
            if not single and "nofollow" not in flags:
                for link in page_links(soup, got.url):
                    link = normalise_url(link, (scope.host, scope.scheme))
                    if (link and link not in seen and link in scope and _page_ext(urllib.parse.urlsplit(link).path)
                            in _PAGE_EXT and not _SKIP_URL_RE.search(link)):
                        seen.add(link)
                        queue.append(link)
            if "noindex" in flags:
                continue
            pages += 1
            try:
                _, md = html_markdown(soup)
            except NavigationPage:
                writer.dropped["navigation"] += 1
                continue
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                failures[f"conversion: {str(exc)[:50]}"] += 1
                continue
            writer.add(scope.rel(final), final, md)
        except RowError:
            raise
        except Exception as exc:  # noqa: BLE001 - one URL fails, not the row
            if transient(exc):
                bad += 1
                streak += 1
            failures[str(exc)[:40] if isinstance(exc, HttpError) else type(exc).__name__] += 1
            continue
        if pages % 25 == 0:
            print(f"  {pages} pages, {len(writer.files)} files, {len(queue)} links waiting", flush=True)
    if bad * 10 > requests_made and not writer.incomplete:
        writer.incomplete.append(f"{bad} of {requests_made} requests failed (server or network)")
    if failures:
        notes.append("skipped: " + ", ".join(f"{k} x{v}" for k, v in failures.most_common(4)))
    if not writer.files:
        why = [f"{k} x{v}" for k, v in failures.most_common(3)] + [f"{k} x{v}" for k, v in writer.dropped.items()]
        raise RowError(("no page had documentation text" if pages else "no page could be fetched")
                       + (f" ({', '.join(why)})" if why else ""))
    return notes


def do_site(ctx, row, writer):
    return crawl(ctx, row["url"], writer)


def do_page(ctx, row, writer):
    return crawl(ctx, row["url"], writer, single=True)


# ---------------------------------------------------------------------------
# zip archives: read in memory, member by member, never extracted
# ---------------------------------------------------------------------------
def safe_member(name):
    """The member's path inside the archive, or None for an absolute path, a
    drive letter, '..' or a NUL byte (names that could leave the folder)."""
    name = name.replace("\\", "/")
    if "\x00" in name or name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        return None
    parts = [p for p in name.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        return None
    return "/".join(parts)


class ZipReader:
    """Members of a zip file read into memory with size limits; symbolic links,
    encrypted and unsafe names are left out. strip_top removes the archive's one
    top folder (repo-main/...)."""

    def __init__(self, path, budget, strip_top=True):
        self.zf = zipfile.ZipFile(path)
        self.budget, self.members = budget, {}
        infos = []
        for info in self.zf.infolist():
            name = safe_member(info.filename)
            if name is None or info.is_dir() or info.flag_bits & 0x1:
                continue
            if (info.external_attr >> 16) & 0o170000 == 0o120000:  # a symbolic link
                continue
            infos.append((name, info))
        tops = {n.split("/", 1)[0] for n, _ in infos}
        strip = strip_top and len(tops) == 1 and all("/" in n for n, _ in infos)
        for name, info in infos:
            self.members[name.split("/", 1)[1] if strip else name] = info

    def entries(self):
        return [(name, info.file_size) for name, info in self.members.items()]

    def read(self, name, limit):
        info = self.members[name]
        if info.file_size > limit:
            raise TooLarge(f"{name} is larger than {limit >> 20} MB")
        with self.zf.open(info) as handle:
            data = handle.read(limit + 1)
        if len(data) > limit:
            raise TooLarge(f"{name} is larger than its stated size")
        self.budget -= len(data)
        if self.budget < 0:
            raise TooLarge("the archive unpacks to more than --max-mb")
        return data

    def close(self):
        self.zf.close()


# ---------------------------------------------------------------------------
# github
# ---------------------------------------------------------------------------
_SKIP_DIRS = {"node_modules", "third_party", "thirdparty", "third-party", "3rdparty", "vendor", "vendored",
              "external", "externals", "extern", "external_libs", "ext", "deps", "dependencies", "submodules",
              "site-packages", "__pycache__", "build", "dist", "_build", "venv", "htmlcov", "test", "tests",
              "testthat", "testdata", "test_data", "_snaps", "lib", "libs", "contrib", "eigen", "boost", "htslib",
              "zlib", "zstd", "libdeflate", "pybind11", "googletest", "gtest", "catch2", "cmake",
              # website themes and assets: templates, not documentation
              "theme", "themes", "_templates", "templates", "overrides", "_static", "static", "assets", "css",
              "js", "fonts", "img", "images", "_includes", "_layouts", "_sass", "_site"}
_DOC_DIR_RE = re.compile(r"^(docs?|documentation|man|manuals?|vignettes?|wiki|guides?|tutorials?|examples?|demos?|"
                         r"help|user[_-]?guide|walkthroughs?|notebooks?|pages)$", re.I)
_EXAMPLE_DIR_RE = re.compile(r"^(examples?|tutorials?|demos?|walkthroughs?|vignettes?|notebooks?)([_-].*)?$", re.I)
_EXAMPLE_NAME_RE = re.compile(r"(example|tutorial|demo|quick_?start|walkthrough)", re.I)
_SKIP_NAME_RE = re.compile(
    r"(licen[cs]e|copying|copyright|notice|change_?log|changes|news|history|authors|contributors|contributing|"
    r"code[_-]of[_-]conduct|security|codeowners|citation|funding|maintainers|release[_-]?(log|notes?)|"
    r"requirements|cmakelists|pull_request_template|issue_template|support)([._-].*)?", re.I)
_DOC_NAME_RE = re.compile(r"(readme|manual|usage|tutorial|guide|howto|how_to|faq|install|instructions|help|intro)",
                          re.I)
_GENERATOR_RE = re.compile(r"(^|/)(_?pkgdown\.ya?ml|mkdocs\.ya?ml|_config\.ya?ml|docusaurus\.config\.[jt]s|book\.toml|"
                           r"_quarto\.ya?ml|_bookdown\.ya?ml|_output\.ya?ml)$|(^|/)(docs?|documentation)/(source/)?"
                           r"conf\.py$")
SCRIPT_LANG = {".sh": "bash", ".bash": "bash", ".r": "r", ".py": "python", ".wdl": "wdl", ".nf": "nextflow",
               ".smk": "snakemake"}


def github_parts(url):
    """(owner, repo, ref or None, subfolder or '') of a GitHub URL."""
    parts = urllib.parse.urlsplit(url)
    if (parts.hostname or "").lower() not in ("github.com", "www.github.com"):
        raise RowError("a github row needs https://github.com/OWNER/REPO")
    segs = [s for s in parts.path.split("/") if s]
    if len(segs) < 2:
        raise RowError("a github row needs https://github.com/OWNER/REPO")
    owner, repo = segs[0], re.sub(r"\.git$", "", segs[1])
    ref, sub = None, ""
    if len(segs) >= 4 and segs[2] in ("tree", "blob"):
        ref, sub = segs[3], "/".join(segs[4:])
    if not re.fullmatch(r"[\w.-]+", owner) or not re.fullmatch(r"[\w.-]+", repo):
        raise RowError(f"unexpected GitHub owner/repository: {owner}/{repo}")
    return owner, repo, ref, sub


def repo_action(rel, size, html_ok):
    """How to treat one repository file ('md', 'rst', 'rmd', 'txt', 'rd',
    'ipynb', 'html', 'tex', 'pdf', 'script') or None to skip it."""
    parts = rel.split("/")
    dirs, name = [p.lower() for p in parts[:-1]], parts[-1]
    low = name.lower()
    stem, ext = os.path.splitext(low)
    if name.startswith(".") or any(d.startswith(".") or d in _SKIP_DIRS for d in dirs):
        return None
    if _SKIP_NAME_RE.fullmatch(stem) or _SKIP_NAME_RE.fullmatch(low):
        return None
    in_doc = any(_DOC_DIR_RE.match(d) for d in dirs)
    in_example = any(_EXAMPLE_DIR_RE.match(d) for d in dirs)
    if ext == ".pdf":
        return "pdf" if in_doc and size <= MAX_REPO_PDF else None
    if size > MAX_FILE or size == 0:
        return None
    if low.startswith("readme"):
        return {".md": "md", ".markdown": "md", ".rst": "rst", ".rmd": "rmd", ".txt": "txt", "": "txt"}.get(ext)
    if ext in (".md", ".markdown"):
        return "md"
    if ext in (".rst", ".rmd", ".ipynb", ".rd"):
        return ext[1:]
    if ext == ".txt":
        return "txt" if in_doc or in_example or _DOC_NAME_RE.search(stem) else None
    if ext in (".html", ".htm"):
        return "html" if in_doc and html_ok else None
    if ext == ".tex":
        return "tex" if in_doc else None
    if (ext in SCRIPT_LANG or low == "snakefile") and (in_example or (_EXAMPLE_NAME_RE.search(stem) and
                                                                       (not dirs or in_doc))):
        return "script"
    return None


def repo_document(ctx, rel, data, action):
    """(text, extension) of one repository file, or (None, reason)."""
    if action == "pdf":
        return pdf_markdown(ctx, data), ".md"
    if b"\x00" in data[:8000]:
        return None, "binary"
    text = decode_text(data)
    if action == "md":
        return pandoc_md(text, "gfm"), ".md"
    if action == "rst":
        return pandoc_md(text, "rst"), ".md"
    if action == "tex":
        return pandoc_md(text, "latex"), ".md"
    if action == "ipynb":
        return pandoc_md(notebook_markdown(data), "gfm"), ".md"
    if action == "rd":
        return rd_markdown(text), ".md"
    if action == "rmd":
        return clean_rmd(text), ".Rmd"
    if action == "txt":
        if not os.path.basename(rel).lower().startswith("readme") and not looks_like_prose(text):
            return None, "not prose"
        return tidy_raw(text), ".txt"
    if action == "html":
        if re.search(r"\{%|\{\{", text[:20000]):  # a website template (Jinja/Liquid), not a page
            return None, "template"
        try:
            return html_markdown(parse_html(data))[1], ".md"
        except NavigationPage:
            return None, "navigation"
    if action == "script":
        ext = os.path.splitext(rel.lower())[1]
        lang = SCRIPT_LANG.get(ext, "snakemake")
        return tidy_raw(f"# {rel}\n\n{ctm._fence(text, lang)}"), ".md"
    return None, "unknown"


def github_api_files(ctx, owner, repo, ref):
    """(entries, reader, notes) from the GitHub API's file list and raw.githubusercontent.com."""
    api = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {"Accept": "application/vnd.github+json"}
    got = ctx.http.get(api, 5 << 20, headers=headers)
    if got is None:
        raise RowError(f"repository not found: {owner}/{repo}")
    branch = ref or json.loads(got.body).get("default_branch") or "HEAD"
    got = ctx.http.get(f"{api}/git/trees/{urllib.parse.quote(branch, safe='')}?recursive=1", 100 << 20,
                       headers=headers)
    if got is None:
        raise RowError(f"no file list for {owner}/{repo} at {branch}")
    tree = json.loads(got.body)
    notes = ["GitHub's file list was truncated"] if tree.get("truncated") else []
    entries = [(safe_member(t.get("path") or ""), int(t.get("size") or 0)) for t in tree.get("tree", [])
               if t.get("type") == "blob"]
    entries = [(p, s) for p, s in entries if p]

    def reader(rel, limit):
        url = (f"https://raw.githubusercontent.com/{owner}/{repo}/{urllib.parse.quote(branch, safe='')}/"
               f"{urllib.parse.quote(rel)}")
        got = ctx.http.get(url, min(limit, max(ctx.left(), 0)))
        if got is None or got.body is None:
            raise RuntimeError("missing on raw.githubusercontent.com")
        return got.body

    return entries, reader, branch, notes


def do_github(ctx, row, writer):
    owner, repo, ref, sub = github_parts(row["url"])
    notes, failures = [], collections.Counter()
    with tempfile.TemporaryDirectory(prefix="docs_repo_") as tmp:
        zpath = os.path.join(tmp, "repo.zip")
        archive = f"https://codeload.github.com/{owner}/{repo}/zip/{urllib.parse.quote(ref or 'HEAD', safe='')}"
        try:
            got = ctx.http.get(archive, ctx.left(), to_file=zpath, timeout=120)
        except TooLarge:
            got = "too_large"
        if got is None:
            raise RowError(f"repository archive not found (404): {owner}/{repo}")
        zr, fallback = None, got == "too_large"
        if fallback:
            print(f"  the archive is larger than --max-mb {ctx.args.max_mb:g}; fetching documentation files one by "
                  "one", flush=True)
            entries, reader, ref_used, more = github_api_files(ctx, owner, repo, ref)
            notes += ["archive over --max-mb: files fetched one by one"] + more
        else:
            if not zipfile.is_zipfile(zpath):
                raise RowError("the repository archive is not a zip file")
            zr = ZipReader(zpath, ctx.args.max_mb * (1 << 20))
            entries, reader, ref_used = zr.entries(), zr.read, ref or "HEAD"
        try:
            names = {rel.lower() for rel, _ in entries}
            # a docs generator's config: its HTML output (pkgdown, mkdocs, Sphinx, Jekyll ...) duplicates sources
            # that are taken anyway
            html_ok = not any(_GENERATOR_RE.search(n) for n in names)
            theme_dirs = {n.rsplit("/", 1)[0] + "/" for n in names if n.endswith(("mkdocs_theme.yml", "theme.toml",
                                                                                   "_config.yml"))}
            chosen = []
            for rel, size in entries:
                if sub and not rel.startswith(sub.rstrip("/") + "/"):
                    continue
                if any(rel.lower().startswith(d) for d in theme_dirs if d != "/") and rel.lower().endswith(
                        (".html", ".htm")):
                    continue
                action = repo_action(rel, size, html_ok)
                if action:
                    chosen.append((rel, size, action))
            # one copy of each document: its source rather than a rendered copy in the same folder
            # (doc/manual.tex, not doc/manual.pdf; README.Rmd, not the README.md knitted from it)
            rank = {"tex": 0, "rmd": 0, "rst": 0, "md": 1, "html": 2, "pdf": 2}
            best = {}
            for rel, _, action in chosen:
                key = os.path.splitext(rel.lower())[0]
                if action in rank and (key not in best or rank[action] < rank[best[key]]):
                    best[key] = action
            kept = [c for c in chosen if c[2] not in rank or best[os.path.splitext(c[0].lower())[0]] == c[2]]
            if len(kept) < len(chosen):
                writer.dropped["rendered copy"] += len(chosen) - len(kept)
            chosen = kept
            chosen.sort(key=lambda c: (not c[0].lower().startswith("readme"), c[0].count("/"), c[0].lower()))
            if fallback and len(chosen) > ctx.args.max_pages:
                notes.append(f"first {ctx.args.max_pages} of {len(chosen)} files (--max-pages)")
                chosen = chosen[:ctx.args.max_pages]
            print(f"  {len(chosen)} documentation files of {len(entries)} in the repository", flush=True)
            bad = streak = 0  # files that failed for the network or a busy server (fallback only), in a row
            for k, (rel, size, action) in enumerate(chosen, 1):
                if fallback:
                    if ctx.left() <= 0:
                        notes.append(f"stopped at --max-mb {ctx.args.max_mb:g}")
                        break
                    if streak >= 5:
                        writer.incomplete.append(f"stopped after {streak} failed downloads in a row")
                        break
                    if size > ctx.left():  # the file list gives its size: no request that cannot succeed
                        failures["over --max-mb"] += 1
                        continue
                try:
                    data = reader(rel, MAX_REPO_PDF if action == "pdf" else MAX_FILE)
                    text, ext = repo_document(ctx, rel, data, action)
                except TooLarge as exc:
                    if "--max-mb" in str(exc):
                        notes.append(f"stopped at --max-mb {ctx.args.max_mb:g}")
                        break
                    failures["too large"] += 1
                    continue
                except Exception as exc:  # noqa: BLE001 - one file fails, not the row
                    if transient(exc):
                        bad += 1
                        streak += 1
                    failures[f"{action}: {str(exc)[:50] or type(exc).__name__}"] += 1
                    continue
                streak = 0
                if text is None:
                    writer.dropped[ext] += 1  # ext is the reason here: not prose, binary, template
                else:
                    url = f"https://github.com/{owner}/{repo}/blob/{ref_used}/{urllib.parse.quote(rel)}"
                    writer.add(rel, url, text, ext)
                if k % 100 == 0:
                    print(f"  {k}/{len(chosen)} files, {len(writer.files)} written", flush=True)
            if bad * 10 > len(chosen) and not writer.incomplete:
                writer.incomplete.append(f"{bad} files could not be downloaded (server or network)")
        finally:
            if zr is not None:
                zr.close()
    if failures:
        notes.append("skipped: " + ", ".join(f"{k} x{v}" for k, v in failures.most_common(3)))
    return notes


# ---------------------------------------------------------------------------
# rtd: the HTML zip, else a crawl
# ---------------------------------------------------------------------------
_RTD_SKIP_RE = re.compile(r"(^|/)(genindex|search|py-modindex|modindex)\.html?$|(^|/)(_modules|_sources|_static|"
                          r"_images|_downloads)/", re.I)


def html_zip_pages(ctx, zpath, writer, base_url):
    zr = ZipReader(zpath, ctx.args.max_mb * (1 << 20))
    try:
        pages = [rel for rel, _ in zr.entries() if rel.lower().endswith((".html", ".htm")) and not
                 _RTD_SKIP_RE.search(rel)]
        pages.sort(key=lambda r: (r.lower() != "index.html", r.count("/"), r.lower()))
        notes = []
        if len(pages) > ctx.args.max_pages:
            notes.append(f"first {ctx.args.max_pages} of {len(pages)} pages (--max-pages)")
            pages = pages[:ctx.args.max_pages]
        failures = collections.Counter()
        for rel in pages:
            try:
                _, md = html_markdown(parse_html(zr.read(rel, MAX_PAGE)))
            except NavigationPage:
                writer.dropped["navigation"] += 1
                continue
            except (TooLarge, RuntimeError, subprocess.TimeoutExpired) as exc:
                failures[type(exc).__name__] += 1
                continue
            writer.add(rel, urllib.parse.urljoin(base_url, rel), md)
        if failures:
            notes.append("skipped: " + ", ".join(f"{k} x{v}" for k, v in failures.items()))
        return len(pages), notes
    finally:
        zr.close()


def do_rtd(ctx, row, writer):
    url = normalise_url(row["url"])
    if not url:
        raise RowError("not an http(s) URL")
    parts = urllib.parse.urlsplit(url)
    m = re.match(r"^/([a-z]{2}(?:[-_][A-Za-z]{2,4})?)/([^/]+)/", parts.path)
    if not m:  # e.g. https://project.readthedocs.io/ -> /en/latest/
        if not ctx.http.allowed(url):
            raise RowError(ctx.http.refusal(url))
        got = ctx.http.get(url, MAX_PAGE, want={"html"})
        if got is not None:
            url = normalise_url(got.url) or url
            parts = urllib.parse.urlsplit(url)
            m = re.match(r"^/([a-z]{2}(?:[-_][A-Za-z]{2,4})?)/([^/]+)/", parts.path)
    lang, version = (m.group(1), m.group(2)) if m else ("en", "latest")
    root = f"{parts.scheme}://{parts.netloc}/{lang}/{version}/"
    zip_url = f"{parts.scheme}://{parts.netloc}/_/downloads/{lang}/{version}/htmlzip/"
    reason = None
    if ctx.http.allowed(zip_url):
        with tempfile.TemporaryDirectory(prefix="docs_rtd_") as tmp:
            zpath = os.path.join(tmp, "docs.zip")
            try:
                got = ctx.http.get(zip_url, ctx.left(), want={"zip"}, to_file=zpath, timeout=120)
                if got is None:
                    reason = "no HTML zip"
                elif got.body is None or not zipfile.is_zipfile(zpath):
                    reason = "the HTML zip link gave no zip file"
                else:
                    print(f"  HTML zip: {os.path.getsize(zpath) / (1 << 20):.1f} MB", flush=True)
                    pages, notes = html_zip_pages(ctx, zpath, writer, root)
                    if writer.files:
                        return ["from the HTML zip"] + notes
                    reason = f"the HTML zip held {pages} pages but no usable text"
            except TooLarge:
                reason = "HTML zip over --max-mb"
            except (HttpError, zipfile.BadZipFile) as exc:
                reason = f"HTML zip: {exc}"
    else:
        reason = "robots.txt disallows the HTML zip"
    print(f"  {reason}; crawling {root}", flush=True)
    return [f"{reason}: crawled"] + crawl(ctx, root, writer)


# ---------------------------------------------------------------------------
# pdf, cran, bioc
# ---------------------------------------------------------------------------
def fetch_document(ctx, url, writer, rel, kinds=("html", "pdf")):
    """Fetch one HTML page or PDF (robots.txt obeyed) and write its Markdown; '' or why it was not written."""
    if not ctx.http.allowed(url):
        return "robots.txt"
    got = ctx.http.get(url, min(MAX_PAGE if url.lower().endswith((".html", ".htm")) else ctx.left(), ctx.left()),
                       want=set(kinds))
    if got is None:
        return "404"
    if got.body is None:
        return f"not {'/'.join(kinds)} ({got.ctype or 'no type'})"
    md = pdf_markdown(ctx, got.body) if got.kind == "pdf" else html_markdown(parse_html(got.body))[1]
    return "" if writer.add(rel, got.url, md) else "too short or duplicate"


def do_pdf(ctx, row, writer):
    url = row["url"]
    if not ctx.http.allowed(url):
        raise RowError(ctx.http.refusal(url))
    try:
        got = ctx.http.get(url, ctx.left(), want={"pdf"}, timeout=120)
    except TooLarge:
        raise RowError(f"the PDF is larger than --max-mb {ctx.args.max_mb:g}") from None
    if got is None:
        raise RowError("PDF not found (404)")
    if got.body is None or got.kind != "pdf":
        raise RowError(f"not a PDF ({got.ctype or 'no content type'})")
    name = urllib.parse.unquote(urllib.parse.urlsplit(got.url).path.rsplit("/", 1)[-1]) or "manual"
    if not writer.add(name, got.url, pdf_markdown(ctx, got.body)):
        raise RowError("the PDF has no usable text (a scan?)")
    return []


def _package_docs(ctx, writer, links, refman, readme=None):
    """Fetch the reference manual, the README and the vignettes (HTML preferred
    to PDF for each vignette); notes for the manifest."""
    vignettes = collections.OrderedDict()
    for link in links:
        path = urllib.parse.urlsplit(link).path
        stem, ext = os.path.splitext(path.rsplit("/", 1)[-1])
        if ext.lower() in (".html", ".htm", ".pdf"):
            current = vignettes.get(stem)
            if current is None or (current.lower().endswith(".pdf") and ext.lower() != ".pdf"):
                vignettes[stem] = link
    todo = [("reference_manual", u) for u in refman[:1]] + ([("README", readme)] if readme else []) + \
           [(f"vignette_{stem}", u) for stem, u in vignettes.items()]
    failures, done = collections.Counter(), 0
    for rel, url in todo:
        if ctx.left() <= 0:
            return [f"stopped at --max-mb {ctx.args.max_mb:g}"]
        try:
            why = fetch_document(ctx, url, writer, rel)
        except (HttpError, TooLarge, RuntimeError, requests.RequestException, subprocess.SubprocessError) as exc:
            why = str(exc)[:60]
            if transient(exc):  # the server or the network: the row is fetched again by the next run
                writer.incomplete.append(f"{rel}: {why}")
        if why:
            failures[f"{rel}: {why}"] += 1
        else:
            done += 1
    if not todo:
        raise RowError("no reference manual or vignettes found on the package page")
    notes = [f"{len(vignettes)} vignettes"]
    if failures:
        notes.append("skipped: " + ", ".join(k for k in list(failures)[:4]))
    if not done:
        raise RowError("nothing could be fetched (" + "; ".join(list(failures)[:3]) + ")")
    return notes


def do_cran(ctx, row, writer):
    parts = urllib.parse.urlsplit(row["url"])
    m = re.search(r"package=([A-Za-z][A-Za-z0-9.]*)", parts.path + "?" + parts.query) or \
        re.search(r"/packages/([A-Za-z][A-Za-z0-9.]*)", parts.path)
    if not m:
        raise RowError("a cran row needs https://cran.r-project.org/package=NAME")
    name = m.group(1)
    base = f"{parts.scheme or 'https'}://{parts.netloc}/web/packages/{name}/"
    index = base + "index.html"
    if not ctx.http.allowed(index):
        raise RowError(ctx.http.refusal(index))
    got = ctx.http.get(index, MAX_PAGE, want={"html"})
    if got is None or not got.body:
        raise RowError(f"CRAN has no package {name} (archived or misspelt?)")
    links = [normalise_url(u) for u in page_links(parse_html(got.body), got.url)]
    links = [u for u in links if u and u.startswith(base)]
    refman = [u for u in links if u.endswith(f"/refman/{name}.html")] or [base + f"{name}.pdf"]
    readme = next((u for u in links if "/readme/" in u.lower() and u.lower().endswith(".html")), None)
    vignettes = [u for u in links if "/vignettes/" in u]
    return _package_docs(ctx, writer, vignettes, refman, readme)


def do_bioc(ctx, row, writer):
    parts = urllib.parse.urlsplit(row["url"])
    segs = [s for s in parts.path.split("/") if s]
    name = re.sub(r"\.html?$", "", segs[-1]) if segs else ""
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9.]*", name or ""):
        raise RowError("a bioc row needs https://bioconductor.org/packages/NAME")
    url = normalise_url(row["url"])
    if not ctx.http.allowed(url):
        raise RowError(ctx.http.refusal(url))
    got = ctx.http.get(url, MAX_PAGE, want={"html"})
    if got is None or not got.body:
        raise RowError(f"Bioconductor has no package {name}")
    if f"/{name}.html" not in urllib.parse.urlsplit(got.url).path:
        raise RowError(f"the package page moved to {got.url} (removed from Bioconductor?)")
    links = [normalise_url(u) for u in page_links(parse_html(got.body), got.url)]
    if "/packages/release/" in got.url:
        # the release page links its files under the release's number (/packages/3.23/...), which robots.txt
        # disallows; the same files are served, and allowed, under /packages/release/
        links = [re.sub(r"/packages/\d+\.\d+/", "/packages/release/", u) if u else u for u in links]
    links = list(dict.fromkeys(u for u in links if u))
    vignettes = [u for u in links if f"/vignettes/{name}/inst/doc/" in u]
    refman = [u for u in links if f"/manuals/{name}/man/" in u and u.lower().endswith(".pdf")]
    return _package_docs(ctx, writer, vignettes, refman)


HANDLERS = {"github": do_github, "site": do_site, "rtd": do_rtd, "pdf": do_pdf, "page": do_page, "cran": do_cran,
            "bioc": do_bioc}


# ---------------------------------------------------------------------------
# sources table, manifest, main
# ---------------------------------------------------------------------------
def read_sources(path):
    """Rows of the sources table (checked); exits with a message on errors."""
    if not os.path.isfile(path):
        raise SystemExit(f"Sources table not found: {path}")
    rows, problems, header, seen = [], [], None, set()
    with open(path, encoding="utf-8-sig") as handle:
        for n, line in enumerate(handle, 1):
            line = line.rstrip("\r\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            cells = [c.strip() for c in line.split("\t")]
            if header is None:
                if [c.lower() for c in cells[:5]] != SOURCE_COLUMNS:
                    raise SystemExit(f"{path}: the header must be: {chr(9).join(SOURCE_COLUMNS)} (tab-separated)")
                header = cells
                continue
            row = dict(zip(SOURCE_COLUMNS, cells + [""] * len(SOURCE_COLUMNS)))
            row["kind"] = row["kind"].lower()
            row["licence"] = row["licence"] or "unknown"
            if not TOOL_RE.fullmatch(row["tool"]):
                problems.append(f"line {n}: tool {row['tool']!r} must be lowercase letters, digits, '-' or '_'")
            elif row["kind"] not in KINDS:
                problems.append(f"line {n}: kind {row['kind']!r} is not one of {', '.join(KINDS)}")
            elif not re.match(r"^https?://[^\s/]+", row["url"]):
                problems.append(f"line {n}: url {row['url']!r} is not an http(s) URL")
            elif (row["tool"], row["kind"], row["url"]) in seen:
                print(f"Note: line {n} repeats an earlier row; it is ignored", flush=True)
            else:
                seen.add((row["tool"], row["kind"], row["url"]))
                rows.append(row)
    if header is None:
        raise SystemExit(f"{path}: no header line")
    if problems:
        raise SystemExit(f"{path}:\n  " + "\n  ".join(problems))
    return rows


def _row_label(row):
    """A row's folder name before name clashes: kind and a short name from the URL."""
    parts = urllib.parse.urlsplit(row["url"])
    host = (parts.hostname or "").removeprefix("www.")
    segs = [s for s in parts.path.split("/") if s]
    kind = row["kind"]
    if kind == "github":
        base = "_".join(segs[:2] + (segs[3:] if len(segs) > 3 and segs[2] in ("tree", "blob") else []))
    elif kind == "cran":
        m = re.search(r"package=([\w.]+)", parts.path + "?" + parts.query) or re.search(r"/packages/([\w.]+)",
                                                                                      parts.path)
        base = m.group(1) if m else host
    elif kind == "bioc":
        base = re.sub(r"\.html?$", "", segs[-1]) if segs else host
    elif kind == "rtd":
        base = "_".join([host.split(".")[0] if host.endswith((".readthedocs.io", ".readthedocs.org")) else host]
                        + segs[1:2])
    elif kind == "pdf":
        base = os.path.splitext(urllib.parse.unquote(segs[-1]))[0] if segs else host
    else:
        base = "_".join([host] + [_DOC_EXT_RE.sub("", s) for s in segs])
    label = f"{kind}_" + (re.sub(r"[^A-Za-z0-9.-]+", "_", base).strip("._-") or "doc")
    if len(label) > 60:
        label = label[:51].rstrip("._-") + "_" + _url_digest(row["url"])
    return label


def _url_digest(url):
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]


def row_labels(rows, manifest):
    """{(tool, kind, url): folder name of the row inside its tool folder}: kind
    and a short name from the URL, e.g. github_rgcgithub_regenie. A row that ran
    before keeps its folder (the label in the manifest); a new row whose name
    another row of its tool already has gets the URL's digest appended. So adding,
    moving or deleting rows never gives one row's folder to another."""
    def key(row):
        return row["tool"], row["kind"], row["url"]

    def first_come(rows, used):
        labels = {}
        for row in rows:
            label = _row_label(row)
            if label.lower() in used[row["tool"]]:
                label = f"{label}_{_url_digest(row['url'])}"
            used[row["tool"]].add(label.lower())
            labels[key(row)] = label
        return labels

    # a manifest written before labels were kept: the labels that run gave (first come, in table order)
    earlier = first_come(rows, collections.defaultdict(set))
    labels, used = {}, collections.defaultdict(set)
    for row in rows:
        entry = manifest.get(key(row))
        if entry:
            labels[key(row)] = entry.get("label") or earlier[key(row)]
            used[row["tool"]].add(labels[key(row)].lower())
    labels.update(first_come([row for row in rows if key(row) not in labels], used))
    return labels


def read_manifest(path):
    rows = collections.OrderedDict()
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            head = handle.readline().rstrip("\n").split("\t")
            for line in handle:
                if line.strip():
                    r = dict(zip(head, line.rstrip("\n").split("\t")))
                    rows[(r.get("tool"), r.get("kind"), r.get("url"))] = r
    return rows


def write_manifest(path, rows):
    with open(path + ".part", "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(MANIFEST_COLUMNS) + "\n")
        for r in rows.values():
            handle.write("\t".join(_flat(r.get(c, "")) for c in MANIFEST_COLUMNS) + "\n")
    os.replace(path + ".part", path)


def finished(entry):
    return bool(entry) and (entry.get("status") or "").split(" ")[0] in FINISHED


def clean_leftovers(out):
    """Row folders and files left by an interrupted run (.part); a row folder that commit() had set
    aside (.<label>.old) goes back into place if the new one never arrived."""
    if not os.path.isdir(out):
        return
    for tool in os.listdir(out):
        folder = os.path.join(out, tool)
        if not os.path.isdir(folder):
            if tool.endswith(".part"):
                os.remove(folder)
            continue
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if name.endswith(".part"):
                shutil.rmtree(path, ignore_errors=True) if os.path.isdir(path) else os.remove(path)
            elif name.startswith(".") and name.endswith(".old") and os.path.isdir(path):
                final = os.path.join(folder, name[1:-len(".old")])
                if os.path.isdir(final):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.replace(path, final)


_HELD = []  # open lock files: the locks go when the process ends


def lock_out(out):
    """Hold <out>/_manifest.lock for this run, so that a second run on the same folder stops
    instead of overwriting this run's manifest and deleting its work. The lock goes with the
    process, also after a crash."""
    handle = open(os.path.join(out, "_manifest.lock"), "a+")
    try:
        if sys.platform == "win32":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise SystemExit(f"Another download_docs.py run is using {out}; wait for it to end, or use another --out.")
    _HELD.append(handle)


def run_row(ctx, row, label):
    """Fetch one sources row into its folder; the manifest entry."""
    writer = RowWriter(ctx.args.out, row["tool"], label, row["url"])
    ctx.budget_end = ctx.http.bytes + ctx.args.max_mb * (1 << 20)
    env_before = ENV_FAILURES.copy()
    try:
        notes = HANDLERS[row["kind"]](ctx, row, writer) or []
        env = ENV_FAILURES - env_before
        if env:
            writer.incomplete.append(", ".join(f"{k} x{v}" for k, v in env.items()))
        if writer.incomplete and not writer.files:
            raise RowError("; ".join(writer.incomplete + notes))
        dropped = ", ".join(f"{v} {k}" for k, v in writer.dropped.items())
        if dropped:
            notes.append(f"left out: {dropped}")
        files, chars = writer.commit()
        status = "ok" if files else "empty"
        if writer.incomplete:  # not finished: the next run fetches the row again
            status = "partial"
            notes.insert(0, "fetched again by the next run: " + "; ".join(writer.incomplete))
    except BaseException:
        writer.discard()
        raise
    if notes:
        status += " (" + "; ".join(notes) + ")"
    return status, files, chars


def print_list(rows, manifest, labels):
    print(f"{'tool':<16} {'kind':<6} {'files':>6} {'chars':>12}  {'status':<10} url")
    for row in rows:
        entry = manifest.get((row["tool"], row["kind"], row["url"])) or {}
        status = (entry.get("status") or "new").split(" ")[0].rstrip(":")
        files = entry.get("files") or "-"
        chars = f"{int(entry['chars']):,}" if (entry.get("chars") or "").isdigit() else "-"
        print(f"{row['tool']:<16} {row['kind']:<6} {files:>6} {chars:>12}  {status:<10} {row['url']}")
        if status in ("error", "partial"):
            print(f"{'':<16} {entry.get('status', '')[:160]}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", default=os.path.join(HERE, "doc_sources.tsv"),
                   help="sources table (default doc_sources.tsv next to this script)")
    p.add_argument("--out", default=os.path.join("~", "ai_training_data", "software_doc"),
                   help="output folder (default %%USERPROFILE%%\\ai_training_data\\software_doc)")
    p.add_argument("--only", help="TOOL[,TOOL]: only these tools' rows")
    p.add_argument("--refresh", action="store_true", help="download finished rows again")
    p.add_argument("--max-pages", type=int, default=400, help="pages per site/rtd row (default 400)")
    p.add_argument("--wait", default="1-2", help="random pause between two requests to one host, MIN-MAX seconds "
                                                 "(default 1-2)")
    p.add_argument("--max-mb", type=float, default=300, help="megabytes downloaded per row (default 300)")
    p.add_argument("--pdf-seconds", type=float, default=PDF_SECONDS,
                   help=f"time limit of one PDF conversion (default {PDF_SECONDS})")
    p.add_argument("--list", action="store_true", help="print the rows and their status; download nothing")
    args = p.parse_args(argv)
    if args.max_pages < 1:
        p.error("--max-pages must be >= 1")
    if args.max_mb <= 0 or args.pdf_seconds <= 0:
        p.error("--max-mb and --pdf-seconds must be > 0")
    try:
        low, _, high = args.wait.partition("-")
        args.wait = (float(low), float(high or low))
    except ValueError:
        p.error("--wait must be MIN-MAX seconds, e.g. 1-2")
    if not 0 <= args.wait[0] <= args.wait[1] <= 600:
        p.error("--wait must be MIN-MAX with 0 <= MIN <= MAX <= 600")
    # %USERPROFILE% / $HOME / ~ are expanded here too: PowerShell leaves %VAR% as it is
    args.out = os.path.abspath(os.path.expanduser(os.path.expandvars(args.out)))
    args.sources = os.path.abspath(os.path.expanduser(os.path.expandvars(args.sources)))
    args.only = {t.strip() for t in (args.only or "").split(",") if t.strip()}
    return args


class Context:
    def __init__(self, args, http, pdf_problem):
        self.args, self.http, self.pdf_problem, self.budget_end = args, http, pdf_problem, 0

    def left(self):
        """Bytes this row may still download (--max-mb)."""
        return int(self.budget_end - self.http.bytes)


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    args = parse_args(argv)
    rows = read_sources(args.sources)
    if args.only:
        unknown = args.only - {r["tool"] for r in rows}
        if unknown:
            raise SystemExit(f"--only: no rows for {', '.join(sorted(unknown))} in {args.sources}")
        rows = [r for r in rows if r["tool"] in args.only]
    manifest_path = os.path.join(args.out, "_manifest.tsv")
    if args.list:
        manifest = read_manifest(manifest_path)
        print_list(rows, manifest, row_labels(rows, manifest))
        return
    PANDOC["path"] = ax.pandoc_path()
    if not PANDOC["path"]:
        raise SystemExit("pandoc is needed (pypandoc_binary, or pandoc on PATH)")
    os.makedirs(args.out, exist_ok=True)
    lock_out(args.out)
    manifest = read_manifest(manifest_path)
    labels = row_labels(rows, manifest)
    for key, entry in manifest.items():  # labels are kept from now on, also for rows finished earlier
        if key in labels and not entry.get("label"):
            entry["label"] = labels[key]
    clean_leftovers(args.out)
    ax._lower_priority()
    todo = [r for r in rows if args.refresh or not finished(manifest.get((r["tool"], r["kind"], r["url"])))]
    print(f"{len(rows)} rows in {args.sources}; {len(rows) - len(todo)} finished, {len(todo)} to fetch into "
          f"{args.out}", flush=True)
    if not todo:
        return
    pdf_problem = check_pdf_route()
    if pdf_problem:
        print(f"Warning: the PDF converter cannot run: {pdf_problem}. PDFs are skipped.", flush=True)
    ctx = Context(args, Http(args.wait), pdf_problem)
    counts = collections.Counter()
    with tempfile.TemporaryDirectory(prefix="docs_lua_") as tmp:
        PANDOC["lua"] = os.path.join(tmp, "plain_gfm.lua")
        with open(PANDOC["lua"], "w", encoding="utf-8") as handle:
            handle.write(LUA_FILTER)
        try:
            for n, row in enumerate(todo, 1):
                key = (row["tool"], row["kind"], row["url"])
                print(f"[{n}/{len(todo)}] {row['tool']}: {row['kind']} {row['url']}", flush=True)
                t0 = time.time()
                try:
                    status, files, chars = run_row(ctx, row, labels[key])
                except RowError as exc:
                    status, files, chars = f"error: {exc}", 0, 0
                except Exception as exc:  # noqa: BLE001 - the row fails, the run goes on
                    status, files, chars = f"error: {type(exc).__name__}: {str(exc)[:150]}", 0, 0
                earlier = manifest.get(key) or {}
                if status.startswith("error") and os.path.isdir(os.path.join(args.out, row["tool"], labels[key])):
                    # the earlier run's files are still in place
                    files, chars = (int(earlier[c]) if (earlier.get(c) or "").isdigit() else 0
                                    for c in ("files", "chars"))
                manifest[key] = {"tool": row["tool"], "kind": row["kind"], "url": row["url"],
                                 "licence": row["licence"], "status": status, "files": str(files),
                                 "chars": str(chars), "date": datetime.datetime.now().isoformat(timespec="minutes"),
                                 "label": labels[key]}
                write_manifest(manifest_path, manifest)
                counts[status.split(" ")[0].rstrip(":")] += 1
                print(f"  {status[:300]}: {files} files, {chars:,} characters ({time.time() - t0:,.0f} s)",
                      flush=True)
        except KeyboardInterrupt:
            print("\nStopped; rerun the same command to continue.")
            raise SystemExit(130)
    print(f"Done: {dict(counts)}; manifest {manifest_path}")


if __name__ == "__main__":
    main()
