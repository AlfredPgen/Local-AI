r"""Download open-access full-text papers from Europe PMC (PubMed Central) and
convert them to Markdown for data_prep.py.

1. list     Europe PMC search (open access, full text, MEDLINE-indexed), sorted
            (default: most cited first), up to --max-papers papers; saved in
            <meta>\list.tsv, so a rerun continues where it stopped.
2. fetch    the papers' JATS XML (the publisher's structured full text) from
            NCBI PubMed Central, 100 papers per request (NCBI E-utilities efetch;
            at most 3 requests per second, as NCBI asks without an API key),
            with retries.
3. convert  title, abstract and body to Markdown: section headings, paragraphs,
            lists, equations as LaTeX (from MathML or TeX), tables as Markdown
            tables, figure and table captions. References, acknowledgements,
            funding, competing interests, author contributions, abbreviations
            and supplementary files are left out. Papers whose XML has no body
            (abstract only) are skipped and listed.

Outputs: <out>\PMC<id>__<title>.md and <meta>\report.tsv (one row per paper:
status, characters, journal, year, citations, licence). The XML is not kept.
Existing outputs are skipped, so the script can be stopped (Ctrl+C) and rerun.
The XML is parsed as data only (no entities are fetched, nothing is executed).
NCBI asks for large jobs to run at weekends or between 9 pm and 5 am US Eastern
time (2 am to 10 am UK time).

Example (the 100,000 most-cited open-access genetics/genomics papers):
    python download_pmc.py --out %USERPROFILE%\ai_training_data\pmc --max-papers 100000
"""

import argparse
import concurrent.futures as cf
import gzip
import http.client
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
DEFAULT_QUERY = 'OPEN_ACCESS:y AND HAS_FT:y AND SRC:MED AND (genetic* OR genom* OR allele* OR "gene expression")'
USER_AGENT = "tinyGPT-pmc-downloader/1.0 (personal research; python urllib)"
MAX_XML_BYTES = 400 << 20
MIN_REQUEST_GAP = 0.35  # seconds between request starts: NCBI allows 3 per second without an API key
FINAL = {"ok", "no_fulltext", "no_body", "not_english", "bad_xml"}  # statuses not retried on a rerun
REPORT_COLUMNS = ["pmcid", "status", "chars", "journal", "year", "cited", "licence", "file"]


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
_RATE_LOCK = threading.Lock()
_LAST_START = [0.0]


def _wait_turn():
    with _RATE_LOCK:
        pause = _LAST_START[0] + MIN_REQUEST_GAP - time.monotonic()
        if pause > 0:
            time.sleep(pause)
        _LAST_START[0] = time.monotonic()


def http_get(url, data=None, timeout=300, tries=6):
    """Bytes of url (POST when data is given), None on 404. Rate limits, server
    errors and network failures are retried with exponential backoff."""
    delay = 3.0
    for attempt in range(tries):
        _wait_turn()
        try:
            request = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT,
                                                                      "Accept-Encoding": "gzip"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read(MAX_XML_BYTES + 1)
                if len(body) > MAX_XML_BYTES:
                    raise ValueError(f"response larger than {MAX_XML_BYTES >> 20} MB")
                if response.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
                return body
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException, OSError):
            if attempt == tries - 1:
                raise
        time.sleep(delay + random.random())
        delay = min(delay * 2, 120)
    return None


# ---------------------------------------------------------------------------
# 1. list
# ---------------------------------------------------------------------------
def build_list(query, sort, limit, path):
    """[(pmcid, pmid, year, cited, journal, title)] of the first `limit` search
    hits with a PMCID, cached in `path` (reused when query and sort match)."""
    header = [f"# query: {query}", f"# sort: {sort}"]
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        if lines[:2] == header:
            rows = [tuple(line.split("\t")) for line in lines[3:] if line]
            if len(rows) >= limit or (len(lines) > 2 and lines[2].endswith("(all search hits)")):
                print(f"Paper list: {min(len(rows), limit):,} papers from {path}")
                return rows[:limit]
    rows, seen, cursor, t0, exhausted = [], set(), "*", time.time(), False
    while len(rows) < limit:
        params = {"query": query, "format": "json", "resultType": "lite", "pageSize": 1000,
                  "cursorMark": cursor, "sort": sort}
        data = json.loads(http_get(SEARCH_URL + "?" + urllib.parse.urlencode(params)))
        results = data.get("resultList", {}).get("result", [])
        for r in results:
            pmcid = r.get("pmcid")
            if pmcid and pmcid not in seen:
                seen.add(pmcid)
                rows.append((pmcid, str(r.get("pmid") or ""), str(r.get("pubYear") or ""),
                             str(r.get("citedByCount") or 0), _flat(r.get("journalTitle")), _flat(r.get("title"))))
        nxt = data.get("nextCursorMark")
        if len(rows) % 10_000 < 1000:
            print(f"  listed {len(rows):,} of {min(limit, data.get('hitCount', limit)):,} papers "
                  f"({time.time() - t0:,.0f} s)", flush=True)
        if not results or not nxt or nxt == cursor:
            exhausted = True
            break
        cursor = nxt
    rows = rows[:limit]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".part", "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(header) + f"\n# complete: {len(rows)} papers"
                     + (" (all search hits)" if exhausted else "") + "\n")
        for row in rows:
            handle.write("\t".join(row) + "\n")
    os.replace(path + ".part", path)
    print(f"Paper list: {len(rows):,} papers saved to {path}")
    return rows


def _flat(value):
    return " ".join(str(value or "").replace("\t", " ").split())


# ---------------------------------------------------------------------------
# 3. convert: JATS XML -> Markdown
# ---------------------------------------------------------------------------
def _tag(el):
    return el.tag.rsplit("}", 1)[-1] if isinstance(el.tag, str) else ""


def _clean(text):
    return " ".join(text.split())


SKIP_SECTION_RE = re.compile(
    r"^\W*(competing interests?|conflicts? of interests?|declaration of (competing )?interests?|declarations?|"
    r"(author|authors|author's|authors'|author’s|authors’) contributions?|contributions|contributors|"
    r"funding( information| statement| sources?)?|financial (support|disclosure)|acknowledge?ments?|"
    r"data (and code )?availability( statement)?|availability of data( and materials?)?|"
    r"code availability|ethics( approval| statement| declarations?)?( and consent( to participate)?)?|"
    r"consent( for publication)?|abbreviations|supplementary (material|materials|information|data)|"
    r"additional (files?|information)|electronic supplementary material|publisher'?’?s note|references|"
    r"footnotes|disclosures?( forms?)?|disclaimer|open access|reporting summary|contributor information|"
    r"online content|additional data( files)?|peer review( information)?|"
    r"(author|authors|authors'|authors’)( information| details| affiliations)|correspondence|orcid)\b",
    re.I)
SKIP_SEC_TYPES = {"supplementary-material", "COI-statement", "funding-information", "ethics-statement",
                  "data-availability", "author-contributions", "acknowledgment", "abbreviations"}
SKIP_BLOCKS = {"supplementary-material", "graphic", "media", "ref-list", "fn-group", "ack", "notes", "sec-meta",
               "table-wrap-foot", "attrib", "permissions", "object-id", "alt-text", "label", "title", "kwd-group"}
BLOCKS_IN_P = {"list", "disp-formula", "table-wrap", "fig", "disp-quote", "boxed-text", "def-list", "preformat",
               "code", "statement", "fig-group", "table-wrap-group", "graphic", "media", "supplementary-material",
               "chem-struct-wrap", "array"}
MAX_TABLE_ROWS = 60
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")


class Jats:
    def __init__(self, article):
        self.root = article  # an <article> element

    # ---------------- metadata ----------------
    def meta(self):
        r = self.root
        title_el = r.find(".//article-meta/title-group/article-title")
        journal = r.find(".//journal-meta//journal-title")
        year = r.find(".//article-meta/pub-date/year")
        return {"title": _clean(self.inline(title_el)) if title_el is not None else "",
                "journal": _clean("".join(journal.itertext())) if journal is not None else "",
                "year": _clean(year.text or "") if year is not None else "",
                "lang": r.get("{http://www.w3.org/XML/1998/namespace}lang", "en"),
                "licence": self.licence()}

    def licence(self):
        for lic in self.root.iterfind(".//article-meta/permissions/license"):
            href = lic.get("{http://www.w3.org/1999/xlink}href") or ""
            if not href:
                for el in lic.iter():
                    if _tag(el) in ("license_ref", "ext-link"):
                        href = (el.get("{http://www.w3.org/1999/xlink}href") or "".join(el.itertext())).strip()
                        if "creativecommons" in href:
                            break
            m = re.search(r"creativecommons\.org/(licenses|publicdomain)/([a-z-]+)/?([\d.]+)?", href, re.I)
            if m:
                return ("CC0" if m.group(2).lower() == "zero" else "CC " + m.group(2).upper()) + \
                    (f" {m.group(3)}" if m.group(3) else "")
            if lic.get("license-type"):
                return lic.get("license-type")
            text = _clean("".join(lic.itertext()))
            if text:
                return text[:60]
        return ""

    # ---------------- inline text ----------------
    def inline(self, el):
        parts = [el.text or ""]
        for ch in el:
            parts.append(self.inline_child(ch))
            parts.append(ch.tail or "")
        return "".join(parts)

    def inline_child(self, ch):
        t = _tag(ch)
        if t in ("inline-formula", "disp-formula"):
            f = formula(ch)
            return f" ${f}$ " if f else ""
        if t == "math":
            f = mathml(ch)
            return f" ${f}$ " if f else ""
        if t in ("sup", "sub"):
            s = _clean(self.inline(ch))
            if not s:
                return ""
            if t == "sup" and any(_tag(x) == "xref" and x.get("ref-type") == "bibr" for x in ch) \
                    and re.fullmatch(r"[\d\s,–\-−]+", s):
                return f" [{s}]"  # citation superscript
            mark = "^" if t == "sup" else "_"
            return mark + (s if len(s) == 1 else "{" + s + "}")
        if t in ("fn", "inline-graphic", "graphic", "media", "alt-text", "object-id", "label"):
            return ""
        if t == "break":
            return " "
        if t in BLOCKS_IN_P:  # a block inside a title or table cell: its text only
            return " " + _clean(self.inline(ch)) + " "
        return self.inline(ch)

    # ---------------- blocks ----------------
    def blocks(self, container, depth, out):
        for ch in container:
            self.block(ch, depth, out)

    def block(self, ch, depth, out):
        t = _tag(ch)
        if t == "sec":
            self.section(ch, depth, out)
        elif t == "p":
            self.paragraph(ch, depth, out)
        elif t == "list":
            self.list(ch, depth, out)
        elif t == "disp-formula":
            f = formula(ch)
            if f:
                out.append(f"$${f}$$")
        elif t == "fig":
            self.caption(ch, "Figure", out)
        elif t == "table-wrap":
            self.table(ch, out)
        elif t in ("fig-group", "table-wrap-group", "boxed-text", "statement", "app", "body", "chem-struct-wrap"):
            title = ch.find("title")
            if title is not None and t in ("boxed-text", "statement", "app"):
                label = ch.find("label")
                if SKIP_SECTION_RE.search(_clean(self.inline(title))):
                    return
                text = _clean(((self.inline(label) + " ") if label is not None else "") + self.inline(title))
                if text:
                    out.append("#" * min(depth, 4) + " " + text)
            self.blocks(ch, depth + 1, out)
        elif t == "def-list":
            for item in ch.iter():
                if _tag(item) == "def-item":
                    term, definition = item.find("term"), item.find("def")
                    text = _clean(self.inline(definition)) if definition is not None else ""
                    head = _clean(self.inline(term)) if term is not None else ""
                    if head or text:
                        out.append(f"- {head}: {text}" if head else f"- {text}")
        elif t == "disp-quote":
            inner = []
            self.blocks(ch, depth, inner)
            out.extend("> " + b.replace("\n", "\n> ") for b in inner)
        elif t in ("preformat", "code"):
            code = "".join(ch.itertext()).strip("\n")
            if code.strip():
                fence = "```"
                while fence in code:
                    fence += "`"
                out.append(f"{fence}\n{code}\n{fence}")
        elif t in SKIP_BLOCKS:
            return
        else:
            text = _clean(self.inline(ch))
            if text:
                out.append(text)

    def section(self, sec, depth, out):
        title_el, label_el = sec.find("title"), sec.find("label")
        title = _clean(self.inline(title_el)) if title_el is not None else ""
        label = _clean(self.inline(label_el)) if label_el is not None else ""
        if sec.get("sec-type") in SKIP_SEC_TYPES or (title and SKIP_SECTION_RE.search(title)):
            return
        if title:
            heading = f"{label} {title}" if label and not title.startswith(label) else title
            out.append("#" * min(depth, 4) + " " + heading)
        self.blocks(sec, depth + 1, out)

    def paragraph(self, p, depth, out):
        buf = [p.text or ""]
        for ch in p:
            if _tag(ch) in BLOCKS_IN_P:
                self._flush(buf, out)
                buf = []
                self.block(ch, depth, out)
            else:
                buf.append(self.inline_child(ch))
            buf.append(ch.tail or "")
        self._flush(buf, out)

    @staticmethod
    def _flush(buf, out):
        text = _clean("".join(buf))
        text = re.sub(r"\$ ([,.;:!?)\]])", r"$\1", re.sub(r"([(\[]) \$", r"\1$", text))  # "( $x$ ," -> "($x$,"
        if text and re.search(r"\w", text):
            out.append(text)

    def list(self, lst, depth, out):
        ordered = lst.get("list-type") in ("order", "arabic", "alpha-lower", "alpha-upper", "roman-lower",
                                           "roman-upper")
        n = 0
        for item in lst:
            if _tag(item) != "list-item":
                continue
            inner = []
            self.blocks(item, depth, inner)
            text = " ".join(b for b in inner if b)
            if text:
                n += 1
                out.append((f"{n}. " if ordered else "- ") + text)

    def caption(self, el, kind, out):
        label = el.find("label")
        cap = el.find("caption")
        head = _clean(self.inline(label)) if label is not None else kind
        text = ""
        if cap is not None:
            parts = []
            for ch in cap:
                if _tag(ch) in ("title", "p"):
                    parts.append(_clean(self.inline(ch)))
            text = " ".join(p for p in parts if p)
        if text:
            out.append(f"{head.rstrip('.:')}. {text}")
        return text

    def table(self, wrap, out):
        self.caption(wrap, "Table", out)
        table = next((t for t in wrap.iter() if _tag(t) == "table"), None)
        if table is None:
            return
        rows = []
        for tr in table.iter():
            if _tag(tr) == "tr":
                cells = [_clean(self.inline(c)).replace("|", "\\|") for c in tr if _tag(c) in ("td", "th")]
                if any(cells):
                    rows.append(cells)
        if len(rows) < 2:
            return
        ncol = max(len(r) for r in rows)
        rows = [r + [""] * (ncol - len(r)) for r in rows]
        lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * ncol]
        lines += ["| " + " | ".join(r) + " |" for r in rows[1:MAX_TABLE_ROWS + 1]]
        if len(rows) > MAX_TABLE_ROWS + 1:
            lines.append(f"({len(rows) - MAX_TABLE_ROWS - 1} more rows)")
        out.append("\n".join(lines))

    # ---------------- document ----------------
    def markdown(self):
        info = self.meta()
        out = [f"# {info['title']}"] if info["title"] else []
        for ab in self.root.iterfind(".//article-meta/abstract"):
            if ab.get("abstract-type") in ("graphical", "teaser", "toc", "web-summary"):
                continue
            title = ab.find("title")
            name = _clean(self.inline(title)) if title is not None else ""
            out.append("## " + (name if name and name.lower() != "abstract" else "Abstract"))
            self.blocks(ab, 3, out)
        body_start = len(out)
        body = self.root.find("body")
        if body is not None:
            self.blocks(body, 2, out)
        back = self.root.find("back")
        if back is not None:
            for group in back:
                if _tag(group) in ("app-group", "app"):
                    self.blocks(group if _tag(group) == "app-group" else [group], 2, out)
        floats = self.root.find("floats-group")
        if floats is not None:
            self.blocks(floats, 2, out)
        info["body_chars"] = sum(len(b) for b in out[body_start:])
        text = "\n\n".join(b for b in out if b.strip())
        text = EMAIL_RE.sub("[email]", text)  # no personal contact details in training text
        return info, re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


# ---------------------------------------------------------------------------
# Equations: TeX when the publisher supplies it, else MathML -> LaTeX
# ---------------------------------------------------------------------------
FUNCS = {"log", "ln", "exp", "sin", "cos", "tan", "max", "min", "lim", "sup", "inf", "det", "arg", "mod", "Pr",
         "var", "cov", "logit", "argmax", "argmin", "tanh", "sinh", "cosh", "sgn", "diag", "tr", "rank", "dim"}
BIG_OPS = ("\\sum", "\\prod", "\\int", "\\lim", "\\max", "\\min", "\\sup", "\\inf", "\\arg", "\\bigcup", "\\bigcap")
MO = {"∑": "\\sum", "∏": "\\prod", "∫": "\\int", "×": "\\times", "≤": "\\le", "⩽": "\\le", "≥": "\\ge", "⩾": "\\ge",
      "≠": "\\ne", "±": "\\pm", "∓": "\\mp", "∼": "\\sim", "≈": "\\approx", "≃": "\\simeq", "→": "\\to",
      "←": "\\leftarrow", "⇒": "\\Rightarrow", "⇔": "\\Leftrightarrow", "↔": "\\leftrightarrow", "∞": "\\infty",
      "∈": "\\in", "∉": "\\notin", "⊂": "\\subset", "⊆": "\\subseteq", "∪": "\\cup", "∩": "\\cap",
      "·": "\\cdot", "⋅": "\\cdot", "∙": "\\cdot", "−": "-", "…": "\\ldots", "⋯": "\\cdots", "∂": "\\partial",
      "∇": "\\nabla", "∝": "\\propto", "≡": "\\equiv", "∀": "\\forall", "∃": "\\exists", "¬": "\\neg",
      "∧": "\\wedge", "∨": "\\vee", "⊗": "\\otimes", "⊕": "\\oplus", "∘": "\\circ", "′": "'", "″": "''",
      "{": "\\{", "}": "\\}", "%": "\\%", "#": "\\#", "&": "\\&", "⟨": "\\langle", "⟩": "\\rangle", "‖": "\\|",
      "∣": "|", "∥": "\\parallel", "⊥": "\\perp", "≪": "\\ll", "≫": "\\gg", "\u2061": "", "\u2062": "",
      "\u2063": "", "\u2064": "", "\u200b": ""}
ACCENTS = {"^": "\\hat", "ˆ": "\\hat", "¯": "\\bar", "‾": "\\bar", "_": "\\bar", "~": "\\tilde", "˜": "\\tilde",
           "˙": "\\dot", "¨": "\\ddot", "→": "\\vec", "⃗": "\\vec"}
TEX_ENCODINGS = {"application/x-tex", "tex", "latex", "text/x-latex", "mathematica"}


def _cmd(s):
    """A LaTeX command word needs a space before a following letter."""
    return s + " " if re.fullmatch(r"\\[A-Za-z]+", s) else s


def _text(el):
    return "".join(el.itertext()).strip()


def formula(el):
    """LaTeX for a formula: from MathML when present (it converts cleanly), else
    from the publisher's TeX (often generated, with {\\rm{...}} noise)."""
    for node in el.iter():
        if _tag(node) == "math":
            tex = _clean(mathml(node))
            if tex:
                return tex
    for node in el.iter():
        if _tag(node) == "tex-math":
            tex = tex_body("".join(node.itertext()))
            if tex:
                return tex
    return ""


def tex_body(s):
    m = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", s, re.S)
    if m:
        s = m.group(1)
    s = re.sub(r"^[\s{}]*(?=\$)", "", s.strip())  # "}{}$$...$$" left by some typesetters
    s = re.sub(r"\\(begin|end)\{(equation|displaymath|math)\*?\}", "", s).strip()
    for left, right in (("$$", "$$"), ("\\[", "\\]"), ("$", "$"), ("\\(", "\\)")):
        if s.startswith(left) and s.endswith(right) and len(s) > len(left) + len(right):
            s = s[len(left):-len(right)].strip()
            break
    return _clean(s)


def mathml(el, top=True):
    t = _tag(el)
    kids = [k for k in el if _tag(k) not in ("annotation", "annotation-xml")]
    if t == "semantics":
        for k in el:
            if _tag(k) == "annotation" and (k.get("encoding") or "").lower() in TEX_ENCODINGS and _text(k):
                return _text(k)
        return mathml(kids[0], top) if kids else ""
    if t == "mi":
        s = _text(el)
        if len(s) > 1:
            return _cmd("\\" + s) if s in FUNCS else "\\mathrm{" + s + "}"
        return MO.get(s, s)
    if t == "mn":
        return _text(el)
    if t == "mo":
        s = _text(el)
        if s in FUNCS:
            return _cmd("\\" + s)
        return _cmd(MO.get(s, s))
    if t in ("mtext", "ms"):
        s = "".join(el.itertext())
        if not s.strip():
            return " "
        if el.get("mathvariant") == "italic" and re.fullmatch(r"\w+", s.strip()):
            return s.strip()  # italic letters are ordinary math symbols
        s = re.sub(r"([{}%#&_$])", r"\\\1", s)
        return "\\text{" + s + "}"
    if t == "mspace":
        return " "
    if t in ("mphantom", "none", "mprescripts"):
        return ""
    k = [mathml(x, False) for x in kids]
    g = [s.strip() if len(s.strip()) == 1 or re.fullmatch(r"\\[A-Za-z]+\s?", s) else "{" + s + "}" for s in k]
    if t == "msub" and len(k) >= 2:
        return f"{g[0]}_{{{k[1]}}}"
    if t == "msup" and len(k) >= 2:
        return f"{g[0]}^{{{k[1]}}}"
    if t == "msubsup" and len(k) >= 3:
        return f"{g[0]}_{{{k[1]}}}^{{{k[2]}}}"
    if t == "mfrac" and len(k) >= 2:
        return f"\\frac{{{k[0]}}}{{{k[1]}}}"
    if t == "msqrt":
        return "\\sqrt{" + "".join(k) + "}"
    if t == "mroot" and len(k) >= 2:
        return f"\\sqrt[{k[1]}]{{{k[0]}}}"
    if t in ("munder", "mover", "munderover") and len(k) >= 2:
        base = k[0].strip()
        big = base.startswith(BIG_OPS)
        if t == "munderover" and len(k) >= 3:
            return f"{g[0]}_{{{k[1]}}}^{{{k[2]}}}"
        mark = _text(kids[1])
        if t == "mover" and mark in ACCENTS:
            return f"{ACCENTS[mark]}{{{base}}}"
        if t == "munder" and mark in ("_", "‾", "¯"):
            return f"\\underline{{{base}}}"
        if big:
            return f"{g[0]}{'_' if t == 'munder' else '^'}{{{k[1]}}}"
        return f"\\{'underset' if t == 'munder' else 'overset'}{{{k[1]}}}{{{base}}}"
    if t == "mfenced":
        open_, close = el.get("open", "("), el.get("close", ")")
        seps = (el.get("separators", ",") or "").strip()
        body = (seps[:1] or "").join(k)
        return MO.get(open_, open_) + body + MO.get(close, close)
    if t == "mtable":
        rows = []
        for tr in kids:
            cells = [c for c in tr if _tag(c) == "mtd"]
            if _tag(tr) == "mlabeledtr":
                cells = cells[1:]
            rows.append(" & ".join(_clean(mathml(c, False)) for c in cells))
        env = "aligned" if top else "matrix"
        return f"\\begin{{{env}}}" + " \\\\ ".join(rows) + f"\\end{{{env}}}"
    if t in ("math", "mrow", "mstyle", "mpadded", "menclose", "merror", "mtd", "mtr") and len(kids) == 1:
        return mathml(kids[0], top)
    return "".join(k)


def jats_to_markdown(xml_bytes):
    """(info dict, markdown) for one JATS article."""
    return Jats(ET.fromstring(xml_bytes)).markdown()


# ---------------------------------------------------------------------------
# 2. fetch + convert, in parallel
# ---------------------------------------------------------------------------
def safe_name(title, limit=90):
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", title or "untitled")
    name = " ".join(name.split())[:limit].rstrip(" .")
    return name or "untitled"


def article_pmcid(article):
    for el in article.iterfind("front/article-meta/article-id"):
        if el.get("pub-id-type") in ("pmc", "pmcid", "pmcaid") and (el.text or "").strip():
            digits = re.sub(r"\D", "", el.text)
            if digits:
                return "PMC" + digits
    return None


def fetch_articles(pmcids):
    """{pmcid: <article> element} for one efetch request. A batch the XML parser
    rejects is split in halves until the bad paper is alone (then missing)."""
    data = urllib.parse.urlencode({"db": "pmc", "id": ",".join(p[3:] for p in pmcids), "retmode": "xml",
                                   "tool": "tinygpt-downloader"}).encode()
    raw = http_get(EFETCH_URL, data=data)
    if raw is None:
        return {}
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        if len(pmcids) == 1:
            return {pmcids[0]: None}
        half = len(pmcids) // 2
        return {**fetch_articles(pmcids[:half]), **fetch_articles(pmcids[half:])}
    return {pmcid: art for art in root.iter("article") if (pmcid := article_pmcid(art))}


def process_batch(rows, out_dir, min_body):
    found = fetch_articles([r[0] for r in rows])
    results = []
    for pmcid, _pmid, year, cited, journal, title in rows:
        base = {"pmcid": pmcid, "chars": 0, "journal": journal, "year": year, "cited": cited, "licence": "",
                "file": ""}
        article = found.get(pmcid, False)
        if article is False:
            results.append({**base, "status": "no_fulltext"})
            continue
        if article is None:
            results.append({**base, "status": "bad_xml"})
            continue
        info, text = Jats(article).markdown()
        base.update(journal=info["journal"] or journal, year=info["year"] or year, licence=info["licence"])
        if not (info["lang"] or "en").lower().startswith("en"):
            results.append({**base, "status": "not_english"})
        elif info["body_chars"] < min_body:
            results.append({**base, "status": "no_body", "chars": len(text)})
        else:
            name = f"{pmcid}__{safe_name(info['title'] or title)}.md"
            path = os.path.join(out_dir, name)
            with open(path + ".part", "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
            os.replace(path + ".part", path)
            results.append({**base, "status": "ok", "chars": len(text), "file": name})
    return results


def read_report(path):
    done = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            head = handle.readline().rstrip("\n").split("\t")
            for line in handle:
                row = dict(zip(head, line.rstrip("\n").split("\t")))
                done[row.get("pmcid")] = row.get("status")
    return done


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="folder for the Markdown files (e.g. ai_training_data\\pmc)")
    p.add_argument("--meta", help="folder for list.tsv and report.tsv (default: _<out name>_meta next to --out, "
                                  "which data_prep --text-root skips)")
    p.add_argument("--query", default=DEFAULT_QUERY, help="Europe PMC search query")
    p.add_argument("--sort", default="CITED desc", help="Europe PMC sort, e.g. 'CITED desc' or 'P_PDATE_D desc'")
    p.add_argument("--max-papers", type=int, default=100_000)
    p.add_argument("--workers", type=int, default=2, help="requests in flight (the 3-per-second limit applies to all)")
    p.add_argument("--batch", type=int, default=100, help="papers per request")
    p.add_argument("--min-body-chars", type=int, default=1500, help="skip papers whose XML body is shorter")
    p.add_argument("--list-only", action="store_true", help="build the paper list and stop")
    args = p.parse_args(argv)
    if not 1 <= args.workers <= 3:
        p.error("--workers must be 1..3")
    if not 1 <= args.batch <= 200:
        p.error("--batch must be 1..200")
    if args.max_papers < 1:
        p.error("--max-papers must be >= 1")
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
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.meta, exist_ok=True)
    rows = build_list(args.query, args.sort, args.max_papers, os.path.join(args.meta, "list.tsv"))
    if args.list_only:
        return
    report_path = os.path.join(args.meta, "report.tsv")
    done = read_report(report_path)
    have = {name.split("__", 1)[0] for name in os.listdir(args.out) if name.endswith(".md")}
    todo = [r for r in rows if r[0] not in have and done.get(r[0]) not in FINAL]
    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    print(f"{len(rows) - len(todo):,} of {len(rows):,} papers already done; fetching {len(todo):,} in "
          f"{len(batches):,} requests of up to {args.batch} into {args.out}", flush=True)
    new_report = not os.path.isfile(report_path)
    counts, t0, n = {}, time.time(), 0
    with open(report_path, "a", encoding="utf-8", newline="\n") as report, \
            cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        if new_report:
            report.write("\t".join(REPORT_COLUMNS) + "\n")

        def record(future, batch):
            nonlocal n
            try:
                results = future.result()
            except Exception as exc:  # noqa: BLE001 - network trouble: retried on the next run
                status = f"error: {type(exc).__name__}: {str(exc)[:80]}"
                results = [{"pmcid": r[0], "status": status, "chars": 0, "journal": r[4], "year": r[2],
                            "cited": r[3], "licence": "", "file": ""} for r in batch]
            for result in results:
                report.write("\t".join(_flat(result[c]) for c in REPORT_COLUMNS) + "\n")
                key = result["status"].split(":")[0]
                counts[key] = counts.get(key, 0) + 1
            report.flush()
            n += len(batch)
            rate = n / max(time.time() - t0, 1e-9)
            print(f"  {n:,}/{len(todo):,} papers ({rate:.1f}/s, about {(len(todo) - n) / max(rate, 1e-9) / 60:.0f} min "
                  f"left) {counts}", flush=True)

        pending = {}
        try:
            for batch in batches:
                pending[pool.submit(process_batch, batch, args.out, args.min_body_chars)] = batch
                if len(pending) >= 2 * args.workers:
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

    ok = counts.get("ok", 0)
    print(f"Done: {ok:,} papers written to {args.out}; statuses {counts}; report {report_path}")


if __name__ == "__main__":
    main()
