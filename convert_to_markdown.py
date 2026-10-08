r"""Convert documents into Markdown text for data_prep.py (replaces pdf_to_markdown.py).

A text-only language model can only learn from text, so every source type is
turned into readable Markdown here, before tokenization:

  .pdf                 pymupdf4llm (headings, lists, tables where detected); plot-heavy pages as plain
                       text; running headers, footers, page numbers, per-page notices and the text
                       inside figures removed; scanned pages read with OCR when --ocr is given
  .docx                paragraphs (headings as #), tracked insertions, content controls, equations
                       as $...$, tables as Markdown tables, footnotes and endnotes under Notes
  .pptx                slides in deck order: titles, bullet text, tables and speaker notes (no extra package)
  .xlsx                each sheet as a Markdown table, first --max-rows rows and --max-cols columns,
                       dates as dates (no extra package)
  .csv / .tsv / .txt   tables become Markdown tables; plain text is copied unchanged (UTF-8, UTF-16
                       or Windows-1252, detected)
  GWAS summary stats   (.tsv/.txt/.csv/.gz with CHR, BP, P columns; tab, comma or space separated)
                       become a written summary: variant count, lambda GC, independent loci, top hits
  .ipynb               Markdown cells as text, code cells as fenced code (outputs dropped)
  .html / .htm         visible text (table cells separated)
  .md / .markdown      copied (normalised line endings)
  code (.py .R ...)    fenced code blocks
  images (.png .jpg)   NOT converted: a text model cannot see pixels. Use the figure caption
                       from the PDF/slide text, or OCR / a vision model upstream.

Examples
--------
    python convert_to_markdown.py D:\sources\papers --out D:\converted --by-type --recursive
    python convert_to_markdown.py lecture.pptx --out D:\converted
    python convert_to_markdown.py "D:\Papers" --out %USERPROFILE%\ai_training_data --by-type --types pdf

With --by-type, outputs go to <out>\books (PDFs of --book-pages pages or more),
<out>\articles, <out>\slides, <out>\tables and <out>\codes. data_prep.py
--text-root <out> then reads each folder as its own named source.

Existing outputs are skipped (rerun after an interruption); two inputs that
would get the same output name are told apart (notes.docx.md, name__folder.md).
conversion_manifest.json (in --out, or in the input folder without --out)
records which input made each output, so a rerun keeps every name, never takes
an earlier output for a new input and never converts its own outputs. Each PDF
is converted in its own process, so a PDF that crashes MuPDF or hangs (--timeout)
costs only that file; the report and manifest are saved as the batch goes. Starting
that process costs about 1.6 s per PDF (loading pymupdf4llm), about 1.2 hours over
2,760 PDFs; --workers N spreads it over N processes.
"""

import argparse
import bisect
import csv
import gzip
import html.parser
import io
import json
import math
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".svg", ".webp"}
CODE_EXT = {".py": "python", ".r": "r", ".rmd": "r", ".sh": "bash", ".js": "javascript", ".sql": "sql",
            ".cpp": "cpp", ".c": "c", ".java": "java", ".jl": "julia", ".m": "matlab"}
TYPE_FOLDER = {"book": "books", "article": "articles", "docx": "articles", "pptx": "slides", "xlsx": "tables",
               "table": "tables", "code": "codes", "html": "articles", "text": "articles", "ipynb": "codes"}
MAX_ZIP_MEMBER = 200 * 1024 * 1024  # refuse absurdly large XML parts (zip bombs)
MAX_ZIP_TOTAL = 1024 * 1024 * 1024
MAX_TEXT_BYTES = 500 * 1024 * 1024  # a plain-text "document" larger than this is not prose
MIN_TEXT_CHARS = 30                 # a PDF with less text than this (scanned, no text layer) is not written
# compressed files that are data, not prose or tables (read whole they would be GBs of text)
NOT_TEXT_GZ = {".vcf", ".bcf", ".fa", ".fasta", ".fna", ".faa", ".fq", ".fastq", ".gtf", ".gff", ".gff3", ".bed",
               ".bedgraph", ".sam", ".bam", ".cram", ".tar", ".pdf", ".json", ".xml", ".bgen", ".pgen", ".npy"}
MANIFEST_NAME = "conversion_manifest.json"  # which input produced each output (stable names on reruns)
csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))  # one long cell (an abstract, a gene list) is not an error


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def md_table(rows, max_rows, total=None, max_cols=None):
    """Markdown table of the first max_rows data rows (and the first max_cols
    columns, when given). `total` is the number of data rows when known;
    otherwise it is taken from `rows`."""
    rows = [["" if c is None else str(c).replace("|", "\\|").replace("\n", " ").strip() for c in r] for r in rows]
    rows = [r for r in rows if any(c for c in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    hidden_cols = 0
    if max_cols and width > max_cols:  # an expression or dosage matrix would be megabytes of numbers
        hidden_cols, width = width - max_cols, max_cols
        rows = [r[:width] for r in rows]
    rows = [r + [""] * (width - len(r)) for r in rows]
    shown = rows[: max_rows + 1]
    out = ["| " + " | ".join(shown[0]) + " |", "|" + "---|" * width]
    out += ["| " + " | ".join(r) + " |" for r in shown[1:]]
    data_rows = len(rows) - 1 if total is None else total
    if data_rows > len(shown) - 1:
        hidden = data_rows - (len(shown) - 1)
        out.append(f"\n*{hidden:,} more rows not shown (limit --max-rows {max_rows}).*" if total is not None or
                   len(rows) > len(shown) + 1 else f"\n*More rows not shown (limit --max-rows {max_rows}).*")
    if hidden_cols:
        out.append(f"\n*{hidden_cols:,} more columns not shown (limit --max-cols {max_cols}).*")
    return "\n".join(out)


def check_zip(path):
    """Refuse Office files whose parts would expand to absurd sizes (zip bombs)."""
    with zipfile.ZipFile(path) as zf:
        sizes = [i.file_size for i in zf.infolist()]
    if sizes and (max(sizes) > MAX_ZIP_MEMBER or sum(sizes) > MAX_ZIP_TOTAL):
        raise ValueError("an archive part is too large to parse safely")


def read_zip_xml(zf, name):
    info = zf.getinfo(name)
    if info.file_size > MAX_ZIP_MEMBER:
        raise ValueError(f"{name} is too large to parse safely ({info.file_size:,} bytes)")
    return ET.fromstring(zf.read(name))


def natural_key(name):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def text_encoding(path):
    """Encoding of a text file, from its first MB: the one a BOM names (UTF-8,
    UTF-16, UTF-32); UTF-16 without a BOM when NUL bytes fill every other byte
    (Excel "Unicode Text" and PowerShell 5.1 '>' files); Windows-1252 when
    invalid UTF-8 outnumbers valid non-ASCII UTF-8 (a real cp1252 file); else
    UTF-8, where a few stray bytes become U+FFFD instead of mangling the rest."""
    import codecs
    raw = gzip.open(path, "rb") if path.lower().endswith(".gz") else open(path, "rb")
    with raw:
        sample = raw.read(1 << 20)
    if sample.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        return "utf-32"
    if sample.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return "utf-16"
    if sample.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    head = sample[:4096] if len(sample) >= 4096 else sample[: len(sample) // 2 * 2]
    if len(head) >= 4:
        even, odd = head[0::2].count(0), head[1::2].count(0)
        half = len(head) // 2
        if odd >= 0.3 * half and even <= 0.05 * half:
            return "utf-16-le"
        if even >= 0.3 * half and odd <= 0.05 * half:
            return "utf-16-be"
    decoded = codecs.getincrementaldecoder("utf-8")(errors="replace").decode(sample, final=False)  # cut tail ignored
    invalid = decoded.count("\ufffd") - sample.count(b"\xef\xbf\xbd")
    valid = sum(1 for ch in decoded if ord(ch) > 127) - decoded.count("\ufffd")
    return "cp1252" if invalid > 0 and invalid > valid else "utf-8-sig"


def open_text(path):
    raw = gzip.open(path, "rb") if path.lower().endswith(".gz") else open(path, "rb")
    return io.TextIOWrapper(raw, encoding=text_encoding(path), errors="replace", newline="")


def read_text(path):
    with open_text(path) as handle:
        return handle.read()


def inner_ext(path):
    """Extension ignoring a trailing .gz (x.tsv.gz -> .tsv)."""
    name = path.lower()
    if name.endswith(".gz"):
        name = name[:-3]
    return os.path.splitext(name)[1]


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
# pymupdf4llm's layout analysis slows down with the number of vector shapes on a page
# (about 30-40 s for a plot of a few hundred to thousands of paths). Such pages are read as
# plain text; get_cdrawings() also sees shapes inside embedded figures (Form XObjects).
HEAVY_PDF_PAGE_DRAWINGS = 500
HEAVY_PDF_PAGE_BYTES = 200_000


def _heavy_page(page):
    if len(page.read_contents()) > HEAVY_PDF_PAGE_BYTES:
        return True
    try:
        return len(page.get_cdrawings()) > HEAVY_PDF_PAGE_DRAWINGS
    except Exception:  # noqa: BLE001 - an odd page: let pymupdf4llm try
        return False


SCAN_CHARS_PER_PAGE = 200   # a page of text has 2,000-4,000 characters; a page with fewer may be a scan
SCAN_IMAGE_COVER = 0.8      # ... when images cover at least this share of it (a scan fills the page)
OCR_SECONDS_PER_PAGE = 15   # --ocr: a child process gets at least this long per page before it is stopped
PDF_SECONDS_PER_PAGE = 5    # without --ocr (layout analysis of a page takes well under a second on a CPU)
_PICTURE_TEXT = re.compile(r"(?:<!--|-----) Start of picture text (?:-->|-----).*?(?:<!--|-----) End of picture "
                           r"text (?:-->|-----)\n?", re.S)
# pymupdf4llm writes each image link on a line of its own ('\n![](path)\n'): matching the whole line lets
# the path run to the last ')', so names like 'paper (1).pdf-0001-03.png' stay whole
_IMAGE_LINE = re.compile(r"^[ \t]*!\[[^\]\n]*\]\((.*)\)[ \t]*$", re.M)
_IMAGE_LINK = re.compile(r"!\[[^\]\n]*\]\(([^)\n]*)\)")  # any other (inline) image link
_IMAGE_LINK_PLAIN = re.compile(r"!\[[^\]\n]*\]\(([^)<\n][^)\n]*)\)")  # ... not one already written as ![](<...>)


def _scan_page(page, chars):
    """A page with (almost) no text layer whose area is mostly covered by images."""
    import pymupdf
    if chars >= SCAN_CHARS_PER_PAGE:
        return False
    try:
        covered = sum(abs(pymupdf.Rect(info["bbox"]) & page.rect) for info in page.get_image_info())
    except Exception:  # noqa: BLE001 - an odd page: judged by the document's text alone
        return False
    return covered >= SCAN_IMAGE_COVER * (abs(page.rect) or 1.0)


def _clean_pdf_markdown(text, image_dir=None):
    """pymupdf4llm extras that are not prose: the text inside figures (tick labels and
    legend fragments, wrapped in '<!-- Start of picture text -->' markers), '<br>' line
    breaks and image links. Links are kept only with --extract-images, pointing at the
    images folder next to the Markdown file."""
    text = _PICTURE_TEXT.sub("", text)
    text = "\n".join(line.replace("<br>", " ") if line.lstrip().startswith("|") else line.replace("<br>", "\n")
                     for line in text.split("\n"))
    if image_dir is None:
        return _IMAGE_LINK.sub("", _IMAGE_LINE.sub("", text))
    folder = os.path.basename(image_dir)

    def link(m):
        url = f"{folder}/{os.path.basename(m.group(1).strip().strip('<>').replace(chr(92), '/'))}"
        return f"![](<{url}>)" if re.search(r"[\s()<>]", url) else f"![]({url})"  # <...> keeps spaces valid
    return _IMAGE_LINK_PLAIN.sub(link, _IMAGE_LINE.sub(link, text))


def convert_pdf(path, args, image_dir=None, info=None):
    """Markdown text of a PDF. `image_dir` is the folder for --extract-images (default:
    <out or the PDF's folder>/<stem>_images). Pages with no text layer whose area is
    mostly images count as scans: with --ocr only those pages are read by OCR and put
    back in place. `info` (a dict), when given, receives pages, scan_pages and ocr_pages."""
    try:
        import pymupdf
        import pymupdf4llm
    except ImportError:
        raise RuntimeError("PDF conversion needs pymupdf4llm: python -m pip install pymupdf4llm")
    kwargs = {"use_ocr": False}  # OCR only when --ocr asks for it (else it runs whenever Tesseract is on PATH)
    if getattr(args, "extract_images", False):
        if image_dir is None:
            folder = getattr(args, "out", None) or os.path.dirname(os.path.abspath(path))
            image_dir = os.path.join(folder, os.path.splitext(os.path.basename(path))[0] + "_images")
        os.makedirs(image_dir, exist_ok=True)
        kwargs.update(write_images=True, image_path=image_dir, image_format="png")
    else:
        image_dir = None
    with pymupdf.open(path) as doc:
        heavy = {i for i in range(doc.page_count) if _heavy_page(doc[i])}
        pages, run = [], []
        for i in range(doc.page_count + 1):
            if i < doc.page_count and i not in heavy:
                run.append(i)
                continue
            if run:  # consecutive ordinary pages keep pymupdf4llm's headings, lists and tables
                chunks = pymupdf4llm.to_markdown(doc, pages=run, page_chunks=True, **kwargs)
                pages.extend(_clean_pdf_markdown(chunk["text"], image_dir) for chunk in chunks)
                run = []
            if i < doc.page_count:
                pages.append(doc[i].get_text("text"))
        # scans are judged page by page: a mostly digital book can hold scanned chapters and vice versa
        chars = [len(doc[i].get_text("text").strip()) for i in range(doc.page_count)]
        scans = [i for i in range(doc.page_count) if _scan_page(doc[i], chars[i])]
        if sum(chars) < SCAN_CHARS_PER_PAGE * doc.page_count:  # nearly no text at all: every thin page may be a scan
            scans = sorted(set(scans) | {i for i in range(doc.page_count) if chars[i] < SCAN_CHARS_PER_PAGE})
        ocr_pages, tessdata = 0, None
        if scans and getattr(args, "ocr", False):
            tessdata = find_tessdata()
            if tessdata is None:
                if len(scans) >= doc.page_count:  # a whole scan: without OCR there is nothing to convert
                    _tessdata_or_fail()
                if info is not None:  # scanned pages in a digital PDF: keep its text, note the missing OCR
                    info["ocr_unavailable"] = True
        if tessdata is not None:
            for k, i in enumerate(scans, 1):
                text = _ocr_page(doc[i], tessdata)
                if len(text) > 2 * len(pages[i].strip()):
                    pages[i], ocr_pages = text, ocr_pages + 1
                if k % 25 == 0 or k == len(scans):
                    print(f"  OCR {os.path.basename(path)}: {k} of {len(scans)} scanned pages", file=sys.stderr,
                          flush=True)
        if info is not None:
            info.update(pages=doc.page_count, scan_pages=len(scans), ocr_pages=ocr_pages)
    return "\n\n".join(p.strip() for p in strip_page_furniture(pages) if p.strip())


_PAGE_FORM = re.compile(r"(?:page\s*)?(\d{1,4})\s*(?:of|/)\s*(\d{1,4})|page\s*(\d{1,4})", re.I)  # "3 of 9", "page 3"
_ROMAN = re.compile(r"(?=[mdclxvi])m{0,3}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})", re.I)
_ROMAN_VALUE = {"m": 1000, "d": 500, "c": 100, "l": 50, "x": 10, "v": 5, "i": 1}
_EDGE_NUMBER = (re.compile(r"^(?:page\s*)?(\d{1,4})\s*[|·•-]?\s+(.*)$"),
                re.compile(r"^(.*?)\s+[|·•-]?\s*(?:page\s*)?(\d{1,4})$"))


def _normal(line):
    s = re.sub(r"<[^>]+>|[*_`#>]", " ", line)
    return re.sub(r"\s+", " ", s).strip().lower()


def _split_page_number(s):
    """('584 | nature | vol 631' -> ('nature | vol 631', 584)); (s, None) if there is none
    or stripping it would leave fewer than two words ('chapter 1' stays whole)."""
    m = _EDGE_NUMBER[0].match(s)
    if m and len(m.group(2).strip(" |·•-").split()) >= 2:
        return m.group(2).strip(" |·•-"), int(m.group(1))
    m = _EDGE_NUMBER[1].match(s)
    if m and len(m.group(1).strip(" |·•-").split()) >= 2:
        return m.group(1).strip(" |·•-"), int(m.group(2))
    return s.strip(" |·•-"), None


def _stepping(pairs, pages):
    """True if the numbers step with the page index like page numbers: one
    (number - page index) offset accounts for most occurrences."""
    if len(pairs) < 3:
        return False
    offsets = {}
    for page, number in pairs:
        offsets[number - page] = offsets.get(number - page, 0) + 1
    return max(offsets.values()) >= max(3, 0.6 * len(pairs))


def _step_offsets(pairs, n_pages=3):
    """{number - page index} offsets shared by 3 or more (page, number) pairs (every
    page of a shorter document): the numbering a run of page labels follows ('ii',
    'iii', 'iv' or '3 of 9', '4 of 9')."""
    offsets = {}
    for page, number in pairs:
        offsets[number - page] = offsets.get(number - page, 0) + 1
    return {k for k, n in offsets.items() if n >= min(3, max(n_pages, 2))}


def _page_label(plain):
    """('roman', value, None) or ('form', number, total or None) for 'xiv', 'page 3',
    '3 of 9', '3/9'; else None."""
    if _ROMAN.fullmatch(plain):
        value = [_ROMAN_VALUE[c] for c in plain.lower()]
        return "roman", sum(-v if i + 1 < len(value) and v < value[i + 1] else v for i, v in enumerate(value)), None
    m = _PAGE_FORM.fullmatch(plain)
    if m:
        return "form", int(m.group(1) or m.group(3)), int(m.group(2)) if m.group(2) else None
    return None


def strip_page_furniture(pages, edge=4):
    """Remove running headers, footers, page numbers and repeated notices.

    Only the first and last `edge` non-empty lines of each page are candidates.
    A number at either end of a line is ignored only when it behaves like a page
    number (it steps with the page), so 'Supplementary Table 3' is kept. A
    candidate is removed if it is a page number (a bare number, a roman numeral
    or 'page 3 of 9' that steps with the page); if it is in capitals (like a
    book's running section title) and sits at a page edge on 2 or more pages; or if it has 2+
    words and sits at a page edge on 3 or more pages and on at least 10% of pages.
    A line of 2+ words found on at least half of all pages is removed wherever it
    sits. Table rows and HTML comments are never removed."""
    if len(pages) < 2:
        return pages
    split = [p.splitlines() for p in pages]
    edges = []
    for lines in split:
        idx = [i for i, l in enumerate(lines) if l.strip()]
        edges.append(sorted(set(idx[:edge] + idx[-edge:])))
    # 1. which "text + page number" lines, bare numbers and page labels behave like page numbering
    numbered, bare, labels = {}, [], {"roman": [], "form": []}
    for page, (lines, idx) in enumerate(zip(split, edges)):
        for i in idx:
            raw = lines[i].strip()
            plain = re.sub(r"[*_`#>]", "", raw).strip()
            if plain.isdigit() and len(plain) <= 4:
                bare.append((page, int(plain)))
                continue
            label = _page_label(plain)
            if label:
                labels[label[0]].append((page, label[1]))
            text, number = _split_page_number(_normal(raw))
            if number is not None:
                numbered.setdefault(text, []).append((page, number))
    stepping_text = {t for t, pairs in numbered.items() if _stepping(pairs, len(pages))}
    bare_offsets = {}
    for page, number in bare:
        bare_offsets[number - page] = bare_offsets.get(number - page, 0) + 1
    page_offsets = set()
    if bare_offsets:
        best = max(bare_offsets, key=bare_offsets.get)
        if bare_offsets[best] >= max(3, 0.2 * len(pages)):
            page_offsets = {best}
    # roman numerals ('xiv') and 'page 3' / '3 of 9' go only when they step with the page ('MCMC', 'LD' stay)
    label_offsets = {kind: _step_offsets(pairs, len(pages)) for kind, pairs in labels.items()}

    def key_of(raw):
        s = _normal(raw)
        text, number = _split_page_number(s)
        return text if (number is not None and text in stepping_text) else s.strip(" |·•-")

    # 2. how often each candidate appears at a page edge, and anywhere
    counts, anywhere = {}, {}
    for lines, idx in zip(split, edges):
        for key in {key_of(lines[i]) for i in idx}:
            if len(key) >= 3:
                counts[key] = counts.get(key, 0) + 1
        for key in {key_of(l) for l in lines if 0 < len(l.strip()) <= 150}:
            if len(key) >= 3:
                anywhere[key] = anywhere.get(key, 0) + 1
    # a line on at least half of all pages is furniture wherever it sits (plot pages scramble the order)
    everywhere = {k for k, n in anywhere.items() if n >= max(3, len(pages) / 2) and len(k.split()) >= 2}
    repeated = max(3, 0.1 * len(pages))  # a heading reused in three chapters of a long book is not furniture
    out = []
    for page, (lines, idx) in enumerate(zip(split, edges)):
        drop = {i for i, l in enumerate(lines) if not l.strip().startswith(("|", "<!--"))
                and 0 < len(l.strip()) <= 150 and key_of(l) in everywhere}
        for i in idx:
            raw = lines[i].strip()
            if raw.startswith(("|", "<!--")) or len(raw) > 150:
                continue
            plain = re.sub(r"[*_`#>]", "", raw).strip()
            if plain.isdigit() and len(plain) <= 4:
                if int(plain) - page in page_offsets:
                    drop.add(i)
                continue
            label = _page_label(plain)
            # 'page 2 of 9' in a 9-page document is that page's label even where the numbering does not step
            if label and (label[1] - page in label_offsets[label[0]]
                          or (label[0] == "form" and label[2] == len(pages) and 1 <= label[1] <= label[2])):
                drop.add(i)
                continue
            key = key_of(raw)
            letters = [c for c in key if c.isalpha()]
            capitals = len(letters) >= 6 and sum(c.isupper() for c in raw if c.isalpha()) >= 0.8 * len(letters)
            if capitals and counts.get(key, 0) >= 2:
                drop.add(i)
            elif len(key.split()) >= 2 and counts.get(key, 0) >= repeated:
                drop.add(i)
        out.append("\n".join(l for i, l in enumerate(lines) if i not in drop))
    return out


# ---------------------------------------------------------------------------
# Office files
# ---------------------------------------------------------------------------
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
_MC = "{http://schemas.openxmlformats.org/markup-compatibility/2006}"
# deleted text, field codes and the old-Word copies of text boxes (mc:Fallback) are not part of the text
_DOCX_SKIP = {_W + "del", _W + "moveFrom", _W + "delText", _W + "instrText", _W + "rPr", _W + "pPr",
              _MC + "Fallback"}


def _omml(node):
    """A Word equation (OMML) as linear text: fractions a/b, x^{2}, x_{i}, sqrt(...)."""
    tag = node.tag if isinstance(node.tag, str) else ""
    kids = {c.tag: c for c in node if isinstance(c.tag, str)}
    part = lambda name: _omml(kids[_M + name]) if _M + name in kids else ""  # noqa: E731
    group = lambda s: s if len(s) <= 1 else "{" + s + "}"  # noqa: E731
    if tag == _M + "t" or tag == _W + "t":
        return node.text or ""
    if tag.endswith("Pr") or tag in _DOCX_SKIP:  # properties (m:fPr, m:rPr, w:rPr ...) hold no text
        return ""
    if tag == _M + "f":
        num, den = part("num"), part("den")
        return f"{num if len(num) <= 1 else '(' + num + ')'}/{den if len(den) <= 1 else '(' + den + ')'}"
    if tag == _M + "sSup":
        return part("e") + "^" + group(part("sup"))
    if tag == _M + "sSub":
        return part("e") + "_" + group(part("sub"))
    if tag == _M + "sSubSup":
        return part("e") + "_" + group(part("sub")) + "^" + group(part("sup"))
    if tag == _M + "rad":
        degree = part("deg")
        return (f"root{group(degree)}" if degree else "sqrt") + "(" + part("e") + ")"
    if tag == _M + "nary":
        pr = kids.get(_M + "naryPr")
        sign = pr.find(_M + "chr") if pr is not None else None
        sign = sign.get(_M + "val") if sign is not None else "\u222b"  # integral sign
        sub, sup = part("sub"), part("sup")
        return sign + ("_" + group(sub) if sub else "") + ("^" + group(sup) if sup else "") + " " + part("e")
    if tag == _M + "d":
        pr = kids.get(_M + "dPr")
        chars = {}
        for name, default in (("begChr", "("), ("endChr", ")"), ("sepChr", ",")):
            el = pr.find(_M + name) if pr is not None else None
            chars[name] = el.get(_M + "val", "") if el is not None else default
        inner = chars["sepChr"].join(_omml(c) for c in node if c.tag == _M + "e")
        return chars["begChr"] + inner + chars["endChr"]
    return "".join(_omml(c) for c in node if isinstance(c.tag, str))


def _docx_text(element, notes=None):
    """Text of a Word paragraph (or table cell) from all its runs in document order,
    including tracked insertions, content controls (citations), smart tags, fields
    and text boxes; deletions and field codes left out. Equations become $...$,
    footnote and endnote marks [^n] / [^en] (their text is listed under Notes)."""
    out = []

    def visit(node, top):
        tag = node.tag if isinstance(node.tag, str) else ""
        if tag in _DOCX_SKIP or (tag.startswith(_W) and tag.endswith("Pr")):  # w:sdtPr, w:tcPr ... hold no text
            return
        if tag == _W + "t":
            out.append(node.text or "")
        elif tag == _W + "tab":
            out.append("\t")
        elif tag in (_W + "br", _W + "cr"):
            out.append("\n")
        elif tag == _W + "noBreakHyphen":
            out.append("-")
        elif tag == _M + "oMath":
            math = re.sub(r"\s+", " ", _omml(node)).strip()
            if math:
                out.append(f"\x01${math}$\x02")  # markers: a space is added only where a word touches it
        elif tag in (_W + "footnoteReference", _W + "endnoteReference"):
            prefix = "" if tag == _W + "footnoteReference" else "e"
            out.append(f"[^{prefix}{node.get(_W + 'id')}]")
            if notes is not None:
                notes.add(prefix + str(node.get(_W + "id")))
        else:
            if tag == _W + "p" and not top:
                out.append("\n")  # a paragraph inside a cell or text box
            for child in node:
                visit(child, False)

    visit(element, True)
    text = re.sub(r"\x02(?=\w)", " ", re.sub(r"(?<=\w)\x01", " ", "".join(out)))
    return text.replace("\x01", "").replace("\x02", "")


def _docx_notes(path, used):
    """'[^n]: text' for the footnotes and endnotes the document refers to (Word's
    separator entries skipped)."""
    lines = []
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        for part, prefix, tag in (("word/footnotes.xml", "", "footnote"), ("word/endnotes.xml", "e", "endnote")):
            if part not in names:
                continue
            for note in read_zip_xml(zf, part).iter(_W + tag):
                key = prefix + str(note.get(_W + "id"))
                if note.get(_W + "type") or key not in used:  # separator, continuationSeparator, ...
                    continue
                text = re.sub(r"\s+", " ", _docx_text(note)).strip()
                if text:
                    lines.append(f"[^{key}]: {text}")
    return lines


def convert_docx(path, args):
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    check_zip(path)
    document = docx.Document(path)
    w = _W
    parts, notes = [], set()

    def paragraph(p):
        text = _docx_text(p._p, notes).strip()
        if not text:
            return
        style = (p.style.name or "").lower() if p.style is not None else ""
        level = re.match(r"heading (\d)", style)
        if level:
            parts.append("#" * min(int(level.group(1)), 6) + " " + text)
        elif style == "title":
            parts.append("# " + text)
        elif "list" in style:
            parts.append("- " + text)
        else:
            parts.append(text)

    def table(t):
        rows = []
        for row in t.rows:
            cells, last = [], None
            for cell in row.cells:  # a merged cell is returned once per grid column
                if cell._tc is not last:
                    cells.append(_docx_text(cell._tc, notes))
                last = cell._tc
            rows.append(cells)
        parts.append(md_table(rows, args.max_rows))

    def walk(element):
        for child in element.iterchildren():
            if child.tag == w + "p":
                paragraph(Paragraph(child, document))
            elif child.tag == w + "tbl":
                table(Table(child, document))
            elif child.tag in (w + "sdt", w + "sdtContent", w + "customXml", w + "smartTag"):
                walk(child)  # content controls (template fields, abstracts) hold ordinary paragraphs

    walk(document.element.body)
    note_lines = _docx_notes(path, notes) if notes else []
    if note_lines:
        parts += ["## Notes", "\n".join(note_lines)]
    return "\n\n".join(p for p in parts if p)


def _pptx_text(para):
    """Text of one a:p, with line breaks (a:br) as spaces."""
    out = []
    for node in para.iter():
        if node.tag == f"{{{NS['a']}}}t":
            out.append(node.text or "")
        elif node.tag == f"{{{NS['a']}}}br":
            out.append(" ")
    return re.sub(r"\s+", " ", "".join(out)).strip()


def _rels(zf, part):
    """{relationship id: (type, absolute part name)} for one package part."""
    folder, name = os.path.split(part)
    rel_part = f"{folder}/_rels/{name}.rels"
    if rel_part not in zf.namelist():
        return {}
    out = {}
    for r in read_zip_xml(zf, rel_part).findall("rel:Relationship", NS):
        target = r.get("Target", "")
        full = os.path.normpath(os.path.join(folder, target)).replace("\\", "/") if not target.startswith("/") \
            else target.lstrip("/")
        out[r.get("Id")] = (r.get("Type", ""), full)
    return out


def convert_pptx(path, args):
    check_zip(path)
    parts = [f"# {os.path.splitext(os.path.basename(path))[0]}"]
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        slides = []
        if "ppt/presentation.xml" in names:  # deck order, not file-name order
            rels = _rels(zf, "ppt/presentation.xml")
            pres = read_zip_xml(zf, "ppt/presentation.xml")
            for sld in pres.iter(f"{{{NS['p']}}}sldId"):
                target = rels.get(sld.get(f"{{{NS['r']}}}id"), ("", ""))[1]
                if target in names:
                    slides.append(target)
        if not slides:
            slides = sorted((n for n in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)), key=natural_key)
        for number, name in enumerate(slides, 1):
            root = read_zip_xml(zf, name)
            lines, title = [], None
            for shape in root.iter(f"{{{NS['p']}}}sp"):
                placeholder = shape.find(".//p:nvSpPr/p:nvPr/p:ph", NS)
                is_title = placeholder is not None and placeholder.get("type") in ("title", "ctrTitle")
                for para in shape.iter(f"{{{NS['a']}}}p"):
                    text = _pptx_text(para)
                    if not text:
                        continue
                    if is_title and title is None:
                        title = text
                    else:
                        level = para.find("a:pPr", NS)
                        indent = int(level.get("lvl", "0")) if level is not None else 0
                        lines.append("  " * indent + "- " + text)
            for table in root.iter(f"{{{NS['a']}}}tbl"):
                rows = [[" ".join(t for t in (_pptx_text(p) for p in cell.iter(f"{{{NS['a']}}}p")) if t)
                         for cell in tr.findall("a:tc", NS)] for tr in table.findall("a:tr", NS)]
                lines.append(md_table(rows, args.max_rows))
            parts.append(f"## Slide {number}" + (f": {title}" if title else ""))
            parts.extend(lines)
            notes = [target for kind, target in _rels(zf, name).values() if kind.endswith("/notesSlide")]
            if notes and notes[0] in names:
                texts = []
                for shape in read_zip_xml(zf, notes[0]).iter(f"{{{NS['p']}}}sp"):
                    placeholder = shape.find(".//p:nvSpPr/p:nvPr/p:ph", NS)
                    if placeholder is not None and placeholder.get("type") in ("sldNum", "dt", "hdr", "ftr", "sldImg"):
                        continue
                    texts += [t for t in (_pptx_text(p) for p in shape.iter(f"{{{NS['a']}}}p")) if t]
                note = " ".join(texts).strip()
                if note:
                    parts.append(f"Speaker notes: {note}")
    return "\n\n".join(parts)


def _col_index(ref, previous):
    m = re.match(r"[A-Z]+", ref or "")
    if not m:
        return previous + 1  # cells may omit their reference: they follow the previous one
    index = 0
    for ch in m.group(0):
        index = index * 26 + (ord(ch) - 64)
    return index


_XLSX_DATE_IDS = set(range(14, 23)) | {45, 46, 47}  # Excel's built-in date and time formats


def _xlsx_date_styles(zf):
    """{cell style index: (has date, has time)} for styles that show dates or times."""
    s = f"{{{NS['s']}}}"
    if "xl/styles.xml" not in zf.namelist():
        return {}
    styles = read_zip_xml(zf, "xl/styles.xml")
    custom = {}
    for fmt in styles.iter(f"{s}numFmt"):
        code = re.sub(r'"[^"]*"|\\.|\[[^\]]*\]', "", fmt.get("formatCode", "")).lower()  # quoted text, [Red], [$-409]
        if "[h]" in fmt.get("formatCode", "").lower() or "[m]" in fmt.get("formatCode", "").lower():
            continue  # elapsed time ([h]:mm) is a duration, not a clock time
        if re.search(r"[dmyhs]", code):
            custom[int(fmt.get("numFmtId", "-1"))] = ("d" in code or "y" in code, "h" in code or "s" in code)
    out = {}
    xfs = styles.find(f"{s}cellXfs")
    for index, xf in enumerate(xfs if xfs is not None else []):
        fid = int(xf.get("numFmtId", "0"))
        if fid in custom:
            out[index] = custom[fid]
        elif fid in _XLSX_DATE_IDS:
            out[index] = (fid in (14, 15, 16, 17, 22), fid in (18, 19, 20, 21, 22, 45, 46, 47))
    return out


def _xlsx_number(text, date_format, date1904):
    """A numeric cell as shown: ISO date/time for date styles, else up to 15 significant digits."""
    try:
        value = float(text)
    except (TypeError, ValueError):
        return text or ""
    if not math.isfinite(value):
        return text
    if date_format and 0 <= value < 2_958_466:  # Excel's last date is 9999-12-31
        import datetime
        has_date, has_time = date_format
        if date1904:
            base = datetime.datetime(1904, 1, 1)
        else:  # Excel counts 1900-02-29, which did not exist: serials below 60 start a day later
            base = datetime.datetime(1899, 12, 31) if value < 60 else datetime.datetime(1899, 12, 30)
        moment = base + datetime.timedelta(days=value)
        moment = moment.replace(microsecond=0) + datetime.timedelta(seconds=round(moment.microsecond / 1e6))
        if has_time and (value % 1 or not has_date):
            clock = moment.strftime("%H:%M:%S" if moment.second else "%H:%M")
            return clock if value < 1 and not has_date else f"{moment:%Y-%m-%d} {clock}"
        return f"{moment:%Y-%m-%d}"
    return f"{value:.15g}"


def convert_xlsx(path, args):
    check_zip(path)
    parts = [f"# {os.path.splitext(os.path.basename(path))[0]}"]
    s = f"{{{NS['s']}}}"
    with zipfile.ZipFile(path) as zf:
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            for si in read_zip_xml(zf, "xl/sharedStrings.xml").findall("s:si", NS):
                shared.append("".join(t.text or "" for t in si.iter(f"{s}t")))
        date_styles = _xlsx_date_styles(zf)
        workbook = read_zip_xml(zf, "xl/workbook.xml")
        pr = workbook.find(f"{s}workbookPr")
        date1904 = pr is not None and pr.get("date1904", "0").lower() in ("1", "true")
        rels = read_zip_xml(zf, "xl/_rels/workbook.xml.rels")
        targets = {r.get("Id"): r.get("Target") for r in rels.findall("rel:Relationship", NS)}
        for sheet in workbook.find("s:sheets", NS):
            target = targets.get(sheet.get(f"{{{NS['r']}}}id"), "")
            member = "xl/" + target.lstrip("/").replace("xl/", "") if not target.startswith("xl/") else target
            if member not in zf.namelist():
                continue
            rows, more = [], False
            with zf.open(member) as handle:  # stream: only --max-rows rows are kept, so stop reading there
                for _, row in ET.iterparse(handle, events=("end",)):
                    if row.tag != f"{s}row":
                        continue
                    if len(rows) > args.max_rows:
                        more = True
                        break
                    cells, col = {}, 0
                    for c in row.findall(f"{s}c"):
                        col = _col_index(c.get("r"), col)
                        kind, v = c.get("t"), c.find(f"{s}v")
                        if kind == "s" and v is not None:
                            value = shared[int(v.text)]
                        elif kind == "inlineStr":
                            value = "".join(t.text or "" for t in c.iter(f"{s}t"))
                        elif kind == "b" and v is not None:
                            value = "TRUE" if v.text == "1" else "FALSE"
                        elif kind in (None, "n") and v is not None:
                            value = _xlsx_number(v.text, date_styles.get(int(c.get("s", "0") or 0)), date1904)
                        else:  # formula strings, errors, ISO dates (t="d")
                            value = v.text if v is not None else ""
                        cells[col] = value
                    if cells:
                        rows.append([cells.get(i, "") for i in range(1, max(cells) + 1)])
                    row.clear()
            parts.append(f"## Sheet: {sheet.get('name')}")
            table = md_table(rows, args.max_rows, max_cols=getattr(args, "max_cols", None))
            if more and table:
                table += f"\n\n*More rows not shown (limit --max-rows {args.max_rows}).*"
            parts.append(table or "*empty sheet*")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# tables and GWAS summary statistics
# ---------------------------------------------------------------------------
GWAS_COLUMNS = {
    "snp": ("snp", "rsid", "variant_id", "markername", "id", "snpid", "variant"),
    "chr": ("chr", "chrom", "chromosome", "#chrom", "#chr", "hg19chrc"),
    "pos": ("bp", "pos", "position", "base_pair_location", "genpos", "bp_hg19"),
    "p": ("p", "pval", "p_value", "pvalue", "p-value", "p.value", "p_bolt_lmm", "log10p", "neglog10p", "mlogp"),
    "beta": ("beta", "b", "effect", "logor", "or"),
    "allele": ("effect_allele", "ea", "a1", "allele1", "alt", "tested_allele", "alt_allele"),
}
NEGLOG_P = ("log10p", "neglog10p", "mlogp")


def gwas_columns(header):
    lower = [h.strip().lower() for h in header]
    found = {}
    for key, names in GWAS_COLUMNS.items():
        for n in names:
            if n in lower:
                found[key] = lower.index(n)
                break
    return found if {"chr", "pos", "p"} <= set(found) else None


def sniff_delimiter(line):
    """Tab, comma or semicolon (Excel's CSV in EU locales, where the comma is the
    decimal mark) for delimited files, None when none is present."""
    if "\t" in line and line.count("\t") >= line.count(","):
        return "\t"
    if line.count(";") > line.count(","):
        return ";"
    return "," if "," in line else None


def _looks_tabular(lines, delim):
    """Most sample lines split into the same number (>= 2) of fields."""
    counts = [l.count(delim) for l in lines if l.strip()] if delim else []
    if len(counts) < 2:
        return bool(counts) and counts[0] >= 2
    mode = max(set(counts), key=counts.count)
    return mode >= 1 and counts.count(mode) >= 0.8 * len(counts)


def _neglog10(text):
    """-log10 of a p-value string, exact even below 1e-308 ('1e-400' -> 400)."""
    value = float(text)
    if value > 0:
        return -math.log10(value)
    if value == 0:  # underflowed: the strongest hits ('0' in PLINK output), as in _gwas_fast
        if "e" in text.lower():
            from decimal import Decimal, InvalidOperation
            try:
                d = Decimal(text.strip())
                return -float(d.log10()) if d > 0 else math.inf
            except InvalidOperation:
                return math.inf
        return math.inf
    return math.nan


def _fmt_p(neglog):
    if not math.isfinite(neglog):
        return "< 1e-300"
    exponent = math.floor(-neglog)
    mantissa = 10 ** (-neglog - exponent)
    if mantissa >= 9.995:
        mantissa, exponent = 1.0, exponent + 1
    return f"{mantissa:.2f}e{exponent:+03d}"


def _gwas_fast(path, header, delim, cols, neglog):
    """pyarrow's multi-threaded C++ CSV reader (gzip/bgzip handled natively) plus
    vectorised statistics: a few seconds for a 500 MB compressed file."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pacsv
    from scipy.stats import chi2 as chi2_dist
    names = {k: header[i].strip() for k, i in cols.items()}
    strings = {names[k]: pa.string() for k in ("chr", "snp", "allele") if k in names}
    reader = pacsv.open_csv(
        path, read_options=pacsv.ReadOptions(block_size=64 << 20),
        parse_options=pacsv.ParseOptions(delimiter=delim),
        convert_options=pacsv.ConvertOptions(
            include_columns=list(dict.fromkeys(names.values())), null_values=["NA", "nan", "NaN", "", ".", "-"],
            column_types={**strings, names["p"]: pa.float64(), names["pos"]: pa.float64()}))
    n, chisq, top, per_chr = 0, [], [], {}
    for batch in reader:
        pv = batch.column(names["p"]).to_numpy(zero_copy_only=False).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            nl = pv if neglog else np.where(pv > 0, -np.log10(pv), np.where(pv == 0, np.inf, np.nan))
        chrom = pc.fill_null(pc.replace_substring(batch.column(names["chr"]), "chr", ""), "?").to_numpy(
            zero_copy_only=False)
        pos = batch.column(names["pos"]).to_numpy(zero_copy_only=False)
        ok = ~np.isnan(nl) & (nl >= 0) & np.isfinite(pos)
        n += int(ok.sum())
        p = np.power(10.0, -np.minimum(nl[ok], 300.0))
        chisq.append(chi2_dist.isf(p, 1).astype(np.float32))
        values, counts = np.unique(chrom[ok], return_counts=True)
        for v, c in zip(values, counts):
            per_chr[v] = per_chr.get(v, 0) + int(c)
        sig = np.flatnonzero(ok & (nl > -math.log10(5e-8)))
        if len(sig):
            snps = batch.column(names["snp"]).to_pylist() if "snp" in names else None
            betas = batch.column(names["beta"]).to_pylist() if "beta" in names else None
            alleles = batch.column(names["allele"]).to_pylist() if "allele" in names else None
            for i in sig:
                top.append((float(nl[i]), chrom[i], int(pos[i]), snps[i] if snps else f"{chrom[i]}:{int(pos[i])}",
                            betas[i] if betas else "?", alleles[i] if alleles else None))
    all_chi = np.concatenate(chisq) if chisq else np.zeros(0)
    lam = float(np.median(all_chi) / 0.4549) if len(all_chi) else float("nan")
    return n, top, per_chr, lam


def _gwas_slow(path, delim, cols, neglog):
    """Forgiving line-by-line reader: ragged rows, odd missing-value codes, runs of spaces."""
    import array
    import numpy as np
    from scipy.stats import chi2 as chi2_dist
    n, top, per_chr, neglogs = 0, [], {}, array.array("d")
    decimal_comma = delim == ";"  # EU-locale CSV: '0,5' means 0.5
    with open_text(path) as handle:
        # each line is parsed on its own: quoted fields holding the delimiter ("A,B" gene lists) stay
        # whole, and a stray quote cannot swallow the lines after it
        next(handle, None)
        for line in handle:
            if delim:
                try:
                    row = next(csv.reader([line], delimiter=delim), [])
                except csv.Error:  # one malformed line: skip it
                    continue
            else:
                row = line.split()
            field = lambda key: row[cols[key]].strip('"')  # noqa: E731
            try:
                p_text, pos_text = field("p"), field("pos")
                if decimal_comma:
                    p_text, pos_text = p_text.replace(",", "."), pos_text.replace(",", ".")
                nl = float(p_text) if neglog else _neglog10(p_text)
                chrom, pos = field("chr").replace("chr", ""), int(float(pos_text))
            except (ValueError, IndexError, OverflowError):
                continue
            if math.isnan(nl) or nl < 0:
                continue
            n += 1
            neglogs.append(min(nl, 300.0))
            per_chr[chrom] = per_chr.get(chrom, 0) + 1
            if nl > -math.log10(5e-8):
                snp = field("snp") if "snp" in cols and cols["snp"] < len(row) else f"{chrom}:{pos}"
                beta = field("beta") if "beta" in cols and cols["beta"] < len(row) else "?"
                allele = field("allele") if "allele" in cols and cols["allele"] < len(row) else None
                top.append((nl, chrom, pos, snp, beta, allele))
    values = np.frombuffer(neglogs, dtype=np.float64)
    lam = float(np.median(chi2_dist.isf(np.power(10.0, -values), 1)) / 0.4549) if len(values) else float("nan")
    return n, top, per_chr, lam


def convert_gwas(path, args, header, delim, cols):
    """Stream a summary-statistics file and describe it in words (the rows
    themselves would teach a language model nothing useful)."""
    neglog = header[cols["p"]].strip().lower() in NEGLOG_P
    if delim:
        try:
            return _gwas_text(path, args, *_gwas_fast(path, header, delim, cols, neglog))
        except Exception:  # noqa: BLE001 - fall back to the slow, forgiving reader
            pass
    return _gwas_text(path, args, *_gwas_slow(path, delim, cols, neglog))


def _gwas_text(path, args, n, top, per_chr, lam):
    top.sort(key=lambda t: (-t[0], t[1], t[2]))
    loci, lead_positions = [], {}
    for hit in top:  # greedy 1 Mb clumping: a hit starts a locus unless a stronger lead lies within 1 Mb
        _, chrom, pos = hit[:3]
        leads = lead_positions.setdefault(chrom, [])
        k = bisect.bisect_left(leads, pos)
        if (k < len(leads) and leads[k] - pos < 1_000_000) or (k > 0 and pos - leads[k - 1] < 1_000_000):
            continue
        bisect.insort(leads, pos)
        loci.append(hit)
    name = re.sub(r"(\.(tsv|txt|csv|tab))?(\.gz)?$", "", os.path.basename(path), flags=re.IGNORECASE)
    lines = [f"# GWAS summary statistics: {name}", "",
             f"This file reports association results for {n:,} genetic variants on {len(per_chr)} chromosomes. "
             f"{len(top):,} variants reach genome-wide significance (p < 5e-8), forming {len(loci)} distinct loci "
             f"when variants within 1 Mb of a stronger signal are grouped. The genomic inflation factor lambda GC "
             f"is {lam:.3f}" + (" (median chi-square / 0.4549)." if math.isfinite(lam) else "."), ""]
    if loci:
        with_allele = any(h[5] for h in loci)
        header = ["rank", "variant", "chromosome", "position", "p-value", "effect"] + (["effect allele"]
                                                                                      if with_allele else [])
        rows = [[i + 1, snp, c, f"{pos:,}", _fmt_p(nl), beta] + ([allele or "?"] if with_allele else [])
                for i, (nl, c, pos, snp, beta, allele) in enumerate(loci[:args.max_rows])]
        lines += ["## Strongest independent loci", "", md_table([header] + rows, args.max_rows, total=len(loci))]
    lines += ["", "## Variants per chromosome", "",
              ", ".join(f"chromosome {c}: {k:,}" for c, k in sorted(per_chr.items(), key=lambda kv: natural_key(kv[0])))]
    return "\n".join(lines)


def _gz_size(path, limit):
    """Decompressed size of a .gz file, counted up to limit + 1 bytes (nothing is kept)."""
    size = 0
    with gzip.open(path, "rb") as handle:
        while size <= limit:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
    return size


def convert_table_or_text(path, args):
    ext = inner_ext(path)
    gz = path.lower().endswith(".gz")
    if gz and ext in NOT_TEXT_GZ:
        raise RuntimeError(f"not converted: a compressed {ext} file is data, not text")
    with open_text(path) as handle:
        # at most 1 MB per line: a huge file without newlines is not read whole here (the size check refuses it)
        sample = [line for line in (handle.readline(1 << 20) for _ in range(20)) if line]
    first = sample[0] if sample else ""
    delim = sniff_delimiter(first)
    if not args.no_gwas:
        header = next(csv.reader([first], delimiter=delim)) if delim else first.split()
        cols = gwas_columns(header)
        if cols and (delim or len(header) >= 3):
            return convert_gwas(path, args, header, delim, cols), "gwas"
    # commas in a .txt are nearly always prose; a .txt counts as a table only when consistently tab-separated
    if ext in (".csv", ".tsv", ".tab") or (delim == "\t" and _looks_tabular(sample, "\t")):
        delim = delim or ("\t" if ext in (".tsv", ".tab") else ",")
        rows, total = [], 0
        with open_text(path) as handle:
            for i, row in enumerate(csv.reader(handle, delimiter=delim)):
                if i <= args.max_rows:
                    rows.append(row)
                elif any(c.strip() for c in row):
                    total += 1
        title = f"# {os.path.splitext(os.path.basename(path))[0]}\n\n"
        data_rows = sum(1 for r in rows[1:] if any(str(c).strip() for c in r)) + total
        return title + md_table(rows, args.max_rows, total=data_rows, max_cols=getattr(args, "max_cols", None)), \
            "table"
    joined = "".join(sample)
    if joined.count("\x00") > max(10, len(joined) // 1000):
        raise RuntimeError("not converted: binary data, not text")
    size = _gz_size(path, MAX_TEXT_BYTES) if gz else os.path.getsize(path)  # a .gz is judged by what it holds
    if size > MAX_TEXT_BYTES:
        raise RuntimeError(f"not converted: {'over ' if gz else ''}{size / 2 ** 20:,.0f} MB of undelimited text "
                           "is not a document")
    return read_text(path), "text"


# ---------------------------------------------------------------------------
# HTML, notebooks, code
# ---------------------------------------------------------------------------
_BLOCK_TAGS = {"p", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "div", "section", "article", "header",
               "footer", "table", "blockquote", "pre", "dl", "dt", "dd", "figure", "figcaption", "ul", "ol", "hr"}


class _TextOnly(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self.skip += 1
        if tag in _BLOCK_TAGS:
            self.out.append("\n")
        elif tag in ("td", "th"):
            self.out.append(" | ")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript") and self.skip:
            self.skip -= 1
        if tag in _BLOCK_TAGS:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def convert_html(path, args):
    parser = _TextOnly()
    parser.feed(read_text(path))
    text = "".join(parser.out)
    text = re.sub(r"^[ \t]*\|\s*", "", text, flags=re.M)  # a row starts with its first cell, not a separator
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def _fence(body, lang):
    longest = max((len(m) for m in re.findall(r"`+", body)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{lang}\n{body.rstrip()}\n{fence}"


def convert_ipynb(path, args):
    notebook = json.loads(read_text(path))
    lang = ((notebook.get("metadata") or {}).get("kernelspec") or {}).get("language") or "python"
    parts = [f"# {os.path.basename(path)}"]
    for cell in notebook.get("cells", []):
        source = cell.get("source", "")
        source = "".join(source) if isinstance(source, list) else str(source)
        if not source.strip():
            continue
        if cell.get("cell_type") == "markdown":
            parts.append(source.strip())
        elif cell.get("cell_type") == "code":
            parts.append(_fence(source, lang))  # outputs (and their base64 images) are dropped
    return "\n\n".join(parts)


def convert_one(path, args, image_dir=None, info=None):
    """Return (markdown, type) or raise with a reason. For a PDF, `image_dir` and
    `info` are passed to convert_pdf."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return convert_pdf(path, args, image_dir, info), "pdf"
    if ext == ".docx":
        return convert_docx(path, args), "docx"
    if ext == ".pptx":
        return convert_pptx(path, args), "pptx"
    if ext == ".xlsx":
        return convert_xlsx(path, args), "xlsx"
    if ext in (".html", ".htm"):
        return convert_html(path, args), "html"
    if ext in (".md", ".markdown"):
        return read_text(path), "text"
    if ext == ".ipynb":
        return convert_ipynb(path, args), "ipynb"
    if ext in (".csv", ".tsv", ".txt", ".tab", ".gz"):
        return convert_table_or_text(path, args)
    if ext in CODE_EXT:
        return f"# {os.path.basename(path)}\n\n{_fence(read_text(path), CODE_EXT[ext])}", "code"
    if ext in IMAGE_EXT:
        raise RuntimeError("image: a text-only model cannot use pixels; keep its caption from the PDF/slide text "
                           "or run OCR / a vision model first")
    if ext in (".doc", ".ppt", ".xls"):
        raise RuntimeError("old binary Office format: save it as .docx/.pptx/.xlsx first")
    raise RuntimeError(f"unsupported file type {ext or '(none)'}")


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", nargs="?", help="file or folder to convert")
    p.add_argument("--out", help="output folder (default: next to each input)")
    p.add_argument("--recursive", action="store_true", help="include sub-folders")
    p.add_argument("--types", help="folder input: convert only these extensions, e.g. pdf or pdf,docx")
    p.add_argument("--by-type", action="store_true",
                   help="write into <out>/books (PDFs >= --book-pages pages), articles, slides, tables, codes")
    p.add_argument("--book-pages", type=int, default=150, help="PDFs with at least this many pages count as books")
    p.add_argument("--name-prefix", default="", help="prefix for output names, e.g. Coloc__ (avoids clashes)")
    p.add_argument("--workers", type=int, default=1, help="convert this many files in parallel (subprocesses)")
    p.add_argument("--timeout", type=int, default=3600,
                   help="seconds allowed per file run in its own process (every PDF, and every file when "
                        f"--workers > 1); a PDF gets at least {PDF_SECONDS_PER_PAGE} s per page "
                        f"({OCR_SECONDS_PER_PAGE} s with --ocr)")
    p.add_argument("--target", help=argparse.SUPPRESS)
    p.add_argument("--image-dir", help=argparse.SUPPRESS)
    p.add_argument("--overwrite", action="store_true", help="replace existing .md outputs")
    p.add_argument("--max-rows", type=int, default=200, help="table rows kept per table/sheet")
    p.add_argument("--max-cols", type=int, default=30,
                   help=".csv/.tsv/.xlsx: table columns kept (a wide matrix would be megabytes of numbers)")
    p.add_argument("--no-gwas", action="store_true", help="treat summary-statistics files as plain tables")
    p.add_argument("--extract-images", action="store_true", help="PDF only: also save embedded images")
    p.add_argument("--ocr", action="store_true",
                   help="scanned PDF pages (no text layer, page covered by an image): read them with Tesseract OCR "
                        "(slow: seconds per page); other pages keep their text layer")
    # options of the old pdf_to_markdown.py
    p.add_argument("--input-folder", "-i", help=argparse.SUPPRESS)
    p.add_argument("--input-name", "-n", help=argparse.SUPPRESS)
    p.add_argument("--output-folder", "-o", help=argparse.SUPPRESS)
    p.add_argument("--output-name", "-O", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.input_name:
        args.input = os.path.join(args.input_folder or ".", args.input_name)
        args.out = args.out or args.output_folder
    if not args.input:
        p.error("give a file or folder to convert")
    if not os.path.exists(args.input):
        p.error(f"not found: {args.input}")
    if args.by_type and not args.out and not args.target:
        p.error("--by-type needs --out (the folder that receives books, articles, slides, tables, codes)")
    if not 1 <= args.max_rows <= 100_000:
        p.error("--max-rows must be between 1 and 100000")
    if not 1 <= args.max_cols <= 100_000:
        p.error("--max-cols must be between 1 and 100000")
    if not 1 <= args.workers <= 16:
        p.error("--workers must be between 1 and 16")
    return args


def pdf_pages(path):
    try:
        import pymupdf
        with pymupdf.open(path) as document:
            return document.page_count
    except Exception:  # noqa: BLE001
        return 0


def plan_target(path, args, n_files):
    """Output path decided before converting, so existing outputs are skipped cheaply."""
    base = os.path.basename(path)
    base = re.sub(r"(\.gz)$", "", base, flags=re.IGNORECASE)
    base = args.name_prefix + os.path.splitext(base)[0]
    folder = args.out or os.path.dirname(os.path.abspath(path))
    if args.by_type and args.out:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            kind = "book" if pdf_pages(path) >= args.book_pages else "article"
        elif ext in CODE_EXT or ext == ".ipynb":
            kind = "code"
        elif ext in (".docx", ".pptx", ".xlsx"):
            kind = ext[1:]
        elif ext in (".html", ".htm"):
            kind = "html"
        elif inner_ext(path) in (".csv", ".tsv", ".tab") or ext == ".gz":
            kind = "table"
        else:
            kind = "text"
        folder = os.path.join(args.out, TYPE_FOLDER.get(kind, "articles"))
    name = args.output_name if (n_files == 1 and args.output_name) else base + ".md"
    return os.path.join(folder, name)


def _norm(path):
    return os.path.normcase(os.path.abspath(path))  # Windows names ignore case


def manifest_file(args):
    """Where the manifest of this batch lives: the output folder, or the input folder
    when outputs go next to their inputs; None for a single file without --out."""
    if args.out:
        return os.path.join(args.out, MANIFEST_NAME)
    return os.path.join(args.input, MANIFEST_NAME) if os.path.isdir(args.input) else None


def load_manifest(path):
    """{normalised output path: (output path, input path)} recorded by earlier runs;
    records whose input no longer exists are dropped (their names are free again)."""
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            outputs = json.load(handle).get("outputs") or {}
    except (OSError, ValueError, AttributeError):
        return {}
    base = os.path.dirname(os.path.abspath(path))
    full = lambda p: os.path.abspath(p if os.path.isabs(p) else os.path.join(base, *p.split("/")))  # noqa: E731
    owners = {}
    for target, source in outputs.items():
        if isinstance(target, str) and isinstance(source, str) and os.path.exists(full(source)):
            owners[_norm(full(target))] = (full(target), full(source))
    return owners


def save_manifest(path, owners):
    """Write {output: input} (paths relative to the manifest's folder when possible),
    keeping only outputs that exist."""
    base = os.path.dirname(os.path.abspath(path))

    def rel(p):
        try:
            return os.path.relpath(p, base).replace(os.sep, "/")
        except ValueError:  # another drive (Windows)
            return p
    outputs = {rel(t): rel(s) for t, s in sorted(owners.values()) if os.path.exists(t)}
    os.makedirs(base, exist_ok=True)
    with open(path + ".part", "w", encoding="utf-8", newline="\n") as handle:
        json.dump({"note": "which input produced each output of convert_to_markdown.py", "outputs": outputs},
                  handle, indent=1, ensure_ascii=False)
    os.replace(path + ".part", path)


def _name_candidates(path, args, n_files):
    """Output names for one input, best first: its plain name, then (for a clash) its
    source extension (notes.docx.md, not for Markdown), its folder name (media__p2.md)
    and a number."""
    target = plan_target(path, args, n_files)
    stem = target[:-3] if target.lower().endswith(".md") else target
    ext = os.path.basename(path).lower().replace(".gz", "").rsplit(".", 1)[-1]
    parent = re.sub(r"[^\w.-]+", "_", os.path.basename(os.path.dirname(os.path.abspath(path)))) or "folder"
    candidates = [target] + ([] if _is_md(path) else [f"{stem}.{ext}.md"]) + [f"{stem}__{parent}.md"]
    return candidates + [f"{stem}__{k}.md" for k in range(2, 1000)]


def _is_md(path):
    return path.lower().endswith((".md", ".markdown"))


def legacy_outputs(files, args):
    """Outputs of runs made before manifests existed that now sit among the inputs
    (no --out, or --out inside the input tree): notes.docx.md, media__p2.md or
    paper__2.md next to the input whose clash name it is. A plain paper.md cannot be
    told from the user's own notes, so it stays an input."""
    names = {_norm(f): f for f in files if _is_md(f)}
    owners = {}
    for path in files:
        if _is_md(path):
            continue
        for candidate in _name_candidates(path, args, len(files))[1:22]:
            key = _norm(candidate)
            if key in names and key not in owners:
                owners[key] = (names[key], os.path.abspath(path))
    return owners


def plan_targets(files, args, owners=None):
    """[(path, target)] in input order, where no two inputs share an output and no
    output is another input. An input keeps the output an earlier run recorded for
    it (`owners`, from the manifest); outputs recorded for other inputs are taken.
    Markdown inputs claim their plain names first, then the others in sorted order;
    a later claimant gets its source extension (notes.docx.md) or, for Markdown,
    its folder name (media__p2.md)."""
    owners = owners or {}
    inputs = {_norm(f) for f in files}
    recorded = {_norm(source): target for target, source in owners.values()}
    # every recorded name is reserved, even one whose output was deleted: its owner makes it again
    claimed = set(owners)
    chosen = {}
    for path in sorted(files, key=lambda f: not _is_md(f)):  # stable: keeps sorted order within each group
        candidates = _name_candidates(path, args, len(files))
        if _norm(candidates[0]) == _norm(path):
            chosen[path] = candidates[0]  # reported as "would overwrite the input"
            continue
        mine = recorded.get(_norm(path))
        if mine and _norm(mine) in {_norm(c) for c in candidates[:22]} and _norm(mine) not in inputs:
            target = mine  # the name an earlier run gave it (same folder and options)
        else:
            target = next((c for c in candidates if _norm(c) not in claimed and _norm(c) not in inputs),
                          candidates[-1])
        claimed.add(_norm(target))
        chosen[path] = target
    return [(path, chosen[path]) for path in files]


def _lower_priority():
    """Below-normal priority, so the PC stays responsive. Never raises it: a worker
    of a process set to idle stays idle (it inherits idle from its parent)."""
    try:
        import psutil
        proc = psutil.Process()
        if sys.platform == "win32":
            if proc.nice() != psutil.IDLE_PRIORITY_CLASS:
                proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        elif proc.nice() < 10:
            proc.nice(10)
    except Exception:  # noqa: BLE001
        pass


def _pdf_page_count(path):
    try:
        import pymupdf
        with pymupdf.open(path) as doc:
            return max(doc.page_count, 1)
    except Exception:  # noqa: BLE001
        return 1


def find_tessdata():
    """Folder of Tesseract's language files (TESSDATA_PREFIX, the tesseract
    program's folder, or the usual install places), or None."""
    import shutil
    candidates = [os.environ.get("TESSDATA_PREFIX", "")]
    exe = shutil.which("tesseract")
    if exe:
        candidates.append(os.path.join(os.path.dirname(exe), "tessdata"))
    candidates += [r"C:\Program Files\Tesseract-OCR\tessdata", r"C:\Program Files (x86)\Tesseract-OCR\tessdata",
                   "/opt/homebrew/share/tessdata", "/usr/local/share/tessdata", "/usr/share/tesseract-ocr/5/tessdata",
                   "/usr/share/tesseract-ocr/4.00/tessdata", "/usr/share/tessdata"]
    for folder in candidates:
        if folder and os.path.isfile(os.path.join(folder, "eng.traineddata")):
            return folder
    return None


def _tessdata_or_fail():
    tessdata = find_tessdata()
    if tessdata is None:
        raise RuntimeError("OCR needs Tesseract's language files: install Tesseract OCR "
                           "(Windows: winget install UB-Mannheim.TesseractOCR) or set TESSDATA_PREFIX")
    return tessdata


def _ocr_page(page, tessdata, dpi=300):
    """Text of one page rendered at `dpi` and read by Tesseract (built into PyMuPDF;
    only the language files are needed), one paragraph per text block."""
    textpage = page.get_textpage_ocr(language="eng", dpi=dpi, full=True, tessdata=tessdata)
    blocks = [_reflow(b[4]) for b in page.get_text("blocks", textpage=textpage, sort=True) if b[6] == 0]
    return "\n\n".join(b for b in blocks if b)


def ocr_pdf(path, dpi=300):
    """Text of a scanned PDF: every page read by OCR, then running headers and
    page numbers removed as for ordinary PDFs."""
    import pymupdf
    tessdata = _tessdata_or_fail()
    with pymupdf.open(path) as doc:
        pages = [_ocr_page(page, tessdata, dpi) for page in doc]
    return "\n\n".join(p for p in strip_page_furniture(pages) if p.strip())


def _reflow(block):
    """One OCR text block as a paragraph: lines joined, words split by a line-end hyphen rejoined."""
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    out = ""
    for ln in lines:
        if out.endswith("-") and len(out) > 1 and out[-2].isalpha() and ln[:1].islower():
            out = out[:-1] + ln
        else:
            out = f"{out} {ln}" if out else ln
    return out


def convert_to_file(path, target, args):
    """Convert one file and write it; returns (status, detail)."""
    if os.path.exists(target) and not getattr(args, "overwrite", False):
        # written since this batch was planned (for example by a separate --ocr run): never replace it
        return "skipped", f"{target} already exists (use --overwrite)"
    info = {}
    image_dir = os.path.splitext(target)[0] + "_images" if getattr(args, "extract_images", False) else None
    text, kind = convert_one(path, args, image_dir, info)  # images go next to the output, named like it
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    ocr_note, ocr_tried = "", False
    pages, scans, ocr_pages = info.get("pages", 0), info.get("scan_pages", 0), info.get("ocr_pages", 0)
    if kind == "pdf" and scans:
        if info.get("ocr_unavailable"):
            ocr_note = f" ({scans} of {pages} pages look scanned; OCR unavailable: install Tesseract)"
        elif getattr(args, "ocr", False):
            ocr_tried = True
            if ocr_pages:
                ocr_note = (" (read with OCR)" if ocr_pages >= pages
                            else f" ({ocr_pages} of {pages} pages read with OCR)")
        elif scans >= pages:
            ocr_note = " (text layer nearly empty: a scan? rerun with --ocr --overwrite)"
        else:
            ocr_note = f" ({scans} of {pages} pages look scanned: rerun with --ocr --overwrite)"
    if kind == "pdf" and len(text) < MIN_TEXT_CHARS:
        if info.get("ocr_unavailable"):
            return "skipped", "no text layer and OCR is unavailable (install Tesseract); nothing written"
        return "skipped", ("OCR found no text; nothing written" if ocr_tried
                           else "no text layer (a scanned PDF? rerun with --ocr); nothing written")
    if not text:
        return "skipped", "no text found; nothing written"
    os.makedirs(os.path.dirname(target), exist_ok=True)
    tmp = target + ".part"
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text + "\n")
    os.replace(tmp, target)
    return f"converted ({kind})", f"{target} ({len(text) + 1:,} characters){ocr_note}"


def _reason(exc):
    return str(exc).splitlines()[0][:200] if str(exc) else type(exc).__name__


def _run_child(cmd, timeout):
    """Run one worker process; returns (return code or None on timeout, stdout, stderr
    lines). OCR progress lines from the worker are shown as they come."""
    import subprocess
    import threading
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                            errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    out, err = [], []

    def pump_out():
        out.extend(proc.stdout)

    def pump_err():
        for line in proc.stderr:
            if line.startswith("  OCR "):
                print(line.rstrip(), flush=True)
            else:
                err.append(line.rstrip())
                del err[:-50]
    threads = [threading.Thread(target=f, daemon=True) for f in (pump_out, pump_err)]
    for t in threads:
        t.start()
    try:
        code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        code = None
    for t in threads:
        t.join(timeout=10)
    return code, "".join(out), [line for line in err if line.strip()]


_RERUN_NOTE = " [later run: output exists (use --overwrite)]"


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")  # the parent reads a worker's as UTF-8
    except (AttributeError, OSError):
        pass
    args = parse_args()
    _lower_priority()
    if args.target:  # internal: one file converted by a worker subprocess
        try:
            status, detail = convert_to_file(args.input, args.target, args)
        except Exception as exc:  # noqa: BLE001 - reported by the parent like an in-process failure
            status, detail = "skipped", _reason(exc)
        print(f"{status}\t{detail}")
        return
    manifest = manifest_file(args)
    owners = load_manifest(manifest)
    if os.path.isdir(args.input):
        walker = os.walk(args.input) if args.recursive else [(args.input, [], os.listdir(args.input))]
        files = sorted(os.path.join(d, f) for d, _, names in walker for f in names
                       if os.path.isfile(os.path.join(d, f)))
        if args.types:
            wanted = {"." + t.strip().lower().lstrip(".") for t in args.types.split(",") if t.strip()}
            files = [f for f in files if os.path.splitext(f)[1].lower() in wanted or inner_ext(f) in wanted]
        # the converter's own files are not inputs: outputs of earlier runs (recorded in the manifest),
        # the manifest and reports, unfinished '.part' files, and anything under --out inside the input tree
        out_inside = args.out and _norm(args.out) != _norm(args.input) and \
            _norm(args.out).startswith(_norm(args.input).rstrip("\\/") + os.sep)
        own = {_norm(manifest)} if manifest else set()

        def is_own(f):
            key, name = _norm(f), os.path.basename(f).lower()
            return (key in own or (key in owners and _norm(owners[key][1]) != key) or name.endswith(".md.part")
                    or (args.out and re.fullmatch(r"conversion_report.*\.tsv", name)
                        and _norm(os.path.dirname(f)) == _norm(args.out))
                    or (out_inside and key.startswith(_norm(args.out) + os.sep)))
        files = [f for f in files if not is_own(f)]
        if not os.path.isfile(manifest or ""):  # outputs of runs made before manifests existed
            legacy = legacy_outputs(files, args)
            owners.update(legacy)
            files = [f for f in files if _norm(f) not in legacy]
    else:
        files = [args.input]
    planned = plan_targets(files, args, owners)

    report_path, merged = None, {}
    if args.out and len(files) > 1:
        suffix = ("_" + args.name_prefix.strip("_")) if args.name_prefix else ""
        report_path = os.path.join(args.out, f"conversion_report{suffix}.tsv")
        # merge into an existing report (a rerun, say with --ocr, updates its rows and keeps all others)
        if os.path.isfile(report_path):
            with open(report_path, encoding="utf-8") as handle:
                next(handle, None)
                for line in handle:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 3:
                        merged[parts[0]] = parts[:3]

    def record(path, status, detail):
        row = [str(x).replace("\t", " ") for x in (path, status, detail)]
        old = merged.get(row[0])
        if status == "skipped" and "exists (use --overwrite)" in detail and old and old[1].startswith("converted"):
            # still the output of an earlier run: keep what that run found (kind, size, scan notes)
            row = [row[0], old[1], old[2].replace(_RERUN_NOTE, "") + _RERUN_NOTE]
        merged[row[0]] = row

    def save(final=False):
        if report_path:
            os.makedirs(args.out, exist_ok=True)
            with open(report_path + ".part", "w", encoding="utf-8", newline="\n") as handle:
                handle.write("input\tstatus\tdetail\n")
                for row in merged.values():
                    handle.write("\t".join(row) + "\n")
            os.replace(report_path + ".part", report_path)
        if manifest and (final or owners_changed[0]):
            for path, target in planned:
                key = _norm(target)
                if key == _norm(path) or key in owners or not os.path.exists(target):
                    continue
                # an unrecorded output is this input's only if this run (or one in the report) made it from
                # this input: an output merely skipped as 'exists' may have come from another input
                row = merged.get(str(path).replace("\t", " "))
                if key in made or (row and row[1].startswith("converted") and row[2].startswith(target + " ")):
                    owners[key] = (os.path.abspath(target), os.path.abspath(path))
            for key, value in load_manifest(manifest).items():  # keep what a parallel run into --out recorded
                owners.setdefault(key, value)
            if owners or os.path.isfile(manifest):
                save_manifest(manifest, owners)
            owners_changed[0] = False
    owners_changed, made = [True], set()
    target_of = dict(planned)

    todo = []
    for path, target in planned:
        if _norm(target) == _norm(path):
            record(path, "skipped", "output would overwrite the input")
        elif os.path.exists(target) and not args.overwrite:
            record(path, "skipped", f"{target} exists (use --overwrite)")
        else:
            todo.append((path, target))
    print(f"{len(files)} files: {len(todo)} to convert, {len(files) - len(todo)} already done or skipped "
          f"({args.workers} worker{'s' if args.workers > 1 else ''})", flush=True)

    def run_one(item):
        path, target = item
        # a PDF always runs in its own process: a MuPDF crash or a stuck page then costs one file, not the batch
        if args.workers == 1 and not path.lower().endswith(".pdf"):
            try:
                return path, *convert_to_file(path, target, args)
            except Exception as exc:  # noqa: BLE001 - report every failure and continue
                return path, "skipped", _reason(exc)
        cmd = [sys.executable, os.path.abspath(__file__), path, "--target", target, "--max-rows", str(args.max_rows),
               "--max-cols", str(args.max_cols)]
        if args.no_gwas:
            cmd.append("--no-gwas")
        if getattr(args, "ocr", False):
            cmd.append("--ocr")
        if args.overwrite:
            cmd.append("--overwrite")
        if args.extract_images:
            cmd.append("--extract-images")
        timeout = args.timeout
        if path.lower().endswith(".pdf"):  # a long book needs longer (OCR takes seconds per page)
            timeout = max(timeout, _pdf_page_count(path) * (OCR_SECONDS_PER_PAGE if getattr(args, "ocr", False)
                                                             else PDF_SECONDS_PER_PAGE))
        code, stdout, stderr = _run_child(cmd, timeout)
        if code is None:
            return path, "skipped", f"timed out after {timeout} s (raise --timeout to allow longer)"
        lines = [line for line in stdout.strip().splitlines() if "\t" in line]
        if code == 0 and lines:
            status, detail = lines[-1].split("\t", 1)
            return path, status, detail
        return path, "skipped", (stderr[-1][:200] if stderr else f"the converter crashed on this file (exit code "
                                                                  f"{code})")

    import time
    done, last_save = 0, time.monotonic()
    if args.workers == 1:
        results = map(run_one, todo)
    else:
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=args.workers)
        results = pool.map(run_one, todo)
    try:
        for i, (path, status, detail) in enumerate(results, 1):
            record(path, status, detail)
            done += status.startswith("converted")
            if status.startswith("converted"):
                owners_changed[0] = True
                made.add(_norm(target_of[path]))
            print(f"[{i}/{len(todo)}] {status:<22} {os.path.basename(path)} -> {detail}", flush=True)
            if time.monotonic() - last_save > 30:  # a killed run still leaves its report and manifest
                save()
                last_save = time.monotonic()
    finally:
        save(final=True)
    skipped = len(files) - done
    print(f"\n{done} of {len(files)} files converted; {skipped} skipped or already present"
          + (f" (details: {report_path})" if report_path else ""))


if __name__ == "__main__":
    main()
