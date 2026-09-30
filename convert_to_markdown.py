r"""Convert documents into Markdown text for data_prep.py (replaces pdf_to_markdown.py).

A text-only language model can only learn from text, so every source type is
turned into readable Markdown here, before tokenization:

  .pdf                 pymupdf4llm (headings, lists, tables where detected); plot-heavy pages as plain
                       text; running headers, footers, page numbers and per-page notices removed
  .docx                paragraphs (headings as #), content controls, tables as Markdown tables
  .pptx                slides in deck order: titles, bullet text, tables and speaker notes (no extra package)
  .xlsx                each sheet as a Markdown table, first --max-rows rows (no extra package)
  .csv / .tsv / .txt   tables become Markdown tables; plain text is copied unchanged
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


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def md_table(rows, max_rows, total=None):
    """Markdown table of the first max_rows data rows. `total` is the number of
    data rows when known; otherwise it is taken from `rows`."""
    rows = [["" if c is None else str(c).replace("|", "\\|").replace("\n", " ").strip() for c in r] for r in rows]
    rows = [r for r in rows if any(c for c in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    shown = rows[: max_rows + 1]
    out = ["| " + " | ".join(shown[0]) + " |", "|" + "---|" * width]
    out += ["| " + " | ".join(r) + " |" for r in shown[1:]]
    data_rows = len(rows) - 1 if total is None else total
    if data_rows > len(shown) - 1:
        hidden = data_rows - (len(shown) - 1)
        out.append(f"\n*{hidden:,} more rows not shown (limit --max-rows {max_rows}).*" if total is not None or
                   len(rows) > len(shown) + 1 else f"\n*More rows not shown (limit --max-rows {max_rows}).*")
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
    """utf-8-sig when the first MB decodes as UTF-8 (a BOM is dropped), else Windows-1252."""
    raw = gzip.open(path, "rb") if path.lower().endswith(".gz") else open(path, "rb")
    with raw:
        sample = raw.read(1 << 20)
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError as exc:
        if exc.start < len(sample) - 4:  # not just a character cut at the end of the sample
            return "cp1252"
    return "utf-8-sig"


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


def convert_pdf(path, args, image_dir=None):
    try:
        import pymupdf
        import pymupdf4llm
    except ImportError:
        raise RuntimeError("PDF conversion needs pymupdf4llm: python -m pip install pymupdf4llm")
    kwargs = {}
    if args.extract_images:
        folder = image_dir or args.out or os.path.dirname(os.path.abspath(path))
        image_dir = os.path.join(folder, os.path.splitext(os.path.basename(path))[0] + "_images")
        os.makedirs(image_dir, exist_ok=True)
        kwargs.update(write_images=True, image_path=image_dir, image_format="png")
    with pymupdf.open(path) as doc:
        heavy = {i for i in range(doc.page_count) if _heavy_page(doc[i])}
        pages, run = [], []
        for i in range(doc.page_count + 1):
            if i < doc.page_count and i not in heavy:
                run.append(i)
                continue
            if run:  # consecutive ordinary pages keep pymupdf4llm's headings, lists and tables
                chunks = pymupdf4llm.to_markdown(doc, pages=run, page_chunks=True, **kwargs)
                pages.extend(chunk["text"] for chunk in chunks)
                run = []
            if i < doc.page_count:
                pages.append(doc[i].get_text("text"))
    return "\n\n".join(p.strip() for p in strip_page_furniture(pages) if p.strip())


_PAGE_NUMBER = re.compile(r"(page\s*)?(\d{1,4}|[ivxlcdm]{1,7})(\s*(of|/)\s*\d{1,4})?", re.I)
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


def strip_page_furniture(pages, edge=4):
    """Remove running headers, footers, page numbers and repeated notices.

    Only the first and last `edge` non-empty lines of each page are candidates.
    A number at either end of a line is ignored only when it behaves like a page
    number (it steps with the page), so 'Supplementary Table 3' is kept. A
    candidate is removed if it is a page number (a bare number that steps with
    the page, or a roman numeral); if it is in capitals (like a book's running
    section title) and sits at a page edge on 2 or more pages; or if it has 2+
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
    # 1. which "text + page number" lines and bare numbers behave like page numbering
    numbered, bare = {}, []
    for page, (lines, idx) in enumerate(zip(split, edges)):
        for i in idx:
            raw = lines[i].strip()
            plain = re.sub(r"[*_`#>]", "", raw).strip()
            if plain.isdigit() and len(plain) <= 4:
                bare.append((page, int(plain)))
                continue
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
            if _PAGE_NUMBER.fullmatch(plain):  # roman numerals and "page 3 of 9"
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
def convert_docx(path, args):
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    check_zip(path)
    document = docx.Document(path)
    w = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    parts = []

    def paragraph(p):
        text = p.text.strip()
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
                    cells.append(cell.text)
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


def convert_xlsx(path, args):
    check_zip(path)
    parts = [f"# {os.path.splitext(os.path.basename(path))[0]}"]
    s = f"{{{NS['s']}}}"
    with zipfile.ZipFile(path) as zf:
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            for si in read_zip_xml(zf, "xl/sharedStrings.xml").findall("s:si", NS):
                shared.append("".join(t.text or "" for t in si.iter(f"{s}t")))
        workbook = read_zip_xml(zf, "xl/workbook.xml")
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
                        else:
                            value = v.text if v is not None else ""
                        cells[col] = value
                    if cells:
                        rows.append([cells.get(i, "") for i in range(1, max(cells) + 1)])
                    row.clear()
            parts.append(f"## Sheet: {sheet.get('name')}")
            table = md_table(rows, args.max_rows)
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
    """Tab or comma for delimited files, None when neither is present."""
    if "\t" in line and line.count("\t") >= line.count(","):
        return "\t"
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
    if value == 0 and ("e" in text.lower()):
        from decimal import Decimal, InvalidOperation
        try:
            d = Decimal(text.strip())
            return -float(d.log10()) if d > 0 else math.inf
        except InvalidOperation:
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
    with open_text(path) as handle:
        rows = csv.reader(handle, delimiter=delim) if delim else (line.split() for line in handle)
        next(rows, None)
        for row in rows:
            try:
                nl = float(row[cols["p"]]) if neglog else _neglog10(row[cols["p"]])
                chrom, pos = row[cols["chr"]].replace("chr", ""), int(float(row[cols["pos"]]))
            except (ValueError, IndexError, OverflowError):
                continue
            if math.isnan(nl) or nl < 0:
                continue
            n += 1
            neglogs.append(min(nl, 300.0))
            per_chr[chrom] = per_chr.get(chrom, 0) + 1
            if nl > -math.log10(5e-8):
                snp = row[cols["snp"]] if "snp" in cols and cols["snp"] < len(row) else f"{chrom}:{pos}"
                beta = row[cols["beta"]] if "beta" in cols and cols["beta"] < len(row) else "?"
                allele = row[cols["allele"]] if "allele" in cols and cols["allele"] < len(row) else None
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


def convert_table_or_text(path, args):
    with open_text(path) as handle:
        sample = [line for _, line in zip(range(20), handle)]
    first = sample[0] if sample else ""
    delim = sniff_delimiter(first)
    ext = inner_ext(path)
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
        return title + md_table(rows, args.max_rows, total=data_rows), "table"
    size = os.path.getsize(path)
    if size > MAX_TEXT_BYTES:
        raise RuntimeError(f"not converted: {size / 2 ** 20:,.0f} MB of undelimited text is not a document")
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


def convert_one(path, args, image_dir=None):
    """Return (markdown, type) or raise with a reason."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return convert_pdf(path, args, image_dir), "pdf"
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
    p.add_argument("--timeout", type=int, default=3600, help="seconds allowed per file when --workers > 1")
    p.add_argument("--target", help=argparse.SUPPRESS)
    p.add_argument("--image-dir", help=argparse.SUPPRESS)
    p.add_argument("--overwrite", action="store_true", help="replace existing .md outputs")
    p.add_argument("--max-rows", type=int, default=200, help="table rows kept per table/sheet")
    p.add_argument("--no-gwas", action="store_true", help="treat summary-statistics files as plain tables")
    p.add_argument("--extract-images", action="store_true", help="PDF only: also save embedded images")
    p.add_argument("--ocr", action="store_true",
                   help="PDFs without a text layer (scans): read the page images with Tesseract OCR (slow: seconds "
                        "per page)")
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


def plan_targets(files, args):
    """[(path, target)] in input order, where no two inputs share an output and no
    output is another input. Markdown inputs claim their plain names first, then
    the others in sorted order; a later claimant gets its source extension
    (notes.docx.md) or, for Markdown, its folder name (media__p2.md)."""
    norm = lambda p: os.path.normcase(os.path.abspath(p))  # noqa: E731 - Windows names ignore case
    inputs = {norm(f) for f in files}
    claimed, chosen = set(), {}
    is_md = lambda f: f.lower().endswith((".md", ".markdown"))  # noqa: E731
    for path in sorted(files, key=lambda f: not is_md(f)):  # stable: keeps sorted order within each group
        target = plan_target(path, args, len(files))
        if norm(target) == norm(path):
            chosen[path] = target  # reported as "would overwrite the input"
            continue
        stem = target[:-3] if target.lower().endswith(".md") else target
        ext = os.path.basename(path).lower().replace(".gz", "").rsplit(".", 1)[-1]
        parent = re.sub(r"[^\w.-]+", "_", os.path.basename(os.path.dirname(os.path.abspath(path)))) or "folder"
        candidates = [target] + ([] if is_md(path) else [f"{stem}.{ext}.md"]) + [f"{stem}__{parent}.md"]
        candidates += [f"{stem}__{k}.md" for k in range(2, 1000)]
        for candidate in candidates:
            if norm(candidate) not in claimed and norm(candidate) not in inputs:
                target = candidate
                break
        claimed.add(norm(target))
        chosen[path] = target
    return [(path, chosen[path]) for path in files]


def _lower_priority():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if sys.platform == "win32" else 10)
    except Exception:  # noqa: BLE001
        pass


SCAN_CHARS_PER_PAGE = 200   # a page of text has 2,000-4,000 characters; fewer than this on average means a scan


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


def ocr_pdf(path, dpi=300):
    """Text of a scanned PDF: each page is rendered at `dpi` and read by
    Tesseract (built into PyMuPDF; only the language files are needed), then
    running headers and page numbers are removed as for ordinary PDFs."""
    import pymupdf
    tessdata = find_tessdata()
    if tessdata is None:
        raise RuntimeError("OCR needs Tesseract's language files: install Tesseract OCR "
                           "(Windows: winget install UB-Mannheim.TesseractOCR) or set TESSDATA_PREFIX")
    pages = []
    with pymupdf.open(path) as doc:
        for page in doc:
            textpage = page.get_textpage_ocr(language="eng", dpi=dpi, full=True, tessdata=tessdata)
            blocks = [_reflow(b[4]) for b in page.get_text("blocks", textpage=textpage, sort=True) if b[6] == 0]
            pages.append("\n\n".join(b for b in blocks if b))
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
    text, kind = convert_one(path, args, getattr(args, "image_dir", None))
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    ocr_note = ""
    if kind == "pdf":
        pages = _pdf_page_count(path)
        if len(text) < SCAN_CHARS_PER_PAGE * pages:  # a nearly empty text layer: a scan (maybe with a cover line)
            if getattr(args, "ocr", False):
                ocr_text = ocr_pdf(path).strip()
                if len(ocr_text) > 2 * len(text):
                    text, ocr_note = ocr_text, " (read with OCR)"
            else:
                ocr_note = " (text layer nearly empty: a scan? rerun with --ocr --overwrite)"
    if kind == "pdf" and len(text) < MIN_TEXT_CHARS:
        return "skipped", ("no text layer (a scanned PDF? rerun with --ocr); nothing written" if not ocr_note
                           else "OCR found no text; nothing written")
    if not text:
        return "skipped", "no text found; nothing written"
    os.makedirs(os.path.dirname(target), exist_ok=True)
    tmp = target + ".part"
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text + "\n")
    os.replace(tmp, target)
    return f"converted ({kind})", f"{target} ({len(text) + 1:,} characters){ocr_note}"


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    args = parse_args()
    _lower_priority()
    if args.target:  # internal: one file converted by a worker subprocess
        status, detail = convert_to_file(args.input, args.target, args)
        print(f"{status}\t{detail}")
        return
    if os.path.isdir(args.input):
        walker = os.walk(args.input) if args.recursive else [(args.input, [], os.listdir(args.input))]
        files = sorted(os.path.join(d, f) for d, _, names in walker for f in names
                       if os.path.isfile(os.path.join(d, f)))
        if args.types:
            wanted = {"." + t.strip().lower().lstrip(".") for t in args.types.split(",") if t.strip()}
            files = [f for f in files if os.path.splitext(f)[1].lower() in wanted or inner_ext(f) in wanted]
    else:
        files = [args.input]
    report, todo = [], []
    for path, target in plan_targets(files, args):
        if os.path.normcase(os.path.abspath(target)) == os.path.normcase(os.path.abspath(path)):
            report.append((path, "skipped", "output would overwrite the input"))
        elif os.path.exists(target) and not args.overwrite:
            report.append((path, "skipped", f"{target} exists (use --overwrite)"))
        else:
            todo.append((path, target))
    print(f"{len(files)} files: {len(todo)} to convert, {len(files) - len(todo)} already done or skipped "
          f"({args.workers} worker{'s' if args.workers > 1 else ''})", flush=True)

    def run_one(item):
        path, target = item
        if args.workers == 1:
            try:
                return path, *convert_to_file(path, target, args)
            except Exception as exc:  # noqa: BLE001 - report every failure and continue
                return path, "skipped", str(exc).splitlines()[0][:200] if str(exc) else type(exc).__name__
        import subprocess
        cmd = [sys.executable, os.path.abspath(__file__), path, "--target", target, "--max-rows", str(args.max_rows)]
        if args.no_gwas:
            cmd.append("--no-gwas")
        if getattr(args, "ocr", False):
            cmd.append("--ocr")
        if args.extract_images:
            cmd += ["--extract-images", "--image-dir", args.out or os.path.dirname(target)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                  timeout=args.timeout)
        except subprocess.TimeoutExpired:
            return path, "skipped", f"timed out after {args.timeout} s"
        if proc.returncode == 0 and "\t" in proc.stdout:
            status, detail = proc.stdout.strip().splitlines()[-1].split("\t", 1)
            return path, status, detail
        err = (proc.stderr.strip().splitlines() or ["failed"])[-1]
        return path, "skipped", err[:200]

    done = 0
    if args.workers == 1:
        results = map(run_one, todo)
    else:
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=args.workers)
        results = pool.map(run_one, todo)
    for i, (path, status, detail) in enumerate(results, 1):
        report.append((path, status, detail))
        done += status.startswith("converted")
        print(f"[{i}/{len(todo)}] {status:<22} {os.path.basename(path)} -> {detail}", flush=True)
    report_path = None
    if args.out and len(files) > 1:
        os.makedirs(args.out, exist_ok=True)
        suffix = ("_" + args.name_prefix.strip("_")) if args.name_prefix else ""
        report_path = os.path.join(args.out, f"conversion_report{suffix}.tsv")
        with open(report_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("input\tstatus\tdetail\n")
            for row in report:
                handle.write("\t".join(str(x).replace("\t", " ") for x in row) + "\n")
    skipped = len(files) - done
    print(f"\n{done} of {len(files)} files converted; {skipped} skipped or already present"
          + (f" (details: {report_path})" if report_path else ""))


if __name__ == "__main__":
    main()
