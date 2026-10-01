r"""Prepare an auditable, tokenized training dataset for tiny_gpt.py.

Every step below is explicit, counted and written to the dataset report; no
record is dropped without a reason in docs.tsv / report.md.

  1. read      Markdown/text folders, a Hugging Face `save_to_disk` Arrow folder
               (e.g. Wikipedia), Parquet shards (e.g. FineWeb-Edu) and JSONL.
  2. clean     UTF-8 decode, BOM/CRLF/NFC normalisation, control characters,
               trailing spaces, runs of blank lines.
  3. filter    empty/short records, replacement-character damage, optional
               case-insensitive whole-word include/exclude keywords.
  4. dedup     exact duplicates (normalised-text hash) are skipped; near
               duplicates (MinHash-LSH, word 5-grams) are grouped so they land
               in the same split, or dropped with --near-dup drop.
  5. strip     lines repeated across many documents of one source
               (boilerplate) are removed and listed in boilerplate.tsv.
  6. split     train/validation per source by document group: a book's parts
               and a near-duplicate cluster never straddle the split.
  7. tokenize  SentencePiece BPE trained on training documents only, with
               lossless whitespace/newlines, split digits, byte fallback and
               BOS/EOS document markers.
  8. write     flat uint16/uint32 token files per (split, source), manifest.json
               with SHA-256 checksums, report.md, docs.tsv and audit tables.

Documents stream through a work spool on disk and token files are written
incrementally, so the text itself is never held in RAM; memory grows only with
the number of kept documents (about 1 KB each: metadata and near-duplicate
signatures). The trainer memory-maps the token files.

Text extraction from PDF, DOCX, PPTX or spreadsheets belongs *before* this
step (convert to Markdown/text first with convert_to_markdown.py); this script
only handles text.

Examples
--------
    python data_prep.py --out datasets\bio_v1 --md-dir %USERPROFILE%\ai_training_data ^
        --wiki-dir %USERPROFILE%\ai_training_data\wikipedia --wiki-max-docs 100000 ^
        --include-keywords keywords.txt --keyword-min-distinct 3

    python data_prep.py --out datasets\fineweb_bio --parquet "D:\fineweb-edu\*.parquet" ^
        --parquet-name fineweb --min-score 3 --include-keywords keywords.txt

    python data_prep.py --out datasets\probe --md-dir %USERPROFILE%\ai_training_data --scan-only
"""

import argparse
import collections
import datetime
import fnmatch
import glob
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import unicodedata

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FORMAT = "tiny_gpt_dataset"
FORMAT_VERSION = 1
MD_PATTERNS_DEFAULT = ("*.md", "*.markdown", "*.rmd", "*.Rmd", "*.txt")
SOURCE_NAME_RE = r"[A-Za-z][A-Za-z0-9_-]{0,31}"


# ---------------------------------------------------------------------------
# Tokenizer wrapper (shared with tiny_gpt.py and view_pt.py)
# ---------------------------------------------------------------------------
class Tokenizer:
    """SentencePiece model plus the encoding convention it was trained with.

    ``lines`` mode (datasets built by this script): text is split on "\\n",
    each line is encoded separately and the lines are joined with the
    dedicated newline token. Line-initial words then tokenize exactly like
    words elsewhere, and whitespace/newlines round-trip losslessly.

    ``plain`` mode: the whole text goes to SentencePiece at once. This is how
    the v3 tiny_gpt checkpoints were trained (no BOS/EOS, runs of spaces
    collapsed, newline as a byte-fallback token).
    """

    def __init__(self, proto, encode_mode="lines"):
        try:
            import sentencepiece as spm
        except ImportError:
            raise SystemExit("SentencePiece is required: python -m pip install sentencepiece")
        if encode_mode not in ("lines", "plain"):
            raise ValueError(f"unknown tokenizer encode mode {encode_mode!r}")
        self.sp = spm.SentencePieceProcessor()
        if not self.sp.LoadFromSerializedProto(proto):
            raise ValueError("could not parse the SentencePiece model")
        self.proto = bytes(proto)
        self.encode_mode = encode_mode
        self.vocab_size = self.sp.get_piece_size()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.unk_id = self.sp.unk_id()
        self.newline_id = -1
        if encode_mode == "lines":
            nl = self.sp.piece_to_id("\n")
            if nl == self.unk_id:
                raise ValueError("lines-mode tokenizer has no newline piece")
            self.newline_id = nl

    @classmethod
    def from_file(cls, path, encode_mode="lines"):
        with open(path, "rb") as handle:
            return cls(handle.read(), encode_mode)

    @property
    def sha256(self):
        return hashlib.sha256(self.proto).hexdigest()

    def meta(self):
        return {
            "type": "sentencepiece",
            "encode_mode": self.encode_mode,
            "vocab_size": self.vocab_size,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "unk_id": self.unk_id,
            "newline_id": self.newline_id,
            "sha256": self.sha256,
        }

    def encode(self, text):
        return self.encode_batch([text])[0]

    def encode_batch(self, texts, threads=1):
        if self.encode_mode == "plain":
            if not texts:
                return []
            return [list(ids) for ids in self.sp.encode(list(texts), out_type=int, num_threads=threads)]
        lines, counts = [], []
        for text in texts:
            parts = text.split("\n")
            lines.extend(parts)
            counts.append(len(parts))
        encoded = self.sp.encode(lines, out_type=int, num_threads=threads) if lines else []
        out, pos, nl = [], 0, self.newline_id
        for count in counts:
            ids = []
            for j in range(count):
                if j:
                    ids.append(nl)
                ids.extend(encoded[pos + j])
            pos += count
            out.append(ids)
        return out

    def decode(self, ids):
        markers = {i for i in (self.bos_id, self.eos_id) if i >= 0}
        ids = [int(i) for i in ids if int(i) not in markers]
        if self.encode_mode == "plain":
            return self.sp.decode(ids)
        pieces, current = [], []
        for i in ids:
            if i == self.newline_id:
                pieces.append(self.sp.decode(current))
                current = []
            else:
                current.append(i)
        pieces.append(self.sp.decode(current))
        return "\n".join(pieces)

    def token_bytes(self):
        """UTF-8 bytes each token stands for on its own (word-start marker = 1
        space byte, byte-fallback tokens = 1, newline = 1, control tokens = 0).
        For exact counts in lines mode use byte_tables()."""
        out = np.zeros(self.vocab_size, dtype=np.int64)
        for i in range(self.vocab_size):
            if self.sp.is_control(i) or self.sp.is_unknown(i):
                continue
            if self.sp.is_byte(i):
                out[i] = 1
            else:
                out[i] = len(self.sp.id_to_piece(i).replace("▁", " ").encode("utf-8"))
        return out

    def byte_tables(self):
        """(bytes, marked, line_break) arrays over the vocabulary for exact byte
        counts (bits per byte). In lines mode every encoded line starts with a
        word-start marker that is not in the text: it is the marked piece right
        after a newline or BOS token. Bytes of tokens y whose preceding tokens
        are x = sum(bytes[y]) - sum(marked[y] & line_break[x])."""
        size = self.token_bytes()
        marked = np.zeros(self.vocab_size, dtype=bool)
        line_break = np.zeros(self.vocab_size, dtype=bool)
        if self.encode_mode == "lines":
            for i in range(self.vocab_size):
                if size[i] and not self.sp.is_byte(i):
                    marked[i] = self.sp.id_to_piece(i).startswith("▁")
            line_break[self.newline_id] = True
            if self.bos_id >= 0:
                line_break[self.bos_id] = True
        return size, marked, line_break

    def piece(self, token_id):
        return self.sp.id_to_piece(int(token_id))

    def is_byte(self, token_id):
        return self.sp.is_byte(int(token_id))

    def is_special(self, token_id):
        token_id = int(token_id)
        return self.sp.is_control(token_id) or self.sp.is_unknown(token_id)


def token_category(tok, token_id):
    """Human-readable class of a vocabulary entry (used in reports and plots)."""
    if tok.is_special(token_id):
        return "special"
    if tok.is_byte(token_id):
        return "byte fallback"
    piece = tok.piece(token_id)
    if piece in ("\n", "\t") or set(piece) <= {"\u2581", " "}:
        return "whitespace/newline"
    if _piece_words(piece) > 1:
        return "multi-word"
    body = piece.replace("\u2581", "")
    if any(ch.isdigit() for ch in body) and all(ch.isdigit() or ch in ".,+-" for ch in body):
        return "number"
    cats = {unicodedata.category(ch)[0] for ch in body}
    if cats <= {"P"}:
        return "punctuation"
    if cats <= {"S"} or cats <= {"S", "P"}:
        return "symbol/math"
    if cats <= {"L", "M"}:
        if all(ord(ch) < 0x250 for ch in body):
            return "word start" if piece.startswith("\u2581") else "word piece"
        return "non-Latin letters"
    return "mixed"


def load_tokenizer_from_any(path):
    """Tokenizer from a .model file, a dataset folder or a tiny_gpt checkpoint."""
    if os.path.isdir(path):
        manifest = read_manifest(path)
        meta = manifest["tokenizer"]
        return Tokenizer.from_file(os.path.join(path, meta["file"]), meta.get("encode_mode", "lines"))
    if path.lower().endswith(".model"):
        return Tokenizer.from_file(path, "lines" if _has_newline_piece(path) else "plain")
    import torch
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - report any unpickling refusal
        raise SystemExit(f"Could not read tokenizer from {path}: {type(exc).__name__}: {exc}")
    if isinstance(ckpt.get("tokenizer"), dict):
        meta = ckpt["tokenizer"]
        return Tokenizer(meta["proto"], meta.get("encode_mode", "lines"))
    if ckpt.get("tokenizer_proto"):
        return Tokenizer(ckpt["tokenizer_proto"], "plain")
    raise SystemExit(f"{path} does not contain a SentencePiece tokenizer.")


def _has_newline_piece(model_path):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=model_path)
    return sp.piece_to_id("\n") != sp.unk_id()


# ---------------------------------------------------------------------------
# Dataset manifest helpers (used by the trainer and the viewer)
# ---------------------------------------------------------------------------
def read_manifest(dataset_dir):
    path = os.path.join(dataset_dir, "manifest.json")
    if not os.path.isfile(path):
        raise SystemExit(f"{dataset_dir} is not a prepared dataset (manifest.json missing). "
                         "Build one with data_prep.py.")
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format") != FORMAT:
        raise SystemExit(f"{path} is not a tiny_gpt dataset manifest.")
    if int(manifest.get("format_version", 0)) > FORMAT_VERSION:
        raise SystemExit(f"{path} was written by a newer data_prep.py "
                         f"(format {manifest['format_version']} > {FORMAT_VERSION}).")
    return manifest


def sha256_file(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def open_token_files(dataset_dir, verify=True):
    """Memory-map every token file listed in the manifest.

    Returns (manifest, {"train": {source: memmap}, "val": {source: memmap}}).
    Sizes are always checked; SHA-256 checksums when verify=True.
    """
    manifest = read_manifest(dataset_dir)
    dtype = np.dtype(manifest["dtype"])
    root = os.path.realpath(dataset_dir)
    arrays = {"train": {}, "val": {}}
    for source, info in manifest["sources"].items():
        for split in ("train", "val"):
            entry = info.get(split)
            if not entry or not entry.get("tokens"):
                continue
            path = os.path.realpath(os.path.join(root, entry["file"]))
            if os.path.dirname(path) != root:
                raise SystemExit(f"Manifest entry escapes the dataset folder: {entry['file']}")
            expected = int(entry["tokens"]) * dtype.itemsize
            if not os.path.isfile(path) or os.path.getsize(path) != expected:
                raise SystemExit(f"{path} is missing or has the wrong size "
                                 f"(expected {expected:,} bytes). Rebuild the dataset.")
            if verify and entry.get("sha256") and sha256_file(path) != entry["sha256"]:
                raise SystemExit(f"{path} failed its SHA-256 check. Rebuild the dataset.")
            arrays[split][source] = np.memmap(path, dtype=dtype, mode="r", shape=(int(entry["tokens"]),))
    return manifest, arrays


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
_CTRL_DROP_RE = re.compile(r"[\x00-\x08\x0e-\x1f\x7f]")
_CTRL_BREAK_RE = re.compile(r"[\x0b\x0c]")
_TRAILING_RE = re.compile(r"[ \t]+(?=\n)")
_BLANK_RUN_RE = re.compile(r"\n{3,}")
_MOJIBAKE_RE = re.compile("[\u00c2\u00c3][\u0080-\u00bf]")


def decode_bytes(raw, stats):
    """UTF-8 (BOM tolerated); invalid bytes are replaced and counted."""
    if raw.startswith(b"\xef\xbb\xbf"):
        stats["bom_removed"] += 1
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        stats["decode_errors_docs"] += 1
        text = raw.decode("utf-8-sig", errors="replace")
        stats["decode_errors_chars"] += text.count("\ufffd")
        return text


def clean_text(text, stats):
    if text.startswith("\ufeff"):
        text = text[1:]
        stats["bom_removed"] += 1
    if "\r" in text:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        stats["newlines_normalised_docs"] += 1
    if not unicodedata.is_normalized("NFC", text):
        text = unicodedata.normalize("NFC", text)
        stats["unicode_nfc_docs"] += 1
    text, n_break = _CTRL_BREAK_RE.subn("\n", text)
    text, n_drop = _CTRL_DROP_RE.subn("", text)
    if n_break or n_drop:
        stats["control_char_docs"] += 1
        stats["control_chars"] += n_break + n_drop
    before = len(text)
    text = _TRAILING_RE.sub("", text)
    text = _BLANK_RUN_RE.sub("\n\n", text)
    if len(text) != before:
        stats["whitespace_trimmed_docs"] += 1
    if _MOJIBAKE_RE.search(text):
        stats["mojibake_suspect_docs"] += 1
    return text.strip()


def strip_replacement_chars(text, stats):
    """Remove lines that are mostly U+FFFD (garbled PDF glyphs, broken indexes)
    and then any remaining U+FFFD characters; both are counted."""
    kept, dropped_lines = [], 0
    for line in text.split("\n"):
        bad = line.count("�")
        visible = len(line) - line.count(" ")
        if bad and bad >= 0.2 * max(visible, 1):
            dropped_lines += 1
            continue
        kept.append(line)
    text = "\n".join(kept)
    remaining = text.count("�")
    stats["replacement_docs_repaired"] += 1
    stats["replacement_lines_removed"] += dropped_lines
    stats["replacement_chars_removed"] += remaining
    return _BLANK_RUN_RE.sub("\n\n", text.replace("�", "")).strip()


_EN_FUNCTION_WORDS = frozenset(
    "the of and to in a is that for it as was with be by on not he this are or his from at which but have an they "
    "you were her she there been one all we their its has had would will can if more when so no who what may also "
    "into than only other some these do such them could two then most our many".split())
_LETTER_WORDS_RE = re.compile(r"[^\W\d_]+")


def looks_english(text):
    """Heuristic language check, calibrated on this corpus: English prose has
    30-40% function words; narrations in another language had 0% and ~7% non-ASCII
    letters. Short texts, tables and code (ASCII) are always kept."""
    body = text[:8_000]  # ~1,300 words decide the language as well as the whole text
    words = _LETTER_WORDS_RE.findall(body)
    if len(words) < 50:
        return True
    share = sum(w.lower() in _EN_FUNCTION_WORDS for w in words) / len(words)
    if share >= 0.10:
        return True
    letters = "".join(words)
    non_ascii = (len(letters) - len(letters.encode("ascii", "ignore"))) / max(len(letters), 1)
    return non_ascii <= 0.005


def normalised_hash(text):
    squashed = " ".join(text.lower().split())
    return hashlib.blake2b(squashed.encode("utf-8"), digest_size=16).hexdigest()


def split_long_text(text, max_chars):
    """Split at paragraph (then line, then hard) boundaries into <= max_chars parts.
    A cut keeps its line breaks at the start of the next part, so the parts put
    back together are exactly the text and, in lines mode, encode to the same
    tokens (only a hard cut inside a line longer than max_chars/2 adds a space)."""
    if len(text) <= max_chars:
        return [text]
    parts, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            cut = text.rfind("\n\n", start + max_chars // 2, end)
            if cut < 0:
                cut = text.rfind("\n", start + max_chars // 2, end)
            if cut > start:
                end = cut
        parts.append(text[start:end])
        start = end
    return parts


def stable_unit(*parts):
    """Deterministic float in [0, 1) from strings (independent of PYTHONHASHSEED)."""
    digest = hashlib.blake2b("\x1f".join(map(str, parts)).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") / 2.0 ** 64


# ---------------------------------------------------------------------------
# Keyword filter
# ---------------------------------------------------------------------------
def read_terms(value):
    """Terms from keyword files (one term per line, # comments) and/or literal
    terms, comma-separated: "keywords.txt,keywords_extra.txt,CRISPR"."""
    if not value:
        return []
    items = [value] if os.path.isfile(value) else [v.strip() for v in value.split(",") if v.strip()]
    lines = []
    for item in items:
        if os.path.isfile(item):
            with open(item, encoding="utf-8") as handle:
                lines.extend(handle.read().splitlines())
        elif os.path.splitext(item)[1].lower() in (".txt", ".tsv", ".csv", ".lst"):
            raise SystemExit(f"Keyword file not found: {item}")
        else:
            lines.append(item)
    terms = []
    for line in lines:
        term = line.strip()
        if not term or term.startswith("#"):
            continue
        if len(term) > 100:
            raise SystemExit(f"Keyword too long (max 100 characters): {term[:40]}...")
        terms.append(term)
    return list(dict.fromkeys(terms))


def _term_parts(term):
    """(core text, case_sensitive, leading wildcard, trailing wildcard).
    '=MET' is case-sensitive (gene symbols that are also English words);
    'genom*' matches any ending, '*mab' any beginning (drug-name suffixes)."""
    case_sensitive = term.startswith("=")
    body = term[1:] if case_sensitive else term
    lead, trail = body.startswith("*"), body.endswith("*")
    core = body.strip("*").strip()
    if not core:
        raise SystemExit(f"Invalid keyword {term!r}")
    return core, case_sensitive, lead, trail


_WORDS_RE = re.compile(r"\w+")


def _term_words(term):
    """The words a term matches, as the Python matcher sees them: (words,
    [(lowercase word, 'exact' | 'prefix' | 'suffix')], case_sensitive, lead, trail);
    words is empty for a term without letters or digits (never matches)."""
    core, case_sensitive, lead, trail = _term_parts(term)
    words = _WORDS_RE.findall(core)
    lead = lead and len(words) == 1  # a leading * counts for single words only
    shape = [(w.lower(), "exact") for w in words]
    if lead:
        shape[0] = (shape[0][0], "suffix")
    elif trail and shape:
        shape[-1] = (shape[-1][0], "prefix")
    return words, shape, case_sensitive, lead, trail


def _same_word_possible(a, b):
    """Could one word match both word patterns (exact / prefix* / *suffix)?"""
    (x, kx), (y, ky) = sorted((a, b), key=lambda p: p[1])  # order: exact, prefix, suffix
    if kx == "exact":
        return {"exact": x == y, "prefix": x.startswith(y), "suffix": x.endswith(y)}[ky]
    if kx == ky:
        return (x.startswith(y) or y.startswith(x)) if kx == "prefix" else (x.endswith(y) or y.endswith(x))
    return True  # a prefix term and a suffix term can both match one long word


def _can_overlap(a, b):
    """Could matches of two terms (word sequences) share a word of the text?"""
    n, m = len(a), len(b)
    return any(all(_same_word_possible(a[i], b[i - d]) for i in range(max(0, d), min(n, d + m)))
               for d in range(-(m - 1), n))


def _re2_term(words, case_sensitive, lead, trail):
    rx = r"\W+".join(re.escape(w) for w in words)
    ascii_word = lambda ch: ch.isascii() and (ch.isalnum() or ch == "_")  # noqa: E731 - RE2 \b is ASCII-only
    rx = (r"\b\w*" if lead else r"\b" if ascii_word(words[0][0]) else "") + rx
    rx += r"\w*\b" if trail else r"\b" if ascii_word(words[-1][-1]) else ""
    return f"(?-i:{rx})" if case_sensitive else rx


PREFILTER_TERMS_PER_PATTERN = 200


def _prefilter_patterns(terms):
    """RE2 patterns for the vectorised C++ prefilter; the SUM of their match
    counts is never below the Python matcher's count, so no document the full
    check would keep is rejected early. One pattern counts overlapping terms
    once ('genetic drift' also contains 'genetic*'), so terms whose matches
    could overlap go to different groups (2,385 terms -> 5 groups).
    Words are joined by any non-word run, as the Python matcher allows; word
    boundaries are only required next to ASCII word characters.

    Each group is split into patterns of at most 200 terms (2,385 terms -> 16
    patterns): a larger alternation outgrows RE2's fast automaton memory and
    falls back to a far slower matcher (one pattern of 2,104 terms ran at 0.22
    MB/s on Wikipedia text; 16 patterns of <= 200 terms at 14.4 MB/s, with
    identical counts, since terms within a group never overlap)."""
    groups = []
    items = sorted(((t,) + _term_words(t) for t in terms), key=lambda it: (-len(it[1]), it[0]))
    for term, words, shape, case_sensitive, lead, trail in items:
        if not words:
            continue
        rx = _re2_term(words, case_sensitive, lead, trail)
        for group in groups:
            if not any(_can_overlap(shape, other) for other, _ in group):
                group.append((shape, rx))
                break
        else:
            groups.append([(shape, rx)])
    patterns = []
    for group in groups:
        rxs = sorted((rx for _, rx in group), key=len, reverse=True)
        for i in range(0, len(rxs), PREFILTER_TERMS_PER_PATTERN):
            patterns.append("(?:" + "|".join(rxs[i:i + PREFILTER_TERMS_PER_PATTERN]) + ")")
    return patterns


class _Affixes:
    """Wildcard terms ('genom*' = prefix, '*mab' = suffix) grouped by length. A
    word is looked up only if its first (or last) three letters begin (or end)
    some term, which settles most words with one set lookup."""

    def __init__(self, suffix):
        self.suffix, self.by_length, self.heads = suffix, collections.defaultdict(dict), None

    def add(self, text, term):
        self.by_length[len(text)][text] = term

    def finish(self):
        self.lengths = sorted(self.by_length)
        if self.lengths and self.lengths[0] >= 3:
            self.heads = {t[-3:] if self.suffix else t[:3] for table in self.by_length.values() for t in table}
        return self

    def count(self, freq, counter, weight):
        heads, suffix = self.heads, self.suffix
        for w, c in freq.items():
            if heads is not None and (w[-3:] if suffix else w[:3]) not in heads:
                continue
            size = len(w)
            for length in self.lengths:
                if length > size:
                    break
                term = self.by_length[length].get(w[-length:] if suffix else w[:length])
                if term is not None:
                    counter[term] += c * weight


class _TermMatcher:
    """Counts keyword hits with dictionary lookups instead of one huge regex
    alternation (Python's re managed ~0.5 MB/s with 600 terms). Words are
    counted once (C-level Counter); single-word terms are looked up on the
    distinct words; phrases are checked only where their first word occurs.
    Whole words, any separator between the words of a phrase, case-insensitive
    unless the term starts with '=', '*' wildcards at either end of single-word
    terms and at the end of phrases. The RE2 prefilter (_prefilter_patterns)
    never counts fewer hits than this."""

    def __init__(self, terms):
        self.single, self.single_cs = {}, {}
        self.phrase = collections.defaultdict(list)       # first word -> [(rest, term, trail)]
        self.phrase_cs = collections.defaultdict(list)
        self.prefix, self.prefix_cs = _Affixes(False), _Affixes(False)
        self.suffix, self.suffix_cs = _Affixes(True), _Affixes(True)
        for term in terms:
            core, case_sensitive, lead, trail = _term_parts(term)
            words = _WORDS_RE.findall(core)
            if not words:
                continue
            if not case_sensitive:
                words = [w.lower() for w in words]
            if len(words) == 1 and lead:
                (self.suffix_cs if case_sensitive else self.suffix).add(words[0], term)
            elif len(words) == 1 and trail:
                (self.prefix_cs if case_sensitive else self.prefix).add(words[0], term)
            elif len(words) == 1:
                (self.single_cs if case_sensitive else self.single)[words[0]] = term
            else:
                (self.phrase_cs if case_sensitive else self.phrase)[words[0]].append((words[1:], term, trail))
        self.affixes = [a.finish() for a in (self.prefix, self.suffix) if a.by_length]
        self.affixes_cs = [a.finish() for a in (self.prefix_cs, self.suffix_cs) if a.by_length]
        self.need_case = bool(self.single_cs or self.phrase_cs or self.affixes_cs)

    @staticmethod
    def _phrases(seq, present, table, counter, weight):
        n = len(seq)
        for i, w in enumerate(seq):
            if w not in present:
                continue
            for rest, term, trail in table[w]:
                k = len(rest)
                if i + k >= n or seq[i + 1:i + k] != rest[:-1]:
                    continue
                last = seq[i + k]
                if last.startswith(rest[-1]) if trail else last == rest[-1]:
                    counter[term] += weight

    def count(self, text, counter, weight=1):
        words = _WORDS_RE.findall(text)
        if not words:
            return counter
        lower = [w.lower() for w in words]
        freq = collections.Counter(lower)
        for w in freq.keys() & self.single.keys():
            counter[self.single[w]] += freq[w] * weight
        for affixes in self.affixes:
            affixes.count(freq, counter, weight)
        present = freq.keys() & self.phrase.keys()
        if present:
            self._phrases(lower, present, self.phrase, counter, weight)
        if self.need_case:
            freq_cs = collections.Counter(words)
            for w in freq_cs.keys() & self.single_cs.keys():
                counter[self.single_cs[w]] += freq_cs[w] * weight
            for affixes in self.affixes_cs:
                affixes.count(freq_cs, counter, weight)
            present = freq_cs.keys() & self.phrase_cs.keys()
            if present:
                self._phrases(words, present, self.phrase_cs, counter, weight)
        return counter


class KeywordFilter:
    """Case-insensitive whole-word include/exclude filter.

    A document is kept when its weighted hit count (title hits x title_weight
    + body hits) reaches ``min_hits`` AND it matches at least ``min_distinct``
    different include terms, and it matches no exclude term. Only the first
    ``scan_chars`` characters of the body are scanned (speed; long articles
    reveal their topic early). The include terms also compile to a few RE2
    patterns (include_patterns) for the vectorised C++ prefilter over
    Arrow/Parquet/JSONL data; exclusions are decided by the full check only.
    """

    def __init__(self, include, exclude, min_hits=3, min_distinct=2, title_weight=3,
                 scan_chars=100_000):
        self.include, self.exclude = include, exclude
        self.min_hits, self.min_distinct = max(0, min_hits), max(0, min_distinct)
        self.title_weight, self.scan_chars = title_weight, scan_chars
        self.include_patterns = _prefilter_patterns(include)
        self._include = _TermMatcher(include) if include else None
        self._exclude = _TermMatcher(exclude) if exclude else None

    @property
    def active(self):
        return bool(self.include or self.exclude)

    def evaluate(self, title, text):
        """Return (keep, reason, weighted_hits, Counter(term -> hits))."""
        body = text[: self.scan_chars]
        if self._exclude is not None:
            found = self._exclude.count(body, self._exclude.count(title or "", collections.Counter()))
            if found:
                return False, "keyword_excluded", 0, found
        if self._include is None:
            return True, "", 0, collections.Counter()
        terms = self._include.count(body, collections.Counter())
        weighted = sum(terms.values())
        title_hits = self._include.count(title or "", collections.Counter())
        weighted += self.title_weight * sum(title_hits.values())
        terms.update(title_hits)
        if weighted < self.min_hits:
            return False, "keyword_min_hits", weighted, terms
        if len(terms) < self.min_distinct:
            return False, "keyword_min_distinct", weighted, terms
        return True, "", weighted, terms


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------
def parse_text_sources(entries):
    """--md-dir values: "path" (source name "md") or "name=path" (own source),
    e.g. papers=D:\\converted\\papers. Returns {name: [paths]} in order."""
    groups = {}
    for entry in entries:
        name, path = "md", entry
        head, sep, tail = entry.partition("=")
        if sep and re.fullmatch(SOURCE_NAME_RE, head) and not os.path.exists(entry):
            name, path = head, tail
        groups.setdefault(name, []).append(path)
    return groups


def iter_markdown(dirs, patterns, excludes, recursive, stats, ledger, source="md"):
    """Documents of one text source. With several folders, ids start with the
    folder name so equal file names in different folders stay distinct."""
    seen = set()
    bases = [os.path.basename(os.path.normpath(d)) or f"folder{k + 1}" for k, d in enumerate(dirs)]
    if len({b.lower() for b in bases}) < len(bases):
        bases = [f"{k + 1}-{b}" for k, b in enumerate(bases)]
    for k, directory in enumerate(dirs):
        if not os.path.isdir(directory):
            raise SystemExit(f"--md-dir is not a folder: {directory}")
        found = []
        for pattern in patterns:
            spec = os.path.join(directory, "**", pattern) if recursive else os.path.join(directory, pattern)
            found.extend(glob.glob(spec, recursive=recursive))
        for path in sorted(set(found)):
            real = os.path.realpath(path)
            if real in seen or not os.path.isfile(real):
                continue
            seen.add(real)
            rel = os.path.relpath(path, directory)
            doc_id = os.path.join(bases[k], rel) if len(dirs) > 1 else rel
            stats["scanned"] += 1
            if any(fnmatch.fnmatch(os.path.basename(path), ex) or fnmatch.fnmatch(rel, ex) for ex in excludes):
                stats["skipped"]["excluded_by_glob"] += 1
                ledger.append(_ledger_row(source, doc_id, rel, "skipped", "excluded_by_glob", os.path.getsize(path)))
                continue
            size = os.path.getsize(path)
            if size > 512 * 1024 * 1024:
                stats["skipped"]["file_too_large"] += 1
                ledger.append(_ledger_row(source, doc_id, rel, "skipped", "file_too_large", size))
                continue
            with open(path, "rb") as handle:
                raw = handle.read()
            yield doc_id, os.path.splitext(os.path.basename(path))[0], decode_bytes(raw, stats["cleaning"]), {"path": path}


class ArrowTableSource:
    """Record batches from a HF `save_to_disk` folder (Arrow IPC stream files)
    or from Parquet files. Only the requested columns are materialised."""

    def __init__(self, kind, paths, text_col, title_col, id_col, score_col=None, batch_rows=1000,
                 require_score=False):
        import pyarrow as pa  # noqa: F401 - fail early with a clear message
        self.kind, self.paths = kind, paths
        self.text_col, self.title_col, self.id_col, self.score_col = text_col, title_col, id_col, score_col
        self.batch_rows, self.require_score = batch_rows, require_score

    @staticmethod
    def hf_files(folder, split):
        if os.path.isfile(os.path.join(folder, "dataset_dict.json")):
            with open(os.path.join(folder, "dataset_dict.json"), encoding="utf-8") as handle:
                splits = json.load(handle).get("splits", [])
            if split not in splits:
                raise SystemExit(f"{folder} has splits {splits}; choose one with --wiki-split.")
            folder = os.path.join(folder, split)
        state = os.path.join(folder, "state.json")
        if not os.path.isfile(state):
            raise SystemExit(f"{folder} is not a Hugging Face save_to_disk folder (state.json missing).")
        with open(state, encoding="utf-8") as handle:
            files = [os.path.join(folder, f["filename"]) for f in json.load(handle)["_data_files"]]
        missing = [f for f in files if not os.path.isfile(f)]
        if missing:
            raise SystemExit(f"Arrow shards missing: {missing[:3]}")
        return files

    def batches(self, file_indices=None):
        """Yield (file_index, batch_index, dict of pyarrow arrays)."""
        import pyarrow as pa
        cols = [c for c in (self.text_col, self.title_col, self.id_col, self.score_col) if c]
        for file_index, path in enumerate(self.paths):
            if file_indices is not None and file_index not in file_indices:
                continue
            if self.kind == "hf_arrow":
                with pa.memory_map(path, "r") as src:
                    reader = pa.ipc.open_stream(src)
                    names = reader.schema.names
                    self._check_columns(names, path)
                    for batch_index, batch in enumerate(reader):
                        yield file_index, batch_index, {c: batch.column(c) for c in cols if c in names}
            else:
                import pyarrow.parquet as pq
                handle = pq.ParquetFile(path)
                names = handle.schema_arrow.names
                self._check_columns(names, path)
                use = [c for c in cols if c in names]
                for batch_index, batch in enumerate(handle.iter_batches(batch_size=self.batch_rows, columns=use)):
                    yield file_index, batch_index, {c: batch.column(c) for c in use}

    def _check_columns(self, names, path):
        if self.text_col not in names:
            raise SystemExit(f"{path}: text column {self.text_col!r} not found; columns are {names}. "
                             "Use --text-column.")
        if self.require_score and self.score_col not in names:
            raise SystemExit(f"{path}: score column {self.score_col!r} (needed by --min-score) not found; columns "
                             f"are {names}. Use --score-column, or drop --min-score.")


def _vector_scores(arrays, kw, text_col, title_col, score_col, min_score):
    """Vectorised (RE2, C++) prefilter; returns numpy arrays (weighted hits, chars, score_ok).
    Hits are an upper bound of the full check's weighted hits (see _prefilter_patterns)."""
    import pyarrow.compute as pc
    text = arrays[text_col]
    n = len(text)
    chars = np.asarray(pc.fill_null(pc.utf8_length(text), 0).to_numpy(zero_copy_only=False), dtype=np.int64)
    hits = np.zeros(n, dtype=np.int64)

    def count(values, pattern):
        return np.asarray(pc.fill_null(pc.count_substring_regex(values, pattern=pattern, ignore_case=True), 0)
                          .to_numpy(zero_copy_only=False), dtype=np.int64)

    if kw is not None and kw.include_patterns:
        body = pc.utf8_slice_codeunits(text, 0, kw.scan_chars)  # characters, as the Python check slices
        for pattern in kw.include_patterns:
            hits += count(body, pattern)
            if title_col in arrays:
                hits += kw.title_weight * count(arrays[title_col], pattern)
    score_ok = np.ones(n, dtype=bool)
    if score_col and min_score is not None and score_col in arrays:
        import pyarrow as pa
        as_float = pc.cast(arrays[score_col], pa.float64())  # int_score is int64, score is float
        scores = np.asarray(pc.fill_null(as_float, -1e30).to_numpy(zero_copy_only=False), dtype=np.float64)
        score_ok = scores >= min_score
    return hits, chars, score_ok


def _score_file(source, file_index, kw, min_score):
    """Pass A of 'top' selection for one file (runs in a worker thread; pyarrow
    compute releases the GIL)."""
    need = kw is not None and bool(kw.include_patterns)
    counts = collections.Counter({"scanned": 0})
    cands = []
    for _, batch_index, arrays in source.batches({file_index}):
        hits, chars, score_ok = _vector_scores(arrays, kw, source.text_col, source.title_col,
                                               source.score_col, min_score)
        counts["scanned"] += len(chars)
        keep = score_ok & (chars > 0)
        counts["score_below_min"] += int((~score_ok).sum())
        counts["empty"] += int((score_ok & (chars == 0)).sum())
        if need:
            counts["keyword_min_hits"] += int((keep & (hits < kw.min_hits)).sum())
            keep &= hits >= kw.min_hits
        for row in np.nonzero(keep)[0]:
            score = hits[row] / math.sqrt(max(int(chars[row]), 500))
            cands.append((-float(score), file_index, batch_index, int(row)))
    return counts, cands


def _prescored(source, kw, min_score, threads):
    """Record batches with their vectorised keyword scores, in order. Scoring is
    C++ (pyarrow releases the GIL), so the next batches are scored in background
    threads while the current one is being used."""
    from concurrent.futures import ThreadPoolExecutor
    pending = collections.deque()
    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        for file_index, batch_index, arrays in source.batches():
            pending.append((file_index, batch_index, arrays,
                            pool.submit(_vector_scores, arrays, kw, source.text_col, source.title_col,
                                        source.score_col, min_score)))
            if len(pending) > 2 * max(1, threads):
                f, b, a, future = pending.popleft()
                yield f, b, a, future.result()
        while pending:
            f, b, a, future = pending.popleft()
            yield f, b, a, future.result()


def iter_table_source(name, source, kw, max_docs, select, min_score, sample_fraction, seed, stats,
                      threads=1):
    """Yield (doc_id, title, text, meta) from an Arrow/Parquet source.

    select='first': stream order, stop once max_docs documents have been kept
                    (stats["accepted"], counted by the caller after the full check).
    select='top'  : score every record first (weighted keyword hits divided by
                    sqrt(characters), in C++), then stream back the best
                    max_docs*1.5 candidates in file order with their rank in
                    meta["rank"]; the exact max_docs cut is made by rank after
                    cleaning, filtering and boilerplate removal.
    """
    need_vector = kw is not None and kw.active
    t0 = time.time()
    if select == "top":
        from concurrent.futures import ThreadPoolExecutor
        cands = []
        with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
            futures = [pool.submit(_score_file, source, f, kw, min_score) for f in range(len(source.paths))]
            for done, future in enumerate(futures, 1):
                counts, found = future.result()
                stats["scanned"] += counts.pop("scanned", 0)
                for reason, count in counts.items():
                    if count:
                        stats["skipped"][reason] += count
                cands.extend(found)
                print(f"  [{name}] scored file {done}/{len(futures)}: {stats['scanned']:,} records, "
                      f"{len(cands):,} candidates ({time.time() - t0:,.0f} s)", flush=True)
        cands.sort()
        limit = len(cands) if not max_docs else min(len(cands), int(max_docs * 1.5) + 100)
        stats["skipped"]["not_selected_rank"] += len(cands) - limit
        wanted = collections.defaultdict(dict)
        for rank, (_, f, b, r) in enumerate(cands[:limit]):
            wanted[(f, b)][r] = rank
        files = {f for f, _ in wanted}
        del cands
        print(f"  [{name}] reading the top {limit:,} candidates", flush=True)
        for file_index, batch_index, arrays in source.batches(files):
            picks = wanted.get((file_index, batch_index))
            if not picks:
                continue
            for r in sorted(picks):
                doc_id, title, text, meta = _row(arrays, r, source, file_index, batch_index)
                meta["rank"] = picks[r]
                yield doc_id, title, text, meta
        return

    for file_index, batch_index, arrays, scored in _prescored(source, kw, min_score, threads):
        hits, chars, score_ok = scored
        for r in range(len(chars)):
            if max_docs and stats["accepted"] >= max_docs:
                stats["stopped_early"] = True
                return
            stats["scanned"] += 1
            if not score_ok[r]:
                stats["skipped"]["score_below_min"] += 1
                continue
            if chars[r] == 0:
                stats["skipped"]["empty"] += 1
                continue
            if need_vector and kw.include_patterns and hits[r] < kw.min_hits:
                stats["skipped"]["keyword_min_hits"] += 1
                continue
            doc_id, title, text, meta = _row(arrays, r, source, file_index, batch_index)
            if sample_fraction < 1.0 and stable_unit(seed, "sample", name, doc_id) >= sample_fraction:
                stats["skipped"]["not_sampled"] += 1
                continue
            yield doc_id, title, text, meta
        if stats["scanned"] % 500_000 < len(chars):
            print(f"  [{name}] scanned {stats['scanned']:,} records ({time.time() - t0:,.0f} s)", flush=True)


def _row(arrays, r, source, file_index, batch_index):
    text = arrays[source.text_col][r].as_py() or ""
    title = arrays[source.title_col][r].as_py() if source.title_col in arrays else ""
    doc_id = arrays[source.id_col][r].as_py() if source.id_col in arrays else None
    if doc_id is None:
        doc_id = f"f{file_index}:b{batch_index}:r{r}"
    meta = {"file": file_index}
    if source.score_col in arrays:
        meta["score"] = arrays[source.score_col][r].as_py()
    return str(doc_id), str(title or ""), text, meta


def iter_jsonl(paths, text_key, title_key, id_key, max_docs, stats, kw=None, batch=5000, name="jsonl", threads=1):
    """JSON Lines (optionally .gz, e.g. peS2o shards).

    Fast path: pyarrow parses the JSON in C++ (all cores) into Arrow batches, the
    keyword prefilter runs on them directly, and only matching records become
    Python strings. If a file does not fit that reader (for example numeric ids or
    other type changes), it falls back to line-by-line json.loads from the first
    record not yet read. Reading stops once max_docs documents have been kept
    (stats["accepted"]). Progress is printed every 500,000 records."""
    t0 = time.time()
    progress = {"next": 500_000}

    def report():
        if stats["scanned"] >= progress["next"]:
            print(f"  [{name}] scanned {stats['scanned']:,} records, {stats.get('accepted_raw', 0):,} passed the "
                  f"prefilter ({time.time() - t0:,.0f} s)", flush=True)
            progress["next"] = (stats["scanned"] // 500_000 + 1) * 500_000

    for path in paths:
        done_rows = 0
        try:
            for scanned, counts, docs in _jsonl_arrow_blocks(path, text_key, title_key, id_key, kw, threads):
                done_rows += scanned
                stats["scanned"] += scanned
                for reason, count in counts.items():
                    if count:
                        stats["skipped"][reason] += count
                for doc in docs:
                    if max_docs and stats["accepted"] >= max_docs:
                        stats["stopped_early"] = True
                        return
                    stats["accepted_raw"] = stats.get("accepted_raw", 0) + 1
                    yield doc
                report()
            continue
        except Exception as exc:  # noqa: BLE001 - pyarrow.ArrowInvalid and friends
            print(f"  [{name}] {os.path.basename(path)}: fast JSON reader stopped after {done_rows:,} records "
                  f"({type(exc).__name__}: {str(exc)[:120]}); continuing with the line-by-line reader", flush=True)
        yield from _jsonl_python(path, text_key, title_key, id_key, max_docs, stats, kw, batch, done_rows, report)


def _jsonl_arrow_blocks(path, text_key, title_key, id_key, kw, threads=1, block_bytes=16 << 20):
    """(records in block, skip counts, [(doc_id, title, text, meta)]) per Arrow block, in order.
    The keyword prefilter of several blocks runs in parallel threads (C++, no GIL)."""
    from concurrent.futures import ThreadPoolExecutor
    pending = collections.deque()
    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        for block, row0 in _jsonl_arrow_raw(path, text_key, title_key, id_key, block_bytes):
            pending.append(pool.submit(_jsonl_arrow_block, block, row0, path, text_key, title_key, id_key, kw))
            if len(pending) > 2 * max(1, threads):
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def _jsonl_arrow_raw(path, text_key, title_key, id_key, block_bytes):
    import pyarrow as pa
    import pyarrow.json as pj
    names = list(dict.fromkeys([text_key, title_key, id_key]))
    schema = pa.schema([pa.field(n, pa.string()) for n in names])
    stream = pa.input_stream(path, compression="gzip" if path.lower().endswith(".gz") else None)
    reader = pj.open_json(stream, read_options=pj.ReadOptions(block_size=block_bytes),
                          parse_options=pj.ParseOptions(explicit_schema=schema, unexpected_field_behavior="ignore"))
    row0 = 0
    for block in reader:
        yield block, row0
        row0 += block.num_rows


def _jsonl_arrow_block(block, row0, path, text_key, title_key, id_key, kw):
    import pyarrow as pa
    import pyarrow.compute as pc
    base = os.path.basename(path)
    n = block.num_rows
    counts = collections.Counter()
    text = block.column(text_key)
    missing = np.asarray(pc.is_null(text).to_numpy(zero_copy_only=False), dtype=bool)
    counts["missing_text"] = int(missing.sum())
    arrays = {"text": pc.fill_null(text, ""), "title": pc.fill_null(block.column(title_key), "")}
    keep = ~missing
    if kw is not None and kw.include_patterns:
        hits, chars, _ = _vector_scores(arrays, kw, "text", "title", None, None)
        counts["empty"] = int((keep & (chars == 0)).sum())
        keep &= chars > 0
        counts["keyword_min_hits"] = int((keep & (hits < kw.min_hits)).sum())
        keep &= hits >= kw.min_hits
    rows = np.nonzero(keep)[0]
    if len(rows):
        picked = block.take(pa.array(rows))
        texts = picked.column(text_key).to_pylist()
        titles = picked.column(title_key).to_pylist()
        ids = picked.column(id_key).to_pylist()
        docs = [(str(i) if i else f"{base}:{row0 + int(r) + 1}", t or "", x, {"path": path})
                for r, i, t, x in zip(rows, ids, titles, texts)]
    else:
        docs = []
    return n, counts, docs


def _jsonl_python(path, text_key, title_key, id_key, max_docs, stats, kw, batch, skip_rows, report):
    """The line-by-line reader (json.loads); skips the first `skip_rows` records.
    Records without an id are named <file>:<record number>, as in the fast reader."""
    import gzip
    opener = gzip.open if path.lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        pending, seen = [], 0
        for line in handle:
            if not line.strip():
                continue
            seen += 1
            if seen <= skip_rows:
                continue
            pending.append((seen, line))
            if len(pending) >= batch:
                if max_docs and stats["accepted"] >= max_docs:
                    stats["stopped_early"] = True
                    return
                yield from _jsonl_batch(pending, path, text_key, title_key, id_key, stats, kw)
                pending = []
                report()
        if pending and not (max_docs and stats["accepted"] >= max_docs):
            yield from _jsonl_batch(pending, path, text_key, title_key, id_key, stats, kw)


def _jsonl_batch(pending, path, text_key, title_key, id_key, stats, kw):
    records = []
    for record_no, line in pending:
        stats["scanned"] += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            stats["skipped"]["invalid_json"] += 1
            continue
        if not isinstance(record, dict) or not isinstance(record.get(text_key), str):
            stats["skipped"]["missing_text"] += 1
            continue
        records.append((record_no, record))
    if kw is not None and kw.include_patterns and records:
        import pyarrow as pa
        arrays = {"text": pa.array([r[text_key] for _, r in records]),
                  "title": pa.array([str(r.get(title_key) or "") for _, r in records])}
        hits, chars, _ = _vector_scores(arrays, kw, "text", "title", None, None)
        keep = []
        for (record_no, record), h, c in zip(records, hits, chars):
            if c == 0:
                stats["skipped"]["empty"] += 1
            elif h < kw.min_hits:
                stats["skipped"]["keyword_min_hits"] += 1
            else:
                keep.append((record_no, record))
        records = keep
    for record_no, record in records:
        stats["accepted_raw"] = stats.get("accepted_raw", 0) + 1
        doc_id = str(record.get(id_key) or f"{os.path.basename(path)}:{record_no}")
        yield doc_id, str(record.get(title_key) or ""), record[text_key], {"path": path}


# ---------------------------------------------------------------------------
# Near-duplicate detection (MinHash + LSH), numpy only
# ---------------------------------------------------------------------------
_WORD_BYTE = np.zeros(256, dtype=bool)  # bytes that belong to words: ASCII letters, digits, _ and all
_WORD_BYTE[list(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")] = True
_WORD_BYTE[128:] = True                   # UTF-8 bytes of non-ASCII characters (letters such as e-acute)
_P = np.uint64(0x100000001B3)
_POW = {"p": np.ones(1, dtype=np.uint64), "inv": np.ones(1, dtype=np.uint64)}


def _powers(kind, n):
    """P^i (or P^-i) for i = 0..n-1, modulo 2^64 (numpy unsigned arithmetic wraps)."""
    table = _POW[kind]
    if len(table) < n:
        base = _P if kind == "p" else np.uint64(pow(int(_P), -1, 1 << 64))
        size = max(n, 2 * len(table), 1 << 16)
        with np.errstate(over="ignore"):
            table = np.concatenate([np.ones(1, dtype=np.uint64), np.cumprod(np.full(size - 1, base, dtype=np.uint64))])
        _POW[kind] = table
    return table


def _mix64(x):
    x = x.copy()
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return x


class ShingleHasher:
    """Word n-gram hashes, computed with numpy over the text's UTF-8 bytes (no
    Python work per word). A word is a run of letters, digits, _ or non-ASCII
    bytes; its hash is a polynomial over its bytes, taken from prefix sums:
    hash(s..e) = (S[e] - S[s]) * P^-s with S[i] = sum_k<i (b_k + 1) P^k, all
    modulo 2^64. Deterministic in every process (no salted Python hash)."""

    def word_hashes(self, text):
        b = np.frombuffer(text.lower().encode("utf-8"), dtype=np.uint8)
        if len(b) == 0:
            return np.zeros(0, dtype=np.uint64)
        edges = np.diff(np.concatenate(([False], _WORD_BYTE[b], [False])).astype(np.int8))
        starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
        if len(starts) == 0:
            return np.zeros(0, dtype=np.uint64)
        with np.errstate(over="ignore"):
            prefix = np.concatenate(([np.uint64(0)], np.cumsum((b.astype(np.uint64) + np.uint64(1))
                                                                * _powers("p", len(b))[:len(b)])))
            raw = (prefix[ends] - prefix[starts]) * _powers("inv", int(starts[-1]) + 1)[starts]
        return _mix64(raw ^ (ends - starts).astype(np.uint64))

    def shingles(self, text, n):
        words = self.word_hashes(text)
        if len(words) == 0:
            return words
        if len(words) < n:
            return _mix64(np.asarray([int(np.bitwise_xor.reduce(words))], dtype=np.uint64))
        m = len(words) - n + 1
        with np.errstate(over="ignore"):
            h = np.zeros(m, dtype=np.uint64)
            for j in range(n):
                h = h * _P + words[j:j + m]
        return _mix64(h)


class MinHasher:
    """One-permutation MinHash (Li, Owen & Zhang 2012): each shingle hash is used
    once; its top bits pick one of num_perm buckets and its low 32 bits compete
    for that bucket's minimum. Empty buckets (short documents) borrow the next
    filled bucket's value with an offset (rotation densification, Shrivastava &
    Li 2014), so similarity estimates stay unbiased. On PubMed Central papers it
    flagged the same near-duplicates as 128 separate hash functions, with the
    same error, at a fraction of the cost."""

    EMPTY = np.uint64(0xFFFFFFFF)

    def __init__(self, num_perm=128, seed=1):
        if num_perm & (num_perm - 1):
            raise ValueError("num_perm must be a power of 2")
        self.num_perm = num_perm
        self.shift = np.uint64(64 - (num_perm.bit_length() - 1))
        self.salt = _mix64(np.asarray([seed], dtype=np.uint64))[0]

    def signature(self, shingles):
        k = self.num_perm
        sig = np.full(k, self.EMPTY, dtype=np.uint64)
        if len(shingles) == 0:
            return sig.astype(np.uint32)
        h = _mix64(shingles ^ self.salt)
        np.minimum.at(sig, (h >> self.shift).astype(np.intp), h & np.uint64(0xFFFFFFFF))
        empty = np.nonzero(sig == self.EMPTY)[0]
        if len(empty):
            full = np.nonzero(sig != self.EMPTY)[0]
            donor = full[np.searchsorted(full, empty) % len(full)]
            with np.errstate(over="ignore"):
                sig[empty] = (sig[donor] + ((donor - empty) % k).astype(np.uint64) * np.uint64(0x9E3779B1)) \
                    & np.uint64(0xFFFFFFFF)
        return sig.astype(np.uint32)


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)


def lsh_clusters(signatures, threshold, bands=16, active=None):
    """Near-duplicate clusters. Per band, documents whose band rows are identical
    share a bucket; each member is compared with the bucket's first (lowest-index)
    document and joined if their full signatures agree on >= threshold of rows.
    Buckets come from sorting the band rows (numpy), not a Python loop per document.
    `active` (ascending row indices) limits the search; other rows stay alone."""
    n, k = signatures.shape
    rows = k // bands
    idx = np.arange(n) if active is None else np.asarray(active, dtype=np.int64)
    m = len(idx)
    uf = UnionFind(n)
    for band in range(bands):
        block = np.ascontiguousarray(signatures[idx, band * rows:(band + 1) * rows])
        keys = block.view(np.dtype((np.void, block.dtype.itemsize * rows))).ravel()
        local = np.argsort(keys, kind="stable")          # equal keys stay in index order
        sorted_keys = keys[local]
        order = idx[local]
        new_group = np.ones(m, dtype=bool)
        new_group[1:] = sorted_keys[1:] != sorted_keys[:-1]
        first_of = order[np.maximum.accumulate(np.where(new_group, np.arange(m), 0))]
        members = np.nonzero(~new_group)[0]
        if not len(members):
            continue
        i_idx, f_idx = order[members], first_of[members]
        for start in range(0, len(i_idx), 100_000):
            a, b = i_idx[start:start + 100_000], f_idx[start:start + 100_000]
            sim = (signatures[a] == signatures[b]).mean(axis=1)
            for i, f in zip(a[sim >= threshold].tolist(), b[sim >= threshold].tolist()):
                uf.union(f, i)
    return [uf.find(i) for i in range(n)]


# ---------------------------------------------------------------------------
# Main preparation
# ---------------------------------------------------------------------------
def _ledger_row(source, doc_id, title, status, reason, chars=0, tokens=0, hits=0, group=""):
    return {"source": source, "doc_id": doc_id, "title": title, "status": status, "reason": reason,
            "chars": chars, "tokens": tokens, "keyword_hits": hits, "group": group}


def _new_stats():
    return {"scanned": 0, "accepted": 0, "skipped": collections.Counter(),
            "cleaning": collections.Counter(), "stopped_early": False, "ledger_rows": 0}


_W_KW, _W_CFG = None, {}  # set in each worker process (and in the main process) by _init_worker


def _init_worker(kw, cfg):
    global _W_KW, _W_CFG
    _W_KW, _W_CFG = kw, cfg
    try:
        _lower_priority()
    except Exception:  # noqa: BLE001
        pass


def _examine(title, text, use_kw):
    """Everything done to one document that needs no shared state: cleaning,
    language, replacement characters, length, keywords, the exact-duplicate key
    and the boilerplate line keys. Pure, so it can run in worker processes; the
    main process applies the stateful steps (duplicates, counts, spool) in order."""
    cfg = _W_CFG
    cleaning = collections.Counter()
    raw_chars = len(text)
    text = clean_text(text, cleaning)
    reason, hits, terms = "", 0, collections.Counter()
    bad = text.count("\ufffd")
    if cfg["language"] == "en" and text and not looks_english(text):
        reason = "not_english"
    elif bad and bad / max(len(text), 1) > cfg["max_replacement_ratio"]:
        reason = "replacement_chars"
    elif bad:
        text = strip_replacement_chars(text, cleaning)
    if reason:
        pass
    elif not text:
        reason = "empty"
    elif len(text) < cfg["min_chars"]:
        reason = "too_short"
    elif use_kw and _W_KW is not None:
        keep, why, hits, terms = _W_KW.evaluate(title, text)
        if not keep:
            reason = why
    key, line_keys = None, []
    if not reason:
        if cfg["exact_dedup"] == "skip":
            key = normalised_hash(text)
        seen = set()
        for idx, line in enumerate(text.split("\n") if cfg["boilerplate"] != "off" else ()):
            stripped = line.strip()
            if len(stripped) >= cfg["boilerplate_min_len"] and not _structural_line(stripped):
                k = _line_key(stripped)
                if k not in seen:
                    seen.add(k)
                    line_keys.append((k, idx))
    return reason, text, hits, terms, key, line_keys, cleaning, raw_chars


def _examine_chunk(jobs):
    return [_examine(title, text, use_kw) for title, text, use_kw in jobs]


class _Pool:
    """Worker processes for pure per-document functions. The same initializer
    runs in this process too, so small inputs (and the first `warmup` items)
    are handled here and the processes start only when a source proves large."""

    def __init__(self, workers, init, initargs, warmup=2_000):
        self.workers, self.init, self.initargs, self.warmup, self.pool = workers, init, initargs, warmup, None
        init(*initargs)

    def get(self):
        if self.pool is None and self.workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            self.pool = ProcessPoolExecutor(max_workers=self.workers, initializer=self.init, initargs=self.initargs)
            print(f"  (using {self.workers} worker processes)", flush=True)
        return self.pool

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()
            self.pool = None


def _ordered(items, job_of, run_chunk, pool, chunk=64, size_of=None, max_chars=4_000_000):
    """Yield (item, result) in the original order. Results never depend on which
    process computed them, so the output is identical for any number of workers.
    A chunk holds up to `chunk` items or `max_chars` characters (size_of(item))
    and at most 3 chunks per worker are in flight, so memory stays flat even for
    whole books."""
    n, pending, batch, batch_chars, ex = 0, collections.deque(), [], 0, None
    for item in items:
        if ex is None:
            if n < pool.warmup or pool.workers <= 1:
                n += 1
                yield item, run_chunk([job_of(item)])[0]
                continue
            ex = pool.get()
        batch.append(item)
        if size_of is not None:
            batch_chars += size_of(item)
        if len(batch) >= chunk or batch_chars >= max_chars:
            pending.append((batch, ex.submit(run_chunk, [job_of(i) for i in batch])))
            batch, batch_chars = [], 0
            while len(pending) >= 3 * pool.workers:
                done, future = pending.popleft()
                yield from zip(done, future.result())
    if batch:
        pending.append((batch, ex.submit(run_chunk, [job_of(i) for i in batch])))
    while pending:
        done, future = pending.popleft()
        yield from zip(done, future.result())


_P2 = {}


def _init_pass2(boiler, mode, min_len, min_chars, near, seed):
    global _P2
    _P2 = {"boiler": boiler, "mode": mode, "min_len": min_len, "min_chars": min_chars, "near": near,
           "hasher": ShingleHasher(), "minhasher": MinHasher(128, seed)}
    try:
        _lower_priority()
    except Exception:  # noqa: BLE001
        pass


def _pass2_doc(source, text):
    """Boilerplate stripping and the MinHash signature of one document (pure)."""
    p = _P2
    removed = []
    table = p["boiler"].get(source)
    if table and p["mode"] != "off":
        kept_lines = []
        for ln in text.split("\n"):
            stripped = ln.strip()
            if len(stripped) >= p["min_len"] and not _structural_line(stripped):
                k = _line_key(stripped)
                if k in table:
                    removed.append(k)
                    if p["mode"] == "strip":
                        continue
            kept_lines.append(ln)
        text = _BLANK_RUN_RE.sub("\n\n", "\n".join(kept_lines)).strip()
    sig = None
    if p["near"] and len(text) >= p["min_chars"]:
        sig = p["minhasher"].signature(p["hasher"].shingles(text, 5))
    return text, removed, sig


def _pass2_chunk(jobs):
    return [_pass2_doc(source, text) for source, text in jobs]


def _structural_line(line):
    """Lines never counted as boilerplate: no letters or digits (rules, table
    borders), Markdown headings ("## Materials and methods" repeats in thousands of
    papers but is structure, not boilerplate) and short ALL-CAPS headings."""
    return (not re.search(r"[A-Za-z0-9]", line) or line.startswith("#")
            or (line.isupper() and len(line.split()) <= 8))


def _line_key(line):
    return hashlib.blake2b(" ".join(line.lower().split()).encode("utf-8"), digest_size=8).digest()


def _lower_priority():
    """Below-normal priority, so the PC stays responsive. Never raises it: a
    process started at idle (or a worker of one) stays idle."""
    try:
        import psutil
        proc = psutil.Process()
        if sys.platform == "win32":
            if proc.nice() != psutil.IDLE_PRIORITY_CLASS:
                proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        elif proc.nice() < 10:
            proc.nice(10)
    except Exception:  # noqa: BLE001 - priority is a courtesy, never fatal
        pass


def auto_vocab_size(train_chars):
    if train_chars < 20e6:
        return 4096
    if train_chars < 1.5e9:
        return 8192
    return 16384


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output dataset folder (must not exist unless --overwrite)")
    p.add_argument("--overwrite", action="store_true",
                   help="rename an existing --out folder to <out>.bak-<time> and rebuild")
    g = p.add_argument_group("Markdown / text files")
    g.add_argument("--md-dir", action="append", default=[],
                   help="folder of .md/.rmd/.txt files (repeatable). NAME=FOLDER makes it a separate source with "
                        "its own validation loss and sampling weight, e.g. papers=D:\\converted\\papers")
    g.add_argument("--md-glob", action="append", default=[], help=f"file patterns (default {MD_PATTERNS_DEFAULT})")
    g.add_argument("--md-exclude", action="append", default=[], help="glob to skip, e.g. *.lt.md (repeatable)")
    g.add_argument("--md-recursive", action="store_true", help="also search sub-folders")
    g.add_argument("--text-root", help="folder whose sub-folders (books, articles, scripts, ...) each become a "
                                       "named source; wikipedia, web, images and _-prefixed folders are skipped")
    g = p.add_argument_group("Hugging Face save_to_disk folder (Wikipedia)")
    g.add_argument("--wiki-dir", help="folder written by datasets.save_to_disk")
    g.add_argument("--wiki-split", default="train")
    g.add_argument("--wiki-max-docs", type=int, default=0, help="max documents kept (0 = all that pass)")
    g.add_argument("--wiki-select", choices=("auto", "first", "top"), default="auto",
                   help="first = dump order; top = rank all by keyword density (auto: top when keywords are set)")
    g = p.add_argument_group("Parquet / JSONL shards (FineWeb-Edu etc.)")
    g.add_argument("--parquet", action="append", default=[], help="glob of .parquet files (repeatable)")
    g.add_argument("--parquet-name", default="web", help="source name used in reports and token files")
    g.add_argument("--parquet-max-docs", type=int, default=0)
    g.add_argument("--parquet-select", choices=("auto", "first", "top"), default="first")
    g.add_argument("--jsonl", action="append", default=[], help="glob of .jsonl / .jsonl.gz / .json.gz files "
                                                                 "(repeatable), e.g. peS2o shards")
    g.add_argument("--jsonl-name", default="jsonl", help="source name for --jsonl data, e.g. pes2o")
    g.add_argument("--jsonl-max-docs", type=int, default=0)
    g.add_argument("--text-column", default="text")
    g.add_argument("--title-column", default="title")
    g.add_argument("--id-column", default="id")
    g.add_argument("--score-column", default="int_score", help="FineWeb-Edu quality column")
    g.add_argument("--min-score", type=float, help="keep Parquet rows with score-column >= this")
    g.add_argument("--sample-fraction", type=float, default=1.0,
                   help="deterministic hash sample of streamed web records (first-mode only)")
    g = p.add_argument_group("Keyword filter (case-insensitive, whole word, * = any ending)")
    g.add_argument("--include-keywords", help="file (one term per line) or comma-separated terms")
    g.add_argument("--exclude-keywords", help="file or comma-separated terms; any match skips the record")
    g.add_argument("--keyword-sources", default="web",
                   help="'web' = every source except Markdown/text folders (default), 'all', or names like wiki,web")
    g.add_argument("--keyword-min-hits", type=int, default=3)
    g.add_argument("--keyword-min-distinct", type=int, default=2)
    g.add_argument("--title-weight", type=int, default=3)
    g = p.add_argument_group("Cleaning and deduplication")
    g.add_argument("--min-chars", type=int, default=200)
    g.add_argument("--language", choices=("en", "any"), default="en",
                   help="en (default): skip documents that are clearly not English (few English function words and "
                        "non-English letters such as \u0105 \u010d \u0117 \u0161); any: keep every language")
    g.add_argument("--max-doc-chars", type=int, default=100_000,
                   help="longer documents are split into parts (same split, never discarded)")
    g.add_argument("--max-replacement-ratio", type=float, default=0.02,
                   help="skip records whose share of U+FFFD characters exceeds this; below it, "
                        "mostly-garbled lines and stray U+FFFD are removed (counted in the report)")
    g.add_argument("--boilerplate", choices=("strip", "report", "off"), default="strip")
    g.add_argument("--boilerplate-min-docs", type=int, default=20)
    g.add_argument("--boilerplate-min-len", type=int, default=20)
    g.add_argument("--exact-dedup", choices=("skip", "keep"), default="skip")
    g.add_argument("--near-dup", choices=("group", "drop", "off"), default="group")
    g.add_argument("--near-dup-threshold", type=float, default=0.8)
    g.add_argument("--near-dup-max-docs", type=int, default=2_000_000)
    g = p.add_argument_group("Split and tokenizer")
    g.add_argument("--val-fraction", type=float, default=0.05)
    g.add_argument("--val-max-chars", type=int, default=20_000_000, help="per-source cap on validation text")
    g.add_argument("--max-val-group-share", type=float, default=0.25,
                   help="groups bigger than this share of a source's validation budget stay in train")
    g.add_argument("--seed", type=int, default=1234)
    g.add_argument("--vocab-size", type=int, default=0, help="0 = automatic from training characters")
    g.add_argument("--tokenizer-sample-chars", type=int, default=60_000_000)
    g.add_argument("--tokenizer-weights",
                   help="over-sample sources in the tokenizer's training text, e.g. md=4 (default: proportional). "
                        "Domain words then get their own tokens instead of being split into pieces")
    g.add_argument("--tokenizer-from", help="reuse a tokenizer: .model file, dataset folder or tiny_gpt checkpoint")
    g.add_argument("--superbpe", action="store_true",
                   help="SuperBPE tokenizer: after ordinary BPE, the last --superbpe-fraction of the vocabulary is "
                        "learnt with word boundaries open, giving tokens such as 'of the' (fewer tokens per text)")
    g.add_argument("--superbpe-fraction", type=float, default=0.1,
                   help="share of the vocabulary for SuperBPE's multi-word stage (default 0.1, the paper's best)")
    g.add_argument("--superbpe-max-words", type=int, default=4, help="most words in one SuperBPE token (default 4)")
    g.add_argument("--keep-split-from", help="earlier dataset folder: its training documents stay in training and its "
                                             "validation documents in validation (for training further from a model "
                                             "built on it); only new documents are split")
    g.add_argument("--leakage-check", choices=("on", "off"), default="on")
    g.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 2),
                   help="threads for the C++ keyword prefilter and SentencePiece (default: all CPU threads)")
    g.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                   help="processes that check documents in parallel (1 = all in this process)")
    g.add_argument("--worker-warmup", type=int, default=2_000, help=argparse.SUPPRESS)
    g.add_argument("--scan-only", action="store_true", help="stop after filtering/dedup; write the report only")
    g.add_argument("--keep-work", action="store_true", help="keep the _work spool files")
    args = p.parse_args(argv)
    _validate(args, p)
    return args


def _validate(args, parser):
    if args.text_root:
        if not os.path.isdir(args.text_root):
            parser.error(f"--text-root not found: {args.text_root}")
        for name in sorted(os.listdir(args.text_root)):
            folder = os.path.join(args.text_root, name)
            if (os.path.isdir(folder) and not name.startswith((".", "_"))
                    and name.lower() not in ("wikipedia", "web", "images")
                    and re.fullmatch(SOURCE_NAME_RE, name)):
                args.md_dir.append(f"{name}={folder}")
        args.md_recursive = True
    if not (args.md_dir or args.wiki_dir or args.parquet or args.jsonl):
        parser.error("give at least one source: --md-dir, --wiki-dir, --parquet or --jsonl")
    for flag, value in (("--parquet-name", args.parquet_name), ("--jsonl-name", args.jsonl_name)):
        if not re.fullmatch(SOURCE_NAME_RE, value):
            parser.error(f"{flag} must be a short identifier (letters, digits, _ or -)")
    if args.min_score is not None and not args.parquet:
        parser.error("--min-score applies to --parquet sources only")
    checks = [
        (0 < args.val_fraction < 0.5, "--val-fraction must be between 0 and 0.5"),
        (0 < args.sample_fraction <= 1, "--sample-fraction must be in (0, 1]"),
        (0.5 <= args.near_dup_threshold <= 1, "--near-dup-threshold must be in [0.5, 1]"),
        (args.min_chars >= 0, "--min-chars must be >= 0"),
        (args.max_doc_chars >= 1000, "--max-doc-chars must be >= 1000"),
        (args.vocab_size == 0 or 512 <= args.vocab_size <= 262_144, "--vocab-size must be 0 or 512..262144"),
        (args.threads >= 1, "--threads must be >= 1"),
        (args.workers >= 1, "--workers must be >= 1"),
        (0 <= args.max_replacement_ratio <= 1, "--max-replacement-ratio must be in [0, 1]"),
        (min(args.wiki_max_docs, args.parquet_max_docs, args.jsonl_max_docs) >= 0, "max-docs must be >= 0"),
    ]
    for ok, message in checks:
        if not ok:
            parser.error(message)
    if args.wiki_dir and not os.path.isdir(args.wiki_dir):
        parser.error(f"--wiki-dir not found: {args.wiki_dir}")
    if args.tokenizer_from and not os.path.exists(args.tokenizer_from):
        parser.error(f"--tokenizer-from not found: {args.tokenizer_from}")
    # file patterns are expanded by Python, not the shell: check them now, not after an hour of reading.
    # Git Bash passes /c/Users/... for patterns with * (it converts only plain paths): accept that form.
    for flag, patterns in (("--parquet", args.parquet), ("--jsonl", args.jsonl)):
        for i, pattern in enumerate(patterns):
            m = re.match(r"^/([A-Za-z])/(.*)$", pattern) if sys.platform == "win32" else None
            if m and not glob.glob(pattern):
                patterns[i] = pattern = f"{m.group(1).upper()}:/{m.group(2)}"
            if not glob.glob(pattern):
                parser.error(f"{flag}: no files match {pattern}")
    if args.superbpe and args.tokenizer_from:
        parser.error("--superbpe trains a new tokenizer; a reused one (--tokenizer-from) keeps its own pieces")
    if not 0 < args.superbpe_fraction < 0.5:
        parser.error("--superbpe-fraction must be between 0 and 0.5")
    if not 2 <= args.superbpe_max_words <= 8:
        parser.error("--superbpe-max-words must be between 2 and 8")
    if args.keep_split_from and not os.path.isfile(os.path.join(args.keep_split_from, "docs.tsv")):
        parser.error(f"--keep-split-from: no docs.tsv in {args.keep_split_from}")


def prepare(argv=None):
    """Build a dataset; returns its folder. An interrupted or failed build
    deletes its work spool (the partial folder is rebuilt on the next run)."""
    args = parse_args(argv)
    state = {}
    try:
        return _prepare(args, state)
    except BaseException:
        work = state.get("work")  # only a spool this run created (never one kept by an earlier --keep-work)
        if work and os.path.isdir(work) and not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)
            print(f"Build stopped; removed the work spool {work}. Rerun the same command to rebuild.")
        raise


def _prepare(args, state=None):
    state = {} if state is None else state
    t_start = time.time()
    _lower_priority()
    out = os.path.abspath(args.out)
    if os.path.exists(out) and os.listdir(out) and _is_incomplete_build(out):
        print(f"Removing an incomplete earlier build in {out} (no manifest.json; only data_prep outputs).")
        shutil.rmtree(out)
    if os.path.exists(out) and os.listdir(out):
        if not args.overwrite:
            raise SystemExit(f"{out} already exists and is not empty. Choose a new --out or pass --overwrite "
                             "(the old folder is renamed, not deleted).")
        backup = f"{out}.bak-{datetime.datetime.now():%Y%m%d-%H%M%S}"
        os.replace(out, backup)
        print(f"Existing dataset moved to {backup}")
    os.makedirs(out, exist_ok=True)
    work = os.path.join(out, "_work")
    os.makedirs(work, exist_ok=True)
    state["work"] = work
    print(f"Preparing dataset in {out}")

    include = read_terms(args.include_keywords)
    exclude = read_terms(args.exclude_keywords)
    kw = KeywordFilter(include, exclude, args.keyword_min_hits, args.keyword_min_distinct, args.title_weight)
    if kw.active:
        print(f"Keyword filter: {len(include)} include terms, {len(exclude)} exclude terms, "
              f"min hits {args.keyword_min_hits}, min distinct {args.keyword_min_distinct}, "
              f"sources: {args.keyword_sources}")

    # ------------------------------------------------------------------
    # Pass 1: read, clean, filter, exact dedup -> spool1
    # ------------------------------------------------------------------
    sources = {}
    ledger = []
    exact_seen = {}
    duplicates = []
    line_counts = collections.defaultdict(collections.Counter)
    line_examples = {}
    spool1 = os.path.join(work, "spool1.jsonl")
    keyword_totals = collections.defaultdict(collections.Counter)
    notes = []

    text_sources = parse_text_sources(args.md_dir)
    spec = args.keyword_sources.strip().lower()
    wanted = None if spec in ("all", "web") else {s.strip() for s in spec.split(",") if s.strip()}

    def filtered(name, kind):
        if spec == "all":
            return True
        if spec == "web":
            return kind != "markdown"
        return name.lower() in wanted

    readers = []
    for text_name in text_sources:
        readers.append((text_name, "markdown"))
    if args.wiki_dir:
        readers.append(("wiki", "hf_arrow"))
    if args.parquet:
        readers.append((args.parquet_name, "parquet"))
    if args.jsonl:
        readers.append((args.jsonl_name, "jsonl"))
    names = [n for n, _ in readers]
    if len({n.lower() for n in names}) != len(names):  # train.Books.bin and train.books.bin are one file on Windows
        raise SystemExit(f"Two sources share a name, ignoring case ({names}); rename a NAME=FOLDER source, "
                         "--parquet-name or --jsonl-name.")
    if wanted is not None and wanted - {n.lower() for n in names}:
        raise SystemExit(f"--keyword-sources: unknown source(s) {sorted(wanted - {n.lower() for n in names})}; "
                         f"the sources are {names} (or use 'web' or 'all').")

    workers = _Pool(args.workers, _init_worker, (kw, {
        "language": args.language, "max_replacement_ratio": args.max_replacement_ratio,
        "min_chars": args.min_chars, "exact_dedup": args.exact_dedup, "boilerplate": args.boilerplate,
        "boilerplate_min_len": args.boilerplate_min_len}), args.worker_warmup)
    with open(spool1, "w", encoding="utf-8", newline="\n") as spool:
        for name, kind in readers:
            stats = _new_stats()
            sources[name] = {"kind": kind, "stats": stats}
            use_kw = kw if (kw.active and filtered(name, kind)) else None
            print(f"\n[{name}] reading ({kind}){' with keyword filter' if use_kw else ''} ...", flush=True)
            t0 = time.time()
            if kind == "markdown":
                patterns = args.md_glob or list(MD_PATTERNS_DEFAULT)
                stream = iter_markdown(text_sources[name], patterns, args.md_exclude, args.md_recursive, stats,
                                       ledger, name)
                sources[name]["paths"] = text_sources[name]
                max_docs = 0
            elif kind in ("hf_arrow", "parquet"):
                if kind == "hf_arrow":
                    paths = ArrowTableSource.hf_files(args.wiki_dir, args.wiki_split)
                    max_docs, select = args.wiki_max_docs, args.wiki_select
                    min_score = None
                    sources[name]["paths"] = [args.wiki_dir]
                else:
                    paths = sorted({p for pattern in args.parquet for p in glob.glob(pattern)})
                    if not paths:
                        raise SystemExit(f"No Parquet files match {args.parquet}")
                    max_docs, select, min_score = args.parquet_max_docs, args.parquet_select, args.min_score
                    sources[name]["paths"] = args.parquet
                if select == "auto":
                    select = "top" if (use_kw and use_kw.include_patterns and max_docs) else "first"
                sources[name]["select"] = select
                table = ArrowTableSource(kind, paths, args.text_column, args.title_column, args.id_column,
                                         args.score_column if kind == "parquet" else None,
                                         require_score=kind == "parquet" and args.min_score is not None)
                stream = iter_table_source(name, table, use_kw, max_docs, select, min_score,
                                           args.sample_fraction, args.seed, stats, args.threads)
            else:
                paths = sorted({p for pattern in args.jsonl for p in glob.glob(pattern)})
                if not paths:
                    raise SystemExit(f"No JSONL files match {args.jsonl}")
                sources[name]["paths"] = args.jsonl
                max_docs = args.jsonl_max_docs
                stream = iter_jsonl(paths, args.text_column, args.title_column, args.id_column, max_docs, stats,
                                    use_kw, name=name, threads=args.threads)

            sources[name]["max_docs"] = max_docs
            kept = 0
            ranked = sources[name].get("select") == "top"
            flag = use_kw is not None
            for (doc_id, title, _raw, meta), found in _ordered(stream, lambda it: (it[1], it[2], flag),
                                                               _examine_chunk, workers,
                                                               size_of=lambda it: len(it[2])):
                reason, text, hits, terms, key, line_keys, cleaning, raw_chars = found
                stats["cleaning"].update(cleaning)
                if max_docs and kept >= max_docs and not ranked:
                    stats["skipped"]["over_max_docs"] += 1
                    continue
                if not reason and key is not None:
                    first = exact_seen.get(key)
                    if first is not None:
                        reason = "exact_duplicate"
                        duplicates.append(("exact", name, doc_id, title, first[0], first[1], first[2], 1.0))
                    else:
                        exact_seen[key] = (name, doc_id, title)
                if reason:
                    stats["skipped"][reason] += 1
                    if kind == "markdown" or stats["ledger_rows"] < 2000:
                        stats["ledger_rows"] += 1
                        ledger.append(_ledger_row(name, doc_id, title, "skipped", reason, raw_chars, 0, hits))
                    continue
                keyword_totals[name].update(terms)
                kept += 1
                stats["accepted"] += 1
                counts, lines = line_counts[name], None
                for k, idx in line_keys:
                    counts[k] += 1
                    if counts[k] == 2 and len(line_examples) < 2_000_000:
                        if lines is None:
                            lines = text.split("\n")
                        line_examples[k] = lines[idx].strip()[:200]
                if len(counts) > 6_000_000:  # prune to half, so this does not run again for every document
                    floor = 1
                    while len(counts) > 3_000_000:
                        counts = collections.Counter({k: c for k, c in counts.items() if c > floor})
                        floor += 1
                    line_counts[name] = counts
                json.dump({"s": name, "id": doc_id, "t": title, "x": text, "h": hits,
                           "d": len(terms), "r": meta.get("rank")}, spool, ensure_ascii=False)
                spool.write("\n")
            print(f"[{name}] scanned {stats['scanned']:,} | kept {stats['accepted']:,} | skipped "
                  f"{sum(stats['skipped'].values()):,} {dict(stats['skipped'].most_common())} "
                  f"({time.time() - t0:,.0f} s)", flush=True)

    workers.close()
    del exact_seen

    # ------------------------------------------------------------------
    # Pass 2: boilerplate strip + MinHash -> spool2
    # ------------------------------------------------------------------
    boiler = {}
    for name, counter in line_counts.items():
        threshold = max(args.boilerplate_min_docs, 2)
        boiler[name] = {k: c for k, c in counter.items() if c >= threshold}
    del line_counts
    line_examples = {k: line_examples[k] for table in boiler.values() for k in table if k in line_examples}
    removed_lines = collections.Counter()
    total_docs = sum(s["stats"]["accepted"] for s in sources.values())
    near_enabled = args.near_dup != "off" and 1 < total_docs <= args.near_dup_max_docs
    if args.near_dup != "off" and total_docs > args.near_dup_max_docs:
        notes.append(f"Near-duplicate detection was skipped: {total_docs:,} documents exceed --near-dup-max-docs "
                     f"{args.near_dup_max_docs:,}.")
    signatures = np.zeros((total_docs, 128), dtype=np.uint32) if near_enabled else None
    docs = []  # compact per-doc metadata: (source, id, title, chars, hits, distinct)
    spool2 = os.path.join(work, "spool2.jsonl")
    print(f"\nPass 2: boilerplate ({args.boilerplate}) and near-duplicate signatures "
          f"({'on' if near_enabled else 'off'}) over {total_docs:,} documents ...", flush=True)
    t0 = time.time()
    pass2 = _Pool(args.workers, _init_pass2, (boiler, args.boilerplate, args.boilerplate_min_len, args.min_chars,
                                              near_enabled, args.seed), args.worker_warmup)

    with open(spool1, encoding="utf-8") as src, open(spool2, "w", encoding="utf-8", newline="\n") as dst:
        for rec, (text, removed, sig) in _ordered((json.loads(line) for line in src), lambda r: (r["s"], r["x"]),
                                                  _pass2_chunk, pass2, size_of=lambda r: len(r["x"])):
            for k in removed:
                removed_lines[(rec["s"], k)] += 1
            if len(text) < args.min_chars:
                sources[rec["s"]]["stats"]["skipped"]["too_short_after_boilerplate"] += 1
                sources[rec["s"]]["stats"]["accepted"] -= 1
                ledger.append(_ledger_row(rec["s"], rec["id"], rec["t"], "skipped",
                                          "too_short_after_boilerplate", len(rec["x"]), 0, rec["h"]))
                continue
            idx = len(docs)
            if near_enabled:
                signatures[idx] = sig
            docs.append([rec["s"], rec["id"], rec["t"], len(text), rec["h"], rec["d"], rec.get("r")])
            rec["x"] = text
            json.dump(rec, dst, ensure_ascii=False)
            dst.write("\n")
    pass2.close()
    del boiler
    if near_enabled:
        signatures = signatures[:len(docs)]
    print(f"Pass 2 done ({time.time() - t0:,.0f} s)", flush=True)

    # 'top' selection: the max_docs best-ranked documents that survived every filter (boilerplate included)
    cut = set()
    for name, info in sources.items():
        limit = info.get("max_docs")
        if info.get("select") != "top" or not limit:
            continue
        ranked = sorted((docs[i][6], i) for i in range(len(docs)) if docs[i][0] == name)
        cut.update(i for _, i in ranked[limit:])
        info["stats"]["skipped"]["over_max_docs_by_rank"] += max(0, len(ranked) - limit)
        info["stats"]["accepted"] -= max(0, len(ranked) - limit)
    for name, info in sources.items():
        limit, st = info.get("max_docs"), info["stats"]
        if limit and st["accepted"] < limit:
            notes.append(f"`{name}` kept {st['accepted']:,} of the {limit:,} documents asked for: " + (
                "reading stopped at the limit and boilerplate removal then left some too short."
                if st["stopped_early"] else "not enough records passed the filters."))

    # clusters
    roots = list(range(len(docs)))
    if near_enabled and len(docs) - len(cut) > 1:
        roots = lsh_clusters(signatures, args.near_dup_threshold,
                             active=[i for i in range(len(docs)) if i not in cut] if cut else None)
    members = collections.defaultdict(list)
    for i, r in enumerate(roots):
        members[r].append(i)
    clusters = {r: m for r, m in members.items() if len(m) > 1}
    near_dropped = set()
    for r, m in clusters.items():
        for i in m:
            if i != r:
                est = float(np.mean(signatures[i] == signatures[r])) if near_enabled else 1.0
                duplicates.append(("near", docs[i][0], docs[i][1], docs[i][2], docs[r][0], docs[r][1], docs[r][2], est))
                if args.near_dup == "drop":
                    near_dropped.add(i)
    dropped = cut | near_dropped
    for i in sorted(near_dropped):
        s = docs[i][0]
        sources[s]["stats"]["skipped"]["near_duplicate"] += 1
        sources[s]["stats"]["accepted"] -= 1
        ledger.append(_ledger_row(s, docs[i][1], docs[i][2], "skipped", "near_duplicate", docs[i][3], 0, docs[i][4]))
    print(f"Near-duplicate clusters: {len(clusters):,} "
          f"({sum(len(m) for m in clusters.values()):,} documents; action: {args.near_dup})")

    # ------------------------------------------------------------------
    # Split by group, per source
    # ------------------------------------------------------------------
    group_of = {}
    for i, r in enumerate(roots):
        group_of[i] = f"{docs[r][0]}:{docs[r][1]}"
    cross_source = {r for r, m in clusters.items() if len({docs[i][0] for i in m}) > 1}
    split_of = {}
    forced_train = collections.Counter()
    previous = _previous_split(args.keep_split_from) if args.keep_split_from else {}
    kept_split = collections.Counter()
    for name in sources:
        idxs = [i for i in range(len(docs)) if docs[i][0] == name and i not in dropped]
        groups = collections.defaultdict(list)  # cluster root -> members (labels alone could collide)
        for i in idxs:
            groups[roots[i]].append(i)
        total_chars = sum(docs[i][3] for i in idxs)
        budget = min(args.val_fraction * total_chars, args.val_max_chars)
        order = sorted(groups, key=lambda g: stable_unit(args.seed, "split", group_of[g]))
        val_chars = 0
        if previous:
            # --keep-split-from: a document the earlier model trained on must never become validation (its loss
            # there would look too good); earlier validation documents stay validation. Only new documents are split.
            fresh = []
            for g in order:
                before = {previous.get((docs[i][0], str(docs[i][1]))) for i in groups[g]}
                target = "train" if "train" in before else ("val" if "val" in before else None)
                if target is None:
                    fresh.append(g)
                    continue
                kept_split[target] += len(groups[g])
                for i in groups[g]:
                    split_of[i] = target
                if target == "val":
                    val_chars += sum(docs[i][3] for i in groups[g])
            order = fresh
        for g in order:
            chars = sum(docs[i][3] for i in groups[g])
            target = "train"
            if len(groups) >= 2 and val_chars < budget:
                if g in cross_source:
                    forced_train["cross_source_cluster"] += 1
                elif chars > args.max_val_group_share * budget:
                    forced_train["group_too_large_for_val"] += 1
                elif val_chars + chars <= budget * 1.1:
                    target = "val"
                    val_chars += chars
            for i in groups[g]:
                split_of[i] = target
        sources[name]["val_chars_target"] = int(budget)
        sources[name]["val_chars"] = int(val_chars)
        sources[name]["train_chars"] = int(total_chars - val_chars)

    if previous:
        notes.append(f"Split kept from {args.keep_split_from}: {kept_split['train']:,} documents stay in training and "
                     f"{kept_split['val']:,} in validation as before; only new documents were split.")
    nothing_kept = len(docs) == len(dropped)
    if nothing_kept and not args.scan_only:
        notes.append("No document passed the filters, so nothing was tokenized.")
    if args.scan_only or nothing_kept:
        _write_report(out, args, sources, ledger, duplicates, removed_lines, line_examples, keyword_totals,
                      None, {}, forced_train, [], None, t_start, scan_only=args.scan_only, notes=notes)
        _write_ledger(out, ledger, docs, split_of, group_of, dropped, {})
        _cleanup(work, args.keep_work)
        if nothing_kept and not args.scan_only:
            raise SystemExit(f"No document passed the filters; see {os.path.join(out, 'report.md')} and docs.tsv. "
                             "Loosen the filters and rebuild into a new --out (or pass --overwrite).")
        print(f"\nScan-only report written to {os.path.join(out, 'report.md')}")
        return out

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    train_chars_total = sum(s.get("train_chars", 0) for s in sources.values())
    if args.tokenizer_from:
        tok = load_tokenizer_from_any(args.tokenizer_from)
        print(f"\nReusing tokenizer from {args.tokenizer_from} ({tok.encode_mode} mode, {tok.vocab_size:,} pieces)")
        tokenizer_info = {"trained": False, "from": os.path.abspath(args.tokenizer_from)}
    else:
        vocab = args.vocab_size or auto_vocab_size(train_chars_total)
        tok, tokenizer_info = _train_tokenizer(spool2, docs, split_of, sources, vocab, args, work)
    with open(os.path.join(out, "tokenizer.model"), "wb") as handle:
        handle.write(tok.proto)
    dtype = np.uint16 if tok.vocab_size <= 65536 else np.uint32

    # ------------------------------------------------------------------
    # Pass 3: encode and write token files
    # ------------------------------------------------------------------
    print(f"\nPass 3: tokenizing ({tok.vocab_size:,} pieces, {np.dtype(dtype).name}) ...", flush=True)
    t0 = time.time()
    writers, counts, doc_counts = {}, collections.Counter(), collections.Counter()
    doc_tokens = {}
    lengths = collections.defaultdict(list)
    chars_by = collections.Counter()

    def writer(split, source):
        key = (split, source)
        if key not in writers:
            writers[key] = open(os.path.join(out, f"{split}.{source}.bin"), "wb")
        return writers[key]

    def flush(batch):
        texts = [part for _, parts in batch for part in parts]
        encoded = tok.encode_batch(texts, threads=args.threads)
        pos = 0
        for i, parts in batch:
            split, source = split_of[i], docs[i][0]
            ids = []
            for j in range(len(parts)):
                ids.extend(encoded[pos + j])
            pos += len(parts)
            if tok.bos_id >= 0:
                ids.insert(0, tok.bos_id)
            if tok.eos_id >= 0:
                ids.append(tok.eos_id)
            elif tok.encode_mode == "plain":
                ids.extend(tok.encode("\n\n"))  # v3 convention: documents joined by blank lines
            arr = np.asarray(ids, dtype=dtype)
            writer(split, source).write(arr.tobytes())
            counts[(split, source)] += len(arr)
            doc_counts[(split, source)] += 1
            doc_tokens[i] = len(arr)
            lengths[source].append(len(arr))
            chars_by[(split, source)] += docs[i][3]

    try:
        batch, batch_chars = [], 0
        with open(spool2, encoding="utf-8") as src:
            for i, line in enumerate(src):
                if i in dropped:
                    continue
                rec = json.loads(line)
                parts = split_long_text(rec["x"], args.max_doc_chars)
                if len(parts) > 1:
                    sources[rec["s"]]["stats"]["cleaning"]["oversized_docs_split"] += 1
                    sources[rec["s"]]["stats"]["cleaning"]["oversized_parts"] += len(parts)
                batch.append((i, parts))
                batch_chars += len(rec["x"])
                if batch_chars > 8_000_000:
                    flush(batch)
                    batch, batch_chars = [], 0
            if batch:
                flush(batch)
    finally:
        for handle in writers.values():
            handle.close()
    print(f"Pass 3 done ({time.time() - t0:,.0f} s)", flush=True)

    # ------------------------------------------------------------------
    # Leakage audit (sampled word 13-grams, train vs val)
    # ------------------------------------------------------------------
    leakage = []
    if args.leakage_check == "on":
        try:  # an informational check: its failure must not throw away a finished dataset
            leakage = _leakage_audit(spool2, docs, split_of, dropped, workers=args.workers,
                                     warmup=args.worker_warmup)
        except (MemoryError, OSError, RuntimeError) as exc:
            print(f"Leakage audit failed ({type(exc).__name__}: {exc}); the dataset is written without it.",
                  flush=True)
            notes.append(f"The leakage audit failed ({type(exc).__name__}) and was skipped.")

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------
    manifest_sources = {}
    for name, info in sources.items():
        entry = {"kind": info["kind"], "paths": info.get("paths"), "select": info.get("select"),
                 "scanned": info["stats"]["scanned"], "kept_docs": info["stats"]["accepted"],
                 "skipped": dict(info["stats"]["skipped"]), "cleaning": dict(info["stats"]["cleaning"]),
                 "stopped_early": info["stats"]["stopped_early"]}
        for split in ("train", "val"):
            if counts[(split, name)]:
                path = os.path.join(out, f"{split}.{name}.bin")
                entry[split] = {"file": f"{split}.{name}.bin", "tokens": int(counts[(split, name)]),
                                "docs": int(doc_counts[(split, name)]), "chars": int(chars_by[(split, name)]),
                                "sha256": sha256_file(path)}
        lens = np.asarray(lengths.get(name, [0]))
        entry["doc_tokens_percentiles"] = {str(q): int(np.percentile(lens, q)) for q in (10, 50, 90, 99)}
        manifest_sources[name] = entry
    tok_meta = tok.meta()
    tok_meta["file"] = "tokenizer.model"
    tok_meta.update(tokenizer_info)
    tok_meta["composition"] = dict(collections.Counter(token_category(tok, i) for i in range(tok.vocab_size)))
    fingerprint = hashlib.sha256()
    for name in sorted(manifest_sources):
        for split in ("train", "val"):
            if split in manifest_sources[name]:
                fingerprint.update(f"{name}/{split}/{manifest_sources[name][split]['sha256']}".encode())
    fingerprint.update(tok.sha256.encode())
    manifest = {
        "format": FORMAT, "format_version": FORMAT_VERSION,
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "dtype": np.dtype(dtype).name,
        "tokenizer": tok_meta,
        "sources": manifest_sources,
        "totals": {
            "train_tokens": int(sum(v for (s, _), v in counts.items() if s == "train")),
            "val_tokens": int(sum(v for (s, _), v in counts.items() if s == "val")),
            "train_docs": int(sum(v for (s, _), v in doc_counts.items() if s == "train")),
            "val_docs": int(sum(v for (s, _), v in doc_counts.items() if s == "val")),
        },
        "near_duplicate_clusters": len(clusters),
        "fingerprint": fingerprint.hexdigest(),
        "prep_args": {k: v for k, v in vars(args).items()},
        "command": " ".join(sys.argv),
    }
    tmp = os.path.join(out, "manifest.json.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    os.replace(tmp, os.path.join(out, "manifest.json"))

    tok_stats = _tokenizer_stats(tok, docs, split_of, doc_tokens)
    _write_report(out, args, sources, ledger, duplicates, removed_lines, line_examples, keyword_totals,
                  manifest, tok_stats, forced_train, leakage, tok, t_start, notes=notes)
    _write_ledger(out, ledger, docs, split_of, group_of, dropped, doc_tokens)
    _cleanup(work, args.keep_work)
    t = manifest["totals"]
    print(f"\nDataset ready: {out}\n  train {t['train_tokens']:,} tokens in {t['train_docs']:,} docs | "
          f"val {t['val_tokens']:,} tokens in {t['val_docs']:,} docs | fingerprint {manifest['fingerprint'][:12]}\n"
          f"  report: {os.path.join(out, 'report.md')} ({time.time() - t_start:,.0f} s total)")
    return out


_OUTPUT_NAMES = {"_work", "tokenizer.model", "report.md", "docs.tsv", "duplicates.tsv", "boilerplate.tsv",
                 "keyword_hits.tsv", "leakage.tsv", "manifest.json.tmp"}


def _previous_split(dataset_dir):
    """{(source, doc_id): 'train' | 'val'} from an earlier dataset's docs.tsv."""
    path = os.path.join(dataset_dir, "docs.tsv")
    if not os.path.isfile(path):
        raise SystemExit(f"--keep-split-from: {path} not found (the earlier dataset folder is needed).")
    out = {}
    with open(path, encoding="utf-8") as handle:
        head = handle.readline().rstrip("\n").split("\t")
        for line in handle:
            row = dict(zip(head, line.rstrip("\n").split("\t")))
            if row.get("status") in ("train", "val"):
                out[(row["source"], row["doc_id"])] = row["status"]
    return out


def _is_incomplete_build(folder):
    """True if a folder holds only files this script writes but no manifest and
    no report (an interrupted run); anything else, including a scan-only report,
    is never deleted."""
    entries = os.listdir(folder)
    return "manifest.json" not in entries and "report.md" not in entries and all(
        e in _OUTPUT_NAMES or re.fullmatch(r"(train|val)\.[A-Za-z][A-Za-z0-9_-]*\.bin", e) for e in entries)


def _train_tokenizer(spool2, docs, split_of, sources, vocab, args, work):
    import sentencepiece as spm
    # Default: every training document has the same inclusion probability, so
    # each source contributes in proportion to its size. --tokenizer-weights
    # multiplies a source's share (total sample size stays near the budget).
    weights = {n: 1.0 for n in sources}
    for item in (args.tokenizer_weights or "").split(","):
        if not item.strip():
            continue
        name, _, value = item.partition("=")
        if name.strip() not in sources:
            raise SystemExit(f"--tokenizer-weights: unknown source {name.strip()!r}; sources are {sorted(sources)}")
        try:
            weights[name.strip()] = float(value)
        except ValueError:
            raise SystemExit(f"--tokenizer-weights: {item!r} is not name=number")
    mass = max(sum(weights[n] * s.get("train_chars", 0) for n, s in sources.items()), 1)
    keep_prob = {n: min(1.0, args.tokenizer_sample_chars * weights[n] / mass) for n in sources}
    corpus = os.path.join(work, "tokenizer_corpus.txt")
    written = collections.Counter()
    with open(spool2, encoding="utf-8") as src, open(corpus, "w", encoding="utf-8", newline="\n") as dst:
        for i, line in enumerate(src):
            if split_of.get(i) != "train":
                continue
            rec = json.loads(line)
            if stable_unit(args.seed, "tok", rec["s"], rec["id"]) >= keep_prob[rec["s"]]:
                continue
            for ln in rec["x"].split("\n"):
                if not ln.strip():
                    continue
                for start in range(0, len(ln), 4000):
                    dst.write(ln[start:start + 4000] + "\n")
            written[rec["s"]] += len(rec["x"])
    extra = max(1, round(vocab * args.superbpe_fraction)) if getattr(args, "superbpe", False) else 0
    plan = f"{vocab - extra:,} + {extra:,} SuperBPE" if extra else f"{vocab:,}"
    print(f"\nTraining SentencePiece BPE (vocab {plan}) on {sum(written.values()) / 1e6:,.1f}M characters "
          f"from training documents {dict(written)} ...", flush=True)
    t0 = time.time()
    prefix = os.path.join(work, "spm")
    spm.SentencePieceTrainer.train(
        input=corpus, model_prefix=prefix, model_type="bpe", vocab_size=vocab - extra,
        character_coverage=0.9995, byte_fallback=True, split_digits=True,
        unk_id=0, bos_id=1, eos_id=2, pad_id=-1, user_defined_symbols=["\n", "\t"],
        remove_extra_whitespaces=False, allow_whitespace_only_pieces=True,
        normalization_rule_name="identity", max_sentence_length=16384, hard_vocab_limit=False,
        input_sentence_size=6_000_000, shuffle_input_sentence=True,
        num_threads=args.threads, minloglevel=2,
    )
    try:  # the model records where and how it was trained (folder, threads): clear that, so identical data
        # gives an identical tokenizer file and dataset fingerprint wherever and however it is built
        from sentencepiece import sentencepiece_model_pb2 as spm_pb
        proto = spm_pb.ModelProto()
        with open(prefix + ".model", "rb") as handle:
            proto.ParseFromString(handle.read())
        del proto.trainer_spec.input[:]
        proto.trainer_spec.model_prefix = ""
        proto.trainer_spec.num_threads = 1
        with open(prefix + ".model", "wb") as handle:
            handle.write(proto.SerializeToString())
    except Exception as exc:  # noqa: BLE001 - the file is still correct, only less reproducible
        print(f"Note: the tokenizer file keeps its build details ({type(exc).__name__}: {exc}), so the dataset "
              "fingerprint depends on the folder it was built in. Installing protobuf fixes this "
              "(python -m pip install protobuf).")
    tok = Tokenizer.from_file(prefix + ".model", "lines")
    info = {"trained": True, "sample_chars": dict(written), "sample_weights": weights, "character_coverage": 0.9995,
            "split_digits": True, "requested_vocab": vocab}
    if extra:
        print(f"SuperBPE stage 2: learning up to {extra:,} tokens that may span words (at most "
              f"{args.superbpe_max_words}) on top of {tok.vocab_size:,} ordinary pieces ...", flush=True)
        t1 = time.time()
        proto, info["superbpe"] = superbpe_extend(tok.proto, corpus, extra, args.superbpe_max_words,
                                                  seed=args.seed, threads=args.threads)
        with open(prefix + ".model", "wb") as handle:
            handle.write(proto)
        tok = Tokenizer(proto, "lines")
        sb = info["superbpe"]
        print(f"  {sb['added_pieces']:,} added ({sb['multiword_pieces']:,} span several words) in "
              f"{time.time() - t1:,.0f} s; {sb['token_saving']:.1%} fewer tokens on "
              f"{'held-out ' if sb['held_out'] else ''}lines of the tokenizer text; e.g. "
              + ", ".join(repr(t) for t in sb["examples"][:8]))
    print(f"Tokenizer trained in {time.time() - t0:,.0f} s ({tok.vocab_size:,} pieces)")
    return tok, info


def _piece_words(piece):
    """Number of words in a piece: '▁of▁the' -> 2, 'ing▁the' -> 2, '▁▁' -> 0."""
    return sum(1 for part in piece.split("\u2581") if part)


def superbpe_extend(proto, corpus, extra, max_words=4, sample_tokens=20_000_000, seed=1234, threads=1,
                    max_chars=64):
    """Stage 2 of SuperBPE (Liu et al. 2025, "SuperBPE: Space Travel for
    Language Models"): continue BPE on the stage-1 tokens of the tokenizer
    corpus with word boundaries no longer blocking merges, so frequent phrases
    ("▁of▁the", "▁linkage▁disequilibrium") become single tokens.

    Up to ``extra`` merges are learnt, each of at most ``max_words`` words.
    Digits, newlines, tabs, byte-fallback and control tokens are never merged
    (numbers stay one digit per token). The new tokens are appended to the
    SentencePiece model in the order learnt, scored below every existing piece,
    so SentencePiece's own encoder applies them after all stage-1 merges:
    encoding speed, decoding, exact byte counts and checkpoints are unchanged.
    Deterministic. Returns (model bytes, info)."""
    import heapq
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2 as spm_pb

    model = spm_pb.ModelProto()
    model.ParseFromString(proto)
    sp = spm.SentencePieceProcessor()
    if not sp.LoadFromSerializedProto(proto):
        raise ValueError("could not parse the SentencePiece model")
    base = sp.get_piece_size()
    strs = [sp.id_to_piece(i) for i in range(base)]
    normal = spm_pb.ModelProto.SentencePiece.NORMAL
    size = base + extra + 1  # the last id stands for "end of line": no pair crosses it
    sep = size - 1
    allowed = np.zeros(size, dtype=bool)
    for i, item in enumerate(model.pieces):
        allowed[i] = (item.type == normal and strs[i] not in ("\n", "\t")
                      and not any(ch.isdigit() for ch in strs[i]))
    allowed[base:sep] = True

    # a deterministic sample of corpus lines, spread over the whole file (~4 bytes per token)
    keep = min(1.0, sample_tokens * 4.0 / max(os.path.getsize(corpus), 1))
    chunks, batch, checks = [], [], []

    def flush():
        flat = []
        for ids in sp.encode(batch, out_type=int, num_threads=threads):
            flat.extend(ids)
            flat.append(sep)
        chunks.append(np.asarray(flat, dtype=np.int32))
        batch.clear()

    with open(corpus, encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            line = line.rstrip("\n")
            if not line:
                continue
            if stable_unit(seed, "superbpe-check", i) < 0.01:  # held out: measures the saving
                if len(checks) < 20_000:
                    checks.append(line)
                continue
            if keep >= 1.0 or stable_unit(seed, "superbpe", i) < keep:
                batch.append(line)
                if len(batch) >= 20_000:
                    flush()
    if batch:
        flush()
    seq = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int32)
    del chunks
    stage1_tokens = int(np.count_nonzero(seq != sep))
    held_out = bool(checks)
    if not checks:  # a tiny corpus: measure on (part of) the learning text instead
        with open(corpus, encoding="utf-8") as handle:
            checks = [ln.rstrip("\n") for ln in handle if ln.strip()][:20_000]

    def pair_codes(seq, starts):
        x, y = seq[starts], seq[starts + 1]
        ok = allowed[x] & allowed[y]
        return x[ok].astype(np.int64) * size + y[ok]

    codes, freq = np.unique(pair_codes(seq, np.arange(max(len(seq) - 1, 0))), return_counts=True)
    counts = dict(zip(codes.tolist(), freq.tolist()))
    del codes, freq
    heap = [(-c, k) for k, c in counts.items() if c >= 2]
    heapq.heapify(heap)
    existing, banned, added = set(strs), set(), []
    while len(added) < extra and heap:
        neg, code = heapq.heappop(heap)
        current = counts.get(code, 0)
        if current != -neg:  # stale entry: counts only fall after a pair's tokens exist
            if current >= 2:
                heapq.heappush(heap, (-current, code))
            continue
        a, b = divmod(code, size)
        text = strs[a] + strs[b]
        if text in existing or len(text) > max_chars or _piece_words(text) > max_words:
            banned.add(code)
            counts.pop(code, None)
            continue
        new = base + len(added)
        cand = np.flatnonzero(seq[:-1] == a)
        pos = cand[seq[cand + 1] == b]
        if not len(pos):  # cannot happen with exact counts; never add a token that does not occur
            counts.pop(code, None)
            continue
        if a == b and len(pos) > 1:  # runs like x x x: merge left to right without overlap
            chosen, last = [], -2
            for q in pos.tolist():
                if q > last + 1:
                    chosen.append(q)
                    last = q
            pos = np.asarray(chosen, dtype=np.int64)
        old = np.unique(np.concatenate([pos - 1, pos, pos + 1]))
        removed = pair_codes(seq, old[(old >= 0) & (old < len(seq) - 1)])
        seq[pos] = new
        seq = np.delete(seq, pos + 1)
        at = pos - np.arange(len(pos))
        fresh = np.unique(np.concatenate([at - 1, at]))
        created = pair_codes(seq, fresh[(fresh >= 0) & (fresh < len(seq) - 1)])
        both = np.concatenate([removed, created])
        keys, inverse = np.unique(both, return_inverse=True)
        sign = np.concatenate([-np.ones(len(removed)), np.ones(len(created))])
        delta = np.rint(np.bincount(inverse, weights=sign, minlength=len(keys))).astype(np.int64)
        for k, d in zip(keys.tolist(), delta.tolist()):
            if d == 0 or k in banned:
                continue
            c = counts.get(k, 0) + d
            if c > 0:
                counts[k] = c
            else:
                counts.pop(k, None)
            if d > 0 and c >= 2:
                heapq.heappush(heap, (-c, k))
        added.append(text)
        strs.append(text)
        existing.add(text)

    low = min(item.score for item in model.pieces)
    for k, text in enumerate(added):
        item = model.pieces.add()
        item.piece, item.score, item.type = text, low - 1.0 - k, normal
    out = model.SerializeToString()
    before, after = Tokenizer(proto, "lines"), Tokenizer(out, "lines")
    for line in checks[:5000]:
        if after.decode(after.encode(line)) != line:
            raise RuntimeError(f"SuperBPE tokenizer does not round-trip: {line[:80]!r}")
    n_before = sum(len(x) for x in before.encode_batch(checks, threads))
    n_after = sum(len(x) for x in after.encode_batch(checks, threads))
    info = {"stage1_pieces": base, "added_pieces": len(added),
            "multiword_pieces": sum(1 for t in added if _piece_words(t) > 1), "max_words": max_words,
            "learnt_on_tokens": stage1_tokens, "checked_lines": len(checks), "held_out": held_out,
            "token_saving": round(1 - n_after / max(n_before, 1), 4), "examples": added[:40]}
    return out, info


def _tokenizer_stats(tok, docs, split_of, doc_tokens):
    chars, tokens = collections.Counter(), collections.Counter()
    for i, n in doc_tokens.items():
        key = (split_of[i], docs[i][0])
        chars[key] += docs[i][3]
        tokens[key] += n
    per = {f"{s}/{src}": round(chars[(s, src)] / max(tokens[(s, src)], 1), 3) for (s, src) in tokens}
    examples = {}
    for term in ("genetic drift", "heritability", "Wright–Fisher", "GWAS summary statistics", "UK Biobank",
                 "pharmacokinetics", "eigenvalue", "haematoxylin and eosin", "Ne = 1000, p = 0.05",
                 "def f(x):\n    return x ** 2"):
        ids = tok.encode(term)
        examples[term] = [tok.piece(i) for i in ids]
    return {"chars_per_token": per, "examples": examples}


_AUDIT = {}


def _init_audit(n, keep_mod):
    global _AUDIT
    _AUDIT = {"hasher": ShingleHasher(), "n": n, "mod": np.uint64(keep_mod)}
    try:
        _lower_priority()
    except Exception:  # noqa: BLE001
        pass


def _audit_chunk(texts):
    """Sampled word n-gram hashes of each text (1 in keep_mod kept, by hash value)."""
    a = _AUDIT
    out = []
    for text in texts:
        sh = a["hasher"].shingles(text, a["n"])
        out.append(sh[(sh % a["mod"]) == 0])
    return out


def _init_audit_match(n, keep_mod, val_set):
    _init_audit(n, keep_mod)
    _AUDIT["val"] = val_set


def _audit_match_chunk(texts):
    """The sampled n-gram hashes of each training text that also occur in the
    validation set (usually none), so only a few numbers travel back."""
    a, val = _AUDIT, _AUDIT["val"]
    out = []
    for text in texts:
        sh = a["hasher"].shingles(text, a["n"])
        sh = sh[(sh % a["mod"]) == 0]
        if len(sh) and len(val):
            pos = np.minimum(np.searchsorted(val, sh), len(val) - 1)
            sh = sh[val[pos] == sh]
        out.append(np.unique(sh) if len(sh) else sh)
    return out


def _leakage_audit(spool2, docs, split_of, dropped, n=13, keep_mod=16, workers=1, warmup=2_000):
    """Per validation document: the share of its sampled word 13-grams that also
    occur in training documents. The validation side (about 5% of the text) is
    hashed first; the training text is then checked against it in the workers,
    so memory stays at a few hundred MB for any corpus size (sorting every
    training hash needed several GB for 4 billion tokens)."""
    print("\nLeakage audit: sampled word 13-gram overlap of validation documents with training ...", flush=True)

    def texts(split):
        with open(spool2, encoding="utf-8") as src:
            for i, line in enumerate(src):
                if i not in dropped and split_of.get(i) == split:
                    yield i, json.loads(line)["x"]

    pool = _Pool(workers, _init_audit, (n, keep_mod), warmup)
    val = [(i, sh) for (i, _), sh in _ordered(texts("val"), lambda it: it[1], _audit_chunk, pool, chunk=32,
                                              size_of=lambda it: len(it[1]))]
    pool.close()
    val_set = np.unique(np.concatenate([sh for _, sh in val])) if val else np.zeros(0, dtype=np.uint64)
    found = []
    if len(val_set):
        pool = _Pool(workers, _init_audit_match, (n, keep_mod, val_set), warmup)
        found = [sh for _, sh in _ordered(texts("train"), lambda it: it[1], _audit_match_chunk, pool, chunk=32,
                                          size_of=lambda it: len(it[1])) if len(sh)]
        pool.close()
    in_train = np.unique(np.concatenate(found)) if found else np.zeros(0, dtype=np.uint64)
    rows = []
    for i, sh in val:
        if len(sh) < 5:
            continue
        if len(in_train):
            pos = np.minimum(np.searchsorted(in_train, sh), len(in_train) - 1)
            overlap = float((in_train[pos] == sh).mean())  # binary search in the sorted matches
        else:
            overlap = 0.0
        rows.append((overlap, docs[i][0], docs[i][1], docs[i][2], len(sh)))
    rows.sort(reverse=True)
    return rows


def _cleanup(work, keep):
    if keep:
        print(f"Work files kept in {work}")
        return
    shutil.rmtree(work, ignore_errors=True)


def _write_ledger(out, ledger, docs, split_of, group_of, dropped, doc_tokens):
    path = os.path.join(out, "docs.tsv")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("source\tdoc_id\ttitle\tstatus\treason\tchars\ttokens\tkeyword_hits\tgroup\n")
        # sorted: readers add some rows ahead of the checked documents, by an amount that depends on --workers
        rows = sorted(ledger, key=lambda r: (r["source"], str(r["doc_id"]), r["reason"]))
        for i, d in enumerate(docs):
            if i in dropped:
                continue
            rows.append(_ledger_row(d[0], d[1], d[2], split_of.get(i, "?"), "", d[3], doc_tokens.get(i, 0),
                                    d[4], group_of.get(i, "")))
        for r in rows:
            handle.write("\t".join(str(r[k]).replace("\t", " ").replace("\n", " ") for k in
                                   ("source", "doc_id", "title", "status", "reason", "chars", "tokens",
                                    "keyword_hits", "group")) + "\n")


def _write_report(out, args, sources, ledger, duplicates, removed_lines, line_examples, keyword_totals,
                  manifest, tok_stats, forced_train, leakage, tok, t_start, scan_only=False, notes=()):
    L = []
    L.append(f"# Dataset report: {os.path.basename(out)}\n")
    L.append(f"Created {datetime.datetime.now().astimezone().isoformat(timespec='seconds')} in "
             f"{time.time() - t_start:,.0f} s. Command: `{' '.join(sys.argv)}`\n")
    if scan_only:
        L.append("**Scan-only run:** filtering, cleaning and deduplication were measured; nothing was tokenized.\n")
    L.append("## Sources\n")
    L.append("| source | kind | scanned | kept docs | train docs | val docs | train tokens | val tokens | skipped |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for name, info in sources.items():
        st = info["stats"]
        m = (manifest or {}).get("sources", {}).get(name, {})
        tr, va = m.get("train", {}), m.get("val", {})
        skipped = ", ".join(f"{k} {v:,}" for k, v in st["skipped"].most_common()) or "none"
        L.append(f"| {name} | {info['kind']} | {st['scanned']:,} | {st['accepted']:,} | {tr.get('docs', 0):,} | "
                 f"{va.get('docs', 0):,} | {tr.get('tokens', 0):,} | {va.get('tokens', 0):,} | {skipped} |")
    if manifest:
        t = manifest["totals"]
        L.append(f"\n**Totals:** train {t['train_tokens']:,} tokens / {t['train_docs']:,} docs; "
                 f"validation {t['val_tokens']:,} tokens / {t['val_docs']:,} docs. "
                 f"Fingerprint `{manifest['fingerprint'][:16]}`.\n")
    for name, info in sources.items():
        if info["stats"]["stopped_early"]:
            L.append(f"- `{name}`: scanning stopped early once enough documents were accepted "
                     "(later records were not examined).")
        if info.get("select"):
            L.append(f"- `{name}`: selection mode `{info['select']}`.")
    for note in notes:
        L.append(f"- **Note:** {note}")
    L.append("\n## Cleaning (documents affected; nothing here removes a document)\n")
    for name, info in sources.items():
        c = info["stats"]["cleaning"]
        L.append(f"- **{name}:** " + (", ".join(f"{k} {v:,}" for k, v in sorted(c.items())) or "no changes"))
    L.append("\n## Validation split\n")
    L.append(f"Split by document group per source (seed {args.seed}); target {args.val_fraction:.1%} of each "
             f"source's characters, capped at {args.val_max_chars:,} characters. Parts of a split document and "
             "members of a near-duplicate cluster always share a split.\n")
    for name, info in sources.items():
        L.append(f"- {name}: validation {info.get('val_chars', 0):,} chars "
                 f"(target {info.get('val_chars_target', 0):,}); train {info.get('train_chars', 0):,} chars")
    if forced_train:
        L.append("- kept in train: " + ", ".join(f"{k} {v:,}" for k, v in forced_train.items()))
    if any(keyword_totals.values()):
        L.append("\n## Keyword filter\n")
        L.append(f"Include terms {len(read_terms(args.include_keywords))}, exclude terms "
                 f"{len(read_terms(args.exclude_keywords))}; min weighted hits {args.keyword_min_hits}, min distinct "
                 f"{args.keyword_min_distinct}, title weight {args.title_weight}. Full counts in keyword_hits.tsv.\n")
        for name, counter in keyword_totals.items():
            if counter:
                L.append(f"- {name} top terms (documents kept): " +
                         ", ".join(f"{t} {c:,}" for t, c in counter.most_common(25)))
        with open(os.path.join(out, "keyword_hits.tsv"), "w", encoding="utf-8", newline="\n") as handle:
            handle.write("source\tterm\thits_in_kept_docs\n")
            for name, counter in keyword_totals.items():
                for term, count in counter.most_common():
                    handle.write(f"{name}\t{term}\t{count}\n")
    L.append("\n## Duplicates\n")
    exact = [d for d in duplicates if d[0] == "exact"]
    near = [d for d in duplicates if d[0] == "near"]
    L.append(f"Exact duplicates ({args.exact_dedup}): {len(exact):,}. Near duplicates (MinHash word 5-grams, "
             f"Jaccard >= {args.near_dup_threshold}, action {args.near_dup}): {len(near):,}. "
             "All pairs in duplicates.tsv.\n")
    for d in (exact + near)[:25]:
        L.append(f"- {d[0]}: `{d[1]}:{d[3] or d[2]}` ~ `{d[4]}:{d[6] or d[5]}` (est. Jaccard {d[7]:.2f})")
    with open(os.path.join(out, "duplicates.tsv"), "w", encoding="utf-8", newline="\n") as handle:
        handle.write("kind\tsource\tdoc_id\ttitle\tduplicate_of_source\tduplicate_of_id\tduplicate_of_title\tjaccard\n")
        for d in duplicates:
            handle.write("\t".join(str(x).replace("\t", " ") for x in d[:7]) + f"\t{d[7]:.3f}\n")
    L.append("\n## Boilerplate lines\n")
    total_removed = sum(removed_lines.values())
    L.append("Headings (Markdown # lines and short ALL-CAPS lines) are never counted as boilerplate. ")
    L.append(f"Mode `{args.boilerplate}`: lines of >= {args.boilerplate_min_len} characters that occur in >= "
             f"{args.boilerplate_min_docs} documents of the same source. {len(removed_lines):,} distinct lines, "
             f"{total_removed:,} occurrences {'removed' if args.boilerplate == 'strip' else 'flagged'}. "
             "Full list in boilerplate.tsv.\n")
    with open(os.path.join(out, "boilerplate.tsv"), "w", encoding="utf-8", newline="\n") as handle:
        handle.write("source\toccurrences\tline\n")
        for (name, k), c in removed_lines.most_common():
            handle.write(f"{name}\t{c}\t{line_examples.get(k, '?')}\n")
    for (name, k), c in removed_lines.most_common(15):
        L.append(f"- {name} x{c}: `{line_examples.get(k, '?')[:110]}`")
    if manifest and tok is not None:
        L.append("\n## Tokenizer\n")
        meta = manifest["tokenizer"]
        L.append(f"SentencePiece BPE, {meta['vocab_size']:,} pieces, encode mode `{meta['encode_mode']}`, "
                 f"BOS {meta['bos_id']}, EOS {meta['eos_id']}, newline {meta['newline_id']}. "
                 f"{'Trained on training documents only.' if meta.get('trained') else 'Reused: ' + str(meta.get('from'))}\n")
        sb = meta.get("superbpe")
        if sb:
            L.append(f"SuperBPE: {sb['stage1_pieces']:,} ordinary BPE pieces, then {sb['added_pieces']:,} learnt with "
                     f"word boundaries open ({sb['multiword_pieces']:,} of them span 2 to {sb['max_words']} words). "
                     f"{'Held-out lines' if sb.get('held_out', True) else 'Lines'} of the tokenizer text need "
                     f"{sb['token_saving']:.1%} fewer tokens than with the "
                     "ordinary pieces alone. First learnt: "
                     + ", ".join(f"`{t}`" for t in sb["examples"][:20]) + "\n")
        L.append("Composition: " + ", ".join(f"{k} {v:,}" for k, v in sorted(meta["composition"].items(),
                                                                                key=lambda x: -x[1])))
        L.append("\nCharacters per token: " + ", ".join(f"{k} {v}" for k, v in tok_stats["chars_per_token"].items()))
        L.append("\nExamples:\n")
        for term, pieces in tok_stats["examples"].items():
            L.append(f"- `{term!r}` -> {len(pieces)} tokens: `{' '.join(repr(p)[1:-1] for p in pieces)}`")
        L.append("\n## Document lengths (tokens)\n")
        for name, entry in manifest["sources"].items():
            L.append(f"- {name}: " + ", ".join(f"p{q} {v:,}" for q, v in entry["doc_tokens_percentiles"].items()))
    if leakage:
        flagged = [r for r in leakage if r[0] >= 0.5]
        L.append("\n## Train/validation leakage audit\n")
        L.append(f"Share of each validation document's sampled word 13-grams (1 in 16) that also occur in "
                 f"training documents. {len(leakage):,} validation documents checked; {len(flagged):,} have "
                 f">= 50% overlap (listed in leakage.tsv). Median overlap "
                 f"{np.median([r[0] for r in leakage]):.1%}.\n")
        for r in leakage[:10]:
            L.append(f"- {r[0]:.0%} `{r[1]}:{r[3] or r[2]}` ({r[4]} sampled 13-grams)")
        with open(os.path.join(out, "leakage.tsv"), "w", encoding="utf-8", newline="\n") as handle:
            handle.write("overlap\tsource\tdoc_id\ttitle\tsampled_13grams\n")
            for r in leakage:
                handle.write(f"{r[0]:.4f}\t{r[1]}\t{r[2]}\t{str(r[3]).replace(chr(9), ' ')}\t{r[4]}\n")
    L.append("\n## Files\n")
    L.append("- `docs.tsv`: every kept document (split, tokens, group), every skipped Markdown file and "
             "up to 2,000 web records per source that were skipped after full-text inspection, with the "
             "reason. Web records rejected by the vectorised keyword prefilter are counted in the table above.")
    L.append("- `duplicates.tsv`, `boilerplate.tsv`, `keyword_hits.tsv`, `leakage.tsv`: audit tables.")
    L.append("- `train.<source>.bin` / `val.<source>.bin`: token IDs (see manifest.json for dtype and SHA-256).")
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(L) + "\n")


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    prepare(argv)


if __name__ == "__main__":
    main()
