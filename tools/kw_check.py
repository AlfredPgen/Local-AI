"""Mechanical checks for a keyword batch file (the keywords.txt format: '# category'
lines followed by one term per line). Prints a JSON report and exits 0.

    python tools/kw_check.py BATCH.txt [N_FINEWEB_DOCS]

Flags, per term (a flag means "a person should look", not "wrong"):
  parse_error      data_prep cannot read the term
  duplicate        already in keywords.txt (same words, same case rule)
  short_token      a word of 1-2 characters (3 for a single-word term) that is not case-sensitive (=)
  short_prefix     a prefix wildcard whose stem has fewer than 4 letters
  common_in_web    matches more than 1% of a FineWeb-Edu sample (likely an everyday word or phrase)
  in_batch_twice   the same term twice in this batch
"new_categories" lists the batch's '# category' names that keywords.txt does not have yet.
"""
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import data_prep as dp  # noqa: E402


def read_batch(path):
    out, cat = [], None
    for n, line in enumerate(open(path, encoding="utf-8").read().splitlines(), 1):
        s = line.strip()
        if s.startswith("# "):
            cat = s[2:].strip()
        elif s and not s.startswith("#"):
            out.append((n, cat, s))
    return out


def norm(t):
    return t if t.startswith("=") else t.lower()


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # JSON may hold non-ASCII names, also when redirected to a file
    path = sys.argv[1]
    n_docs = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    rows = read_batch(path)
    existing = {norm(t) for t in dp.read_terms(os.path.join(HERE, "keywords.txt"))}
    flags = collections.defaultdict(list)
    seen, good = set(), []
    for n, cat, t in rows:
        key = norm(t)
        if key in seen:
            flags[t].append("in_batch_twice")
        seen.add(key)
        try:
            words, shape, cs, lead, trail = dp._term_words(t)
        except Exception as exc:  # noqa: BLE001
            flags[t].append(f"parse_error: {type(exc).__name__}: {exc}")
            continue
        if not words:
            flags[t].append("parse_error: no words")
            continue
        if key in existing:
            flags[t].append("duplicate")
        if not cs:
            limit = 3 if len(words) == 1 else 2
            if any(len(w) <= limit for w in words):
                flags[t].append("short_token")
        if t.endswith("*") and len(t.rstrip("*").lstrip("=").strip()) < 4:
            flags[t].append("short_prefix")
        good.append(t)
    # how often each term matches ordinary educational web text
    common = {}
    files = sorted(glob.glob(os.path.join(os.path.expanduser("~"),
                                          "ai_training_data/web/fineweb-edu/sample/10BT/*.parquet")))
    if files and good and n_docs > 0:
        import pyarrow.parquet as pq
        batch = next(pq.ParquetFile(files[0]).iter_batches(batch_size=n_docs, columns=["text"]))
        texts = batch.column(0).to_pylist()
        kf = dp.KeywordFilter(good, [], 1, 1)
        df = collections.Counter()
        for text in texts:
            for term in kf.evaluate("", text)[3]:
                df[term] += 1
        for term, c in df.items():
            share = c / len(texts)
            if share > 0.01:
                flags[term].append(f"common_in_web: {share:.1%} of {len(texts):,} FineWeb-Edu documents")
                common[term] = share
    known = {ln[2:].strip() for ln in open(os.path.join(HERE, "keywords.txt"), encoding="utf-8")
             if ln.startswith("# ")}
    cats = collections.Counter(c for _, c, _ in rows)
    report = {"batch": path, "terms": len(rows), "flagged": len(flags),
              "flags": {t: f for t, f in sorted(flags.items())},
              "categories": dict(cats), "new_categories": sorted(c for c in cats if c not in known)}
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
