"""Stack Overflow and other Stack Exchange sites for pre-training, from Hugging Face's
Markdown dump (HuggingFaceTB/stackexchange_2025_md: one Parquet row per question, posts to
2024-12-31, CC BY-SA 2.5/3.0/4.0 by post; usernames are already anonymised).

    python tools/download_stackexchange.py                       # Stack Overflow + science sites
    python tools/download_stackexchange.py --sites stats.stackexchange.com --min-score 1

The raw dump is ~127 GB (Stack Overflow ~90 GB in 177 files of ~510 MB), mostly comments,
metadata and low-scored posts, so it is never kept: each file is downloaded to --scratch,
filtered into <out>/<site>/<file>.jsonl.gz, and deleted before the next one. A finished
file is skipped on a rerun (resume after a stop by running the same command).

Kept per question: open (not closed), score >= the site's minimum, at least one answer with
score >= --min-answer-score or accepted; the text is the question (title, body, tags) and
its best --top-answers such answers, highest score first, in the dump's Markdown. JSONL
fields: id, title, text, site, score, tags, license (data_prep.py --jsonl reads id, title, text).
"""
import argparse
import gzip
import json
import os
import re
import sys
import time

REPO = "HuggingFaceTB/stackexchange_2025_md"
DEFAULT_SITES = ("stackoverflow.com,bioinformatics.stackexchange.com,stats.stackexchange.com,"
                 "biology.stackexchange.com,datascience.stackexchange.com")
ANSWER_HEAD = re.compile(r"^### Answer by .*?\. Score: (-?\d+)( \| Accepted answer)?\s*$", re.M)
ASKED = re.compile(r"^\(Asked by: [^)]*\)\s*$", re.M)
COUNT = re.compile(r"^## \d+ answers?\s*$", re.M)


def build_text(thread, min_answer, top):
    """(text, answers kept) from a ThreadText: the question part, then the best answers. None when no answer
    qualifies or the thread does not have the expected layout."""
    heads = list(ANSWER_HEAD.finditer(thread))
    if not heads:
        return None, 0
    question = thread[:heads[0].start()]
    question = COUNT.sub("", ASKED.sub("", question)).rstrip().rstrip("-").rstrip()
    answers = []
    for i, h in enumerate(heads):
        body = thread[h.end():heads[i + 1].start() if i + 1 < len(heads) else len(thread)].strip()
        score, accepted = int(h.group(1)), bool(h.group(2))
        if body and (score >= min_answer or accepted):
            answers.append((score, accepted, body.rstrip("-").rstrip()))
    answers.sort(key=lambda a: -a[0])
    answers = answers[:top]
    if not answers:
        return None, 0
    parts = [question, "", "---", ""]
    for score, accepted, body in answers:
        parts += ["### Accepted answer" if accepted else "### Answer", "", body, ""]
    return "\n".join(parts).strip() + "\n", len(answers)


def filter_file(src, dst, site, min_score, min_answer, top):
    """Filters one dump file into dst (.jsonl.gz, written to .part and renamed when complete)."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(src)
    cols = ["Id", "Title", "Score", "Tags", "ClosedDate", "ContentLicense", "ThreadText"]
    kept = seen = chars = 0
    with gzip.open(dst + ".part", "wt", encoding="utf-8") as out:
        for g in range(pf.num_row_groups):
            table = pf.read_row_group(g, columns=cols)
            seen += table.num_rows
            # score, closed and empty threads are checked in Arrow (C++): only the questions that pass become
            # Python objects (at score >= 3 about one in seven on Stack Overflow)
            closed = pc.fill_null(pc.not_equal(pc.utf8_length(table["ClosedDate"]), 0), False)
            keep = pc.and_(pc.and_(pc.greater_equal(pc.fill_null(table["Score"], 0), min_score), pc.invert(closed)),
                           pc.is_valid(table["ThreadText"]))
            for r in table.filter(keep).to_pylist():
                text, n = build_text(r["ThreadText"], min_answer, top)
                if text is None:
                    continue
                tags = [t for t in re.split(r"[<>]", r["Tags"] or "") if t]
                out.write(json.dumps({"id": f"{site}/{r['Id']}", "title": r["Title"] or "", "text": text,
                                      "site": site, "score": r["Score"], "tags": tags,
                                      "license": r["ContentLicense"]}, ensure_ascii=False) + "\n")
                kept += 1
                chars += len(text)
    os.replace(dst + ".part", dst)
    return seen, kept, chars


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join(os.path.expanduser("~"), "ai_training_data", "stackexchange"))
    p.add_argument("--scratch", default=None, help="where each raw file waits while it is filtered "
                                                   "(default: <out>/_raw; deleted file by file)")
    p.add_argument("--sites", default=DEFAULT_SITES, help="comma-separated site folders of the dump")
    p.add_argument("--so-min-score", type=int, default=5, help="minimum question score on stackoverflow.com")
    p.add_argument("--min-score", type=int, default=1, help="minimum question score on the other sites")
    p.add_argument("--min-answer-score", type=int, default=1, help="answers kept: this score or accepted")
    p.add_argument("--top-answers", type=int, default=3)
    p.add_argument("--max-files", type=int, default=0, help="stop after this many files (a trial run)")
    args = p.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    scratch = args.scratch or os.path.join(args.out, "_raw")
    sites = [s.strip() for s in args.sites.split(",") if s.strip()]
    files = [f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
             if f.endswith(".parquet") and f.split("/")[0] in sites]
    missing = [s for s in sites if not any(f.startswith(s + "/") for f in files)]
    if missing:
        raise SystemExit(f"not in the dump: {', '.join(missing)}")
    todo = [f for f in files if not os.path.isfile(os.path.join(args.out, f.replace(".parquet", ".jsonl.gz")))]
    print(f"{len(files)} files in {len(sites)} sites; {len(files) - len(todo)} done before, {len(todo)} to do. "
          f"Output: {args.out}", flush=True)
    if args.max_files:
        todo = todo[:args.max_files]
    def fetch(f):
        """(path, seconds) of one downloaded file."""
        t = time.time()
        for attempt in range(5):  # the Hub drops connections now and then
            try:
                return hf_hub_download(REPO, f, repo_type="dataset", local_dir=scratch), time.time() - t
            except Exception as exc:  # noqa: BLE001 - retried, then reported
                if attempt == 4:
                    raise
                print(f"  download of {f} failed ({exc.__class__.__name__}: {exc}); retrying in 30 s", flush=True)
                time.sleep(30)

    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(1)  # the next file downloads while this one is filtered (network and CPU overlap)
    nxt = pool.submit(fetch, todo[0]) if todo else None
    t0, total_kept, total_chars = time.time(), 0, 0
    for i, f in enumerate(todo, 1):
        site = f.split("/")[0]
        dst = os.path.join(args.out, f.replace(".parquet", ".jsonl.gz"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        t1 = time.time()
        raw, download_s = nxt.result()
        nxt = pool.submit(fetch, todo[i]) if i < len(todo) else None
        waited = time.time() - t1
        t2 = time.time()
        min_score = args.so_min_score if site == "stackoverflow.com" else args.min_score
        seen, kept, chars = filter_file(raw, dst, site, min_score, args.min_answer_score, args.top_answers)
        os.remove(raw)
        total_kept, total_chars = total_kept + kept, total_chars + chars
        rate = (time.time() - t0) / i
        print(f"[{i}/{len(todo)}] {f}: kept {kept:,} of {seen:,} questions, {chars / 1e6:,.0f} MB text | "
              f"download {download_s:,.0f} s (waited {waited:,.0f}), filter {time.time() - t2:,.0f} s | total {total_kept:,} questions, "
              f"{total_chars / 1e9:,.2f} GB | left ~{(len(todo) - i) * rate / 60:,.0f} min", flush=True)
    print(f"done: {total_kept:,} questions, {total_chars / 1e9:,.2f} GB of text in {(time.time() - t0) / 60:,.0f} min",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
