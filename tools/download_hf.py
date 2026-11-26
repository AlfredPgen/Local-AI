"""Hugging Face datasets for tinyGPT v2, file by file into ai_training_data (raw files are
never kept: each is downloaded to <out>/_raw, converted to <out>/<name>/<file>.jsonl.gz and
deleted, while the next file downloads). A rerun skips finished files.

    python tools/download_hf.py medrag_textbooks pubmed cosmopedia finemath smoltalk
    python tools/download_hf.py finemath --fraction 0.2

Presets (pre-training text goes to ai_training_data/<name>, which data_prep --text-root reads
as a named JSONL source; chat data goes to ai_training_data/_sft/<name>, which it skips):
  medrag_textbooks  MedRAG/textbooks: 18 medical textbooks (Gray's Anatomy, Alberts, Harrison, ...);
                    the ~1,000-character chunks of a book are joined back into documents of
                    about --doc-chars characters, in book order
  pubmed            MedRAG/pubmed: PubMed abstracts (2023 baseline), title + abstract, one per paper
  cosmopedia        HuggingFaceTB/cosmopedia: the curated synthetic textbook parts (openstax,
                    khanacademy, stanford, wikihow, auto_math_text); not web_samples or stories
  finemath          HuggingFaceTB/finemath, finemath-4plus (the best-rated maths pages);
                    --fraction samples documents by a stable hash of their id
  smoltalk          HuggingFaceTB/smoltalk 'all' (SFT conversations) and SmolTalk2's preference
                    pairs without reasoning traces: for fine-tuning, not pre-training

Licences: Cosmopedia Apache-2.0, FineMath ODC-BY, SmolTalk per its card; MedRAG states none
(the textbooks are copyrighted books; PubMed abstracts follow NLM's terms): check before
sharing a model trained on them.
"""
import argparse
import fnmatch
import gzip
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.join(os.path.expanduser("~"), "ai_training_data")


def _stable_unit(key):
    return int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "big") / 2 ** 64


def _jsonl_rows(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _parquet_rows(path, columns):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    for g in range(pf.num_row_groups):
        yield from pf.read_row_group(g, columns=columns).to_pylist()


# each converter: (raw path, repo file name, args) -> iterator of output records (id, title, text, ...)
def conv_textbook(path, name, args):
    book = os.path.basename(name).rsplit(".", 1)[0]
    rows = sorted(_jsonl_rows(path), key=lambda r: int(r["id"].rsplit("_", 1)[1]))
    part, size, k = [], 0, 0
    for r in rows + [None]:
        if r is None or (size >= args.doc_chars and part):
            if part:
                yield {"id": f"{book}/{k}", "title": book.replace("_", " "), "text": "\n\n".join(part) + "\n",
                       "source": "MedRAG/textbooks"}
                k += 1
            part, size = [], 0
            if r is None:
                break
        part.append(r["content"].strip())
        size += len(r["content"])


def conv_pubmed(path, name, args):
    for r in _jsonl_rows(path):
        title, text = (r.get("title") or "").strip(), (r.get("content") or "").strip()
        if len(text) < 200:
            continue
        yield {"id": f"pmid{r.get('PMID')}", "title": title, "text": f"# {title}\n\n{text}\n" if title else text + "\n",
               "pmid": r.get("PMID"), "source": "MedRAG/pubmed"}


def conv_cosmopedia(path, name, args):
    part = name.split("/")[1]
    base = os.path.basename(name).rsplit(".", 1)[0]
    for i, r in enumerate(_parquet_rows(path, ["text", "format", "audience"])):
        text = (r["text"] or "").strip()
        if text:
            yield {"id": f"{part}/{base}/{i}", "title": "", "text": text + "\n", "subset": part,
                   "format": r["format"], "audience": r["audience"], "source": "HuggingFaceTB/cosmopedia"}


def conv_finemath(path, name, args):
    for r in _parquet_rows(path, ["url", "text", "score"]):
        key = r.get("url") or ""
        if args.fraction < 1 and _stable_unit(key) >= args.fraction:
            continue
        text = (r["text"] or "").strip()
        if text:
            yield {"id": key, "title": "", "text": text + "\n", "score": r.get("score"),
                   "source": "HuggingFaceTB/finemath"}


def conv_smoltalk(path, name, args):
    cols = None if "Preference" in name else ["messages", "source"]
    for i, r in enumerate(_parquet_rows(path, cols)):
        if "Preference" in name:
            rec = {"chosen": r.get("chosen"), "rejected": r.get("rejected"), "prompt": r.get("prompt")}
        else:
            rec = {"messages": r.get("messages"), "subset": r.get("source")}
        yield {"id": f"{os.path.basename(name)}/{i}", **rec, "source": "smoltalk"}


PRESETS = {
    "medrag_textbooks": {"repo": "MedRAG/textbooks", "files": ["chunk/*.jsonl"], "conv": conv_textbook},
    "pubmed": {"repo": "MedRAG/pubmed", "files": ["chunk/*.jsonl"], "conv": conv_pubmed},
    "cosmopedia": {"repo": "HuggingFaceTB/cosmopedia",
                   "files": [f"data/{p}/*.parquet" for p in ("openstax", "khanacademy", "stanford", "wikihow",
                                                             "auto_math_text")], "conv": conv_cosmopedia},
    "finemath": {"repo": "HuggingFaceTB/finemath", "files": ["finemath-4plus/*.parquet"], "conv": conv_finemath},
    "smoltalk": {"repo": "HuggingFaceTB/smoltalk", "files": ["data/all/*.parquet"], "conv": conv_smoltalk,
                 "sft": True, "extra": ("HuggingFaceTB/smoltalk2",
                                        ["Preference/llama_3.1_tulu_3_8b_preference_mixture_no_think-*.parquet"])},
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("presets", nargs="+", choices=sorted(PRESETS))
    p.add_argument("--root", default=ROOT)
    p.add_argument("--fraction", type=float, default=0.2, help="finemath: share of documents kept (stable hash)")
    p.add_argument("--doc-chars", type=int, default=20_000, help="medrag_textbooks: characters per document")
    p.add_argument("--max-files", type=int, default=0, help="a trial: at most this many files per preset")
    args = p.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    scratch = os.path.join(args.root, "_raw_hf")
    jobs = []  # (preset, repo, file, destination)
    for name in args.presets:
        pre = PRESETS[name]
        out_dir = os.path.join(args.root, "_sft", name) if pre.get("sft") else os.path.join(args.root, name)
        sources = [(pre["repo"], pre["files"])] + ([pre["extra"]] if pre.get("extra") else [])
        for repo, patterns in sources:
            files = sorted(f for f in api.list_repo_files(repo, repo_type="dataset")
                           if any(fnmatch.fnmatch(f, pat) for pat in patterns))
            if not files:
                raise SystemExit(f"{name}: no files in {repo} match {patterns}")
            for f in files:
                # flat in the source folder (data_prep looks in it and one level down): repo__path__file
                stem = f"{repo.split('/')[-1]}__{f}".rsplit(".", 1)[0].replace("/", "__")
                dst = os.path.join(out_dir, stem + ".jsonl.gz")
                if not os.path.isfile(dst):
                    jobs.append((name, repo, f, dst))
    if args.max_files:
        jobs = [j for name in args.presets for j in [j for j in jobs if j[0] == name][:args.max_files]]
    print(f"{len(jobs)} files to do ({', '.join(args.presets)}); finished files are skipped", flush=True)

    def fetch(job):
        t = time.time()
        for attempt in range(5):
            try:
                return hf_hub_download(job[1], job[2], repo_type="dataset", local_dir=scratch), time.time() - t
            except Exception as exc:  # noqa: BLE001 - retried, then reported
                if attempt == 4:
                    raise
                print(f"  download of {job[2]} failed ({exc.__class__.__name__}); retrying in 30 s", flush=True)
                time.sleep(30)

    pool = ThreadPoolExecutor(1)  # the next file downloads while this one is converted
    nxt = pool.submit(fetch, jobs[0]) if jobs else None
    t0, totals = time.time(), {}
    for i, job in enumerate(jobs, 1):
        name, repo, f, dst = job
        t1 = time.time()
        raw, dl = nxt.result()
        nxt = pool.submit(fetch, jobs[i]) if i < len(jobs) else None
        waited, t2 = time.time() - t1, time.time()
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        n = chars = 0
        with gzip.open(dst + ".part", "wt", encoding="utf-8") as out:
            for rec in PRESETS[name]["conv"](raw, f, args):
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                chars += len(rec.get("text") or "")
        os.replace(dst + ".part", dst)
        os.remove(raw)
        tn, tc = totals.get(name, (0, 0))
        totals[name] = (tn + n, tc + chars)
        left = (time.time() - t0) / i * (len(jobs) - i)
        print(f"[{i}/{len(jobs)}] {name} {f}: {n:,} records, {chars / 1e6:,.0f} MB text | download {dl:,.0f} s "
              f"(waited {waited:,.0f}), convert {time.time() - t2:,.0f} s | {name} so far {totals[name][0]:,} "
              f"records, {totals[name][1] / 1e9:,.2f} GB | left ~{left / 60:,.0f} min", flush=True)
    print("done: " + "; ".join(f"{k} {v[0]:,} records, {v[1] / 1e9:,.2f} GB text" for k, v in totals.items()),
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
