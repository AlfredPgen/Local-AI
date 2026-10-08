r"""Is checkpoint A really better than checkpoint B? A paired, document-level bootstrap.

Both models read the same validation documents (decoded to plain text, so
models with different tokenizers can be compared). For each document we get
bits per byte (BPB): total negative log-likelihood / (ln 2 x UTF-8 bytes).
The difference A - B is computed per document ("paired": the same text for
both models), and documents - not tokens - are resampled 2,000 times to get a
95% interval, because tokens inside one document are strongly correlated and a
token-level standard error would be far too optimistic.

    python compare_models.py tinyGPT_best.pt tiny_gpt_v4_best.pt
    python compare_models.py run1_best.pt run2_best.pt --dataset datasets\bio_v2 --docs 300 --device cpu

Reading the result: a negative difference means A compresses the text better
(lower BPB). If the 95% interval excludes 0 the difference is unlikely to be
noise. Use --device cpu while a training run is using the GPU.

A model trained on another dataset may have trained on some of these
validation documents. Its training dataset's docs.tsv says which (same source
and document id, or the same text under another name or source, by content
hash); those documents are left out. If that dataset is not on this
computer, a warning is printed and no verdict is given (--allow-different-datasets
gives one anyway).
"""

import argparse
import csv
import datetime
import gc
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data_prep  # noqa: E402
import tiny_gpt  # noqa: E402


def validation_documents(dataset_dir, per_source, max_chars, layout=None):
    """Evenly spaced validation documents per source, as text. With a dict as
    `layout`, layout[source] = (segments found, offset): segment k is the
    (k + offset)-th validation document of that source (offset None: unknown)."""
    manifest, arrays = data_prep.open_token_files(dataset_dir, verify=False)
    meta = manifest["tokenizer"]
    tok = data_prep.Tokenizer.from_file(os.path.join(dataset_dir, meta["file"]), meta.get("encode_mode", "lines"))
    docs = []
    for source, arr in sorted(arrays["val"].items()):
        boundary = tok.bos_id if tok.bos_id >= 0 else tok.eos_id
        starts = np.flatnonzero(np.asarray(arr) == boundary) if boundary >= 0 else np.array([0])
        if len(starts) == 0:
            starts = np.array([0])
        ends = np.append(starts[1:], len(arr))
        if layout is not None:  # a segment starts at a document's start token, or after its end token
            layout[source] = (len(starts), 0 if tok.bos_id >= 0 else 1 if boundary >= 0 else None)
        chosen = np.unique(np.linspace(0, len(starts) - 1, min(per_source, len(starts))).astype(int))
        for k in chosen:
            text = tok.decode(np.asarray(arr[starts[k]:ends[k]]).tolist()).strip()
            if len(text) >= 200:
                docs.append((source, int(k), text[:max_chars]))
    return docs


@torch.no_grad()
def document_nll(model, tok, text, device):
    """(total NLL in nats, UTF-8 bytes of the tokens the model predicted) for one
    document, scored in non-overlapping windows of the model's context. The bytes
    are those of the predicted tokens themselves, so a tokenizer that drops
    characters (collapsed spaces) or has no start token (the first token is never
    predicted) is not credited with bytes it never scored."""
    ids = ([tok.bos_id] if tok.bos_id >= 0 else []) + tok.encode(text)
    if len(ids) < 2:
        return 0.0, 0
    targets = [t for t in ids[1:] if not tok.is_special(t)]
    predicted_bytes = len(tok.decode(targets).encode("utf-8"))
    rows_x, rows_y = [], []
    for start in range(0, len(ids) - 1, model.ctx):
        chunk = ids[start:start + model.ctx + 1]
        if len(chunk) < 2:
            break
        rows_x.append(chunk[:-1])
        rows_y.append(chunk[1:])
    total = 0.0
    width = max(len(r) for r in rows_x)
    x = torch.zeros((len(rows_x), width), dtype=torch.long)
    y = torch.full((len(rows_y), width), -100, dtype=torch.long)
    for i, (a, b) in enumerate(zip(rows_x, rows_y)):
        x[i, :len(a)] = torch.tensor(a)
        y[i, :len(b)] = torch.tensor(b)
    for i in range(0, len(rows_x), 16):
        # the output layer in pieces: no (windows x context x vocabulary) table; -100 labels count 0
        total += -tiny_gpt.target_logprobs(model, x[i:i + 16].to(device), y[i:i + 16].to(device)).sum().item()
    return total, predicted_bytes


MIN_DOCS = 20  # fewer documents: numbers are shown, but no verdict


def _ledger_rows(dataset_dir):
    """Yields the column names of a dataset's docs.tsv, then each row as a dict
    (streamed: a full ledger can hold millions of documents); nothing without a
    docs.tsv. Rows end at \\n only, as data_prep reads them (older files kept \\r
    inside titles)."""
    path = os.path.join(dataset_dir, "docs.tsv")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8", newline="\n") as handle:
        head = handle.readline().rstrip("\r\n").split("\t")
        yield head
        for line in handle:
            yield dict(zip(head, line.rstrip("\r\n").split("\t")))


def _hash(row):
    """A row's content hash as an int, or None (older builds, or no hash)."""
    try:
        return int(row.get("content_hash") or "", 16)
    except ValueError:
        return None


def read_validation_ledger(dataset_dir, sources):
    """({source: [(doc_id, content hash or None), ...] in file order}, has hashes)
    for the validation rows of `sources` in a dataset's docs.tsv, or (None, False)
    without one. Documents are written to the token files in docs.tsv's order,
    so the k-th validation document of a source is its k-th validation row."""
    rows = _ledger_rows(dataset_dir)
    head = next(rows, None)
    if head is None:
        return None, False
    val = {}
    for row in rows:
        if row.get("status") == "val" and row.get("source") in sources:
            val.setdefault(row["source"], []).append((row["doc_id"], _hash(row)))
    return val, "content_hash" in head


def read_training_matches(dataset_dir, keys, hashes):
    """(the (source, doc_id) keys, the content hashes) among `keys` / `hashes`
    that a dataset's docs.tsv marks 'train', and whether it has content hashes;
    None without a docs.tsv. Only the wanted rows are kept (a full ledger can
    hold millions of documents). Keys are compared as data_prep's
    --keep-split-from compares them (/ and \\ alike)."""
    rows = _ledger_rows(dataset_dir)
    head = next(rows, None)
    if head is None:
        return None
    hit_keys, hit_hashes = set(), set()
    for row in rows:
        if row.get("status") != "train":
            continue
        key = data_prep._split_key(row.get("source", ""), row.get("doc_id", ""))
        if key in keys:
            hit_keys.add(key)
        h = _hash(row) if hashes else None
        if h is not None and h in hashes:
            hit_hashes.add(h)
    return hit_keys, hit_hashes, "content_hash" in head


def document_ids(dataset_dir, docs, layout):
    """(docs.tsv's (doc_id, content hash) of each (source, k, text) document, None
    where it cannot be told; whether the ledger has content hashes)."""
    val, has_hash = read_validation_ledger(dataset_dir, {source for source, _, _ in docs})
    out = []
    for source, k, _ in docs:
        ids = (val or {}).get(source, [])
        n, offset = layout.get(source, (0, None))
        j = None if offset is None or n != len(ids) else k + offset
        out.append(ids[j] if j is not None and j < len(ids) else None)
    return out, has_hash


def same_folder(a, b):
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def training_overlap(docs, dataset, layout, models):
    """Documents of `dataset`'s validation split that a model may have trained on.
    models: {label: the checkpoint's 'dataset' entry}. A model trained on exactly
    this dataset (same fingerprint) never saw them. For a model trained on another
    dataset, that dataset's docs.tsv says which documents it trained on: the same
    source and doc_id, or the same text (content hash: files renamed, moved or
    re-numbered, or the same paper in another source), as data_prep's
    --keep-split-from matches them. Returns (indices of docs to drop, {label: why
    its dataset differs but cannot be checked}, {label: why its dataset is
    unknown})."""
    try:
        with open(os.path.join(dataset, "manifest.json"), encoding="utf-8") as handle:
            fingerprint = json.load(handle).get("fingerprint")
    except (OSError, ValueError):
        fingerprint = None
    drop, unchecked, unknown, ids, has_hash = set(), {}, {}, None, False
    for label, info in models.items():
        path, fp = (info or {}).get("path"), (info or {}).get("fingerprint")
        if fp and fingerprint and fp == fingerprint or (not fp and path and same_folder(path, dataset)):
            continue
        if not path:
            unknown[label] = "its checkpoint does not say which dataset it was trained on"
            continue
        if same_folder(path, dataset):
            unchecked[label] = f"it was trained on an earlier build of {dataset} (another fingerprint)"
            continue
        if not os.path.isfile(os.path.join(path, "docs.tsv")):
            unchecked[label] = f"its training dataset {path} (or its docs.tsv) is not on this computer"
            continue
        if ids is None:
            ids, has_hash = document_ids(dataset, docs, layout)
        if any(d is None for d in ids):
            unchecked[label] = f"the documents of {dataset} cannot be matched to its docs.tsv"
            continue
        keys = [data_prep._split_key(source, d[0]) for (source, _, _), d in zip(docs, ids)]
        hashes = [d[1] for d in ids]
        found = read_training_matches(path, set(keys), {h for h in hashes if h is not None})
        if found is None:
            unchecked[label] = f"its training dataset {path} (or its docs.tsv) is not on this computer"
            continue
        hit_keys, hit_hashes, other_has_hash = found
        hits = {i for i, (key, h) in enumerate(zip(keys, hashes)) if key in hit_keys or h in hit_hashes}
        print(f"  model {label} was trained on {path}: {len(hits)} of these documents were in its training split; "
              "they are left out")
        if not (has_hash and other_has_hash):
            print(f"  note: {dataset if not has_hash else path} has no content hashes (an older build): only "
                  "documents with the same source and id were checked, not the same text under another name")
        drop |= hits
    return drop, unchecked, unknown


def bootstrap(nll_a, nll_b, bytes_a, bytes_b, reps, seed):
    """BPB(A) - BPB(B), each model divided by the bytes it predicted, with a
    percentile interval from resampling documents and a two-sided p-value."""
    rng = np.random.default_rng(seed)
    n = len(nll_a)
    ln2 = math.log(2)
    point = nll_a.sum() / (ln2 * bytes_a.sum()) - nll_b.sum() / (ln2 * bytes_b.sum())
    idx = rng.integers(0, n, (reps, n))
    stats = nll_a[idx].sum(1) / (ln2 * bytes_a[idx].sum(1)) - nll_b[idx].sum(1) / (ln2 * bytes_b[idx].sum(1))
    low, high = np.percentile(stats, [2.5, 97.5])
    p = min(1.0, 2 * min((stats <= 0).mean(), (stats >= 0).mean()))
    return point, low, high, p


def fmt_p(p, reps):
    return f"<{1 / reps:.0e}" if p == 0 else f"{p:.3f}"


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a", help="checkpoint A (.pt or --export folder)")
    p.add_argument("b", help="checkpoint B")
    p.add_argument("--dataset", help="dataset whose validation documents are used (default: A's dataset)")
    p.add_argument("--docs", type=int, default=150, help="documents per source")
    p.add_argument("--max-chars", type=int, default=20_000, help="characters used per document")
    p.add_argument("--reps", type=int, default=2000, help="bootstrap resamples")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    p.add_argument("--trust-checkpoint", action="store_true")
    p.add_argument("--allow-different-datasets", action="store_true",
                   help="give a verdict even when a model was trained on another dataset that cannot be checked "
                        "for these validation documents")
    args = p.parse_args()
    for path in (args.a, args.b):
        if not os.path.exists(path):
            p.error(f"not found: {path}")
    if not 1 <= args.docs <= 5000 or not 100 <= args.reps <= 100_000:
        p.error("--docs must be 1-5000 and --reps 100-100000")
    device = tiny_gpt.select_device(args.device)
    trained_on = {}
    model_a, tok_a, cfg_a, obj_a = tiny_gpt.load_for_inference(args.a, device, args.trust_checkpoint)
    trained_on["A"] = obj_a.get("dataset") if isinstance(obj_a.get("dataset"), dict) else None
    del obj_a  # the raw checkpoint (weights, optimizer state) is not needed beside the model
    gc.collect()
    model_b, tok_b, cfg_b, obj_b = tiny_gpt.load_for_inference(args.b, device, args.trust_checkpoint)
    trained_on["B"] = obj_b.get("dataset") if isinstance(obj_b.get("dataset"), dict) else None
    del obj_b
    gc.collect()
    dataset = args.dataset or (trained_on["A"] or {}).get("path")
    if not dataset or not os.path.isdir(dataset):
        p.error("no dataset: pass --dataset (checkpoint A's dataset folder is missing)")
    layout = {}
    docs = validation_documents(dataset, args.docs, args.max_chars, layout)
    print(f"A: {args.a}\nB: {args.b}\n{len(docs)} validation documents from {dataset} on {tiny_gpt.device_label(device)}")
    # a model trained on another dataset may have trained on some of these documents: its BPB would look too good
    drop, unchecked, unknown = training_overlap(docs, dataset, layout, trained_on)
    if drop:
        docs = [d for i, d in enumerate(docs) if i not in drop]
        print(f"  {len(docs)} documents left that neither model trained on")
    for who, why in unknown.items():
        print(f"  note: model {who} cannot be checked for training on these documents: {why}")
    for who, why in unchecked.items():
        print(f"  WARNING: model {who} was trained on another dataset and may have trained on some of these "
              f"documents ({why}); its bits per byte may look too good")
    rows, skipped, lossy = [], 0, {"A": 0, "B": 0}
    for i, (source, k, text) in enumerate(docs, 1):
        n_bytes = len(text.encode("utf-8"))
        nll_a, bytes_a = document_nll(model_a, tok_a, text, device)
        nll_b, bytes_b = document_nll(model_b, tok_b, text, device)
        if not (math.isfinite(nll_a) and math.isfinite(nll_b)) or not bytes_a or not bytes_b:
            skipped += 1
            continue
        lossy["A"] += bytes_a < 0.99 * n_bytes
        lossy["B"] += bytes_b < 0.99 * n_bytes
        rows.append((source, k, n_bytes, nll_a, nll_b, bytes_a, bytes_b))
        if i % 50 == 0:
            print(f"  scored {i}/{len(docs)} documents", flush=True)
    if skipped:
        print(f"  {skipped} documents skipped (no tokens, or a non-finite loss)")
    for who, count in lossy.items():
        if count:
            print(f"  note: model {who}'s tokenizer drops over 1% of the characters in {count} documents "
                  "(e.g. collapsed spaces); its bits per byte count only the bytes it predicts")
    if not rows:
        raise SystemExit("No documents could be scored.")
    sources = sorted({r[0] for r in rows})
    print(f"\n{'documents':<14} {'n':>5} {'BPB A':>8} {'BPB B':>8} {'A - B':>9} {'95% interval':>21} {'p':>7}  A better on")
    for label, sel in [("all", rows)] + [(s, [r for r in rows if r[0] == s]) for s in sources]:
        na, nbb = np.array([r[3] for r in sel]), np.array([r[4] for r in sel])
        ba, bb = np.array([r[5] for r in sel], dtype=float), np.array([r[6] for r in sel], dtype=float)
        point, low, high, pval = bootstrap(na, nbb, ba, bb, args.reps, args.seed)
        bpb_a, bpb_b = na.sum() / (math.log(2) * ba.sum()), nbb.sum() / (math.log(2) * bb.sum())
        wins = int(((na / ba) < (nbb / bb)).sum())
        print(f"{label:<14} {len(sel):>5} {bpb_a:>8.4f} {bpb_b:>8.4f} {point:>+9.4f} [{low:>+9.4f}, {high:>+9.4f}] "
              f"{fmt_p(pval, args.reps):>7}  {wins}/{len(sel)} docs" + ("  (too few for a verdict)"
                                                                          if len(sel) < MIN_DOCS else ""))
    point, low, high, pval = bootstrap(np.array([r[3] for r in rows]), np.array([r[4] for r in rows]),
                                       np.array([r[5] for r in rows], dtype=float),
                                       np.array([r[6] for r in rows], dtype=float), args.reps, args.seed)
    if len(rows) < MIN_DOCS:
        verdict = f"too few documents ({len(rows)}) for a verdict; use at least {MIN_DOCS}"
    elif unchecked and not args.allow_different_datasets:
        verdict = (f"none: model {' and '.join(unchecked)} may have trained on these documents (see the warning "
                   "above; --allow-different-datasets gives one anyway)")
    else:
        verdict = ("A is better" if high < 0 else "B is better" if low > 0 else "no reliable difference")
    print(f"\nVerdict: {verdict} (difference in bits per byte {point:+.4f}, 95% interval {low:+.4f} to {high:+.4f}).")
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    stem = os.path.join(os.path.dirname(os.path.abspath(args.a)),
                        f"compare_{os.path.splitext(os.path.basename(os.path.normpath(args.a)))[0]}_vs_"
                        f"{os.path.splitext(os.path.basename(os.path.normpath(args.b)))[0]}_{stamp}")
    out, k = stem + ".csv", 2
    while os.path.exists(out):
        out, k = f"{stem}_{k}.csv", k + 1
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "document_index", "bytes", "nll_a_nats", "nll_b_nats", "bytes_a", "bytes_b",
                         "bpb_a", "bpb_b"])
        for source, k, n_bytes, na, nbb, ba, bb in rows:
            writer.writerow([source, k, n_bytes, f"{na:.3f}", f"{nbb:.3f}", ba, bb,
                             f"{na / (math.log(2) * ba):.5f}", f"{nbb / (math.log(2) * bb):.5f}"])
    print(f"Per-document results: {out}")
    tiny_gpt.record_experiment(os.path.dirname(os.path.abspath(args.a)), {
        "event": "comparison", "name": f"{os.path.basename(args.a)} vs {os.path.basename(args.b)}",
        "checkpoint": os.path.abspath(args.a), "dataset": os.path.basename(os.path.normpath(dataset)),
        "notes": f"{verdict}; delta BPB {point:+.4f} [{low:+.4f}, {high:+.4f}], p={fmt_p(pval, args.reps)}, "
                 f"{len(rows)} docs; {os.path.basename(out)}"})


if __name__ == "__main__":
    main()
