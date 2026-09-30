r"""Estimate what data_prep.py would build, without building it: kept documents,
characters and tokens per source, and the model sizes those tokens support,
with training time on this PC.

    python estimate_dataset.py                    # everything in ~\ai_training_data
    python estimate_dataset.py --text-root D:\data --jsonl "D:\pes2o\*.json.gz" --jsonl-name pes2o
    python estimate_dataset.py --sample 0.05 --hours 12,24,48,72

Any data_prep.py option means the same here (they are passed to data_prep's own
parser, so a data_prep command can be pasted with estimate_dataset.py in front).
Without data_prep options it uses the usual layout: the folders of
ai_training_data, Wikipedia (200,000 articles), FineWeb-Edu (score >= 3) and
peS2o, with keywords_biology.txt (at least 3 different terms) on the web sources.

How: a sample of every source (--sample of the web records, at least 300
files of each folder) goes through data_prep's real per-document checks:
cleaning, English test, minimum length, quality score and keyword filter. The
kept share and size are scaled up to the whole source. Characters per token
come from a SentencePiece tokenizer trained on the kept sample with data_prep's
settings (or --tokenizer: a dataset folder, .model file or checkpoint). Not
modelled: duplicates and repeated boilerplate lines (a few per cent in bio_v3).
With --wiki-max-docs (and --parquet-select top) the sample is ranked the way
data_prep ranks, so the kept size reflects the shorter top-ranked documents.

Model sizes follow tiny_gpt.py's planner: at most --max-epochs passes over the
data and at least --tokens-per-param training tokens per parameter. Training
time uses the speed of the last training run in experiments.csv, converted to
floating-point operations per second so that it carries over to other model
sizes (or --tflops). The last column is the Chinchilla fit of loss against
parameters and tokens (Hoffmann et al. 2022): its values belong to their data,
but it ranks the options.
"""

import argparse
import csv
import fnmatch
import glob
import math
import os
import random
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(os.path.expanduser("~"), "ai_training_data")
sys.path.insert(0, HERE)
import data_prep as dp  # noqa: E402

CHINCHILLA = (1.69, 406.4, 410.7, 0.34, 0.28)  # E, A, B, alpha, beta (Hoffmann et al. 2022, approach 3)


def default_prep_args():
    args = ["--text-root", ROOT]
    wiki = os.path.join(ROOT, "wikipedia")
    if os.path.isdir(wiki):
        args += ["--wiki-dir", wiki, "--wiki-max-docs", "200000"]
    fineweb = os.path.join(ROOT, "web", "fineweb-edu", "sample", "10BT", "*.parquet")
    if glob.glob(fineweb):
        args += ["--parquet", fineweb, "--parquet-name", "fineweb", "--min-score", "3"]
    pes2o = os.path.join(ROOT, "web", "pes2o", "data", "v2", "*.json.gz")
    if glob.glob(pes2o):
        args += ["--jsonl", pes2o, "--jsonl-name", "pes2o"]
    return args + ["--include-keywords", os.path.join(HERE, "keywords_biology.txt"), "--keyword-min-distinct", "3"]


class Tally:
    def __init__(self, name, kind):
        self.name, self.kind = name, kind
        self.total = 0            # files or records in the source
        self.sampled = 0          # of them examined
        self.sampled_bytes = 0    # (folders) raw size of the examined files
        self.total_bytes = 0
        self.kept = 0
        self.kept_chars = 0
        self.reasons = {}
        self.texts, self.text_chars = [], 0
        self.max_docs, self.select = 0, "first"
        self.ranked = []          # (keyword score, characters) of kept sample documents, for 'top' selection

    def keep(self, text):
        self.kept += 1
        self.kept_chars += len(text)
        if self.text_chars < 6_000_000:
            self.texts.append(text)
            self.text_chars += len(text)

    def skip(self, reason):
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def estimate(self):
        """(documents, characters) kept from the whole source."""
        if self.kind == "markdown":
            docs = self.total * self.kept / max(self.sampled, 1)
            chars = self.total_bytes * self.kept_chars / max(self.sampled_bytes, 1)
        else:
            scale = self.total / max(self.sampled, 1)
            docs, chars = self.kept * scale, self.kept_chars * scale
        if self.max_docs and docs > self.max_docs:
            if self.select == "top" and self.ranked:
                # data_prep keeps the best-ranked documents, which are shorter than average: take the same
                # top share of the sample
                share = self.max_docs / docs
                top = sorted(self.ranked, reverse=True)[:max(1, round(len(self.ranked) * share))]
                chars = self.max_docs * sum(c for _, c in top) / len(top)
            else:
                chars *= self.max_docs / docs
            docs = self.max_docs
        return docs, chars


def examine(tally, title, text, use_kw):
    reason, cleaned = dp._examine(title, text, use_kw)[:2]
    if reason:
        tally.skip(reason)
        return None
    tally.keep(cleaned)
    return len(cleaned)


def markdown_source(name, dirs, args, use_kw, rng):
    t = Tally(name, "markdown")
    patterns = args.md_glob or list(dp.MD_PATTERNS_DEFAULT)
    files, seen = [], set()
    for directory in dirs:
        found = []
        for pattern in patterns:
            spec = os.path.join(directory, "**", pattern) if args.md_recursive else os.path.join(directory, pattern)
            found.extend(glob.glob(spec, recursive=args.md_recursive))
        for path in sorted(set(found)):
            real = os.path.realpath(path)
            rel = os.path.relpath(path, directory)
            if real in seen or not os.path.isfile(real):
                continue
            seen.add(real)
            if any(fnmatch.fnmatch(os.path.basename(path), ex) or fnmatch.fnmatch(rel, ex) for ex in args.md_exclude):
                continue
            size = os.path.getsize(path)
            if size <= 512 * 1024 * 1024:
                files.append((path, size))
    t.total, t.total_bytes = len(files), sum(s for _, s in files)
    n = min(len(files), max(300, int(args.sample * len(files))))
    for path, size in rng.sample(files, n):
        with open(path, "rb") as handle:
            text = dp.decode_bytes(handle.read(), {"bom_removed": 0, "decode_errors_docs": 0,
                                                   "decode_errors_chars": 0})
        t.sampled += 1
        t.sampled_bytes += size
        examine(t, os.path.splitext(os.path.basename(path))[0], text, use_kw)
    return t


def table_source(name, kind, source, args, use_kw, min_score, max_docs, select):
    """Wikipedia (Arrow) or Parquet: every k-th record batch is examined."""
    import pyarrow.parquet as pq
    t = Tally(name, kind)
    t.max_docs = max_docs
    t.select = "top" if select == "top" or (select == "auto" and use_kw and max_docs) else "first"
    k = max(1, round(1 / args.sample))
    kw = dp._W_KW if use_kw else None
    need = kw is not None and bool(kw.include_patterns)

    def batches():
        if kind == "parquet":
            for f, path in enumerate(source.paths):
                handle = pq.ParquetFile(path)
                t.total += handle.metadata.num_rows
                names = handle.schema_arrow.names
                cols = [c for c in (source.text_col, source.title_col, source.id_col, source.score_col) if c in names]
                for g in range(0, handle.num_row_groups, k):
                    for b, batch in enumerate(handle.read_row_group(g, columns=cols).to_batches()):
                        yield f, b, {c: batch.column(c) for c in cols}
        else:
            for n, (f, b, arrays) in enumerate(source.batches()):
                t.total += len(arrays[source.text_col])
                if n % k == 0:
                    yield f, b, arrays

    for f, b, arrays in batches():
        hits, chars, score_ok = dp._vector_scores(arrays, kw, source.text_col, source.title_col, source.score_col,
                                                  min_score)
        t.sampled += len(chars)
        for r in range(len(chars)):
            if not score_ok[r]:
                t.skip("score_below_min")
            elif chars[r] == 0:
                t.skip("empty")
            elif need and hits[r] < kw.min_hits:
                t.skip("keyword_min_hits")
            else:
                _, title, text, _ = dp._row(arrays, r, source, f, b)
                kept = examine(t, title, text, use_kw)
                if kept is not None:  # data_prep's 'top' score: keyword hits / sqrt(characters)
                    t.ranked.append((hits[r] / math.sqrt(max(int(chars[r]), 500)), kept))
    return t


def jsonl_source(name, paths, args, use_kw, max_docs):
    t = Tally(name, "jsonl")
    t.max_docs = max_docs
    k = max(1, round(1 / args.sample))
    kw = dp._W_KW if use_kw else None
    for path in paths:
        for n, (block, row0) in enumerate(dp._jsonl_arrow_raw(path, args.text_column, args.title_column,
                                                              args.id_column, 4 << 20)):
            t.total += block.num_rows
            if n % k:
                continue
            count, skipped, docs = dp._jsonl_arrow_block(block, row0, path, args.text_column, args.title_column,
                                                         args.id_column, kw)
            t.sampled += count
            for reason, c in skipped.items():
                if c:
                    t.reasons[reason] = t.reasons.get(reason, 0) + c
            for _, title, text, _ in docs:
                examine(t, title, text, use_kw)
    return t


def train_tokenizer(tallies, vocab, work, args=None):
    """A tokenizer like data_prep.py's, trained on the kept sample (with --superbpe:
    the same two stages)."""
    import sentencepiece as spm
    extra = max(1, round(vocab * args.superbpe_fraction)) if getattr(args, "superbpe", False) else 0
    corpus = os.path.join(work, "corpus.txt")
    with open(corpus, "w", encoding="utf-8", newline="\n") as out:
        for t in tallies:
            budget = 3_000_000
            for text in t.texts:
                for line in text.split("\n"):
                    if line.strip():
                        out.write(line[:4000] + "\n")
                budget -= len(text)
                if budget <= 0:
                    break
    spm.SentencePieceTrainer.train(
        input=corpus, model_prefix=os.path.join(work, "spm"), model_type="bpe", vocab_size=vocab - extra,
        character_coverage=0.9995, byte_fallback=True, split_digits=True, unk_id=0, bos_id=1, eos_id=2, pad_id=-1,
        user_defined_symbols=["\n", "\t"], remove_extra_whitespaces=False, allow_whitespace_only_pieces=True,
        normalization_rule_name="identity", max_sentence_length=16384, hard_vocab_limit=False,
        input_sentence_size=3_000_000, shuffle_input_sentence=True, num_threads=os.cpu_count() or 4, minloglevel=2)
    tok = dp.Tokenizer.from_file(os.path.join(work, "spm.model"), "lines")
    if extra:
        proto, _ = dp.superbpe_extend(tok.proto, corpus, extra, args.superbpe_max_words,
                                      threads=os.cpu_count() or 4)
        tok = dp.Tokenizer(proto, "lines")
    return tok


def chars_per_token(tok, texts, limit=2_000_000):
    chars = tokens = 0
    for text in texts:
        chars += len(text)
        tokens += len(tok.encode(text))
        if chars >= limit:
            break
    return chars / tokens if tokens else float("nan")


def measured_speed(path):
    """FLOP/s of the last GPU training run in experiments.csv (6N + 12*L*ctx*d per token)."""
    try:
        with open(path, encoding="utf-8-sig", newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if r.get("event") == "training run"]
    except OSError:
        return None, ""
    for r in reversed(rows):
        try:
            n, layers, ctx, d = (int(r[c]) for c in ("params", "layers", "ctx", "d_model"))
            tokens, hours = float(r["tokens_seen"]), float(r["train_hours"])
        except (KeyError, TypeError, ValueError):
            continue
        if hours > 0.2 and tokens > 0:
            tok_s = tokens / (hours * 3600)
            flops = tok_s * (6 * n + 12 * layers * ctx * d)
            return flops, f"{r['name']} run of {r['time'][:10]}: {n / 1e6:.1f} M parameters at {tok_s:,.0f} tokens/s"
    return None, ""


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    est = argparse.ArgumentParser(add_help=False)
    est.add_argument("--sample", type=float, default=0.02, help="share of web records examined")
    est.add_argument("--tokenizer", help="measure characters per token with this tokenizer instead of a new one")
    est.add_argument("--hours", default="12,24,48,72", help="training time budgets to tabulate")
    est.add_argument("--tflops", type=float, help="training speed in TFLOP/s (default: from experiments.csv)")
    est.add_argument("--tokens-per-param", type=float, default=20.0)
    est.add_argument("--max-epochs", type=float, default=4.0)
    est.add_argument("--sample-seed", dest="est_seed", type=int, default=0)
    est.add_argument("-h", "--help", action="store_true")
    opts, rest = est.parse_known_args(argv)
    if opts.help:
        print(__doc__)
        est.print_help()
        return
    if not 0 < opts.sample <= 1:
        raise SystemExit("--sample must be in (0, 1]")
    rest = rest or default_prep_args()
    args = dp.parse_args(rest + ["--out", os.path.join(tempfile.gettempdir(), "estimate_dataset_unused")])
    args.sample = opts.sample
    rng = random.Random(opts.est_seed)
    t0 = time.time()

    include = dp.read_terms(args.include_keywords)
    exclude = dp.read_terms(args.exclude_keywords)
    kw = dp.KeywordFilter(include, exclude, args.keyword_min_hits, args.keyword_min_distinct, args.title_weight)
    dp._init_worker(kw, {"language": args.language, "max_replacement_ratio": args.max_replacement_ratio,
                         "min_chars": args.min_chars, "exact_dedup": args.exact_dedup,
                         "boilerplate": "off", "boilerplate_min_len": args.boilerplate_min_len})
    spec = args.keyword_sources.strip().lower()

    def filtered(name, kind):
        if not kw.active:
            return False
        if spec in ("all", "web"):
            return spec == "all" or kind != "markdown"
        return name.lower() in {s.strip() for s in spec.split(",")}

    tallies = []
    for name, dirs in dp.parse_text_sources(args.md_dir).items():
        print(f"[{name}] examining a sample of the folder ...", flush=True)
        tallies.append(markdown_source(name, dirs, args, filtered(name, "markdown"), rng))
    if args.wiki_dir:
        print(f"[wiki] examining 1 in {round(1 / args.sample)} record batches ...", flush=True)
        src = dp.ArrowTableSource("hf_arrow", dp.ArrowTableSource.hf_files(args.wiki_dir, args.wiki_split),
                                  args.text_column, args.title_column, args.id_column)
        tallies.append(table_source("wiki", "hf_arrow", src, args, filtered("wiki", "hf_arrow"), None,
                                    args.wiki_max_docs, args.wiki_select))
    if args.parquet:
        name = args.parquet_name
        print(f"[{name}] examining 1 in {round(1 / args.sample)} row groups ...", flush=True)
        paths = sorted({p for pattern in args.parquet for p in glob.glob(pattern)})
        src = dp.ArrowTableSource("parquet", paths, args.text_column, args.title_column, args.id_column,
                                  args.score_column)
        tallies.append(table_source(name, "parquet", src, args, filtered(name, "parquet"), args.min_score,
                                    args.parquet_max_docs, args.parquet_select))
    if args.jsonl:
        name = args.jsonl_name
        print(f"[{name}] examining 1 in {round(1 / args.sample)} blocks (the whole file is read to count) ...",
              flush=True)
        paths = sorted({p for pattern in args.jsonl for p in glob.glob(pattern)})
        tallies.append(jsonl_source(name, paths, args, filtered(name, "jsonl"), args.jsonl_max_docs))

    estimates = {t.name: t.estimate() for t in tallies}
    train_chars = sum(c for _, c in estimates.values())
    vocab = args.vocab_size or dp.auto_vocab_size(train_chars)
    with tempfile.TemporaryDirectory() as work:
        if opts.tokenizer:
            tok = dp.load_tokenizer_from_any(opts.tokenizer)
            how = f"tokenizer {opts.tokenizer}"
        else:
            kind = "SuperBPE" if args.superbpe else "SentencePiece"
            print(f"training a {vocab:,}-piece {kind} tokenizer on the kept sample ...", flush=True)
            tok = train_tokenizer(tallies, vocab, work, args)
            how = f"a {vocab:,}-piece {kind} tokenizer trained on the kept sample"
        cpt = {t.name: chars_per_token(tok, t.texts) for t in tallies}

    print(f"\nEstimated dataset (sample {args.sample:.0%} of web records, >= 300 files per folder; "
          f"characters per token from {how}; {time.time() - t0:,.0f} s)\n")
    print(f"{'source':<11}{'kind':<10}{'in source':>12}{'kept docs':>12}{'kept chars':>13}{'chars/tok':>10}"
          f"{'train tokens':>15}  main skip reasons")
    total_tokens = total_val = 0
    for t in tallies:
        docs, chars = estimates[t.name]
        val = min(args.val_fraction * chars, args.val_max_chars) if docs >= 2 else 0
        ratio = cpt[t.name] if math.isfinite(cpt[t.name]) else 4.0
        train = (chars - val) / ratio
        total_tokens += train
        total_val += val / ratio
        reasons = ", ".join(f"{r} {c / max(t.sampled, 1):.0%}" for r, c in
                            sorted(t.reasons.items(), key=lambda x: -x[1])[:3])
        cap = f" (cap {t.max_docs:,})" if t.max_docs and docs >= t.max_docs else ""
        print(f"{t.name:<11}{t.kind:<10}{t.total:>12,}{docs:>12,.0f}{chars / 1e6:>12,.0f}M{ratio:>10.2f}"
              f"{train / 1e6:>14,.0f}M  {reasons}{cap}")
    print(f"{'total':<21}{'':>12}{'':>12}{'':>13}{'':>10}{total_tokens / 1e6:>14,.0f}M  "
          f"(+ {total_val / 1e6:,.0f}M validation tokens)")

    # ---------------- model sizes ----------------
    import tiny_gpt
    ladder = tiny_gpt.shape_ladder(vocab)
    shapes = [(tiny_gpt.count_params(c)[0], c) for c in ladder]
    r, epochs, unique = opts.tokens_per_param, opts.max_epochs, total_tokens
    data_tokens = epochs * unique
    flops, source = (opts.tflops * 1e12, f"--tflops {opts.tflops:g}") if opts.tflops else \
        measured_speed(os.path.join(HERE, "experiments.csv"))

    def ctx_of(c):  # tiny_gpt's rule: long documents (books) take the width's maximum context
        return 256 if c.d_model <= 192 else (512 if c.d_model <= 512 else 1024)

    def per_token(n, c):
        return 6 * n + 12 * c.n_layers * ctx_of(c) * c.d_model

    def chinchilla(n, d):
        e, a, b, alpha, beta = CHINCHILLA
        return e + a / n ** alpha + b / d ** beta

    commands = []

    def row(label, n, c, d):
        hours = d * per_token(n, c) / flops / 3600 / 0.92 if flops else float("nan")
        commands.append(f"{label}: python tiny_gpt.py --dataset <folder> --d-model {c.d_model} --layers {c.n_layers} "
                        f"--train-tokens {int(d / 1e7) * 10_000_000}")
        return (f"{label:<16}{n / 1e6:>9.1f}M  d{c.d_model:<5} L{c.n_layers:<3}{d / 1e9:>9.2f}B{d / n:>9.0f}"
                f"{d / unique:>8.2f}{hours:>10,.1f}{chinchilla(n, d):>11.2f}")

    print(f"\nModel sizes (tiny_gpt planner rules: <= {epochs:g} epochs, >= {r:g} tokens per parameter)")
    if flops:
        print(f"speed: {flops / 1e12:.1f} TFLOP/s, from the {source}")
    else:
        print("speed unknown (no training run in experiments.csv): pass --tflops for training hours")
    print(f"{'option':<16}{'params':>10}  {'shape':<11}{'tokens':>10}{'tok/par':>9}{'epochs':>8}{'hours':>10}"
          f"{'Chinchilla':>11}")
    fit = [(n, c) for n, c in shapes if n * r <= data_tokens]
    if fit:
        n, c = fit[-1]
        print(row("data only", n, c, data_tokens))
    if flops:
        for hours in (float(h) for h in opts.hours.split(",") if h.strip()):
            best = None
            for n, c in shapes:
                tokens = min(flops * hours * 3600 * 0.92 / per_token(n, c), data_tokens)
                if tokens >= r * n:
                    best = (n, c, tokens)
            if best:
                print(row(f"{hours:g} h budget", *best))
    print("\nChinchilla = fitted loss (nats per token) of Hoffmann et al. 2022 on their data: lower is better; "
          "use it to rank the options, not as a prediction of your validation loss.")
    print("\nCommands (--train-tokens sets how much is trained; --steps alone only splits the data-set budget into "
          "steps):\n  " + "\n  ".join(commands))


if __name__ == "__main__":
    main()
