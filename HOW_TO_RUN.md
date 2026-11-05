# How to run the tinyGPT pipeline

A short, practical run sheet. The long explanations are in `tinyGPT_learning_guide.docx`.

Open a terminal in VS Code (PowerShell) and go to the project folder first:

```
cd $HOME\local-ai
```

Long commands below are split over several lines with a backtick (`` ` ``) at the end of each line; PowerShell
joins them. You can also type them on one line without the backticks.

**One rule:** only one job may use the GPU at a time (two at once crashed the graphics driver before). While a
training run is going, give every other script `--device cpu`.

---

## The simple way: run_pipeline.py

Once the text is in `ai_training_data` (step 1 below), three commands do the rest:

```
python run_pipeline.py build               # build the dataset datasets\bio_all (~2 hours)
python run_pipeline.py plan --hours 48     # show which model fits in 48 hours (trains nothing)
python run_pipeline.py train --hours 48    # train it
python run_pipeline.py resume              # continue after Ctrl+C or a shutdown
python run_pipeline.py continue --hours 24 # after adding more text: train further from the first model
python run_pipeline.py posttrain           # teach the trained model to answer questions (SFT, DPO, evaluation)
```

- `--hours` is how long you are willing to wait. The planner picks the largest model that can read about 20
  tokens of text per parameter in that time, using the speed this laptop reached in its last training run.
- Each command prints the full `data_prep.py` or `tiny_gpt.py` command it runs, so you can see (and copy) the
  options. The settings (data folder, dataset name, run name) are at the top of `run_pipeline.py`.
- `python run_pipeline.py estimate` shows the expected dataset size first, without building anything.
- `plan`, `train`, `resume`, `continue` and `posttrain` refuse to start while the GPU is busy (the plan times
  training steps on the GPU, and a busy GPU may mean the run is still training).
- The settings at the top of `run_pipeline.py` describe the current run: `DATASET` `datasets\bio_all`, `RUN_NAME`
  `tinyGPT_muon` (so `resume` continues it), `SUPERBPE = True` and `OPTIMIZER = "muon"` (Muon for the block matrices,
  Step 4). `train` refuses a name that already exists: give a new one with `--name`, for example
  `python run_pipeline.py train --hours 48 --name tinyGPT_48h`.

The rest of this file explains the steps and options behind these commands.

---

## The whole pipeline at a glance

| Step | What happens | Script | How long |
|---|---|---|---|
| 1 | Put text into `$HOME\ai_training_data` | `convert_to_markdown.py`, `download_pmc.py`, `download_arxiv.py`, `download_rxiv.py`, `download_docs.py`, `youtube_transcripts.py` | minutes to days |
| 2 | (Optional) See how big the dataset will be and which model it supports | `estimate_dataset.py` | ~7 min |
| 3 | Clean, filter and tokenise everything into a dataset folder | `data_prep.py` | ~2-2.5 h |
| 4 | Train the model | `tiny_gpt.py` | 12 h to days |
| 5 | Look at the results | `view_pt.py`, `tiny_gpt.py --benchmark`, `compare_models.py` | minutes |
| 6 | Use or fine-tune the model | `tiny_gpt.py --generate`, `finetune.py`, `posttrain.py`, `detect_text.py` | minutes to hours |

---

## Step 1: put text into ai_training_data

Every sub-folder of `ai_training_data` becomes one **source** (books, articles, notes, scripts, pmc, lectures...),
with its own validation loss. Folders starting with `_` and the folders `wikipedia`, `web` and `images` are not
read as text folders (Wikipedia and the web data are read separately in step 3).

**PDFs, Word, PowerPoint, Excel -> Markdown**

```
python convert_to_markdown.py "D:\SomeFolder" --out $HOME\ai_training_data `
    --by-type --types pdf --name-prefix SomeFolder__ --recursive --workers 3
```

- `--by-type` puts long PDFs in `books`, others in `articles` (and slides, tables, code in their folders).
- `--types pdf` converts only PDFs from the folder (use `pdf,docx` for more).
- `--name-prefix` avoids name clashes between collections.
- Rerunning skips files already converted. `conversion_manifest.json` (in `--out`) remembers which input made
  each output, so a file added later gets a name of its own instead of being skipped, and nothing is converted twice.
- `--ocr` also reads scanned pages (pages that are only pictures) with Tesseract OCR. It is slow, a few seconds
  per page, so only the scanned pages are read this way: under 200 characters of text and mostly covered by
  images. The other pages keep their normal conversion. Without `--ocr` the report says, for example, `3 of 120
  pages look scanned: rerun with --ocr --overwrite`.
- Each PDF is converted in its own process with a time limit (`--timeout`, default 3600 s), so one damaged PDF
  costs only that file.
- `--max-cols 30` (default): CSV, TSV and Excel tables keep their first 30 columns, with a note saying how many
  were left out.
- Compressed data files (`.vcf.gz`, `.fa.gz`, `.bam.gz` ...) are refused: they are data, not text. GWAS summary
  statistics are summarised instead (see the guide).

**Open-access papers from PubMed Central**

```
python download_pmc.py --out $HOME\ai_training_data\pmc --max-papers 100000
python download_pmc.py --out $HOME\ai_training_data\pmc --keywords keywords.txt --max-papers 300000   # select with keywords.txt
```

Already done: 99,664 papers (about 1.25 billion tokens). `--max-papers 200000` would add the next 100,000.

**arXiv papers**

```
python download_arxiv.py --out $HOME\ai_training_data\arxiv              # first run; a rerun continues
python download_arxiv.py --out $HOME\ai_training_data\arxiv --update     # later: add papers new on arXiv
```

- Papers are chosen by keywords in the title and abstract: biology first, then statistics, then AI and
  mathematics. The text comes from the LaTeX source (equations kept); papers without usable source come from
  their PDF.
- `--wait 5-60` (default): a random pause of 5 to 60 s before each paper, on top of arXiv's 3 s minimum; about 110
  papers an hour. `--wait 0` keeps only the 3 s minimum (faster, but "too many requests" answers become more
  likely). After a "too many requests" answer (429 or 503) every request waits for the time arXiv asks for, or 1,
  2, 4 ... up to 10 minutes; no paper is skipped because of it.
- A PDF conversion is stopped after 180 s (`--pdf-seconds`) and the paper is marked `pdf_timeout`. A later run with
  a larger limit, for example `--pdf-seconds 600`, tries those papers again.
- `%USERPROFILE%`, `$HOME` and `~` in `--out` work in PowerShell too (also in `download_pmc.py` and
  `youtube_transcripts.py`).
- 21 subject areas by default: q-bio, physics.bio-ph and physics.med-ph (biology); stat, econ.EM and physics.data-an
  (statistics); cs.AI, LG, CL, CV, NE, MA, IR, CE, DS, DC, MS, CR, CY and eess.IV (AI and computing); math.
  `--sets` chooses others (a new area also needs a group in `GROUPS`).

**bioRxiv and medRxiv preprints** (most statistical genetics and genetic epidemiology preprints are here)

```
python download_rxiv.py --server biorxiv            # first run: harvests the metadata (hours), then downloads
python download_rxiv.py --server medrxiv
python download_rxiv.py --server biorxiv --update   # later: add new preprints
```

- Same rules as arXiv: keywords in the title and abstract (3 matches from 3 terms, the title counts 3 times),
  best matches first, `--list-only`, `--rebuild-list`, `report.tsv` with each paper's licence (bioRxiv licences
  vary from CC BY to "all rights reserved").
- Text from the server's JATS XML (sections, equations, figure captions; references left out), else Europe PMC's
  full text, else the PDF. Output in `ai_training_data\biorxiv` and `ai_training_data\medrxiv`.
- `--wait 5-15` (default) before each paper and at least 1 s between requests; a bot-check page stops the run after
  5 papers in a row so it never hammers the site.
- What happened to each paper, with its licence: `ai_training_data\_biorxiv_meta\report.tsv` (and
  `_medrxiv_meta`). Stop with Ctrl+C and rerun the same command to continue.

**Software documentation** (manuals, tutorials and worked examples of the tools the model should learn to use)

```
python download_docs.py --list                           # the sources in doc_sources.tsv and their status
python download_docs.py                                  # download what is not done yet; a rerun continues
python download_docs.py --only plink2,regenie --refresh  # fetch some tools again
```

- `doc_sources.tsv` lists about 1,200 sources for about 920 tools (columns: tool, kind, url, licence, notes). Kinds:
  GitHub repositories (README, docs, vignettes, examples), documentation sites (crawled under the URL's folder only,
  robots.txt obeyed, at most `--max-pages 400` pages), Read the Docs projects, PDF manuals, single pages, CRAN and
  Bioconductor packages. Add a row to add a tool.
- Output: `ai_training_data\software_doc\<tool>\...` as Markdown, one file per page; `data_prep.py` reads it as the
  source `software_doc`. `software_doc\_manifest.tsv` lists each row's status, files, characters and licence; rows
  that failed (`error`, `partial`) are tried again on the next run.
- Pacing: 1-2 s between requests to one host, a pause after "too many requests", at most `--max-mb 300` per row.
  Nothing downloaded is run.
- Some sites refuse scripts: GCTA, SMR and GSMR (yanglab.westlake.edu.cn), LDAK (dougspeed.com) and KING. Those rows
  stay errors; save their pages from the browser and convert them with `convert_to_markdown.py`.

**YouTube lecture transcripts**

Add channels (one per line) to `youtube_channels.txt`, then:

```
python youtube_transcripts.py --channels-file youtube_channels.txt
python youtube_transcripts.py "Human Cell Atlas" --list-only        # check a channel name first
python youtube_transcripts.py --channels-file youtube_channels.txt --whisper   # later, when not training
python youtube_transcripts.py --channels-file youtube_channels.txt --recheck-off-topic   # after editing the keywords
```

Only videos whose transcripts match `keywords.txt` are kept. The `--whisper` run transcribes videos that
have no usable captions (it uses the GPU if nothing else does; on the CPU it uses 4 threads, `--whisper-threads`).
What happened to each video is listed in `ai_training_data\_lectures_meta\<channel>_report.tsv` (kept, off topic,
queued for Whisper...). Videos that failed for a passing reason (a rate limit, "format not available"), and
premieres or live streams that have not aired yet, are tried again on the next run. If YouTube says the session
is rate-limited, the run stops: wait a few hours.

---

## Step 2 (optional): how big will the dataset be?

```
python estimate_dataset.py
```

Prints, for every source, how many documents and tokens `data_prep.py` would keep, then a table of model sizes
with the exact training command for each. It builds nothing.

---

## Step 3: build the dataset

```
$D = "$HOME\ai_training_data"
python data_prep.py --out datasets\bio_all --text-root $D `
    --wiki-dir $D\wikipedia --wiki-max-docs 0 `
    --parquet "$D\web\fineweb-edu\sample\10BT\*.parquet" --parquet-name fineweb --min-score 3 `
    --jsonl "$D\web\pes2o\data\v2\*.json.gz" --jsonl-name pes2o `
    --include-keywords keywords.txt --keyword-min-distinct 3 `
    --tokenizer-weights books=3,articles=3 --near-dup drop
```

What the options mean:

| Option | Meaning |
|---|---|
| `--out datasets\bio_all` | The new dataset folder (must not exist yet). |
| `--text-root $D` | Read every sub-folder of ai_training_data as a source. |
| `--wiki-dir ... --wiki-max-docs 0` | Keep every Wikipedia article that matches the keywords (a number keeps only that many, best matches first). |
| `--parquet ... --min-score 3` | FineWeb-Edu web pages with an education score of 3 or more. |
| `--jsonl ... --jsonl-name pes2o` | Scientific papers from peS2o. |
| `--include-keywords keywords.txt` | Wikipedia, FineWeb and peS2o documents must match these terms (your own folders are not filtered). |
| `--keyword-min-distinct 3` | ...with at least 3 different terms. |
| `--tokenizer-weights books=3,articles=3` | Let books and articles shape the vocabulary more. The tokenizer's sample is drawn in 4,000-character windows from every source, so a source of a few long documents (books) is never left out. |
| `--superbpe` | A SuperBPE tokenizer, whose tokens can span words ("of the"): about 12% fewer tokens at 16,384 pieces. `bio_all` and the `tinyGPT_muon` run use SuperBPE (16,384 pieces), and `SUPERBPE = True` in run_pipeline.py. The only measurement against ordinary BPE (20.5M parameters, 2026-09-30) was 3.1% worse in bits per byte and has not been repeated at this size; see the guide. |
| `--near-dup drop` | Keep only one copy of near-identical documents (for example the 5 copies of *The Germ-Plasm*). |
| `--near-dup-max-docs N` | Optional guard: stop if more than N documents would be searched for near-duplicates (default 0 = no limit; the search works at tens of millions of documents). |
| `--work-dir D:\tinygpt_work` | Optional: put the compressed text spool on another drive (a build needs about 0.4x the corpus there; at 3-5x the current corpus 40-70 GB). |
| `--tokenizer-from tinyGPT_muon_best.pt` | Build for an EXISTING model: reuse its tokenizer. Without it the automatic vocabulary rule picks a size for the text (32,768 for the current corpus, while the running model uses 16,384), and a vocabulary change needs a fresh model. A rebuild meant for an existing model must pass `--tokenizer-from` and `--keep-split-from datasets\bio_all` (its train/validation split); `run_pipeline.py continue` passes both. |
| `--threads 8` | Optional: use all 8 CPU threads for tokenising. |
| `--scan-only` | Add this to count and filter without tokenising (fast check of the settings). |
| `--resume` | A build that failed after pass 2 (disk full, crash) keeps its work files; `python data_prep.py --out datasets\bio_all --resume` redoes only the tokenizer and the tokenisation, not the hours of reading. |

The result: `datasets\bio_all\report.md` explains, per source, what was kept, removed and why, and how the tokenizer's
pieces are used on the training data (a chart, `token_summary.png`; every piece with its count is in `token_counts.tsv`). Lines repeated in
20 or more documents (licence notices, navigation) are removed as boilerplate, but headings and display maths never
are. Validation documents whose text largely appears in training (an abstract and its paper, a lightly edited copy)
are moved to training before the tokenizer is trained and listed in `leakage.tsv`; one text under 5 or more file
names is listed as a possible download placeholder.

`python run_pipeline.py build` runs this command with `--superbpe` added (the books and articles weights only for
folders that exist). A rebuild of `datasets\bio_all` takes about 3 hours and gives a dataset the running model cannot
use unless it passes `--tokenizer-from tinyGPT_muon_best.pt --keep-split-from datasets\bio_all`.

- A folder whose name is not a valid source name (for example `lecture notes`, with a space) is skipped with a
  WARNING on screen and in `report.md`: rename it (`lecture_notes`).
- While a build runs, `datasets\bio_all\_build.lock` stops a second build into the same folder.
- `manifest.json` is written only when the dataset is complete; a folder without it is not a finished dataset.

---

## Step 4: train

### How the model size is chosen from your data

The planner in `tiny_gpt.py` uses two rules of thumb from published research:

1. **Do not reread the same text more than 4 times.** `datasets\bio_all` has 4.08 billion training tokens, so
   training can use at most 4 x 4.08 = 16.3 billion tokens.
2. **A model needs about 20 tokens of training text per parameter** to be properly trained (a model with too many
   parameters for its text is under-trained and wastes its size). So 16.3 billion tokens can "feed" at most
   16.3 / 20 = 0.82 billion parameters (817M).

It then takes the largest model on a fixed list of sizes that is below that number and fits in the GPU's memory
(parameters for the 16,384-piece vocabulary of `bio_all`; memory as the planner estimates it for Muon, the least it
can use):

| Parameters | Width (`--d-model`) | Layers | Context | Fits in 8 GB? |
|---:|---:|---:|---:|---|
| 20.5M | 384 | 8 | 512 | yes |
| 29.3M | 448 | 9 | 512 | yes |
| 36.6M | 512 | 10 | 512 | yes |
| 70.0M | 640 | 12 | 1,024 | yes |
| 99.3M | 768 | 14 | 1,024 | yes |
| 170.7M | 896 | 16 | 1,024 | yes |
| 216.2M | 1024 | 18 | 1,024 | yes: 5.0 GiB at micro-batch 2 without checkpointing |
| 457.2M | 1280 | 22 | 2,048 | no: at least 7.1 GiB |
| 669.3M | 1536 | 26 | 2,048 | no: at least 9.9 GiB (with checkpointing, micro-batch 1) |

The data's choice is 669M (the largest below 817M), but it needs about 10 GiB even with every memory saving, and
457M does not fit either (the planner's budget is 80 % of the free GPU memory, about 5.6-6.3 GiB on this card). So
the plan is the **216M model on 16.3 billion tokens: at the ~5,800 tokens per second this laptop sustains for it,
about 33 days**. That is the run now going (`tinyGPT_muon`: 60,000 steps of 270,336 tokens = 16.2 billion tokens).
It is the best model *for the data and this GPU*, but it ignores how long you are willing to wait.

**With a time limit**, the choice changes: in a fixed number of hours, a bigger model reads fewer tokens (each
token costs more computation). The planner takes the largest model that can still read 20 tokens per parameter
in the time, and gives a smaller model more text rather than a bigger one too little.

The planner looks only at *how much* text there is, never at what it says.

### Choose a run

**Simplest:** give the time and let the planner choose (this is what `run_pipeline.py` does):

```
python tiny_gpt.py --dataset datasets\bio_all --time-budget-hours 48 --plan    # look first
python tiny_gpt.py --dataset datasets\bio_all --time-budget-hours 48           # then train
```

It times each model size briefly on the GPU, and uses the speed your last training run actually sustained for
hours (a cool GPU runs faster for a few seconds than it does all day; before the first run it allows for that).

**To see what N hours buy,** run `python tiny_gpt.py --dataset datasets\bio_all --time-budget-hours N --plan` while
the GPU is idle (it times each size on the GPU) and read the model, steps and tokens it prints. That is always up to
date; a fixed table of runs goes stale with every new dataset and every change of speed (the one that was here was
measured on the September dataset). **Or pick the model yourself:** `--d-model` and `--layers` from the table
above, and `--train-tokens` for how much it reads.

- More time gives a better model, but each extra day helps less. By the loss formula of the 2022 "Chinchilla"
  study (which predicts final loss from model size and training text), going from 12 h to 24 h improves the
  model about as much as 24 h to 48 h, and the third day adds half as much again.
- **Always preview first:** add `--plan` to any command to print the model, batch, steps and ratios without
  training.
- `--train-tokens` sets how much is trained. **Do not use `--steps` for this**: `--steps` on its own only splits
  the data's full budget (4 passes, 16.3 billion tokens) into that many steps, so it still trains for weeks.

### The other planner rules (what `--plan` prints)

| Setting | Rule | Override |
|---|---|---|
| Context | By width: 256 up to d 192, 512 up to d 512, 1,024 up to d 1,024, 2,048 up to d 2,048, 4,096 above. The plan prints attention's share of the training FLOPs and a note when a large source's median document is longer than the context | `--ctx` |
| RoPE base | 10,000 up to ctx 2,048, then 10,000 x (ctx / 2,048)^2 (40,000 at 4,096) | `--rope-base` |
| Tokens per step | Half of DeepSeek's fitted optimum batch, 0.5 x 0.292 x C^0.327 with C = 6 N D (N parameters, D training tokens), rounded to a power of two between 16,384 and 2,097,152: 262,144 at 216M / 16.2B, 524,288 at 1.45B / 29B, 1,048,576 at 3.2B / 64B | `--tokens-per-step`, or `--batch-size` and `--grad-accum` |
| Peak learning rate | A log-log table: 4e-3 at 1M parameters, 2.4e-3 at 10M, 1.2e-3 at 100M, 6e-4 at 1B (twice the rates of published models, from the 5 November sweep; 9.5e-4 at 216M), times a batch factor: the square root of the planned batch over the old 16 sqrt(N) batch, rounded to a power of two first and kept within 0.7-1.4 | `--lr`, or `--lr-scale` (a factor, for sweeps) |
| Weight decay | 0.05 on the matrices (half the published 0.1, so the shrink per step, lr x weight decay, stays where the published rates put it). The plan prints the decay timescale 1/(lr x weight decay) in steps and its share of the run | `--weight-decay` |
| Schedule | Warmup-stable-decay; warm-up 2 % of the steps (50-2,000); the last 20 % decay as 1 - sqrt to 0 (cosine keeps a tenth of the peak) | `--warmup-steps`, `--decay-frac`, `--min-lr-ratio`, `--schedule` |
| Dropout | 0 up to 4 passes over the data, else 0.1 | `--dropout` |
| Validation | Fixed windows of about 0.5 % of the tokens trained between evaluations, at least 65,536 tokens per source | `--eval-tokens` |
| Architecture | QK-norm on; tied embeddings; GQA with 4 query heads per key/value head from 8 heads on; dense SwiGLU (no experts); no weight average; no document mask | `--qk-norm off`, `--tie-embeddings off`, `--kv-heads`, `--experts N`, `--ema on`, `--doc-mask on` |

A resumed run never re-plans: it keeps the learning rate, weight decay, schedule, architecture and objective
recorded in its checkpoint (the options it may change are listed under "Resuming the 216M run" below).

### Memory: does the model fit on the GPU?

The plan prints a `memory:` line. The planner estimates the GPU memory each model needs and, only when the model
would not fit otherwise, switches on one or both of these savings. Neither changes what the model learns:

- **The loss in pieces** (`--loss-chunk-tokens`): the output layer and the loss are computed 4,096 tokens at a
  time, so the large table of scores (every token against every vocabulary entry) never exists in full. Measured
  cost: 1-8 % of step time, nothing measurable when the model is compiled.
- **Activation checkpointing** (`--activation-checkpointing`): the values inside each block are recomputed in the
  backward pass instead of being kept. Much less memory; measured cost 21-32 % of step time.

The planner takes the cheapest combination that reaches a micro-batch of about 2,048 tokens (2 sequences at ctx
1,024, 1 at ctx 2,048 and above), and otherwise the cheapest that fits at all. A bigger micro-batch is never worth
checkpointing: at the 216M shape a micro-batch of 4 is only about 7 % faster than 2 when compiled. The estimate
counts the weights, gradients, the half-precision copies autocast keeps, the optimizer state, the activations and the
logits, and is calibrated to the measured peak (216M at micro-batch 2 without checkpointing: estimated 5.0 GiB,
4.88 GiB allocated). Every metrics line and the experiments table record the real peak as `peak_mem_gib`.

If the first step still runs out of memory, the run tries, in order: the loss in pieces, a smaller micro-batch
(down to about 2,048 tokens), activation checkpointing, then smaller micro-batches down to 1, always on the same
windows and with the same tokens per step. Later in a run, an out-of-memory error replays the same step (at most
twice) instead of ending the run. It never switches on a saving you switched off. Set them yourself only to force a
choice: `--activation-checkpointing on` or `off`, `--loss-chunk-tokens 0` (off) or a number. A resumed run keeps
its settings unless you give these options.

### AdamW or Muon

`--optimizer muon` trains the weight matrices inside the blocks with Muon and the rest (embedding, norm gains) with
AdamW; one learning rate and schedule drive both. It needs PyTorch 2.9 or newer, applies to new runs only (a resumed
run keeps its optimizer), and needs about half of AdamW's optimizer memory for those matrices. The default of
`tiny_gpt.py` is AdamW; `run_pipeline.py` uses Muon (`OPTIMIZER = "muon"`), as the `tinyGPT_muon` run does. A short
AdamW-against-Muon comparison on this data is still to be run (TODO.md); compare two runs with `compare_models.py`.

### Is a bigger or deeper model better?

Only if it also gets enough text. For a given number of parameters, research found that the exact split between
width and depth matters little; the list above grows both together. Adding layers alone makes the model slower
and, without more text, not better.

### During training

- The first minute or so of a run is spent compiling the model for the GPU (`torch.compile`); after that it trains
  about 1.45 times faster. The header line says `compile: on`.
- Every 200 steps a **metrics line**: training loss, validation loss per source, bits per byte, speed, energy used,
  CO2 and electricity cost, time left. `NEW BEST` means the model improved; `<name>_best.pt` is saved then
  (`tinyGPT_muon_best.pt` for the current run).
- **`val(mix)`** is the validation loss with the sources weighted as training samples them. Compare it with the
  training loss: the plain `val` weights every source equally, so its gap to the training loss mixes overfitting
  with the difference between the mixtures.
- **A step whose gradient is not finite** (inf or NaN) is skipped and reported (`update skipped (1/20)`); 20 in a
  row end the run without overwriting the last good checkpoint. A save that fails (disk full, file locked) is
  retried a minute later while training goes on.
- **Pause:** press Ctrl+C. **Continue:** `python tiny_gpt.py --name tinyGPT_muon --resume` (the run's `--name`;
  without it the name is `tinyGPT`). A step is either applied in full or not at all, and the resumed run continues
  on the same windows, schedule and optimizer state (GPU kernels are not bit-deterministic, so a replayed step
  differs at about 1e-5 in the loss).
- **Dashboard:** after each `NEW BEST` the dashboard is redrawn in the background on the CPU (log:
  `logs\dashboard.log`) and pushed to GitHub (`dashboards\`) at most every 3 hours (`--dashboard-push-hours`;
  `--dashboard off` switches it off).
- **Overfitting** (memorising text) would show as validation loss rising while training loss keeps falling. The
  planner caps reading at 4 passes over the data, where repeated text is still almost as good as new (the 216M run
  reads it about 4 times by its end); watch `val(mix)` against the training loss, and the best checkpoint is kept
  anyway.
- A second run with a different name: add `--name tinyGPT_70M`. Reusing a name needs `--overwrite` (old files are
  renamed, not deleted).

### Resuming the 216M run (tinyGPT_muon) with the new code

`python tiny_gpt.py --name tinyGPT_muon --resume` (or `python run_pipeline.py resume`) continues the run exactly as
recorded in `tinyGPT_muon.pt`: 4 x 66 micro-batches with activation checkpointing, learning rate 4.76e-4, weight
decay 0.1, decay to 4.76e-5, no QK-norm, no weight average. The new defaults of the planner (learning-rate table,
QK-norm, weight decay 0.05) apply to new runs only. These optional flags change the run; each is written into
`tinyGPT_muon.pt` at the next save and kept from then on:

| Flag | What it changes |
|---|---|
| `--batch-size 2 --grad-accum 132 --activation-checkpointing off` | The same 270,336 tokens per step and the same mathematics, without the recomputation: 5-16 % faster per step (measured 5 November, see below), peak memory 4.9 GiB instead of 4.2. The random windows drawn after the resume differ from an uninterrupted run (same mixture). If step 1 ran out of memory anyway, the run would try the loss in pieces and then micro-batch 1 x 264, never checkpointing (the command line switched it off): resume again without these three flags. |
| `--min-lr-ratio 0` | The decay phase (steps 48,001-60,000) ends at learning rate 0 instead of 4.76e-5. Nothing changes before step 48,001; expected 0.005-0.03 nats lower at the end. Ignored, with a note, once the run is inside the decay. |
| `--ema 0.999` | Keeps a debiased moving average of the weights (on the CPU) and scores it beside the raw weights at each evaluation while the learning rate is above 5 % of its peak; the better of the two becomes `tinyGPT_muon_best.pt` (its record says which). Costs about 0.9 GB of RAM, 1.7 GB while the average is scored, a second validation pass per evaluation and 0.9 GB more in `tinyGPT_muon.pt`. Expected to give a better `_best.pt` in the stable phase; not yet measured on a long run. `--ema off` removes it again. |

Options that a resume ignores (with a note): the shape, `--lr`, `--lr-scale`, `--qk-norm`, `--doc-mask`,
`--z-loss`, `--eval-tokens` and the optimizer.

### Measuring speed on this laptop, and the 5 November A/B results

Speeds measured on this laptop are comparable only within one session: the GPU switches between about 85 W (boost,
1.2-1.4 GHz) and its 55 W cap (0.3-0.8 GHz) during a day, so the same step can take 1.5 times longer an hour later.
Compare two settings by alternating them in one session, or by time x clock. The 55 W power profile is your
deliberate setting; raising it would be the largest single speed-up (about 1.2-1.4x), and it is your choice.

Three short GPU experiments were run on 5 November while the 216M run was paused (all numbers in
`results\ab_2026-11-05.md`; differences below 0.02 bits per byte were treated as noise):

| Experiment | Result | What changed |
|---|---|---|
| Speed: 216M as 4 x 66 with checkpointing against 2 x 132 without | 2 x 132 is 5-16 % faster per step (49.0 against 41.0 s throttled, 28.6 against 25.7 s at boost; the rest of the gap is clock), the same loss, peak 4.88 against 4.20 GiB, as the planner estimated (5.0 GiB) | Nothing: the planner already chooses 2 x 132 for new runs; for the running run it is the first resume flag above |
| QK-norm (20.5M, 500 steps) | 0.106 bits per byte better | On for new runs |
| Learning rate (20.5M and 36.6M) | 2x the old table beat 1x by 0.060; 4x beat 2x at both sizes (0.019 and 0.047) | The table is doubled (a 500-step optimum overstates the rate of a 60,000-step run by about 1.7x), weight decay 0.1 -> 0.05 |
| Weight average (EMA) | Never better in 500 steps; 1.1 bits per byte worse at steps 300-400, because the average still held the initial weights | Off by default (`--ema on` to opt in); the average is now debiased |
| Document mask (ctx 512) | -0.0016 bits per byte (noise) for about 8 % more forward and backward time | Stays opt-in (`--doc-mask on`) |
| Mixture of experts (99M active / 298M total, 50 steps) | Memory as estimated (5.02 against 5.3 GiB); 2.2-2.3x slower per step than dense; 0.26 nats behind at step 50 | Dense stays the default; no MoE run on this laptop |

### Energy and cost

Shown in every metrics line. Default price: Flexible Octopus, region H (change it with `--region`). If you are on another Octopus
tariff, add for example `--tariff AGILE-24-10-01` (Agile) or `--tariff GO-VAR-22-10-14` (Go), or a fixed price
`--tariff 24.5` (pence per kWh).

---

## Later: more data, and training further from the first model

After adding more text to `ai_training_data`, the model does not have to start from zero:

```
python run_pipeline.py continue --hours 24
```

This does two things:

1. **Builds a new dataset, `datasets\bio_all_v2`, from all of `ai_training_data`** (old and new text) with the
   *same tokenizer* as the first model (`data_prep.py --tokenizer-from tinyGPT_muon_best.pt`). The same tokenizer is
   essential: the model's weights belong to its vocabulary, and a new tokenizer would give every token a different
   meaning. Keeping the old text in stops the model from forgetting it while it learns the new, and the old
   train/validation split is kept too (`--keep-split-from datasets\bio_all`): a document the first model trained on
   never becomes a validation document, where its loss would look better than the model really is. This holds even
   when the file was renamed or moved, or when a near-copy of it is the one kept.
2. **Trains further from the first model's weights**
   (`tiny_gpt.py --dataset datasets\bio_all_v2 --init-from tinyGPT_muon_best.pt --name tinyGPT_muon_continued
   --time-budget-hours 24 --optimizer muon`): same model (its exact architecture, read from the checkpoint), a fresh
   learning-rate schedule, and as much of the new dataset as fits in the hours given. Without `--steps`, the planned
   steps come on top of the step the first model reached.

Why not LoRA? LoRA freezes the model and trains small add-on matrices. It saves memory when the model has billions
of parameters (it is the plan for fine-tuning large open models on the Mac Studio), but it limits how much new
knowledge a model can take in. tinyGPT is small enough to keep training all of its weights, which learns more.

Reusing the same text is fine and intended. What matters is how often each text is read in total, over both
runs: keep it to about 4 passes. For example, if the first run read the old text 0.4 times and the second reads the
combined data 0.8 times, the old text has been read 1.2 times. Duplicated documents count several times, which is
why `--near-dup drop` matters.

Two limits: the model size stays that of the first run (a bigger model starts from zero), and a tokenizer from the
first data splits new-field words into more pieces (slightly less efficient, still correct).

## Step 5: look at the results

```
python view_pt.py tinyGPT_muon_best.pt --plot                # dashboard picture: tinyGPT_muon_best_dashboard.png
python view_pt.py tinyGPT_muon_best.pt --plot --dpi 300      # the same, sharper (7,800 x 6,900 px)
python view_pt.py tinyGPT_muon_best.pt --plot --pca-labels 50 --sim-tokens 100
                                                             # how many tokens the two embedding panels show
                                                             # (these are the defaults)
python tiny_gpt.py --name tinyGPT_muon --benchmark           # 336 biology fact questions, per category
python compare_models.py A_best.pt B_best.pt --device cpu    # is A really better than B?
```

- The dashboard draws the fact benchmark from the results training stored at that step, so it takes about 30 s.
  For a fine-tuned file it scores the fine-tuned weights itself, and labels the base model's numbers as such.
- `view_pt.py --device auto` uses the GPU (or Apple's MPS on a Mac) for the panels that run the model; the default
  is the CPU.
- `compare_models.py` leaves out validation documents that the other model trained on, when the two were trained
  on different datasets. If it cannot check this, it prints a WARNING and gives no verdict, unless you add
  `--allow-different-datasets`.

`results\experiments.csv` (opens in Excel) has one row per training run, benchmark, comparison and fine-tune, with the full
recipe: model shape, batch, optimizer, learning-rate schedule, weight decay, clipping, dropout, precision and the
memory settings. When a new version adds columns, the old file is kept as a backup and rewritten once with the
new header. It is the file pushed to GitHub, so paths show your home folder as `~`.

### Public benchmarks: benchmark.py

```
python benchmark.py download                                  # ~180 MB into benchmarks\ (once; --only mmlu,medqa)
python benchmark.py sandbox-setup                             # numpy for the code sandbox in WSL (once)
python benchmark.py run --checkpoint tinyGPT_muon_best.pt --limit 200 --device cpu
python benchmark.py run --checkpoint tinyGPT_muon_best.pt --benchmarks mmlu,medqa,humaneval
python benchmark.py plot                                      # PNGs in benchmarks\plots\
```

- `download` fetches 21 open benchmarks (MMLU, MMLU-Pro, MedQA, MedMCQA, PubMedQA, MedXpertQA, SciQ, ARC,
  LAB-Bench, MedCalc-Bench, HumanEval(+), MBPP(+), DS-1000 and others) and writes `benchmarks\manifest.json` with
  each source, revision, licence and checksum. Nothing gated is downloaded. Never train on these files.
- `run` scores multiple-choice sets like the fact benchmark (log-probability of each option per character), with
  chance and the shuffled-question floor; MedCalc by the first number of a greedy answer; coding sets by running
  the tests of a greedy completion (pass@1). `--limit N` takes the same N items every time; leave it out for the
  full sets (best on the GPU). Results go to `results\benchmarks.csv`, per-item CSVs to `benchmarks\runs\`, and one
  row to `experiments.csv` (`--no-publish`: not pushed to GitHub).
- Generated code runs only in a sandbox without network: Docker if Docker Desktop is running, otherwise WSL
  Ubuntu with bubblewrap (no Windows drives visible). Without either, coding benchmarks are refused.
  `sandbox-setup --datasci` adds pandas, scipy, scikit-learn and matplotlib for DS-1000.
- Open-answer and picture-graded sets (GeneTuring, HealthBench, CADBench, BioCoder, BigCodeBench) are downloaded
  but not scored; `run` says why.

## Step 6: use the model

```
python tiny_gpt.py --name tinyGPT_muon --generate "Genetic drift is" --length 100
python tiny_gpt.py --name tinyGPT_muon --export exported_model   # safetensors + tokenizer, for other tools
python finetune.py sft --base tinyGPT_muon_best.pt --data finetune_examples\sft_examples.jsonl --name tinyGPT_sft
python finetune.py ask --checkpoint tinyGPT_sft.pt "What does genetic drift do to small populations?"
python detect_text.py some_text.txt --checkpoint tinyGPT_muon_best.pt
python posttrain.py data                                     # question data from open datasets (once)
python posttrain.py run --base tinyGPT_muon_best.pt --name tinyGPT_muon   # SFT, then DPO, then evaluation of all three
python tools\edit_weights.py tinyGPT_muon_best.pt tinyGPT_muon_best_noloop.pt   # remove repetition loops in a copy
python tools\compare_edit.py tinyGPT_muon_best.pt tinyGPT_muon_best_noloop.pt   # did the edit help? (picture + report)
```

- Text is generated with a key/value cache: each new token runs the model on that token only, not on the whole
  text again (the same output, many times faster for long texts). The training samples, `finetune.py ask` and
  `benchmark.py` use it too.
- `--checkpoint FILE` picks any checkpoint instead of `<name>_best.pt`.

- `finetune.py --batch 8` is the number of examples per update. `--micro-batch N` processes N at a time and adds
  up their gradients (same result, less memory). An out-of-memory error halves it by itself, and at 1 switches on
  `--activation-checkpointing`.
- Batches hold examples of similar length, so little time goes into padding; `--no-length-grouping` gives plain
  shuffled batches.
- `finetune.py` and `posttrain.py eval` refuse the GPU while another job uses it: use `--device cpu`, or `--force`.
- `posttrain.py` writes `posttrain_data\posttrain_report.md`: multiple-choice accuracy on held-out questions
  against a shuffled-question floor, the fact benchmark, and example answers.

---

## Options of tiny_gpt.py you are likely to need

| Option | What it does | Default |
|---|---|---|
| `--dataset FOLDER` | Dataset built by data_prep.py | required for training |
| `--plan` | Show the plan and stop | |
| `--train-tokens N` | How many tokens to train on | 4 passes over the data |
| `--d-model N --layers N` | Model width and depth (pick a pair from the table) | chosen from the data |
| `--time-budget-hours H` | Let the planner fit the run into H hours | off |
| `--name NAME` | Name of the checkpoint files (the current run: `tinyGPT_muon`) | tinyGPT |
| `--resume` | Continue a paused run | |
| `--overwrite` | Start again under an existing name (old files renamed) | |
| `--eval-every N` | Steps between metrics lines | 200 |
| `--save-minutes M` | Save the training state every M minutes | 30 |
| `--stop-after-steps N` | End this session cleanly after N more steps (continue with `--resume`) | |
| `--optimizer adamw` or `muon` | Optimizer of a new run | adamw |
| `--batch-size B --grad-accum A` | Micro-batch and accumulation; on `--resume` they may change if B x A x ctx stays the same | planned |
| `--activation-checkpointing auto`, `on` or `off` | Recompute blocks in the backward pass to save memory | auto |
| `--loss-chunk-tokens N` | Compute the loss N tokens at a time (0 = off) | auto |
| `--min-lr-ratio R` | Learning rate at the end as a share of the peak; on `--resume` until the decay starts | 0 (wsd), 0.1 (cosine) |
| `--lr-scale F` | Multiply the planner's learning rate (sweeps) | 1 |
| `--ema on`, `off` or a decay | Debiased moving average of the weights, scored beside them; the better is saved as best | off |
| `--qk-norm on` or `off` | Normalise queries and keys in attention (new runs) | on |
| `--doc-mask on` or `off` | Attention stays within each document of a window (new runs) | off |
| `--experts N` | Mixture of experts with N experts per block (new runs; `--experts-per-token`, `--moe-every`) | 0 (dense) |
| `--best-bf16` | Store `<name>_best.pt` in bfloat16 (half the file; for models of about 1B and more) | off |
| `--dashboard auto`, `on` or `off` | Redraw the dashboard after each new best | auto |
| `--dashboard-push-hours H` | Push the dashboard to GitHub at most every H hours (0 = every time) | 3 |
| `--mix wiki=1,notes=3` | Read some sources more often than their size | by size |
| `--mix-alpha A` | Sampling share ~ weight x tokens^A (below 1 flattens a mix one source dominates) | 1 |
| `--device cpu` | Run on the CPU (for anything while the GPU trains) | auto |
| `--tariff`, `--region` | Electricity price for the cost figure | Flexible Octopus, H |
| `--compile auto`, `on` or `off` | torch.compile (auto: on CUDA when Triton is installed) | auto |
| `--generate TEXT --length N` | Write text from the best checkpoint | |
| `--benchmark` | Score the fact benchmark | |

`python tiny_gpt.py --help` lists every option.

## When something goes wrong

| Message | What to do |
|---|---|
| `... already exists` | Use a new `--name`, or add `--overwrite`. |
| `Out of memory at the first step; retrying ...` | Nothing: it tries the memory savings and smaller batches by itself. |
| `Out of memory at the first step even at micro-batch 1` | Choose a smaller `--d-model`, `--layers` or `--ctx`, or drop the `--loss-chunk-tokens 0` / `--activation-checkpointing off` the message names. |
| `... is a newer training state than tinyGPT.pt` | A save went to `tinyGPT.HHMMSS.pt` because `tinyGPT.pt` was locked by another program. Rename it to `tinyGPT.pt` and resume again. |
| `... is being trained by another process` | A run with that name is already training. Wait for it or stop it; delete `<name>.train.lock` only if nothing is training. |
| `The GPU is busy` | Another job uses the GPU. Wait, or use `--device cpu`. |
| `--precision fp16` refused on a Mac | This PyTorch cannot scale FP16 gradients on MPS: use `--precision bf16` or `fp32`. |
| `MPS backend out of memory` | The run is capped at Metal's recommended working set on purpose (no swapping to the SSD). At step 1 the memory savings are tried by themselves; otherwise choose a smaller model or `--ctx`. |
| `non-finite gradient norm ...; update skipped` | Nothing, if it is rare: the update was dropped and training goes on. 20 in a row stop the run with the last good checkpoint kept. |
| `Out of memory at step N; retrying the same step` | Nothing: another program took GPU memory; the step is replayed (twice at most). |
| `training loss became nan` | Restart from the last checkpoint with a lower `--lr` (see the message). |
| The screen goes black / driver reset | Two GPU jobs ran at once. Resume with `--resume`; keep other jobs on `--device cpu`. |
| `manifest.json missing` | The dataset folder was deleted or is incomplete: rebuild it (step 3). |
| `Nothing to resume` / `No checkpoint ...tinyGPT_best.pt` | The default name is `tinyGPT`: add `--name tinyGPT_muon` (or the run's name), or `--checkpoint FILE`. |

---

## Mac Studio (MPS)

`tiny_gpt.py` runs on Apple silicon without code changes (PyTorch 2.9 or newer, which Muon needs). What is
different there, and what to check on the first day:

**First-run checklist** (in a Terminal, in the project folder, nothing else running):

1. `python gpu_check.py`: the chip, its GPU core count, Metal's recommended working set and the macOS version.
2. `python gpu_check.py --train-bench --device mps --d-model 1024 --layers 18 --heads 16 --kv-heads 4 --ctx 1024
   --batch 4`: a kernel benchmark at the 216M shape (tokens per second and memory; not used by the planner).
3. `python tiny_gpt.py --dataset datasets/bio_all --plan`: the budget line, the model, the memory estimate. The
   first plan on MPS also runs the attention-memory probe: if the plan's estimate includes the attention scores, the
   MPS attention kernel keeps them for the backward pass (the composite path), and long contexts prefer activation
   checkpointing.
4. One short run twice, `--compile on` and `--compile off` (for example `--name mps_test --steps 200
   --stop-after-steps 50`), and compare tok/s. torch.compile on MPS is a prototype: auto mode leaves it off, and a
   graph that fails falls back to eager. If compiled is 10 % or more faster, keep `--compile on` for Mac runs.
5. Compare `peak_mem_gib` in the metrics with the plan's estimate. It is read from `torch.mps.driver_allocated_memory`
   (an upper bound that includes the allocator's cache).

**Memory rules on MPS:**

- The planner's budget is 75 % of Metal's recommended working set (which is itself about 75 % of the unified
  memory): on the 96 GB Mac Studio about 72 GB recommended, about 54 GiB budget. It is lower than CUDA's 80 % because
  the same memory also holds the memory-mapped token files and macOS.
- The run caps itself at the recommended working set (`torch.mps.set_per_process_memory_fraction(1.0)`), so an
  over-budget plan fails fast with "MPS backend out of memory", which the first-step handler and the planner catch,
  instead of paging to the SSD for hours.
- At that budget (Muon, vocabulary 16,384, micro-batch 1, before any attention-score term): the 1.45B ladder shape
  (d 2,048, 32 layers, ctx 2,048) fits without checkpointing at about 25 GiB; 2.25B (d 2,560, ctx 4,096) needs
  checkpointing, about 32 GiB; 3.2B (d 3,072, ctx 4,096) with checkpointing about 45 GiB.
- `sudo sysctl iogpu.wired_limit_mb=88000` raises the GPU's share of the 96 GB (at most about 88,000 MB; leave the
  rest to macOS). The setting is lost at a restart. Only after a run's `peak_mem_gib` shows the estimate holds.

**Environment variables:** the run header prints `PYTORCH_MPS_HIGH_WATERMARK_RATIO` and
`PYTORCH_MPS_LOW_WATERMARK_RATIO` when they are set (the cap above makes them unnecessary), and warns about
`PYTORCH_ENABLE_MPS_FALLBACK=1`: it runs operations MPS has no kernel for on the CPU, 10-100 times slower, so set
it only to diagnose a missing operation.

**Also different on a Mac:** there is no nvidia-smi, so energy is not measured (the experiments table leaves the
energy columns empty; `sudo powermetrics --samplers cpu_power,gpu_power -i 1000 -n 10` gives a package power to pass
as `--cpu-watts` for an estimate), and only the lock files tell whether another job is training. Time budgets
assume a desktop Mac hardly throttles (a 1.1 allowance instead of the laptop GPU's 1.6) and learn from the Mac's own
runs after the first.
