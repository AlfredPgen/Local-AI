# Local-AI: tinyGPT

A small biology language model trained from scratch on a single laptop GPU (RTX 3070, 8 GB), together with the
whole pipeline around it: collecting text, building an audited dataset, choosing the model size from the data and
the time available, training, evaluation, fine-tuning, and tracking the energy, carbon and cost of every run.

Everything is plain Python and PyTorch: no training framework, no hidden defaults. Each step writes a report of
what it did and why, and the learning guide explains the methods behind it.

- **Run it:** [HOW_TO_RUN.md](HOW_TO_RUN.md) (a step-by-step run sheet), or the short version below.
- **Understand it:** `tinyGPT_learning_guide.docx` (built from `guide/tinyGPT_learning_guide.md`): how language
  models are trained, every design choice here, the mathematics, and how to scale up.

## Quick start

```
python -m pip install -r requirements.txt       # after installing PyTorch for your GPU
python run_pipeline.py estimate                 # optional: how many tokens, which model sizes (~7 min)
python run_pipeline.py build                    # build the dataset from ~/ai_training_data (~2 hours)
python run_pipeline.py plan --hours 48          # which model fits in 48 hours (trains nothing)
python run_pipeline.py train --hours 48         # train it; Ctrl+C pauses
python run_pipeline.py resume                   # continue after a pause or a shutdown
python tiny_gpt.py --generate "Genetic drift is"
```

`run_pipeline.py` prints every command it runs; its settings (data folder, dataset and run names) are at the top.

## What is in the repository

| Script | Purpose |
|---|---|
| `run_pipeline.py` | Build the dataset, plan and train in a few commands |
| `convert_to_markdown.py` | PDF (with OCR for scans), Word, PowerPoint, Excel, HTML, notebooks, code and GWAS summary statistics to Markdown |
| `download_pmc.py` | Open-access papers from PubMed Central, converted from their XML to Markdown with LaTeX equations |
| `download_arxiv.py` | arXiv papers that match the keywords, converted from their LaTeX source (or PDF) to Markdown; `--update` adds new papers |
| `download_wikipedia.py` | English Wikipedia in the format `data_prep.py --wiki-dir` reads (needs about 45 GB free while it runs) |
| `youtube_transcripts.py` | On-topic lecture transcripts from YouTube channels (captions, or Whisper speech recognition) |
| `estimate_dataset.py` | Tokens per source and the model sizes they support, without building anything |
| `data_prep.py` | Cleaning, filtering, deduplication, validation split, tokenizer and token files, with a full report |
| `tiny_gpt.py` | Model, planner, training, evaluation, text generation, fact benchmark, export |
| `view_pt.py` | Dashboard of a checkpoint: losses, calibration, embeddings, attention, weights |
| `compare_models.py` | Is model A really better than B? Paired bootstrap over documents |
| `finetune.py` | Supervised fine-tuning (SFT) and preference tuning (DPO) |
| `posttrain.py` | Post-training from open question datasets: builds the data, runs SFT then DPO, and scores base, SFT and DPO |
| `detect_text.py` | Watermark and likelihood tests for text written by the model |
| `calibrate_probes.py` | Difficulty labels for the fact benchmark, from open reference models |
| `gpu_check.py` | GPU clocks, power limits and a short training benchmark |
| `test_tiny_gpt.py` | Test suite: `python -m unittest test_tiny_gpt.py` (CPU only, a few minutes) |
| `keywords.txt` | 4,100 terms (biology, genetics and gene symbols, proteins, microbes, diseases, chemistry and drugs, nutrition, species, molecular structure and 3D rendering, mathematics, statistics, machine learning, databases, scientists' names) that decide which web, Wikipedia, arXiv and YouTube documents are on topic |
| `probes_biology.tsv` | The 336-question fact benchmark (16 categories, difficulty-labelled) |
| `youtube_channels.txt` | Channels for `youtube_transcripts.py` |
| `finetune_examples/` | Example SFT and DPO data |
| `guide/` | Source and build scripts of the learning guide |

Not included: model weights, datasets, and personal run records (see `.gitignore`). Two things are published
automatically for runs in this folder: the experiments table as `results/experiments.csv` (home folder shown as
`~`) and the latest dashboard of each checkpoint in `dashboards/`.

## Requirements

- Python 3.10 or newer (developed on 3.12), PyTorch 2.6 or newer (tested with 2.10, see Safety).
- For about 1.45 times faster training on an NVIDIA GPU: Triton, used by `torch.compile` (on Windows the
  `triton-windows` package matching your PyTorch version; see `requirements.txt`).
- An NVIDIA GPU for training in reasonable time; Apple silicon (MPS) and CPU also work, more slowly.
- The packages in `requirements.txt`. Optional tools (yt-dlp, faster-whisper, Tesseract OCR, pandoc) are listed
  there with what needs them.

## The data

The pipeline reads `~/ai_training_data`. Each sub-folder of Markdown or text files becomes a **source** with its own
validation loss and sampling weight; web-scale data sits in its own folders:

| Folder | Content | How it gets there |
|---|---|---|
| `books`, `articles`, `slides`, `tables`, `codes` | Your documents | `convert_to_markdown.py --by-type` |
| `notes`, `scripts` | Your own notes and texts | copy Markdown in |
| `pmc` | Open-access PubMed Central papers | `download_pmc.py` |
| `arxiv` | arXiv papers on the keyword topics | `download_arxiv.py` |
| `lectures` | YouTube lecture transcripts | `youtube_transcripts.py` |
| `wikipedia` | English Wikipedia (Hugging Face `save_to_disk`) | `download_wikipedia.py` |
| `web/fineweb-edu`, `web/pes2o` | FineWeb-Edu web pages, peS2o scientific papers | Hugging Face downloads |

Folders starting with `_` hold reports (for example `_pmc_meta`, `_lectures_meta`) and are not read as text.

## How it works

**1. Data preparation (`data_prep.py`).** Unicode and whitespace cleaning; an English-language test; removal of
damaged text; a keyword filter for the web sources (a fast C++ prefilter that never rejects what the exact check
would keep, then the exact check); exact deduplication and near-duplicate detection (one-permutation MinHash with
locality-sensitive hashing); removal of boilerplate lines that repeat across documents (headings are kept); a
train/validation split by document, so no text leaks across; a SentencePiece BPE tokenizer trained on the training
split only (optionally SuperBPE, whose tokens can span words: 12-15% fewer tokens); and a train-versus-validation
leakage audit. Heavy steps run on all CPU cores, with byte-identical
results for any number of worker processes. `report.md` records every decision.

**2. Choosing the model size (`tiny_gpt.py`).** The planner follows two rules of thumb from published scaling
studies: reread text at most 4 times, and train on about 20 tokens per parameter. With a time budget
(`--time-budget-hours`) it picks the largest model that still reaches that ratio in the time, using the speed
this computer sustained in its last run. `--plan` shows the choice and the reasons without training.

**3. The model.** A decoder-only transformer, as in GPT and Llama: rotary position embeddings, RMSNorm, SwiGLU
feed-forward layers, grouped-query attention and tied input/output embeddings. Training uses AdamW (or Muon for
the block matrices, `--optimizer muon`), a warmup-stable-decay learning-rate schedule, z-loss, gradient clipping and
BF16 mixed precision, with exact resumption after Ctrl+C or a shutdown. For models too big for the GPU's memory the
planner switches on two savings that leave the mathematics unchanged: the loss computed a few thousand tokens at a
time, and activation checkpointing (blocks recomputed in the backward pass). An out-of-memory error at the first
step is recovered automatically.

**4. Evaluation.** Validation loss per source on fixed windows, bits per byte (comparable across tokenizers), a
biology fact benchmark scored against a shuffled-question floor, calibration of the model's confidence, a
dashboard (`view_pt.py`) and a paired bootstrap for comparing two models (`compare_models.py`).

**5. Energy, carbon and cost.** Every metrics line shows the electricity used so far, its CO2e and its cost, for
example `0.42 kWh, 52 g CO2e, £0.11`. GPU power is measured with nvidia-smi; CPU and memory are estimated. Carbon
uses a grid intensity (`--grid-intensity`, default: Great Britain's recent average), and cost uses Octopus Energy's
published unit rates for a tariff and region (`--tariff`, `--region`) or a fixed price.

**6. After pre-training.** Supervised fine-tuning and DPO (`finetune.py`, or `posttrain.py` for the whole
round from open datasets), text generation with optional watermarking, detection of the model's own text
(`detect_text.py`) and export to safetensors.

## Data sources and licences

| Source | Licence |
|---|---|
| FineWeb-Edu, peS2o | ODC-By |
| Wikipedia | CC BY-SA |
| PubMed Central open-access papers | Per article (mostly CC BY; some CC BY-NC or NIH author manuscripts); each paper's licence is recorded in `_pmc_meta/report.tsv` |
| arXiv papers | Mostly arXiv's own licence (reading, not reuse); some Creative Commons; each paper's licence is in `_arxiv_meta/report.tsv` |
| Post-training data | Per dataset (MIT, Apache-2.0, CC BY-NC 3.0 ...); listed in `posttrain_data/sources.md` |
| YouTube transcripts | Copyright of the creators; YouTube's terms restrict downloading. `--creative-commons-only` keeps only Creative Commons videos |

Check the licences before sharing a model trained on this data.

## Safety

- Checkpoints are loaded with `weights_only=True`, so they cannot run code. PyTorch before 2.6 has a known bypass
  (CVE-2025-32434): with an older PyTorch, load only checkpoints you made yourself.
- Downloaded and converted data is parsed as data only; nothing in it is executed.
- Only one job should use a laptop GPU at a time. `run_pipeline.py` (plan, train, resume, continue, posttrain),
  `finetune.py` and `posttrain.py` refuse the GPU while it is busy (`--force` overrides in the last two).
- Existing checkpoints and datasets are never overwritten without `--overwrite` (old files are renamed). Saves are
  written to a temporary file, flushed to disk, then renamed.
- Lock files stop two processes from training the same run (`<name>.train.lock`) or building the same dataset
  (`<out>/_build.lock`); a lock left by a process that has ended is removed.

## Tests

```
python -m unittest test_tiny_gpt.py
```

About 30 tests on the CPU: data preparation (filters, duplicates, splits, identical results for any number of
workers), converters, the model and training loop (exact resumption, Muon, the memory savings), the planner,
evaluation, fine-tuning, detection and checkpoint safety.

## Licence

No licence has been chosen for this repository yet.
