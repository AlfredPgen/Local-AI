# tinyGPT: to do

Last updated 11 October 2026.

## Data

- [ ] **Before the next `data_prep.py` run:** compress `ai_training_data` with NTFS compression (about 2x less
      space; scripts keep working): `compact /c /s:"$env:USERPROFILE\ai_training_data" /i /q`
      (undo: `compact /u /s:"$env:USERPROFILE\ai_training_data" /i /q`).

- [ ] **Rebuild the dataset.** `datasets\` is empty (bio_all is gone). Rebuild with `python run_pipeline.py build`
      (about 2 h) once the downloads below have added more text. It picks up the 4,527-term `keywords.txt`.
- [ ] **arXiv download** is running (21 subject areas, `--wait 8-12`). Rerun the same command to continue;
      add `--update` for new papers.
- [ ] **bioRxiv and medRxiv:** first runs still to do, each in its own window:
      `python download_rxiv.py --server biorxiv` and `--server medrxiv` (the bioRxiv harvest takes hours).
- [ ] **Retry 4 arXiv papers** that failed on the page-number bug (fixed in 341f88f): remove their rows
      (`pdf failed: ... invalid literal for int()`) from `_arxiv_meta\report.tsv`, then rerun.
- [ ] **Code data** (planned): Stack-Edu (Python, R) first, then The Stack v2 subsets, Bioconductor/CRAN,
      nf-core/Snakemake, statistical-genetics tools, Stack Exchange dumps. Target about 5-10% of training tokens,
      into the `codes` folder; keep each file's licence. A downloader script is still to be written.
- [ ] **Tokenizer sample (after the current SuperBPE build finishes):** draw it uniformly from all training text in
      pieces of about 4,000 characters, with no per-source weights (`--tokenizer-weights` no longer needed). Today it
      samples whole documents, so a source of a few long documents can drop out entirely: in the 2026-10-11 SuperBPE
      build all 94 books (and the slides) were missing from the tokenizer sample (about a 37% chance with 84 books).
- [ ] **Train/validation split per source:** every source with 2+ documents gets at least one validation document
      (slides and synthetic had none); sources of a few large documents (books) may put whole large documents in
      validation (today a book above 25% of the source's validation budget is forced into training); report.md
      shows each source's validation share against its target and warns when a source has none.
- [ ] **Protect equations from boilerplate removal** in `data_prep.py`: a line that looks like maths (`$...$`,
      `\begin{equation}`, `\frac` and similar) should never count as boilerplate. Today a standard equation on its
      own line that appears in 20+ documents of a source is removed from all of them.
- [ ] **Licences for commercial use:** if a model might ever be sold, train it only on permissively licensed text
      (arXiv's default licence, CC BY-NC PMC articles, SciQ, YouTube transcripts and many bioRxiv preprints are
      not). The downloaders record each document's licence.

## Training

- [ ] **SuperBPE tokenizer** on the rebuilt dataset: set `SUPERBPE = True` in `run_pipeline.py` (or
      `data_prep.py --superbpe`). Compare against ordinary BPE with `compare_models.py` on bits per byte (not loss
      per token, since the tokenizers differ). The small 2026-09-30 test favoured ordinary BPE, but the paper's gains
      grow with data size.
- [ ] **Check for new optimization strategies** before each big run: new optimizers, learning-rate schedules,
      memory and speed tricks for PyTorch/MPS, tokenizer and data-mixing results. Search recent papers and release
      notes; try only what can be tested with short A/B runs.
- [ ] **Muon test** before the next big run: a few short runs (about 500 steps), AdamW against `--optimizer muon`
      at 2-3 learning rates; keep Muon only if its loss is clearly lower at the same step.
- [ ] **8-bit optimizer** (`bitsandbytes`) would let models above about 300M parameters fit on the 8 GB GPU.
      Needs approval to install: `python -m pip install bitsandbytes`.
- [ ] **Dashboard for big models:** use fewer text windows for the CPU forward-pass panels in `view_pt.py`
      when the model has 1B+ parameters.
- [ ] **Benchmark suite** (`benchmark.py`, 21 free benchmarks in `benchmarks\`): only a 50-question trial of
      tinyGPT_best.pt was run (all near chance). Still to do: run the coding benchmarks end to end through the WSL
      sandbox (or Docker when Docker Desktop is running), and run the full suite on the next, larger model.
- [ ] **Post-training at the real size:** `python run_pipeline.py posttrain` once a new pre-trained model exists
      (fine-tuning now accumulates micro-batches, so it fits).

## Mac Studio (delivery 29 January - 12 February 2027)

- [ ] Order: M5 Ultra 36/80, 96 GB, 1 TB (£6,269 with education pricing); keep the box for resale.
- [ ] External drive: ACASIS TB501Pro-SL Thunderbolt 5 case + a 4 TB PCIe 4.0 NVMe without heatsink
      (Samsung 990 Pro or a cheaper Crucial P3 Plus); buy shortly before delivery; format as APFS.
- [ ] First checks on the Mac: training on MPS (BF16, compile), the planner's memory choice, one Blender frame
      timed against the laptop (expected about 6x faster).
- [ ] Pre-training size on the Mac: about 1B parameters is practical (a few weeks per run); 3-5B needs rented
      GPUs or continued training of an open model.

## Housekeeping

- [ ] Organise the local-ai scripts into folders (asked for earlier; moving files must not break running jobs).
- [ ] Rename "tinyGPT" (pending).
- [ ] Describe `download_rxiv.py` and the 21 arXiv areas in the learning guide too (only HOW_TO_RUN has them), then
      rebuild the DOCX with `guide\build_guide.py`.
- [ ] `AUDIT_2026-10-01.md` (local, not committed) contains a home-folder path; delete it or keep it local.

## Ideas

- [ ] Clinical-trial success classifier: AACT (ClinicalTrials.gov) labels plus genetic evidence (Open Targets,
      colocalisation, MR), gradient-boosted trees, a time-based split for testing.
- [ ] Statistical-genetics assistant: an open model fine-tuned on the field, with a paper search index and the fast
      regenie/susieR forks as tools.
