# tinyGPT: to do

Last updated 11 October 2026.

## Data

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
