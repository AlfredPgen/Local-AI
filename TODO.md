# tinyGPT: to do

Last updated 5 November 2026.

## Data

- [x] **Compression (27 October):** the text folders of `ai_training_data` use NTFS compression (new files are
      compressed automatically; undo per folder: `compact /u /s:<folder> /i /q`); data_prep.py and
      estimate_dataset.py also read `.md.gz` and `.txt.gz` files.
- [ ] **PMC by keywords:** `python download_pmc.py --out $HOME\ai_training_data\pmc --keywords keywords.txt
      --max-papers 300000` (harvest about 13 h once; best overnight UK time, as NCBI asks).

- [ ] **Rebuild the dataset** once the downloads below have added text: `datasets\bio_all` (built 11 October,
      4.08B training tokens, SuperBPE 16,384) is what tinyGPT_muon trains on. A rebuild (about 3 h) picks up the
      22,650-term `keywords.txt`; for this model it must keep its tokenizer and split
      (`python run_pipeline.py continue` does: `--tokenizer-from tinyGPT_muon_best.pt --keep-split-from
      datasets\bio_all`), otherwise the vocabulary rule gives 32,768 pieces and needs a fresh model.
- [ ] **arXiv download** is running (21 subject areas, `--wait 8-12`). Rerun the same command to continue;
      add `--update` for new papers.
- [ ] **bioRxiv and medRxiv** downloads are running. After the 27 October keyword update, rerun each with
      `--rebuild-list` so the list uses the new terms (also `download_arxiv.py --rebuild-list` and
      `youtube_transcripts.py --recheck-off-topic`).
- [ ] **Software documentation** (`download_docs.py`, 1,205 sources for 919 tools in `doc_sources.tsv`) is running;
      rerun the same command to continue or retry failed rows. Five sites refuse scripts: save the GCTA, SMR, GSMR,
      LDAK and KING manual pages from the browser into a folder and convert them with `convert_to_markdown.py`.
      www.gnu.org (bash, coreutils, gawk, sed, grep, make, parallel manuals) timed out on 27 October; the next run
      retries it.
- [ ] **NVIDIA SimReady 3D assets** (SimReady Central, more than 2,000 OpenUSD assets on Hugging Face; licence per
      asset, often CC BY 4.0, some non-commercial): a converter that keeps the prim hierarchy, part names, sizes,
      materials and physics settings as text (vertex arrays replaced by counts) into `ai_training_data\3d`. Needs
      `usd-core` (approval: `python -m pip install usd-core`).
- [ ] **Retry 4 arXiv papers** that failed on the page-number bug (fixed in 341f88f): remove their rows
      (`pdf failed: ... invalid literal for int()`) from `_arxiv_meta\report.tsv`, then rerun.
- [ ] **Code data** (planned): Stack-Edu (Python, R) first, then The Stack v2 subsets, Bioconductor/CRAN,
      nf-core/Snakemake, statistical-genetics tools, Stack Exchange dumps. Target about 5-10% of training tokens,
      into the `codes` folder; keep each file's licence. A downloader script is still to be written.
- [x] **Tokenizer sample (5 November):** drawn in 4,000-character windows from all training text, so every source
      takes part in proportion (in the 2026-10-11 SuperBPE build all 89 books and the slides were missing from the
      sample). Takes effect at the next build; bio_all keeps its tokenizer.
- [ ] **Train/validation split per source:** report.md now warns when a source of 2+ documents has no validation
      document (slides and synthetic had none); the split rule itself is unchanged (a 1-2 document source cannot
      have a meaningful validation loss). Still open: sources of a few large documents (books), where a book above
      25% of the source's validation budget is forced into training.
- [x] **Protect equations from boilerplate removal (5 November):** lines with `$$`, `\begin{` or `\end{` and
      headings followed by a paragraph are never boilerplate. Takes effect at the next build.
- [ ] **Licences for commercial use:** if a model might ever be sold, train it only on permissively licensed text
      (arXiv's default licence, CC BY-NC PMC articles, SciQ, YouTube transcripts and many bioRxiv preprints are
      not). The downloaders record each document's licence.

## Training

- [ ] **SuperBPE against ordinary BPE at this size:** bio_all IS SuperBPE (16,384 pieces) and `SUPERBPE = True`
      in `run_pipeline.py` now says so. The comparison is still to run: the same model on a BPE and a SuperBPE
      build of the same text, judged by bits per byte with `compare_models.py` (not loss per token, since the
      tokenizers differ). The only test (2026-09-30, 20.5M parameters) favoured ordinary BPE by 3.1%; the paper's
      gains grow with model and data size.
- [ ] **Check for new optimization strategies** before each big run: new optimizers, learning-rate schedules,
      memory and speed tricks for PyTorch/MPS, tokenizer and data-mixing results. Search recent papers and release
      notes; try only what can be tested with short A/B runs.
- [ ] **Muon test, still owed** (the 216M run started with Muon without it): a few short runs (about 500 steps),
      AdamW against `--optimizer muon` at 2-3 learning rates; keep Muon only if its loss is clearly lower at the same
      step. The 5 November learning-rate sweep was Muon only.
- [ ] **Check the doubled learning-rate table at real size:** stop the first long run planned with the new table at
      2,000 steps if its bits per byte is worse than an old-table run at the same step (`--lr-scale 0.5
      --weight-decay 0.1` gives the old setting). The sweep that doubled it ran 500 steps at 20.5M and 36.6M
      parameters (`results\ab_2026-11-05.md`).
- [ ] **Resume tinyGPT_muon with the new code** (each needs your approval; HOW_TO_RUN.md, Step 4): `--batch-size 2
      --grad-accum 132 --activation-checkpointing off` (5-16% faster, same mathematics), `--min-lr-ratio 0` (decay to
      zero; nothing changes before step 48,001), `--ema 0.999` (+1.7 GB RAM; a better stable-phase `_best.pt`).
- [ ] **EMA on a long run:** its stable-phase gain is untested (the 500-step A/B could not show it).
- [ ] **Mixture of experts:** 2.2-2.3x slower than dense per step on this GPU in the 5 November smoke test. Profile
      the loop implementation under compile (a host sync per MoE block), try `--moe-impl grouped`, and judge expert
      balance after 500+ steps. Dense stays the default until an A/B at a size the Mac can train says otherwise.
- [ ] **8-bit optimizer** (`bitsandbytes`) would let models above about 300M parameters fit on the 8 GB GPU.
      Needs approval to install: `python -m pip install bitsandbytes`.
- [x] **Dashboard for big models:** use fewer text windows for the CPU forward-pass panels in `view_pt.py`
      when the model has 1B+ parameters.
- [ ] **Benchmark suite** (`benchmark.py`, 21 free benchmarks in `benchmarks\`): only a 50-question trial of
      tinyGPT_best.pt (the 70M October run) was run (all near chance). Still to do: run the coding benchmarks end
      to end through the WSL sandbox (or Docker when Docker Desktop is running), and run the full suite on the
      216M model.
- [ ] **Post-training at the real size:** `python run_pipeline.py posttrain` once a new pre-trained model exists
      (fine-tuning now accumulates micro-batches, so it fits).

## Mac Studio (delivery 29 January - 12 February 2027)

- [ ] Order: M5 Ultra 36/80, 96 GB, 1 TB (£6,269 with education pricing); keep the box for resale.
- [ ] External drive: ACASIS TB501Pro-SL Thunderbolt 5 case + a 4 TB PCIe 4.0 NVMe without heatsink
      (Samsung 990 Pro or a cheaper Crucial P3 Plus); buy shortly before delivery; format as APFS.
- [ ] First checks on the Mac: the checklist in HOW_TO_RUN.md (Mac Studio): `gpu_check.py`, the 216M kernel
      benchmark, `--plan` with the attention-memory probe, one short run with `--compile on` and `off`, `peak_mem_gib`
      against the estimate; then raise the MPS budget from 75% to 80% if the estimate holds. Also one Blender frame
      timed against the laptop (expected about 6x faster).
- [ ] Pre-training size on the Mac: about 1B parameters is practical (a few weeks per run); 3-5B needs rented
      GPUs or continued training of an open model.

## Housekeeping

- [ ] Organise the local-ai scripts into folders (asked for earlier; moving files must not break running jobs).
- [ ] Rename "tinyGPT" (pending).
- [x] `AUDIT_2026-10-01.md` (local, not committed) contained a home-folder path; it has been deleted.

## Ideas

- [ ] **Count overlapping keyword matches once.** `data_prep.py` counts every term that matches, so one mention of
      "linear regression" scores both "linear regression" and "regression" (two distinct terms of the three needed).
      Phrases inside phrases ('China Kadoorie Biobank' and 'China Kadoorie') do the same. Counting each stretch of
      text once (the longest match) would make the distinct-term rule mean what it says; `tools/kw_covered.py
      --nested` lists the nested pairs. Measure the keep rate before and after on FineWeb-Edu and peS2o samples.
- [ ] **SFT pairs from the software documentation:** question-and-answer pairs ("run REGENIE step 1 on these
      files") made from `software_doc`, so the model learns to write the commands, not only to recognise them.

- [ ] Clinical-trial success classifier: AACT (ClinicalTrials.gov) labels plus genetic evidence (Open Targets,
      colocalisation, MR), gradient-boosted trees, a time-based split for testing.
- [ ] Statistical-genetics assistant: an open model fine-tuned on the field, with a paper search index and the fast
      regenie/susieR forks as tools.
