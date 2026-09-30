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
python run_pipeline.py build               # build the dataset datasets\bio_v4 (~2 hours)
python run_pipeline.py plan --hours 48     # show which model fits in 48 hours (trains nothing)
python run_pipeline.py train --hours 48    # train it
python run_pipeline.py resume              # continue after Ctrl+C or a shutdown
python run_pipeline.py continue --hours 24 # after adding more text: train further from the first model
```

- `--hours` is how long you are willing to wait. The planner picks the largest model that can read about 20
  tokens of text per parameter in that time, using the speed this laptop reached in its last training run.
- Each command prints the full `data_prep.py` or `tiny_gpt.py` command it runs, so you can see (and copy) the
  options. The settings (data folder, dataset name, run name) are at the top of `run_pipeline.py`.
- `python run_pipeline.py estimate` shows the expected dataset size first, without building anything.

The rest of this file explains the steps and options behind these commands.

---

## The whole pipeline at a glance

| Step | What happens | Script | How long |
|---|---|---|---|
| 1 | Put text into `$HOME\ai_training_data` | `convert_to_markdown.py`, `download_pmc.py`, `youtube_transcripts.py` | minutes to hours |
| 2 | (Optional) See how big the dataset will be and which model it supports | `estimate_dataset.py` | ~7 min |
| 3 | Clean, filter and tokenise everything into a dataset folder | `data_prep.py` | ~2-2.5 h |
| 4 | Train the model | `tiny_gpt.py` | 12 h to days |
| 5 | Look at the results | `view_pt.py`, `tiny_gpt.py --benchmark`, `compare_models.py` | minutes |
| 6 | Use or fine-tune the model | `tiny_gpt.py --generate`, `finetune.py`, `detect_text.py` | minutes |

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
- Rerunning skips files already converted.
- `--ocr` also reads scanned PDFs (pages that are only pictures) with Tesseract OCR. It is slow, a few seconds
  per page, so it is used only for PDFs with (almost) no text layer: under 200 characters per page on average.

**Open-access papers from PubMed Central**

```
python download_pmc.py --out $HOME\ai_training_data\pmc --max-papers 100000
```

Already done: 99,664 papers (about 1.25 billion tokens). `--max-papers 200000` would add the next 100,000.

**YouTube lecture transcripts**

Add channels (one per line) to `youtube_channels.txt`, then:

```
python youtube_transcripts.py --channels-file youtube_channels.txt
python youtube_transcripts.py "Human Cell Atlas" --list-only        # check a channel name first
python youtube_transcripts.py --channels-file youtube_channels.txt --whisper   # later, when not training
python youtube_transcripts.py --channels-file youtube_channels.txt --recheck-off-topic   # after editing the keywords
```

Only videos whose transcripts match `keywords_biology.txt` are kept. The `--whisper` run transcribes videos that
have no usable captions (it uses the GPU if nothing else does). What happened to each video is listed in
`ai_training_data\_lectures_meta\<channel>_report.tsv` (kept, off topic, queued for Whisper...).

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
python data_prep.py --out datasets\bio_v4 --text-root $D `
    --wiki-dir $D\wikipedia --wiki-max-docs 200000 `
    --parquet "$D\web\fineweb-edu\sample\10BT\*.parquet" --parquet-name fineweb --min-score 3 `
    --jsonl "$D\web\pes2o\data\v2\*.json.gz" --jsonl-name pes2o `
    --include-keywords keywords_biology.txt --keyword-min-distinct 3 `
    --tokenizer-weights books=3,articles=3 --near-dup drop --threads 8
```

What the options mean:

| Option | Meaning |
|---|---|
| `--out datasets\bio_v4` | The new dataset folder (must not exist yet). |
| `--text-root $D` | Read every sub-folder of ai_training_data as a source. |
| `--wiki-dir ... --wiki-max-docs 200000` | Keep the 200,000 Wikipedia articles that match the keywords best. |
| `--parquet ... --min-score 3` | FineWeb-Edu web pages with an education score of 3 or more. |
| `--jsonl ... --jsonl-name pes2o` | Scientific papers from peS2o. |
| `--include-keywords keywords_biology.txt` | Wikipedia, FineWeb and peS2o documents must match these terms (your own folders are not filtered). |
| `--keyword-min-distinct 3` | ...with at least 3 different terms. |
| `--tokenizer-weights books=3,articles=3` | Let books and articles shape the vocabulary more. |
| `--near-dup drop` | Keep only one copy of near-identical documents (for example the 5 copies of *The Germ-Plasm*). |
| `--threads 8` | Use all 8 CPU threads for tokenising. |
| `--scan-only` | Add this to count and filter without tokenising (fast check of the settings). |

The result: `datasets\bio_v4\report.md` explains, per source, what was kept, removed and why. Lines repeated in
20 or more documents (licence notices, navigation) are removed as boilerplate, but headings never are.

`python run_pipeline.py build` runs exactly this command.

---

## Step 4: train

### How the model size is chosen from your data

The planner in `tiny_gpt.py` uses two rules of thumb from published research:

1. **Do not reread the same text more than 4 times.** Your data will be about 2.6 billion tokens, so training can
   use at most 4 x 2.6 = 10.4 billion tokens.
2. **A model needs about 20 tokens of training text per parameter** to be properly trained (a model with too many
   parameters for its text is under-trained and wastes its size). So 10.4 billion tokens can "feed" at most
   10.4 / 20 = 0.52 billion parameters.

It then takes the largest model on a fixed list of sizes that is below that number and fits in the GPU's 8 GB:

| Parameters | Width (`--d-model`) | Layers |
|---:|---:|---:|
| 20.5M | 384 | 8 |
| 29.3M | 448 | 9 |
| 36.6M | 512 | 10 |
| 70.0M | 640 | 12 |
| 99.3M | 768 | 14 |
| 170.7M | 896 | 16 |
| 216.2M | 1024 | 18 |
| 457.2M | 1280 | 22 (does not fit in 8 GB) |

With no time limit this gives the **216M model on 10.4 billion tokens: about 33 days** on this laptop. That is
the best model *for the data*, but it ignores how long you are willing to wait.

**With a time limit**, the choice changes: in a fixed number of hours, a bigger model reads fewer tokens (each
token costs more computation). The planner takes the largest model that can still read 20 tokens per parameter
in the time. That is why 12 h and 24 h both give 36.6M: in 24 h the next size (70M) could read only about
13 tokens per parameter, too few, so the 24 h run gives the 36.6M model twice as much text instead.

The planner looks only at *how much* text there is, never at what it says.

### Choose a run

**Simplest:** give the time and let the planner choose (this is what `run_pipeline.py` does):

```
python tiny_gpt.py --dataset datasets\bio_v4 --time-budget-hours 48 --plan    # look first
python tiny_gpt.py --dataset datasets\bio_v4 --time-budget-hours 48           # then train
```

It times each model size briefly on the GPU, and uses the speed your last training run actually sustained for
hours (a cool GPU runs faster for a few seconds than it does all day; before the first run it allows for that).

**Or pick the model yourself** from this table. Times use the measured speed of your last run (the GPU was capped
at 55 W then; at 40 W expect longer).

| Time | Model | Command |
|---|---|---|
| ~12 h | 36.6M | `python tiny_gpt.py --dataset datasets\bio_v4 --d-model 512 --layers 10 --train-tokens 950000000` |
| ~24 h | 36.6M | `python tiny_gpt.py --dataset datasets\bio_v4 --d-model 512 --layers 10 --train-tokens 1890000000` |
| ~48 h | 70M | `python tiny_gpt.py --dataset datasets\bio_v4 --d-model 640 --layers 12 --train-tokens 1850000000` |
| ~3 days | 99M | `python tiny_gpt.py --dataset datasets\bio_v4 --d-model 768 --layers 14 --train-tokens 2000000000` |

- More time gives a better model, but each extra day helps less. By the loss formula of the 2022 "Chinchilla"
  study (which predicts final loss from model size and training text), going from 12 h to 24 h improves the
  model about as much as 24 h to 48 h, and the third day adds half as much again.
- **Always preview first:** add `--plan` to any command to print the model, batch, steps and ratios without
  training.
- `--train-tokens` sets how much is trained. **Do not use `--steps` for this**: `--steps` on its own only splits
  the data's full budget (4 passes, 10.4 billion tokens) into that many steps, so it still trains for weeks.

### Is a bigger or deeper model better?

Only if it also gets enough text. For a given number of parameters, research found that the exact split between
width and depth matters little; the list above grows both together. Adding layers alone makes the model slower
and, without more text, not better.

### During training

- The first minute or so of a run is spent compiling the model for the GPU (`torch.compile`); after that it trains
  about 1.45 times faster. The header line says `compile: on`.
- Every 200 steps a **metrics line**: training loss, validation loss per source, bits per byte, speed, energy used,
  CO2 and electricity cost, time left. `NEW BEST` means the model improved; `tinyGPT_best.pt` is saved then.
- **Pause:** press Ctrl+C. **Continue:** `python tiny_gpt.py --resume` (same `--name` if you used one).
- **Overfitting** (memorising text) would show as validation loss rising while training loss keeps falling. The
  runs above read the data less than once, so it is not a risk; the best checkpoint is kept anyway.
- A second run with a different name: add `--name tinyGPT_70M`. Reusing a name needs `--overwrite` (old files are
  renamed, not deleted).

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

1. **Builds a new dataset, `datasets\bio_v5`, from all of `ai_training_data`** (old and new text) with the
   *same tokenizer* as the first dataset (`data_prep.py --tokenizer-from datasets\bio_v4`). The same tokenizer is
   essential: the model's weights belong to its vocabulary, and a new tokenizer would give every token a different
   meaning. Keeping the old text in stops the model from forgetting it while it learns the new, and the old
   train/validation split is kept too (`--keep-split-from datasets\bio_v4`): a document the first model trained on
   never becomes a validation document, where its loss would look better than the model really is.
2. **Trains further from the first model's weights**
   (`tiny_gpt.py --dataset datasets\bio_v5 --init-from tinyGPT_best.pt --name tinyGPT_continued
   --time-budget-hours 24`): same model size, a fresh learning-rate schedule, and as much of the new dataset as fits
   in the hours given.

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
python view_pt.py tinyGPT_best.pt --plot                     # dashboard picture: tinyGPT_best_dashboard.png
python tiny_gpt.py --benchmark                               # 336 biology fact questions, per category
python compare_models.py A_best.pt B_best.pt --device cpu    # is A really better than B?
```

`experiments.csv` (opens in Excel) has one row per training run, benchmark and comparison.

## Step 6: use the model

```
python tiny_gpt.py --generate "Genetic drift is" --length 100
python tiny_gpt.py --export exported_model                   # safetensors + tokenizer, for other tools
python finetune.py sft --base tinyGPT_best.pt --data finetune_examples\sft_examples.jsonl --name tinyGPT_sft
python finetune.py ask --checkpoint tinyGPT_sft.pt "What does genetic drift do to small populations?"
python detect_text.py some_text.txt --checkpoint tinyGPT_best.pt
```

---

## Options of tiny_gpt.py you are likely to need

| Option | What it does | Default |
|---|---|---|
| `--dataset FOLDER` | Dataset built by data_prep.py | required for training |
| `--plan` | Show the plan and stop | |
| `--train-tokens N` | How many tokens to train on | 4 passes over the data |
| `--d-model N --layers N` | Model width and depth (pick a pair from the table) | chosen from the data |
| `--time-budget-hours H` | Let the planner fit the run into H hours | off |
| `--name NAME` | Name of the checkpoint files | tinyGPT |
| `--resume` | Continue a paused run | |
| `--overwrite` | Start again under an existing name (old files renamed) | |
| `--eval-every N` | Steps between metrics lines | 200 |
| `--save-minutes M` | Save the training state every M minutes | 30 |
| `--mix wiki=1,notes=3` | Read some sources more often than their size | by size |
| `--device cpu` | Run on the CPU (for anything while the GPU trains) | auto |
| `--tariff`, `--region` | Electricity price for the cost figure | Flexible Octopus, H |
| `--generate TEXT --length N` | Write text from the best checkpoint | |
| `--benchmark` | Score the fact benchmark | |

`python tiny_gpt.py --help` lists every option.

## When something goes wrong

| Message | What to do |
|---|---|
| `... already exists` | Use a new `--name`, or add `--overwrite`. |
| `Out of memory at the first step; retrying ...` | Nothing: it halves the batch automatically. |
| `training loss became nan` | Restart from the last checkpoint with a lower `--lr` (see the message). |
| The screen goes black / driver reset | Two GPU jobs ran at once. Resume with `--resume`; keep other jobs on `--device cpu`. |
| `manifest.json missing` | The dataset folder was deleted or is incomplete: rebuild it (step 3). |
