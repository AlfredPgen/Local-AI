---
title: "tinyGPT: a learning guide"
subtitle: "Training a small biology language model on your own data, understanding every step, and scaling it up"
date: "29 September 2026"
---

# Start here

## What tinyGPT is

tinyGPT is a small language model that you train from scratch on a laptop with an RTX 3070 GPU. It uses the same design as GPT,
Llama and Qwen (a decoder-only transformer), and it learns from your own biology, genetics and statistics text plus
selected parts of Wikipedia and the web. The run started on 29 September has **17,308,032 parameters** (not 31
million) and will read **778 million tokens**.

At this size tinyGPT won't become an expert assistant. Its purpose is to let you:

- learn how every part of LLM training works, on your own machine;
- measure honestly whether each change helps;
- produce a domain-flavoured text model that can grow later.

Chapter 11 explains what it would take to build something much stronger.

This guide replaces three earlier documents: *HOW_LLMS_ARE_TRAINED.md* (the concepts), *GUIDE.md* (the Mac Studio
and cloud plan) and *tinyGPT_advanced_techniques.docx* (the technique catalogue). The two Markdown files stay in the
folder as sources.

![**The tinyGPT pipeline.** **a**, text sources. **b**, data preparation turns raw text into clean, deduplicated,
English, on-topic token shards, split by document. **c**, pre-training repeats a four-step loop (forward pass, loss,
gradients, update) and evaluates every 200 steps. **d**, post-training turns the base model into a question
answerer. **e**, tools to inspect, compare, export and use checkpoints.](figures/fig1_pipeline.png){width=16.5cm}

## The files

| File | What it does |
|-------------------------|---------------------------------------------------------------------------|
| data_prep.py | Builds a tokenised dataset from your folders, Wikipedia, FineWeb-Edu and peS2o: cleaning, English filter, keyword filter, deduplication, document-level split, tokenizer, leakage audit and a report |
| tiny_gpt.py | Plans and trains the model; also `--plan`, `--generate`, `--benchmark` and `--export` |
| view_pt.py | Dashboard picture of a checkpoint: weights, embeddings, attention, outputs, calibration, benchmark and loss curves |
| compare_models.py | Is checkpoint A really better than B? A paired bootstrap over validation documents |
| finetune.py | Post-training: `sft` (question and answer pairs), `dpo` (preferred and rejected answers) and `ask` |
| detect_text.py | Was this text written by my model? A watermark test and a likelihood score |
| convert_to_markdown.py | Turns PDF, DOCX, PPTX, XLSX, CSV, HTML, code and GWAS summary statistics into Markdown, sorted by type |
| gpu_check.py | GPU readings, a load test and a training benchmark; refuses to run while the GPU is busy |
| test_tiny_gpt.py | 18 automated tests that run on the CPU |
| keywords_biology.txt | 2,385 topic terms for the keyword filter |
| probes_biology.tsv | The fact benchmark: 336 questions in 16 categories, with near-miss wrong answers |
| calibrate_probes.py | Sets the benchmark's difficulty labels from how open reference models score it |
| finetune_examples/ | Format templates: 20 SFT examples and 12 DPO pairs |
| experiments.csv | A permanent table with one row per training run, benchmark, comparison or fine-tune |
| log.txt | Everything the scripts print, appended |

## Everyday commands

The simplest route is `run_pipeline.py` (build, plan, train, resume); `HOW_TO_RUN.md` is a short run sheet with
every step and option.

```
cd $HOME\local-ai
python run_pipeline.py build                               # build datasets\bio_v4 from ai_training_data
python run_pipeline.py plan --hours 48                     # which model fits in 48 hours
python run_pipeline.py train --hours 48                    # train it
python tiny_gpt.py --dataset datasets\bio_v2 --plan        # what would be trained
python tiny_gpt.py --dataset datasets\bio_v2               # train, planned for you
python tiny_gpt.py --resume                                # continue after Ctrl+C
python tiny_gpt.py --generate "Genetic drift is"           # write text
python tiny_gpt.py --benchmark                             # the 336-question fact test
python view_pt.py tinyGPT_best.pt --plot                   # dashboard picture
python compare_models.py A_best.pt B_best.pt --device cpu  # A versus B, 95% interval
python finetune.py sft --base tinyGPT_best.pt --name tinyGPT_sft `
    --data finetune_examples\sft_examples.jsonl
python detect_text.py sample.txt --checkpoint tinyGPT_best.pt
```

Long commands are split with a backtick (`` ` ``) at the end of each line, as PowerShell (the VS Code terminal)
expects. In the old Command Prompt, use `^` instead.

> **One rule while a run is training:** give every other script `--device cpu`. Two GPU jobs at once caused a driver
> reset (a "TDR") that killed an earlier run. `gpu_check.py` refuses to start while the GPU is busy.

## What changed on 29 September 2026

- **Bits per byte** is now in every metrics line. Unlike loss per token, it doesn't depend on the tokenizer
  (Section 9.1).
- **compare_models.py** gives a paired, document-level bootstrap between two checkpoints (Section 6.3).
- **z-loss** keeps the output scores from drifting. It's on by default at $10^{-4}$.
- **experiments.csv** keeps a permanent table of every run and test (Section 5.3).
- **finetune.py** adds supervised fine-tuning and DPO (Chapter 7).
- **detect_text.py** and **generation watermarks** (Chapter 8).
- **The English-only filter** is on by default, and the **keyword filter is 11 times faster** (Chapter 4).
- **Training data** is sorted into books, articles, scripts, notes and other folders, and each folder becomes a named
  source.
- **New inputs**:
  - JSON Lines (gzipped) input, used for peS2o;
  - FineWeb-Edu and peS2o samples downloaded;
  - many more document types in the converter, with a fast GWAS reader.
- **The benchmark** was rebuilt: 336 questions with near-miss wrong answers, difficulty set from reference models,
  and a floor (shuffled questions) reported next to every score.

## What changed on 30 September 2026: the script audit

Every script was reviewed for bugs, stale code and speed. What you will notice:

- **Bits per byte is now exact.** The old count missed one byte for every blank line, so bits per byte came out 0.2
  to 0.6% too high (books 0.6%, Wikipedia 0.3%). Runs started after the audit show slightly lower values than
  earlier runs of the same quality (Section 9.1).
- **The keyword prefilter never rejects a document the full check would keep.** The old prefilter counted
  overlapping terms once ("genetic drift" also contains "genetic\*"), so it wrongly rejected 0.3% of the FineWeb-Edu
  and 0.65% of the peS2o documents the full check keeps. bio_v3 was built before this fix; the loss is too small to
  rebuild for.
- **`--wiki-max-docs`, `--parquet-max-docs` and `--jsonl-max-docs` now deliver the number asked for** whenever
  enough documents pass. bio_v3 asked for 200,000 Wikipedia articles and kept 196,817; the report now says so when a
  source falls short.
- **Long documents keep their paragraph breaks** where they are cut into parts for the tokenizer. Before, the two
  paragraphs at each cut ran together.
- **docs.tsv** now really lists up to 2,000 rejected web records per source (before, almost none), in a fixed order.
- **Safer reruns:** a scan-only report is never deleted as an "incomplete build", and a refused rerun no longer
  deletes a `_work` folder kept with `--keep-work`.
- **Clearer errors:** `--keyword-sources` ignores case and rejects unknown names; source names that differ only in
  case are refused (their token files would collide on Windows); `--min-score` needs the score column; a build in
  which no document survives stops with a message instead of a tokenizer crash.
- **tinyGPT training:** the seed now fixes the model's starting weights even with `--time-budget-hours`; a diverged
  run (NaN) is never saved as the training state; a NEW BEST is on disk before anything else can fail;
  `--grad-clip 0` really switches clipping off; resuming mid-interval, with dropout, or with a changed benchmark file
  is handled correctly.
- **compare_models.py** needs at least 20 documents for a verdict. **detect_text.py** counts each (previous token,
  token) pair once and reports an exact p-value (Section 9.7).
- **PyTorch 2.10 and torch.compile (same day):** PyTorch was upgraded from 2.5.1 to 2.10, which also closes the
  checkpoint-loading flaw CVE-2025-32434, and the `triton-windows` package makes `torch.compile` work on this
  laptop: training is about 1.45 times faster (Section 3.1).
- **New tools the same day:**
  - `download_pmc.py`: 100,000 open-access genetics and genomics papers from PubMed Central as Markdown
    (Section 4.8).
  - `youtube_transcripts.py`: lecture transcripts from YouTube channels, kept only if on topic (Section 4.9).
  - `estimate_dataset.py`: tokens per source and the model sizes they support, without building (Section 4.5).
  - Energy, carbon and electricity cost in every metrics line of tiny_gpt.py (Section 5.4).
  - `run_pipeline.py` and `HOW_TO_RUN.md`: the whole pipeline in a few commands, and a plain run sheet.
  - `convert_to_markdown.py --ocr`: scanned PDFs read with Tesseract OCR (Section 4.2).
- **Faster, better data preparation:** near-duplicate detection is about 10 times faster (one-permutation
  MinHash, Section 9.8), and headings are no longer removed as boilerplate.
- **Realistic time budgets:** `--time-budget-hours` now uses the speed your last training run sustained, not a
  few seconds on a cool GPU, which had been 1.7 times too optimistic.

# How language models are trained

A language model is **one very large function**. It takes the text so far and gives a probability for every possible
next piece of text. Training adjusts its numbers (the **parameters**, or weights) so that those probabilities match
real text. Everything else makes that prediction useful.

Most of it maps onto statistics you already use: maximum likelihood, logistic regression, priors, cross-validation and
paired comparisons. The **"For a statistician"** notes point out those links.

![**The four training stages.** Pre-training costs most of the compute. The later stages decide most of how useful
the model is.](figures/fig3_stages.png){width=16cm}

## Tokens: text becomes numbers

A model can't read letters. Text is split into **tokens**: common words, word pieces and punctuation, each with an
integer ID. The vocabulary is built by **byte-pair encoding (BPE)**:

1. Start from single characters.
2. Find the pair of neighbouring tokens that occurs most often in a large sample of text, and merge it into a new
   token.
3. Repeat until the vocabulary has the size you want.

Frequent words become one token. Rare ones are split into pieces: *Weinberg* might become *We* + *inberg*. Large
models use 100,000 to 250,000 tokens; in English, one word is about 1.3 tokens.

**In tinyGPT:**

- **Tokenizer:** a SentencePiece BPE tokenizer with **8,192 pieces**, trained by data_prep.py on your own data. Its
  vocabulary therefore contains pieces such as *allele*, *heritab* and *GWAS*.
- **Digits** are split one by one, so the model learns numbers digit by digit.
- **Lines** are encoded one at a time, with a newline token between them, so Markdown structure is kept.
- **Special tokens:** begin-of-document (1) and end-of-document (2).
- **Starting point:** a model that guesses uniformly has loss $\ln 8192 = 9.01$. Your log's baseline line shows
  exactly this.

## The transformer

**Embeddings.** Each token ID is replaced by a **vector** of numbers: 384 in your run, 4,096 in a mid-sized model.
Tokens used in similar ways end up with similar vectors; view_pt.py plots them.

**Layers.** The vectors pass through a stack of identical **blocks**: 8 in your run, 30 to 100 in large models. Each
block has two parts.

- **Attention** lets every token gather information from the tokens before it. Each token computes three vectors:
  - a *query* ("what am I looking for?");
  - a *key* ("what do I contain?");
  - a *value* ("what will I pass on?").

  The query is compared with the keys of all earlier tokens by dot products. A softmax turns the scores into
  weights that sum to 1, and the token takes the weighted average of those tokens' values. A token never looks
  ahead: that's the *causal mask*.
- The **feed-forward** part is a small two-layer network applied to each token separately. Much of the model's stored
  knowledge lives in these weights.

Each part *adds* its result to the running vector (a **residual connection**) instead of replacing it. That's what
makes deep stacks trainable.

**Output.** The final vector at each position is turned into one score (a **logit**) per vocabulary token, and a
softmax turns the scores into probabilities.

> **For a statistician:**
>
> - Attention is a **kernel smoother** (Nadaraya–Watson) whose similarity function is learnt. Several *heads* run in
>   parallel, each learning a different similarity.
> - The output layer is **multinomial logistic regression**. Its predictors are the final vector, which the rest of
>   the network learns to construct.

**Mixture of experts (MoE).** Here the feed-forward part is split into many *experts*, and a small router picks a few
of them for each token. That's why "320B total, 18B active" models exist: memory holds all the parameters, but the
speed per token depends only on the active ones.

## What it learns: predict the next token

Take any sentence, for example "Genetic drift is the random change in allele frequency". For **every position**, the
model predicts the next token from the ones before it. The text is its own answer key, so no human labels are
needed.

The loss for one prediction is $-\ln$ of the probability the model gave to the correct token:

| Probability given to the correct token | Loss |
|---|---|
| 0.9 | 0.11 |
| 0.6 | 0.51 |
| 0.01 | 4.61 |

Training minimises the average of this loss over all predictions: the **cross-entropy**. **Perplexity** is
$e^{\text{loss}}$, the effective number of tokens the model hesitates between. Your run's validation loss fell from
9.07 (guessing) to 4.89 after 200 steps, a perplexity of 133. Good large models reach a perplexity of about 5 to 7 on
general text.

> **For a statistician:** this is exactly **maximum likelihood**. The model is a conditional multinomial
> distribution for the next token, and the loss is the negative log-likelihood per token. Weight decay acts as an L2
> penalty, like a Gaussian prior (MAP estimation).

Predicting text well rewards learning grammar, facts and patterns of reasoning, because all of them lower the loss.
That's why next-token prediction produces capable models at scale.

## How it learns: gradient descent

1. **Take a batch.** In your run that's 128 windows of 512 tokens: 65,536 predictions per step.
2. **Forward pass:** compute the probabilities and the average loss.
3. **Backward pass (back-propagation):** the chain rule gives the **gradient**, the direction in which each parameter
   should move to reduce the loss. It costs about twice the forward pass.
4. **Update** every parameter a small step against its gradient. The step size is the **learning rate**.
5. Repeat: 11,870 times in your run.

The details that make it work:

- **AdamW** keeps running averages of each parameter's gradient and squared gradient. Every parameter then gets its
  own step size, a cheap stand-in for second-order methods such as Newton–Raphson in a GLM.
- **Learning-rate schedule.** A short *warm-up*, a long *stable* phase, then a *decay* (Section 9.3).
- **Mixed precision.** Most arithmetic is done in 16-bit bfloat16, which halves memory and roughly doubles speed.
  Sensitive sums stay in 32-bit.
- **Compute rule of thumb.** Training costs about $6ND$ floating-point operations for $N$ parameters and $D$ tokens.
  Your run: $6 \times 1.73\times10^{7} \times 7.78\times10^{8} = 8.1\times10^{16}$.

## How it writes: generation

The model writes **one token at a time**. It gives probabilities for the next token, one is sampled and appended, and
the loop repeats. Three settings control the sampling (`--temperature`, `--top-k` and `--top-p` in tiny_gpt.py):

- **Temperature** divides the logits before the softmax. Low values (0.3) almost always pick the top token; high
  values (1.2) give variety and more mistakes.
- **Top-k** keeps only the $k$ most likely tokens.
- **Top-p** keeps the smallest set of tokens whose probabilities add up to $p$.

The model has no separate fact store and no plan: a long answer is thousands of next-token steps. That's also why
reasoning models write out their thinking first, because the written reasoning becomes context for the answer.

## The four training stages

**1. Pre-training.** Raw text, next-token loss.

- **Frontier scale:** 15 to 40 trillion tokens of web pages, books, code, papers and maths, after heavy
  deduplication and quality filtering.
- **Example:** Llama 3.1 405B saw 15.6T tokens, about $6 \times 4.05\times10^{11} \times 1.56\times10^{13} =
  3.8\times10^{25}$ operations, or roughly 30 million H100 GPU-hours.
- **Result:** a *base model* that continues text but doesn't follow instructions.

**2. Supervised fine-tuning (SFT).**

- **Data:** $10^3$ to $10^6$ (prompt, ideal answer) pairs.
- **Method:** the same loss, but computed on the answer tokens only.
- **Cost:** under 1% of pre-training compute, yet it turns an autocomplete engine into an assistant, because the
  knowledge is already there and SFT teaches the format.

**3. Preference tuning.** Many qualities are easier to *judge* than to *write*, so people or AI judges compare pairs
of answers.

- **RLHF** fits a reward model (Bradley–Terry) and then optimises the model against it, with a penalty for drifting
  from the SFT model.
- **DPO** skips the reward model and uses a logistic loss on the pairs directly (Section 9.6).

> **For a statistician:** the reward model is the **Bradley–Terry model** for paired comparisons. The KL penalty is
> shrinkage towards the reference model, like a prior.

**4. Reinforcement learning with checkable rewards.** This is the big advance of 2024 to 2026, behind the recent gains
in maths, code and analysis.

- The model tries each checkable problem many times.
- Correct attempts are made more likely and failed ones less likely.
- **GRPO** scores each attempt against the others for the same problem: (reward − group mean) ÷ group sd, a
  standardised residual.
- Checking one's own work and backtracking emerge because they lead to right answers.

## Measuring progress, and the overfitting trap

- **Validation loss** is the loss on text the model never trained on, the equivalent of out-of-sample deviance.
  - Training loss always falls.
  - If validation loss **rises** while training loss falls, the model is memorising: that's overfitting.
  - tinyGPT splits by **document**, so no document appears on both sides.
- **Benchmarks** are fixed question sets. **Contamination** (benchmark items leaking into training data) inflates
  scores. data_prep.py's 13-gram audit checks the train/validation split for this.
- **Your own evaluation set** beats any public benchmark for your use.

## Making models smaller and cheaper

- **Mixture of experts:** large knowledge, small compute per token.
- **Distillation:** train a small student on a large teacher's outputs or probability distributions. Most "Flash" and
  "mini" models are made this way.
- **Quantisation:** store each weight in fewer bits.
  - 4-bit with block scales takes about 0.56 bytes per parameter.
  - The loss in quality is small at 4-bit, noticeable at 3-bit and large at 2-bit.

## Fine-tuning at home: LoRA and QLoRA

Updating all the weights needs about 16 bytes per parameter (the weight, its gradient and Adam's two averages).
**LoRA** freezes the original weights and learns a low-rank correction $W' = W + BA$, where $B$ is $d\times r$, $A$ is
$r\times d$ and the rank $r$ is 8 to 64.

- For a 4,096 × 4,096 matrix, a rank of 16 trains 131,072 numbers instead of 16.8 million (0.8%).
- **QLoRA** does the same on a 4-bit copy of the model.

> **For a statistician:** LoRA restricts the *change* in each weight matrix to a rank-$r$ subspace, like fitting only
> the first $r$ principal components of the update.

**What fine-tuning can change:**

- style and format;
- house rules;
- focus;
- familiarity with your field's vocabulary.

**What it can't change much:** raw reasoning, and reliable recall of new facts. Recall is better served by searching
your documents at answer time.

## Why the top closed models lead

In order of importance:

1. **Post-training scale:** expert data, preference data and, above all, reinforcement-learning environments.
2. **Size:** frontier models are probably far larger than anything that fits on a desktop.
3. **Compression:** quantisation and distillation each cost a little quality.
4. **Time:** open models tend to reach a given level some months later.

Search over your papers plus tools that run R and Python close much of the gap *for your own tasks*.

# The tinyGPT model

![**The tinyGPT architecture.** **a**, one forward pass: embedding, $L$ identical blocks (each with two residual
additions), final normalisation and the tied output layer. **b**, causal attention in one head. **c**, where the
17,308,032 parameters of your run sit.](figures/fig2_architecture.png){width=16.5cm}

## Design choices, and why

| Part | Choice | Why |
|---------------|-----------------------------------|--------------------------------------------------|
| Positions | **RoPE** (rotary embeddings) | Attention scores depend on the distance between tokens; no learnt position table |
| Normalisation | **RMSNorm**, applied before each part (pre-norm) | Simpler and more stable than LayerNorm; used by Llama and Qwen |
| Feed-forward | **SwiGLU**, hidden width $8d/3$ rounded up to a multiple of 64 | Better loss per parameter than a GELU network of the same size |
| Attention | **Grouped-query attention (GQA)** once there are 8 or more heads | Several query heads share one key/value head; less memory at generation. Your run has 6 heads, so it uses ordinary multi-head attention |
| Output | **Tied embeddings** | The output layer reuses the embedding matrix, saving 3.1M parameters (18% of your model) |
| Biases | None | They add little and complicate weight decay |
| Attention kernel | PyTorch SDPA, `repeat_kv` path chosen by a timing race | On Windows the native GQA path falls back to a kernel about 5 times slower (17 times with torch 2.5.1) |
| Compilation | `torch.compile` when Triton is installed (`triton-windows` on Windows) | Fuses many small GPU operations into few: 1.46 times faster for 36.6M and 1.44 times for 70M parameters on the RTX 3070, measured alternating both modes; the first steps take 10-30 s longer to compile |

## Why decoder-only, and not encoder-decoder?

Transformers come in three families. They share the same blocks; they differ in which tokens each token may attend to,
and in what they're trained to predict.

| Family | Examples | Attention | Trained to | Best at |
|--------------|----------------------|----------------------|--------------------------|--------------------------|
| Encoder-only | BERT, PubMedBERT, BioLinkBERT | Both directions: every token sees the whole text | Fill in masked words (15% of tokens) | Understanding: classification, entity tagging, embeddings for search |
| Encoder-decoder | T5, BART | Encoder: both directions over the input. Decoder: causal, plus attention to the encoder's output | Rebuild corrupted spans, or map an input to a separate output | Clear input → output tasks: translation, summarisation |
| Decoder-only | GPT, Llama, Qwen, tinyGPT | Causal: each token sees only earlier tokens | Predict the next token at every position | Open-ended generation; after SFT, any task phrased as text in, text out |

**Why decoder-only suits tinyGPT:**

1. **Every token is a training example.** The next-token loss is computed at every position. BERT's and T5's
   objectives score only about 15% of tokens, which gives less learning signal from the same limited data.
2. **One sequence does everything.** Prompt and answer are just one text. SFT, DPO, RL, few-shot examples and
   retrieved passages pasted into the prompt need no change to the model. An encoder-decoder must decide what counts
   as input and what counts as output.
3. **Evidence.** In a controlled comparison at 5B parameters, Wang et al. (2022) found:
   - a decoder-only model trained on next-token prediction generalised best to new tasks straight after pre-training;
   - an encoder-decoder trained to fill in masks did best only after extra supervised training on many tasks;
   - either can be adapted to the other at modest cost.

   At large scale the field settled on decoder-only for its simplicity and predictable scaling.
4. **Everything you'd build on is decoder-only.** That includes every open model you might continue training (Llama,
   Qwen, OLMo, SmolLM), fast generation with a key–value cache, and the tools (TRL, MLX, llama.cpp,
   lm-evaluation-harness).

**Where the other families are better for you:**

- **Encoder-only models** are better for search embeddings and classification (for example "is this abstract about
  colocalisation?"). Use an existing biomedical encoder or embedding model rather than training one.
- **Encoder-decoder models** suit fixed transformations with many paired examples, such as English → another language
  in narration or paper → summary. Modern decoder-only models do these well too.

## Your run's settings explained

The run you started on 29 September (`--dataset datasets\bio_v2 --d-model 384 --layers 8`) printed a `model{...}`
and a `train{...}` block. Here is what each value means and where it came from.

**model{...}: the shape of the network**

| Setting | Value | What it means | How it was chosen |
|----------------|-------------|-------------------------------|----------------------------------------|
| vocab_size | 8,192 | Number of distinct tokens | Fixed by the dataset's tokenizer (data_prep.py, automatic size) |
| ctx | 512 | Tokens the model sees at once (the context window) | 90th percentile of document length rounded up to a power of 2, capped at 512 for widths up to 512 |
| d_model | 384 | Width of every token vector | Your `--d-model 384` |
| n_layers | 8 | Number of transformer blocks | Your `--layers 8` |
| n_heads | 6 | Attention heads per block | $384 / 64$: heads of width 64 |
| n_kv_heads | 6 | Key/value heads | Equal to n_heads (GQA starts at 8 heads, with 4 query heads per key/value head) |
| ffn_hidden | 1,024 | Width inside the SwiGLU feed-forward part | $8 \times 384 / 3 = 1{,}024$ |
| rope_base | 10,000 | Base of RoPE's rotation frequencies | Standard value; larger bases suit longer contexts |
| dropout | 0 | Share of activations zeroed at random during training | Off, because each token is seen at most 4 times; dropout only helps with heavy repetition |
| norm_eps | 1e-6 | Small constant inside RMSNorm | Prevents division by zero |
| tie_embeddings | true | The output layer shares the embedding matrix | Default; saves 3,145,728 parameters |

**train{...}: how it learns**

| Setting | Value | What it means | How it was chosen |
|----------------|-------------|-------------------------------|----------------------------------------|
| steps | 11,870 | Number of parameter updates | planned_train_tokens ÷ tokens_per_step, rounded down |
| micro_batch | 32 | Windows processed together on the GPU | The largest that fits the measured GPU memory |
| grad_accum | 4 | Micro-batches whose gradients are summed before one update | tokens_per_step ÷ (micro_batch × ctx) |
| tokens_per_step | 65,536 | Tokens behind each update: $32 \times 4 \times 512$ | About $16\sqrt{N}$, rounded to a power of 2 |
| lr | 0.001017 | Peak learning rate | Interpolated on a log-log scale from published models (Pythia-70M $10^{-3}$, GPT-3 125M $6\times10^{-4}$, ...) at 17.3M parameters |
| min_lr | 0.0001017 | Learning rate at the very end | 10% of the peak (`--min-lr-ratio 0.1`) |
| warmup_steps | 237 | Steps of linear ramp from 0 to the peak | 2% of the steps |
| schedule | wsd | Warm-up, stable, decay | Default. It allows a run to be extended, and matches cosine schedules at the end |
| decay_frac | 0.2 | Share of steps in the final decay | The last 2,374 steps, from step 9,496 |
| decay_shape | 1-sqrt | Shape of the decay: $1-\sqrt{t}$ | The best-performing shape in Hägele et al. (2024) |
| weight_decay | 0.1 | Each step shrinks weight matrices slightly towards 0 | Standard; not applied to normalisation gains |
| betas | 0.9, 0.95 | Adam's averaging factors for the gradient and the squared gradient | Standard for language models; 0.95 reacts faster to loss spikes than 0.999 |
| grad_clip | 1.0 | If the gradient's overall length exceeds 1, it is scaled down to 1 | Prevents rare huge updates |
| planned_train_tokens | 777,912,320 | Tokens read in total | 4 passes (`--max-epochs 4`) over 194,493,906 unique training tokens |
| mix | null | Weights for sampling between sources | None given, so each source is sampled in proportion to its size |

**What this means for your run:**

- **Data per parameter:** 45 tokens per parameter (778M ÷ 17.3M), or 55 per non-embedding parameter. That's more
  than Chinchilla's 20, which is fine: it trades a little compute for a better small model.
- **Repetition:** the 4-epoch cap follows Muennighoff et al. (2023). Up to about 4 passes, repeated data is almost as
  good as fresh data (Section 9.4).
- **Time:** at 4,826 tokens per second the run takes about 45 hours (Chapter 10 explains why the GPU is slow).
- **Settings file:** your settings are saved inside every checkpoint and written to experiments.csv, so they are
  never lost.

## How the planner chooses a model

`python tiny_gpt.py --dataset datasets\bio_v2 --plan` prints the plan and its reasons without training. The rules:

1. **Training tokens:** $D = \text{max\_epochs} \times \text{unique tokens}$, unless you give `--train-tokens`,
   `--epochs` or `--steps`.
2. **Model size:** the largest shape on a ladder of widths (128 to 2,048, with depth growing with width) whose
   parameter count satisfies $D \geq r N$, where $r$ is `--tokens-per-param` (default 20). Your `--d-model` and
   `--layers` override this.
3. **Batch, learning rate, warm-up and dropout** follow from $N$ and the number of passes (the table above).
4. **Time budget:** `--time-budget-hours 10` binary-searches the ladder for the largest model that still reads
   $r$ tokens per parameter in that time. Each size is timed briefly on the GPU, but a laptop GPU runs faster for a
   few seconds than it does for hours (power and heat limits; 1.7 times on this RTX 3070). So the planner uses the
   speed your last training run sustained, converted to floating-point operations per second so that it carries
   over to other sizes; before the first run it slows the short measurement down by 1.6.

## Hands-on exercises

Each takes minutes to a few hours. Run them only when no other training is running, give each a new `--name` so
nothing is overwritten, and compare the results with compare_models.py.

1. **Plan versus time:** `python tiny_gpt.py --dataset datasets\bio_v2 --time-budget-hours 3 --plan`. What model does
   3 hours buy?
2. **Learning rate:** `--lr 3e-3 --name lr_high --stop-after-steps 1500`, then `--lr 3e-4 --name lr_low`. The first
   can be unstable; the second learns slowly.
3. **Size:** `--d-model 256 --layers 6 --name small`. The loss is higher but each step is faster: a scaling law in
   miniature.
4. **Context:** `--ctx 128 --name short_ctx`. The samples lose coherence over long passages.
5. **Stability:** `--z-loss 0 --name no_zloss`. Compare the loss spikes and the final bits per byte.
6. **Temperature:** `python tiny_gpt.py --generate "Genetic drift is" --temperature 0.3`, then `1.2`.
7. **Overfitting:** build a dataset from one folder, for example
   `python data_prep.py --out datasets\scripts_only --md-dir scripts=$HOME\ai_training_data\scripts`. Then
   train with `--epochs 30 --name overfit`. The training loss keeps falling while the validation loss turns up.

# Your training data

## Folder layout

`$HOME\ai_training_data` is organised by type. With `--text-root`, each subfolder becomes a named
**source**, with its own validation loss and bits per byte in every metrics line.

| Folder | Contents | Used as text? |
|-------------|-----------------------------------------------|----------------------------------------|
| books | Long PDFs (150 pages or more) converted to Markdown | Yes, source "books" |
| articles | Papers and short documents | Yes, "articles" |
| scripts | Narration scripts | Yes, "scripts" |
| notes | Research notes and Markdown files | Yes, "notes" |
| slides | Converted PowerPoint files | Yes, "slides" |
| tables | Spreadsheets, CSV and GWAS summaries (as text) | Yes, "tables" |
| codes | R, Python and other code (as text; never run) | Yes, "codes" |
| synthetic | AI-written text (GLM and DeepSeek genetics notes) | Yes, "synthetic". Rename it to `_synthetic` to leave it out |
| images | Pictures for future multimodal work | No |
| wikipedia | English Wikipedia (Hugging Face Arrow format, 6.4M articles) | Through `--wiki-dir` |
| web | FineWeb-Edu (Parquet) and peS2o (JSON Lines) | Through `--parquet` and `--jsonl` |

**Rules for this folder:**

- Folders whose names start with `_` or `.` are skipped.
- Every file move of 29 September is listed in `_moves_2026-09-29.tsv` (old path, new path), so the reorganisation
  can be undone.
- The running bio_v2 dataset was built before the reorganisation and is unaffected. Its exact file list can't be
  rebuilt from the new layout without undoing the moves.

## Converting documents

convert_to_markdown.py turns documents into Markdown, the only format data_prep.py reads from folders:

```
python convert_to_markdown.py "D:\Papers" --out $HOME\ai_training_data `
    --by-type --types pdf --name-prefix Papers__ --recursive --workers 2
```

| Option | Effect |
|------------------------------|----------------------------------------------------------------------|
| `--by-type` | Sorts output into books, articles, slides, tables and codes |
| `--types pdf` | Converts only these file types from a folder (for example `pdf,docx`) |
| `--book-pages 150` | PDFs with at least this many pages count as books |
| `--name-prefix Papers__` | Avoids name clashes between collections |
| `--workers 2` | Converts files in parallel, each in its own process with a time limit |
| `--ocr` | Reads scanned PDFs (no text layer) with Tesseract OCR, a few seconds per page; formulas in old papers come out garbled |
| (default) | Skips files already converted, so an interrupted batch can simply be restarted |

- **Log:** every batch writes a `conversion_report_*.tsv` listing what was converted, skipped or failed, and why.
- **Tools:** PDFs go through pymupdf4llm, which keeps headings, lists and tables. Word, PowerPoint and Excel files are
  read directly from their XML, without Office.
- **Plot pages:** a page whose drawing instructions exceed 200 KB is a plot made of tens of thousands of shapes.
  pymupdf4llm can spend about 30 seconds on each such page, so these pages are read as plain text instead (captions and
  axis labels are kept). In BDA3, 24 of 677 pages are like this.
- **Page furniture removed:** the converter removes running headers and footers, page numbers, and notices printed on
  every page.
  - A line counts as furniture when it keeps recurring at the top or bottom of pages, ignoring page numbers and case.
    Book running heads are in capitals and are recognised after 2 repeats. Other lines need 2+ words and 10% of pages.
  - Ordinary words ("where") and headings that repeat in a few chapters are kept.
  - In BDA3 this removed 1,779 lines, including the notice "This electronic edition is for non-commercial purposes
    only" from 670 pages. In Nature papers it removes footers such as "Nature | Vol 631 | 18 July 2024 | 585".

## English only

data_prep.py keeps English documents only (`--language en`, the default). The test for each document:

1. **Function words:** if at least 10% of its words are English function words (*the*, *of*, *and*, *is* and so on),
   it's kept.
2. **Non-English letters:** otherwise it's kept only if at most 0.5% of its letters are non-ASCII letters (*ą*, *č*,
   *ė*...).
3. **Short documents:** under 50 words are kept, because the test is unreliable there.

**Tested on your files:** it caught all 145 narrations in another language and wrongly rejected none of 162 English ones.
Rejected documents appear in the report as `not_english`. Use `--language any` to switch the filter off.

## Keyword selection

Wikipedia, FineWeb-Edu and peS2o are far too broad, so data_prep.py keeps only on-topic documents. Your own folders
are never keyword-filtered.

**The keyword list:**

- keywords_biology.txt has 2,385 terms. It covers molecular and cell biology, genetics and genomics, evolution and
  phylogenetics, statistics and mathematics, machine learning, medicine and specific diseases, drugs and other
  chemicals, nutrition and metabolism, enzymes and other proteins, gene symbols, species and microbes, molecular
  structure and 3D modelling and rendering, and the names of well-known scientists.
- **History:** 608 terms until 30 September 2026. Then mathematics, statistics and more genetics themes were added
  (1,006), then diseases, chemistry, genes, proteins and 3D modelling (1,686), then nutrition, species, microbes,
  phylogenetics, more diseases and scientists' names (2,385).
- **Everyday words are left out or used only in phrases.** Examples: "stroke" appears only as "ischaemic stroke";
  "Godot" only as "Godot engine" (not the play); "Falconer" only as "Douglas Falconer". Common surnames (Fisher,
  Wright, Snow) appear only as full names ("Ronald Fisher", "Sewall Wright", "John Snow").
- **No term is covered by another one.** "polymerase\*" was dropped because "polymer\*" already matches it: two
  terms matching the same word would count as two different terms and let a document pass the distinct-term rule
  too easily.
- Terms starting with `=` are case-sensitive, for gene symbols such as `=BRCA1` that would otherwise match ordinary
  words.
- `*` is a wildcard at either end of a word: `*mab` finds antibody drug names, `immuno*` finds all immuno- words.

**How a document is scored:**

- It must contain at least `--keyword-min-hits` matches (default 3) from at least `--keyword-min-distinct` different
  terms (default 2; use 3 for stricter selection).
- A match in the title counts three times (`--title-weight 3`).
- `--wiki-select top` ranks articles by score and keeps the best `--wiki-max-docs`. The best 1.5 times that many
  are read, and the cut to exactly `--wiki-max-docs` is made after cleaning, filtering and boilerplate removal.

**Why the keywords are not built into the tokenizer.** Giving every keyword its own token sounds like it
would save tokens, and so training time. It doesn't. The test: 16,384-token tokenizers were trained on the same 86
million characters (PubMed Central papers and FineWeb-Edu), then each encoded 27 million other characters.

| Tokenizer | Tokens | Keyword words that are one token |
|---|---|---|
| As data_prep.py builds it | 6,686,742 | 32% |
| Keywords learned as whole words (each keyword word added 300 times to the tokenizer's training text) | 6,840,465 (+2.3%) | 99% |
| Keywords forced in as fixed symbols | 7,345,794 (+9.9%) | (split from their space) |

- **The most it could ever save is 0.4%.** Keywords are 7.6% of the words, and most already take only one or two
  tokens. Even if every keyword cost exactly one token and nothing else changed, the text would shrink by 0.4%.
- **In practice it costs more than it saves.** The whole-word keyword tokens take about 2,000 of the 16,384
  vocabulary slots from pieces that common words need, so everything else gets longer.
- **Fixed symbols are worse still:** they ignore word boundaries ("general" became "gene" + "ral").
- **The plain tokenizer already learns your field:** it is trained on your own training split, so frequent terms
  ("genetics", "proteins", "diets") are already single tokens, and rare ones ("Tajima") cost a few pieces.

**Why keywords, and not "text that looks like my Markdown":** your Markdown doesn't cover every topic you care about.
Importance resampling towards it would make the corpus narrower. Keyword selection stays.

**Speed, now 11 times faster:**

- **Prefilter:** a C++ regular-expression engine (RE2, inside pyarrow) first discards documents with too few
  keyword matches. It scans all 6.4 million Wikipedia articles in about 2 minutes on 4 threads. Its count is never
  below the exact count: terms whose matches could overlap ("genetic drift" and "genetic\*") are counted by
  separate patterns (still five for the 2,385 terms). On 100,000 FineWeb-Edu and peS2o documents it rejected none
  that the exact count keeps. For the 2,385-term list, every term was also written out in nine spellings (plural,
  capitals, punctuation around it; 21,438 test texts): the prefilter never counted fewer matches than the exact check.
- **Counting:** the survivors are counted with dictionary lookups instead of one large regular expression. Speed rose
  from 0.5 to 5.7 MB of text per second per thread.
- **Accuracy:** it made the same decision as the old method on 2,979 of 3,000 test documents (99.3%). The differences
  are overlapping phrases, which the new method counts more consistently.
- **Wildcards:** each word is checked against the wildcard terms only if its first (or last) three letters could
  start (or end) one, which halved the counting time.

**Using all CPU cores (since 30 September).** One process reads the files and runs the C++ keyword prefilter, with
the next batches scored in background threads. The Python checks on each document go to worker processes (default:
all cores but two, `--workers`). These are cleaning, the English test, the exact keyword count, and the duplicate
and boilerplate keys. The per-document part of pass 2 and of the leakage audit go to workers as well.

- **Same output:** results come back in the original order, so the dataset is byte-identical for any number of
  workers; a test checks this.
- **Speed:** on FineWeb-Edu the reading pass ran twice as fast with 6 workers while another build shared the
  machine, and pass 2 ran 3.2 times as fast. Expect about 3 times overall on an idle machine.
- **Near-duplicates on large corpora:** detection now runs up to 2 million documents (`--near-dup-max-docs`,
  formerly 400,000). Above the old limit it had been switched off silently.

**Further speed-ups, if needed:**

- `--parquet-max-docs 150000 --parquet-select top` ranks a Parquet source like Wikipedia. Every record is scored in
  C++ first, and only the best-matching documents get the slower checks: faster, and a more on-topic selection.
  Without `--parquet-select top`, and always for JSON Lines (`--jsonl-max-docs`), reading simply stops once that many
  documents have been kept.
- `--threads 6` gives pyarrow and the tokenizer more cores.
- `--scan-only` runs the filters and writes the report without tokenising, so you can tune thresholds first.
- `--sample-fraction 0.25` tests settings on a quarter of the data.
- Keep the list focused: very common terms such as *cell* or *model* add matches but not selectivity.

## How big will the dataset be? estimate_dataset.py

```
python estimate_dataset.py                 # everything in ai_training_data, with the usual settings
python estimate_dataset.py --sample 0.05   # a larger sample: slower, more precise
```

It predicts what data_prep.py would keep, without building anything: documents, characters and tokens per source,
then the model sizes those tokens support and how long they take on this PC.

- **Same rules:** it takes data_prep.py's options (paste a data_prep command after `estimate_dataset.py`). A
  sample of every source (2% of web records, at least 300 files of each folder) goes through data_prep's real
  checks: cleaning, English test, length, quality score and keywords. The kept share is scaled up.
- **Tokens:** characters per token come from a tokenizer trained on the kept sample with data_prep's settings.
- **Model sizes:** the planner's rules (at most 4 epochs, at least 20 tokens per parameter), for the data alone
  and for time budgets (`--hours 12,24,48,72`), each with the exact command. Note that in tiny_gpt.py
  `--train-tokens` sets how much is trained; `--steps` on its own only splits the data's token budget (4 epochs)
  into that many steps, so `--steps 15500` with a 70M shape still trains 10 billion tokens. Training hours use the speed of your last run in
  experiments.csv, as floating-point operations per second, so it carries over to other sizes.
- **Ranking column:** the Chinchilla fit of loss against parameters $N$ and tokens $D$ (Hoffmann et al. 2022),
  $L = 1.69 + 406.4/N^{0.34} + 410.7/D^{0.28}$. Its values belong to their data, but it ranks the options.
- **Not modelled:** duplicates and repeated boilerplate lines (a few per cent).

## Cleaning, deduplication, split and audit

1. **Cleaning:** Unicode normalisation, control characters removed, files with too many replacement characters
   (broken encodings) dropped.
2. **Exact duplicates** are removed. **Near-duplicates** are found with MinHash and locality-sensitive hashing
   (Section 9.8) and kept together on one side of the split.
3. **Boilerplate lines** that recur in 20 or more documents (headers, licences, navigation) are stripped.
   Headings are never counted: "## Materials and methods" repeats in 40% of PubMed Central papers, but it is
   structure, not boilerplate (Markdown headings and short ALL-CAPS lines are exempt).
4. **The split is by document**, with no document split across training and validation. It is capped so no single
   group dominates validation.
5. **The tokenizer** is trained on a sample. `--tokenizer-weights books=3,articles=3` gives those sources more
   influence over the vocabulary.
6. **Leakage audit:** the share of validation word 13-grams that also occur in training data. A high share means the
   validation loss is optimistic.

Every decision is counted in `report.md` inside the dataset folder, per source and per reason.

## Downloaded data, and the recommended next dataset

Downloaded on 29 September into `ai_training_data\web` (both are ODC-By licensed, which allows training):

| Dataset | Downloaded | What it is |
|----------------------|--------------------------------------|----------------------------------------|
| FineWeb-Edu, sample-10BT | 2 files, 4.3 GB (`000_00000.parquet`, `001_00000.parquet`) | Educational web pages, each with a quality score from 0 to 5 |
| peS2o v2 | 1 of 20 shards, 1.6 GB (`train-00000-of-00020.json.gz`) | Full text of open-access scientific papers, all fields |

A new dataset that uses everything (run it when the current training has finished, because it keeps the CPU busy for
a few hours):

```
$D = "$HOME\ai_training_data"
python data_prep.py --out datasets\bio_v3 --text-root $D `
    --wiki-dir $D\wikipedia --wiki-max-docs 100000 `
    --parquet "$D\web\fineweb-edu\sample\10BT\*.parquet" `
    --parquet-name fineweb --min-score 3 `
    --jsonl "$D\web\pes2o\data\v2\*.json.gz" --jsonl-name pes2o `
    --include-keywords keywords_biology.txt --keyword-min-distinct 3 `
    --tokenizer-weights books=3,articles=3
```

- **Scan first:** add `--scan-only` to see the kept counts before committing hours.
- **Keyword filter:** it applies to wiki, fineweb and pes2o, but not to your folders.
- **More data later:** the other FineWeb-Edu and peS2o files can be added to the same folders. The glob patterns
  pick them up.

## Open-access papers from PubMed Central: download_pmc.py

```
python download_pmc.py --out $HOME\ai_training_data\pmc --max-papers 100000
```

1. **List:** a Europe PMC search for open-access papers with full text that mention genetic\*, genom\*, allele\* or
   "gene expression" (2.95 million match), sorted by citations, keeps the first `--max-papers`.
2. **Fetch:** the publishers' structured XML (JATS) comes from NCBI PubMed Central, 100 papers per request, at most
   3 requests per second (NCBI's limit without an API key). 100,000 papers take about an hour. NCBI asks that
   large jobs run at weekends or between 2 am and 10 am UK time.
3. **Convert:** title, abstract and body become Markdown: section headings, paragraphs, lists, equations as LaTeX,
   tables as Markdown tables, figure and table captions. References, acknowledgements, funding, competing
   interests, author contributions, contact details and supplementary files are dropped; email addresses become
   `[email]`.

- **Why XML, not PDF:** the XML has the real section structure and equations, with no page headers, columns or
  hyphenation to repair.
- **Output:** `ai_training_data\pmc`, a separate source for `--text-root`, with its own validation loss.
  `ai_training_data\_pmc_meta` (skipped by `--text-root`) holds the paper list and `report.tsv`: status, length,
  journal, year, citations and licence of every paper.
- **Stop and restart:** Ctrl+C, then the same command; finished papers are skipped.
- **Licences:** most papers are CC BY; some are CC BY-NC or NIH author manuscripts (text mining allowed). Fine for a
  personal model; check `report.tsv` before publishing a model trained on them.

## Lecture transcripts from YouTube: youtube_transcripts.py

```
python youtube_transcripts.py @statquest @mitocw --max-videos 300
python youtube_transcripts.py --channels-file youtube_channels.txt
python youtube_transcripts.py --channels-file youtube_channels.txt --whisper   # later: the videos without captions
```

A channel can be given as `@handle`, channel name, channel or playlist URL, or single video. For each video, newest
first, the transcript comes from:

1. **English captions written by the creator**;
2. **YouTube's automatic captions**, if they are punctuated (recent videos usually are);
3. **Whisper** speech recognition of the audio track, only with `--whisper`. Without it these videos are marked
   `queued_whisper`, and a later `--whisper` run transcribes them. Whisper (distil-large-v3) uses the GPU only when
   no other job does, and moves to the CPU at once if a training run starts. On the CPU (small.en, 4 threads) it
   runs about 4 times faster than real time.

- **On topic only:** a transcript is kept if it matches keywords_biology.txt at least 10 times, with 3 different
  terms and at least 4 matches per 1,000 words. The density rule matters because channels repeat their theme in
  every intro: every StatQuest video mentions "statistics", "machine learning" and "neural networks", which alone
  passed the looser web-text test. Whisper is used only for videos whose title or description mentions a keyword.
- **Nothing large is stored:** no video is downloaded; Whisper's audio track is deleted as soon as its transcript is
  written.
- **Output:** `ai_training_data\lectures\<channel>\<date> <title> [<id>].md` (a separate source for `--text-root`)
  and `ai_training_data\_lectures_meta\<channel>_report.tsv`: status (ok, off_topic, queued_whisper, not_english,
  too_short...), transcript source, duration, words, keyword matches and licence of every video.
- **Reruns** skip finished videos, so running the same command later adds new uploads.
- **If YouTube asks to "confirm you're not a bot",** the run stops. Wait a few hours, or add
  `--cookies-from-browser firefox` (this uses your own YouTube login).
- **Rights:** YouTube's terms restrict downloading content. The licence column shows Creative Commons videos, and
  `--creative-commons-only` keeps only those.

## More data worth getting

| Source | What | Licence | How it fits |
|----------------------|----------------------------|----------------|----------------------------------|
| peS2o v2, remaining 19 shards | About 30 GB of open papers | ODC-By | Ready: same folder, `--jsonl` |
| FineWeb-Edu, more files | Each file is about 2.15 GB | ODC-By | Ready: same folder, `--parquet` |
| PubMed Central Open Access subset | Several million full-text biomedical articles in XML | Per article (CC BY, CC BY-NC, ...) | Done for 100,000 papers with download_pmc.py (above); raise `--max-papers` for more |
| PubMed baseline | More than 35 million citations, most with abstracts | NLM terms (free) | Dense biomedical vocabulary; needs the same XML step |
| Europe PMC, bioRxiv, medRxiv | Open full text and preprints | Mostly CC BY | Bulk services; the PDF route already works |
| OpenStax textbooks | Biology 2e, Microbiology, Anatomy and Physiology, Introductory Statistics | CC BY 4.0 | Download the PDFs, then convert_to_markdown `--by-type` (they land in books) |
| LibreTexts | Biology and statistics textbooks | Mostly CC BY-NC-SA | Web pages, then HTML conversion |
| OpenWebMath | 14.7B tokens of mathematics | ODC-By | Parquet, `--parquet-name math` |
| The Stack v2 (R and Python subsets) | Source code | Permissive licences only | Parquet; needs a Hugging Face login and terms acceptance |
| Cosmopedia | 25B tokens of synthetic textbooks | Apache 2.0 | Parquet; treat as synthetic |

**For evaluation only, never for training:** MMLU, GPQA, MedQA, PubMedQA, MedMCQA, BioASQ, SciQ and ARC (Section
6.5).

**Paywalled journals:** use your institution's lawful access and the publishers' text-and-data-mining routes. UK law (CDPA
section 29A) allows text-and-data-mining copies for non-commercial research.

## GWAS summary statistics

- **Reading them is easy and fast.** convert_to_markdown.py reads bgzipped files directly (bgzip is ordinary
  multi-part gzip), with pyarrow's multi-threaded reader. It recognises the usual column names and handles a
  3-million-variant file in about 15 seconds. A 300 to 800 MB file takes about 1 to 2 minutes, so a thousand files is
  roughly a day of CPU time.
- **Training on the rows is not useful.** A file with 10 million rows would be about 150 million tokens of numbers
  that teach a language model nothing.
- **What the converter writes instead** is a short text summary: variant count, genomic inflation $\lambda_{GC}$,
  genome-wide significant loci after 1 Mb clumping with their top variants and effects, and counts per chromosome.
  That's about 1 to 3 thousand tokens per study.
- **Better uses of the numbers:**
  - as data for tools the model calls (R or Python code that queries the files);
  - as material for question-and-answer pairs ("Which loci pass $5\times10^{-8}$ for trait X?").
- **Watch for:**
  - varying column names (the GWAS-SSF standard helps);
  - genome build (GRCh37 versus GRCh38);
  - which allele the effect refers to;
  - licence terms, for example UK Biobank-derived results.

## Pictures (PNG) in the future

A text-only model can't use pixels. The `images` folder is kept out of text training. There are three routes, in
order of practicality:

1. **Pictures to text, now.** Use figure captions (the PDF converter keeps them) and descriptions of each figure
   written by a vision-language model on the Mac Studio. The text then trains tinyGPT as usual. For charts, keep the
   underlying data table instead.
2. **Continuous image features, the LLaVA route.** A pretrained image encoder (a ViT such as SigLIP) turns each image
   into patch vectors. A small projection layer maps them into the model's width, and the model reads them like
   tokens.
   - Training starts with the projection only, then adds LoRA.
   - It needs hundreds of thousands of image–caption pairs; PMC-OA figure–caption collections exist for biomedicine.
3. **Discrete image tokens, the Chameleon route.** A VQ-VAE or VQGAN turns an image into, for example, 256 codes from
   an 8,192-entry codebook. These codes are added to the vocabulary, and the model reads and writes them like words.
   It's elegant, but costly to train well.

# Training a run

## Reading the log

Every 200 steps tiny_gpt.py prints a metrics line and a sample, and writes both to log.txt. An example (one line in
the log, split here to fit the page):

```
step 400/11870 | train 4.912 | val 4.470 @400 [md 4.51 wiki 4.43] | ppl 87.36
  | bpb 1.372 | lr 1.02e-03 | 4,826 tok/s | elapsed 1:31:10 | ETA 43h 58m | NEW BEST
```

| Field | Meaning |
|------------------|----------------------------------------------------------------------------------|
| train | Average training loss since the last line |
| val @400 | Validation loss measured fresh at step 400 on fixed windows, then per source in brackets |
| ppl | Validation perplexity, $e^{\text{val}}$ |
| bpb | Validation bits per byte: comparable across tokenizers (Section 9.1) |
| lr | Current learning rate |
| tok/s | Training throughput |
| elapsed | Time since this session started. After `--resume`, `(whole run 7:45:12)` adds the earlier sessions |
| NEW BEST | Shown only when validation loss improves; the best checkpoint is then saved |

- **Samples:** after each line, a short sample continues "Genetic drift is", so you can watch the text improve.
- **End of a run:** the last line gives this session's time and the whole run's: total time, how much of it was
  training, and the energy, CO2e and cost, for example `this session 2:10:03 | whole run: 26:41:10 in total,
  25:58:47 of it training, 9.81 kWh, 1,226 g CO2e, £2.61`.
- **Heartbeat:** a progress line starting with `...` appears every 2 minutes between evaluations, so a slow step never
  looks like a hang. It shows the training loss since the previous line, speed and time to the next validation; it
  includes no validation. The run header explains this once, so the line itself stays short.
- **Why the sample always starts with "Genetic drift is":** the prompt (`--sample-prompt`) and the random seed are
  fixed on purpose, so samples from different steps differ only because the model has learnt more. Choose another
  prompt with `--sample-prompt "Linkage disequilibrium is"` when starting or resuming a run.

## Checkpoints, pausing and resuming

- **tinyGPT.pt** is the full training state. It is saved every 1,000 steps, every 30 minutes (`--save-minutes`) and
  on Ctrl+C. A shutdown or crash therefore loses at most about half an hour. It holds:
  - the weights;
  - Adam's averages;
  - the step number;
  - the random-number states;
  - the tokenizer;
  - the settings.
- **tinyGPT_best.pt** holds the weights of the best validation loss, for inference only.
- `python tiny_gpt.py --resume` continues exactly where the run stopped. The learning-rate schedule is stateless, so
  resuming is exact.
- **No training state yet?** If the PC stopped before tinyGPT.pt was first written, `--resume` continues from
  tinyGPT_best.pt at its step. Weights, schedule position and history are kept. Adam's averages start again from zero,
  so the learning rate is ramped back up over 50 steps.
- `--resume --steps 20000` extends a finished run. `--init-from other.pt` starts a new run from another checkpoint's
  weights: that's continued pre-training on new data.
- Existing files are never overwritten unless you pass `--overwrite`. Saves are atomic: written to a temporary file,
  then renamed.

## More data later: training further from the first model

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

## The experiments table

`experiments.csv`, next to the checkpoints, gets one row for every training run (when it ends or is stopped),
benchmark, comparison and fine-tune.

- **Columns include:** time, event, name, status, step, parameters, shape, context, vocabulary, tokens seen, learning
  rate, z-loss, dataset, best and last validation loss, bits per byte, perplexity, benchmark accuracy, throughput,
  training hours (`train_hours`, without evaluations and saving), total hours of the run over all sessions
  (`run_hours`) and checkpoint.
- **Opening it:** it opens directly in Excel. The scripts only ever append, and nothing deletes it.
- **Why keep it:** yes, storing the parameters in a table is useful. It is the only reliable way to know, months
  later, which settings gave which result. Add your own comments in the *notes* column.

## Energy, carbon and cost

Every metrics line shows the electricity the run has used so far, its carbon footprint and what it cost, for
example `0.42 kWh, 52 g CO2e, £0.11`. The last line of a run splits the energy into GPU, CPU and memory, and
experiments.csv records `energy_kwh`, `co2e_kg` and `cost_gbp`. The totals are stored in the checkpoint, so they
add up across `--resume` sessions.

- **GPU: measured.** nvidia-smi reports the graphics card's power draw every 5 seconds, and the energy is the
  area under that curve.
- **CPU and memory: estimated.** The CPU's rated power (`--cpu-watts`, 35 W for this laptop's i7-11370H) times the
  run's share of all CPU time, plus 0.375 W per GB of memory the run holds.
- **Carbon:** energy times the grid's carbon intensity (`--grid-intensity`, grams CO2e per kWh). The default, 125,
  is the British grid's average from October 2025 to September 2026 (National Grid ESO Carbon Intensity API).
  Monthly averages ranged from 97 to 152, and the grid is cleaner on windy nights than on still winter evenings.
- **Cost:** each half-hour's energy times that half-hour's unit rate, VAT included. By default the rates are Flexible
  Octopus for region H (Southern Electric; set yours with `--region`), read from Octopus's public price list (no account
  or personal data): 26.4 p/kWh on 30 September 2026. On Agile (`--tariff AGILE-24-10-01`) or Go
  (`--tariff GO-VAR-22-10-14`) the price changes through the day, so a run overnight costs much less: on Go,
  8.6 p/kWh between 00:30 and 05:30. A fixed price works too: `--tariff 24.5`. The standing charge is not counted,
  because it is paid anyway.
- **Not included:** the screen, fans, charger losses, building the dataset, and the carbon of making the laptop.

For scale: a 12-hour run at about 60 W in total uses about 0.7 kWh: about 90 g CO2e and 19p on Flexible Octopus,
about as much electricity as boiling a full kettle four times (about 0.17 kWh each).

## Exporting a model

`python tiny_gpt.py --export tinyGPT_v1` writes a folder with the following files. Every script accepts that folder
wherever it accepts a checkpoint.

| File | Contents |
|------------------------------|----------------------------------------------------------------------|
| `model.safetensors` | The weights (a safe format with no code inside) |
| `config.json` | The model settings |
| `tokenizer.model` | The SentencePiece tokenizer used to train the model |

# Evaluating a model

## Loss, perplexity and bits per byte

- **Validation loss** is measured on the same fixed windows every time, so values are comparable along a run.
- **Per-source values** show which kind of text the model handles well (for example books versus Wikipedia).
- **Loss per token depends on the tokenizer**: a bigger vocabulary packs more text into each token. **Bits per
  byte** divides by the raw text length instead, so it is the right number for comparing models with different
  tokenizers or vocabulary sizes.

## The fact benchmark

`probes_biology.tsv` has **336 questions in 16 categories**: 24 each in population, quantitative and statistical
genetics and in statistics, and 20 in each other category.

**Format:** each line is a sentence start, the correct continuation and three wrong ones, for example: "Cells that break
down bone matrix during remodelling are called" → *osteoclasts* (versus *osteoblasts*, *osteocytes*, *chondrocytes*).

**Why it was rebuilt:** the first version (171 items, kept in `backup_v3_2026-09-29`) was too easy.

- Its wrong answers often came from another topic (*germline mutations pass to the → offspring* against *liver /
  environment / neighbours*), so knowing the topic alone gave about 35%.
- Many items finished a stock phrase ("Benjamini and → Hochberg"), which small models learn long before the fact
  itself.
- Open models of 135M and 360M parameters scored 81% and 88% on it.

**How it was built:**

1. Four writers drafted the items. Every wrong answer is a near miss from the same class as the right one:
   - statistics against other statistics, and cell types against neighbouring cell types;
   - numbers against nearby numbers;
   - directions against the other directions.
2. Two independent reviewers checked all 336 items for:
   - wrong or disputable answers;
   - a second defensible option;
   - grammar that gives the answer away;
   - stock phrases and weak distractors.

   They agreed that no answer was wrong, and 26 items were rewritten.
3. **Difficulty comes from how models actually score** (`calibrate_probes.py`, below), not from the writers' guesses.

**Scoring:**

- A question is correct when the model gives the right continuation a higher log-probability *per character* than
  every wrong one. This is the length-normalised multiple-choice method used by the standard evaluation harness.
- Chance is 25%, but the **floor** is the number to beat. The floor is the accuracy when each question's words are
  shuffled: the topic words stay, the sentence is destroyed. It is averaged over 3 shuffles and reported next to every
  score:
  - in `--benchmark` output;
  - in the training history;
  - in experiments.csv;
  - on the dashboard, where a category's bar is green only when it beats its own floor.

  Accuracy above the floor needs the sentence itself.
- Results are shown per category with **Wilson 95% intervals** (Section 9.5), and every item's result is saved as a
  CSV next to the checkpoint.

**What it measures:** factual reliability, not a hallucination rate. With 336 items it can detect differences of
about 10 percentage points (Section 9.5).

**Difficulty labels: `calibrate_probes.py`.** Small open models with the Llama design (SmolLM2 135M, 360M and 1.7B;
TinyLlama 1.1B) are run on the benchmark by tinyGPT's own model code, with no extra packages. Their Hugging Face
weights need only a reordering of the attention rows for tinyGPT's RoPE layout; a unit test checks the conversion.

- **Labels:** an item is *easy* if at least 3/4 of these reference models answer it correctly, *hard* if at most 1/4
  do, and *medium* otherwise.
- **No self-grading:** tinyGPT itself is never used, so its scores per difficulty stay unbiased.
- **Suspect items:** items that every reference model gets wrong in the same way are listed for a manual check.
- **Report:** `probes_biology_calibration.md`.
- **Running it:** `python calibrate_probes.py` runs on the CPU (models that don't fit in memory are skipped). Once
  no training is running, `python calibrate_probes.py --device cuda` adds the 1.7B model. Scores are cached, so
  reruns are quick.

**Results of the first calibration (29 September 2026):**

| Reference model | Old benchmark (171 items) | New benchmark (336 items) |
|------------------|--------------------|--------------------|
| SmolLM2-135M | 81% | 38% |
| SmolLM2-360M | 88% | 52% |
| TinyLlama-1.1B | not scored | 43% |
| SmolLM2-1.7B | not scored | 68% |

- **Labels:** 144 easy, 44 medium and 148 hard.
- **Hard items:** 48 items were answered wrongly by all four models, all choosing the same distractor. All 48 were
  checked and are correct. Most need a small calculation, and the models took the intuitive wrong option:
  - $N_e = 4 \times 10 \times 90 / 100 = 36$ rather than 100;
  - $1 - 0.95^{20} = 0.64$ rather than 0.05;
  - $\det(2A) = 8\det(A)$ for a 3 × 3 matrix, rather than $2\det(A)$.
- **Headroom:** the benchmark now has room at both ends. The strongest small model scores 68%, and larger or
  better-trained models should do clearly better.

**Adding your own:** add lines in the same tab-separated format. Keep them out of every training set.

## Is A really better than B? compare_models.py

```
python compare_models.py run1_best.pt run2_best.pt --device cpu
```

1. Both models read the same validation documents, decoded to plain text so different tokenizers work.
2. For each document it computes both models' bits per byte.
3. It resamples **documents** 2,000 times to get a 95% interval for the difference. Each model's bits per byte uses
   the bytes its own tokens cover. With fewer than 20 documents it prints the numbers but gives no verdict.

Tokens within a document are strongly correlated, so an interval over tokens would be far too narrow. This is the
same logic as clustered standard errors.

**Reading the result:**

- A negative difference means A is better.
- If the interval excludes 0, the difference is unlikely to be noise.
- The table is printed per source, saved as a CSV next to A, and recorded in experiments.csv.

## The dashboard

`python view_pt.py tinyGPT_best.pt --plot` writes a 4 × 4 dashboard picture, `tinyGPT_best_dashboard.png`.
- **Name:** it's the same every time, so each new dashboard replaces the previous one. The step and the time it was
  drawn are printed in its title.
- **Keeping one:** to keep a particular dashboard, give it a name of its own with `--out`.

Its panels:

- weight statistics per layer;
- the embedding map, with PCA colours and labelled tokens;
- attention patterns;
- next-token predictions for a test sentence;
- output preferences;
- calibration;
- the benchmark per category;
- loss, perplexity and bits-per-byte curves.

Panels that need the model to run are labelled as inference-only. `--html` writes an interactive embedding explorer.

## Towards standard LLM benchmarks, with a biology bias

The next benchmark step is to score tinyGPT and any open model on the same public tests the field reports, using the
same method as EleutherAI's *lm-evaluation-harness*:

- multiple choice by comparing log-likelihoods;
- accuracy and length-normalised accuracy;
- 0-shot and 5-shot;
- standard errors.

| Benchmark | Items (test) | What it tests | Meaningful from about |
|------------------------|------------------------------------|-----------------------|-----------------|
| SciQ | 1,000 | School science, with a support passage | 10–100M parameters |
| ARC-Easy / ARC-Challenge | 2,376 / 1,172 | Grade-school science | 100M / 1B |
| MMLU biology and medicine subsets | college biology 144, high-school biology 310, medical genetics 100, anatomy 135, college medicine 173, clinical knowledge 265, virology 166, nutrition 306 | University-level knowledge | 1B+ (chance is 25%) |
| PubMedQA (labelled) | 1,000 | Yes/no/maybe questions about abstracts | 1B+ |
| MedMCQA (dev) / MedQA-USMLE | 4,183 / 1,273 | Medical exam questions | 7B+ |
| BioASQ | Yearly sets | Biomedical question answering | 7B+ |
| GPQA | 448 (198 "diamond") | Graduate-level, "Google-proof" questions | Frontier models |

**Plan:**

1. Add a `--eval-suite` option that loads these from Hugging Face and scores them like the probes.
2. Report per-subject accuracy with Wilson intervals.
3. Run the 13-gram leakage check against the training data first.

**Expected results:** at 17M parameters, expect chance on MMLU and above chance on SciQ. The suite becomes
informative for comparing open 1B to 8B models on your biology questions.

# Post-training: SFT, preference tuning and RL

## Separate scripts, sharing one model

Post-training lives in **finetune.py**, not in tiny_gpt.py, because almost everything differs from pre-training:

| | Pre-training (tiny_gpt.py) | Post-training (finetune.py) |
|------------------|-----------------------------------------|-----------------------------------------|
| Data | Token shards of raw text | JSON Lines of questions and answers |
| Loss | Every token | Answer tokens only (SFT); pairs (DPO) |
| Learning rate | $10^{-3}$ | $10^{-4}$ (SFT), $10^{-5}$ (DPO) |
| Batches | Random windows | Padded examples |
| Extra model | None | A frozen reference copy (DPO) |

finetune.py imports the model code from tiny_gpt.py, so there's one definition of the network. It writes an ordinary
checkpoint, which every other script can read.

## Supervised fine-tuning

```
python finetune.py sft --base tinyGPT_best.pt --name tinyGPT_sft --device cpu `
    --data finetune_examples\sft_examples.jsonl
```

**Data:** one JSON object per line:

```
{"prompt": "What is an allele?", "response": "One of the alternative forms of a gene."}
```

**Template:** each example is written as `### Question:` + prompt + `### Answer:` + response + end-of-document.

**Training:**

- **The prompt tokens are masked**, so the loss is computed on the answer only.
- 10% of examples are held out, and the held-out loss is printed each epoch.
- Defaults: 3 epochs, batch 8, learning rate $10^{-4}$.

**Data needed:** the 20 examples are a format template. A useful SFT set has thousands of examples. They can come
from:

- your own questions and answers;
- textbook exercises with worked answers;
- answers drafted by a large open model and checked by you, marked as synthetic.

## Preference tuning (DPO)

```
python finetune.py dpo --base tinyGPT_sft.pt --name tinyGPT_dpo --device cpu `
    --data finetune_examples\dpo_examples.jsonl
```

**Data:** each line holds a prompt, a better answer (*chosen*) and a worse one (*rejected*). The rejected answers
should be *plausible* mistakes, such as the classic p-value misreading, not nonsense.

**Training:**

- The loss (Section 9.6) raises the chosen answer's probability relative to the rejected one's, measured against a
  frozen copy of the starting model.
- `--beta` (default 0.1) sets how far the model may move from that copy.
- The printed **reward accuracy** is the share of pairs where the model now prefers the chosen answer.

**Asking questions:** `python finetune.py ask "What is linkage disequilibrium?" --checkpoint tinyGPT_dpo.pt` uses the
same template.

## Reinforcement learning: the design for later

RL isn't implemented yet, on purpose. It learns from attempts that sometimes succeed, and a 17M-parameter model almost
never produces a correct free-form answer, so the reward would always be zero. RL becomes worthwhile when the base
model solves roughly 10% or more of the problems.

The design, for an open 1 to 8B model:

1. **Problems with checkable answers:**
   - Hardy–Weinberg and allele-frequency calculations;
   - heritability from variance components;
   - $\chi^2$ tests;
   - R or Python functions checked by unit tests.
2. **Several attempts per problem:** 8 to 16, each scored 1 (correct) or 0.
3. **GRPO:** each attempt's advantage is its reward standardised within its group. Correct reasoning is reinforced,
   with a KL penalty towards the starting model (Section 9.6).
4. **Tools:** Hugging Face TRL's `GRPOTrainer`, or Unsloth on a single GPU.

## What to expect at this size

- **SFT** will teach tinyGPT the question-and-answer *format* quickly. Its answers will be fluent but often wrong: the
  model stores little (Section 9.4).
- **DPO** will move its choices on the trained pairs, but it won't generalise far.
- **Value:** treat both as a way to learn the methods with fast iterations. Apply them seriously to an open model
  (Chapter 11).

# Was this text written by my model?

detect_text.py gives two independent answers.

## Watermark: certain, but only for text you marked

Generate with a secret key:

```
python tiny_gpt.py --generate "Genetic drift is" --watermark-key mysecret
```

1. **At every step,** the key and the previous token pick a pseudo-random "green" quarter of the vocabulary.
2. **Green tokens get a small boost** (`--watermark-delta 2.0`). The text reads the same, but it contains more green
   tokens than chance would give.
3. **Detection:** `python detect_text.py text.txt --watermark-key mysecret` counts green tokens and reports a z-score
   (Section 9.7).
   - A z above 4 is very strong evidence (a false-alarm probability of about 3 in 100,000).
   - Unmarked text stays near 0.
   - Without the key nobody can detect or remove the mark reliably, although heavy paraphrasing weakens it.

In the tests, 120 marked tokens gave z > 4 and unmarked text z < 3.

## Likelihood score: evidence, for any text

```
python detect_text.py text.txt --checkpoint tinyGPT_best.pt
```

**The idea:** a model's own samples sit in a characteristic range of its probabilities. The analytic Fast-DetectGPT
criterion compares the log-probability of the actual tokens with what the model would expect from its own samples,
and standardises the difference (Section 9.7).

**Reading the score:**

- Above 3: model-like.
- 1 to 3: ambiguous.
- Below 1: human-like.

`--calibrate` scores 5 fresh model samples and 5 validation documents first, so you can see where each falls for your
checkpoint.

**Limits:**

- Short texts (under about 100 tokens) are unreliable.
- Text from *other* language models may also look model-like.
- Editing lowers the score.

Treat it as evidence, not proof. Only the watermark gives near-certainty.

# The mathematics and statistics

This chapter collects the equations behind every step, with $\theta$ for the parameters, $V$ the vocabulary size,
$d$ the model width and $T$ the number of tokens.

## Likelihood, perplexity and bits per byte

The model factorises the probability of a text by the chain rule:

$$p_\theta(x_1,\dots,x_T) = \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t})$$

The next-token probability is a softmax over the logits $z_1,\dots,z_V$:

$$p_\theta(x_t = v \mid x_{<t}) = \frac{e^{z_v}}{Z_t}, \qquad Z_t = \sum_{u=1}^{V} e^{z_u}$$

Training minimises the average negative log-likelihood (cross-entropy):

$$\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \ln p_\theta(x_t \mid x_{<t})$$

- **Perplexity** is $\mathrm{PPL} = e^{\mathcal{L}}$. Uniform guessing gives $\mathcal{L} = \ln V$, which is
  $\ln 8192 = 9.01$ for your tokenizer.
- **Bits per byte** divides the total negative log-likelihood by the number of UTF-8 bytes $B$ of the text instead of
  the number of tokens. $B$ is counted exactly: each token's bytes, minus the word-start space the tokenizer adds at
  the start of every line (it is not in the text). Checked against the decoded text of every bio_v3 validation source,
  the count matches to the byte:

$$\mathrm{BPB} = \frac{\sum_{t=1}^{T} -\ln p_\theta(x_t \mid x_{<t})}{B \ln 2} = \frac{\mathcal{L}}{\ln 2}\cdot\frac{T}{B}$$

For example, a loss of 3.0 nats per token with 4 bytes per token is $3.0/(0.693\times4) = 1.08$ bits per byte. Large
modern models reach well under 1 bit per byte on typical English text.

## The transformer

**Attention.** For the token vectors $X$ (one row per position), each head computes queries, keys and values
$Q = XW_Q$, $K = XW_K$, $V = XW_V$ with head width $d_h = 64$:

$$\mathrm{Attn}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_h}} + M\right)V, \qquad M_{ts} = \begin{cases} 0 & s \le t \\ -\infty & s > t \end{cases}$$

The mask $M$ makes attention causal. Dividing by $\sqrt{d_h}$ keeps the scores at unit scale.

**RoPE.** Each pair of coordinates $(2i, 2i+1)$ of a query or key at position $t$ is rotated by the angle
$\phi_{t,i} = t\,b^{-2i/d_h}$, with base $b = 10{,}000$:

$$\begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos\phi_{t,i} & -\sin\phi_{t,i} \\ \sin\phi_{t,i} & \cos\phi_{t,i} \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$

Because rotations compose, $\langle R_t q, R_s k\rangle = \langle q, R_{s-t} k\rangle$: the score depends only on the
distance $s-t$.

**RMSNorm and SwiGLU.** With a learnt gain $g$ and $\epsilon = 10^{-6}$:

$$\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \epsilon}} \odot g, \qquad \mathrm{FFN}(x) = W_{\text{down}}\big(\mathrm{SiLU}(W_{\text{gate}}x) \odot W_{\text{up}}x\big)$$

Here $\mathrm{SiLU}(u) = u\,\sigma(u)$ and $\sigma$ is the logistic function.

**One block, with residual connections:**

$$h = x + \mathrm{Attn}(\mathrm{RMSNorm}(x)), \qquad y = h + \mathrm{FFN}(\mathrm{RMSNorm}(h))$$

**Parameter count.** With $d_{kv} = n_{kv} d_h$ and feed-forward width $h$, each block has:

$$P_{\text{block}} = \underbrace{2d^2 + 2\,d\,d_{kv}}_{\text{attention}} + \underbrace{3\,d\,h}_{\text{feed-forward}} + \underbrace{2d}_{\text{norms}}$$

The whole model has $N = L\,P_{\text{block}} + d + Vd$ parameters. Your run:

- per block: $2(384^2) + 2(384^2) + 3(384)(1024) + 768 = 1{,}770{,}240$;
- all blocks: $N = 8 \times 1{,}770{,}240 + 384 + 8192 \times 384 = 17{,}308{,}032$.

**Training compute.** Each token costs about $2N$ operations forward and $4N$ backward:

$$C \approx 6ND = 6 \times 1.73\times10^{7} \times 7.78\times10^{8} = 8.1\times10^{16}\ \text{operations}$$

At your measured 4,826 tokens per second that is $6 \times 1.73\times10^7 \times 4826 \approx 5\times10^{11}$
operations per second, or 0.5 TFLOPS. That's a small fraction of what the RTX 3070 Laptop can do (Chapter 10).

## Optimisation

**Gradient clipping** with threshold $c = 1$:

$$g_t \leftarrow g_t \cdot \min\!\left(1, \frac{c}{\lVert g_t\rVert}\right)$$

**AdamW** with $\beta_1 = 0.9$, $\beta_2 = 0.95$ and weight decay $\lambda = 0.1$:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$

$$\theta_t = \theta_{t-1} - \eta_t\left(\frac{m_t/(1-\beta_1^t)}{\sqrt{v_t/(1-\beta_2^t)} + \epsilon} + \lambda\,\theta_{t-1}\right)$$

Here $m_t/(1-\beta_1^t)$ is the bias-corrected mean gradient. Dividing by $\sqrt{v_t}$ gives every parameter its own
step size.

**Warmup–stable–decay (WSD) schedule.** For step $s$ of $S$, warm-up length $s_w$ and decay start $s_d$:

$$\eta(s) = \begin{cases} \eta_{\max}\, s/s_w & s \le s_w \\ \eta_{\max} & s_w < s \le s_d \\ \eta_{\min} + (\eta_{\max}-\eta_{\min})\left(1 - \sqrt{\dfrac{s-s_d}{S-s_d}}\right) & s > s_d \end{cases}$$

Your run: $s_w = 237$, $s_d = 9{,}496$, $S = 11{,}870$, $\eta_{\max} = 1.017\times10^{-3}$ and
$\eta_{\min} = 1.017\times10^{-4}$.

**z-loss.** A penalty on the log of the softmax normaliser $Z_t$ keeps the logits from drifting upwards, which
prevents loss spikes (PaLM):

$$\mathcal{L}_{\text{total}} = \mathcal{L} + \lambda_z\,\frac{1}{T}\sum_{t=1}^{T} (\ln Z_t)^2, \qquad \lambda_z = 10^{-4}$$

**Weight decay as a prior.** For plain gradient descent, adding $\frac{\lambda}{2}\lVert\theta\rVert^2$ is MAP
estimation under a Gaussian prior $\theta \sim \mathcal{N}(0, \lambda^{-1}I)$. AdamW's *decoupled* decay is a close
cousin that shrinks every weight by the same fraction at each step.

## Scaling laws and repeated data

Hoffmann et al. (2022, the Chinchilla paper) fitted the loss as a function of parameters $N$ and tokens $D$:

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

Their fitted values were $E = 1.69$, $A = 406.4$, $B = 410.7$, $\alpha = 0.34$ and $\beta = 0.28$.

- **Compute-optimal sizes:** minimising $L$ subject to $C = 6ND$ gives $N_{\text{opt}}$ and $D_{\text{opt}}$ that both
  grow as roughly $\sqrt{C}$, with $D \approx 20N$.
- **Inverting it:** for a compute budget $C$, $N_{\text{opt}} \approx \sqrt{C/120}$.
- **The constants are for their data and tokenizer.** Your own curve needs 6 to 10 short runs (the roadmap in
  Chapter 12).

**Repeated data.** Muennighoff et al. (2023) model the value of repeating $U$ unique tokens $R$ extra times as an
effective amount of data:

$$D' = U + U R^{*}\left(1 - e^{-R/R^{*}}\right), \qquad R^{*} \approx 15.4$$

- **Four passes** ($R = 3$) are worth $D' = 3.73\,U$: 93% of fresh data. That's why tinyGPT caps repetition at 4
  epochs.
- **Forty passes** would be worth only about 15 fresh epochs.

**Knowledge capacity.** Allen-Zhu and Li (2024) measured about **2 bits of factual knowledge per parameter** in
well-trained language models.

- Your 17.3M-parameter model can store at most about 35 million bits, roughly 4 MB of facts.
- A 1B-parameter model could store about 250 MB.
- This is the main reason small models hallucinate: they can't *hold* many facts, however well they write.

## Evaluation statistics

**Paired, document-level bootstrap** (compare_models.py). For documents $i = 1,\dots,n$ with byte counts $B_i$ and
negative log-likelihoods $\ell_i^A$ and $\ell_i^B$ under the two models, the byte-weighted difference in bits per
byte is:

$$\hat\Delta = \frac{\sum_i (\ell_i^A - \ell_i^B)}{\ln 2\,\sum_i B_i}$$

1. Resample documents with replacement 2,000 times and recompute $\hat\Delta^{*}$ each time.
2. The 2.5th and 97.5th percentiles give the 95% interval.
3. The two-sided p-value is $2\min\{P(\hat\Delta^{*}\le 0), P(\hat\Delta^{*}\ge 0)\}$.
4. Resampling documents, not tokens, respects the correlation between tokens within a document: this is the cluster
   bootstrap.

**Wilson score interval** for a benchmark accuracy $\hat p$ on $n$ items, with $z = 1.96$:

$$\frac{\hat p + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$

Unlike $\hat p \pm z\sqrt{\hat p(1-\hat p)/n}$, it stays inside $[0,1]$ and behaves well for small $n$.

**Power.** The number of items needed to detect accuracy $p_1$ versus $p_2$ (two-sided, 5% level, 80% power) is:

$$n \approx \frac{(z_{0.975} + z_{0.8})^2\,[\,p_1(1-p_1) + p_2(1-p_2)\,]}{(p_1 - p_2)^2}$$

- 25% versus 40% needs about 149 items.
- 25% versus 35% needs about 325 items.
- So the 336-item benchmark detects gains of about 10 percentage points (the old 171 items: about 15). Paired tests
  on the same items (McNemar) need somewhat fewer.

**Expected calibration error.** Group predictions into $M$ bins by confidence. With $n_b$ predictions in bin $b$:

$$\mathrm{ECE} = \sum_{b=1}^{M} \frac{n_b}{n}\,\big|\,\mathrm{acc}(b) - \mathrm{conf}(b)\,\big|$$

A well-calibrated model is right 70% of the time when it says 70%. The dashboard shows the reliability diagram.

## Post-training

**Supervised fine-tuning** averages the loss over the answer positions $\mathcal{A}$ only:

$$\mathcal{L}_{\text{SFT}} = -\frac{1}{|\mathcal{A}|}\sum_{t \in \mathcal{A}} \ln p_\theta(y_t \mid x, y_{<t})$$

**Bradley–Terry preferences.** The probability that answer $y_w$ is preferred to $y_l$ is:

$$P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big)$$

**RLHF objective.** Maximise reward with a KL penalty towards the reference model:

$$\max_{\theta}\ \mathbb{E}_{y \sim \pi_\theta(\cdot\mid x)}\big[r(x,y)\big] - \beta\,\mathrm{KL}\big(\pi_\theta(\cdot\mid x)\,\big\|\,\pi_{\text{ref}}(\cdot\mid x)\big), \qquad \mathrm{KL}(p\,\|\,q) = \sum_{y} p(y)\ln\frac{p(y)}{q(y)}$$

Its solution is $\pi^{*}(y\mid x) \propto \pi_{\text{ref}}(y\mid x)\,e^{r(x,y)/\beta}$. Solving for $r$ and
substituting into Bradley–Terry gives **DPO**, which needs no reward model:

$$\mathcal{L}_{\text{DPO}} = -\ln\sigma\!\left(\beta\left[\ln\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \ln\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right]\right)$$

The bracket is the difference in *implicit rewards*. The reward accuracy printed by finetune.py is the share of
pairs where it is positive.

**GRPO.** For $G$ attempts $y_1,\dots,y_G$ at one problem with rewards $r_i$, the advantages are
$A_i = (r_i - \bar r)/s_r$. With $\rho_i = \pi_\theta(y_i\mid x)/\pi_{\text{old}}(y_i\mid x)$ the objective to
maximise is:

$$\frac{1}{G}\sum_{i=1}^{G}\min\!\big(\rho_i A_i,\ \mathrm{clip}(\rho_i, 1-\varepsilon, 1+\varepsilon)A_i\big) - \beta\,\mathrm{KL}\big(\pi_\theta\,\|\,\pi_{\text{ref}}\big)$$

**Distillation and LoRA.** A student $S$ learns a teacher's full distributions, and LoRA learns a scaled low-rank
update:

$$\mathcal{L}_{\text{KD}} = \sum_{t}\mathrm{KL}\big(p_T(\cdot\mid x_{<t})\,\big\|\,p_S(\cdot\mid x_{<t})\big), \qquad W' = W + \frac{\alpha}{r}BA$$

## Generation and detection

**Sampling.** Temperature $\tau$ rescales the logits, $p_i \propto e^{z_i/\tau}$. Top-$p$ keeps the smallest set of
tokens whose probabilities sum to at least $p$.

**Watermark test** (Kirchenbauer et al. 2023). For $T$ scored tokens, $|s|_G$ of them green, and green share
$\gamma = 0.25$:

$$z = \frac{|s|_G - \gamma T}{\sqrt{T\gamma(1-\gamma)}}$$

Only distinct (previous token, token) pairs are scored: a repeated phrase repeats the same green or red verdict, and
counting it again would inflate $z$. Without a watermark, $|s|_G \sim \mathrm{Binomial}(T,\gamma)$, so $z$ is
approximately standard normal and $P(z > 4) \approx 3\times10^{-5}$. detect_text.py reports the exact binomial tail
$P(X \ge |s|_G)$ as the p-value, which is also right for short texts, where the normal approximation is poor.

**Likelihood score** (analytic Fast-DetectGPT, Bao et al. 2024). At each position, the model's expected log-probability
and its variance under its own distribution are:

$$\mu_t = \sum_{v} p(v\mid x_{<t})\ln p(v\mid x_{<t}), \qquad \sigma_t^2 = \sum_{v} p(v\mid x_{<t})\big(\ln p(v\mid x_{<t})\big)^2 - \mu_t^2$$

$$\hat d(x) = \frac{\sum_{t}\big[\ln p(x_t\mid x_{<t}) - \mu_t\big]}{\sqrt{\sum_t \sigma_t^2}}$$

Model-written text has $\hat d$ well above 0, because it tends to pick tokens more probable than a typical draw.

## Data statistics

**MinHash.** For a random hash function, the probability that two documents share the minimum hash of their word
shingles equals their Jaccard similarity:

$$P\big[h_{\min}(A) = h_{\min}(B)\big] = J(A,B) = \frac{|A\cap B|}{|A\cup B|}$$

**One-permutation hashing.** Computing 128 separate hash functions for every shingle was the slowest part of
data preparation. data_prep.py hashes each shingle once instead: the top 7 bits of the hash choose one of 128
buckets, and each bucket keeps the smallest remaining value (Li, Owen and Zhang 2012). A document shorter than 128
shingles leaves some buckets empty; they copy the next filled bucket's value with an offset ("densification",
Shrivastava and Li 2014), which keeps the estimate unbiased. On PubMed Central papers it flagged the same
near-duplicates as 128 hash functions, with the same error, in a tenth of the time.

**Locality-sensitive hashing.** data_prep.py uses the 128 bucket values split into $b = 16$ bands of $r = 8$ rows. Two documents
become duplicate candidates if all rows of any one band agree:

$$P(\text{candidate}) = 1 - \big(1 - J^{r}\big)^{b}$$

This gives 0.95 at $J = 0.8$ but only 0.06 at $J = 0.5$: a sharp threshold that avoids comparing every pair.

**Leakage audit.** The share of validation word 13-grams that also occur in the training data. Long n-grams rarely
repeat by chance, so a high share signals copied text.

# The laptop GPU throttle

## What was measured

| Date | Reading |
|----------------------|------------------------------------------------------------------------------|
| 26 September | Enforced power limit 55 W, against the 80 W default |
| 29 September, during your run | The GPU sat in performance state **P8 at 210 MHz** at 100% use, with a 20 W enforced power limit |

- **Things checked:** Turbo mode in Armoury Crate didn't change the limit. The NVIDIA Platform Controllers and
  Framework (NVPCF) driver, which handles Dynamic Boost, is installed and working.
- **Effect on your run:** it trains at about 4,800 tokens per second, which is 0.5 TFLOPS of useful work.
  - At full power the same run is estimated to be 5 to 6 times faster: about 8 hours instead of 45.
  - This is an estimate; measure it after a fix.

## Likely causes

This is an observation to investigate, not a proven diagnosis. On ASUS TUF laptops, a GPU held at its lowest state
under load usually has one of these causes:

1. **The power source.**
   - Running from a USB-C (Power Delivery) charger, or from any adapter below the laptop's rating, makes the firmware
     cap the dedicated GPU.
   - A USB-C dock or monitor that also supplies power can do the same.
   - This is the leading suspect.
2. **A firmware or embedded-controller state** that got stuck, for example after sleep or hibernation.
3. **Driver or control-software mismatch:** BIOS, ASUS System Control Interface, Armoury Crate and the NVIDIA driver
   out of step.
4. **Battery condition:** a nearly empty or faulty battery can make the firmware limit power.
5. **Heat.** It's unlikely at 20 W, but it was 79 to 84 °C on 26 September at higher power.

## What to try, in order

None of these was done without asking. Each is your decision. Test after each one with
`python gpu_check.py --load 60`, but only when no training is running. A fix shows as P0 or P2, a graphics clock above
1,200 MHz and 60 to 80 W drawn.

1. **Read the throttle reason.** Run `nvidia-smi -q -d PERFORMANCE,POWER` during load and look at the lines under
   "Clocks Event Reasons".
   - *HW Power Brake Slowdown: Active* means the laptop itself is signalling a power problem, which points to the
     adapter or charging circuit.
   - *SW Power Cap* means a software or firmware limit.
2. **Use the original ASUS barrel-plug adapter, directly in a wall socket.** Unplug every USB-C cable that could
   supply power: chargers, docks and monitor cables with charging.
3. **Reset the embedded controller.**
   1. Shut down fully.
   2. Unplug the adapter.
   3. Hold the power button for 40 seconds.
   4. Reconnect the adapter and start up.

   This is ASUS's standard "hard reset" and changes no settings.
4. **Check the modes.**
   - Armoury Crate: operating mode Turbo (only effective on mains power), and GPU mode Standard or Ultimate, not Eco.
   - Windows: power mode "Best performance".
5. **Update, in this order,** from MyASUS and NVIDIA:
   1. BIOS;
   2. ASUS System Control Interface;
   3. Armoury Crate;
   4. the NVIDIA Studio driver, with the "clean installation" option.

   These change system software, so do them yourself when convenient.
6. **NVIDIA Control Panel:** under Manage 3D settings, give `python.exe` the power management mode "Prefer maximum
   performance".
7. **Cooling:** clean the vents and raise the back of the laptop.
8. **If the power-brake reason persists with the original adapter:** try another genuine ASUS adapter of the same
   rating. If that doesn't help, it's a hardware fault for ASUS support (adapter, DC-in board or battery).

# Scaling up

## What each machine could train with tiny_gpt.py

All figures use $C = 6ND$ and $N_{\text{opt}} \approx \sqrt{C/120}$ (20 tokens per parameter). Useful throughput is
**estimated** where marked; measure it with `gpu_check.py --train-bench` before relying on it.

| Machine | Useful speed | Sensible model | Notes |
|--------------------|--------------------|------------------------|------------------------------------|
| This laptop, throttled (20 W, P8) | 0.5 TFLOPS (measured) | 17M on 0.8B tokens: 45 h | The current run |
| This laptop at full power | About 3–8 TFLOPS (estimated) | 100–200M on 2–4B tokens in a week | 8 GB of GPU memory limits the model to a few hundred million parameters |
| 1 × H100 for $100 | About 250 TFLOPS (estimated) | About 450M on 9B tokens | Needs at least 2.3B *unique* tokens (4 passes). Your run would take under half an hour there |
| 8 × H100, a day | About 3,000 TFLOPS | 1B on 20B tokens in about 11 h | Needs multi-GPU training (DDP), not yet in tiny_gpt.py |
| Mac Studio M5 Ultra 256 GB | About 20–40 TFLOPS with PyTorch MPS (assumed) | 0.5–1B from scratch in a month | Compute-bound: memory would hold about 8B parameters, but training one would take years |

**Right now, data limits you more than hardware.** bio_v2 has 194.5M unique training tokens, which supports about 40M
parameters at 4 passes. Keyword-filtered FineWeb-Edu and peS2o can add billions (Chapter 4).

## Renting an H100 (Lambda Cloud)

**Prices** (Lambda's list prices in September 2026; check lambda.ai/pricing before renting):

- **One H100 PCIe:** about **$3.29 per hour**.
- **H100 SXM:** about **$3.99 to $4.29 per GPU-hour**, in 8-GPU machines.
- Billing runs while the instance exists, even when idle. Storage is billed separately.

**What $100 buys:**

- **Time:** about 30 hours on one H100.
- **Compute:** at about 250 useful TFLOPS that's $30 \times 3600 \times 2.5\times10^{14} \approx 2.7\times10^{19}$
  operations.
- **The model:** $N_{\text{opt}} = \sqrt{2.7\times10^{19}/120} \approx 470$M parameters on about 9.5B tokens,
  allowing a few hours for setup and uploading.
- **For comparison,** Karpathy's llm.c reproduced GPT-2 small (124M) for about $20, and GPT-2 XL (1.5B) in 24 hours
  on 8 H100s for about $672.

**The better use of $100:** continued pre-training or LoRA of an open model. For example, a LoRA pass of an open 8B
model over 200M tokens of your text costs $4\times(8\times10^{9})\times(2\times10^{8}) = 6.4\times10^{18}$
operations, about 6 hours ($20).

**Steps with tiny_gpt.py:**

1. Build the dataset at home. Only the token files are needed: bio_v2 is about 400 MB.
2. Rent one H100 with Linux, where `torch.compile` works out of the box (on Windows it needs the `triton-windows`
   package).
3. Upload the dataset and scripts.
4. Run `python tiny_gpt.py --dataset datasets\bio_v3 --time-budget-hours 25`.
5. Download the best checkpoint and experiments.csv.
6. **Terminate the instance.**

## A 1B model from scratch

One billion-parameter shape: width 2,048, 22 layers, 16 query heads, 4 key/value heads and a 32,768-token vocabulary
(1.04B parameters).

| Requirement | Value |
|-------------------------|---------------------------------------------------------------------------|
| Training tokens | About 20B (at least 5B unique) |
| Compute | $6 \times 1.04\times10^{9} \times 2.08\times10^{10} \approx 1.3\times10^{20}$ operations |
| Memory | About 16.7 GB for weights, gradients and Adam states, plus activations. It can't fit in the laptop's 8 GB |
| Time on 1 × H100 | About 4 to 6 days ($300–500) |
| Time on 8 × H100 | About 11 to 15 hours ($350–500), once multi-GPU training is added |
| Time on the Mac Studio | About 1.5 to 2.5 months (assumed speed) |

## Is 1B enough for a specialised field?

It depends on the job.

- **Fluent domain text:** yes. 1B is plenty for vocabulary, style, summarising and classification.
- **Knowledge:** at about 2 bits per parameter, a 1B model holds about 250 MB of facts. That's a lot of textbook
  knowledge, but not the literature. It will still invent details.
- **Reasoning:** multi-step statistical-genetics reasoning comes mainly from scale, plus RL on checkable problems.
  Open models of 7B and more are clearly better.
- **Evidence from the field:** domain-trained small models (BioGPT, BioMedLM) beat general models of their own size.
  Large general models with search over the literature beat them.

**Recommendation:**

- 1B from scratch is an excellent learning project and a useful domain autocomplete.
- For an assistant you'd rely on, adapt an open model and add search over your papers and tools. That model can be
  anything from 1–8B for fine-tuning up to a 235–355B mixture of experts for running on the Mac.
- You don't need to train 10B or 100B models yourself: a 70B model from scratch costs millions of pounds of compute.

## Continued pre-training of an open model: do the weights match?

They don't have to. Continued pre-training uses the **open model's own architecture and tokenizer**. You feed it
your text, and its weights keep their shapes.

- **Shapes differ from tinyGPT.** Llama 3.2 1B has width 2,048, 16 layers, 32 query heads, 8 key/value heads and a
  128,256-token vocabulary. tinyGPT's current weights (width 384, vocabulary 8,192) can't be merged into it.
- **The design family is the same:** RoPE, RMSNorm, SwiGLU and GQA. A converter could load such a model into
  tiny_gpt.py's code. It would need to:
  - use the open model's tokenizer instead of yours;
  - match its RoPE settings (for example base 500,000 with scaling);
  - handle details such as Qwen's attention biases.
- **Simplest route:** Hugging Face `transformers` with TRL, Unsloth or Axolotl, reading your text as JSON Lines.
  data_prep.py's cleaned, filtered documents can be exported for this.
- **Candidates** (all with open weights; OLMo also has open data):
  - Llama 3.2 1B;
  - Qwen 1.5–1.7B;
  - SmolLM2 1.7B;
  - OLMo 2 1B.
- **Memory:**
  - A full update of a 1B model needs about 16 GB plus activations: an H100 or the Mac.
  - LoRA or QLoRA of a 1B model fits in the laptop's 8 GB.
- **Cost:** about $6\times10^{18}$ operations per billion tokens at 1B parameters, or about 6 H100-hours (about $20).

## The Mac Studio plan (from GUIDE.md)

This section keeps the plan written on 27 September for a Mac Studio M5 Ultra with 256 GB. Numbers marked
*(assumed)* must be re-measured on the real machine, and model names re-checked in February 2027.

### The best setup

The best setup combines four parts:

- **A model:** the strongest open mixture-of-experts model that fits, about 235–355B parameters in total with
  20–35B active, at 4-bit.
- **Search:** an index over your papers.
- **Tools:** tools that run R and Python.
- **A fine-tune:** your own LoRA for editing and your conventions.

| Ability | Where it comes from | What you do |
|----------------------------------|------------------------------------|------------------------------|
| Reasoning, statistics, coding, catching errors | Pre-training on 15–36T tokens plus costly RL | Choose the strongest base model |
| Your field's voice, editing the way you want | Fine-tuning (LoRA) on examples | You train this |
| Facts, numbers, real citations | Searching your papers at answer time | You build a search index |
| Correct calculations and plots | Actually running code | You connect tools |

**Why 20,000 papers can't make a better scientist:**

- They are about 160M tokens: about 0.001% of the 15T or more a modern base model has read.
- Fine-tuning at that scale changes *how the model writes*, not *how well it reasons*.
- Facts learnt by fine-tuning are recalled unreliably, so they belong in the search index.

### What fits in 256 GB

At 4-bit a parameter takes about 0.56 bytes. macOS lets the GPU use about 75% of memory by default; raise it with
`sudo sysctl iogpu.wired_limit_mb=235000`.

| Model | Memory at 4-bit | Runs? | QLoRA training? |
|----------------------------|------------------|------------------------------|------------------------|
| 70B dense | 39 GB | Yes | Yes |
| 120B MoE (5B active) | 67 GB | Yes, very fast | Yes |
| 235B MoE (22B active) | 132 GB | Yes, with room for long papers | Yes (about 170 GB) |
| 355B MoE (32B active) | 199 GB | Yes, about 35 GB left for context | No (about 238 GB) |
| 400B MoE | 224 GB | Only just, short context | No |
| 671B MoE | 376 GB | No | No |

**Speed:** it depends on the *active* parameters. Words per second ≈ memory bandwidth (about 800 GB/s, assumed) ÷
bytes read per word. A 235B MoE therefore writes about 25–35 words per second, faster than a 70B dense model (12–15).

**512 GB would matter** only for running 400B+ models or QLoRA of 355B. It doesn't matter for tinyGPT-style training
from scratch, which is limited by compute.

**Candidates in September 2026** (memory at 4-bit):

| Model | Memory | Fits? |
|--------------------------------------------------|--------------------|------------------------------|
| GLM-5.3-Flash (320B/18B) | 179 GB | Fits |
| Qwen3.8-Flash-Next (176B/6B) | 99 GB | Fits |
| MiMo-V2.6-Flash (310B/15B) | 174 GB | Fits |
| DeepSeek-V4-Flash (284B/13B) | 159 GB | Fits |
| DeepSeek-V4.1-Flash (552B/16B) | 309 GB | Only at about 3-bit |

### LoRA training time on the Mac

Assuming about 30 TFLOPS (assumed) and LoRA compute of about $4 \times \text{active parameters} \times \text{tokens}$:

| Model | Editing set (30M tokens) | Style corpus (160M tokens) |
|----------------------------------|---------------------------------|---------------------------------|
| 8B dense | About 9 h | About 2 days |
| 70B dense | About 3 days | About 17 days |
| 235B MoE (22B active) | About 2–3 days | About 2–3 weeks |

Use the time for several shorter experiments, not one long run: many passes over the same data make a model
memorise.

**Pausing:**

- MLX saves adapters with `--save-every 200` and resumes with `--resume-adapter-file`. It doesn't restore the
  optimiser, so the loss jumps briefly.
- Hugging Face tools restore everything with `resume_from_checkpoint=True`.
- Use `caffeinate -i` to stop the Mac sleeping.

### Step by step

**Phase A: before the Mac arrives (on the laptop)**

1. Learn the basics: this guide, the Hugging Face LLM course and the MLX-LM LoRA guide.
2. Write your **evaluation set**. It's the most important step: without it you can't tell whether anything helped.
   About 2–3 days of work:
   - 40 proofreading tasks;
   - 20 drafting tasks;
   - 30 analysis tasks with known answers;
   - 20 knowledge questions with citations;
   - 10 "trap" paragraphs hiding a statistical error.
3. Shortlist base models: run the evaluation set on 3–5 open models through OpenRouter, for a few pounds. Don't send
   private data.
4. Collect papers legally: open access first, then the text-and-data-mining routes for paywalled journals.
5. Convert them to clean text. Publisher XML (JATS) is cleanest. Remove references and page furniture, and
   deduplicate.
6. Build three training sets:
   - **editing:** 30–50K examples of degraded paragraph → original, notes → paragraph, and error-spotting;
   - **analysis:** 1–3K task → R or Python code examples, *each one run and checked*;
   - **style corpus:** optional.

   Split by paper, and mix in 10–20% general instruction data.

**Phase B: the Mac's first week.**

1. Install Homebrew, uv, Python 3.12, mlx-lm, LM Studio and R.
2. Raise the GPU memory limit.
3. Download the shortlisted models in MLX 4-bit.
4. Measure speed and memory, and run the evaluation set.
5. Benchmark LoRA speed and redo the time table.

**Phase C: useful without training (weeks 2–4).**

1. Build a search index over your papers: a local embedding model, 500-token passages, and LanceDB or Chroma.
2. Add tools that run R and Python: Continue or Cline in VS Code, or a small tool-calling agent.
3. Rerun the evaluation set. That score is the baseline a fine-tune must beat.

**Phase D: fine-tuning (months 2–3).**

1. Pilot on an 8B model with the editing set:

   ```
   mlx_lm.lora --model <8B-4bit> --train --data data/edit \
     --batch-size 4 --iters 2000 --max-seq-length 4096 \
     --num-layers 16 --learning-rate 1e-4 --mask-prompt \
     --grad-checkpoint --save-every 200 --steps-per-eval 200 \
     --adapter-path adapters/pilot -c lora.yaml
   ```

2. Read the curves and the samples.
3. Run the main job on the chosen MoE, on the Mac or on rented GPUs.
4. Keep the fine-tune only if the evaluation improves.
5. Retrain every few months with your own corrections, the most valuable training data you will ever have.

**Phase E: daily use.**

- The Mac serves the model around the clock. The laptop connects over Wi-Fi at home, or through Tailscale from elsewhere.
- VS Code tools point at the Mac's OpenAI-compatible address.
- Individual-level genetic data never leaves your machine; check your data-access agreements.
- Hard cases go to a frontier model, without confidential data.

### Rented GPUs, and still owning the model

- **Ownership:**
  - Open-weight models (Apache 2.0 or MIT for Qwen, DeepSeek, GLM and gpt-oss; check each licence) can be kept and
    modified.
  - Your adapter is a file you download. Once you delete the cloud storage, the provider keeps nothing.
- **Workflow:**
  1. Test the whole script at home on a small model first.
  2. Rent (RunPod, Lambda or Vast.ai), attach persistent storage, and train with Unsloth or Axolotl.
  3. Save checkpoints to the persistent volume.
  4. Merge the adapter and quantise on the rented machine.
  5. Download, then delete the volume.
- **Free research compute:** your university's research computing service, and the UK AI Research Resource (Isambard-AI, Dawn) for eligible projects.
- **Budget:** about £100–300 in total for the fine-tuning plan, including failed attempts.

### The video model, the subscription and the timeline

**The video model:**

- **Style LoRAs** on open video models (Wan 2.x, HunyuanVideo, LTX-Video) need 50–500 clips and hours on rented
  H100s.
- **Good for** atmosphere, landscapes and background footage.
- **Not reliable for** equations, labels, molecules or a consistent person. Train only on your own renders or
  licensed footage.

**The subscription:** after phases C and D, log for a month which tasks the local model handles well, then decide
whether to downgrade. The log should make the decision.

**The timeline:**

| When | Phase |
|-----------------------------------|-----------------------------------------------------------------|
| Now to January | A1–A3 |
| Now to February | A4–A6 |
| Mac, week 1 | B |
| Mac, weeks 2–4 | C |
| Mac, months 2–3 | D |
| Month 4 onwards | E, and the subscription decision |

# Techniques roadmap

Status of the technique catalogue (the former *tinyGPT_advanced_techniques.docx*). **Gain** is the expected effect for
a 5–50M model: S (a few percent), M (noticeable) or L (large). **Effort:** S is hours, M days, L weeks.

## Done on 29 September

| Technique | Where |
|---------------------------------------------|-------------------------------------------------------|
| Bits per byte | tiny_gpt.py metrics line, dashboard, compare_models.py |
| Paired document-level bootstrap | compare_models.py |
| z-loss | tiny_gpt.py `--z-loss` (default $10^{-4}$) |
| Supervised fine-tuning and DPO | finetune.py |
| Generation watermark and likelihood detector | tiny_gpt.py `--watermark-key`, detect_text.py |
| Permanent experiments table | experiments.csv |
| Earlier: AdamW, WSD, clipping, BF16, RoPE, RMSNorm, SwiGLU, GQA, tied embeddings, data-aware planner, MinHash deduplication, document split, leakage audit, fixed validation windows, calibration, Wilson intervals | tiny_gpt.py, data_prep.py |

## Recommended next, in order

| Technique | What it does | Gain | Effort | Where |
|----------------------|----------------------------------------------------|--------|--------|----------|
| Learning-rate sweep | Five log-spaced rates, 500-step runs. The heuristic can be off by 2–3 times | M | S | Laptop |
| Your own scaling law | Fit $L(N,D)$ to 6–10 short runs to find the best size for *your* data | L | M | Laptop |
| QK-norm | Normalise queries and keys; allows higher learning rates stably | S | S | Laptop |
| Weight averaging (EMA) | Average recent weights; often a free drop in loss | S | S | Laptop |
| Mixture grid | Short runs over `--mix` weights, judged by per-source bits per byte | M | S | Laptop |
| Several seeds | Measure run-to-run noise before trusting a gain | M | S | Laptop |
| Standard benchmark suite | Section 6.5 | M | M | Laptop |
| Document masking | Attention never crosses a document boundary inside a window | S | M | Laptop |
| Multi-GPU (DDP) | Needed for 8-GPU cloud runs | L | M | Cloud |
| Continued pre-training of an open model | Section 11.5 | L | M | Mac/cloud |
| Distillation from an open teacher | Learn a larger model's full distributions, or its written explanations | L | M | Mac |
| Retrieval at answer time | Search your documents and condition on the passages | L | M | Laptop |

## Later or at larger scale

- **Optimisers:** µP (learning rates that transfer across widths), Muon, schedule-free AdamW and critical-batch-size
  estimation.
- **Architecture:** multi-token prediction, RoPE context extension and mixture of experts.
- **Data:** quality classifiers, suffix-array deduplication and synthetic textbook data.
- **Efficiency:** chunked cross-entropy, activation checkpointing, 8-bit optimisers and FlashAttention kernels
  (`torch.compile` is already on).
- **Interpretability:** logit lens, linear probes, sparse autoencoders.
- **Uncertainty:** conformal answer sets and semantic entropy.

**Not planned:** importance resampling as a *replacement* for keyword selection (your decision, Section 4.4).

**Doesn't help at this size:**

- label smoothing;
- dropout without heavy repetition;
- treating 20 tokens per parameter as a law;
- much larger vocabularies (each extra 8,192 tokens costs 3.1M parameters at width 384).

# Tools, safety and naming

## Other training frameworks

| Tool | What it is | Worth it for you? |
|------------------------|--------------------------------------|--------------------------------------|
| **JAX** (Google), with Flax and Optax | NumPy-like arrays compiled by XLA; excellent on TPUs. MaxText and Levanter are reference LLM trainers | Not now. On Windows it runs on the CPU only (CUDA needs WSL2); the Mac plug-in is experimental. Relevant if you get free TPU time (Google's TPU Research Cloud) |
| nanoGPT, llm.c (Karpathy) | Minimal reference trainers | Good for reading; tinyGPT already covers them |
| torchtitan, Megatron-LM, DeepSpeed | Large-scale multi-GPU training | Only for multi-node cloud runs |
| Hugging Face transformers, TRL | Open models, SFT, DPO and GRPO trainers | Yes, for adapting open models |
| Unsloth, Axolotl, LLaMA-Factory | Easy, memory-efficient fine-tuning | Yes, for LoRA on the cloud or the laptop |
| MLX, mlx-lm | Apple's framework | Yes, on the Mac Studio |
| lm-evaluation-harness | Standard benchmarks | Yes, for Section 6.5 |
| datatrove, text-dedup | Large-scale data cleaning | Only for tens of billions of tokens |
| TensorBoard, Weights & Biases | Live training curves | Optional; the log, metrics JSON Lines and experiments.csv cover this |

## Safety rules built into the scripts

- **Safe loading.** Checkpoints are loaded with `weights_only=True`, so they can't run code. `--trust-checkpoint`
  exists only for your own old files. Exports use safetensors.
  - CVE-2025-32434 showed that `weights_only=True` can be bypassed in PyTorch before 2.6. PyTorch 2.10 is
    installed here, so this is closed; on another machine with an older PyTorch, load only checkpoints you made
    yourself.
- **No overwriting.**
  - New names are required; `--overwrite` is always explicit.
  - Saves are atomic.
  - The earlier scripts are kept in `backup_v3_2026-09-29`.
  - Dataset folders are never reused with different settings.
- **Data is never executed.** Code files are read as text. Spreadsheets and Word files are parsed as XML without
  macros.
- **One GPU job at a time.** gpu_check.py refuses to run while the GPU is busy, unless you pass `--force`.
- **No installs.** Every package in requirements.txt is already installed. On a new machine:
  `python -m pip install -r requirements.txt`, with PyTorch installed first from pytorch.org for the right CUDA
  version.
- **Tests.** `python -m unittest test_tiny_gpt.py` runs 24 tests on the CPU in about two and a half minutes.

## A name for the model

"tinyGPT" describes the size, not the purpose, and "GPT" is OpenAI's trademark. Suggestions:

- **Locus** (recommended): short, a genetics word, and it suggests "focus on one place in the genome of knowledge";
- **Codon**;
- **Allele**;
- **Helix**.

**AIFB** ("AI for Biology") is clear but generic and hard to say. Renaming is only a change of `DEFAULT_NAME` plus
file names. Old checkpoints keep working, because the loader recognises them by content, not name.

# Glossary {.unnumbered}

| Term | Meaning |
|-------------------------|---------------------------------------------------------------------------|
| token | A piece of text (word, word part or symbol) with an integer ID |
| BPE | Byte-pair encoding: builds the vocabulary by repeatedly merging frequent pairs |
| parameter, weight | One learnt number |
| embedding | The vector that represents a token |
| attention | The step in which each token gathers information from earlier tokens |
| RoPE | Rotary position embedding: encodes position by rotating queries and keys |
| RMSNorm | Normalisation by the root-mean-square of a vector |
| SwiGLU | A gated feed-forward layer |
| GQA | Grouped-query attention: several query heads share a key/value head |
| logit | A token's score before the softmax |
| loss, cross-entropy | Average $-\ln$ probability of the correct next token |
| perplexity | $e^{\text{loss}}$: the effective number of choices |
| bits per byte | Loss per byte of raw text, in bits; comparable across tokenizers |
| gradient | The direction of change that most reduces the loss |
| learning rate | The step size of each update |
| WSD | Warmup–stable–decay learning-rate schedule |
| z-loss | A penalty keeping the softmax normaliser small |
| epoch | One full pass over the training data |
| tokens per parameter | Training tokens divided by model size (Chinchilla: about 20) |
| overfitting | Memorising the training data; validation loss rises |
| leakage | Validation text that also appears in training |
| MinHash, LSH | Fast estimation of text similarity, used for near-duplicate detection |
| base model | A pre-trained model that continues text |
| SFT | Supervised fine-tuning on question-and-answer pairs |
| RLHF, DPO | Training on preferences, with or without a reward model |
| GRPO | RL that standardises rewards within a group of attempts |
| MoE | Mixture of experts: only some weights run for each token |
| distillation | Training a small model to imitate a big one |
| quantisation | Storing weights in fewer bits |
| LoRA, QLoRA | Low-rank fine-tuning (on a 4-bit model) |
| checkpoint | A saved model, optionally with the full training state |
| decoder-only | A transformer with causal attention trained to predict the next token (GPT, Llama, tinyGPT) |
| encoder-decoder | A transformer that reads the input in both directions, then writes a separate output (T5) |
| safetensors | A weight-file format that can't contain code |
| watermark | A hidden statistical signal in generated text, detectable with a key |
| RAG | Retrieval: searching documents and giving the passages to the model |

# Reading list and references {.unnumbered}

**Start with:**

1. 3Blue1Brown, *Neural networks*, chapters 5–7 on transformers (YouTube).
2. Andrej Karpathy, *Let's build GPT: from scratch, in code, spelled out* (YouTube, 2 h), and *Deep Dive into LLMs
   like ChatGPT* (3.5 h).
3. Sebastian Raschka, *Build a Large Language Model (From Scratch)* (2024).
4. Hugging Face *LLM Course* (free), and the MLX-LM `LORA.md` guide.

**Papers:**

- Vaswani et al. 2017. Attention is all you need.
- Devlin et al. 2019. BERT. Raffel et al. 2020. Exploring the limits of transfer learning with a unified text-to-text
  transformer (T5). Wang et al. 2022. What language model architecture and pretraining objective work best for
  zero-shot generalization?
- Su et al. 2021. RoFormer: rotary position embedding. Zhang and Sennrich 2019. Root mean square layer
  normalization. Shazeer 2020. GLU variants improve Transformer. Ainslie et al. 2023. GQA.
- Loshchilov and Hutter 2019. Decoupled weight decay regularization (AdamW).
- Hoffmann et al. 2022. Training compute-optimal large language models (Chinchilla).
- Muennighoff et al. 2023. Scaling data-constrained language models.
- Hägele et al. 2024. Scaling laws and compute-optimal training beyond fixed training durations (WSD).
- Chowdhery et al. 2022. PaLM (z-loss). Wortsman et al. 2023. Small-scale proxies for large-scale Transformer
  training instabilities.
- Allen-Zhu and Li 2024. Physics of language models, part 3.3: knowledge capacity scaling laws.
- Lee et al. 2022. Deduplicating training data makes language models better. Broder 1997. On the resemblance and
  containment of documents (MinHash).
- Penedo et al. 2024. The FineWeb datasets. Soldaini et al. 2024. Dolma (and peS2o).
- Ouyang et al. 2022. Training language models to follow instructions (InstructGPT).
- Rafailov et al. 2023. Direct preference optimization.
- Shao et al. 2024. DeepSeekMath (GRPO). DeepSeek-AI 2025. DeepSeek-R1.
- Hu et al. 2021. LoRA. Dettmers et al. 2023. QLoRA. Hinton et al. 2015. Distilling the knowledge in a neural
  network.
- Kirchenbauer et al. 2023. A watermark for large language models. Bao et al. 2024. Fast-DetectGPT.
- Efron and Tibshirani 1993. An introduction to the bootstrap. Wilson 1927. Probable inference. Guo et al. 2017. On
  calibration of modern neural networks.
- Hendrycks et al. 2021. MMLU. Rein et al. 2023. GPQA. Jin et al. 2019. PubMedQA. Jin et al. 2021. MedQA.
- Liu et al. 2023. Visual instruction tuning (LLaVA). Chameleon Team 2024. Chameleon: mixed-modal early-fusion
  foundation models.
