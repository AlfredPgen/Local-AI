r"""The tinyGPT pipeline in a few simple commands.

    python run_pipeline.py estimate            # optional: how big will the dataset be? (~7 min)
    python run_pipeline.py build               # 1. build the dataset from ai_training_data (~2 hours)
    python run_pipeline.py plan --hours 48     # 2. which model fits in 48 hours? (prints the plan, trains nothing)
    python run_pipeline.py train --hours 48    # 3. train it
    python run_pipeline.py resume              #    continue after Ctrl+C or a shutdown
    python run_pipeline.py all --hours 48      # 1 + 2 + 3, asking before the training starts
    python run_pipeline.py continue --hours 24 # later, after adding more text: rebuild with the same
                                               # tokenizer and train further from the first model's weights
    python run_pipeline.py posttrain           # 4. teach the pre-trained model to answer questions (SFT, then
                                               # DPO, then a before/after evaluation; posttrain.py)

Every step prints the full command it runs, so you can also copy it and change
options by hand (HOW_TO_RUN.md explains them). The settings below say where
the data is, what the dataset is called and what the model run is called.

How the model is chosen: the planner in tiny_gpt.py takes the largest model
that can still read about 20 tokens of text per parameter in the hours you give,
using the speed this computer reached in its last training run.
"""

import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tiny_gpt  # noqa: E402

# ---------------- settings: edit these ----------------
DATA = os.path.join(os.path.expanduser("~"), "ai_training_data")  # the folder with all training text
DATASET = os.path.join(HERE, "datasets", "bio_all")     # the dataset folder that 'build' creates
RUN_NAME = "tinyGPT_muon"                               # checkpoints are <RUN_NAME>.pt and <RUN_NAME>_best.pt
KEYWORDS = os.path.join(HERE, "keywords.txt")   # topic filter for Wikipedia, FineWeb and peS2o
WIKI_MAX_DOCS = 0         # Wikipedia articles kept, best keyword matches first; 0 = all that pass (611,091)
SUPERBPE = True           # True: SuperBPE tokenizer, with tokens that span words ("of the"); fewer tokens per text
OPTIMIZER = "muon"        # "muon": Muon for the block matrices, AdamW for the rest (new runs only)
# -------------------------------------------------------
# Once DATASET is built, its manifest's data settings win over the ones above for estimate and continue, so both
# describe the dataset that exists rather than constants that may have been edited since.

# data_prep's former --near-dup-max-docs default: manifests written before it became 0 (no limit) record it although
# nobody chose it, and repeating it would stop a larger rebuild after pass 1
_OLD_NEAR_DUP_DEFAULT = 3_000_000


def manifest_prep_args(dataset=None):
    """The data_prep settings recorded in a built dataset's manifest.json
    (prep_args), {} when there is none or it cannot be read."""
    path = os.path.join(dataset or DATASET, "manifest.json")
    try:
        with open(path, encoding="utf-8") as f:
            args = json.load(f).get("prep_args")
    except (OSError, ValueError, AttributeError):
        return {}
    return args if isinstance(args, dict) else {}


def build_command():
    """The data_prep command that builds DATASET. Where DATASET already has a
    manifest, its recorded superbpe, tokenizer_weights, wiki_max_docs and
    near_dup_max_docs are used instead of the constants above (estimate then
    counts the tokens of the tokenizer that exists: SuperBPE gives ~13 % fewer)."""
    prep = manifest_prep_args()
    superbpe = bool(prep.get("superbpe", SUPERBPE))
    wiki_max = int(prep.get("wiki_max_docs", WIKI_MAX_DOCS) or 0)
    near_dup_max = int(prep.get("near_dup_max_docs", 0) or 0)
    if near_dup_max == _OLD_NEAR_DUP_DEFAULT:
        near_dup_max = 0
    cmd = [sys.executable, "data_prep.py", "--out", DATASET, "--text-root", DATA]
    wiki = os.path.join(DATA, "wikipedia")
    if os.path.isdir(wiki):
        cmd += ["--wiki-dir", wiki, "--wiki-max-docs", str(wiki_max)]
    fineweb = os.path.join(DATA, "web", "fineweb-edu", "sample", "10BT", "*.parquet")
    if glob.glob(fineweb):
        cmd += ["--parquet", fineweb, "--parquet-name", "fineweb", "--min-score", "3"]
    pes2o = os.path.join(DATA, "web", "pes2o", "data", "v2", "*.json.gz")
    if glob.glob(pes2o):
        cmd += ["--jsonl", pes2o, "--jsonl-name", "pes2o"]
    # books and articles count 3x when the tokenizer is trained; only those sub-folders that exist (data_prep
    # refuses a weight for a source it does not read)
    try:
        folders = {n.lower() for n in os.listdir(DATA) if os.path.isdir(os.path.join(DATA, n))}
    except OSError:
        folders = set()
    weights = ",".join(f"{n}=3" for n in ("books", "articles") if n in folders)
    if prep.get("tokenizer_weights"):
        # the recorded weights, minus sources this command no longer reads (data_prep refuses a weight it cannot
        # apply): the text-root folders plus the named web sources added above
        sources = folders | {name for name, flag in (("wiki", "--wiki-dir"), ("fineweb", "--parquet"),
                                                     ("pes2o", "--jsonl")) if flag in cmd}
        recorded = [w.strip() for w in str(prep["tokenizer_weights"]).split(",") if w.strip()]
        weights = ",".join(w for w in recorded if w.split("=")[0].strip().lower() in sources)
    return cmd + ["--include-keywords", KEYWORDS, "--keyword-min-distinct", "3"] + (
        ["--tokenizer-weights", weights] if weights else []) + [
        "--near-dup", "drop"] + (["--near-dup-max-docs", str(near_dup_max)] if near_dup_max else []) + (
        ["--superbpe"] if superbpe else [])


def checkpoint_optimizer(path):
    """The optimizer a checkpoint was trained with (its train_config), read with
    weights_only=True; OPTIMIZER when the file records none or cannot be read."""
    try:
        obj = tiny_gpt.safe_load(path, mmap=True)
        found = (obj.get("train_config") or {}).get("optimizer")
    except (SystemExit, Exception):  # noqa: BLE001 - an unreadable file only loses the hint
        found = None
    return found if found in ("adamw", "muon") else OPTIMIZER


def train_command(hours, name, plan=False):
    return ([sys.executable, "tiny_gpt.py", "--dataset", DATASET, "--name", name, "--time-budget-hours", str(hours),
             "--optimizer", OPTIMIZER] + (["--plan"] if plan else []))


def run(cmd):
    print("\n> " + " ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n", flush=True)
    return subprocess.call(cmd, cwd=HERE)


def dataset_ready():
    return os.path.isfile(os.path.join(DATASET, "manifest.json"))


def gpu_busy():
    """tiny_gpt.gpu_busy for this folder's runs: a live training lock on any device, or
    GPU memory/utilization where nvidia-smi exists (kept here so the other scripts and
    the tests can refer to run_pipeline.gpu_busy)."""
    return tiny_gpt.gpu_busy(HERE)


def step_build():
    if dataset_ready():
        print(f"The dataset is already built: {DATASET}\n(To rebuild, delete that folder or change DATASET at the "
              "top of run_pipeline.py. A rebuild trains a new tokenizer (32,768 pieces for this much text): existing "
              "models can only be continued with `continue`, not resumed.)")
        return 0
    print(f"Building {DATASET} from {DATA}. This takes about 2 hours; the report is {DATASET}\\report.md.")
    return run(build_command())


GPU_BUSY = tiny_gpt.GPU_BUSY_MESSAGE


def step_plan(hours, name):
    if not dataset_ready():
        print("Build the dataset first: python run_pipeline.py build")
        return 1
    if gpu_busy():  # the planner times training steps on the GPU; next to another job that is risky and misleading
        print(GPU_BUSY + " (The plan measures training speed on the GPU.)")
        return 1
    return run(train_command(hours, name, plan=True))


def step_train(hours, name):
    if not dataset_ready():
        print("Build the dataset first: python run_pipeline.py build")
        return 1
    if gpu_busy():
        print(GPU_BUSY)
        return 1
    if os.path.exists(os.path.join(HERE, f"{name}.pt")) or os.path.exists(os.path.join(HERE, f"{name}_best.pt")):
        print(f"A run called '{name}' already exists. Continue it with: python run_pipeline.py resume --name {name}\n"
              f"or start a new one with another name: python run_pipeline.py train --hours {hours} --name {name}_2")
        return 1
    return run(train_command(hours, name))


def step_continue(hours, name, new_dataset):
    """Build a new dataset with the SAME tokenizer (so the trained weights still
    fit), then train further from the best checkpoint of the earlier run."""
    base = os.path.join(HERE, f"{name}_best.pt")
    if not os.path.isfile(base):
        print(f"No earlier model {base}: train one first (python run_pipeline.py train --hours N).")
        return 1
    if not dataset_ready():
        print(f"The earlier run's dataset {DATASET} is needed for its train/validation split; it is missing.")
        return 1
    if not os.path.isfile(os.path.join(new_dataset, "manifest.json")):
        cmd = build_command()
        cmd[cmd.index("--out") + 1] = new_dataset
        print(f"Building {new_dataset} from all of {DATA} (old and new text, so the model does not forget), with the "
              f"tokenizer and the train/validation split of {DATASET}.")
        if "--superbpe" in cmd:  # the reused tokenizer already has its SuperBPE pieces; data_prep refuses both
            cmd.remove("--superbpe")
        # the tokenizer comes from the model itself (its embedding belongs to that vocabulary), not from the dataset
        # folder, which may have been rebuilt with another vocabulary since
        if run(cmd + ["--tokenizer-from", base, "--keep-split-from", DATASET]):
            return 1
    if gpu_busy():
        print(GPU_BUSY)
        return 1
    # the optimizer the earlier model was trained with, not the constant (which may describe another run)
    return run([sys.executable, "tiny_gpt.py", "--dataset", new_dataset, "--init-from", base, "--name",
                f"{name}_continued", "--time-budget-hours", str(hours), "--optimizer", checkpoint_optimizer(base)])


def step_posttrain(name):
    """Post-training of <name>_best.pt: data (once), SFT, DPO, evaluation."""
    base = os.path.join(HERE, f"{name}_best.pt")
    if not os.path.isfile(base):
        print(f"No pre-trained model {base}: train one first (python run_pipeline.py train --hours N).")
        return 1
    data = os.path.join(HERE, "posttrain_data")
    # sources.md is written last: without it (or any other file) the data build stopped partway
    if not all(os.path.isfile(os.path.join(data, f)) for f in ("sft.jsonl", "dpo.jsonl", "eval_mcq.jsonl",
                                                                "sources.md")):
        if run([sys.executable, "posttrain.py", "data"]):
            return 1
    if gpu_busy():
        print(GPU_BUSY)
        return 1
    return run([sys.executable, "posttrain.py", "run", "--base", base, "--name", name])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=("estimate", "build", "plan", "train", "resume", "continue", "posttrain", "all"))
    p.add_argument("--hours", type=float, default=24, help="training time you are willing to wait (default 24)")
    p.add_argument("--name", default=RUN_NAME, help=f"name of the model run (default {RUN_NAME})")
    p.add_argument("--new-dataset", default=os.path.join(HERE, "datasets", "bio_all_v2"),
                   help="continue: the new dataset to build (default datasets\\bio_all_v2)")
    args = p.parse_args()
    if args.step == "continue":
        return step_continue(args.hours, args.name, args.new_dataset)
    if args.step == "posttrain":
        return step_posttrain(args.name)
    if args.step == "estimate":
        # the same data settings as 'build' (everything after data_prep.py --out <folder>)
        return run([sys.executable, "estimate_dataset.py"] + build_command()[4:])
    if args.step == "build":
        return step_build()
    if args.step == "plan":
        return step_plan(args.hours, args.name)
    if args.step == "train":
        return step_train(args.hours, args.name)
    if args.step == "resume":
        if gpu_busy():  # perhaps the run itself is still training: a second copy would overwrite its checkpoints
            print(GPU_BUSY + f" If '{args.name}' is still training, it does not need resuming.")
            return 1
        return run([sys.executable, "tiny_gpt.py", "--name", args.name, "--resume"])
    # all
    if step_build() or step_plan(args.hours, args.name):
        return 1
    answer = input(f"\nStart training this model for about {args.hours:g} hours? [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        print("Not started. Later: python run_pipeline.py train --hours", f"{args.hours:g}")
        return 0
    return step_train(args.hours, args.name)


if __name__ == "__main__":
    sys.exit(main())
