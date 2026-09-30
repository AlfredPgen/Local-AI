r"""The tinyGPT pipeline in a few simple commands.

    python run_pipeline.py estimate            # optional: how big will the dataset be? (~7 min)
    python run_pipeline.py build               # 1. build the dataset from ai_training_data (~2 hours)
    python run_pipeline.py plan --hours 48     # 2. which model fits in 48 hours? (prints the plan, trains nothing)
    python run_pipeline.py train --hours 48    # 3. train it
    python run_pipeline.py resume              #    continue after Ctrl+C or a shutdown
    python run_pipeline.py all --hours 48      # 1 + 2 + 3, asking before the training starts
    python run_pipeline.py continue --hours 24 # later, after adding more text: rebuild with the same
                                               # tokenizer and train further from the first model's weights

Every step prints the full command it runs, so you can also copy it and change
options by hand (HOW_TO_RUN.md explains them). The settings below say where
the data is, what the dataset is called and what the model run is called.

How the model is chosen: the planner in tiny_gpt.py takes the largest model
that can still read about 20 tokens of text per parameter in the hours you give,
using the speed this computer reached in its last training run.
"""

import argparse
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------- settings: edit these ----------------
DATA = os.path.join(os.path.expanduser("~"), "ai_training_data")  # the folder with all training text
DATASET = os.path.join(HERE, "datasets", "bio_v4")      # the dataset folder that 'build' creates
RUN_NAME = "tinyGPT"                                    # checkpoints are <RUN_NAME>.pt and <RUN_NAME>_best.pt
KEYWORDS = os.path.join(HERE, "keywords_biology.txt")   # topic filter for Wikipedia, FineWeb and peS2o
WIKI_MAX_DOCS = 200_000   # Wikipedia articles kept, best keyword matches first; 0 = all that pass (~465,000)
# -------------------------------------------------------


def build_command():
    cmd = [sys.executable, "data_prep.py", "--out", DATASET, "--text-root", DATA]
    wiki = os.path.join(DATA, "wikipedia")
    if os.path.isdir(wiki):
        cmd += ["--wiki-dir", wiki, "--wiki-max-docs", str(WIKI_MAX_DOCS)]
    fineweb = os.path.join(DATA, "web", "fineweb-edu", "sample", "10BT", "*.parquet")
    if glob.glob(fineweb):
        cmd += ["--parquet", fineweb, "--parquet-name", "fineweb", "--min-score", "3"]
    pes2o = os.path.join(DATA, "web", "pes2o", "data", "v2", "*.json.gz")
    if glob.glob(pes2o):
        cmd += ["--jsonl", pes2o, "--jsonl-name", "pes2o"]
    return cmd + ["--include-keywords", KEYWORDS, "--keyword-min-distinct", "3",
                  "--tokenizer-weights", "books=3,articles=3", "--near-dup", "drop"]


def train_command(hours, name, plan=False):
    return ([sys.executable, "tiny_gpt.py", "--dataset", DATASET, "--name", name, "--time-budget-hours", str(hours)]
            + (["--plan"] if plan else []))


def run(cmd):
    print("\n> " + " ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n", flush=True)
    return subprocess.call(cmd, cwd=HERE)


def dataset_ready():
    return os.path.isfile(os.path.join(DATASET, "manifest.json"))


def gpu_busy():
    """True if another job holds GPU memory (a training run uses several GB)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return float(out.strip().splitlines()[0]) > 2500
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return False


def step_build():
    if dataset_ready():
        print(f"The dataset is already built: {DATASET}\n(To rebuild, delete that folder or change DATASET at the "
              "top of run_pipeline.py.)")
        return 0
    print(f"Building {DATASET} from {DATA}. This takes about 2 hours; the report is {DATASET}\\report.md.")
    return run(build_command())


def step_plan(hours, name):
    if not dataset_ready():
        print("Build the dataset first: python run_pipeline.py build")
        return 1
    return run(train_command(hours, name, plan=True))


def step_train(hours, name):
    if not dataset_ready():
        print("Build the dataset first: python run_pipeline.py build")
        return 1
    if gpu_busy():
        print("The GPU is busy (another training run?). Two GPU jobs at once can crash the graphics driver; "
              "wait for the other job or stop it first.")
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
        print(f"The earlier run's dataset {DATASET} is needed for its tokenizer; it is missing.")
        return 1
    if not os.path.isfile(os.path.join(new_dataset, "manifest.json")):
        cmd = build_command()
        cmd[cmd.index("--out") + 1] = new_dataset
        print(f"Building {new_dataset} from all of {DATA} (old and new text, so the model does not forget), with the "
              f"tokenizer and the train/validation split of {DATASET}.")
        if run(cmd + ["--tokenizer-from", DATASET, "--keep-split-from", DATASET]):
            return 1
    if gpu_busy():
        print("The GPU is busy (another training run?); wait for it or stop it first.")
        return 1
    return run([sys.executable, "tiny_gpt.py", "--dataset", new_dataset, "--init-from", base, "--name",
                f"{name}_continued", "--time-budget-hours", str(hours)])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=("estimate", "build", "plan", "train", "resume", "continue", "all"))
    p.add_argument("--hours", type=float, default=24, help="training time you are willing to wait (default 24)")
    p.add_argument("--name", default=RUN_NAME, help=f"name of the model run (default {RUN_NAME})")
    p.add_argument("--new-dataset", default=os.path.join(HERE, "datasets", "bio_v5"),
                   help="continue: the new dataset to build (default datasets\\bio_v5)")
    args = p.parse_args()
    if args.step == "continue":
        return step_continue(args.hours, args.name, args.new_dataset)
    if args.step == "estimate":
        return run([sys.executable, "estimate_dataset.py", "--hours", "12,24,48,72"])
    if args.step == "build":
        return step_build()
    if args.step == "plan":
        return step_plan(args.hours, args.name)
    if args.step == "train":
        return step_train(args.hours, args.name)
    if args.step == "resume":
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
