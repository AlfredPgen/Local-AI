"""Continue a tinyGPT training run on a cloud GPU at Modal (modal.com).

The run, its dataset and its checkpoints live on a Modal volume named
"tinygpt"; the code is copied from this folder at every launch. One session
lasts at most --hours (Modal ends any job after 24 h): the trainer is then
stopped like Ctrl+C, saves its training state, and the next launch resumes it.

One-time setup (in this folder):
    pip install modal
    modal setup
    modal volume create tinygpt
    modal volume put tinygpt datasets/bio_all /datasets/bio_all
    modal volume put tinygpt tinyGPT_muon.pt /run/tinyGPT_muon.pt
    modal volume put tinygpt tinyGPT_muon_metrics.jsonl /run/tinyGPT_muon_metrics.jsonl

Short test (30 steps), then a full session that keeps running when this
computer is off:
    modal run modal_train.py --steps 30
    modal run --detach modal_train.py --hours 23

Watch and fetch (never over the local checkpoint):
    modal app logs tinygpt
    modal volume get tinygpt /run/log.txt cloud/
    modal volume get tinygpt /run/tinyGPT_muon.pt cloud/
"""

import os

import modal

GPU = "H100"
NAME = "tinyGPT_muon"
DATASET = "/vol/datasets/bio_all"
RUN_DIR = "/vol/run"
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = ["tiny_gpt.py", "data_prep.py", "view_pt.py", "probes_biology.tsv"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("gcc", "g++")  # Triton builds a small launcher with the C compiler when torch.compile runs
    .pip_install("torch==2.10.0", "numpy", "sentencepiece", "psutil")
)
for name in CODE:
    image = image.add_local_file(os.path.join(HERE, name), f"/app/{name}")

app = modal.App("tinygpt")
volume = modal.Volume.from_name("tinygpt", create_if_missing=True)


@app.function(image=image, gpu=GPU, volumes={"/vol": volume}, timeout=24 * 3600, max_containers=1)
def train(extra, hours):
    import signal
    import subprocess
    import time

    env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR="/vol/compile_cache", PYTHONUNBUFFERED="1")
    cmd = ["python", "/app/tiny_gpt.py", "--dataset", DATASET, "--name", NAME, "--resume",
           "--out-dir", RUN_DIR, "--log-file", f"{RUN_DIR}/log.txt", "--dashboard", "off"] + extra
    print(" ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd="/app", env=env)
    deadline = time.time() + hours * 3600
    last_commit = time.time()
    stopping = False
    while proc.poll() is None:
        time.sleep(20)
        if not stopping and time.time() > deadline:
            print(f"--hours {hours} reached: stopping the trainer (it saves its state)", flush=True)
            proc.send_signal(signal.SIGINT)
            stopping = True
        if time.time() - last_commit > 900:
            volume.commit()  # so a crash of the container loses at most 15 minutes of saved files
            last_commit = time.time()
    volume.commit()
    print(f"trainer exited with code {proc.returncode}", flush=True)
    return proc.returncode


@app.local_entrypoint()
def main(hours: float = 23.0, steps: int = 0, batch: int = 2, accum: int = 132):
    """--steps N ends after N steps (a test); --batch/--accum split the 264 sequences of a step."""
    if batch * accum != 264:
        raise SystemExit(f"--batch x --accum must be 264 (the run's sequences per step), not {batch * accum}")
    extra = ["--batch-size", str(batch), "--grad-accum", str(accum)]
    if steps:
        extra += ["--stop-after-steps", str(steps)]
    code = train.remote(extra, min(hours, 23.5))
    print("finished" if code == 0 else f"the trainer ended with code {code}: see the log above")
