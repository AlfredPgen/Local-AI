"""Download English Wikipedia (Hugging Face wikimedia/wikipedia, 20231101.en)
into %USERPROFILE%\\ai_training_data\\wikipedia, the folder data_prep.py
--wiki-dir reads (datasets.save_to_disk format).

Only that one copy is kept: the downloaded Parquet files and the Arrow cache
that datasets builds from them go to a temporary folder next to it, which is
deleted at the end (by default they would stay in hidden cache folders, about
30 GB more). The copy is written to <folder>.part and renamed when complete,
so an interrupted run never leaves a folder that looks finished.
"""
import gc
import os
import shutil
import sys

NEEDED_GB = 45  # at the peak: the Parquet download (~12 GB) + the Arrow cache (~20 GB), then cache + copy (~40 GB)
save_path = os.path.join(os.path.expanduser("~"), "ai_training_data", "wikipedia")


def _remove(folder):
    shutil.rmtree(folder, ignore_errors=True)
    if os.path.exists(folder):
        print(f"Could not delete all of {folder} (a file still in use?); delete it by hand.")


def main():
    if os.path.isfile(os.path.join(save_path, "dataset_dict.json")):
        print(f"Wikipedia is already in {save_path}; delete that folder to download it again.")
        return
    if os.path.isdir(save_path) and os.listdir(save_path):
        sys.exit(f"{save_path} holds files but no complete download (dataset_dict.json is missing): "
                 "delete or move them, then run this again.")
    parent = os.path.dirname(save_path)
    os.makedirs(parent, exist_ok=True)
    free_gb = shutil.disk_usage(parent).free / 1e9
    if free_gb < NEEDED_GB:
        sys.exit(f"Only {free_gb:.0f} GB free on the drive of {parent}; about {NEEDED_GB} GB are needed while "
                 "downloading (about 20 GB are kept at the end).")
    work = save_path + ".download"  # Parquet download and Arrow cache, deleted at the end
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface"), "hub")
    if os.path.isdir(os.path.join(hub, "datasets--wikimedia--wikipedia")):
        print(f"Using the Wikipedia files downloaded earlier into {hub} (delete its datasets--wikimedia--wikipedia "
              "folder afterwards to free about 12 GB).")
    else:
        # the Hub download goes where HF_HUB_CACHE points, read when huggingface_hub is imported: set it first
        hub = os.environ["HF_HUB_CACHE"] = os.path.join(work, "hub")
    from datasets import load_dataset

    print("Downloading Wikipedia dataset. This may take a long time...")
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", cache_dir=os.path.join(work, "datasets"))
    if hub.startswith(work):
        _remove(hub)  # the Arrow cache now holds everything; the Parquet files are not needed

    print(f"Saving dataset to {save_path}...")
    part = save_path + ".part"
    _remove(part)  # left by an interrupted run
    dataset.save_to_disk(part)  # Arrow format, read by data_prep.py --wiki-dir
    del dataset  # the cache files are memory-mapped: Windows deletes them only once they are closed
    gc.collect()
    if os.path.isdir(save_path) and not os.listdir(save_path):
        os.rmdir(save_path)  # an empty folder made by an earlier version of this script
    os.replace(part, save_path)
    _remove(work)
    print("Download and save complete!")


if __name__ == "__main__":
    main()
