import os
from datasets import load_dataset

# Define your target directory (using raw string to handle Windows backslashes)
save_path = os.path.join(os.path.expanduser("~"), "ai_training_data", "wikipedia")

# Create the directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

print("Downloading Wikipedia dataset. This may take a long time...")
# Load the dataset from Hugging Face (this downloads it to the cache first)
dataset = load_dataset("wikimedia/wikipedia", "20231101.en")

print(f"Saving dataset to {save_path}...")
# Save it to your specified directory in Arrow format
dataset.save_to_disk(save_path)

print("Download and save complete!")