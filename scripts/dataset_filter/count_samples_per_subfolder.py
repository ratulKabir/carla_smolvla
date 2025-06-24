import json
import os
from collections import defaultdict

dataset_path = "datasets/vqa_train.jsonl"  # <-- path to your dataset file
base_folder = "/home/ratul/Workstation/ratul/simlingo/database/simlingo_1_data/data/simlingo/training_1_scenario/routes_training/random_weather_seed_1_balanced_150"

counts = defaultdict(int)

# Read dataset and count by subfolder
with open(dataset_path, "r") as f:
    for line in f:
        sample = json.loads(line)
        image_path = sample["image"]
        # Extract subfolder path after the base folder
        if image_path.startswith(base_folder):
            rel_path = image_path[len(base_folder):].strip("/")
            subfolder = rel_path.split("/")[0]
            counts[subfolder] += 1

# Print counts per subfolder
print("\nSample counts per subfolder:")
for folder, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{folder}: {count}")

# Find and show the one with the most samples
max_folder = max(counts, key=counts.get)
print(f"\n📌 Highest sample count is in: {max_folder} ({counts[max_folder]} samples)")
