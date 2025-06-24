import json

input_path = "datasets/vqa_train.jsonl"  # <-- update with your real file path
output_path = "datasets/vqa_train_Town12_Rep0_1392_route0_01_11_15_19_20.jsonl"
folder_key = "Town12_Rep0_1392_route0_01_11_15_19_20"

filtered_data = []

# Step 1: Read and filter
with open(input_path, "r") as f:
    for line in f:
        item = json.loads(line)
        if folder_key in item["image"]:
            filtered_data.append(item)

# Step 2: Print how many items found
print(f"Total entries in '{folder_key}': {len(filtered_data)}")

# Step 3: Save a small subset (e.g., first 10)
small_subset = filtered_data#[:10]
with open(output_path, "w") as f:
    for item in small_subset:
        f.write(json.dumps(item) + "\n")

print(f"Saved {len(small_subset)} samples to '{output_path}'")
