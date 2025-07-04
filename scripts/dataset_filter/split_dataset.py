import json
import random

# Paths
input_path = "datasets/Town12_Rep0_1306_route0_01_11_23_54_47/vqa_train.jsonl"  # replace with your file
train_path = "datasets/Town12_Rep0_1306_route0_01_11_23_54_47/train.jsonl"
val_path = "datasets/Town12_Rep0_1306_route0_01_11_23_54_47/val.jsonl"
test_path = "datasets/Town12_Rep0_1306_route0_01_11_23_54_47/test.jsonl"

# Load data
with open(input_path, "r") as f:
    data = [json.loads(line) for line in f]

# Shuffle deterministically
random.seed(42)
random.shuffle(data)

# Split
n = len(data)
train_end = int(0.8 * n)
val_end = int(0.9 * n)

train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]

# Save
def save_jsonl(path, data):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

save_jsonl(train_path, train_data)
save_jsonl(val_path, val_data)
save_jsonl(test_path, test_data)

print(f"Total: {n}")
print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
