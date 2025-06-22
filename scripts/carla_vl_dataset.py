import os
import gzip
import json
from datasets import load_dataset

def extract_samples(root_dir: str) -> list:
    """
    Recursively extract (image, query, label, human_or_machine, category) samples from all .json.gz files.
    """
    samples = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".json.gz"):
                json_path = os.path.join(dirpath, filename)
                with gzip.open(json_path, 'rt', encoding='utf-8') as f:
                    try:
                        content = json.load(f)
                        raw_image_path = content.get('image_paths', {}).get('CAM_FRONT')
                        if raw_image_path and raw_image_path.startswith("database/"):
                            image_path = raw_image_path.replace(
                                "database/simlingo_v2_2025_01_10/",
                                "/home/ratul/Workstation/ratul/simlingo/database/simlingo_1_data/"
                            )
                            qa_sections = content.get('QA', {})
                            count = 0
                            for category in qa_sections:
                                for qa in qa_sections[category]:
                                    question = qa.get('Q')
                                    answer = qa.get('A')
                                    if question and answer:
                                        samples.append({
                                            "image": image_path,
                                            "query": question,
                                            "label": [answer],
                                            "human_or_machine": 0,
                                            "category": category
                                        })
                                        count += 1
                            # print(f"✓ Processed {filename} — extracted {count} QAs")
                    except Exception as e:
                        print(f"✗ Error reading {json_path}: {e}")
    return samples

def save_dataset_jsonl(samples: list, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"✓ Saved {len(samples)} samples to {output_path}")

def load_jsonl_dataset(path: str):
    return load_dataset("json", data_files=path, split="train")

if __name__ == "__main__":
    # Set your root directories here
    train_root_dir = "/home/ratul/Workstation/ratul/simlingo/database/simlingo_1_data/drivelm_simlingo_training_1_scenario_routes_training_random_weather_seed_1_balanced_150_chunk_001"
    val_root_dir = "/home/ratul/Workstation/ratul/simlingo/database/simlingo_1_data/drivelm_simlingo_validation_1_scenario_routes_validation_random_weather_seed_2_balanced_150_chunk_001.tar.gz"

    # Output paths
    train_output = "datasets/vqa_train.jsonl"
    val_output = "datasets/vqa_val.jsonl"

    # Process and save
    train_samples = extract_samples(train_root_dir)
    save_dataset_jsonl(train_samples, train_output)
    print(f"Saved {len(train_samples)} training samples to {train_output}")

    val_samples = extract_samples(val_root_dir)
    save_dataset_jsonl(val_samples, val_output)
    print(f"Saved {len(val_samples)} validation samples to {val_output}")

    # (Optional) Load back the dataset for quick check
    # train_dataset = load_jsonl_dataset(train_output)
    # val_dataset = load_jsonl_dataset(val_output)

    # print("✓ Loaded training samples:", len(train_dataset))
    # print("✓ Loaded validation samples:", len(val_dataset))
    # print("Example sample:", train_dataset[0])
