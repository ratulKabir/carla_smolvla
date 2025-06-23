from PIL import Image

from datasets import load_dataset
from configs.smolvlm_config import DATASET_ID, SYS_MSG, NUM_TRAIN_SAMPLES, NUM_VAL_SAMPLES, STREAM
from scripts.carla_vl_dataset import load_jsonl_dataset


def get_dataset(test=False):
    if "chartqa" in DATASET_ID.lower(): 
        train_dataset, eval_dataset, test_dataset = load_dataset(
            DATASET_ID, split=["train[:1%]", "val[:1%]", "test[:1%]"]
        )

    elif "carla_vqa" in DATASET_ID.lower():
        eval_dataset = load_jsonl_dataset("datasets/vqa_val.jsonl", f"train[:{int(NUM_VAL_SAMPLES * 2)}]")
        eval_len = len(eval_dataset)
        split_idx = eval_len // 2

        if not test:
            if STREAM:
                train_dataset = load_jsonl_dataset("datasets/vqa_train.jsonl", streaming=STREAM).take(NUM_TRAIN_SAMPLES)
            else:
                # Load the sub training dataset if not streaming
                train_dataset = load_jsonl_dataset("datasets/vqa_train.jsonl", f"train[:{NUM_TRAIN_SAMPLES}]")
            eval_dataset = eval_dataset.select(range(split_idx))
        else:
            test_dataset = eval_dataset.select(range(split_idx, eval_len))

    # Define batched=True map step
    def map_fn(example):
        image_path = example["image"]

        # Skip invalid image paths
        if not isinstance(image_path, str) or not image_path.endswith((".jpg", ".png", ".jpeg")):
            print(f"Skipping invalid image: {image_path}")
            return {"messages": []}

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Failed to load image {image_path}: {e}")
            return {"messages": []}

        return {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYS_MSG}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": example["query"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": example["label"][0]}],
                },
            ]
        }


    if not test:
        train_dataset = train_dataset.map(map_fn)
        eval_dataset = eval_dataset.map(map_fn)
        return train_dataset, eval_dataset, None
    else:
        test_dataset = test_dataset.map(map_fn)
        return None, None, test_dataset

if __name__ == "__main__":
    train_data, eval_data, _ = get_dataset()
    print(f"Train samples: {len(train_data)}")
    print(f"Eval samples: {len(eval_data)}")
    # print(f"Test samples: {len(test_data)}")