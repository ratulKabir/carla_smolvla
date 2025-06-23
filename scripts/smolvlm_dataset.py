from PIL import Image

from datasets import load_dataset
from configs.smolvlm_config import DATASET_ID, SYS_MSG, NUM_TRAIN_SAMPLES, NUM_VAL_SAMPLES
from scripts.carla_vl_dataset import load_jsonl_dataset

def format_data(sample):
    # Ensure the image is a PIL.Image object in RGB mode
    image = sample["image"]
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")
    
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYS_MSG}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": sample["query"],
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["label"][0]}],
        },
    ]

def get_dataset(test=False):
    if "chartqa" in DATASET_ID.lower(): 
        train_dataset, eval_dataset, test_dataset = load_dataset(DATASET_ID, split=["train[:1%]", "val[:1%]", "test[:1%]"])

    elif "carla_vqa" in DATASET_ID.lower():
        eval_dataset = load_jsonl_dataset("datasets/vqa_val.jsonl", f"train[:{int(NUM_VAL_SAMPLES*2)}]")
        eval_len = len(eval_dataset)
        split_idx = eval_len // 2
        
        if not test:
            train_dataset = load_jsonl_dataset("datasets/vqa_train.jsonl", f"train[:{NUM_TRAIN_SAMPLES}]")
            eval_dataset = eval_dataset.select(range(split_idx))

        else:
            test_dataset = eval_dataset.select(range(split_idx, len(eval_dataset)))

    if not test:
        train_dataset = [format_data(sample) for sample in train_dataset]
        eval_dataset = [format_data(sample) for sample in eval_dataset]

        return train_dataset, eval_dataset, None
    else:
        test_dataset = [format_data(sample) for sample in test_dataset]

        return None, None, test_dataset

if __name__ == "__main__":
    train_data, eval_data, _ = get_dataset()
    print(f"Train samples: {len(train_data)}")
    print(f"Eval samples: {len(eval_data)}")
    # print(f"Test samples: {len(test_data)}")