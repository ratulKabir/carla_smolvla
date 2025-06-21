from datasets import load_dataset
from configs.smolvlm_config import DATASET_ID, SYS_MSG

def format_data(sample):
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
                    "image": sample["image"],
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

def get_dataset():
    train_dataset, eval_dataset, test_dataset = load_dataset(DATASET_ID, split=["train[:1%]", "val[:1%]", "test[:1%]"])

    train_dataset = [format_data(sample) for sample in train_dataset]
    eval_dataset = [format_data(sample) for sample in eval_dataset]
    test_dataset = [format_data(sample) for sample in test_dataset]

    return train_dataset, eval_dataset, test_dataset


if __name__ == "__main__":
    train_data, eval_data, test_data = get_dataset()
    print(f"Train samples: {len(train_data)}")
    print(f"Eval samples: {len(eval_data)}")
    print(f"Test samples: {len(test_data)}")