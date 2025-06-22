import torch
from transformers import Idefics3ForConditionalGeneration, AutoProcessor
from scripts.smolvlm_utils import clear_memory, generate_text_from_sample
from peft import PeftModel, PeftConfig

from configs.smolvlm_config import MODEL_ID, DATASET_ID
from scripts.smolvlm_dataset import get_dataset

pretrained_model_name_or_path = "outputs/smolvlm-256m-instruct-trl-sft-ChartQA"
train_data_id = 10  # Index of the training data sample to use for evaluation

# Clear memory before starting
clear_memory()
# Load the dataset
_, _, test_dataset = get_dataset()

model = Idefics3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2",
)

processor = AutoProcessor.from_pretrained(MODEL_ID)

output = generate_text_from_sample(model, processor, test_dataset[train_data_id])
print(f"Generated output before fine-tuning:\n {output}")


model = PeftModel.from_pretrained(model, pretrained_model_name_or_path)
model.eval()


output = generate_text_from_sample(model, processor, test_dataset[train_data_id])
print(f"\nGenerated output after fine-tuning:\n {output}")