import io
import torch
from transformers import Idefics3ForConditionalGeneration, AutoProcessor
from scripts.smolvlm_utils import clear_memory, generate_text_from_sample, save_vqa_chat_vis
from peft import PeftModel, PeftConfig
from PIL import Image

from configs.smolvlm_config import MODEL_ID, OUTPUT_DIR
from scripts.smolvlm_dataset import get_dataset


pretrained_model_name_or_path = OUTPUT_DIR
train_data_id = 15  # Index of the training data sample to use for evaluation

# Clear memory before starting
clear_memory()
# Load the dataset
_, _, test_dataset = get_dataset(test=True)

model = Idefics3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2",
)

processor = AutoProcessor.from_pretrained(MODEL_ID)

output = generate_text_from_sample(model, processor, test_dataset[train_data_id])
print(f"Ground truth output:\n{test_dataset[train_data_id]['messages'][2]['content'][0]['text']}")
print(f"\nGenerated output before fine-tuning:\n{output}")

save_vqa_chat_vis(
    image=Image.open(io.BytesIO(test_dataset[train_data_id]['messages'][1]["content"][0]["image"]['bytes'])),
    query=test_dataset[train_data_id]['messages'][1]["content"][1]["text"],
    ground_truth=test_dataset[train_data_id]['messages'][2]["content"][0]["text"],
    prediction=output,
    save_path=f"outputs/figures/vqa_chat_vis_{train_data_id}.png",
)


model = PeftModel.from_pretrained(model, pretrained_model_name_or_path)
model.eval()


output = generate_text_from_sample(model, processor, test_dataset[train_data_id])
print(f"\nGenerated output after fine-tuning:\n{output}")

save_vqa_chat_vis(
    image=Image.open(io.BytesIO(test_dataset[train_data_id]['messages'][1]["content"][0]["image"]['bytes'])),
    query=test_dataset[train_data_id]['messages'][1]["content"][1]["text"],
    ground_truth=test_dataset[train_data_id]['messages'][2]["content"][0]["text"],
    prediction=output,
    save_path=f"outputs/figures/vqa_chat_vis_{train_data_id}_trained.png",
)