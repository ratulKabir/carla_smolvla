import gc
import time
import torch
import os
import textwrap
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def clear_memory():
    # Delete variables if they exist in the current global scope
    if "inputs" in globals():
        del globals()["inputs"]
    if "model" in globals():
        del globals()["model"]
    if "processor" in globals():
        del globals()["processor"]
    if "trainer" in globals():
        del globals()["trainer"]
    if "peft_model" in globals():
        del globals()["peft_model"]
    if "bnb_config" in globals():
        del globals()["bnb_config"]
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")



def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        sample[1:2], add_generation_prompt=True  # Use the sample without the system message
    )

    image_inputs = []
    image = sample[1]["content"][0]["image"]
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_inputs.append([image])

    # Prepare the inputs for the model
    model_inputs = processor(
        # text=[text_input],
        text=text_input,
        images=image_inputs,
        return_tensors="pt",
    ).to(
        device
    )  # Move inputs to the specified device

    # Generate text with the model
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]

    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]  # Return the first decoded output text


def save_vqa_chat_vis(image: Image.Image, query: str, ground_truth: str, prediction: str, save_path: str):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    # Left: Image
    ax_img = fig.add_axes([0.05, 0.1, 0.4, 0.8])
    ax_img.imshow(image)
    ax_img.axis('off')

    # Right: Chat-style text
    ax_text = fig.add_axes([0.5, 0.1, 0.45, 0.8])
    ax_text.axis('off')

    def draw_bubble(label, text, y, side="left", color="#eeeeee", text_color="#000"):
        # Draw label
        label_y = y
        x_label = 0.0 if side == "left" else 0.99
        ax_text.text(
            x_label,
            label_y,
            label,
            fontsize=9,
            va='top',
            ha='left' if side == "left" else "right",
            color="#444",
            fontweight='bold'
        )

        # Compute wrapped text
        wrapped = textwrap.fill(text, width=40)
        text_height = 0.05 + 0.018 * wrapped.count("\n")
        
        # Draw bubble
        y -= 0.035  # spacing after label
        ax_text.text(
            x_label,
            y,
            wrapped,
            fontsize=10,
            va='top',
            ha='left' if side == "left" else "right",
            color=text_color,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=color, edgecolor='gray'),
        )

        return y - text_height - 0.04  # spacing after bubble

    y = 1.0
    y = draw_bubble("User", query, y, side="right", color="#d1e7dd")
    y = draw_bubble("Assistant (GT)", ground_truth, y, side="left", color="#f8d7da")
    y = draw_bubble("Assistant (Pred)", prediction, y, side="left", color="#cfe2ff")

    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
