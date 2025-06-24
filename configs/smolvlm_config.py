# DATASET_ID = "HuggingFaceM4/ChartQA"
DATASET_ID = "carla_vqa"

TRAIN_DATASET_ID = "datasets/Town12_Rep0_1306_route0_01_11_23_54_47/train.jsonl"
VAL_DATASET_ID = "datasets/Town12_Rep0_1306_route0_01_11_23_54_47/val.jsonl"
TEST_DATASET_ID = "datasets/Town12_Rep0_1306_route0_01_11_23_54_47/test.jsonl"

NUM_TRAIN_SAMPLES = 10000  # Number of samples to use for training
NUM_VAL_SAMPLES = 64  # Number of samples to use for validation

# SYS_MSG for chartQA
# SYS_MSG = """You are a Vision Language Model specialized in interpreting visual data from chart images.
# Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
# The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
# Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

# SYS_MSG for carla_vqa
SYS_MSG = """You are a Vision Language Model specialized in interpreting driving scenes from vehicle-mounted camera images.
Your task is to analyze the visual input and respond to questions about the current road situation, such as lane markings, obstacles, traffic lights, and vehicle behavior.
Provide clear and concise answers.
Base your answers strictly on what is visible in the image. Do not speculate or provide extra explanations."""

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

# OUTPUT_DIR = "outputs/smolvlm-256m-instruct-trl-sft-ChartQA"
OUTPUT_DIR = "outputs/smolvlm-256m-instruct-trl-sft-carla-vqa-one-scene"

RANDOM_SEED = 42

STREAM = False

BATCH_SIZE = 2  # Adjust based on your GPU memory