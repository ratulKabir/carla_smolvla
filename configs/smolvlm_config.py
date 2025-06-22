DATASET_ID = "HuggingFaceM4/ChartQA"
SYS_MSG = """You are a Vision Language Model specialized in interpreting visual data from chart images.
Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

OUTPUT_DIR = "outputs/smolvlm-256m-instruct-trl-sft-ChartQA"