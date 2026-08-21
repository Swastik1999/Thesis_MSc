# config.py

import os


class GlobalConfig:
    # Directory paths
    BASE_OUTPUT_DIR = "ReAG"

    # Default dataset path based on your directory structure
    TXT_FILE_PATH = os.path.join("Data", "combined_corpus.txt")

    # Hyperparameter defaults & ranges
    CHUNK_SIZE_RANGE = (800, 1200)
    CHUNK_OVERLAP_RANGE = (150, 200)
    TOP_K_RANGE = (20,50)
    FINAL_TOP_K_RANGE = (1, 20)
    TEMPERATURE_RANGE = (0.0, 0.5)

    # Models available in Ollama
    EMBEDDING_MODELS = ["qwen3-embedding:4b","qwen3-embedding:0.6b"]
    CRITIQUE_MODELS = ["llama3.2:3b","deepseek-r1:latest","qwen3.5:2b"]
    GENERATION_MODELS = ["deepseek-r1:latest"]
    EVAL_MODEL = ["llama3"]

    NUM_THREAD_WORKERS = 4