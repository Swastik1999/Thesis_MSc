# config.py

import os


class GlobalConfig:
    # Directory paths
    BASE_OUTPUT_DIR = "ReAG"

    # Default dataset path based on your directory structure
    TXT_FILE_PATH = os.path.join("Data", "combined_corpus.txt")
    RETRIEVAL_TEST_CSV_PATH = os.path.join("Data", "extracted_file.csv")

    # Hyperparameter defaults & ranges
    CHUNK_SIZE_RANGE = (600, 1200)
    CHUNK_OVERLAP_RANGE = (100, 500)
    TOP_K_RANGE = (10,50)
    FINAL_TOP_K_RANGE = (1, 20)
    TEMPERATURE_RANGE = (0.0, 0.5)

    # Models available in Ollama
    EMBEDDING_MODELS = ["qwen3-embedding:4b","qwen3-embedding:0.6b","IEITYuan/Yuan-embedding-2.0-en"]
    CRITIQUE_MODELS = ["llama3.2:3b","deepseek-r1:latest","qwen3.5:2b"]
    GENERATION_MODELS = ["deepseek-r1:latest"]

    NUM_THREAD_WORKERS = 4

    SPARSE_QUERY_PROCESSING = True  # Set to True to enable sparse query processing, False otherwise
    DENSE_QUERY_PROCESSING = False  # Set to True to enable dense query processing, False otherwise

    # Minimum token overlap (0-1) for a dense-retrieved chunk and a
    # sparse-retrieved chunk to be treated as near-duplicates and collapsed
    # into one before RRF fusion. See pipeline.deduplicate_dense_sparse().
    DEDUP_OVERLAP_THRESHOLD = 0.50

    # Master switch for whether the relevancy-filtering stage
    # (pipeline.filter_by_relevance()) runs at all. When False, no extra
    # embedding calls are made and RELEVANCE_SCORE_THRESHOLD is ignored
    # entirely -- chunks pass through unchanged, same as before this stage
    # existed. Defaults to False since this is a new, as-yet-untuned
    # filtering stage; flip to True once you've picked a sensible
    # RELEVANCE_SCORE_THRESHOLD via retrieval_eval.py.
    RELEVANCE_FILTERING_ENABLED = False

    # Minimum cosine-similarity relevance score a (post-dedup) chunk must
    # reach against the raw user query to survive filtering, computed with
    # the same embedding model used for dense retrieval. Only applied when
    # RELEVANCE_FILTERING_ENABLED is True. Score range is embedding-model
    # dependent -- 0.0 effectively passes everything through even when
    # enabled. Tune upward using retrieval_eval.py once you've seen the
    # score distribution for your embedding model.
    # See pipeline.filter_by_relevance().
    RELEVANCE_SCORE_THRESHOLD = 0.3

    RETRIEVAL_TEST_JSON_PATH = os.path.join("Data", "rag_evaluation_dataset_2.json")