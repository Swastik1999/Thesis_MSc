# retrieval_eval.py
#
# Evaluates BOTH retrieval and generation:
#   - Retrieval: Dense / Sparse / Hybrid (RRF) / Final (critique-filtered) hit rates,
#     tested against the "context" chunk(s) from the JSON dataset.
#   - Generation: for every successful FINAL retrieval, an answer is generated from
#     the final chunks and compared with the JSON "ground_truth" answer.

import csv
import json
import math
import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
from bert_score import BERTScorer  # pip install bert-score

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# Import GlobalConfig from config.py
from config import GlobalConfig

# Import existing functions/classes from pipeline.py
from pipeline import (
    load_txt_and_build_vectorstore,
    reciprocal_rank_fusion,
    deduplicate_dense_sparse,
    generate_hyde_document,
    expand_query,
    filter_by_relevance,
    resolve_embedder,
)
from prompts import CRITIQUE_VARIANTS, FILTER_VARIANTS, RAG_VARIANTS

# ==========================================
# CONFIGURATION - Imported from config.py
# ==========================================
# Path to the evaluation JSON file (list of objects with id/question/ground_truth/context/...)
# Falls back to a default path if not defined in config.py.
JSON_PATH = getattr(GlobalConfig, "RETRIEVAL_TEST_JSON_PATH", "Data/eval_questions.json")
TXT_PATH = GlobalConfig.TXT_FILE_PATH

# Model configurations from GlobalConfig
EMBEDDING_MODEL = GlobalConfig.EMBEDDING_MODELS[1]  # "qwen3-embedding:0.6b"
CRITIQUE_MODEL = GlobalConfig.CRITIQUE_MODELS[2]    # "qwen3.5:2b"

# Model used to generate the HyDE hypothetical-answer passage for dense
# retrieval. Falls back to pipeline.py's default if not defined in config.
QUERY_PROCESSOR_MODEL = getattr(GlobalConfig, "QUERY_PROCESSOR_MODEL", "qwen3.5:2b")

# Model + prompt used to generate the final answer (same as pipeline.generate_single_response).
GENERATION_MODEL = GlobalConfig.GENERATION_MODELS[0]   # "deepseek-r1:latest"
RAG_VARIANT_KEY = "rag_v1"
GENERATION_TEMPERATURE = 0.2
# The RAG prompt tells the model to reply with a sentence containing this phrase when the
# context lacks the answer. Used to count "retrieval hit but model refused" separately.
REFUSAL_MARKER = "does not contain enough information"

# Chunking & Retrieval hyperparameters
CHUNK_SIZE = 600     # 800
CHUNK_OVERLAP = 200  # 150
TOP_K = 15           # 20
FINAL_TOP_K = 7      # Set to 5

# Critique criteria come from prompts.py (same as production). They are injected into the
# FILTER_VARIANTS template via its {critique_criteria} placeholder inside retrieve_all_stages().
CRITIQUE_VARIANT_KEY = "critique_v1"
FILTER_VARIANT_KEY = "filter_v1"
CRITIQUE_PROMPT = CRITIQUE_VARIANTS[CRITIQUE_VARIANT_KEY]
OVERLAP_THRESHOLD = 0.50  # 50% minimum token overlap requirement (context hit check)

# If True, a retrieved chunk also counts as a hit when >= OVERLAP_THRESHOLD of ITS OWN
# tokens appear in the gold context chunk (i.e. the retrieved chunk is a sub-span of
# the gold chunk). This matters when your retrieval CHUNK_SIZE is smaller than the
# chunks used to build the dataset: a 600-char retrieved chunk can never cover 50% of a
# 1500-char gold chunk, so the original one-directional check would always miss.
SYMMETRIC_MATCH = True

# Minimum token overlap for a dense chunk and a sparse chunk to be treated
# as near-duplicates before RRF fusion. See pipeline.deduplicate_dense_sparse().
DEDUP_OVERLAP_THRESHOLD = getattr(GlobalConfig, "DEDUP_OVERLAP_THRESHOLD", 0.50)

USE_HYDE = GlobalConfig.DENSE_QUERY_PROCESSING          # Embed a hypothetical answer instead of the raw query (dense)
USE_QUERY_EXPANSION = GlobalConfig.SPARSE_QUERY_PROCESSING  # Append LLM-generated related terms to the query (sparse)
USE_RELEVANCE_FILTERING = GlobalConfig.RELEVANCE_FILTERING_ENABLED  # Embedding-based relevance cutoff before RRF

# Generation monitoring
EVALUATE_GENERATION = True
BERTSCORE_MODEL = "roberta-large"       # bert_score default for lang="en"; e.g. "microsoft/deberta-xlarge-mnli" correlates better with humans but is heavier
BERTSCORE_LANG = "en"
BERTSCORE_RESCALE_WITH_BASELINE = True   # rescales scores to a readable range; needs a baseline for model+lang (built in for roberta-large/en)
BERTSCORE_BATCH_SIZE = 8
# BERTScore F1 (generated vs ground-truth answer) at or above this value counts as a
# "good" generation. With rescale_with_baseline=True scores are spread out so that
# ~0 is an unrelated pair and ~1 is near-identical; 0.50 is an UNTUNED starting point.
# Look at the score distribution from a first run and adjust.
BERTSCORE_THRESHOLD = 0.50

# Output paths
FAILED_RETRIEVALS_PATH = "Data/failed_retrievals.json"
FAILED_RETRIEVALS_SUMMARY_PATH = "Data/failed_retrievals_summary.csv"
LOW_SIMILARITY_PATH = "Data/low_similarity_generations.json"
ALL_RESULTS_PATH = "Data/eval_results.json"
# Written INCREMENTALLY: one entry is appended right after every generation, so you can
# open these mid-run. Both files are overwritten at the start of each run.
GENERATION_LOG_JSONL = "Data/generations_log.jsonl"   # machine-readable, one JSON object per line
GENERATION_LOG_TXT = "Data/generations_log.txt"       # human-readable, easy to skim

GENERATION_SYSTEM_PROMPT = (
    "You are a question-answering assistant. Answer the question using ONLY the "
    "provided context. Be concise and answer in one or two sentences. If the context "
    "does not contain the answer, say that you cannot answer from the context."
)
# ==========================================


# ------------------------------------------------------------------
# Text utilities
# ------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """
    Normalizes text to lowercase, replacing non-alphanumeric characters
    and newlines with a single space for reliable substring matching.
    """
    if not text:
        return ""
    return re.sub(r'\W+', ' ', text).lower().strip()


def calculate_token_overlap(ground_truth: str, retrieved_chunk: str) -> float:
    """
    Calculates what fraction of the ground truth tokens appear in the retrieved chunk.
    """
    gt_tokens = set(normalize_text(ground_truth).split())
    chunk_tokens = set(normalize_text(retrieved_chunk).split())

    if not gt_tokens:
        return 0.0

    intersection = gt_tokens.intersection(chunk_tokens)
    return len(intersection) / len(gt_tokens)


def check_hit(
    gold_context: str,
    retrieved_chunks: List[str],
    threshold: float = OVERLAP_THRESHOLD,
    symmetric: bool = SYMMETRIC_MATCH,
) -> bool:
    """
    Returns True if the gold context chunk is matched by any retrieved chunk:
      - gold context is an exact (normalized) substring of the retrieved chunk, OR
      - >= threshold of the gold context tokens appear in the retrieved chunk, OR
      - (if symmetric) >= threshold of the retrieved chunk's tokens appear in the
        gold context, i.e. the retrieved chunk is a sub-span of the gold chunk.
    """
    norm_gold = normalize_text(gold_context)
    for chunk in retrieved_chunks:
        if norm_gold and norm_gold in normalize_text(chunk):
            return True
        if calculate_token_overlap(gold_context, chunk) >= threshold:
            return True
        if symmetric and calculate_token_overlap(chunk, gold_context) >= threshold:
            return True
    return False


def check_hit_any(
    gold_contexts: List[str],
    retrieved_chunks: List[str],
) -> bool:
    """A hit on ANY of the gold context chunks counts as a hit."""
    return any(check_hit(ctx, retrieved_chunks) for ctx in gold_contexts)


# ------------------------------------------------------------------
# Dataset loading
# ------------------------------------------------------------------
def load_json_dataset(path: str) -> List[Dict[str, Any]]:
    """
    Loads the evaluation dataset from a JSON file: a list of objects with
    id, question, ground_truth, context (list of chunk strings), source, chunk_id.
    Also accepts a {"data": [...]} / {"questions": [...]} wrapper.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        for key in ("data", "questions", "items"):
            if key in raw and isinstance(raw[key], list):
                raw = raw[key]
                break
        else:
            raise ValueError("JSON root is an object; expected a list of question objects.")

    dataset = []
    for i, obj in enumerate(raw, start=1):
        context = obj.get("context", [])
        if isinstance(context, str):
            context = [context]
        context = [c for c in context if c and c.strip()]

        if not obj.get("question") or not context:
            tqdm.write(f"[WARNING] Skipping entry #{i} (id={obj.get('id')}): missing question or context.")
            continue

        dataset.append({
            "id": obj.get("id", f"q{i:04d}"),
            "question": obj["question"],
            "ground_truth": obj.get("ground_truth", ""),
            "context": context,
            "source": obj.get("source"),
            "chunk_id": obj.get("chunk_id"),
        })
    return dataset


# ------------------------------------------------------------------
# Retrieval (unchanged logic)
# ------------------------------------------------------------------
def retrieve_all_stages(
    query: str,
    vectorstore,
    splade_retriever,
    top_k: int,
    final_top_k: int,
    critique_model: str,
    critique_prompt: str = None,
    filter_variant_key: str = "filter_v1",
    use_hyde: bool = USE_HYDE,
    use_query_expansion: bool = USE_QUERY_EXPANSION,
    query_processor_model: str = QUERY_PROCESSOR_MODEL,
    dedup_threshold: float = DEDUP_OVERLAP_THRESHOLD,
    use_relevance_filtering: bool = USE_RELEVANCE_FILTERING,
    embedding_model: str = EMBEDDING_MODEL,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Executes Dense, Sparse, Hybrid (RRF), and Critique filtering steps.
    Returns 4 lists of chunk texts:
      (dense_chunks, sparse_chunks, initial_hybrid_chunks, final_filtered_chunks)

    NOTE ON DEDUPLICATION: dense_chunks/sparse_chunks (used for the
    standalone "Dense Only"/"Sparse Only" hit-rate metrics) are the RAW,
    undeduplicated candidates -- those metrics measure each retriever's
    independent recall. Deduplication (via deduplicate_dense_sparse) is applied
    only to the copies that feed into RRF fusion, mirroring pipeline.py's
    generate_single_response().
    """

    # 0. Query Processing (HyDE for dense, expansion for sparse)
    dense_query = query
    if use_hyde:
        dense_query = generate_hyde_document(query, llm_model=query_processor_model)

    sparse_query = query
    if use_query_expansion:
        sparse_query = expand_query(query, llm_model=query_processor_model)

    # 1. Individual Retrievals (raw -- feeds the standalone Dense/Sparse metrics)
    dense_docs = vectorstore.as_retriever(search_kwargs={"k": top_k}).invoke(dense_query)
    splade_docs = splade_retriever.invoke(sparse_query, top_k=top_k)

    dense_chunks = [doc.page_content for doc in dense_docs]
    sparse_chunks = [doc.page_content for doc in splade_docs]

    # 1.5 Cross-retriever deduplication before fusion
    deduped_dense_docs, deduped_sparse_docs = deduplicate_dense_sparse(
        dense_docs, splade_docs, threshold=dedup_threshold
    )

    # 1.6 Optional relevance filtering (mirrors pipeline.generate_single_response Stage 1.6).
    # Like dedup, this only affects the copies feeding RRF, not the raw Dense/Sparse metrics.
    if use_relevance_filtering:
        embedder = resolve_embedder(vectorstore, embedding_model)
        deduped_dense_docs, deduped_sparse_docs = filter_by_relevance(
            query=query,
            dense_docs=deduped_dense_docs,
            splade_docs=deduped_sparse_docs,
            embeddings=embedder,
        )

    # 2. Hybrid Reciprocal Rank Fusion (RRF) -- unweighted.
    initial_docs = reciprocal_rank_fusion(
        [deduped_dense_docs, deduped_sparse_docs],
        weights=None,
        top_k=top_k,
    )
    initial_chunks = [doc.page_content for doc in initial_docs]

    # 3. Critique Filtering
    if critique_prompt and critique_prompt.strip() and len(initial_docs) > final_top_k:
        critique_llm = ChatOllama(model=critique_model, temperature=0.0)
        candidates_formatted = "\n\n".join(
            [f"--- CHUNK ID: {idx} ---\n{doc.page_content}" for idx, doc in enumerate(initial_docs)]
        )

        filter_system_template = FILTER_VARIANTS.get(filter_variant_key, FILTER_VARIANTS["filter_v1"])
        filter_prompt = ChatPromptTemplate.from_messages([
            ("system", filter_system_template),
            ("user", "USER QUERY: {user_query}\n\nCANDIDATE CHUNKS:\n{candidates}\n\nRemember: respond with ONLY the JSON array, nothing else."),
        ])

        try:
            filter_chain = filter_prompt | critique_llm
            critique_response = filter_chain.invoke({
                "critique_criteria": critique_prompt,
                "final_top_k": final_top_k,
                "user_query": query,
                "candidates": candidates_formatted,
            })

            raw_content = re.sub(r"<think>.*?</think>", "", critique_response.content, flags=re.DOTALL)
            matches = re.findall(
                r"\[\s*[\"']?\d+[\"']?(?:\s*,\s*[\"']?\d+[\"']?)*\s*\]",
                raw_content,
                flags=re.DOTALL,
            )

            if matches:
                cleaned = re.sub(r'["\']', '', matches[-1])
                selected_ids = json.loads(cleaned)
                valid_ids = [
                    idx for idx in selected_ids
                    if isinstance(idx, int) and 0 <= idx < len(initial_docs)
                ][:final_top_k]
                filtered_docs = [initial_docs[idx] for idx in valid_ids]
            else:
                filtered_docs = initial_docs[:final_top_k]

        except Exception as e:
            tqdm.write(f"[WARNING] Critique parsing failed ({e}). Falling back to top {final_top_k}.")
            filtered_docs = initial_docs[:final_top_k]
    else:
        filtered_docs = initial_docs[:final_top_k]

    final_chunks = [doc.page_content for doc in filtered_docs]

    return dense_chunks, sparse_chunks, initial_chunks, final_chunks


# ------------------------------------------------------------------
# Generation + answer similarity
# ------------------------------------------------------------------
def generate_answer(question: str, chunks: List[str], llm: ChatOllama) -> str:
    """
    Generates an answer from the retrieved chunks using the SAME prompt layout as
    pipeline.generate_single_response() (Stage 3), so generation quality measured here
    reflects what production produces.
    """
    context_str = "\n\n--- CHUNK BREAK ---\n\n".join(chunks)
    rag_system_template = RAG_VARIANTS.get(RAG_VARIANT_KEY, RAG_VARIANTS["rag_v1"])
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"{rag_system_template}\n\n=== CONTEXT START ===\n{{retrieved_data}}\n=== CONTEXT END ===",
        ),
        ("user", "Question: {user_query}"),
    ])
    response = (prompt | llm).invoke({"retrieved_data": context_str, "user_query": question})
    # deepseek-r1 emits <think>...</think> reasoning; strip it before scoring
    answer = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL)
    return answer.strip()


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token-level F1 between generated and ground-truth answers."""
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def bertscore_similarity(generated: str, reference: str, scorer: BERTScorer) -> Tuple[float, float, float]:
    """
    BERTScore between the generated answer (candidate) and the ground-truth answer
    (reference). Returns (precision, recall, f1). F1 is the headline number.
    """
    if not generated.strip() or not reference.strip():
        return 0.0, 0.0, 0.0
    p, r, f1 = scorer.score([generated], [reference])
    return p.item(), r.item(), f1.item()


# ------------------------------------------------------------------
# Incremental generation logging
# ------------------------------------------------------------------
def reset_generation_logs() -> None:
    """Creates output folders and truncates the incremental generation logs for a new run."""
    for path in (GENERATION_LOG_JSONL, GENERATION_LOG_TXT, ALL_RESULTS_PATH,
                 FAILED_RETRIEVALS_PATH, FAILED_RETRIEVALS_SUMMARY_PATH, LOW_SIMILARITY_PATH):
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
    for path in (GENERATION_LOG_JSONL, GENERATION_LOG_TXT):
        open(path, "w", encoding="utf-8").close()


def append_generation_log(entry: Dict[str, Any]) -> None:
    """
    Appends one generation to both logs immediately (file is opened/closed per call, so
    everything written so far survives a crash or Ctrl+C).
    """
    with open(GENERATION_LOG_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    lines = [
        "=" * 78,
        f"ID: {entry.get('id')}   |   PASS: {entry.get('generation_pass')}   |   REFUSED: {entry.get('refused')}",
        f"BERTScore  P={entry.get('bertscore_precision')}  R={entry.get('bertscore_recall')}  "
        f"F1={entry.get('bertscore_f1')}   |   Token F1={entry.get('token_f1')}   |   "
        f"Gen time={entry.get('generation_time')}s",
        f"QUESTION     : {entry.get('question')}",
        f"GROUND TRUTH : {entry.get('ground_truth')}",
        f"GENERATED    : {entry.get('generated_answer')}",
    ]
    if entry.get("error"):
        lines.append(f"ERROR        : {entry['error']}")
    lines.append("--- FINAL CHUNKS GIVEN TO THE MODEL ---")
    for i, chunk in enumerate(entry.get("final_chunks") or [], start=1):
        lines.append(f"[Chunk {i}] {chunk}")
    lines.append("")
    with open(GENERATION_LOG_TXT, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ------------------------------------------------------------------
# Main evaluation loop
# ------------------------------------------------------------------
def fast_evaluate_json():
    print(f"[INFO] Loading JSON dataset from: {JSON_PATH}")
    dataset = load_json_dataset(JSON_PATH)

    print("[INFO] Building / Loading Vectorstore and SPLADE retriever...")
    retrievers_data = load_txt_and_build_vectorstore(
        txt_path=TXT_PATH,
        embedding_model=EMBEDDING_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    vectorstore = retrievers_data["vectorstore"]
    splade_retriever = retrievers_data["splade_retriever"]

    generation_llm = None
    bert_scorer = None
    if EVALUATE_GENERATION:
        generation_llm = ChatOllama(model=GENERATION_MODEL, temperature=GENERATION_TEMPERATURE)
        print(f"[INFO] Loading BERTScore model '{BERTSCORE_MODEL}' (downloads on first use)...")
        bert_scorer = BERTScorer(
            model_type=BERTSCORE_MODEL,
            lang=BERTSCORE_LANG,
            rescale_with_baseline=BERTSCORE_RESCALE_WITH_BASELINE,
            batch_size=BERTSCORE_BATCH_SIZE,
        )

    total_questions = len(dataset)
    dense_hits = 0
    sparse_hits = 0
    hybrid_hits = 0
    final_hits = 0

    # Generation metrics (only for questions whose FINAL retrieval succeeded)
    gen_count = 0
    gen_errors = 0
    gen_refusals = 0             # model replied "context does not contain enough information"
    gen_pass = 0                 # BERTScore F1 >= BERTSCORE_THRESHOLD
    sum_bs_p = 0.0
    sum_bs_r = 0.0
    sum_bs_f1 = 0.0
    sum_token_f1 = 0.0
    total_gen_time = 0.0

    failed_cases = []            # retrieval misses / pipeline errors
    low_sim_cases = []           # successful retrievals whose answer was dissimilar to ground truth
    all_results = []             # per-question record

    print(f"\n[INFO] Starting Evaluation for {total_questions} questions...")
    print(f"[INFO] Settings: Chunk Size={CHUNK_SIZE}, Overlap={CHUNK_OVERLAP}, Top-K={TOP_K}, Final Top-K={FINAL_TOP_K}")
    print(f"[INFO] Retrieval hit threshold: {int(OVERLAP_THRESHOLD * 100)}% token overlap "
          f"({'symmetric' if SYMMETRIC_MATCH else 'gold-context coverage only'})")
    print(f"[INFO] HyDE for dense retrieval: {'ENABLED (' + QUERY_PROCESSOR_MODEL + ')' if USE_HYDE else 'DISABLED'}")
    print(f"[INFO] Query expansion for sparse retrieval: {'ENABLED (' + QUERY_PROCESSOR_MODEL + ')' if USE_QUERY_EXPANSION else 'DISABLED'}")
    print(f"[INFO] Relevance filtering before RRF: {'ENABLED' if USE_RELEVANCE_FILTERING else 'DISABLED'}")
    print(f"[INFO] Hybrid fusion: Unweighted RRF")
    print(f"[INFO] Cross-retriever dedup threshold: {int(DEDUP_OVERLAP_THRESHOLD * 100)}% token overlap (applied before Hybrid/Final stages only)")
    if EVALUATE_GENERATION:
        print(f"[INFO] Generation: ENABLED (model={GENERATION_MODEL}, BERTScore F1 threshold={BERTSCORE_THRESHOLD})")
    else:
        print("[INFO] Generation: DISABLED")
    print("-" * 65)

    if EVALUATE_GENERATION:
        reset_generation_logs()
        print(f"[INFO] Logging every generation as it completes to: {GENERATION_LOG_TXT} (and {GENERATION_LOG_JSONL})")

    start_time = time.time()
    pbar = tqdm(dataset, desc="Evaluating", unit="query")

    for idx, row in enumerate(pbar, start=1):
        question_id = row["id"]
        question = row["question"]
        gold_contexts = row["context"]
        ground_truth_answer = row["ground_truth"]

        record: Dict[str, Any] = {
            "id": question_id,
            "question": question,
            "source": row["source"],
            "chunk_id": row["chunk_id"],
            "ground_truth": ground_truth_answer,
        }

        try:
            dense_chunks, sparse_chunks, initial_chunks, final_chunks = retrieve_all_stages(
                query=question,
                vectorstore=vectorstore,
                splade_retriever=splade_retriever,
                top_k=TOP_K,
                final_top_k=FINAL_TOP_K,
                critique_model=CRITIQUE_MODEL,
                critique_prompt=CRITIQUE_PROMPT,
                filter_variant_key=FILTER_VARIANT_KEY,
                use_hyde=USE_HYDE,
                use_query_expansion=USE_QUERY_EXPANSION,
                query_processor_model=QUERY_PROCESSOR_MODEL,
                dedup_threshold=DEDUP_OVERLAP_THRESHOLD,
                use_relevance_filtering=USE_RELEVANCE_FILTERING,
                embedding_model=EMBEDDING_MODEL,
            )

            # ---- Retrieval checks (against the JSON "context" chunk(s)) ----
            dense_hit = check_hit_any(gold_contexts, dense_chunks)
            sparse_hit = check_hit_any(gold_contexts, sparse_chunks)
            hybrid_hit = check_hit_any(gold_contexts, initial_chunks)
            final_hit = check_hit_any(gold_contexts, final_chunks)

            dense_hits += dense_hit
            sparse_hits += sparse_hit
            hybrid_hits += hybrid_hit
            final_hits += final_hit

            record.update({
                "dense_hit": dense_hit,
                "sparse_hit": sparse_hit,
                "hybrid_hit": hybrid_hit,
                "final_hit": final_hit,
                "generated_answer": None,
                "bertscore_precision": None,
                "bertscore_recall": None,
                "bertscore_f1": None,
                "token_f1": None,
                "generation_time": None,
                "generation_pass": None,
                "refused": None,
            })

            if not final_hit:
                failed_cases.append({
                    "id": question_id,
                    "question": question,
                    "ground_truth": ground_truth_answer,
                    "gold_context": gold_contexts,
                    "source": row["source"],
                    "chunk_id": row["chunk_id"],
                    "dense_hit": dense_hit,
                    "sparse_hit": sparse_hit,
                    "hybrid_hit": hybrid_hit,
                    "final_hit": final_hit,
                    # hybrid_hit True but final_hit False => critique dropped a correct chunk
                    "dropped_by_critique": hybrid_hit and not final_hit,
                    "dense_chunks": dense_chunks,
                    "sparse_chunks": sparse_chunks,
                    "hybrid_chunks": initial_chunks,
                    "final_chunks": final_chunks,
                })

            # ---- Generation (only for successful final retrievals) ----
            elif EVALUATE_GENERATION:
                gen_start = time.time()
                answer = None
                try:
                    answer = generate_answer(question, final_chunks, generation_llm)
                    bs_p, bs_r, bs_f1 = bertscore_similarity(answer, ground_truth_answer, bert_scorer)
                    f1 = token_f1(answer, ground_truth_answer)
                    gen_time = time.time() - gen_start

                    gen_count += 1
                    sum_bs_p += bs_p
                    sum_bs_r += bs_r
                    sum_bs_f1 += bs_f1
                    sum_token_f1 += f1
                    total_gen_time += gen_time
                    refused = REFUSAL_MARKER in answer.lower()
                    gen_refusals += refused
                    passed = bs_f1 >= BERTSCORE_THRESHOLD
                    gen_pass += passed

                    record.update({
                        "generated_answer": answer,
                        "bertscore_precision": round(bs_p, 4),
                        "bertscore_recall": round(bs_r, 4),
                        "bertscore_f1": round(bs_f1, 4),
                        "token_f1": round(f1, 4),
                        "generation_time": round(gen_time, 2),
                        "generation_pass": passed,
                        "refused": refused,
                    })
                    append_generation_log({**record, "final_chunks": final_chunks})

                    if not passed:
                        low_sim_cases.append({
                            "id": question_id,
                            "question": question,
                            "ground_truth": ground_truth_answer,
                            "generated_answer": answer,
                            "refused": refused,
                            "bertscore_precision": round(bs_p, 4),
                            "bertscore_recall": round(bs_r, 4),
                            "bertscore_f1": round(bs_f1, 4),
                            "token_f1": round(f1, 4),
                            "final_chunks": final_chunks,
                        })
                except Exception as gen_e:
                    gen_errors += 1
                    record["generation_error"] = str(gen_e)
                    append_generation_log({
                        **record,
                        "generated_answer": answer,
                        "error": str(gen_e),
                        "final_chunks": final_chunks,
                    })
                    tqdm.write(f"[ERROR] Generation failed for ID: {question_id} | {gen_e}")

            postfix = {
                "Dense": f"{(dense_hits / idx) * 100:.1f}%",
                "Sparse": f"{(sparse_hits / idx) * 100:.1f}%",
                "Hybrid": f"{(hybrid_hits / idx) * 100:.1f}%",
                "Final": f"{(final_hits / idx) * 100:.1f}%",
            }
            if EVALUATE_GENERATION and gen_count > 0:
                postfix["BERT-F1"] = f"{sum_bs_f1 / gen_count:.3f}"
                postfix["GenOK"] = f"{(gen_pass / gen_count) * 100:.1f}%"
            pbar.set_postfix(postfix)

        except Exception as e:
            tqdm.write(f"[ERROR] ID: {question_id} | {e}")
            record["error"] = str(e)
            failed_cases.append({
                "id": question_id,
                "question": question,
                "ground_truth": ground_truth_answer,
                "gold_context": gold_contexts,
                "error": str(e),
            })

        all_results.append(record)

    total_time = time.time() - start_time

    # ------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------
    def pct(n: int, d: int) -> float:
        return (n / d) * 100 if d > 0 else 0.0

    dense_rate = pct(dense_hits, total_questions)
    sparse_rate = pct(sparse_hits, total_questions)
    hybrid_rate = pct(hybrid_hits, total_questions)
    final_rate = pct(final_hits, total_questions)

    print("\n" + "=" * 65)
    print("              RETRIEVAL EVALUATION RESULTS")
    print("=" * 65)
    print(f"Total Questions Evaluated       : {total_questions}")
    print(f"Dense Only Hit Rate (Top-{TOP_K})    : {dense_rate:.2f}% ({dense_hits}/{total_questions})")
    print(f"Sparse Only Hit Rate (Top-{TOP_K})   : {sparse_rate:.2f}% ({sparse_hits}/{total_questions})")
    print(f"Hybrid RRF Hit Rate (Top-{TOP_K})    : {hybrid_rate:.2f}% ({hybrid_hits}/{total_questions})")
    print(f"Final Filtered Hit Rate (Top-{FINAL_TOP_K})  : {final_rate:.2f}% ({final_hits}/{total_questions})")

    if EVALUATE_GENERATION:
        print("\n" + "=" * 65)
        print("              GENERATION EVALUATION RESULTS")
        print("=" * 65)
        print(f"Answers Generated (final hits)  : {gen_count}/{final_hits}"
              + (f"  [{gen_errors} generation error(s)]" if gen_errors else ""))
        if gen_count > 0:
            print(f"Model Refusals (retrieval hit, but model said context insufficient): {gen_refusals}/{gen_count}")
            print(f"Mean BERTScore Precision        : {sum_bs_p / gen_count:.4f}")
            print(f"Mean BERTScore Recall           : {sum_bs_r / gen_count:.4f}")
            print(f"Mean BERTScore F1               : {sum_bs_f1 / gen_count:.4f}")
            print(f"Mean Token F1                   : {sum_token_f1 / gen_count:.4f}")
            print(f"BERTScore F1 >= {BERTSCORE_THRESHOLD:.2f} (of generated): "
                  f"{pct(gen_pass, gen_count):.2f}% ({gen_pass}/{gen_count})")
            print(f"End-to-End Success Rate         : {pct(gen_pass, total_questions):.2f}% "
                  f"({gen_pass}/{total_questions})  [retrieval hit AND BERTScore F1 >= {BERTSCORE_THRESHOLD:.2f}]")
            print(f"Avg Generation Time             : {total_gen_time / gen_count:.2f} seconds")
        else:
            print("No answers were generated (no successful retrievals).")

    print("-" * 65)
    print(f"Total Execution Time            : {total_time:.2f} seconds")
    if total_questions > 0:
        print(f"Average Time per Query          : {total_time / total_questions:.2f} seconds")
    print("=" * 65 + "\n")

    # ------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------
    with open(ALL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Per-question results written to: {ALL_RESULTS_PATH}")

    if failed_cases:
        with open(FAILED_RETRIEVALS_PATH, "w", encoding="utf-8") as f:
            json.dump(failed_cases, f, indent=2, ensure_ascii=False)
        print(f"[INFO] {len(failed_cases)} failed retrieval case(s) written to: {FAILED_RETRIEVALS_PATH}")

        with open(FAILED_RETRIEVALS_SUMMARY_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["id", "question", "ground_truth", "dense_hit",
                            "sparse_hit", "hybrid_hit", "final_hit", "dropped_by_critique"],
                extrasaction="ignore",
            )
            writer.writeheader()
            for case in failed_cases:
                writer.writerow(case)
        print(f"[INFO] Summary CSV written to: {FAILED_RETRIEVALS_SUMMARY_PATH}")
    else:
        print("[INFO] No failed retrievals — every gold context chunk was retrieved successfully.")

    if EVALUATE_GENERATION:
        if low_sim_cases:
            with open(LOW_SIMILARITY_PATH, "w", encoding="utf-8") as f:
                json.dump(low_sim_cases, f, indent=2, ensure_ascii=False)
            print(f"[INFO] {len(low_sim_cases)} low-similarity generation(s) written to: {LOW_SIMILARITY_PATH}")
        else:
            print("[INFO] No low-similarity generations.")


if __name__ == "__main__":
    fast_evaluate_json()