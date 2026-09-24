# retrieval_eval.py

import csv
import json
import re
import time
from typing import List, Tuple
from tqdm import tqdm

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
)
from prompts import FILTER_VARIANTS

# ==========================================
# CONFIGURATION - Imported from config.py
# ==========================================
CSV_PATH = GlobalConfig.RETRIEVAL_TEST_CSV_PATH  # Path to evaluation CSV file
TXT_PATH = GlobalConfig.TXT_FILE_PATH

# Model configurations from GlobalConfig
EMBEDDING_MODEL = GlobalConfig.EMBEDDING_MODELS[1]  # "qwen3-embedding:0.6b"
CRITIQUE_MODEL = GlobalConfig.CRITIQUE_MODELS[2]    # "qwen3.5:2b"

# Model used to generate the HyDE hypothetical-answer passage for dense
# retrieval. Falls back to pipeline.py's default if not defined in config.
QUERY_PROCESSOR_MODEL = getattr(GlobalConfig, "QUERY_PROCESSOR_MODEL", "qwen3.5:2b")

# Chunking & Retrieval hyperparameters
CHUNK_SIZE = 600     # 800
CHUNK_OVERLAP = 200 # 150
TOP_K = 15            # 20
FINAL_TOP_K = 7                                     # Set to 5

CRITIQUE_PROMPT = "Select the most relevant chunks."
OVERLAP_THRESHOLD = 0.50                            # 50% minimum token overlap requirement (ground-truth hit check)

# Minimum token overlap for a dense chunk and a sparse chunk to be treated
# as near-duplicates before RRF fusion. See pipeline.deduplicate_dense_sparse().
# Falls back to 0.50 if not defined in config.py.
DEDUP_OVERLAP_THRESHOLD = getattr(GlobalConfig, "DEDUP_OVERLAP_THRESHOLD", 0.50)

USE_HYDE = False                                     # Generate a hypothetical answer and embed that instead of the raw query (dense)
USE_QUERY_EXPANSION = True                          # Append LLM-generated related terms/synonyms to the query (sparse)
# ==========================================


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
    overlap_ratio = len(intersection) / len(gt_tokens)
    return overlap_ratio


def check_hit(ground_truth: str, retrieved_chunks: List[str], threshold: float = OVERLAP_THRESHOLD) -> bool:
    """
    Returns True if ground truth is an exact substring OR has at least threshold token overlap
    with any of the retrieved chunks.
    """
    norm_gt = normalize_text(ground_truth)
    return any(
        (norm_gt in normalize_text(chunk)) or 
        (calculate_token_overlap(ground_truth, chunk) >= threshold)
        for chunk in retrieved_chunks
    )


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
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Executes Dense, Sparse, Hybrid (RRF), and Critique filtering steps.
    Returns 4 lists of chunk texts:
      (dense_chunks, sparse_chunks, initial_hybrid_chunks, final_filtered_chunks)

    NOTE ON DEDUPLICATION: dense_chunks/sparse_chunks (used for the
    standalone "Dense Only"/"Sparse Only" hit-rate metrics) are the RAW,
    undeduplicated candidates -- those metrics are meant to measure each
    retriever's independent recall, and deduping them would make a hit
    disappear from one retriever's score just because the other retriever
    also found something overlapping. Deduplication (via
    deduplicate_dense_sparse, threshold=dedup_threshold) is applied only to
    the copies that feed into RRF fusion, mirroring pipeline.py's
    generate_single_response() so the Hybrid/Final hit rates reflect what
    production actually returns.
    """

    # 0. Query Processing (HyDE for dense, expansion for sparse)
    # Dense: embed a hypothetical answer passage instead of the raw query,
    # since it resembles the target chunk text far more closely than a short
    # question does. Falls back to the raw query if generation fails.
    dense_query = query
    if use_hyde:
        dense_query = generate_hyde_document(query, llm_model=query_processor_model)

    # Sparse: append LLM-generated related terms/synonyms to widen the
    # lexical/token-overlap surface SPLADE can match against. Falls back to
    # the raw query if generation fails.
    sparse_query = query
    if use_query_expansion:
        sparse_query = expand_query(query, llm_model=query_processor_model)

    # 1. Individual Retrievals (raw -- feeds the standalone Dense/Sparse metrics)
    dense_docs = vectorstore.as_retriever(search_kwargs={"k": top_k}).invoke(dense_query)
    splade_docs = splade_retriever.invoke(sparse_query, top_k=top_k)
    
    dense_chunks = [doc.page_content for doc in dense_docs]
    sparse_chunks = [doc.page_content for doc in splade_docs]

    # 1.5 Cross-retriever deduplication before fusion, mirroring
    # pipeline.py's generate_single_response(). Runs on separate copies so
    # dense_chunks/sparse_chunks above stay raw for the standalone metrics.
    deduped_dense_docs, deduped_sparse_docs = deduplicate_dense_sparse(
        dense_docs, splade_docs, threshold=dedup_threshold
    )

    # 2. Hybrid Reciprocal Rank Fusion (RRF) — unweighted.
    # Weighting the RRF terms multiplicatively (e.g. [0.7, 0.3]) can make one
    # list's weight advantage dominate the rank-based score entirely,
    # silently locking sparse-only hits out of the fused top_k regardless of
    # how well they ranked. Leaving weights=None (equal weighting) lets rank
    # position in each list decide fairly, which is the standard formulation.
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
        
        filter_system_template = FILTER_VARIANTS.get(filter_variant_key, "{critique_criteria}")
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


def fast_evaluate_csv():
    print(f"[INFO] Loading CSV dataset from: {CSV_PATH}")
    dataset = []
    with open(CSV_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset.append(row)
            
    print("[INFO] Building / Loading Vectorstore and SPLADE retriever...")
    retrievers_data = load_txt_and_build_vectorstore(
        txt_path=TXT_PATH,
        embedding_model=EMBEDDING_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    vectorstore = retrievers_data["vectorstore"]
    splade_retriever = retrievers_data["splade_retriever"]
    
    total_questions = len(dataset)
    dense_hits = 0
    sparse_hits = 0
    hybrid_hits = 0
    final_hits = 0

    failed_cases = []  # cases where the final RAG output missed the ground truth
    
    print(f"\n[INFO] Starting Fast Evaluation for {total_questions} questions...")
    print(f"[INFO] Settings: Chunk Size={CHUNK_SIZE}, Overlap={CHUNK_OVERLAP}, Top-K={TOP_K}, Final Top-K={FINAL_TOP_K}")
    print(f"[INFO] Threshold: {int(OVERLAP_THRESHOLD * 100)}% Token Overlap")
    print(f"[INFO] HyDE for dense retrieval: {'ENABLED (' + QUERY_PROCESSOR_MODEL + ')' if USE_HYDE else 'DISABLED'}")
    print(f"[INFO] Query expansion for sparse retrieval: {'ENABLED (' + QUERY_PROCESSOR_MODEL + ')' if USE_QUERY_EXPANSION else 'DISABLED'}")
    print(f"[INFO] Hybrid fusion: Unweighted RRF")
    print(f"[INFO] Cross-retriever dedup threshold: {int(DEDUP_OVERLAP_THRESHOLD * 100)}% token overlap (applied before Hybrid/Final stages only)")
    print("-" * 65)
    
    start_time = time.time()
    pbar = tqdm(dataset, desc="Evaluating", unit="query")
    
    for idx, row in enumerate(pbar, start=1):
        question_id = row.get("id", str(idx))
        question = row["question"]
        ground_truth = row["ground_truth_chunk"]
        
        try:
            dense_chunks, sparse_chunks, initial_chunks, final_chunks = retrieve_all_stages(
                query=question,
                vectorstore=vectorstore,
                splade_retriever=splade_retriever,
                top_k=TOP_K,
                final_top_k=FINAL_TOP_K,
                critique_model=CRITIQUE_MODEL,
                critique_prompt=CRITIQUE_PROMPT,
                use_hyde=USE_HYDE,
                use_query_expansion=USE_QUERY_EXPANSION,
                query_processor_model=QUERY_PROCESSOR_MODEL,
                dedup_threshold=DEDUP_OVERLAP_THRESHOLD,
            )
            
            # 1. Check Dense Hit
            dense_hit = check_hit(ground_truth, dense_chunks)
            if dense_hit:
                dense_hits += 1
                
            # 2. Check Sparse Hit
            sparse_hit = check_hit(ground_truth, sparse_chunks)
            if sparse_hit:
                sparse_hits += 1
                
            # 3. Check Hybrid (RRF) Hit
            hybrid_hit = check_hit(ground_truth, initial_chunks)
            if hybrid_hit:
                hybrid_hits += 1
                
            # 4. Check Final (LLM Filtered) Hit
            final_hit = check_hit(ground_truth, final_chunks)
            if final_hit:
                final_hits += 1
            else:
                # Log the miss for later review, including which stage(s)
                # actually had the ground truth available so we can tell
                # retrieval misses apart from critique-filtering misses.
                failed_cases.append({
                    "id": question_id,
                    "question": question,
                    "ground_truth": ground_truth,
                    "dense_hit": dense_hit,
                    "sparse_hit": sparse_hit,
                    "hybrid_hit": hybrid_hit,
                    "final_hit": final_hit,
                    # if hybrid_hit is True but final_hit is False, the
                    # critique/filter step dropped a correct chunk
                    "dropped_by_critique": hybrid_hit and not final_hit,
                    "dense_chunks": dense_chunks,
                    "sparse_chunks": sparse_chunks,
                    "hybrid_chunks": initial_chunks,
                    "final_chunks": final_chunks,
                })
                
            pbar.set_postfix({
                "Dense": f"{(dense_hits/idx)*100:.1f}%",
                "Sparse": f"{(sparse_hits/idx)*100:.1f}%",
                "Hybrid": f"{(hybrid_hits/idx)*100:.1f}%",
                "Final": f"{(final_hits/idx)*100:.1f}%"
            })
            
        except Exception as e:
            tqdm.write(f"[ERROR] ID: {question_id} | {e}")
            failed_cases.append({
                "id": question_id,
                "question": question,
                "ground_truth": ground_truth,
                "error": str(e),
            })
            
    total_time = time.time() - start_time
    
    # Calculate final percentages
    dense_rate = (dense_hits / total_questions) * 100 if total_questions > 0 else 0.0
    sparse_rate = (sparse_hits / total_questions) * 100 if total_questions > 0 else 0.0
    hybrid_rate = (hybrid_hits / total_questions) * 100 if total_questions > 0 else 0.0
    final_rate = (final_hits / total_questions) * 100 if total_questions > 0 else 0.0
    
    print("\n" + "=" * 65)
    print("                 RETRIEVAL EVALUATION RESULTS")
    print("=" * 65)
    print(f"Total Questions Evaluated       : {total_questions}")
    print(f"Dense Only Hit Rate (Top-{TOP_K})    : {dense_rate:.2f}% ({dense_hits}/{total_questions})")
    print(f"Sparse Only Hit Rate (Top-{TOP_K})   : {sparse_rate:.2f}% ({sparse_hits}/{total_questions})")
    print(f"Hybrid RRF Hit Rate (Top-{TOP_K})    : {hybrid_rate:.2f}% ({hybrid_hits}/{total_questions})")
    print(f"Final Filtered Hit Rate (Top-{FINAL_TOP_K})  : {final_rate:.2f}% ({final_hits}/{total_questions})")
    print("-" * 65)
    print(f"Total Execution Time            : {total_time:.2f} seconds")
    print(f"Average Time per Query          : {total_time / total_questions:.2f} seconds")
    print("=" * 65 + "\n")

    # ------------------------------------------------------------
    # Write failed cases to a file for manual review
    # ------------------------------------------------------------
    if failed_cases:
        failures_path = "Data/failed_retrievals.json"
        with open(failures_path, "w", encoding="utf-8") as f:
            json.dump(failed_cases, f, indent=2, ensure_ascii=False)
        print(f"[INFO] {len(failed_cases)} failed case(s) written to: {failures_path}")

        # Also drop a lightweight CSV summary (no chunk text) for quick scanning
        summary_path = "Data/failed_retrievals_summary.csv"
        with open(summary_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["id", "question", "ground_truth", "dense_hit",
                            "sparse_hit", "hybrid_hit", "final_hit", "dropped_by_critique"],
                extrasaction="ignore",
            )
            writer.writeheader()
            for case in failed_cases:
                writer.writerow(case)
        print(f"[INFO] Summary CSV written to: {summary_path}")
    else:
        print("[INFO] No failed cases — every ground truth was retrieved successfully.")


if __name__ == "__main__":
    fast_evaluate_csv()