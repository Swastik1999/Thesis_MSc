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

# Chunking & Retrieval hyperparameters
CHUNK_SIZE = GlobalConfig.CHUNK_SIZE_RANGE[0]       # 800
CHUNK_OVERLAP = GlobalConfig.CHUNK_OVERLAP_RANGE[0] # 150
TOP_K = GlobalConfig.TOP_K_RANGE[0]                 # 20
FINAL_TOP_K = 5                                     # Set to 5

CRITIQUE_PROMPT = "Select the most relevant chunks."
OVERLAP_THRESHOLD = 0.50                            # 50% minimum token overlap requirement
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
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Executes Dense, Sparse, Hybrid (RRF), and Critique filtering steps.
    Returns 4 lists of chunk texts:
      (dense_chunks, sparse_chunks, initial_hybrid_chunks, final_filtered_chunks)
    """
    
    # 1. Individual Retrievals
    dense_docs = vectorstore.as_retriever(search_kwargs={"k": top_k}).invoke(query)
    splade_docs = splade_retriever.invoke(query, top_k=top_k)
    
    dense_chunks = [doc.page_content for doc in dense_docs]
    sparse_chunks = [doc.page_content for doc in splade_docs]

    # 2. Hybrid Reciprocal Rank Fusion (RRF)
    initial_docs = reciprocal_rank_fusion(
        [dense_docs, splade_docs],
        weights=[0.7, 0.3],
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
    
    print(f"\n[INFO] Starting Fast Evaluation for {total_questions} questions...")
    print(f"[INFO] Settings: Chunk Size={CHUNK_SIZE}, Overlap={CHUNK_OVERLAP}, Top-K={TOP_K}, Final Top-K={FINAL_TOP_K}")
    print(f"[INFO] Threshold: {int(OVERLAP_THRESHOLD * 100)}% Token Overlap")
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
            )
            
            # 1. Check Dense Hit
            if check_hit(ground_truth, dense_chunks):
                dense_hits += 1
                
            # 2. Check Sparse Hit
            if check_hit(ground_truth, sparse_chunks):
                sparse_hits += 1
                
            # 3. Check Hybrid (RRF) Hit
            if check_hit(ground_truth, initial_chunks):
                hybrid_hits += 1
                
            # 4. Check Final (LLM Filtered) Hit
            if check_hit(ground_truth, final_chunks):
                final_hits += 1
                
            pbar.set_postfix({
                "Dense": f"{(dense_hits/idx)*100:.1f}%",
                "Sparse": f"{(sparse_hits/idx)*100:.1f}%",
                "Hybrid": f"{(hybrid_hits/idx)*100:.1f}%",
                "Final": f"{(final_hits/idx)*100:.1f}%"
            })
            
        except Exception as e:
            tqdm.write(f"[ERROR] ID: {question_id} | {e}")
            
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


if __name__ == "__main__":
    fast_evaluate_csv()