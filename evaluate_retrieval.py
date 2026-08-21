import json
import os
from typing import Dict, List
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


def load_txt_corpus_as_documents(txt_path: str) -> List[Document]:
    """Loads text corpus and splits it into Document objects with matching IDs."""
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"File not found: {txt_path}")

    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.split("========================================")
    documents = []

    for idx, block in enumerate(blocks):
        block_str = block.strip()
        if not block_str or len(block_str) < 50:
            continue

        vol_match = re.search(r"Volume:\s*(\d+)", block_str)
        sec_match = re.search(r"Section:\s*(\d+)", block_str)

        vol = vol_match.group(1) if vol_match else "0"
        sec = sec_match.group(1) if sec_match else str(idx + 1)
        doc_id = f"doc_v{vol}_s{sec}"

        doc = Document(
            page_content=block_str, metadata={"doc_id": doc_id, "chunk_index": idx}
        )
        documents.append(doc)

    return documents


def run_evaluation(txt_path: str, eval_json_path: str, embedding_model: str):
    # 1. Load documents from TXT
    documents = load_txt_corpus_as_documents(txt_path)

    # 2. Index in ChromaDB
    embedding_fn = OllamaEmbeddings(model=embedding_model)
    vstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_fn,
        collection_name="txt_eval",
    )

    # 3. Load synthetic test cases
    with open(eval_json_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    # 4. Evaluate Recall@K and MRR
    retriever = vstore.as_retriever(search_kwargs={"k": 25})
    mrr_scores = []
    recall_10_scores = []

    for case in test_cases:
        query = case["query"]
        target_id = case["expected_doc_id"]

        results = retriever.invoke(query)
        retrieved_ids = [doc.metadata["doc_id"] for doc in results]

        # Calculate Reciprocal Rank
        if target_id in retrieved_ids:
            rank = retrieved_ids.index(target_id) + 1
            mrr_scores.append(1.0 / rank)
        else:
            mrr_scores.append(0.0)

        # Calculate Recall@10
        recall_10_scores.append(
            1.0 if target_id in retrieved_ids[:10] else 0.0
        )

    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
    avg_recall = (
        sum(recall_10_scores) / len(recall_10_scores) if recall_10_scores else 0.0
    )

    print("\n" + "=" * 50)
    print("RETRIEVAL PERFORMANCE EVALUATION")
    print("=" * 50)
    print(f"Evaluated Test Cases : {len(test_cases)}")
    print(f"Mean Reciprocal Rank : {avg_mrr:.4f}")
    print(f"Recall@10            : {avg_recall * 100:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    run_evaluation(
        txt_path="Data/combined_corpus.txt",
        eval_json_path="Data/synthetic_eval_dataset.json",
        embedding_model="nomic-embed-text",
    )