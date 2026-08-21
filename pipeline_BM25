# pipeline.py

import hashlib
import json
import os
import re
import time
from typing import Dict, List

from config import GlobalConfig
from embeddings import LocalOllamaEmbeddings
from rank_bm25 import BM25Okapi
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import prompt variants from prompts.py
from prompts import FILTER_VARIANTS, RAG_VARIANTS

_vs_cache = {}


# =====================================================================
# 1. BM25 SPARSE RETRIEVER CLASS
# =====================================================================
class BM25Retriever:
    """Fast lexical sparse retriever using BM25Okapi (rank_bm25).

    Replaces the previous SPLADE-based learned sparse retriever. BM25 is a
    pure statistical term-frequency/inverse-document-frequency method with
    no model inference required, making indexing dramatically faster on
    large corpora. It loses SPLADE's semantic-aware term weighting, but
    since this pipeline already fuses sparse results with dense embeddings
    (which handle semantic similarity) via weighted RRF, the trade-off is
    generally acceptable in this hybrid setup.
    """

    def __init__(self, documents: List[Document]):
        print(f"[INFO] Building BM25 index for {len(documents)} document chunks...")
        self.documents = documents
        self.tokenized_corpus = [
            self._tokenize(doc.page_content) for doc in documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple lowercase word tokenizer. Swap for a stemmer/tokenizer
        of your choice (e.g. nltk, spaCy) if you want more sophisticated
        lexical matching."""
        return re.findall(r"\w+", text.lower())

    def invoke(self, query: str, top_k: int = 25) -> List[Document]:
        """Scores all documents against the query using BM25 and returns
        the top_k highest-scoring chunks."""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        return [self.documents[idx] for idx in ranked_indices[:top_k]]


# =====================================================================
# 2. RECIPROCAL RANK FUSION (RRF) HELPER
# =====================================================================
def reciprocal_rank_fusion(
    results_list: List[List[Document]],
    weights: List[float] = None,
    top_k: int = 25,
    c: int = 60,
) -> List[Document]:
    """Combines ranked candidate lists from Sparse (BM25) and Dense retrievers using weighted RRF."""
    if weights is None:
        weights = [1.0] * len(results_list)
    if len(weights) != len(results_list):
        raise ValueError("weights must have the same length as results_list")

    doc_scores: Dict[str, float] = {}
    doc_mapping: Dict[str, Document] = {}

    for weight, ranked_docs in zip(weights, results_list):
        for rank, doc in enumerate(ranked_docs):
            doc_key = doc.page_content
            if doc_key not in doc_scores:
                doc_scores[doc_key] = 0.0
                doc_mapping[doc_key] = doc

            doc_scores[doc_key] += weight * (1.0 / (c + rank + 1))

    sorted_docs = sorted(
        doc_scores.keys(), key=lambda k: doc_scores[k], reverse=True
    )
    return [doc_mapping[k] for k in sorted_docs[:top_k]]

# =====================================================================
# 2.5 QUERY PROCESSING: EXPANSION (for sparse) + HyDE (for dense)
# =====================================================================
def expand_query(query: str, llm_model: str, num_expansions: int = 5) -> str:
    """Generates related terms/synonyms via LLM and appends them to the
    original query, improving BM25's lexical/token-overlap recall."""
    llm = ChatOllama(model=llm_model, temperature=0.3)
    expansion_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You expand search queries for a lexical/keyword search engine. "
                f"Given a query, output ONLY a comma-separated list of {num_expansions} "
                "closely related terms, synonyms, or alternate phrasings that might "
                "appear in a relevant document. No explanations, no numbering.",
            ),
            ("user", "{query}"),
        ]
    )
    chain = expansion_prompt | llm
    try:
        response = chain.invoke({"query": query})
        expansion_terms = response.content.strip()
        return f"{query} {expansion_terms}"
    except Exception as e:
        print(f"[WARNING] Query expansion failed ({e}). Using original query.")
        return query


def generate_hyde_document(query: str, llm_model: str) -> str:
    """Generates a short hypothetical answer passage for the query (HyDE).
    This passage is embedded instead of the raw query for dense retrieval,
    since it more closely resembles the form of the target document chunks."""
    llm = ChatOllama(model=llm_model, temperature=0.3)
    hyde_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Write a short, plausible passage (3-5 sentences) that would answer "
                "the user's question, as if it were an excerpt from a reference "
                "document on the topic. Do not mention you are an AI, do not say "
                "'hypothetically' or similar — just write the passage directly.",
            ),
            ("user", "{query}"),
        ]
    )
    chain = hyde_prompt | llm
    try:
        response = chain.invoke({"query": query})
        return response.content.strip()
    except Exception as e:
        print(f"[WARNING] HyDE generation failed ({e}). Falling back to raw query.")
        return query
# =====================================================================
# 3. VECTORSTORE & HYBRID SEARCH LOADER
# =====================================================================
def load_txt_and_build_vectorstore(
    txt_path: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
):
    cache_key = (txt_path, embedding_model, chunk_size, chunk_overlap)
    if cache_key in _vs_cache:
        return _vs_cache[cache_key]

    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"File not found at: {txt_path}")

    try:
        embeddings = LocalOllamaEmbeddings(model=embedding_model)
    except TypeError:
        embeddings = LocalOllamaEmbeddings(model_name=embedding_model)

    params_str = (
        f"{os.path.abspath(txt_path)}_{embedding_model}_{chunk_size}_{chunk_overlap}"
    )
    folder_hash = hashlib.md5(params_str.encode("utf-8")).hexdigest()[:10]
    persist_dir = os.path.join(
        GlobalConfig.BASE_OUTPUT_DIR,
        f"chroma_db_cs{chunk_size}_co{chunk_overlap}_{folder_hash}",
    )

    loader = TextLoader(txt_path, encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)

    sqlite_file = os.path.join(persist_dir, "chroma.sqlite3")
    if os.path.exists(sqlite_file):
        print(f"[INFO] Loading existing dense vectorstore from: {persist_dir}")
        vectorstore = Chroma(
            persist_directory=persist_dir, embedding_function=embeddings
        )
    else:
        print(f"[INFO] Creating dense vectorstore at: {persist_dir}")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir,
        )

    bm25_retriever = BM25Retriever(documents=chunks)

    cache_data = {
        "vectorstore": vectorstore,
        "splade_retriever": bm25_retriever,  # key name kept for downstream compatibility
    }
    _vs_cache[cache_key] = cache_data
    return cache_data


# =====================================================================
# 4. GENERATION PIPELINE
# =====================================================================
def generate_single_response(
    query: str,
    txt_file_path: str,
    embedding_model: str,
    generation_model: str,
    chunk_size: int,
    chunk_overlap: int,
    system_context: str = None,
    top_k: int = 25,
    temperature: float = 0.0,
    critique_prompt: str = None,
    critique_model: str = "qwen3.5:2b",
    final_top_k: int = 6,
    filter_variant_key: str = "filter_v1",
    rag_variant_key: str = "rag_v1",
    use_query_expansion: bool = False,      # <-- new
    use_hyde: bool = False,                 # <-- new
    query_processor_model: str = "qwen2.5:1.5b",  # <-- new
):
    start_time = time.time()

    # Stage 0: Query Processing (optional)
    sparse_query = query
    dense_query = query

    if use_query_expansion:
        sparse_query = expand_query(query, llm_model=query_processor_model)

    if use_hyde:
        dense_query = generate_hyde_document(query, llm_model=query_processor_model)

    # Stage 1: Hybrid Retrieval (BM25 + Dense)
    retrievers_data = load_txt_and_build_vectorstore(
        txt_path=txt_file_path,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    vectorstore = retrievers_data["vectorstore"]
    splade_retriever = retrievers_data["splade_retriever"]  # now a BM25Retriever

    dense_docs = vectorstore.as_retriever(
        search_kwargs={"k": int(top_k)}
    ).invoke(dense_query)                    # <-- uses HyDE passage if enabled
    splade_docs = splade_retriever.invoke(sparse_query, top_k=int(top_k))  # <-- uses expanded query

    initial_docs = reciprocal_rank_fusion(
        [dense_docs, splade_docs],
        weights=[0.75, 0.25],
        top_k=int(top_k),
    )
    # Stage 2: Critique Filtering
    if (
        critique_prompt
        and critique_prompt.strip()
        and len(initial_docs) > final_top_k
    ):
        # 1. Force JSON format at decoding level
        critique_llm = ChatOllama(
            model=critique_model, 
            temperature=0.0,
            format="json"
        )

        candidates_formatted = "\n\n".join(
            [
                f"[ID: {idx}]\n{doc.page_content}"
                for idx, doc in enumerate(initial_docs)
            ]
        )

        filter_system_template = FILTER_VARIANTS.get(
            filter_variant_key, FILTER_VARIANTS["filter_v1"]
        )

        # 2. Corrected 2-element tuples: ("role", "content_template")
        filter_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", filter_system_template),
                (
                    "user",
                    "USER QUERY: {user_query}\n\n"
                    "CANDIDATE CHUNKS:\n{candidates}\n\n"
                    "Instructions: Respond ONLY with a valid JSON object containing a 'selected_ids' list of integers.\n"
                    'Example JSON output format: {{"selected_ids": [0, 1, 3]}}\n'
                    "You MUST select at least one chunk ID from the candidates above.",
                ),
            ]
        )

        filter_chain = filter_prompt | critique_llm

        try:
            critique_response = filter_chain.invoke(
                {
                    "critique_criteria": critique_prompt,
                    "final_top_k": final_top_k,
                    "user_query": query,
                    "candidates": candidates_formatted,
                }
            )

            # Strip reasoning / <think> tags if present
            clean_content = re.sub(
                r"<think>.*?</think>", "", critique_response.content, flags=re.DOTALL
            ).strip()

            # Parse JSON output
            data = json.loads(clean_content)

            if isinstance(data, list):
                raw_numbers = data
            elif isinstance(data, dict):
                raw_numbers = data.get("selected_ids", data.get("ids", []))
            else:
                raw_numbers = []

            # Fallback regex extraction if JSON dictionary keys were omitted
            if not raw_numbers:
                raw_numbers = re.findall(r"\d+", clean_content)

            # Filter for valid, unique document indices within range
            valid_ids = []
            for item in raw_numbers:
                if str(item).isdigit():
                    idx = int(item)
                    if 0 <= idx < len(initial_docs) and idx not in valid_ids:
                        valid_ids.append(idx)

            if valid_ids:
                filtered_docs = [
                    initial_docs[idx] for idx in valid_ids[:final_top_k]
                ]
            else:
                raise ValueError("JSON returned no valid candidate indices within range.")

        except Exception as e:
            print(
                f"[WARNING] Critique parsing failed ({e}). Defaulting to top {final_top_k} candidates."
            )
            filtered_docs = initial_docs[:final_top_k]
    else:
        filtered_docs = initial_docs[:final_top_k]
    # Stage 3: Final Answer Generation
    context_str = "\n\n--- CHUNK BREAK ---\n\n".join(
        [doc.page_content for doc in filtered_docs]
    )

    rag_system_template = RAG_VARIANTS.get(
        rag_variant_key, RAG_VARIANTS["rag_v1"]
    )

    llm = ChatOllama(model=generation_model, temperature=float(temperature))
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"{rag_system_template}\n\n=== CONTEXT START ===\n{{retrieved_data}}\n=== CONTEXT END ===",
            ),
            ("user", "Question: {user_query}"),
        ]
    )

    chain = prompt | llm
    response = chain.invoke(
        {
            "retrieved_data": context_str,
            "user_query": query,
        }
    )

    latency = round(time.time() - start_time, 2)
    return response.content, context_str, f"{latency}s"