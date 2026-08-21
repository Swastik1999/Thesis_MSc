# pipeline.py
import os

cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
if os.path.exists(cuda_bin):
    os.add_dll_directory(cuda_bin)
    
import hashlib
import json
import pickle
import re
import time
from typing import Dict, List

from config import GlobalConfig
from embeddings import LocalOllamaEmbeddings
from fastembed import SparseTextEmbedding
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import prompt variants from prompts.py
from prompts import FILTER_VARIANTS, RAG_VARIANTS

_vs_cache = {}

# Project root = folder containing this file (pipeline.py)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REAG_DIR = os.path.join(PROJECT_ROOT, "ReAG")
os.makedirs(REAG_DIR, exist_ok=True)

# =====================================================================
# 1. SPLADE SPARSE RETRIEVER CLASS (GPU-accelerated + picklable for disk cache)
# =====================================================================
class SpladeRetriever:
    """Computes SPLADE learned sparse embeddings locally using FastEmbed,
    with GPU execution via ONNX Runtime's CUDAExecutionProvider when
    available, falling back to CPU automatically otherwise.

    NOTE ON DISK CACHING: FastEmbed's SparseTextEmbedding model object
    (self.splade_model) wraps an ONNX Runtime session, which is generally
    NOT reliably picklable across processes/restarts. To keep this class
    disk-cache-safe, the model is NOT stored on the instance that gets
    pickled -- only the computed doc_embeddings and documents are. The
    model is re-initialized (cheap, no need to re-embed 130k+ chunks) each
    time the retriever is loaded, whether freshly built or restored from
    cache. See load_txt_and_build_vectorstore() for how this is used.
    """

    def __init__(
        self,
        documents: List[Document],
        model_name: str = "prithivida/Splade_PP_en_v1",
        use_gpu: bool = True,
        batch_size: int = 64,
        doc_embeddings=None,  # pass pre-computed embeddings to skip re-encoding
    ):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.documents = documents

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu
            else ["CPUExecutionProvider"]
        )

        print(
            f"[INFO] Initializing SPLADE model '{model_name}' "
            f"(providers={providers})..."
        )
        try:
            self.splade_model = SparseTextEmbedding(
                model_name=model_name, providers=providers
            )
        except Exception as e:
            print(
                f"[WARNING] Failed to init SPLADE with providers={providers} ({e}). "
                "Falling back to CPU."
            )
            self.splade_model = SparseTextEmbedding(
                model_name=model_name, providers=["CPUExecutionProvider"]
            )

        if doc_embeddings is not None:
            print(
                f"[INFO] Using {len(doc_embeddings)} cached SPLADE sparse vectors "
                "(skipping re-encoding)."
            )
            self.doc_embeddings = doc_embeddings
        else:
            print(
                f"[INFO] Computing SPLADE sparse vectors for {len(documents)} "
                f"document chunks (batch_size={batch_size})..."
            )
            texts = [doc.page_content for doc in documents]
            self.doc_embeddings = list(
                self.splade_model.embed(texts, batch_size=batch_size)
            )

    def invoke(self, query: str, top_k: int = 25) -> List[Document]:
        """Encodes user query into SPLADE sparse format and computes sparse dot-products."""
        query_embedding = list(self.splade_model.embed([query]))[0]

        # Pre-build dict lookups per doc for O(1) token lookup instead of
        # repeated list.index() scans (meaningful speedup at scale).
        scores = []
        for idx, doc_emb in enumerate(self.doc_embeddings):
            doc_token_to_weight = dict(zip(doc_emb.indices, doc_emb.values))
            score = 0.0
            for token_id, q_weight in zip(
                query_embedding.indices, query_embedding.values
            ):
                if token_id in doc_token_to_weight:
                    score += q_weight * doc_token_to_weight[token_id]
            scores.append((score, idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [self.documents[idx] for _, idx in scores[:top_k]]

    def get_cache_payload(self):
        """Returns the minimal picklable data needed to reconstruct this
        retriever without re-running SPLADE inference. The ONNX model
        session itself is excluded (not reliably picklable)."""
        return {
            "documents": self.documents,
            "doc_embeddings": self.doc_embeddings,
            "model_name": self.model_name,
        }

    @classmethod
    def from_cache_payload(cls, payload: dict, use_gpu: bool = True, batch_size: int = 64):
        """Reconstructs a SpladeRetriever from a cached payload, re-initializing
        the ONNX model (fast) but skipping the expensive document encoding step."""
        return cls(
            documents=payload["documents"],
            model_name=payload["model_name"],
            use_gpu=use_gpu,
            batch_size=batch_size,
            doc_embeddings=payload["doc_embeddings"],
        )


# =====================================================================
# 2. RECIPROCAL RANK FUSION (RRF) HELPER
# =====================================================================
def reciprocal_rank_fusion(
    results_list: List[List[Document]],
    weights: List[float] = None,
    top_k: int = 25,
    c: int = 60,
) -> List[Document]:
    """Combines ranked candidate lists from Sparse (SPLADE) and Dense retrievers using weighted RRF."""
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
    original query, improving SPLADE's lexical/token-overlap recall."""
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
    splade_use_gpu: bool = True,
    splade_batch_size: int = 64,
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
    os.makedirs(persist_dir, exist_ok=True)

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

    # --- SPLADE: load from disk cache if present, else compute + persist ---
    splade_cache_path = os.path.join(REAG_DIR, f"splade_vectors_{folder_hash}.pkl")

    if os.path.exists(splade_cache_path):
        print(f"[INFO] Loading cached SPLADE vectors from: {splade_cache_path}")
        with open(splade_cache_path, "rb") as f:
            payload = pickle.load(f)
        splade_retriever = SpladeRetriever.from_cache_payload(
            payload, use_gpu=splade_use_gpu, batch_size=splade_batch_size
        )
    else:
        splade_retriever = SpladeRetriever(
            documents=chunks,
            use_gpu=splade_use_gpu,
            batch_size=splade_batch_size,
        )
        print(f"[INFO] Persisting SPLADE vectors to: {splade_cache_path}")
        with open(splade_cache_path, "wb") as f:
            pickle.dump(splade_retriever.get_cache_payload(), f)

    cache_data = {
        "vectorstore": vectorstore,
        "splade_retriever": splade_retriever,
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
    use_query_expansion: bool = False,
    use_hyde: bool = False,
    query_processor_model: str = "qwen2.5:1.5b",
    splade_use_gpu: bool = True,
    splade_batch_size: int = 64,
):
    start_time = time.time()

    # Stage 0: Query Processing (optional)
    sparse_query = query
    dense_query = query

    if use_query_expansion:
        sparse_query = expand_query(query, llm_model=query_processor_model)

    if use_hyde:
        dense_query = generate_hyde_document(query, llm_model=query_processor_model)

    # Stage 1: Hybrid Retrieval (SPLADE + Dense)
    retrievers_data = load_txt_and_build_vectorstore(
        txt_path=txt_file_path,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        splade_use_gpu=splade_use_gpu,
        splade_batch_size=splade_batch_size,
    )

    vectorstore = retrievers_data["vectorstore"]
    splade_retriever = retrievers_data["splade_retriever"]

    dense_docs = vectorstore.as_retriever(
        search_kwargs={"k": int(top_k)}
    ).invoke(dense_query)
    splade_docs = splade_retriever.invoke(sparse_query, top_k=int(top_k))

    initial_docs = reciprocal_rank_fusion(
        [dense_docs, splade_docs],
        weights=[0.7, 0.3],
        top_k=int(top_k),
    )
    # Stage 2: Critique Filtering
    if (
        critique_prompt
        and critique_prompt.strip()
        and len(initial_docs) > final_top_k
    ):
        critique_llm = ChatOllama(model=critique_model, temperature=0.0)

        candidates_formatted = "\n\n".join(
            [
                f"--- CHUNK ID: {idx} ---\n{doc.page_content}"
                for idx, doc in enumerate(initial_docs)
            ]
        )

        filter_system_template = FILTER_VARIANTS.get(
            filter_variant_key, FILTER_VARIANTS["filter_v1"]
        )

        filter_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", filter_system_template),
                (
                    "user",
                    "USER QUERY: {user_query}\n\n"
                    "CANDIDATE CHUNKS:\n{candidates}\n\n"
                    "Remember: respond with ONLY the JSON array, nothing else.",
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
                    idx
                    for idx in selected_ids
                    if isinstance(idx, int) and 0 <= idx < len(initial_docs)
                ][:final_top_k]
                filtered_docs = [initial_docs[idx] for idx in valid_ids]
            else:
                raise ValueError("JSON parsing failed")

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