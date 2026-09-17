# embeddings.py

import time
from typing import List
import ollama
from langchain_core.embeddings import Embeddings
from tqdm import tqdm
from config import GlobalConfig


class LocalOllamaEmbeddings(Embeddings):
    """Local Ollama embeddings using native batch requests.

    Prevents connection flooding by sending array batches to Ollama instead of
    threading. Incorporates persistent progress tracking to avoid resetting UI
    progress bars when ChromaDB batches inputs internally.
    """

    def __init__(
            self,
            # Default to the first embedding model in your config
            model_name: str = GlobalConfig.EMBEDDING_MODELS[0].strip(),
            host: str = "http://127.0.0.1:11434",
            batch_size: int = 128,
            max_retries: int = 3,
            # Instruction-aware models (Qwen3-Embedding, E5, BGE, GTE, ...) are
            # trained asymmetrically: the QUERY should be wrapped with a task
            # instruction, but documents should NOT be. Per Qwen3-Embedding's
            # own docs: "No need to add instruction for retrieval documents."
            # Set to None/"" to disable and embed queries plain (e.g. for
            # models that aren't instruction-aware).
            query_instruction: str = "Given a web search query, retrieve relevant passages that answer the query",
            **kwargs,
        ):
            self.model_name = kwargs.get("model", model_name)
            self.host = host
            self.batch_size = batch_size
            self.max_retries = max_retries
            self.query_instruction = query_instruction
            self.client = ollama.Client(host=self.host)
            self._global_pbar = None

    def _embed_batch(self, batch_texts: List[str]) -> List[List[float]]:
        """Sends an array batch to Ollama's embed API with retry support."""
        for attempt in range(self.max_retries):
            try:
                resp = self.client.embed(
                    model=self.model_name,
                    input=batch_texts,
                )
                return resp["embeddings"]
            except Exception as e:
                wait_time = min(10, 2**attempt)
                if self._global_pbar:
                    self._global_pbar.write(
                        f"[Retry {attempt + 1}] Connection issue: {e} -> waiting {wait_time}s"
                    )
                else:
                    print(
                        f"\n[Retry {attempt + 1}] Connection issue: {e} -> waiting {wait_time}s"
                    )
                time.sleep(wait_time)

        # Final attempt
        resp = self.client.embed(
            model=self.model_name,
            input=batch_texts,
        )
        return resp["embeddings"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds documents in batches with clean progress bar updates."""
        all_embeddings = []

        # Create progress bar if not already active
        pbar_created_here = False
        if self._global_pbar is None:
            self._global_pbar = tqdm(
                total=len(texts), desc="Ollama Embedding", leave=True
            )
            pbar_created_here = True
        else:
            # If Chroma called this again with a new sub-batch, update total target count
            self._global_pbar.total += len(texts)
            self._global_pbar.refresh()

        try:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                batch_res = self._embed_batch(batch)
                all_embeddings.extend(batch_res)
                self._global_pbar.update(len(batch))
        finally:
            # Close and clear the global progress bar when the entire job finishes
            if pbar_created_here:
                self._global_pbar.close()
                self._global_pbar = None

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single user query.

        If query_instruction is set, wraps the text in the
        'Instruct: {task}\\nQuery:{text}' format Qwen3-Embedding (and other
        instruction-aware models like E5/BGE/GTE) expect for queries.
        Documents are intentionally left unwrapped -- these models are
        trained asymmetrically and don't want an instruction on the
        document side.
        """
        query_text = text
        if self.query_instruction:
            query_text = f"Instruct: {self.query_instruction}\nQuery:{text}"
        res = self._embed_batch([query_text])
        return res[0]


class LocalSentenceTransformerEmbeddings(Embeddings):
    """Embeddings backed by a locally-loaded sentence-transformers model
    (e.g. Yuan-embedding-2.0-en, or any other HuggingFace model not
    published to Ollama's model library).

    Unlike LocalOllamaEmbeddings, this does NOT call an HTTP server --
    the model is loaded directly into this process via the
    sentence-transformers library. Requires:
        pip install -U sentence-transformers==3.4.1

    QUERY INSTRUCTION HANDLING:
    Yuan-embedding-2.0-en is a fine-tune of Qwen3-Embedding-0.6B, but its
    published usage example shows plain symmetric encoding with no
    'Instruct:'/'Query:' wrapping -- unlike the base Qwen3-Embedding
    model, which documents that format explicitly. It's unconfirmed
    whether the instruction requirement survived fine-tuning.

    Rather than guess, this class defers to the model's OWN bundled
    prompt config if it has one (sentence-transformers models can ship
    a `model.prompts` dict, e.g. {"query": "Instruct: ...\\nQuery:"} --
    if present, `model.encode(..., prompt_name="query")` applies it
    automatically). If no such prompt is bundled, queries and documents
    are encoded identically (matching the documented usage example).

    You can force an explicit prefix via `manual_query_instruction` if
    you determine (e.g. via the eval harness) that one helps, but treat
    that as a hypothesis to A/B test, not a known-correct default.
    """

    def __init__(
        self,
        model_name: str = "IEITYuan/Yuan-embedding-2.0-en",
        device: str = None,  # None = let sentence-transformers auto-pick (cuda if available, else cpu)
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        manual_query_instruction: str = None,  # opt-in override; unverified for this model, see docstring
        **kwargs,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = kwargs.get("model", model_name)
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.manual_query_instruction = manual_query_instruction

        self.model = SentenceTransformer(self.model_name, device=device)

        # Auto-detect a bundled "query" prompt in the model's own config,
        # rather than assuming Qwen3-Embedding's instruction format
        # carried over through fine-tuning.
        bundled_prompts = getattr(self.model, "prompts", None) or {}
        self.has_bundled_query_prompt = "query" in bundled_prompts
        if self.has_bundled_query_prompt:
            print(
                f"[INFO] Model '{self.model_name}' ships a bundled 'query' "
                f"prompt -- using it automatically for embed_query(): "
                f"{bundled_prompts['query']!r}"
            )
        elif manual_query_instruction:
            print(
                f"[INFO] No bundled query prompt found on the model. Using "
                f"manual_query_instruction override (UNVERIFIED for this "
                f"model -- confirm via eval harness): {manual_query_instruction!r}"
            )
        else:
            print(
                f"[INFO] No bundled query prompt found on the model, and no "
                f"manual_query_instruction set. Queries and documents will "
                f"be encoded identically (matches the model's published "
                f"usage example)."
            )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds documents. Never gets a query instruction/prompt --
        matches the asymmetric convention documents shouldn't carry one."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single query, applying the model's bundled 'query'
        prompt if it has one, else a manual override if set, else plain
        (identical to document encoding)."""
        if self.has_bundled_query_prompt:
            embedding = self.model.encode(
                [text],
                prompt_name="query",
                normalize_embeddings=self.normalize_embeddings,
            )
        elif self.manual_query_instruction:
            query_text = f"Instruct: {self.manual_query_instruction}\nQuery:{text}"
            embedding = self.model.encode(
                [query_text],
                normalize_embeddings=self.normalize_embeddings,
            )
        else:
            embedding = self.model.encode(
                [text],
                normalize_embeddings=self.normalize_embeddings,
            )
        return embedding[0].tolist()