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
            **kwargs,
        ):
            self.model_name = kwargs.get("model", model_name)
            self.host = host
            self.batch_size = batch_size
            self.max_retries = max_retries
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
        """Embeds a single user query."""
        res = self._embed_batch([text])
        return res[0]