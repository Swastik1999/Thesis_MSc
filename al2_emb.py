# embeddings.py

from typing import List
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings



class LocalOllamaEmbeddings(Embeddings):
    """Local embedding wrapper utilizing PyTorch & CUDA directly on your GPU.

    Bypasses Ollama HTTP sockets for maximum speed and zero timeout errors.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        **kwargs,
    ):
        # Support accepting both 'model' or 'model_name' keyword arguments
        model_name = kwargs.get("model", model_name)

        # Fallback to local HuggingFace model if Ollama model name passed in
        if "nomic" in model_name or "ollama" in model_name:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"

        print(
            f"[INFO] Initializing Direct GPU Embeddings using CUDA with model: {model_name}"
        )

        # Runs natively on your RTX 3070 CUDA cores with batching
        self._embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda"},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 128,  # High batch size for fast 172MB indexing
            },
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of text chunks in parallel on GPU."""
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single user query."""
        return self._embeddings.embed_query(text)