import time

from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import List



import ollama

from langchain.embeddings.base import Embeddings

from tqdm import tqdm





class LocalOllamaEmbeddings(Embeddings):

    def __init__(

        self,

        model_name: str,

        host: str = "http://127.0.0.1:11440",

        max_workers: int = 4,

        max_retries: int = 3,

    ):

        self.model_name = model_name

        self.max_workers = max_workers

        self.max_retries = max_retries

        self.client = ollama.Client(host=host)



    def _embed_one(self, text: str) -> List[float]:

        """Embed a single text with retries."""

        for attempt in range(self.max_retries):

            try:

                resp = self.client.embed(

                    model=self.model_name,

                    input=[text],

                )

                return resp["embeddings"][0]

            except Exception as e:

                wait_time = min(10, 2**attempt)

                print(f"[Retry {attempt + 1}] {e} -> waiting {wait_time}s")

                time.sleep(wait_time)



        resp = self.client.embed(

            model=self.model_name,

            input=[text],

        )

        return resp["embeddings"][0]



    def embed_documents(self, texts: List[str]) -> List[List[float]]:

        """Embed documents in parallel using threads."""

        results = [None] * len(texts)



        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:

            futures = {

                executor.submit(self._embed_one, text): idx

                for idx, text in enumerate(texts)

            }



            with tqdm(total=len(texts), desc="Embedding", leave=False) as pbar:

                for future in as_completed(futures):

                    idx = futures[future]

                    try:

                        results[idx] = future.result()

                    except Exception as e:

                        print(f"Text {idx} failed: {e}")

                        results[idx] = [0.0] * 768

                    pbar.update(1)



        return results



    def embed_query(self, text: str) -> List[float]:

        return self._embed_one(text)