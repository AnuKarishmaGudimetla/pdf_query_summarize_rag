from embedder import Embedder
from vector_store import VectorStore
import numpy as np
from typing import List, Dict, Optional

class Retriever:
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        index_path: str = "data/faiss_index/index.faiss",
        metadata_path: str = "data/faiss_index/metadata.pkl"
    ):
        self.embedder = Embedder()

        if vector_store:
            self.store = vector_store
        else:
            self.store = VectorStore(dim=384, index_path=index_path, metadata_path=metadata_path)
            self.store.load()

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Returns top-k dicts with 'text' and 'source'.
        """
        q_emb = self.embedder.embed([query]).astype("float32")
        return self.store.search(q_emb, top_k)
