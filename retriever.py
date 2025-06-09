from embedder import Embedder
from vector_store import VectorStore
import numpy as np

class Retriever:
    def __init__(self, index_path: str = "data/faiss_index/index.faiss", metadata_path: str = "data/faiss_index/metadata.pkl"):
        self.embedder = Embedder()
        self.store = VectorStore(dim=384, index_path=index_path, metadata_path=metadata_path)
        self.store.load()

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """
        Retrieve top-k relevant chunks for a given query.
        """
        query_embedding = self.embedder.embed([query])
        query_embedding = np.array(query_embedding).astype("float32")
        return self.store.search(query_embedding, top_k)
