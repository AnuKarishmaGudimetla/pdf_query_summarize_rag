import faiss
import numpy as np
import os
import pickle
from typing import List, Dict

class VectorStore:
    def __init__(
        self,
        dim: int,
        index_path: str = "data/faiss_index/index.faiss",
        metadata_path: str = "data/faiss_index/metadata.pkl"
    ):
        self.dim = dim
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = faiss.IndexFlatL2(dim)
        self.metadata: List[Dict] = []

    def add(self, embeddings: np.ndarray, metadatas: List[Dict]):
        """
        Add embeddings and associated metadata dicts to FAISS.
        """
        self.index.add(embeddings)
        self.metadata.extend(metadatas)

    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            raise FileNotFoundError("FAISS index not found.")
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            raise FileNotFoundError("Metadata file not found.")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Return top_k metadata dicts (each has 'text' and 'source').
        """
        distances, indices = self.index.search(query_embedding, top_k)
        return [ self.metadata[i] for i in indices[0] ]
