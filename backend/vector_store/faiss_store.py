import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import faiss
from loguru import logger
from config.settings import settings


class FAISSVectorStore:
    """
    FAISS-based vector store with metadata support.
    Handles storing, searching, and persisting embeddings.
    """

    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.metadata: List[dict] = []
        self.texts: List[str] = []
        self.dimension: int = 768
        self.index_path = Path(settings.faiss_index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

    def _get_index_file(self) -> Path:
        return self.index_path / "index.faiss"

    def _get_metadata_file(self) -> Path:
        return self.index_path / "metadata.pkl"

    def initialize_index(self) -> None:
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        self.texts = []
        logger.info(f"Initialized fresh FAISS index (dimension={self.dimension})")

    def add_chunks(self, chunks: List[dict]) -> None:
        if self.index is None:
            self.initialize_index()

        embeddings = []
        for chunk in chunks:
            if "embedding" not in chunk:
                raise ValueError(f"Chunk {chunk.get(chr(99)+chr(104)+chr(117)+chr(110)+chr(107)+'_id')} has no embedding")
            embeddings.append(chunk["embedding"])
            self.metadata.append(chunk["metadata"])
            self.texts.append(chunk["text"])

        embeddings_matrix = np.array(embeddings).astype(np.float32)
        self.index.add(embeddings_matrix)

        logger.success(
            f"Added {len(chunks)} chunks to FAISS index. "
            f"Total vectors: {self.index.ntotal}"
        )

    def search(self, query_embedding: np.ndarray, top_k: int = None) -> List[dict]:
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("FAISS index is empty. Please upload documents first.")

        if top_k is None:
            top_k = settings.top_k_results

        query_vector = query_embedding.reshape(1, -1).astype(np.float32)
        top_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            similarity = float(1 / (1 + distance))
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity_score": round(similarity, 4),
                "index": int(idx),
            })

        logger.info(f"Retrieved {len(results)} chunks for query")
        return results

    def save(self) -> None:
        if self.index is None:
            raise ValueError("No index to save")

        faiss.write_index(self.index, str(self._get_index_file()))

        with open(self._get_metadata_file(), "wb") as f:
            pickle.dump({"metadata": self.metadata, "texts": self.texts}, f)

        logger.success(
            f"Saved FAISS index to {self.index_path} "
            f"({self.index.ntotal} vectors)"
        )

    def load(self) -> bool:
        index_file = self._get_index_file()
        metadata_file = self._get_metadata_file()

        if not index_file.exists() or not metadata_file.exists():
            logger.warning("No existing FAISS index found. Starting fresh.")
            self.initialize_index()
            return False

        self.index = faiss.read_index(str(index_file))

        with open(metadata_file, "rb") as f:
            data = pickle.load(f)
            self.metadata = data["metadata"]
            self.texts = data["texts"]

        logger.success(
            f"Loaded FAISS index from {self.index_path} "
            f"({self.index.ntotal} vectors)"
        )
        return True

    def get_stats(self) -> dict:
        if self.index is None:
            return {
                "total_vectors": 0,
                "total_documents": 0,
                "documents": [],
                "dimension": self.dimension,
            }

        unique_docs = list(set(m["filename"] for m in self.metadata))

        return {
            "total_vectors": self.index.ntotal,
            "total_documents": len(unique_docs),
            "documents": unique_docs,
            "dimension": self.dimension,
        }

    def delete_document(self, filename: str) -> int:
        if self.index is None or self.index.ntotal == 0:
            return 0

        keep_indices = [
            i for i, m in enumerate(self.metadata)
            if m["filename"] != filename
        ]

        removed_count = self.index.ntotal - len(keep_indices)

        if removed_count == 0:
            logger.warning(f"Document not found in index: {filename}")
            return 0

        kept_texts = [self.texts[i] for i in keep_indices]
        kept_metadata = [self.metadata[i] for i in keep_indices]
        kept_embeddings = np.array([
            self.index.reconstruct(i) for i in keep_indices
        ]).astype(np.float32)

        self.initialize_index()

        if kept_embeddings.shape[0] > 0:
            self.index.add(kept_embeddings)
            self.texts = kept_texts
            self.metadata = kept_metadata

        self.save()

        logger.success(f"Removed {removed_count} chunks for {filename}")
        return removed_count


# Global singleton instance
vector_store = FAISSVectorStore()
