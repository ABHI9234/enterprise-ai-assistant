from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger
from config.settings import settings


# Load model once at module level so it is not reloaded on every call
_model = None


def get_embedding_model() -> SentenceTransformer:
    """
    Load and cache the embedding model.
    Downloads on first call, uses cache on subsequent calls.
    """
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        logger.info("This may take a few minutes on first run (downloading model)...")
        _model = SentenceTransformer(settings.embedding_model)
        logger.success(f"Embedding model loaded successfully")
    return _model


def generate_embedding(text: str) -> np.ndarray:
    """
    Generate a single embedding vector for a text string.

    Args:
        text: The text to embed

    Returns:
        numpy array of shape (768,)
    """
    model = get_embedding_model()

    # BGE models work best with a query prefix for retrieval
    embedding = model.encode(
        text,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embedding.astype(np.float32)


def generate_embeddings_batch(texts: List[str]) -> np.ndarray:
    """
    Generate embeddings for a list of texts efficiently in batch.

    Args:
        texts: List of text strings to embed

    Returns:
        numpy array of shape (len(texts), 768)
    """
    if not texts:
        raise ValueError("Cannot generate embeddings for empty list")

    model = get_embedding_model()

    logger.info(f"Generating embeddings for {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )

    logger.success(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

    return embeddings.astype(np.float32)


def embed_chunks(chunks: List[dict]) -> List[dict]:
    """
    Add embedding vectors to a list of chunks.

    Args:
        chunks: List of chunk dicts from chunk_document()

    Returns:
        Same chunks with 'embedding' key added to each
    """
    texts = [chunk["text"] for chunk in chunks]
    embeddings = generate_embeddings_batch(texts)

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    logger.success(f"Embedded {len(chunks)} chunks successfully")
    return chunks


def generate_query_embedding(query: str) -> np.ndarray:
    """
    Generate embedding for a user query.
    BGE models use a special prefix for queries to improve retrieval.

    Args:
        query: The user's question

    Returns:
        numpy array of shape (768,)
    """
    # BGE recommendation: prefix queries with this instruction
    prefixed_query = f"Represent this sentence for searching relevant passages: {query}"
    return generate_embedding(prefixed_query)
