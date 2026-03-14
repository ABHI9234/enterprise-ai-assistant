from typing import List
from loguru import logger
from backend.services.embeddings import generate_query_embedding
from backend.vector_store import vector_store
from config.settings import settings


def retrieve_relevant_chunks(query: str, top_k: int = None) -> List[dict]:
    if top_k is None:
        top_k = settings.top_k_results

    logger.info(f'Retrieving top-{top_k} chunks for query: {query[:80]}')

    query_embedding = generate_query_embedding(query)
    results = vector_store.search(query_embedding, top_k=top_k)

    if not results:
        logger.warning('No relevant chunks found for query')
        return []

    logger.info(f'Retrieved {len(results)} chunks. Top score: {results[0]["similarity_score"]}')
    return results


def format_context(chunks: List[dict]) -> str:
    if not chunks:
        return 'No relevant context found.'

    context_parts = []
    for i, chunk in enumerate(chunks):
        filename = chunk['metadata']['filename']
        chunk_id = chunk['metadata']['chunk_id']
        score = chunk['similarity_score']
        text = chunk['text']
        context_parts.append(
            f'[Source {i+1}: {filename} | Chunk {chunk_id} | Relevance: {score}]\n{text}'
        )

    return '\n\n'.join(context_parts)


def format_citations(chunks: List[dict]) -> str:
    if not chunks:
        return ''

    seen = set()
    citations = []

    for chunk in chunks:
        filename = chunk['metadata']['filename']
        chunk_id = chunk['metadata']['chunk_id']
        key = f'{filename}_chunk_{chunk_id}'

        if key not in seen:
            seen.add(key)
            citations.append(f'- {filename} (chunk {chunk_id})')

    return 'Sources:\n' + '\n'.join(citations)