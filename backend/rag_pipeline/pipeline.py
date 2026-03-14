import time
from typing import Iterator, List
from loguru import logger
from backend.rag_pipeline.retriever import (
    retrieve_relevant_chunks,
    format_context,
    format_citations,
)
from backend.rag_pipeline.llm import get_rag_response, stream_rag_response
from backend.evaluation.metrics import evaluate_rag_response


def run_rag_pipeline(
    question: str,
    chat_history: List[dict] = None,
    top_k: int = 5,
) -> dict:
    start_time = time.time()
    logger.info(f'RAG pipeline started for: {question[:80]}')

    chunks = retrieve_relevant_chunks(question, top_k=top_k)

    if not chunks:
        latency_ms = int((time.time() - start_time) * 1000)
        return {
            'answer': 'I could not find the answer in the provided documents.',
            'citations': '',
            'chunks': [],
            'latency_ms': latency_ms,
            'evaluation': None,
        }

    context = format_context(chunks)
    citations = format_citations(chunks)
    answer = get_rag_response(question, context, chat_history)
    latency_ms = int((time.time() - start_time) * 1000)

    evaluation = evaluate_rag_response(question, answer, chunks, latency_ms)

    logger.success(f'RAG pipeline completed in {latency_ms}ms | score={evaluation["overall_score"]}')

    return {
        'answer': answer,
        'citations': citations,
        'chunks': chunks,
        'latency_ms': latency_ms,
        'evaluation': evaluation,
    }


def stream_rag_pipeline(
    question: str,
    chat_history: List[dict] = None,
    top_k: int = 5,
) -> Iterator[str]:
    logger.info(f'Streaming RAG pipeline for: {question[:80]}')

    chunks = retrieve_relevant_chunks(question, top_k=top_k)

    if not chunks:
        yield 'I could not find the answer in the provided documents.'
        return

    context = format_context(chunks)
    citations = format_citations(chunks)

    for token in stream_rag_response(question, context, chat_history):
        yield token

    yield '\n\n---\n' + citations