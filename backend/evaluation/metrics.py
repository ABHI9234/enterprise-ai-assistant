import time
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from loguru import logger


EVAL_LOG_PATH = Path('data/evaluation_logs.json')


def compute_retrieval_score(chunks: List[dict]) -> float:
    '''
    Score retrieval quality based on similarity scores of retrieved chunks.
    Returns a score between 0 and 1.
    '''
    if not chunks:
        return 0.0
    scores = [c['similarity_score'] for c in chunks]
    return round(sum(scores) / len(scores), 4)


def compute_context_utilization(answer: str, chunks: List[dict]) -> float:
    '''
    Estimate how much of the retrieved context was used in the answer.
    Uses keyword overlap as a proxy metric.
    Returns a score between 0 and 1.
    '''
    if not chunks or not answer:
        return 0.0

    answer_words = set(answer.lower().split())
    context_words = set()
    for chunk in chunks:
        context_words.update(chunk['text'].lower().split())

    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'shall', 'can',
        'of', 'in', 'to', 'for', 'on', 'with', 'at', 'by', 'from',
        'and', 'or', 'but', 'not', 'this', 'that', 'it', 'its',
    }
    answer_words -= stop_words
    context_words -= stop_words

    if not context_words:
        return 0.0

    overlap = answer_words.intersection(context_words)
    return round(len(overlap) / max(len(answer_words), 1), 4)


def compute_answer_completeness(answer: str) -> float:
    '''
    Simple heuristic to check if the answer is complete and not a fallback.
    Returns 1.0 for complete answers, 0.0 for fallback responses.
    '''
    fallback_phrases = [
        'could not find',
        'not found in the',
        'no information',
        'i don\'t know',
    ]
    answer_lower = answer.lower()
    for phrase in fallback_phrases:
        if phrase in answer_lower:
            return 0.0
    return 1.0


def evaluate_rag_response(
    question: str,
    answer: str,
    chunks: List[dict],
    latency_ms: int,
) -> dict:
    '''
    Run all evaluation metrics on a RAG response.
    Returns a dict of all metrics.
    '''
    retrieval_score = compute_retrieval_score(chunks)
    context_utilization = compute_context_utilization(answer, chunks)
    answer_completeness = compute_answer_completeness(answer)

    metrics = {
        'timestamp': datetime.utcnow().isoformat(),
        'question': question[:200],
        'answer_length': len(answer),
        'chunks_retrieved': len(chunks),
        'retrieval_score': retrieval_score,
        'context_utilization': context_utilization,
        'answer_completeness': answer_completeness,
        'latency_ms': latency_ms,
        'overall_score': round(
            (retrieval_score + context_utilization + answer_completeness) / 3, 4
        ),
    }

    logger.info(
        f'Eval | retrieval={retrieval_score} | '
        f'context_util={context_utilization} | '
        f'completeness={answer_completeness} | '
        f'latency={latency_ms}ms'
    )

    _log_metrics(metrics)
    return metrics


def _log_metrics(metrics: dict) -> None:
    '''Append metrics to the evaluation log file.'''
    EVAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    logs = []
    if EVAL_LOG_PATH.exists():
        with open(EVAL_LOG_PATH, 'r') as f:
            try:
                logs = json.load(f)
            except Exception:
                logs = []

    logs.append(metrics)

    with open(EVAL_LOG_PATH, 'w') as f:
        json.dump(logs, f, indent=2)


def get_evaluation_summary() -> dict:
    '''
    Read all logged evaluations and return aggregate statistics.
    '''
    if not EVAL_LOG_PATH.exists():
        return {
            'total_queries': 0,
            'avg_retrieval_score': 0,
            'avg_context_utilization': 0,
            'avg_answer_completeness': 0,
            'avg_latency_ms': 0,
            'avg_overall_score': 0,
        }

    with open(EVAL_LOG_PATH, 'r') as f:
        try:
            logs = json.load(f)
        except Exception:
            return {'total_queries': 0}

    if not logs:
        return {'total_queries': 0}

    def avg(key):
        return round(sum(l[key] for l in logs) / len(logs), 4)

    return {
        'total_queries': len(logs),
        'avg_retrieval_score': avg('retrieval_score'),
        'avg_context_utilization': avg('context_utilization'),
        'avg_answer_completeness': avg('answer_completeness'),
        'avg_latency_ms': int(avg('latency_ms')),
        'avg_overall_score': avg('overall_score'),
        'recent_logs': logs[-5:],
    }