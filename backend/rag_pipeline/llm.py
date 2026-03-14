from typing import Iterator, List
from groq import Groq
from loguru import logger
from config.settings import settings
from config.prompts import RAG_SYSTEM_PROMPT, RAG_HUMAN_PROMPT


client = Groq(api_key=settings.groq_api_key)


def get_rag_response(question: str, context: str, chat_history: List[dict] = None) -> str:
    messages = _build_messages(question, context, chat_history)
    logger.info(f'Calling Groq LLM ({settings.groq_model}) for: {question[:80]}')

    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content
    logger.success(f'LLM response received ({len(answer)} characters)')
    return answer


def stream_rag_response(
    question: str,
    context: str,
    chat_history: List[dict] = None
) -> Iterator[str]:
    messages = _build_messages(question, context, chat_history)
    logger.info(f'Streaming Groq LLM response for: {question[:80]}')

    stream = client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


def _build_messages(
    question: str,
    context: str,
    chat_history: List[dict] = None
) -> List[dict]:
    messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]

    if chat_history:
        for turn in chat_history[-6:]:
            messages.append(turn)

    human_message = RAG_HUMAN_PROMPT.format(
        context=context,
        question=question,
    )
    messages.append({"role": "user", "content": human_message})
    return messages