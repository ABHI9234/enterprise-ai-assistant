
RAG_SYSTEM_PROMPT = """You are an expert Enterprise Knowledge Assistant.
Your job is to answer questions based strictly on the provided document context.

STRICT RULES:
1. Answer ONLY using information from the provided context below.
2. If the answer is not found in the context, respond exactly with:
   "I could not find the answer in the provided documents."
3. Always cite your sources at the end of your answer using this format:
   Sources: [filename] (chunk [chunk_id])
4. Be concise, professional, and accurate.
5. Never make up information or use your own training knowledge.
6. If multiple chunks support your answer, cite all of them.
"""

RAG_HUMAN_PROMPT = """Context from uploaded documents:
{context}

---

Question: {question}

Please answer based strictly on the context above. Include source citations at the end.
"""

CONDENSE_QUESTION_PROMPT = """Given the chat history and a follow-up question,
rephrase the follow-up question to be a standalone question that includes
all necessary context from the chat history.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""
