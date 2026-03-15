import os
import httpx
import streamlit as st

BASE_URL = os.environ.get("BACKEND_URL", "http://localhost:8000") + "/api/v1"


def get_health():
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=5)
        return r.json()
    except Exception:
        return None


def upload_document(file_bytes: bytes, filename: str) -> dict:
    files = {"file": (filename, file_bytes, "application/octet-stream")}
    r = httpx.post(f"{BASE_URL}/upload", files=files, timeout=120)
    r.raise_for_status()
    return r.json()


def query_documents(question: str, chat_history: list, top_k: int = 5) -> dict:
    payload = {"question": question, "chat_history": chat_history, "top_k": top_k}
    r = httpx.post(f"{BASE_URL}/query", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def stream_query(question: str, chat_history: list, top_k: int = 5):
    payload = {"question": question, "chat_history": chat_history, "top_k": top_k}
    with httpx.stream("POST", f"{BASE_URL}/stream", json=payload, timeout=60) as r:
        for chunk in r.iter_text():
            if chunk:
                yield chunk


def get_documents() -> list:
    r = httpx.get(f"{BASE_URL}/documents", timeout=10)
    r.raise_for_status()
    return r.json()


def delete_document(filename: str) -> dict:
    r = httpx.delete(f"{BASE_URL}/documents/{filename}", timeout=10)
    r.raise_for_status()
    return r.json()


def get_admin_stats() -> dict:
    r = httpx.get(f"{BASE_URL}/admin/stats", timeout=10)
    r.raise_for_status()
    return r.json()


def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = []
