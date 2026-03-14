from pydantic import BaseModel, Field
from typing import List, Optional


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    chat_history: Optional[List[dict]] = Field(default=None)


class QueryResponse(BaseModel):
    answer: str
    citations: str
    latency_ms: int
    chunks_used: int


class DocumentInfo(BaseModel):
    filename: str
    file_type: str
    chunk_count: int


class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_created: int
    char_count: int


class AdminStats(BaseModel):
    total_documents: int
    total_vectors: int
    documents: List[str]
    embedding_dimension: int


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    index_loaded: bool
    total_vectors: int