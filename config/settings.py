from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Groq
    groq_api_key: str
    groq_model: str = "llama-3.1-8b-instant"

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"

    # Storage paths
    faiss_index_path: str = "data/faiss_index"
    upload_dir: str = "data/uploads"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_results: int = 5

    # App
    app_name: str = "Enterprise AI Knowledge Assistant"
    app_version: str = "1.0.0"
    debug: bool = True
    backend_url: str = "http://localhost:8000"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
