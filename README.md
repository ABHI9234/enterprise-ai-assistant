# Enterprise AI Knowledge Assistant

A production-grade RAG (Retrieval Augmented Generation) system that lets users upload internal company documents and ask questions in natural language.

## Architecture

- **Backend**: FastAPI + LangChain + FAISS
- **Frontend**: Streamlit
- **Embeddings**: BAAI/bge-base-en-v1.5 (HuggingFace)
- **LLM**: Groq (llama-3.1-8b-instant)
- **Vector DB**: FAISS
- **Deployment**: Docker + Render

## Features

- Upload PDF, DOCX, TXT, and Markdown documents
- Natural language question answering using RAG
- Source citations with every answer
- Streaming responses
- Multi-document support
- Chat memory across conversation turns
- Evaluation metrics (retrieval score, context utilization, latency)
- Admin dashboard with system stats
- Fully Dockerized

## Quick Start

### Local Development

1. Clone the repository
2. Create virtual environment:
```bash
   python3.10 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
```
3. Copy `.env.example` to `.env` and add your Groq API key
4. Start the backend:
```bash
   python -m uvicorn backend.main:app --reload --port 8000
```
5. Start the frontend:
```bash
   python -m streamlit run frontend/app.py --server.port 8501
```

### Docker
```bash
cd docker
docker compose up --build
```

Visit http://localhost:8501

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/v1/health | Health check |
| POST | /api/v1/upload | Upload document |
| POST | /api/v1/query | Query documents |
| POST | /api/v1/stream | Stream response |
| GET | /api/v1/documents | List documents |
| DELETE | /api/v1/documents/{filename} | Delete document |
| GET | /api/v1/admin/stats | Admin statistics |
| GET | /api/v1/admin/evaluation | Evaluation metrics |

## Project Structure
```
enterprise-ai-assistant/
├── backend/
│   ├── api/          # FastAPI routes
│   ├── services/     # Ingestion, chunking, embeddings
│   ├── rag_pipeline/ # RAG pipeline, retriever, LLM
│   ├── vector_store/ # FAISS vector database
│   └── evaluation/   # Metrics and evaluation
├── frontend/
│   ├── pages/        # Chat, Upload, Admin pages
│   └── utils/        # API client
├── config/           # Settings and prompts
├── docker/           # Dockerfiles and compose
└── data/             # Uploaded files and FAISS index
```

## Environment Variables
```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.1-8b-instant
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K_RESULTS=5
```
# Enterprise AI Knowledge Assistant(https://abhi9234-enterprise-ai-assistant.hf.space)
