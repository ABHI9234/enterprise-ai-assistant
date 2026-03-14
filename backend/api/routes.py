import os
import time
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from loguru import logger
from backend.models.schemas import (
    QueryRequest, QueryResponse, UploadResponse,
    AdminStats, HealthResponse, DocumentInfo
)
from backend.services.ingestion import load_document, save_uploaded_file
from backend.services.chunking import chunk_document
from backend.services.embeddings import embed_chunks
from backend.vector_store import vector_store
from backend.rag_pipeline import run_rag_pipeline, stream_rag_pipeline
from config.settings import settings


router = APIRouter()


@router.get('/health', response_model=HealthResponse)
async def health_check():
    stats = vector_store.get_stats()
    return HealthResponse(
        status='healthy',
        app_name=settings.app_name,
        version=settings.app_version,
        index_loaded=vector_store.index is not None,
        total_vectors=stats['total_vectors'],
    )


@router.post('/upload', response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    allowed_types = {'.pdf', '.docx', '.txt', '.md'}
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f'Unsupported file type: {ext}. Allowed: {allowed_types}'
        )

    try:
        file_bytes = await file.read()
        file_path = save_uploaded_file(file_bytes, file.filename, settings.upload_dir)
        document = load_document(file_path)
        chunks = chunk_document(document)
        chunks = embed_chunks(chunks)
        vector_store.add_chunks(chunks)
        vector_store.save()

        logger.success(f'Document uploaded and indexed: {file.filename}')

        return UploadResponse(
            message='Document uploaded and indexed successfully',
            filename=file.filename,
            chunks_created=len(chunks),
            char_count=document['char_count'],
        )

    except Exception as e:
        logger.error(f'Upload failed for {file.filename}: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/query', response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    if vector_store.index is None or vector_store.index.ntotal == 0:
        raise HTTPException(
            status_code=400,
            detail='No documents uploaded yet. Please upload documents first.'
        )

    try:
        result = run_rag_pipeline(
            question=request.question,
            chat_history=request.chat_history,
            top_k=request.top_k,
        )

        return QueryResponse(
            answer=result['answer'],
            citations=result['citations'],
            latency_ms=result['latency_ms'],
            chunks_used=len(result['chunks']),
        )

    except Exception as e:
        logger.error(f'Query failed: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/stream')
async def stream_query(request: QueryRequest):
    if vector_store.index is None or vector_store.index.ntotal == 0:
        raise HTTPException(
            status_code=400,
            detail='No documents uploaded yet. Please upload documents first.'
        )

    def generate():
        try:
            for token in stream_rag_pipeline(
                question=request.question,
                chat_history=request.chat_history,
                top_k=request.top_k,
            ):
                yield token
        except Exception as e:
            logger.error(f'Streaming failed: {e}')
            yield f'Error: {str(e)}'

    return StreamingResponse(generate(), media_type='text/plain')


@router.get('/documents', response_model=List[DocumentInfo])
async def list_documents():
    stats = vector_store.get_stats()
    documents = []

    for filename in stats['documents']:
        ext = os.path.splitext(filename)[1].lower()
        chunk_count = sum(
            1 for m in vector_store.metadata
            if m['filename'] == filename
        )
        documents.append(DocumentInfo(
            filename=filename,
            file_type=ext,
            chunk_count=chunk_count,
        ))

    return documents


@router.delete('/documents/{filename}')
async def delete_document(filename: str):
    removed = vector_store.delete_document(filename)

    if removed == 0:
        raise HTTPException(
            status_code=404,
            detail=f'Document not found: {filename}'
        )

    return {'message': f'Deleted {filename}', 'chunks_removed': removed}


@router.get('/admin/stats', response_model=AdminStats)
async def admin_stats():
    stats = vector_store.get_stats()
    return AdminStats(
        total_documents=stats['total_documents'],
        total_vectors=stats['total_vectors'],
        documents=stats['documents'],
        embedding_dimension=stats['dimension'],
    )

@router.get('/admin/evaluation')
async def get_evaluation():
    from backend.evaluation.metrics import get_evaluation_summary
    return get_evaluation_summary()
