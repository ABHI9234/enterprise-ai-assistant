from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger
from config.settings import settings


def chunk_document(document: dict) -> List[dict]:
    """
    Split a document's text content into overlapping chunks.
    Each chunk carries metadata about its source document.

    Args:
        document: dict returned by load_document()

    Returns:
        List of chunk dicts, each with text and metadata
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = splitter.split_text(document["content"])

    if not raw_chunks:
        raise ValueError(f"No chunks generated for document: {document['filename']}")

    chunks = []
    for index, chunk_text in enumerate(raw_chunks):
        chunk = {
            "chunk_id": index,
            "text": chunk_text.strip(),
            "metadata": {
                "filename": document["filename"],
                "file_type": document["file_type"],
                "file_path": document["file_path"],
                "chunk_id": index,
                "total_chunks": len(raw_chunks),
                "char_count": len(chunk_text),
            },
        }
        chunks.append(chunk)

    logger.info(
        f"Chunked '{document['filename']}' into {len(chunks)} chunks "
        f"(size={settings.chunk_size}, overlap={settings.chunk_overlap})"
    )

    return chunks


def chunk_multiple_documents(documents: List[dict]) -> List[dict]:
    """
    Chunk multiple documents and return all chunks in a flat list.

    Args:
        documents: List of document dicts from load_document()

    Returns:
        Flat list of all chunks across all documents
    """
    all_chunks = []

    for document in documents:
        try:
            chunks = chunk_document(document)
            all_chunks.extend(chunks)
            logger.success(
                f"Processed '{document['filename']}': {len(chunks)} chunks"
            )
        except Exception as e:
            logger.error(f"Failed to chunk '{document['filename']}': {e}")
            raise

    logger.info(f"Total chunks across all documents: {len(all_chunks)}")
    return all_chunks
