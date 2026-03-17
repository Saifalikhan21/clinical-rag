from pathlib import Path
from langchain_core.documents import Document
from loguru import logger

from src.core.vectorstore import get_vectorstore
from src.ingestion.loader import load_document, load_directory
from src.ingestion.chunker import chunk_documents


def ingest_file(file_path: str | Path) -> int:
    """Ingest a single document into ChromaDB. Returns number of chunks stored."""
    docs = load_document(file_path)
    chunks = chunk_documents(docs)
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    logger.success(f"Ingested {len(chunks)} chunks from {Path(file_path).name}")
    return len(chunks)


def ingest_directory(directory: str | Path) -> int:
    """Ingest all supported documents from a directory. Returns total chunks stored."""
    docs = load_directory(directory)
    if not docs:
        logger.warning(f"No documents found in {directory}")
        return 0
    chunks = chunk_documents(docs)
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    logger.success(f"Ingested {len(chunks)} total chunks from {directory}")
    return len(chunks)


def list_ingested_sources() -> list[str]:
    """Return unique source document names currently in the vector store."""
    vectorstore = get_vectorstore()
    collection = vectorstore._collection
    results = collection.get(include=["metadatas"])
    sources = {m.get("source", "unknown") for m in results["metadatas"]}
    return sorted(sources)
