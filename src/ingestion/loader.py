import os
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    BSHTMLLoader,
)
from langchain_core.documents import Document
from loguru import logger

SUPPORTED_EXTENSIONS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".html": BSHTMLLoader,
    ".htm": BSHTMLLoader,
}


def load_document(file_path: str | Path) -> list[Document]:
    path = Path(file_path)
    ext = path.suffix.lower()

    loader_cls = SUPPORTED_EXTENSIONS.get(ext)
    if not loader_cls:
        raise ValueError(f"Unsupported file type: {ext}")

    logger.info(f"Loading {path.name} ({ext})")
    loader = loader_cls(str(path))
    docs = loader.load()

    # Enrich metadata
    for doc in docs:
        doc.metadata.setdefault("source", path.name)
        doc.metadata.setdefault("file_path", str(path))
        doc.metadata.setdefault("file_type", ext.lstrip("."))

    return docs


def load_directory(directory: str | Path) -> list[Document]:
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    all_docs: list[Document] = []
    for file_path in dir_path.rglob("*"):
        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                all_docs.extend(load_document(file_path))
            except Exception as e:
                logger.warning(f"Failed to load {file_path.name}: {e}")

    logger.info(f"Loaded {len(all_docs)} pages from {directory}")
    return all_docs
