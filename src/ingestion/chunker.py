from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.core.config import get_settings


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = get_text_splitter()
    chunks = splitter.split_documents(documents)
    # Propagate source metadata to every chunk
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    return chunks
