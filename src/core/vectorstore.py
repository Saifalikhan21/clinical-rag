from functools import lru_cache
import chromadb
from langchain_chroma import Chroma
from src.core.config import get_settings
from src.core.embeddings import get_embeddings


def get_chroma_client() -> chromadb.PersistentClient:
    settings = get_settings()
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


@lru_cache
def get_vectorstore() -> Chroma:
    settings = get_settings()
    return Chroma(
        client=get_chroma_client(),
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embeddings(),
    )


def get_retriever():
    settings = get_settings()
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": settings.retriever_k,
            "score_threshold": settings.retriever_score_threshold,
        },
    )
