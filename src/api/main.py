from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.core.config import get_settings
from src.api.routes import query, documents


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(f"Starting Clinical RAG API — env={settings.environment}")
    # Warm up the embedding model and vectorstore so the first request
    # doesn't pay the cold-start cost (HuggingFace model load ~10-30s).
    logger.info("Warming up embeddings model and vectorstore…")
    from src.core.embeddings import get_embeddings
    from src.core.vectorstore import get_vectorstore
    from src.rag.graph import rag_graph  # noqa: F401 — triggers graph compilation
    get_embeddings()
    get_vectorstore()
    logger.info("Warm-up complete — API ready")
    yield
    logger.info("Shutting down Clinical RAG API")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Clinical Document Intelligence API",
        description="Query clinical guidelines, drug protocols, and WHO/NHS documents using natural language.",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(query.router, prefix="/api/v1")
    app.include_router(documents.router, prefix="/api/v1")

    @app.get("/health", tags=["Health"])
    async def health():
        return {"status": "ok", "service": "clinical-rag"}

    return app


app = create_app()
