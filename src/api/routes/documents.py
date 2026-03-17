import shutil
import tempfile
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from loguru import logger

from src.ingestion.processor import ingest_file, list_ingested_sources

router = APIRouter(prefix="/documents", tags=["Documents"])

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".html", ".htm"}


class IngestResponse(BaseModel):
    filename: str
    chunks_ingested: int
    message: str


class SourcesResponse(BaseModel):
    sources: list[str]
    total: int


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        chunks = ingest_file(tmp_path)
        logger.success(f"Ingested {file.filename}: {chunks} chunks")
        return IngestResponse(
            filename=file.filename,
            chunks_ingested=chunks,
            message=f"Successfully ingested {chunks} chunks from '{file.filename}'.",
        )
    except Exception as e:
        logger.error(f"Ingestion failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.get("/sources", response_model=SourcesResponse)
async def get_ingested_sources():
    sources = list_ingested_sources()
    return SourcesResponse(sources=sources, total=len(sources))
