from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

from src.rag.graph import query as run_query

router = APIRouter(prefix="/query", tags=["Query"])


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, description="Clinical question to answer")


class SourceReference(BaseModel):
    source: str
    page: str | int | None
    chunk_index: int | None
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
    question: str


@router.post("/", response_model=QueryResponse)
def answer_clinical_question(request: QueryRequest):
    logger.info(f"Query received: {request.question[:80]}...")
    try:
        result = run_query(request.question)
        return QueryResponse(
            answer=result["answer"],
            sources=[SourceReference(**s) for s in result["sources"]],
            question=request.question,
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to process query.")
