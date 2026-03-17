# CLAUDE.md — Clinical Document Intelligence System

## Project Purpose

Production-grade RAG application for medical staff to query clinical guidelines,
drug protocols, and WHO/NHS documents via natural language with grounded, cited answers.

## Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangChain + LangGraph |
| Vector DB | ChromaDB (persistent) |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI GPT-4o |
| Backend | FastAPI |
| Frontend | Streamlit |
| Monitoring | LangSmith |
| Container | Docker + Docker Compose |

## Project Layout

```
src/
  api/          FastAPI app, routes (query, documents)
  core/         Config, embeddings, vectorstore, LLM singletons
  rag/          LangGraph pipeline (retrieve → generate)
  ingestion/    Document loaders, chunker, processor
  frontend/     Streamlit app
data/
  raw/          Drop source PDFs/DOCXs here before ingesting
  processed/    Intermediate artifacts (not used yet)
  chroma_db/    Persistent ChromaDB vector store
tests/          pytest unit + integration tests
docs/           Architecture diagram and design notes
docker/         Dockerfiles for API and frontend
```

## Local Development

```bash
# 1. Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Fill in OPENAI_API_KEY and (optionally) LANGCHAIN_API_KEY

# 3. Start API
uvicorn src.api.main:app --reload

# 4. Start frontend (separate terminal)
streamlit run src/frontend/app.py

# 5. Run tests
pytest tests/ -v --cov=src
```

## Docker

```bash
docker-compose up --build
# API  → http://localhost:8000/docs
# UI   → http://localhost:8501
```

## Ingesting Documents

**Via API** — POST `/api/v1/documents/ingest` with a file upload.

**Via script** (bulk ingestion):
```python
from src.ingestion.processor import ingest_directory
ingest_directory("data/raw/")
```

## Important Constraints

- The system prompt enforces **no fabrication** — the LLM must cite sources or decline.
- Retrieval uses **score threshold** (default 0.35) to avoid low-quality matches.
- LangSmith tracing is opt-in via `LANGCHAIN_TRACING_V2=true` in `.env`.
- Never commit `.env` — use `.env.example` as the template.

## Adding New Document Types

1. Add a loader mapping in `src/ingestion/loader.py` → `SUPPORTED_EXTENSIONS`
2. Add the extension to `ALLOWED_EXTENSIONS` in `src/api/routes/documents.py`
3. Add the extension to the Streamlit file uploader in `src/frontend/app.py`
