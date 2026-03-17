# Clinical Document Intelligence System

A production-grade RAG application that lets medical staff query clinical guidelines, drug protocols, and WHO/NHS documents using natural language and receive accurate, cited answers.

## Features

- **Natural language queries** over clinical PDFs, DOCX, HTML, and TXT documents
- **Grounded answers** — the LLM cites source document and page number for every claim
- **No-hallucination policy** — the system declines when relevant context is not found
- **Document ingestion UI** — upload files directly from the browser
- **LangSmith monitoring** — full trace observability out of the box
- **Docker-ready** — single `docker-compose up` to run the full stack

## Quick Start

### Prerequisites

- Python 3.12+
- OpenAI API key
- (Optional) LangSmith API key for tracing

### Local Setup

```bash
git clone <repo>
cd clinical-rag

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env — add your OPENAI_API_KEY

# Start API
uvicorn src.api.main:app --reload --port 8000

# Start UI (new terminal)
streamlit run src/frontend/app.py
```

- API docs: http://localhost:8000/docs
- Streamlit UI: http://localhost:8501

### Docker

```bash
cp .env.example .env   # fill in your keys
docker-compose up --build
```

## Usage

1. Open the Streamlit UI at http://localhost:8501
2. Upload a clinical document (PDF, DOCX, TXT, HTML) via the sidebar
3. Ask a natural language question in the chat input
4. Receive an answer with source citations

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/query/` | Ask a clinical question |
| `POST` | `/api/v1/documents/ingest` | Upload and ingest a document |
| `GET` | `/api/v1/documents/sources` | List ingested documents |

### Example Query

```bash
curl -X POST http://localhost:8000/api/v1/query/ \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the recommended aspirin dose for ACS?"}'
```

## Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full system diagram.

```
User → Streamlit → FastAPI → LangGraph (retrieve → generate) → ChromaDB + GPT-4o
```

## Tech Stack

- **LangChain / LangGraph** — RAG orchestration
- **ChromaDB** — persistent vector store
- **OpenAI** — embeddings (`text-embedding-3-small`) + LLM (`gpt-4o`)
- **FastAPI** — REST API
- **Streamlit** — frontend
- **LangSmith** — observability
- **Docker** — containerisation

## Security Notes

- Never commit `.env` — it contains API keys
- Set `ENVIRONMENT=production` to disable `/docs` and `/redoc`
- Rotate `API_SECRET_KEY` before deploying
