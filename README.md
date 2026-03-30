# Local Context Query

Context-aware AI chat — answers **only** from your uploaded documents.
Everything runs in Docker. Mobile-friendly UI with light/dark mode.

## Quick Start

```bash
cd local-context-query

# CPU (works everywhere):
docker compose up -d --build

# GPU (requires nvidia-docker):
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build

# Pull models (first time only):
docker compose exec ollama ollama pull phi4-mini
docker compose exec ollama ollama pull nomic-embed-text

# Open:
open http://localhost:8300
```

Wait ~60s for Temporal + Postgres to initialize.

## Architecture

```
Browser :8300 → Nginx → FastAPI :8200
                            │
              ┌─────────────┼──────────────────┐
              ▼             ▼                  ▼
        LocalStack S3    Temporal :7233     ChromaDB :8100
        :4566            (Postgres 17)
              │             │
              └───── Worker ────▸ Ollama :11434
```

Models are loaded from `~/.ollama/models` on your host (shared with local Ollama install).

## Services

| Service      | Port  | Image                         |
|-------------|-------|-------------------------------|
| Ollama      | 11434 | ollama/ollama:latest          |
| Frontend    | 8300  | nginx:1.27-alpine             |
| Backend     | 8200  | python:3.12-slim + FastAPI    |
| LocalStack  | 4566  | localstack/localstack:4.0     |
| Temporal    | 7233  | temporalio/auto-setup:1.25.2  |
| Temporal UI | 8080  | temporalio/ui:2.31.2          |
| Postgres 17 | 5432  | postgres:17-alpine            |
| ChromaDB    | 8100  | chromadb/chroma:0.6.3         |

## Features

- **Mobile-first responsive** — works on phones, tablets, desktop
- **Light/dark mode** — toggle in header, persists in localStorage, respects system preference
- **S3 storage** — documents stored in LocalStack S3 before processing
- **Temporal workflows** — reliable document processing with retries
- **Context-only answers** — "I do not know" when docs don't contain relevant info
- **Document filtering** — checkbox to include/exclude docs per query

## Tests

```bash
pip install -r tests/requirements.txt
python -m pytest tests/test_backend.py -v
```

## Stopping

```bash
docker compose down        # Stop
docker compose down -v     # Stop + delete data (models kept)
```
