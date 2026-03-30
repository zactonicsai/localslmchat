# Local Context Query

A self-hosted, context-aware AI chat app that answers questions **only** from your uploaded documents. Uses **Temporal** for reliable document processing workflows, **ChromaDB** for vector search, and **Ollama** for local LLM inference.

## Architecture

```
┌──────────┐    ┌─────────┐    ┌──────────────┐    ┌──────────────┐
│ Browser  │───▸│  Nginx  │───▸│   FastAPI     │───▸│  Temporal    │
│ :8300    │    │  :8300  │    │   :8200       │    │  :7233       │
└──────────┘    └─────────┘    └──────┬────────┘    └──────┬───────┘
                                      │                     │
                                      ▼                     ▼
                               ┌──────────────┐    ┌──────────────┐
                               │  ChromaDB    │    │   Worker     │
                               │  :8100       │    │  (activities)│
                               └──────────────┘    └──────┬───────┘
                                                          │
                                      ┌───────────────────┼──────────┐
                                      ▼                   ▼          ▼
                               ┌────────────┐    ┌────────────┐  ┌────────┐
                               │ Postgres 17│    │  ChromaDB  │  │ Ollama │
                               │ (Temporal) │    │  (vectors) │  │ :11434 │
                               │ :5432      │    │  :8100     │  │ (host) │
                               └────────────┘    └────────────┘  └────────┘
```

## Upload Flow (Temporal Workflow)

```
POST /api/upload → save file → start Temporal workflow
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                   Activity 1:           Activity 2:
                   Extract Text          Chunk Text
                   (PDF/DOCX/TXT)        (800 words, 150 overlap)
                         │                     │
                         └──────────┬──────────┘
                                    ▼
                              Activity 3:
                              Embed + Store
                              (Ollama → ChromaDB)
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                   Frontend polls        ChromaDB has
                   GET /api/upload/      indexed chunks
                   {doc_id}/status       ready for queries
```

Each activity has independent retry policies and timeouts managed by Temporal.

## Prerequisites

1. **Docker & Docker Compose**
2. **Ollama** running on your host:
   ```bash
   ollama serve
   ollama pull phi4-mini
   ollama pull nomic-embed-text
   ```

## Quick Start

```bash
cd local-context-query
docker compose up -d --build
# Wait ~60s for Temporal + Postgres to initialize
open http://localhost:8300
```

## Running Tests

```bash
pip install -r tests/requirements.txt
python -m pytest tests/test_backend.py -v
```

**19 tests** (all mocked, no services needed) covering:
- Health / models endpoints
- Upload triggers Temporal workflow
- Status polling (processing → completed)
- Document listing from ChromaDB
- Query with context → answer + sources
- Query empty DB → "I do not know"
- Document filtering by checkbox
- Document deletion
- Error handling (always JSON)
- Full end-to-end flow

## Services & Ports

| Service       | Port  | Location | Purpose                        |
|---------------|-------|----------|--------------------------------|
| Frontend      | 8300  | Docker   | Nginx serving UI               |
| Backend API   | 8200  | Docker   | FastAPI                        |
| Temporal      | 7233  | Docker   | Workflow orchestration          |
| Temporal UI   | 8080  | Docker   | Workflow monitoring dashboard   |
| ChromaDB      | 8100  | Docker   | Vector store                   |
| Postgres 17   | 5432  | Docker   | Temporal persistence           |
| Ollama        | 11434 | Host     | LLM inference + embeddings     |

## Usage

1. **Select model** from dropdown (auto-populated from Ollama)
2. **Upload documents** — sidebar right; processing runs as a Temporal workflow
3. **Watch progress** — "Processing" section shows active workflows
4. **Ask questions** — answers come only from your document context
5. **Manage docs** — checkbox to enable/disable per query, trash to delete

## Stopping

```bash
docker compose down        # Stop
docker compose down -v     # Stop + delete all data
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Cannot reach Ollama" | `ollama serve` on host |
| Embedding errors | `ollama pull nomic-embed-text` |
| Temporal not starting | Wait 60s; check `docker compose logs temporal` |
| Linux: Ollama unreachable | `OLLAMA_HOST=0.0.0.0 ollama serve` |
| Workflow stuck | Check Temporal UI at http://localhost:8080 |
# localslmchat
# localslmchat
