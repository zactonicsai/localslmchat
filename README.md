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
open http://localhost:6977
```

Wait ~60s for Temporal + Postgres to initialize.

## Architecture

```
Browser :6977 → Nginx → FastAPI :6976
                            │
              ┌─────────────┼──────────────────┐
              ▼             ▼                  ▼
        LocalStack S3    Temporal :6973     ChromaDB :6975
        :6971            (Postgres 17)
              │             │
              └───── Worker ────▸ Ollama :6970
```

Models are loaded from `~/.ollama/models` on your host (shared with local Ollama install).

## Services

| Service      | Port  | Image                         |
|-------------|-------|-------------------------------|
| Ollama      | 6970  | ollama/ollama:latest          |
| LocalStack  | 6971  | localstack/localstack:4.0     |
| Postgres 17 | 6972  | postgres:17-alpine            |
| Temporal    | 6973  | temporalio/auto-setup:1.25.2  |
| Temporal UI | 6974  | temporalio/ui:2.31.2          |
| ChromaDB    | 6975  | chromadb/chroma:0.6.3         |
| Backend     | 6976  | python:3.12-slim + FastAPI    |
| Frontend    | 6977  | nginx:1.27-alpine             |

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

# Temporal RAG Activities — A Line-by-Line Tutorial

> A complete walkthrough of `activities.py`, the Temporal worker module that powers a Retrieval-Augmented Generation (RAG) pipeline using **S3** for storage, **Ollama** for embeddings and LLM inference, and **ChromaDB** for vector search.

---

## Table of Contents

1. [What This Module Does (Big Picture)](#1-what-this-module-does-big-picture)
2. [Architecture & Data Flow](#2-architecture--data-flow)
3. [Module Docstring & Design Philosophy](#3-module-docstring--design-philosophy)
4. [Imports — What Each One Is For](#4-imports--what-each-one-is-for)
5. [Configuration Block](#5-configuration-block)
6. [Infrastructure Wrappers](#6-infrastructure-wrappers)
   - [6.1 `S3Client`](#61-s3client)
   - [6.2 `ChromaStore`](#62-chromastore)
   - [6.3 `OllamaClient` — The Heart of the Module](#63-ollamaclient--the-heart-of-the-module)
7. [The Document Upload Pipeline (`DocumentActivities`)](#7-the-document-upload-pipeline-documentactivities)
8. [The Query Pipeline (`QueryActivities`)](#8-the-query-pipeline-queryactivities)
9. [Module-Level Wiring](#9-module-level-wiring)
10. [How Ollama Is Used (Deep Dive)](#10-how-ollama-is-used-deep-dive)
11. [How ChromaDB Is Used (Deep Dive)](#11-how-chromadb-is-used-deep-dive)
12. [Putting It All Together: Two End-to-End Walkthroughs](#12-putting-it-all-together-two-end-to-end-walkthroughs)

---

## 1. What This Module Does (Big Picture)

This file defines **Temporal activities** — discrete, retryable units of work that a Temporal workflow orchestrator can call. There are two pipelines:

| Pipeline | Activities | Purpose |
|----------|------------|---------|
| **Upload** | `extract_text` → `chunk_text` → `embed_and_store` | Take a file (PDF/DOCX/XLSX/TXT) from S3, pull out its text, split it into overlapping pieces, turn each piece into a vector via Ollama, and save those vectors to ChromaDB. |
| **Query** | `execute_query` | Take a user's question, vectorize it, find the most similar chunks in ChromaDB, and ask Ollama's LLM to answer using those chunks as context. |

In one sentence: **this is the engine of a "chat with your documents" system.**

---

## 2. Architecture & Data Flow

```
┌──────────┐    ┌───────────────────────────────────────────────┐    ┌──────────┐
│  USER    │───▶│              Temporal Workflow                │───▶│  USER    │
│ (upload) │    │                                               │    │ (answer) │
└──────────┘    │  ┌─────────────┐  ┌──────────┐  ┌──────────┐ │    └──────────┘
                │  │ extract_text│─▶│chunk_text│─▶│embed_and_│ │
                │  └─────────────┘  └──────────┘  │  store   │ │
                │         │              │        └──────────┘ │
                │         ▼              ▼              │      │
                │     ┌─────┐        ┌─────┐            ▼      │
                │     │ S3  │        │ S3  │       ┌────────┐  │
                │     │ raw │        │text │       │ Chroma │  │
                │     └─────┘        └─────┘       └────────┘  │
                │                                       ▲      │
                │  ┌──────────────────────────┐         │      │
                │  │     execute_query        │─────────┘      │
                │  │ (embed→retrieve→generate)│                │
                │  └──────────────────────────┘                │
                │              │                                │
                │              ▼                                │
                │         ┌─────────┐                           │
                │         │ Ollama  │ (embed + generate)        │
                │         └─────────┘                           │
                └───────────────────────────────────────────────┘
```

**Three external services are involved:**

- **S3** (or MinIO): persistent blob storage for raw uploads, extracted text, and final answers.
- **Ollama**: a local LLM runtime that does two jobs — turn text into vectors (`/api/embed`), and generate natural-language answers (`/api/generate`).
- **ChromaDB**: a vector database that stores those embeddings and finds the nearest neighbors to a query vector.

---

## 3. Module Docstring & Design Philosophy

## Temporal activities for upload and query processing.

Best practices applied:
- Class-based activities with dependency injection for testability
- Single input/output dataclasses (already present via shared.models)
- Heartbeating on long-running activities
- Non-retryable ApplicationError for permanent failures (404, validation)
- Retryable errors left as plain exceptions for Temporal retry policy
- Explicit timeout-friendly httpx clients (not created per-call)
- Ollama /api/embed fix: handle both string and list input, with
  fallback to legacy /api/embeddings endpoint for older Ollama versions


The docstring tells you the **five key engineering decisions** baked into this module. Read it like a cheat sheet:

1. **Class-based + DI** — activities live on classes whose constructors accept their dependencies. In tests you swap in fakes; in production you get the real S3/Chroma/Ollama clients.
2. **Dataclass I/O** — activities take and return one struct each (e.g. `UploadInput`, `ExtractResult`). Temporal serializes these across the worker boundary.
3. **Heartbeats** — long-running activities call `activity.heartbeat(...)` so Temporal knows they're alive and not hung.
4. **Two-tier error handling**:
   - `ApplicationError(non_retryable=True)` for things retrying won't fix (404, empty file, schema mismatch).
   - Plain `Exception` / `ValueError` for transient failures — Temporal will retry them according to the workflow's retry policy.
5. **The Ollama embed bug** — older Ollama versions choked on certain payload shapes. This module sends a string (not a list), and falls back to the older `/api/embeddings` endpoint if `/api/embed` doesn't exist.

---

## 4. Imports — What Each One Is For

```python
import os, json, tempfile, re
from datetime import datetime, timezone
from typing import List, Optional

import boto3              # S3 client
import httpx              # async HTTP client for Ollama
import chromadb           # vector database client
from botocore.exceptions import ClientError
from temporalio import activity
from temporalio.exceptions import ApplicationError

from shared.models import (
    UploadInput, ExtractResult, ChunkResult, EmbedStoreResult,
    QueryInput, QueryResult, S3_TEXT_PREFIX, S3_ANSWER_PREFIX
)
```

| Import | Why |
|--------|-----|
| `os`, `json`, `tempfile`, `re` | env vars, JSON serialization, temp files for binary parsers, regex for stripping `<think>` blocks |
| `datetime, timezone` | UTC-aware timestamps in metadata |
| `boto3` + `ClientError` | S3 access (uploads, downloads) |
| `httpx` | async HTTP client for talking to Ollama |
| `chromadb` | vector DB client |
| `temporalio.activity` | the `@activity.defn` decorator + `activity.logger` + `activity.heartbeat` |
| `ApplicationError` | how you signal *non-retryable* failure to Temporal |
| `shared.models` | dataclasses defining activity input/output schemas — kept in a shared module so both workflow code and worker code see the same definitions |

---

## 5. Configuration Block

```python
S3_ENDPOINT      = os.getenv("S3_ENDPOINT", "http://host.docker.internal:6971")
S3_BUCKET        = os.getenv("S3_BUCKET", "lcq-documents")
CHROMA_HOST      = os.getenv("CHROMA_HOST", "host.docker.internal")
CHROMA_PORT      = int(os.getenv("CHROMA_PORT", "6975"))
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:6970")
EMBED_MODEL      = os.getenv("EMBED_MODEL", "nomic-embed-text")
BACKEND_URL      = os.getenv("BACKEND_URL", "http://host.docker.internal:6976")
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP", "100"))
EMBED_MAX_CHARS  = int(os.getenv("EMBED_MAX_CHARS", "6000"))
```

Every value is overridable via environment variables. The defaults assume Docker Desktop with `host.docker.internal` resolving to the host machine.

**The interesting numbers:**

- **`CHUNK_SIZE = 500` words** with **`CHUNK_OVERLAP = 100` words** — the comment in the source explains why: nomic-embed-text has a 2048-token context, ~1.3 tokens/word means 500 words ≈ 650 tokens. The previous default of 800 words sometimes blew past the limit.
- **`EMBED_MAX_CHARS = 6000`** — a *belt-and-braces* safety net. Even if word-count is fine, a chunk full of base64 or hex strings can have very long "words" that explode the token count. 6000 chars ≈ 1500 tokens, well under 2048.

> **Tutorial point:** when you embed real-world text, you'll hit "context length exceeded" eventually. This module defends against it three ways: word-based chunking, character-level pre-truncation, and `truncate: true` in the Ollama payload (server-side last resort).

---

## 6. Infrastructure Wrappers

These three classes exist purely so the activity code can be unit-tested without spinning up real infrastructure. Each is a **thin pass-through** with a stable interface.

### 6.1 `S3Client`

```python
class S3Client:
    def __init__(self, endpoint_url=S3_ENDPOINT, bucket=S3_BUCKET):
        self._bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        )

    def get_bytes(self, key: str) -> bytes: ...
    def put_bytes(self, key: str, data: bytes) -> None: ...
```

- Uses `endpoint_url` so it works against MinIO, LocalStack, or real AWS.
- Exposes only two methods — `get_bytes` and `put_bytes` — because that's all the rest of the module needs.
- The `"test"/"test"` defaults make local development with MinIO frictionless.

### 6.2 `ChromaStore`

```python
class ChromaStore:
    def __init__(self, host=CHROMA_HOST, port=CHROMA_PORT):
        client = chromadb.HttpClient(host=host, port=port)
        client.heartbeat()                      # fail fast if Chroma is down
        self._collection = client.get_or_create_collection(
            name="local_context",
            metadata={"hnsw:space": "cosine"},  # cosine similarity for embeddings
        )
```

Three things happen in the constructor:

1. **HTTP client** to a Chroma server (could be `chromadb` running in Docker).
2. **`heartbeat()`** is called immediately — if Chroma is down, you find out at worker startup, not on the first request.
3. **One collection**, named `"local_context"`, configured for **cosine similarity** (the right metric for sentence/document embeddings; angle-based, magnitude-invariant).

`get_or_create_collection` is idempotent — first run creates it, subsequent runs reuse it.

### 6.3 `OllamaClient` — The Heart of the Module

This class has the most logic because Ollama's HTTP API has two awkward edge cases:

1. The `/api/embed` endpoint expects `"input"` (string or list); the older `/api/embeddings` endpoint expects `"prompt"` (string). The class supports **both**, in that order.
2. Inputs can exceed the model's token context, returning a 400 — so we truncate **before** sending and also pass `truncate: true` to let Ollama do server-side truncation as backup.

```python
class OllamaClient:
    def __init__(self, base_url=OLLAMA_BASE_URL, embed_model=EMBED_MODEL):
        self._base_url = base_url.rstrip("/")
        self._embed_model = embed_model
```

#### The `embed()` method, step by step

```python
async def embed(self, text: str) -> List[float]:
    clean = text.strip()
    if not clean:
        raise ApplicationError("Cannot embed an empty string", non_retryable=True)
```
Empty input is a permanent error — retrying won't help, so we mark it non-retryable.

```python
    if len(clean) > EMBED_MAX_CHARS:
        activity.logger.info(f"Truncating embed input from {len(clean)} to {EMBED_MAX_CHARS} chars")
        clean = clean[:EMBED_MAX_CHARS].rsplit(" ", 1)[0]
```
If too long: chop to `EMBED_MAX_CHARS`, then back up to the last space so we don't slice mid-word.

```python
    async with httpx.AsyncClient(timeout=90.0) as client:
        embed_error = await self._try_embed(client, clean)
        if isinstance(embed_error, list):
            return embed_error  # success – got the vector

        legacy_result = await self._try_legacy_embeddings(client, clean)
        if isinstance(legacy_result, list):
            return legacy_result  # success via legacy

        raise embed_error
```

This is a **two-tier fallback**. Each helper either returns a `list[float]` (success, the vector) or returns an `Exception` (failure). If both fail, we raise the *original* error from `/api/embed` — because that's almost always the more informative one.

#### `_try_embed` — the modern endpoint

```python
payload = {"model": self._embed_model, "input": text, "truncate": True}
r = await client.post(f"{self._base_url}/api/embed", json=payload)

if r.status_code == 404:
    raise ApplicationError(
        f"Embedding model '{self._embed_model}' not found on Ollama. "
        f"Pull it with: ollama pull {self._embed_model}",
        non_retryable=True,
    )
```

**Notice the 404 handling**: a 404 means *the model isn't pulled*. Retrying won't fix that — the human has to run `ollama pull`. So we raise a non-retryable error with a friendly hint.

```python
data = r.json()
embeddings = data.get("embeddings")
if embeddings and len(embeddings) > 0:
    return embeddings[0]
```

`/api/embed` returns `{"embeddings": [[0.1, 0.2, ...]]}` — a list of vectors, even for single input. We grab the first one.

#### `_try_legacy_embeddings` — the older endpoint

Same idea, different shape: payload uses `"prompt"`, response uses `"embedding"` (singular):

```python
payload = {"model": self._embed_model, "prompt": text, "truncate": True}
# response: {"embedding": [0.1, 0.2, ...]}
```

#### `generate()` — LLM completion for answers

```python
async def generate(self, model: str, prompt: str, system: str) -> str:
    async with httpx.AsyncClient(timeout=180.0) as client:
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": 0.3, "num_ctx": 4096},
        }
        r = await client.post(f"{self._base_url}/api/generate", json=payload)
        ...
        answer = r.json().get("response", "")
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
        return answer
```

Three things to note:

- **Lower temperature (0.3)** — for RAG, you want grounded, deterministic answers, not creative writing.
- **`num_ctx: 4096`** — explicitly sized context window so context + question + answer all fit.
- **`<think>` stripping** — some "reasoning" models (Qwen, DeepSeek, etc.) emit chain-of-thought wrapped in `<think>...</think>` tags. We hide that from the end user.

---

## 7. The Document Upload Pipeline (`DocumentActivities`)

```python
class DocumentActivities:
    def __init__(self, s3=None, chroma=None, ollama=None):
        self._s3 = s3 or S3Client()
        self._chroma = chroma or ChromaStore()
        self._ollama = ollama or OllamaClient()
```

The `or` pattern is the **dependency injection seam**: pass real clients (or none, for production defaults), or pass mocks for tests.

### Activity 1 — `extract_text`

```python
@activity.defn(name="extract_text_activity")
async def extract_text(self, inp: UploadInput) -> ExtractResult:
    try:
        raw = self._s3.get_bytes(inp.s3_raw_key)
    except ClientError as e:
        raise ApplicationError(f"Failed to fetch raw file from S3: {e}", non_retryable=True)
```

The `name=` argument is the **activity type name** Temporal sees on the wire. Keeping it equal to the original function name preserves backward compatibility with workflows that already reference it.

```python
    ext = inp.filename.rsplit(".", 1)[-1].lower() if "." in inp.filename else ""
    text = self._extract_by_extension(raw, ext, inp.filename)

    if not text.strip():
        raise ApplicationError(f"No text extracted from {inp.filename}", non_retryable=True)

    key = f"{S3_TEXT_PREFIX}{inp.doc_id}.txt"
    self._s3.put_bytes(key, text.encode())
    return ExtractResult(s3_text_key=key, characters=len(text))
```

- Dispatches to a parser based on file extension.
- An empty extraction is a permanent failure — there's nothing to retry.
- Result is written to S3 under `text/{doc_id}.txt` and the key is returned to the workflow.

### `_extract_by_extension` — the parser switchboard

```python
@staticmethod
def _extract_by_extension(raw: bytes, ext: str, filename: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        if ext == "pdf":
            import pdfplumber
            with pdfplumber.open(tmp_path) as pdf:
                return "\n\n".join(p.extract_text() or "" for p in pdf.pages)

        if ext in ("doc", "docx"):
            from docx import Document
            return "\n\n".join(p.text for p in Document(tmp_path).paragraphs if p.text.strip())

        if ext in ("xlsx", "xls"):
            import openpyxl
            wb = openpyxl.load_workbook(tmp_path, read_only=True, data_only=True)
            lines = []
            for ws in wb.worksheets:
                for row in ws.iter_rows(values_only=True):
                    lines.append(" | ".join(str(c) if c is not None else "" for c in row))
            wb.close()
            return "\n".join(lines)

        return raw.decode("utf-8", errors="replace")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
```

**Why a temp file?** The PDF/DOCX/XLSX libraries all want a file path, not bytes. We write the raw bytes to a temp file, parse, then delete in `finally`.

**Imports inside the function** are intentional: pdfplumber, python-docx, and openpyxl are heavy and not always needed. Lazy-loading speeds up worker startup.

The fallback (`raw.decode("utf-8", errors="replace")`) handles plain text and any unrecognized format gracefully.

### Activity 2 — `chunk_text`

```python
@activity.defn(name="chunk_text_activity")
async def chunk_text(self, s3_text_key: str) -> ChunkResult:
    text = self._s3.get_bytes(s3_text_key).decode("utf-8")
    words = text.split()
    if not words:
        return ChunkResult(chunks=[], count=0)

    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + CHUNK_SIZE]))
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return ChunkResult(chunks=chunks, count=len(chunks))
```

This is **classic sliding-window chunking**:

```
words: [w0, w1, w2, ..., w499, w500, ...]
chunk 0: words[0:500]                ← stride starts at 0
chunk 1: words[400:900]              ← stride += (500-100) = 400
chunk 2: words[800:1300]
...
```

The 100-word overlap is critical: if a sentence happens to fall at a chunk boundary, the next chunk still contains the surrounding context, so retrieval doesn't miss it.

### Activity 3 — `embed_and_store`

```python
@activity.defn(name="embed_and_store_activity")
async def embed_and_store(self, doc_id: str, filename: str, chunks: List[str]) -> EmbedStoreResult:
    ids, embeddings, metadatas, documents = [], [], [], []
    now = datetime.now(timezone.utc).isoformat()

    for i, chunk in enumerate(chunks):
        activity.heartbeat(f"Embedding chunk {i + 1}/{len(chunks)}")
        vector = await self._ollama.embed(chunk)

        ids.append(f"{doc_id}_chunk_{i}")
        embeddings.append(vector)
        metadatas.append({
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "uploaded_at": now,
        })
        documents.append(chunk)

    self._chroma.collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    )
    return EmbedStoreResult(chunks_stored=len(ids))
```

This is **the most expensive activity** in the module — one HTTP call to Ollama per chunk. A 100-page PDF could mean dozens of HTTP round-trips.

Three patterns to learn from:

1. **Heartbeat in the loop.** Temporal needs to hear from the activity periodically; otherwise it assumes the worker died and reschedules the work. The heartbeat message includes progress info, which shows up in the Temporal UI.
2. **Batch the Chroma write.** We don't `add` after each embed — we accumulate all four parallel arrays (`ids`, `embeddings`, `metadatas`, `documents`) and call `add` once. ChromaDB indexes in bulk far more efficiently.
3. **Retryable Chroma errors propagate.** Note the bare `raise` after logging. If Chroma is having a transient issue, Temporal will retry the whole activity. (The embeds are idempotent because chunk IDs are deterministic — `{doc_id}_chunk_{i}`.)

---

## 8. The Query Pipeline (`QueryActivities`)

```python
class QueryActivities:
    def __init__(self, s3=None, chroma=None, ollama=None, backend_url=BACKEND_URL):
        self._s3 = s3 or S3Client()
        self._chroma = chroma or ChromaStore()
        self._ollama = ollama or OllamaClient()
        self._backend_url = backend_url
```

Same DI pattern. Plus a `backend_url` for an optional notification webhook.

### `execute_query` — the main RAG activity

```python
@activity.defn(name="execute_query_activity")
async def execute_query(self, inp: QueryInput) -> QueryResult:
    # 1. Load query from S3
    q_data = json.loads(self._s3.get_bytes(inp.s3_query_key))
    query_text = q_data["query"]

    # 2. Retrieve relevant chunks
    coll = self._chroma.collection
    if coll.count() == 0:
        answer_obj = self._no_docs_answer(inp.query_id)
    else:
        answer_obj = await self._rag_answer(inp, query_text, coll)

    # 3. Persist answer to S3
    s3_answer_key = f"{S3_ANSWER_PREFIX}{inp.query_id}.json"
    self._s3.put_bytes(s3_answer_key, json.dumps(answer_obj).encode())

    # 4. Best-effort backend notification (fire-and-forget)
    await self._notify_backend(inp.query_id)

    return QueryResult(query_id=inp.query_id, status="completed", s3_answer_key=s3_answer_key)
```

The query text is read from S3 instead of passed as an argument — this keeps the workflow's history compact (Temporal stores every input/output forever).

### `_rag_answer` — the actual RAG loop

```python
async def _rag_answer(self, inp, query_text, coll) -> dict:
    activity.heartbeat("Embedding query")
    query_vector = await self._ollama.embed(query_text)

    n = min(8, coll.count())
    results = coll.query(
        query_embeddings=[query_vector],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    filtered_docs, sources = self._filter_results(results, inp.enabled_doc_ids)

    if not filtered_docs:
        return self._no_results_answer(inp.query_id)

    activity.heartbeat("Generating answer")
    context = "\n\n---\n\n".join(filtered_docs)
    if len(context) > 12_000:
        context = context[:12_000] + "\n\n[Context truncated]"

    system_prompt = (
        "You are Local Context Query. Answer ONLY from provided context. "
        "If context lacks info say 'I do not know.' Cite document names. Be concise."
    )
    user_prompt = (
        f"CONTEXT:\n{context}\n\nQUESTION:\n{query_text}\n\nAnswer from context only:"
    )
    answer_text = await self._ollama.generate(inp.model, user_prompt, system_prompt)
    if len(answer_text) > 50_000:
        answer_text = answer_text[:50_000] + "\n\n[Answer truncated]"

    return {"query_id": inp.query_id, "answer": answer_text, "sources": sources}
```

**Annotated flow:**

| Step | What happens |
|------|--------------|
| 1. Embed the question | One Ollama `/api/embed` call → 768-dim (or similar) float vector. |
| 2. Top-k retrieval | Ask Chroma for the 8 nearest chunks. `n_results = min(8, count)` so we don't ask for 8 when only 3 exist. |
| 3. Filter | If the user enabled only certain documents, drop any chunk whose `doc_id` isn't in their allow-list. |
| 4. Build context | Join chunks with `\n\n---\n\n` separators. Cap at 12,000 chars to leave room for the question and answer in the LLM's context window. |
| 5. Generate | Call Ollama `/api/generate` with a system prompt that constrains the model to answer only from the context. |
| 6. Cap answer | Truncate at 50,000 chars in case a runaway model floods the response. |

The system prompt is the single most important line for RAG quality:

> *"Answer ONLY from provided context. If context lacks info say 'I do not know.'"*

Without that, the model happily hallucinates from its training data. With it, you get either a grounded answer or a clean "I don't know" — both are useful.

### `_filter_results` — post-retrieval filtering

```python
@staticmethod
def _filter_results(results: dict, enabled_doc_ids: Optional[List[str]]) -> tuple:
    docs, sources = [], []
    if not results["documents"] or not results["documents"][0]:
        return docs, sources

    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]

        if enabled_doc_ids and meta["doc_id"] not in enabled_doc_ids:
            continue

        docs.append(doc)
        sources.append({
            "filename": meta["filename"],
            "doc_id": meta["doc_id"],
            "chunk_index": meta["chunk_index"],
            "distance": round(dist, 4),
        })
    return docs, sources
```

ChromaDB's `query` returns nested lists (one outer list per query vector — we only ever pass one). The `[0]` indexing is unwrapping that.

The `distance` field is preserved in the returned `sources` payload so the UI can display *how relevant* each cited chunk was.

### `_notify_backend` — fire-and-forget

```python
async def _notify_backend(self, query_id: str) -> None:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                f"{self._backend_url}/api/internal/query-complete",
                json={"query_id": query_id},
            )
    except Exception:
        activity.logger.debug(f"Backend notification failed for {query_id} (non-critical)")
```

A bare `except Exception` is appropriate here because the notification is **best-effort**. The answer is already saved to S3; if the backend's webhook fails, the worst case is the UI takes a few seconds longer to learn the answer is ready (presumably via polling).

---

## 9. Module-Level Wiring

```python
_s3 = S3Client()
_chroma = ChromaStore()
_ollama = OllamaClient()

_doc_activities = DocumentActivities(s3=_s3, chroma=_chroma, ollama=_ollama)
_query_activities = QueryActivities(s3=_s3, chroma=_chroma, ollama=_ollama)

extract_text_activity   = _doc_activities.extract_text
chunk_text_activity     = _doc_activities.chunk_text
embed_and_store_activity = _doc_activities.embed_and_store
execute_query_activity  = _query_activities.execute_query
```

This is the **adapter layer** between the class-based design and Temporal's function-style registration API.

- The infrastructure clients are created **once** at module import time.
- Both activity classes share the same client instances (no duplicate connections).
- The bound methods are exposed under their original names so existing code (`from activities import extract_text_activity`) keeps working.

When the Temporal worker starts, it registers these four names. Workflows reference them by their `name=...` attribute (`extract_text_activity`, etc.).

---

## 10. How Ollama Is Used (Deep Dive)

Ollama is used for **two completely different jobs**, exposed via two different HTTP endpoints.

### Job A — Embeddings (`POST /api/embed`)

**Purpose:** turn a piece of text into a fixed-length vector that captures its semantic meaning.

**Where called:**
- `embed_and_store` — once per chunk during upload.
- `_rag_answer` — once per query.

**Request shape (modern API):**
```json
{
  "model": "nomic-embed-text",
  "input": "Sliding-window chunking with 100-word overlap...",
  "truncate": true
}
```

**Response shape:**
```json
{ "embeddings": [[0.012, -0.034, 0.567, ...]] }
```

**Fallback (legacy `/api/embeddings`):**
```json
// request
{ "model": "nomic-embed-text", "prompt": "...", "truncate": true }
// response
{ "embedding": [0.012, -0.034, ...] }
```

**Failure modes the code handles:**

| Status | Treatment |
|--------|-----------|
| 200 + non-empty vector | Success |
| 200 + empty | `ValueError` → fallback path |
| 404 | `ApplicationError(non_retryable=True)` — the model isn't pulled |
| Other non-200 | `ValueError` → fallback path → eventual retry |
| `httpx.RequestError` (network) | Captured & passed to fallback |

### Job B — Generation (`POST /api/generate`)

**Purpose:** given retrieved context + a question, produce a natural-language answer.

**Where called:** `_rag_answer`, exactly once per query.

**Request shape:**
```json
{
  "model": "llama3.1:8b",
  "system": "You are Local Context Query. Answer ONLY from provided context...",
  "prompt": "CONTEXT:\n...\n\nQUESTION:\n...\n\nAnswer from context only:",
  "stream": false,
  "options": { "temperature": 0.3, "num_ctx": 4096 }
}
```

**Response shape:**
```json
{ "response": "The document states...", "done": true, ... }
```

The model name is **not hard-coded** — it comes from `inp.model` on `QueryInput`, so different queries can route to different models (e.g. a fast model for chat, a slow careful model for analysis).

### Why these defenses exist

The `OllamaClient.embed` method has more error handling than seems strictly necessary because **token-limit errors are silent traps**:

- A user uploads a PDF with one page that's nothing but base64 image data.
- That page extracts as a single 50,000-character "word" (no spaces).
- The 500-word chunker happily packages it as a chunk.
- Ollama's `/api/embed` returns 400: "input length exceeds context length."
- The whole upload fails.

The three layers of protection (word chunking + char truncation + `truncate: true`) ensure that even pathological input gets embedded somehow, even if the resulting vector is for a truncated version.

---

## 11. How ChromaDB Is Used (Deep Dive)

ChromaDB is the system's **memory**. Without it, the LLM has no way to know what's in your documents.

### Setup (once per worker process)

```python
client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
client.heartbeat()
collection = client.get_or_create_collection(
    name="local_context",
    metadata={"hnsw:space": "cosine"},
)
```

- **HttpClient** = remote Chroma server (in-process via `chromadb.Client()` is also possible; the HTTP variant lets the vector DB be a separate, scalable container).
- **`hnsw:space="cosine"`** = the index uses Hierarchical Navigable Small Worlds with cosine distance. Cosine is the standard for normalized embedding vectors.

### Write Path — `collection.add(...)`

```python
collection.add(
    ids=["doc_42_chunk_0", "doc_42_chunk_1", ...],     # globally unique
    embeddings=[[0.1, ...], [0.2, ...], ...],          # parallel array of vectors
    metadatas=[                                        # parallel array of dicts
        {"doc_id": "doc_42", "filename": "manual.pdf",
         "chunk_index": 0, "total_chunks": 7,
         "uploaded_at": "2026-04-29T10:00:00Z"},
        ...
    ],
    documents=["original chunk text...", ...],         # parallel array of strings
)
```

All four lists must be the **same length** — they're in lockstep, indexed positionally.

The metadata is what enables filtering and citation:
- **`doc_id`** lets the query path scope retrieval to enabled documents.
- **`filename`** is what the user sees in the source attribution UI.
- **`chunk_index` / `total_chunks`** lets the UI say "chunk 3 of 7."
- **`uploaded_at`** lets you sort by recency or implement TTL pruning later.

### Read Path — `collection.query(...)`

```python
results = collection.query(
    query_embeddings=[query_vector],                       # batch of 1
    n_results=n,
    include=["documents", "metadatas", "distances"],
)
```

The response shape is **batched** because Chroma supports querying many vectors at once:

```python
{
  "ids":        [["doc_42_chunk_3", "doc_19_chunk_0", ...]],  # one inner list per query
  "documents":  [["chunk 3 text", "chunk 0 text", ...]],
  "metadatas":  [[{...}, {...}, ...]],
  "distances":  [[0.1234, 0.1567, ...]],                      # smaller = more similar
}
```

That's why the filter code does `results["documents"][0]` — unwrapping the single-query batch.

### Why `n_results = min(8, coll.count())`

Asking Chroma for more results than exist returns garbage in some configurations. The `min` is a defensive guard for the early state of the system when only a few chunks have been indexed.

---

## 12. Putting It All Together: Two End-to-End Walkthroughs

### Walkthrough A — A user uploads `manual.pdf` (12 pages)

1. **Frontend** uploads the PDF to S3 at `raw/abc-123.pdf` and starts a Temporal upload workflow with `UploadInput(doc_id="abc-123", filename="manual.pdf", s3_raw_key="raw/abc-123.pdf")`.
2. **`extract_text_activity`** runs:
   - `S3Client.get_bytes("raw/abc-123.pdf")` → raw PDF bytes.
   - Writes to a temp file, opens with `pdfplumber`, extracts ~30,000 chars.
   - Saves plain text to `text/abc-123.txt` in S3.
   - Returns `ExtractResult(s3_text_key="text/abc-123.txt", characters=30000)`.
3. **`chunk_text_activity`** runs:
   - Reads the text back from S3.
   - Splits into ~5,000 words.
   - Produces ~13 chunks (5000 / (500-100) stride ≈ 13).
   - Returns `ChunkResult(chunks=[...], count=13)`.
4. **`embed_and_store_activity`** runs:
   - For each of the 13 chunks: heartbeat + `OllamaClient.embed(chunk)` → 768-d vector.
   - One batched `collection.add(...)` call with 13 IDs, 13 vectors, 13 metadata dicts, 13 documents.
   - Returns `EmbedStoreResult(chunks_stored=13)`.
5. **Workflow completes.** The user sees "manual.pdf — 13 chunks indexed."

### Walkthrough B — The user asks: *"What's the warranty period?"*

1. **Frontend** writes the question to S3 at `query/q-789.json` and starts a query workflow with `QueryInput(query_id="q-789", s3_query_key="query/q-789.json", model="llama3.1:8b", enabled_doc_ids=["abc-123"])`.
2. **`execute_query_activity`** runs:
   - Reads `query/q-789.json` from S3 → `query_text = "What's the warranty period?"`.
   - `coll.count()` returns 13 (from the upload above), so we go into `_rag_answer`.
   - **Heartbeat: "Embedding query"** — `OllamaClient.embed("What's the warranty period?")` → 768-d vector.
   - `coll.query(query_embeddings=[v], n_results=8)` → 8 nearest chunks across all docs.
   - `_filter_results` keeps only chunks where `meta["doc_id"] == "abc-123"` (all 8 in this case).
   - Builds context string (8 chunks joined with `\n\n---\n\n`), ~6,000 chars total.
   - **Heartbeat: "Generating answer"** — `OllamaClient.generate(model="llama3.1:8b", ...)`.
   - Ollama returns `"The warranty period is 24 months from date of purchase, per page 4 of manual.pdf."`.
   - Saves answer JSON to `answer/q-789.json` in S3.
   - Fires `_notify_backend("q-789")` (best-effort POST to webhook).
   - Returns `QueryResult(query_id="q-789", status="completed", s3_answer_key="answer/q-789.json")`.
3. **Frontend** receives the webhook (or polls), reads the answer JSON, and displays it with source citations.

---

## Closing notes

A few patterns in this file are worth taking with you to your own Temporal projects:

- **Wrap external services in thin classes** with stable interfaces. Tests stop being painful.
- **Use `ApplicationError(non_retryable=True)` for permanent failures.** Network blips deserve retries; missing models do not.
- **Heartbeat inside loops.** Otherwise Temporal will time out long activities.
- **Defend against token limits in three places.** Word chunking, char truncation, server-side `truncate: true`. Pick all three; pick a fight with none.
- **Embed once, retrieve fast.** The expensive work happens at upload; queries are cheap.
- **Separate worker concerns from workflow data.** S3 is the persistent state; Temporal history is just "which activities ran, in what order, and what they returned." Keep workflow inputs small.



# simple OpenWeb

docker run -d -p 3911:8080 -v ollama:/root/.ollama -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:ollama

docker run -d -p 3911:8080 --gpus all --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:cuda

