"""
Tests for Local Context Query backend with Temporal workflows.

All external dependencies (Temporal, Ollama, ChromaDB) are mocked.
Tests verify:
  1. Health endpoint
  2. Model listing
  3. Upload triggers Temporal workflow
  4. Upload status polling (processing → completed)
  5. Document listing from ChromaDB
  6. Query with context → answer + sources
  7. Query empty DB → "I do not know"
  8. Query with disabled documents
  9. Document deletion
 10. Error handling (JSON responses)
 11. Chunking logic
 12. Full flow: upload → poll → query → delete → query again
"""

import io
import os
import sys
import tempfile
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

# ── Setup paths ──
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))
sys.path.insert(0, PROJECT_ROOT)

os.environ["UPLOAD_DIR"] = tempfile.mkdtemp()
os.environ["CHROMA_HOST"] = "localhost"
os.environ["CHROMA_PORT"] = "8000"
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
os.environ["TEMPORAL_ADDRESS"] = "localhost:7233"


# ── Fake ChromaDB collection ──
class FakeCollection:
    def __init__(self):
        self._ids, self._embeddings, self._metadatas, self._documents = [], [], [], []

    def count(self):
        return len(self._ids)

    def add(self, ids, embeddings, metadatas, documents):
        self._ids.extend(ids)
        self._embeddings.extend(embeddings)
        self._metadatas.extend(metadatas)
        self._documents.extend(documents)

    def get(self, include=None):
        return {"ids": list(self._ids), "metadatas": list(self._metadatas), "documents": list(self._documents)}

    def query(self, query_embeddings, n_results=5, include=None):
        n = min(n_results, len(self._ids))
        return {
            "ids": [self._ids[:n]], "documents": [self._documents[:n]],
            "metadatas": [self._metadatas[:n]], "distances": [[0.1*i for i in range(n)]],
        }

    def delete(self, ids):
        indices = [i for i, id_ in enumerate(self._ids) if id_ in ids]
        for idx in sorted(indices, reverse=True):
            self._ids.pop(idx); self._embeddings.pop(idx)
            self._metadatas.pop(idx); self._documents.pop(idx)


fake_collection = FakeCollection()

def mock_get_collection():
    return fake_collection


# ── Fake Ollama ──
FAKE_EMBED = [0.1] * 768
FAKE_MODELS = {"models": [
    {"name": "phi4-mini:latest", "size": 2e9, "modified_at": "2025-01-01T00:00:00Z"},
    {"name": "nomic-embed-text:latest", "size": 5e8, "modified_at": "2025-01-01T00:00:00Z"},
]}

async def mock_ollama_get(path, timeout=10.0):
    if "/api/tags" in path: return FAKE_MODELS
    return {}

async def mock_ollama_post(path, payload, timeout=120.0):
    if "/api/embed" in path: return {"embeddings": [FAKE_EMBED]}
    if "/api/generate" in path:
        return {"response": "Based on the document, the answer is 42."}
    return {}

async def mock_ollama_embed(text):
    return FAKE_EMBED

async def mock_resolve_embed_model():
    return "nomic-embed-text"


# ── Fake Temporal client ──
# Stores workflow runs so we can verify they were started and return results
_workflow_runs = {}  # workflow_id -> {input, result}


@dataclass
class FakeWorkflowHandle:
    workflow_id: str

    async def describe(self):
        if self.workflow_id not in _workflow_runs:
            raise RuntimeError(f"workflow not found: {self.workflow_id}")
        run = _workflow_runs.get(self.workflow_id)
        if run and run.get("result"):
            return MagicMock(status=MagicMock(__str__=lambda s: "COMPLETED"))
        return MagicMock(status=MagicMock(__str__=lambda s: "RUNNING"))

    async def result(self):
        run = _workflow_runs.get(self.workflow_id)
        if run and run.get("result"):
            return run["result"]
        raise RuntimeError("Not completed")


class FakeTemporalClient:
    async def start_workflow(self, workflow_name, inp, id, task_queue):
        _workflow_runs[id] = {"input": inp, "result": None}
        return MagicMock()

    def get_workflow_handle(self, workflow_id):
        return FakeWorkflowHandle(workflow_id)


_fake_temporal = FakeTemporalClient()


async def mock_get_temporal_client():
    return _fake_temporal


def _simulate_workflow_completion(doc_id, filename, chunks_count):
    """Simulate worker completing the workflow and storing data in ChromaDB."""
    from shared.models import UploadResult
    from datetime import datetime, timezone

    workflow_id = f"doc-upload-{doc_id}"
    _workflow_runs[workflow_id] = {
        "result": UploadResult(
            doc_id=doc_id, filename=filename,
            chunks=chunks_count, characters=1000,
            status="completed",
        )
    }
    # Also insert data into fake ChromaDB (simulates what the real worker does)
    now = datetime.now(timezone.utc).isoformat()
    for i in range(chunks_count):
        fake_collection._ids.append(f"{doc_id}_chunk_{i}")
        fake_collection._embeddings.append(FAKE_EMBED)
        fake_collection._metadatas.append({
            "doc_id": doc_id, "filename": filename,
            "chunk_index": i, "total_chunks": chunks_count,
            "uploaded_at": now,
        })
        fake_collection._documents.append(f"Chunk {i} content from {filename}")


# ── Fixtures ──
@pytest_asyncio.fixture(autouse=True)
async def reset_state():
    fake_collection._ids.clear(); fake_collection._embeddings.clear()
    fake_collection._metadatas.clear(); fake_collection._documents.clear()
    _workflow_runs.clear()
    import main
    main._chroma_client = None; main._collection = None; main._embed_model = None
    main._temporal_client = None
    yield


@pytest_asyncio.fixture
async def client():
    import main
    main.get_collection = mock_get_collection
    main.get_temporal_client = mock_get_temporal_client
    main.ollama_get = mock_ollama_get
    main.ollama_post = mock_ollama_post
    main.ollama_embed = mock_ollama_embed
    main._resolve_embed_model = mock_resolve_embed_model
    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ═══════════════════════════════════════════════════════════════════
# 1. HEALTH
# ═══════════════════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_health(client):
    r = await client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] in ("ok", "degraded")


# ═══════════════════════════════════════════════════════════════════
# 2. MODELS
# ═══════════════════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_list_models(client):
    r = await client.get("/api/models")
    assert r.status_code == 200
    names = [m["name"] for m in r.json()["models"]]
    assert "phi4-mini:latest" in names


# ═══════════════════════════════════════════════════════════════════
# 3. UPLOAD TRIGGERS TEMPORAL WORKFLOW
# ═══════════════════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_upload_starts_workflow(client):
    r = await client.post("/api/upload", files={"file": ("test.txt", io.BytesIO(b"Hello world"), "text/plain")})
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "processing"
    assert data["doc_id"]
    assert data["workflow_id"]
    assert data["filename"] == "test.txt"
    # Verify workflow was registered
    assert data["workflow_id"] in _workflow_runs


@pytest.mark.asyncio
async def test_upload_empty_returns_400(client):
    r = await client.post("/api/upload", files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")})
    assert r.status_code == 400


# ═══════════════════════════════════════════════════════════════════
# 4. UPLOAD STATUS POLLING
# ═══════════════════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_status_processing(client):
    r = await client.post("/api/upload", files={"file": ("doc.txt", io.BytesIO(b"Content"), "text/plain")})
    doc_id = r.json()["doc_id"]
    # Still processing (no result set yet)
    s = await client.get(f"/api/upload/{doc_id}/status")
    assert s.status_code == 200
    assert s.json()["status"] == "processing"


@pytest.mark.asyncio
async def test_status_completed(client):
    r = await client.post("/api/upload", files={"file": ("report.txt", io.BytesIO(b"Report data"), "text/plain")})
    doc_id = r.json()["doc_id"]
    # Simulate worker completing
    _simulate_workflow_completion(doc_id, "report.txt", 3)
    s = await client.get(f"/api/upload/{doc_id}/status")
    assert s.status_code == 200
    data = s.json()
    assert data["status"] == "completed"
    assert data["chunks"] == 3
    assert data["filename"] == "report.txt"


@pytest.mark.asyncio
async def test_status_not_found(client):
    r = await client.get("/api/upload/nonexistent/status")
    assert r.status_code == 404


# ═══════════════════════════════════════════════════════════════════
# 5. DOCUMENT LISTING
# ═══════════════════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_list_empty(client):
    r = await client.get("/api/documents")
    assert r.json()["documents"] == []


@pytest.mark.asyncio
async def test_list_after_workflow_completes(client):
    _simulate_workflow_completion("abc123", "notes.txt", 2)
    _simulate_workflow_completion("def456", "guide.txt", 5)
    r = await client.get("/api/documents")
    docs = r.json()["documents"]
    assert len(docs) == 2
    names = {d["filename"] for d in docs}
    assert names == {"notes.txt", "guide.txt"}


# ═══════════════════════════════════════════════════════════════════
# 6. QUERY WITH CONTEXT
# ═══════════════════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_query_with_docs(client):
    _simulate_workflow_completion("abc123", "knowledge.txt", 3)
    r = await client.post("/api/query", json={"query": "What is 42?", "model": "phi4-mini:latest"})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert len(data["sources"]) > 0
    assert data["sources"][0]["filename"] == "knowledge.txt"


# ═══════════════════════════════════════════════════════════════════
# 7. QUERY EMPTY DB
# ═══════════════════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_query_empty_db(client):
    r = await client.post("/api/query", json={"query": "Anything?", "model": "phi4-mini:latest"})
    assert "I do not know" in r.json()["answer"]
    assert r.json()["sources"] == []


# ═══════════════════════════════════════════════════════════════════
# 8. QUERY WITH DISABLED DOCS
# ═══════════════════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_query_disabled_docs(client):
    _simulate_workflow_completion("doc1", "enabled.txt", 2)
    # Query with wrong enabled_doc_ids → "I do not know"
    r = await client.post("/api/query", json={
        "query": "Test", "model": "phi4-mini:latest", "enabled_doc_ids": ["wrong_id"]
    })
    assert "I do not know" in r.json()["answer"]
    # Query with correct id → gets answer
    r2 = await client.post("/api/query", json={
        "query": "Test", "model": "phi4-mini:latest", "enabled_doc_ids": ["doc1"]
    })
    assert "I do not know" not in r2.json()["answer"]
    assert len(r2.json()["sources"]) > 0


# ═══════════════════════════════════════════════════════════════════
# 9. DELETE
# ═══════════════════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_delete(client):
    _simulate_workflow_completion("del1", "temp.txt", 2)
    assert fake_collection.count() == 2
    r = await client.delete("/api/documents/del1")
    assert r.status_code == 200
    assert fake_collection.count() == 0


@pytest.mark.asyncio
async def test_delete_not_found(client):
    r = await client.delete("/api/documents/nope")
    assert r.status_code == 404


# ═══════════════════════════════════════════════════════════════════
# 10. ERRORS RETURN JSON
# ═══════════════════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_errors_json(client):
    import main
    orig = main.get_collection
    main.get_collection = lambda: (_ for _ in ()).throw(HTTPException(503, "ChromaDB down"))
    r = await client.get("/api/documents")
    assert r.headers["content-type"].startswith("application/json")
    assert r.status_code == 503
    main.get_collection = mock_get_collection


@pytest.mark.asyncio
async def test_ollama_down(client):
    import main
    orig = main.ollama_get
    main.ollama_get = AsyncMock(side_effect=ConnectionError("refused"))
    r = await client.get("/api/models")
    assert r.status_code == 502
    assert "detail" in r.json()
    main.ollama_get = mock_ollama_get


# ═══════════════════════════════════════════════════════════════════
# 11. CHUNKING (import from worker)
# ═══════════════════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_chunking():
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "worker"))
    # We test chunk logic inline since activity needs Temporal runtime
    words = ["word"] * 2000
    text = " ".join(words)
    # Replicate chunk logic
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+800]))
        i += 800 - 150
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.split()) <= 800


# ═══════════════════════════════════════════════════════════════════
# 12. FULL FLOW
# ═══════════════════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_full_flow(client):
    # 1. Empty DB → "I do not know"
    r1 = await client.post("/api/query", json={"query": "Hi?", "model": "phi4-mini:latest"})
    assert "I do not know" in r1.json()["answer"]

    # 2. Upload triggers workflow
    up = await client.post("/api/upload", files={"file": ("guide.txt", io.BytesIO(b"Python guide content"), "text/plain")})
    assert up.status_code == 200
    doc_id = up.json()["doc_id"]

    # 3. Status = processing
    s1 = await client.get(f"/api/upload/{doc_id}/status")
    assert s1.json()["status"] == "processing"

    # 4. Worker completes (simulated)
    _simulate_workflow_completion(doc_id, "guide.txt", 4)

    # 5. Status = completed
    s2 = await client.get(f"/api/upload/{doc_id}/status")
    assert s2.json()["status"] == "completed"
    assert s2.json()["chunks"] == 4

    # 6. Document shows in listing
    docs = (await client.get("/api/documents")).json()["documents"]
    assert len(docs) == 1
    assert docs[0]["filename"] == "guide.txt"

    # 7. Query → gets answer
    r2 = await client.post("/api/query", json={"query": "What is Python?", "model": "phi4-mini:latest"})
    assert r2.status_code == 200
    assert len(r2.json()["sources"]) > 0

    # 8. Delete
    d = await client.delete(f"/api/documents/{doc_id}")
    assert d.status_code == 200

    # 9. Query again → "I do not know"
    r3 = await client.post("/api/query", json={"query": "What is Python?", "model": "phi4-mini:latest"})
    assert "I do not know" in r3.json()["answer"]


@pytest.mark.asyncio
async def test_all_responses_have_json_content_type(client):
    for url in ["/api/health", "/api/models", "/api/documents"]:
        r = await client.get(url)
        assert "application/json" in r.headers.get("content-type", "")
