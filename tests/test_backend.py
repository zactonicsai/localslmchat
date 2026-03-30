"""
Tests for Local Context Query backend.

All external deps (S3/LocalStack, Temporal, ChromaDB, Ollama) are mocked.
Tests:
  1. Health
  2. Models
  3. Upload saves to S3 and starts Temporal workflow
  4. Status polling (processing → completed)
  5. Document listing from ChromaDB
  6. Query with context
  7. Query empty DB → "I do not know"
  8. Query with disabled docs
  9. Delete (ChromaDB + S3)
 10. Error handling (JSON responses)
 11. Full flow: upload → S3 → poll → query → delete → query
"""

import io
import os
import sys
import tempfile
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))
sys.path.insert(0, PROJECT_ROOT)

os.environ.update({
    "CHROMA_HOST": "localhost", "CHROMA_PORT": "8000",
    "OLLAMA_BASE_URL": "http://localhost:11434",
    "TEMPORAL_ADDRESS": "localhost:7233",
    "EMBED_MODEL": "nomic-embed-text",
    "S3_ENDPOINT": "http://localhost:4566",
    "S3_BUCKET": "lcq-documents",
    "AWS_ACCESS_KEY_ID": "test",
    "AWS_SECRET_ACCESS_KEY": "test",
    "AWS_DEFAULT_REGION": "us-east-1",
})


# ── Fake S3 ──
class FakeS3:
    def __init__(self):
        self._objects = {}

    def put_object(self, Bucket, Key, Body):
        self._objects[f"{Bucket}/{Key}"] = Body if isinstance(Body, bytes) else Body.encode()

    def get_object(self, Bucket, Key):
        k = f"{Bucket}/{Key}"
        if k not in self._objects:
            raise Exception(f"NoSuchKey: {Key}")
        return {"Body": io.BytesIO(self._objects[k])}

    def head_bucket(self, Bucket):
        return {}

    def list_objects_v2(self, Bucket, Prefix=""):
        contents = [{"Key": k.split("/", 1)[1]} for k in self._objects if k.startswith(f"{Bucket}/{Prefix}")]
        return {"Contents": contents}

    def delete_object(self, Bucket, Key):
        self._objects.pop(f"{Bucket}/{Key}", None)


fake_s3 = FakeS3()


# ── Fake ChromaDB ──
class FakeCollection:
    def __init__(self):
        self._ids, self._embeddings, self._metadatas, self._documents = [], [], [], []

    def count(self):
        return len(self._ids)

    def add(self, ids, embeddings, metadatas, documents):
        self._ids.extend(ids); self._embeddings.extend(embeddings)
        self._metadatas.extend(metadatas); self._documents.extend(documents)

    def get(self, include=None):
        return {"ids": list(self._ids), "metadatas": list(self._metadatas), "documents": list(self._documents)}

    def query(self, query_embeddings, n_results=5, include=None):
        n = min(n_results, len(self._ids))
        return {"ids": [self._ids[:n]], "documents": [self._documents[:n]],
                "metadatas": [self._metadatas[:n]], "distances": [[0.1 * i for i in range(n)]]}

    def delete(self, ids):
        indices = [i for i, x in enumerate(self._ids) if x in ids]
        for idx in sorted(indices, reverse=True):
            self._ids.pop(idx); self._embeddings.pop(idx)
            self._metadatas.pop(idx); self._documents.pop(idx)


fake_collection = FakeCollection()


# ── Fake Ollama ──
FAKE_EMBED = [0.1] * 768

async def mock_ollama_get(path, timeout=10.0):
    if "/api/tags" in path:
        return {"models": [
            {"name": "phi4-mini:latest", "size": 2e9, "modified_at": "2025-01-01T00:00:00Z"},
            {"name": "nomic-embed-text:latest", "size": 5e8, "modified_at": "2025-01-01T00:00:00Z"},
        ]}
    return {}

async def mock_ollama_post(path, payload, timeout=120.0):
    if "/api/embed" in path:
        return {"embeddings": [FAKE_EMBED]}
    if "/api/generate" in path:
        return {"response": "Based on the document, the answer is 42."}
    return {}

async def mock_ollama_embed(text):
    return FAKE_EMBED

async def mock_ollama_generate(model, prompt, system=""):
    return "Based on the document, the answer is 42."


# ── Fake Temporal ──
_workflow_runs = {}


@dataclass
class FakeWorkflowHandle:
    workflow_id: str

    async def describe(self):
        if self.workflow_id not in _workflow_runs:
            raise RuntimeError(f"workflow not found: {self.workflow_id}")
        run = _workflow_runs[self.workflow_id]
        status_mock = MagicMock()
        if run.get("result"):
            status_mock.name = "COMPLETED"
        else:
            status_mock.name = "RUNNING"
        desc_mock = MagicMock()
        desc_mock.status = status_mock
        return desc_mock

    async def result(self):
        run = _workflow_runs.get(self.workflow_id)
        if run and run.get("result"):
            return run["result"]
        raise RuntimeError("Not completed")


class FakeTemporalClient:
    async def start_workflow(self, workflow_name, inp, id, task_queue):
        _workflow_runs[id] = {"input": inp, "result": None}

    def get_workflow_handle(self, workflow_id):
        return FakeWorkflowHandle(workflow_id)


_fake_temporal = FakeTemporalClient()


def _simulate_complete(doc_id, filename, chunk_count):
    """Simulate worker completing workflow: set result + populate ChromaDB."""
    from datetime import datetime, timezone
    wf_id = f"doc-upload-{doc_id}"
    _workflow_runs[wf_id] = {
        "result": {
            "doc_id": doc_id, "filename": filename,
            "chunks": chunk_count, "characters": 1000,
            "status": "completed", "error": None,
        }
    }
    now = datetime.now(timezone.utc).isoformat()
    for i in range(chunk_count):
        fake_collection._ids.append(f"{doc_id}_chunk_{i}")
        fake_collection._embeddings.append(FAKE_EMBED)
        fake_collection._metadatas.append({
            "doc_id": doc_id, "filename": filename,
            "chunk_index": i, "total_chunks": chunk_count,
            "uploaded_at": now,
        })
        fake_collection._documents.append(f"Chunk {i} content from {filename}. Important data here.")


# ── Fixtures ──
@pytest_asyncio.fixture(autouse=True)
async def reset():
    fake_collection._ids.clear(); fake_collection._embeddings.clear()
    fake_collection._metadatas.clear(); fake_collection._documents.clear()
    fake_s3._objects.clear()
    _workflow_runs.clear()
    import main
    main._temporal = None; main._chroma_client = None; main._collection = None
    yield


@pytest_asyncio.fixture
async def client():
    import main
    main.get_collection = lambda: fake_collection
    main.get_temporal = AsyncMock(return_value=_fake_temporal)
    main._s3 = lambda: fake_s3
    main.ollama_get = mock_ollama_get
    main.ollama_post = mock_ollama_post
    main.ollama_embed = mock_ollama_embed
    main.ollama_generate = mock_ollama_generate
    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ═══════════════════════════════════════════════════════
# 1. HEALTH
# ═══════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_health(client):
    r = await client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] in ("ok", "degraded")


# ═══════════════════════════════════════════════════════
# 2. MODELS
# ═══════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_models(client):
    r = await client.get("/api/models")
    assert r.status_code == 200
    names = [m["name"] for m in r.json()["models"]]
    assert "phi4-mini:latest" in names
    assert "nomic-embed-text:latest" in names


# ═══════════════════════════════════════════════════════
# 3. UPLOAD → S3 + WORKFLOW
# ═══════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_upload_saves_to_s3_and_starts_workflow(client):
    r = await client.post("/api/upload", files={"file": ("test.txt", io.BytesIO(b"Hello world content"), "text/plain")})
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "processing"
    assert data["doc_id"]
    assert data["workflow_id"]
    assert data["filename"] == "test.txt"
    # Verify file is in S3
    s3_key = f"raw/{data['doc_id']}/test.txt"
    assert f"lcq-documents/{s3_key}" in fake_s3._objects
    assert fake_s3._objects[f"lcq-documents/{s3_key}"] == b"Hello world content"
    # Verify workflow registered
    assert data["workflow_id"] in _workflow_runs


@pytest.mark.asyncio
async def test_upload_empty_returns_400(client):
    r = await client.post("/api/upload", files={"file": ("e.txt", io.BytesIO(b""), "text/plain")})
    assert r.status_code == 400
    assert "Empty" in r.json()["detail"]


# ═══════════════════════════════════════════════════════
# 4. STATUS POLLING
# ═══════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_status_processing(client):
    r = await client.post("/api/upload", files={"file": ("d.txt", io.BytesIO(b"Content"), "text/plain")})
    doc_id = r.json()["doc_id"]
    s = await client.get(f"/api/upload/{doc_id}/status")
    assert s.status_code == 200
    assert s.json()["status"] == "processing"


@pytest.mark.asyncio
async def test_status_completed(client):
    r = await client.post("/api/upload", files={"file": ("report.txt", io.BytesIO(b"Report data"), "text/plain")})
    doc_id = r.json()["doc_id"]
    _simulate_complete(doc_id, "report.txt", 3)
    s = await client.get(f"/api/upload/{doc_id}/status")
    assert s.status_code == 200
    d = s.json()
    assert d["status"] == "completed"
    assert d["chunks"] == 3
    assert d["filename"] == "report.txt"


@pytest.mark.asyncio
async def test_status_not_found(client):
    r = await client.get("/api/upload/nonexistent/status")
    assert r.status_code == 404


# ═══════════════════════════════════════════════════════
# 5. DOCUMENT LISTING
# ═══════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_list_empty(client):
    r = await client.get("/api/documents")
    assert r.json()["documents"] == []


@pytest.mark.asyncio
async def test_list_after_complete(client):
    _simulate_complete("aaa", "notes.txt", 2)
    _simulate_complete("bbb", "guide.txt", 5)
    r = await client.get("/api/documents")
    docs = r.json()["documents"]
    assert len(docs) == 2
    assert {d["filename"] for d in docs} == {"notes.txt", "guide.txt"}


# ═══════════════════════════════════════════════════════
# 6. QUERY WITH CONTEXT
# ═══════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_query_with_docs(client):
    _simulate_complete("abc", "knowledge.txt", 3)
    r = await client.post("/api/query", json={"query": "What is 42?", "model": "phi4-mini:latest"})
    assert r.status_code == 200
    d = r.json()
    assert "42" in d["answer"]
    assert len(d["sources"]) > 0
    assert d["sources"][0]["filename"] == "knowledge.txt"


# ═══════════════════════════════════════════════════════
# 7. QUERY EMPTY DB
# ═══════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_query_empty(client):
    r = await client.post("/api/query", json={"query": "Hello?", "model": "phi4-mini:latest"})
    assert "I do not know" in r.json()["answer"]
    assert r.json()["sources"] == []


# ═══════════════════════════════════════════════════════
# 8. QUERY DISABLED DOCS
# ═══════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_query_disabled(client):
    _simulate_complete("d1", "enabled.txt", 2)
    # Wrong IDs → no matches
    r = await client.post("/api/query", json={"query": "Test", "model": "phi4-mini:latest", "enabled_doc_ids": ["wrong"]})
    assert "I do not know" in r.json()["answer"]
    # Correct ID → answer
    r2 = await client.post("/api/query", json={"query": "Test", "model": "phi4-mini:latest", "enabled_doc_ids": ["d1"]})
    assert "I do not know" not in r2.json()["answer"]
    assert len(r2.json()["sources"]) > 0


# ═══════════════════════════════════════════════════════
# 9. DELETE (ChromaDB + S3)
# ═══════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_delete(client):
    # Upload to S3 and simulate completion
    r = await client.post("/api/upload", files={"file": ("tmp.txt", io.BytesIO(b"Temp data"), "text/plain")})
    doc_id = r.json()["doc_id"]
    _simulate_complete(doc_id, "tmp.txt", 2)
    assert fake_collection.count() == 2

    # Also add extracted text to S3 to verify cleanup
    fake_s3.put_object(Bucket="lcq-documents", Key=f"extracted/{doc_id}.txt", Body=b"extracted text")

    d = await client.delete(f"/api/documents/{doc_id}")
    assert d.status_code == 200
    assert fake_collection.count() == 0
    # S3 raw and extracted should be cleaned
    raw_key = f"lcq-documents/raw/{doc_id}/tmp.txt"
    ext_key = f"lcq-documents/extracted/{doc_id}.txt"
    assert raw_key not in fake_s3._objects
    assert ext_key not in fake_s3._objects


@pytest.mark.asyncio
async def test_delete_not_found(client):
    r = await client.delete("/api/documents/nope")
    assert r.status_code == 404


# ═══════════════════════════════════════════════════════
# 10. ERROR HANDLING
# ═══════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_errors_json(client):
    import main
    main.get_collection = lambda: (_ for _ in ()).throw(HTTPException(503, "ChromaDB down"))
    r = await client.get("/api/documents")
    assert r.headers["content-type"].startswith("application/json")
    assert r.status_code == 503
    main.get_collection = lambda: fake_collection


@pytest.mark.asyncio
async def test_ollama_down(client):
    import main
    main.ollama_get = AsyncMock(side_effect=ConnectionError("refused"))
    r = await client.get("/api/models")
    assert r.status_code == 502
    assert "detail" in r.json()
    main.ollama_get = mock_ollama_get


# ═══════════════════════════════════════════════════════
# 11. FULL FLOW
# ═══════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_full_flow(client):
    # 1. Empty DB → I do not know
    r1 = await client.post("/api/query", json={"query": "Hi?", "model": "phi4-mini:latest"})
    assert "I do not know" in r1.json()["answer"]

    # 2. Upload → S3 + workflow
    up = await client.post("/api/upload", files={"file": ("guide.txt", io.BytesIO(b"Python guide content here. " * 50), "text/plain")})
    assert up.status_code == 200
    doc_id = up.json()["doc_id"]
    assert f"lcq-documents/raw/{doc_id}/guide.txt" in fake_s3._objects

    # 3. Status = processing
    s1 = await client.get(f"/api/upload/{doc_id}/status")
    assert s1.json()["status"] == "processing"

    # 4. Worker completes (simulated)
    _simulate_complete(doc_id, "guide.txt", 4)

    # 5. Status = completed
    s2 = await client.get(f"/api/upload/{doc_id}/status")
    assert s2.json()["status"] == "completed"
    assert s2.json()["chunks"] == 4

    # 6. Documents visible
    docs = (await client.get("/api/documents")).json()["documents"]
    assert len(docs) == 1
    assert docs[0]["filename"] == "guide.txt"

    # 7. Query → answer with sources
    r2 = await client.post("/api/query", json={"query": "What is Python?", "model": "phi4-mini:latest"})
    assert r2.status_code == 200
    assert len(r2.json()["sources"]) > 0
    assert "42" in r2.json()["answer"]

    # 8. Delete → cleans ChromaDB + S3
    d = await client.delete(f"/api/documents/{doc_id}")
    assert d.status_code == 200
    assert fake_collection.count() == 0

    # 9. Query again → I do not know
    r3 = await client.post("/api/query", json={"query": "What is Python?", "model": "phi4-mini:latest"})
    assert "I do not know" in r3.json()["answer"]


# ═══════════════════════════════════════════════════════
# 12. ALL JSON CONTENT-TYPE
# ═══════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_json_content_types(client):
    for url in ["/api/health", "/api/models", "/api/documents"]:
        r = await client.get(url)
        assert "application/json" in r.headers.get("content-type", "")
