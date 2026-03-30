"""
Local Context Query — FastAPI Backend

Upload → S3 → Temporal workflow (extract → chunk → embed → store in ChromaDB).
Query → embed with same model → ChromaDB search → Ollama generate.
"""
import os
import sys
import uuid
import textwrap
import traceback

import boto3
import httpx
import chromadb
from temporalio.client import Client as TemporalClient
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.insert(0, "/app")
from shared.models import TASK_QUEUE, WORKFLOW_ID_PREFIX, S3_RAW_PREFIX, UploadInput

# ── Config ──
TEMPORAL_ADDRESS = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://localhost:4566")
S3_BUCKET = os.getenv("S3_BUCKET", "lcq-documents")

app = FastAPI(title="Local Context Query API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(Exception)
async def global_exc(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"detail": f"Server error: {exc}"})


# ── Clients (lazy) ──
_temporal = None
_chroma_client = None
_collection = None


def _s3():
    return boto3.client(
        "s3", endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )


async def get_temporal():
    global _temporal
    if _temporal is None:
        _temporal = await TemporalClient.connect(TEMPORAL_ADDRESS)
    return _temporal


def get_collection():
    global _chroma_client, _collection
    try:
        if _chroma_client is None:
            _chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            _chroma_client.heartbeat()
        if _collection is None:
            _collection = _chroma_client.get_or_create_collection(
                name="local_context", metadata={"hnsw:space": "cosine"})
        return _collection
    except Exception as e:
        _chroma_client = None
        _collection = None
        raise HTTPException(status_code=503, detail=f"ChromaDB unavailable: {e}")


# ── Ollama ──
async def ollama_get(path, timeout=10.0):
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.get(f"{OLLAMA_BASE_URL}{path}")
        r.raise_for_status()
        return r.json()


async def ollama_post(path, payload, timeout=120.0):
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(f"{OLLAMA_BASE_URL}{path}", json=payload)
        r.raise_for_status()
        return r.json()


async def ollama_embed(text):
    """Embed with the SAME model the worker uses. Supports both old and new Ollama API."""
    # Try new API first (/api/embed with "input"), fall back to old (/api/embeddings with "prompt")
    async with httpx.AsyncClient(timeout=60.0) as c:
        # New endpoint (Ollama >= 0.4)
        resp = await c.post(f"{OLLAMA_BASE_URL}/api/embed", json={"model": EMBED_MODEL, "input": text})
        if resp.status_code == 404:
            # Old endpoint (Ollama < 0.4)
            resp = await c.post(f"{OLLAMA_BASE_URL}/api/embed", json={"model": EMBED_MODEL, "prompt": text})
        resp.raise_for_status()
        data = resp.json()

    # New API returns {"embeddings": [[...]]}; old returns {"embedding": [...]}
    embs = data.get("embeddings")
    if embs and len(embs) > 0:
        return embs[0]
    emb = data.get("embedding")
    if emb and len(emb) > 0:
        return emb
    raise ValueError(f"Empty embeddings from {EMBED_MODEL}")


async def ollama_generate(model, prompt, system=""):
    data = await ollama_post("/api/generate", {
        "model": model, "prompt": prompt, "system": system,
        "stream": False, "options": {"temperature": 0.3, "num_ctx": 4096},
    })
    return data.get("response", "")


# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    errors = {}
    try:
        get_collection()
    except Exception as e:
        errors["chromadb"] = str(e)
    try:
        await ollama_get("/api/tags")
    except Exception as e:
        errors["ollama"] = str(e)
    try:
        await get_temporal()
    except Exception as e:
        errors["temporal"] = str(e)
    try:
        _s3().head_bucket(Bucket=S3_BUCKET)
    except Exception as e:
        errors["s3"] = str(e)
    return {"status": "degraded" if errors else "ok", "errors": errors} if errors else {"status": "ok"}


@app.get("/api/models")
async def list_models():
    try:
        data = await ollama_get("/api/tags")
        return {"models": [
            {"name": m["name"], "size": m.get("size", 0), "modified": m.get("modified_at", "")}
            for m in data.get("models", [])
        ]}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Cannot reach Ollama: {e}")


# ── Upload: save to S3, start Temporal workflow ──
@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    doc_id = uuid.uuid4().hex[:12]
    safe_name = file.filename.replace("/", "_").replace("\\", "_")
    s3_raw_key = f"{S3_RAW_PREFIX}{doc_id}/{safe_name}"

    # Upload to S3
    try:
        _s3().put_object(Bucket=S3_BUCKET, Key=s3_raw_key, Body=content)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"S3 upload failed: {e}")

    # Start Temporal workflow
    try:
        temporal = await get_temporal()
        workflow_id = f"{WORKFLOW_ID_PREFIX}-{doc_id}"
        await temporal.start_workflow(
            "DocumentUploadWorkflow",
            UploadInput(doc_id=doc_id, filename=safe_name, s3_raw_key=s3_raw_key),
            id=workflow_id, task_queue=TASK_QUEUE,
        )
        return {"doc_id": doc_id, "filename": safe_name, "workflow_id": workflow_id, "status": "processing"}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to start workflow: {e}")


# ── Upload status polling ──
@app.get("/api/upload/{doc_id}/status")
async def upload_status(doc_id: str):
    try:
        temporal = await get_temporal()
        handle = temporal.get_workflow_handle(f"{WORKFLOW_ID_PREFIX}-{doc_id}")
        desc = await handle.describe()

        # desc.status is WorkflowExecutionStatus enum — get name safely
        status_val = desc.status
        if hasattr(status_val, 'name'):
            status_name = status_val.name
        elif hasattr(status_val, 'value'):
            status_name = str(status_val.value)
        else:
            status_name = str(status_val)

        if status_name == "COMPLETED" or status_name == "2":
            try:
                result = await handle.result()
                # Result could be dict or dataclass depending on SDK serialization
                if isinstance(result, dict):
                    return {
                        "doc_id": doc_id,
                        "status": result.get("status", "completed"),
                        "filename": result.get("filename", ""),
                        "chunks": result.get("chunks", 0),
                        "characters": result.get("characters", 0),
                        "error": result.get("error"),
                    }
                return {
                    "doc_id": doc_id,
                    "status": getattr(result, "status", "completed"),
                    "filename": getattr(result, "filename", ""),
                    "chunks": getattr(result, "chunks", 0),
                    "characters": getattr(result, "characters", 0),
                    "error": getattr(result, "error", None),
                }
            except Exception as e:
                return {"doc_id": doc_id, "status": "completed", "chunks": 0, "error": str(e)}

        elif status_name in ("FAILED", "3"):
            return {"doc_id": doc_id, "status": "failed",
                    "error": "Workflow failed. Check Temporal UI at http://localhost:8080"}
        elif status_name in ("CANCELED", "TERMINATED", "TIMED_OUT", "5", "6", "7"):
            return {"doc_id": doc_id, "status": "failed", "error": f"Workflow {status_name}"}
        else:
            return {"doc_id": doc_id, "status": "processing"}

    except Exception as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=f"Workflow not found for doc {doc_id}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {e}")


# ── Documents ──
@app.get("/api/documents")
async def list_documents():
    coll = get_collection()
    result = coll.get(include=["metadatas"])
    docs = {}
    for meta in result["metadatas"]:
        did = meta["doc_id"]
        if did not in docs:
            docs[did] = {
                "doc_id": did, "filename": meta["filename"],
                "total_chunks": meta["total_chunks"],
                "uploaded_at": meta.get("uploaded_at", ""),
            }
    return {"documents": list(docs.values())}


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    coll = get_collection()
    result = coll.get(include=["metadatas"])
    ids_to_delete = [result["ids"][i] for i, m in enumerate(result["metadatas"]) if m.get("doc_id") == doc_id]
    if not ids_to_delete:
        raise HTTPException(status_code=404, detail="Document not found")
    coll.delete(ids=ids_to_delete)

    # Also clean S3 files
    try:
        s3 = _s3()
        for prefix in [f"raw/{doc_id}/", f"extracted/{doc_id}"]:
            resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
            for obj in resp.get("Contents", []):
                s3.delete_object(Bucket=S3_BUCKET, Key=obj["Key"])
    except Exception:
        pass  # Best-effort S3 cleanup

    return {"deleted": len(ids_to_delete), "doc_id": doc_id}


# ── Query ──
class QueryRequest(BaseModel):
    query: str
    model: str = "phi4-mini:latest"
    enabled_doc_ids: list = []


@app.post("/api/query")
async def query_documents(req: QueryRequest):
    coll = get_collection()

    if coll.count() == 0:
        return {
            "answer": "I do not know. There are no documents in the context database. "
                      "Please upload documents first.",
            "sources": [],
        }

    try:
        query_emb = await ollama_embed(req.query)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding error: {e}")

    n = min(8, coll.count())
    results = coll.query(query_embeddings=[query_emb], n_results=n,
                         include=["documents", "metadatas", "distances"])

    filtered_docs, filtered_sources = [], []
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            dist = results["distances"][0][i]
            # Empty list = all docs; non-empty = filter
            if req.enabled_doc_ids and meta["doc_id"] not in req.enabled_doc_ids:
                continue
            filtered_docs.append(doc)
            filtered_sources.append({
                "filename": meta["filename"], "doc_id": meta["doc_id"],
                "chunk_index": meta["chunk_index"], "distance": round(dist, 4),
            })

    if not filtered_docs:
        return {
            "answer": "I do not know. The enabled documents do not contain relevant information.",
            "sources": [],
        }

    context = "\n\n---\n\n".join(filtered_docs)
    system = textwrap.dedent("""\
        You are Local Context Query, answering ONLY from provided context.
        RULES:
        1. ONLY use context information. 2. If context lacks info, say "I do not know."
        3. Do NOT use general knowledge. 4. Cite document names. 5. Be concise.""")
    prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{req.query}\n\nAnswer from context only:"

    try:
        answer = await ollama_generate(req.model, prompt, system)
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail=f"Cannot connect to Ollama at {OLLAMA_BASE_URL}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {e}")

    return {"answer": answer, "sources": filtered_sources}
