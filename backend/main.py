"""
Local Context Query — FastAPI Backend

Upload triggers a Temporal workflow (extract → chunk → embed → store).
Query reads from ChromaDB and calls Ollama for generation.
"""
import os
import uuid
import textwrap
import traceback
from datetime import datetime, timezone

import httpx
import chromadb
from temporalio.client import Client as TemporalClient
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add shared module to path
import sys
sys.path.insert(0, "/app")

from shared.models import TASK_QUEUE, WORKFLOW_ID_PREFIX, UploadInput, UploadResult

# ── Config ─────────────────────────────────────────────────────────────
TEMPORAL_ADDRESS = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/app/uploads")
COLLECTION_NAME = "local_context"

app = FastAPI(title="Local Context Query API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global exception handler ────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    tb = traceback.format_exc()
    print(f"[ERROR] {request.method} {request.url.path}: {exc}\n{tb}")
    return JSONResponse(status_code=500, content={"detail": f"Server error: {str(exc)}"})


# ── Temporal client (lazy) ──────────────────────────────────────────────
_temporal_client = None


async def get_temporal_client():
    global _temporal_client
    if _temporal_client is None:
        _temporal_client = await TemporalClient.connect(TEMPORAL_ADDRESS)
        print(f"[INFO] Temporal connected at {TEMPORAL_ADDRESS}")
    return _temporal_client


# ── ChromaDB (lazy, reconnect on failure) ───────────────────────────────
_chroma_client = None
_collection = None


def get_collection():
    global _chroma_client, _collection
    try:
        if _chroma_client is None:
            _chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            _chroma_client.heartbeat()
            print(f"[INFO] ChromaDB connected at {CHROMA_HOST}:{CHROMA_PORT}")
        if _collection is None:
            _collection = _chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return _collection
    except Exception as e:
        _chroma_client = None
        _collection = None
        raise HTTPException(status_code=503, detail=f"ChromaDB unavailable: {e}")


# ── Ollama helpers ──────────────────────────────────────────────────────
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


async def ollama_generate(model, prompt, system=""):
    data = await ollama_post("/api/generate", {
        "model": model, "prompt": prompt, "system": system,
        "stream": False, "options": {"temperature": 0.3, "num_ctx": 4096},
    })
    return data.get("response", "")


_embed_model = None


async def _resolve_embed_model():
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    preferred = ["nomic-embed-text", "all-minilm", "mxbai-embed-large"]
    try:
        tags = await ollama_get("/api/tags")
        available = [m["name"].split(":")[0] for m in tags.get("models", [])]
    except Exception:
        available = []
    for m in preferred:
        if m in available:
            _embed_model = m
            return m
    if available:
        _embed_model = available[0]
        return _embed_model
    raise HTTPException(status_code=503, detail="No models in Ollama. Run: ollama pull nomic-embed-text")


async def ollama_embed(text):
    model = await _resolve_embed_model()
    data = await ollama_post("/api/embed", {"model": model, "input": text}, timeout=60.0)
    embs = data.get("embeddings", [])
    if embs and len(embs) > 0:
        return embs[0]
    raise ValueError(f"Empty embeddings from {model}")


# ═══════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════

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
        await get_temporal_client()
    except Exception as e:
        errors["temporal"] = str(e)
    if errors:
        return {"status": "degraded", "errors": errors}
    return {"status": "ok"}


@app.get("/api/models")
async def list_models():
    try:
        data = await ollama_get("/api/tags")
        return {
            "models": [
                {"name": m["name"], "size": m.get("size", 0), "modified": m.get("modified_at", "")}
                for m in data.get("models", [])
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Cannot reach Ollama at {OLLAMA_BASE_URL}: {e}")


# ── Upload: save file locally, start Temporal workflow ──
@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    doc_id = uuid.uuid4().hex[:12]
    safe_name = file.filename.replace("/", "_").replace("\\", "_")
    filepath = os.path.join(UPLOAD_DIR, f"{doc_id}_{safe_name}")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    with open(filepath, "wb") as f:
        f.write(content)

    # Start Temporal workflow
    try:
        temporal = await get_temporal_client()
        workflow_id = f"{WORKFLOW_ID_PREFIX}-{doc_id}"

        await temporal.start_workflow(
            "DocumentUploadWorkflow",
            UploadInput(doc_id=doc_id, filename=safe_name, filepath=filepath),
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )

        return {
            "doc_id": doc_id,
            "filename": safe_name,
            "workflow_id": workflow_id,
            "status": "processing",
        }
    except Exception as e:
        # Clean up file on failure
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(status_code=502, detail=f"Failed to start workflow: {e}")


# ── Upload status: poll Temporal for workflow result ──
@app.get("/api/upload/{doc_id}/status")
async def upload_status(doc_id: str):
    try:
        temporal = await get_temporal_client()
        workflow_id = f"{WORKFLOW_ID_PREFIX}-{doc_id}"
        handle = temporal.get_workflow_handle(workflow_id)
        desc = await handle.describe()

        status = str(desc.status)

        # If completed, get the result
        if "COMPLETED" in status:
            result = await handle.result()
            return {
                "doc_id": doc_id,
                "status": result.status,
                "filename": result.filename,
                "chunks": result.chunks,
                "characters": result.characters,
                "error": result.error,
            }
        elif "FAILED" in status:
            return {
                "doc_id": doc_id,
                "status": "failed",
                "error": "Workflow failed. Check Temporal UI at http://localhost:8080",
            }
        elif "RUNNING" in status:
            return {"doc_id": doc_id, "status": "processing"}
        else:
            return {"doc_id": doc_id, "status": status.lower()}

    except Exception as e:
        error_str = str(e)
        if "not found" in error_str.lower():
            raise HTTPException(status_code=404, detail=f"Workflow not found for doc {doc_id}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {e}")


# ── Document listing from ChromaDB ──
@app.get("/api/documents")
async def list_documents():
    coll = get_collection()
    result = coll.get(include=["metadatas"])

    docs = {}
    for meta in result["metadatas"]:
        did = meta["doc_id"]
        if did not in docs:
            docs[did] = {
                "doc_id": did,
                "filename": meta["filename"],
                "total_chunks": meta["total_chunks"],
                "uploaded_at": meta.get("uploaded_at", ""),
            }
    return {"documents": list(docs.values())}


# ── Delete document ──
@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    coll = get_collection()
    result = coll.get(include=["metadatas"])

    ids_to_delete = [
        result["ids"][i]
        for i, meta in enumerate(result["metadatas"])
        if meta.get("doc_id") == doc_id
    ]
    if not ids_to_delete:
        raise HTTPException(status_code=404, detail="Document not found")

    coll.delete(ids=ids_to_delete)

    if os.path.isdir(UPLOAD_DIR):
        for f in os.listdir(UPLOAD_DIR):
            if f.startswith(doc_id):
                os.remove(os.path.join(UPLOAD_DIR, f))
                break

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
                      "Please upload documents first to provide context for your questions.",
            "sources": [],
        }

    try:
        query_emb = await ollama_embed(req.query)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding error: {e}")

    n = min(8, coll.count())
    results = coll.query(
        query_embeddings=[query_emb], n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    filtered_docs, filtered_sources = [], []
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            dist = results["distances"][0][i]
            if req.enabled_doc_ids and meta["doc_id"] not in req.enabled_doc_ids:
                continue
            filtered_docs.append(doc)
            filtered_sources.append({
                "filename": meta["filename"],
                "doc_id": meta["doc_id"],
                "chunk_index": meta["chunk_index"],
                "distance": round(dist, 4),
            })

    if not filtered_docs:
        return {
            "answer": "I do not know. The enabled documents in the context database "
                      "do not contain information relevant to your question.",
            "sources": [],
        }

    context_text = "\n\n---\n\n".join(filtered_docs)

    system_prompt = textwrap.dedent("""\
        You are Local Context Query, a helpful assistant that answers questions
        ONLY based on the provided context documents.

        STRICT RULES:
        1. ONLY use information from the provided context to answer.
        2. If the context does not contain enough information, respond with:
           "I do not know. The context documents do not contain information to answer this question."
        3. Do NOT make up information or use general knowledge.
        4. Cite which document the information comes from.
        5. Be concise and accurate.
    """)

    user_prompt = (
        f"CONTEXT DOCUMENTS:\n{context_text}\n\n"
        f"USER QUESTION:\n{req.query}\n\n"
        f"Answer using ONLY the context above. If the context doesn't help, say \"I do not know.\":"
    )

    try:
        answer = await ollama_generate(req.model, user_prompt, system_prompt)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {e.response.status_code}")
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail=f"Cannot connect to Ollama at {OLLAMA_BASE_URL}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {e}")

    return {"answer": answer, "sources": filtered_sources}
