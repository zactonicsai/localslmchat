"""
Local Context Query — FastAPI Backend
Query: save to S3 → Temporal workflow → answer to S3 → WebSocket notify.
"""
import os, sys, uuid, json, traceback, asyncio
from typing import Set

import boto3, httpx, chromadb
from temporalio.client import Client as TemporalClient
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.insert(0, "/app")
from shared.models import (TASK_QUEUE, QUERY_TASK_QUEUE, WORKFLOW_ID_PREFIX, QUERY_WORKFLOW_PREFIX,
                           S3_RAW_PREFIX, S3_QUERY_PREFIX, S3_ANSWER_PREFIX,
                           UploadInput, QueryInput)

TEMPORAL_ADDRESS = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
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


# ── WebSocket hub ──
class WSHub:
    def __init__(self):
        self.clients: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients.add(ws)

    def disconnect(self, ws: WebSocket):
        self.clients.discard(ws)

    async def broadcast(self, message: dict):
        dead = set()
        for ws in self.clients:
            try:
                await ws.send_json(message)
            except Exception:
                dead.add(ws)
        self.clients -= dead


ws_hub = WSHub()


# ── Clients ──
_temporal = None
_chroma_client = None
_collection = None


def _s3():
    return boto3.client("s3", endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))


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
            _collection = _chroma_client.get_or_create_collection(name="local_context", metadata={"hnsw:space": "cosine"})
        return _collection
    except Exception as e:
        _chroma_client = None; _collection = None
        raise HTTPException(status_code=503, detail=f"ChromaDB unavailable: {e}")


async def ollama_get(path, timeout=10.0):
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.get(f"{OLLAMA_BASE_URL}{path}"); r.raise_for_status(); return r.json()


def _wf_status(desc):
    s = desc.status
    if hasattr(s, 'name'): return s.name
    return str(getattr(s, 'value', s))


# ═══ WebSocket endpoint ═══
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_hub.connect(ws)
    try:
        while True:
            # Keep alive — client can send pings or we just wait
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        ws_hub.disconnect(ws)
    except Exception:
        ws_hub.disconnect(ws)


# ═══ Internal callback from worker ═══
class QueryCompleteNotify(BaseModel):
    query_id: str


@app.post("/api/internal/query-complete")
async def query_complete_callback(body: QueryCompleteNotify):
    """Called by worker after saving answer to S3. Broadcasts to all WS clients."""
    # Read answer from S3 and broadcast
    try:
        s3 = _s3()
        key = f"{S3_ANSWER_PREFIX}{body.query_id}.json"
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        answer_data = json.loads(obj["Body"].read())
        await ws_hub.broadcast({
            "type": "query_answer",
            "query_id": body.query_id,
            "answer": answer_data.get("answer", ""),
            "sources": answer_data.get("sources", []),
        })
    except Exception as e:
        # Still broadcast that it's done even if S3 read fails
        await ws_hub.broadcast({
            "type": "query_answer",
            "query_id": body.query_id,
            "answer": f"Answer ready but failed to load: {e}",
            "sources": [],
        })
    return {"ok": True}


# ═══ Routes ═══

@app.get("/api/health")
async def health():
    errors = {}
    try: get_collection()
    except Exception as e: errors["chromadb"] = str(e)
    try: await ollama_get("/api/tags")
    except Exception as e: errors["ollama"] = str(e)
    try: await get_temporal()
    except Exception as e: errors["temporal"] = str(e)
    try: _s3().head_bucket(Bucket=S3_BUCKET)
    except Exception as e: errors["s3"] = str(e)
    return {"status": "degraded" if errors else "ok", **({"errors": errors} if errors else {})}


@app.get("/api/models")
async def list_models():
    try:
        data = await ollama_get("/api/tags")
        return {"models": [{"name": m["name"], "size": m.get("size", 0)} for m in data.get("models", [])]}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Cannot reach Ollama: {e}")


# ── Upload ──
@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename: raise HTTPException(status_code=400, detail="No filename")
    content = await file.read()
    if not content: raise HTTPException(status_code=400, detail="Empty file")
    doc_id = uuid.uuid4().hex[:12]
    safe_name = file.filename.replace("/", "_").replace("\\", "_")
    s3_key = f"{S3_RAW_PREFIX}{doc_id}/{safe_name}"
    try: _s3().put_object(Bucket=S3_BUCKET, Key=s3_key, Body=content)
    except Exception as e: raise HTTPException(status_code=502, detail=f"S3 failed: {e}")
    try:
        temporal = await get_temporal()
        wf_id = f"{WORKFLOW_ID_PREFIX}-{doc_id}"
        await temporal.start_workflow("DocumentUploadWorkflow",
            UploadInput(doc_id=doc_id, filename=safe_name, s3_raw_key=s3_key),
            id=wf_id, task_queue=TASK_QUEUE)
        # Broadcast upload started
        await ws_hub.broadcast({"type": "upload_started", "doc_id": doc_id, "filename": safe_name})
        return {"doc_id": doc_id, "filename": safe_name, "workflow_id": wf_id, "status": "processing"}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Workflow failed: {e}")


@app.get("/api/upload/{doc_id}/status")
async def upload_status(doc_id: str):
    try:
        temporal = await get_temporal()
        handle = temporal.get_workflow_handle(f"{WORKFLOW_ID_PREFIX}-{doc_id}")
        desc = await handle.describe()
        sn = _wf_status(desc)
        if sn in ("COMPLETED", "2"):
            try:
                r = await handle.result()
                d = r if isinstance(r, dict) else {"status": getattr(r, "status", "completed"),
                    "filename": getattr(r, "filename", ""), "chunks": getattr(r, "chunks", 0),
                    "characters": getattr(r, "characters", 0), "error": getattr(r, "error", None)}
                d["doc_id"] = doc_id
                # Broadcast upload complete
                if d.get("status") == "completed":
                    await ws_hub.broadcast({"type": "upload_complete", "doc_id": doc_id,
                        "filename": d.get("filename", ""), "chunks": d.get("chunks", 0)})
                return d
            except Exception as e:
                return {"doc_id": doc_id, "status": "completed", "chunks": 0, "error": str(e)}
        elif sn in ("FAILED", "3"):
            return {"doc_id": doc_id, "status": "failed", "error": "Workflow failed"}
        return {"doc_id": doc_id, "status": "processing"}
    except Exception as e:
        if "not found" in str(e).lower(): raise HTTPException(status_code=404, detail="Not found")
        raise HTTPException(status_code=500, detail=str(e))


# ── Documents ──
@app.get("/api/documents")
async def list_documents():
    coll = get_collection()
    result = coll.get(include=["metadatas"])
    docs = {}
    for meta in result["metadatas"]:
        did = meta["doc_id"]
        if did not in docs:
            docs[did] = {"doc_id": did, "filename": meta["filename"],
                         "total_chunks": meta["total_chunks"], "uploaded_at": meta.get("uploaded_at", "")}
    return {"documents": list(docs.values())}


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    coll = get_collection()
    result = coll.get(include=["metadatas"])
    ids_del = [result["ids"][i] for i, m in enumerate(result["metadatas"]) if m.get("doc_id") == doc_id]
    if not ids_del: raise HTTPException(status_code=404, detail="Not found")
    coll.delete(ids=ids_del)
    try:
        s3 = _s3()
        for pfx in [f"raw/{doc_id}/", f"extracted/{doc_id}"]:
            for obj in s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=pfx).get("Contents", []):
                s3.delete_object(Bucket=S3_BUCKET, Key=obj["Key"])
    except Exception: pass
    return {"deleted": len(ids_del), "doc_id": doc_id}


# ── Query: save to S3 → start workflow → return immediately ──
class QueryRequest(BaseModel):
    query: str
    model: str = "qwen3:8b"
    enabled_doc_ids: list = []


@app.post("/api/query")
async def start_query(req: QueryRequest):
    query_id = uuid.uuid4().hex[:12]
    s3_query_key = f"{S3_QUERY_PREFIX}{query_id}.json"

    # Save question to S3
    try:
        _s3().put_object(Bucket=S3_BUCKET, Key=s3_query_key,
            Body=json.dumps({"query": req.query, "model": req.model, "enabled_doc_ids": req.enabled_doc_ids}).encode())
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"S3 failed: {e}")

    # Start Temporal workflow
    try:
        temporal = await get_temporal()
        wf_id = f"{QUERY_WORKFLOW_PREFIX}-{query_id}"
        await temporal.start_workflow("QueryWorkflow",
            QueryInput(query_id=query_id, s3_query_key=s3_query_key, model=req.model, enabled_doc_ids=req.enabled_doc_ids),
            id=wf_id, task_queue=QUERY_TASK_QUEUE)
        return {"query_id": query_id, "status": "queued"}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Query workflow failed: {e}")


# ── Get answer from S3 (fallback polling) ──
@app.get("/api/query/{query_id}/answer")
async def get_answer(query_id: str):
    try:
        s3 = _s3()
        key = f"{S3_ANSWER_PREFIX}{query_id}.json"
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        data = json.loads(obj["Body"].read())
        return {"query_id": query_id, "status": "completed", "answer": data.get("answer", ""), "sources": data.get("sources", [])}
    except Exception as e:
        if "NoSuchKey" in str(e) or "not found" in str(e).lower():
            return {"query_id": query_id, "status": "processing"}
        raise HTTPException(status_code=500, detail=str(e))
