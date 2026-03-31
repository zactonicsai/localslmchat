"""Temporal activities for upload and query processing."""
import os, json, tempfile
from datetime import datetime, timezone
from typing import List
import boto3, httpx, chromadb
from temporalio import activity
from shared.models import (UploadInput, ExtractResult, ChunkResult, EmbedStoreResult,
                           QueryInput, QueryResult, S3_TEXT_PREFIX, S3_ANSWER_PREFIX)

S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://localhost:4566")
S3_BUCKET = os.getenv("S3_BUCKET", "lcq-documents")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8200")
CHUNK_SIZE, CHUNK_OVERLAP = 800, 150

def _s3():
    return boto3.client("s3", endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))

def _chroma():
    c = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT); c.heartbeat()
    return c.get_or_create_collection(name="local_context", metadata={"hnsw:space": "cosine"})

async def _embed(text):
    async with httpx.AsyncClient(timeout=60.0) as c:
        r = await c.post(f"{OLLAMA_BASE_URL}/api/embed", json={"model": EMBED_MODEL, "input": text})
        if r.status_code == 404:
            r = await c.post(f"{OLLAMA_BASE_URL}/api/embed", json={"model": EMBED_MODEL, "prompt": text})
        r.raise_for_status(); d = r.json()
    e = d.get("embeddings")
    if e and len(e) > 0: return e[0]
    e2 = d.get("embedding")
    if e2 and len(e2) > 0: return e2
    raise ValueError(f"No embedding from {EMBED_MODEL}")

@activity.defn
async def extract_text_activity(inp: UploadInput) -> ExtractResult:
    raw = _s3().get_object(Bucket=S3_BUCKET, Key=inp.s3_raw_key)["Body"].read()
    ext = inp.filename.rsplit(".", 1)[-1].lower() if "." in inp.filename else ""
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as f:
        f.write(raw); tp = f.name
    try:
        if ext == "pdf":
            import pdfplumber
            with pdfplumber.open(tp) as pdf:
                text = "\n\n".join(p.extract_text() or "" for p in pdf.pages)
        elif ext in ("doc", "docx"):
            from docx import Document
            text = "\n\n".join(p.text for p in Document(tp).paragraphs if p.text.strip())
        elif ext in ("xlsx", "xls"):
            import openpyxl; wb = openpyxl.load_workbook(tp, read_only=True, data_only=True)
            text = "\n".join(" | ".join(str(c) if c else "" for c in r) for ws in wb.worksheets for r in ws.iter_rows(values_only=True)); wb.close()
        else:
            text = raw.decode("utf-8", errors="replace")
    finally:
        os.unlink(tp)
    if not text.strip(): raise ValueError(f"No text in {inp.filename}")
    k = f"{S3_TEXT_PREFIX}{inp.doc_id}.txt"
    _s3().put_object(Bucket=S3_BUCKET, Key=k, Body=text.encode())
    return ExtractResult(s3_text_key=k, characters=len(text))

@activity.defn
async def chunk_text_activity(s3_text_key: str) -> ChunkResult:
    text = _s3().get_object(Bucket=S3_BUCKET, Key=s3_text_key)["Body"].read().decode("utf-8")
    words = text.split()
    if not words: return ChunkResult(chunks=[], count=0)
    chunks, i = [], 0
    while i < len(words): chunks.append(" ".join(words[i:i+CHUNK_SIZE])); i += CHUNK_SIZE - CHUNK_OVERLAP
    return ChunkResult(chunks=chunks, count=len(chunks))

@activity.defn
async def embed_and_store_activity(doc_id: str, filename: str, chunks: List[str]) -> EmbedStoreResult:
    ids, embs, metas, docs = [], [], [], []
    now = datetime.now(timezone.utc).isoformat()
    for i, ch in enumerate(chunks):
        activity.heartbeat(f"{i+1}/{len(chunks)}")
        ids.append(f"{doc_id}_chunk_{i}"); embs.append(await _embed(ch))
        metas.append({"doc_id": doc_id, "filename": filename, "chunk_index": i, "total_chunks": len(chunks), "uploaded_at": now})
        docs.append(ch)
    _chroma().add(ids=ids, embeddings=embs, metadatas=metas, documents=docs)
    return EmbedStoreResult(chunks_stored=len(ids))

@activity.defn
async def execute_query_activity(inp: QueryInput) -> QueryResult:
    s3 = _s3()
    q_data = json.loads(s3.get_object(Bucket=S3_BUCKET, Key=inp.s3_query_key)["Body"].read())
    query_text = q_data["query"]
    activity.logger.info(f"Query [{inp.query_id}]: {query_text[:80]}")
    coll = _chroma()
    if coll.count() == 0:
        answer_obj = {"query_id": inp.query_id, "answer": "I do not know. No documents uploaded yet.", "sources": []}
    else:
        activity.heartbeat("Embedding query"); qemb = await _embed(query_text)
        n = min(8, coll.count())
        res = coll.query(query_embeddings=[qemb], n_results=n, include=["documents", "metadatas", "distances"])
        fdocs, fsrc = [], []
        if res["documents"] and res["documents"][0]:
            for i, doc in enumerate(res["documents"][0]):
                m, d = res["metadatas"][0][i], res["distances"][0][i]
                if inp.enabled_doc_ids and m["doc_id"] not in inp.enabled_doc_ids: continue
                fdocs.append(doc); fsrc.append({"filename": m["filename"], "doc_id": m["doc_id"], "chunk_index": m["chunk_index"], "distance": round(d, 4)})
        if not fdocs:
            answer_obj = {"query_id": inp.query_id, "answer": "I do not know. No relevant information in enabled documents.", "sources": []}
        else:
            activity.heartbeat("Generating answer")
            ctx = "\n\n---\n\n".join(fdocs)
            # Cap context to ~12k chars to keep prompt + response within safe limits
            if len(ctx) > 12_000:
                ctx = ctx[:12_000] + "\n\n[Context truncated]"
            sys_p = "You are Local Context Query. Answer ONLY from provided context. If context lacks info say 'I do not know.' Cite document names. Be concise."
            prompt = f"CONTEXT:\n{ctx}\n\nQUESTION:\n{query_text}\n\nAnswer from context only:"
            async with httpx.AsyncClient(timeout=180.0) as c:
                r = await c.post(f"{OLLAMA_BASE_URL}/api/generate",
                    json={"model": inp.model, "prompt": prompt, "system": sys_p, "stream": False,
                          "options": {"temperature": 0.3, "num_ctx": 4096}})
                r.raise_for_status()
                answer_text = r.json().get("response", "")
                # Truncate to avoid exceeding Temporal's 2MB payload limit
                if len(answer_text) > 50_000:
                    answer_text = answer_text[:50_000] + "\n\n[Answer truncated due to length]"
                answer_obj = {"query_id": inp.query_id, "answer": answer_text, "sources": fsrc}

    s3_answer_key = f"{S3_ANSWER_PREFIX}{inp.query_id}.json"
    s3.put_object(Bucket=S3_BUCKET, Key=s3_answer_key, Body=json.dumps(answer_obj).encode())
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            await c.post(f"{BACKEND_URL}/api/internal/query-complete", json={"query_id": inp.query_id})
    except Exception: pass
    return QueryResult(query_id=inp.query_id, status="completed", s3_answer_key=s3_answer_key)
