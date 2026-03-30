"""Temporal activities for document processing."""
import os
from datetime import datetime, timezone

import httpx
import chromadb
from temporalio import activity

from shared.models import (
    UploadInput, ExtractResult, ChunkResult, EmbedStoreResult,
)

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def _get_collection():
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    client.heartbeat()
    return client.get_or_create_collection(
        name="local_context",
        metadata={"hnsw:space": "cosine"},
    )


# ── Activity 1: Extract text from file ──
@activity.defn
async def extract_text_activity(inp: UploadInput) -> ExtractResult:
    activity.logger.info(f"Extracting text from {inp.filename}")

    ext = inp.filename.rsplit(".", 1)[-1].lower() if "." in inp.filename else ""

    if ext == "pdf":
        import pdfplumber
        parts = []
        with pdfplumber.open(inp.filepath) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t:
                    parts.append(t)
        text = "\n\n".join(parts)

    elif ext in ("doc", "docx"):
        from docx import Document
        doc = Document(inp.filepath)
        text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

    elif ext in ("xlsx", "xls"):
        import openpyxl
        wb = openpyxl.load_workbook(inp.filepath, read_only=True, data_only=True)
        rows = []
        for ws in wb.worksheets:
            for row in ws.iter_rows(values_only=True):
                rows.append(" | ".join(str(c) if c is not None else "" for c in row))
        wb.close()
        text = "\n".join(rows)

    else:
        with open(inp.filepath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

    if not text.strip():
        raise ValueError(f"No text content found in {inp.filename}")

    activity.logger.info(f"Extracted {len(text)} characters from {inp.filename}")
    return ExtractResult(text=text, characters=len(text))


# ── Activity 2: Chunk text ──
@activity.defn
async def chunk_text_activity(text: str) -> ChunkResult:
    words = text.split()
    if not words:
        return ChunkResult(chunks=[], count=0)

    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i: i + CHUNK_SIZE]))
        i += CHUNK_SIZE - CHUNK_OVERLAP

    activity.logger.info(f"Created {len(chunks)} chunks")
    return ChunkResult(chunks=chunks, count=len(chunks))


# ── Activity 3: Embed chunks and store in ChromaDB ──
@activity.defn
async def embed_and_store_activity(doc_id: str, filename: str, chunks: list) -> EmbedStoreResult:
    activity.logger.info(f"Embedding {len(chunks)} chunks for {filename}")

    # Resolve embedding model
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            tags_resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            available = [m["name"].split(":")[0] for m in tags_resp.json().get("models", [])]
        except Exception:
            available = []

    preferred = ["nomic-embed-text", "all-minilm", "mxbai-embed-large"]
    embed_model = None
    for m in preferred:
        if m in available:
            embed_model = m
            break
    if embed_model is None and available:
        embed_model = available[0]
    if embed_model is None:
        raise RuntimeError("No models available in Ollama. Run: ollama pull nomic-embed-text")

    activity.logger.info(f"Using embedding model: {embed_model}")

    # Embed each chunk and collect
    ids, embeddings, metadatas, documents = [], [], [], []
    now = datetime.now(timezone.utc).isoformat()

    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, chunk in enumerate(chunks):
            # Heartbeat so Temporal knows we're alive for large docs
            activity.heartbeat(f"Embedding chunk {i+1}/{len(chunks)}")

            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json={"model": embed_model, "input": chunk},
            )
            resp.raise_for_status()
            data = resp.json()
            embs = data.get("embeddings", [])
            if not embs or len(embs) == 0:
                raise ValueError(f"No embedding returned for chunk {i}")

            ids.append(f"{doc_id}_chunk_{i}")
            embeddings.append(embs[0])
            metadatas.append({
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "uploaded_at": now,
            })
            documents.append(chunk)

    # Store in ChromaDB
    activity.heartbeat("Storing in ChromaDB")
    coll = _get_collection()
    coll.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    activity.logger.info(f"Stored {len(ids)} chunks in ChromaDB for doc {doc_id}")
    return EmbedStoreResult(chunks_stored=len(ids))
