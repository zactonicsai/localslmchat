"""Temporal activities for document processing via S3."""
import io
import os
import tempfile
from datetime import datetime, timezone
from typing import List

import boto3
import httpx
import chromadb
from temporalio import activity

from shared.models import (
    UploadInput, ExtractResult, ChunkResult, EmbedStoreResult,
    S3_TEXT_PREFIX,
)

S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://localhost:4566")
S3_BUCKET = os.getenv("S3_BUCKET", "lcq-documents")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def _s3():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )


def _chroma():
    c = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    c.heartbeat()
    return c.get_or_create_collection(name="local_context", metadata={"hnsw:space": "cosine"})


@activity.defn
async def extract_text_activity(inp: UploadInput) -> ExtractResult:
    """Download raw file from S3, extract text, save extracted text back to S3."""
    activity.logger.info(f"Extracting text from s3://{S3_BUCKET}/{inp.s3_raw_key}")

    s3 = _s3()
    obj = s3.get_object(Bucket=S3_BUCKET, Key=inp.s3_raw_key)
    raw_bytes = obj["Body"].read()

    ext = inp.filename.rsplit(".", 1)[-1].lower() if "." in inp.filename else ""

    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        if ext == "pdf":
            import pdfplumber
            parts = []
            with pdfplumber.open(tmp_path) as pdf:
                for p in pdf.pages:
                    t = p.extract_text()
                    if t:
                        parts.append(t)
            text = "\n\n".join(parts)

        elif ext in ("doc", "docx"):
            from docx import Document
            doc = Document(tmp_path)
            text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

        elif ext in ("xlsx", "xls"):
            import openpyxl
            wb = openpyxl.load_workbook(tmp_path, read_only=True, data_only=True)
            rows = []
            for ws in wb.worksheets:
                for row in ws.iter_rows(values_only=True):
                    rows.append(" | ".join(str(c) if c is not None else "" for c in row))
            wb.close()
            text = "\n".join(rows)

        else:
            text = raw_bytes.decode("utf-8", errors="replace")
    finally:
        os.unlink(tmp_path)

    if not text.strip():
        raise ValueError(f"No text content found in {inp.filename}")

    s3_text_key = f"{S3_TEXT_PREFIX}{inp.doc_id}.txt"
    s3.put_object(Bucket=S3_BUCKET, Key=s3_text_key, Body=text.encode("utf-8"))
    activity.logger.info(f"Saved extracted text ({len(text)} chars) to s3://{S3_BUCKET}/{s3_text_key}")

    return ExtractResult(s3_text_key=s3_text_key, characters=len(text))


@activity.defn
async def chunk_text_activity(s3_text_key: str) -> ChunkResult:
    """Download extracted text from S3 and split into overlapping chunks."""
    activity.logger.info(f"Chunking from s3://{S3_BUCKET}/{s3_text_key}")

    s3 = _s3()
    obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_text_key)
    text = obj["Body"].read().decode("utf-8")

    words = text.split()
    if not words:
        return ChunkResult(chunks=[], count=0)

    chunks: List[str] = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i: i + CHUNK_SIZE]))
        i += CHUNK_SIZE - CHUNK_OVERLAP

    activity.logger.info(f"Created {len(chunks)} chunks")
    return ChunkResult(chunks=chunks, count=len(chunks))


@activity.defn
async def embed_and_store_activity(doc_id: str, filename: str, chunks: List[str]) -> EmbedStoreResult:
    """Embed each chunk via Ollama and store in ChromaDB."""
    activity.logger.info(f"Embedding {len(chunks)} chunks for {filename} with {EMBED_MODEL}")

    ids, embeddings, metadatas, documents = [], [], [], []
    now = datetime.now(timezone.utc).isoformat()

    async with httpx.AsyncClient(timeout=120.0) as client:
        for i, chunk in enumerate(chunks):
            activity.heartbeat(f"Embedding chunk {i+1}/{len(chunks)}")

            # Try new API (/api/embed), fall back to old (/api/embeddings)
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json={"model": EMBED_MODEL, "input": chunk},
            )
            if resp.status_code == 404:
                resp = await client.post(
                    f"{OLLAMA_BASE_URL}/api/embed",
                    json={"model": EMBED_MODEL, "prompt": chunk},
                )
            resp.raise_for_status()
            data = resp.json()

            # New API: {"embeddings": [[...]]}  Old API: {"embedding": [...]}
            emb = None
            embs = data.get("embeddings")
            if embs and len(embs) > 0:
                emb = embs[0]
            else:
                emb = data.get("embedding")
            if not emb:
                raise ValueError(f"No embedding returned for chunk {i}")

            ids.append(f"{doc_id}_chunk_{i}")
            embeddings.append(emb)
            metadatas.append({
                "doc_id": doc_id, "filename": filename,
                "chunk_index": i, "total_chunks": len(chunks),
                "uploaded_at": now,
            })
            documents.append(chunk)

    activity.heartbeat("Storing in ChromaDB")
    coll = _chroma()
    coll.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    activity.logger.info(f"Stored {len(ids)} chunks for {doc_id} in ChromaDB")
    return EmbedStoreResult(chunks_stored=len(ids))
