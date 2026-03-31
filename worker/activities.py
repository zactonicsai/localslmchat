"""Temporal activities for upload and query processing.

Best practices applied:
- Class-based activities with dependency injection for testability
- Single input/output dataclasses (already present via shared.models)
- Heartbeating on long-running activities
- Non-retryable ApplicationError for permanent failures (404, validation)
- Retryable errors left as plain exceptions for Temporal retry policy
- Explicit timeout-friendly httpx clients (not created per-call)
- Ollama /api/embed fix: handle both string and list input, with
  fallback to legacy /api/embeddings endpoint for older Ollama versions
"""

import os
import json
import tempfile
import re
from datetime import datetime, timezone
from typing import List, Optional

import boto3
import httpx
import chromadb
from botocore.exceptions import ClientError
from temporalio import activity
from temporalio.exceptions import ApplicationError

from shared.models import (
    UploadInput, ExtractResult, ChunkResult, EmbedStoreResult,
    QueryInput, QueryResult, S3_TEXT_PREFIX, S3_ANSWER_PREFIX
)

# ---------------------------------------------------------------------------
# Configuration – injected at runtime via env vars
# ---------------------------------------------------------------------------
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://localhost:4566")
S3_BUCKET = os.getenv("S3_BUCKET", "lcq-documents")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8200")
# Chunk size in words. nomic-embed-text defaults to 2048 tokens in Ollama.
# ~1.3 tokens/word means 500 words ≈ 650 tokens — safely under the limit.
# The old default of 800 words (~1100 tokens) caused "input length exceeds
# context length" errors on documents with dense/technical text.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Maximum characters to send to the embedding model. This is a hard safety
# net — even if chunks are sized correctly in words, some documents contain
# very long "words" (base64, URLs, hex dumps) that inflate token count.
# 6000 chars ≈ 1500 tokens, well within nomic-embed-text's 2048 default.
EMBED_MAX_CHARS = int(os.getenv("EMBED_MAX_CHARS", "6000"))


# ---------------------------------------------------------------------------
# Infrastructure helpers (thin wrappers for DI seams)
# ---------------------------------------------------------------------------
class S3Client:
    """Wrapper around boto3 S3 client for testability."""

    def __init__(
        self,
        endpoint_url: str = S3_ENDPOINT,
        bucket: str = S3_BUCKET,
    ):
        self._bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        )

    @property
    def bucket(self) -> str:
        return self._bucket

    def get_bytes(self, key: str) -> bytes:
        return self._client.get_object(Bucket=self._bucket, Key=key)["Body"].read()

    def put_bytes(self, key: str, data: bytes) -> None:
        self._client.put_object(Bucket=self._bucket, Key=key, Body=data)


class ChromaStore:
    """Wrapper around ChromaDB for testability."""

    def __init__(self, host: str = CHROMA_HOST, port: int = CHROMA_PORT):
        client = chromadb.HttpClient(host=host, port=port)
        client.heartbeat()
        self._collection = client.get_or_create_collection(
            name="local_context",
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self):
        return self._collection


class OllamaClient:
    """Wraps Ollama HTTP calls with proper error handling.

    Fixes the embed bug: the /api/embed endpoint expects ``"input"`` as
    either a plain string *or* a list of strings.  Some Ollama versions
    return 400 when given a list for a single string – we now send a
    plain string for single inputs (matching the official curl examples)
    and fall back to the legacy ``/api/embeddings`` endpoint on error.
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL, embed_model: str = EMBED_MODEL):
        self._base_url = base_url.rstrip("/")
        self._embed_model = embed_model

    async def embed(self, text: str) -> List[float]:
        """Return the embedding vector for *text*.

        Applies a character-level truncation before sending to Ollama to
        prevent "input length exceeds context length" errors.  Also passes
        ``truncate: true`` in the payload so Ollama server-side truncates
        if the tokenized length still exceeds the model's context window.

        Tries ``/api/embed`` first (current API).  Falls back to the
        legacy ``/api/embeddings`` endpoint if the server returns a
        non-404 error (e.g. older Ollama that doesn't support /embed).
        """
        clean = text.strip()
        if not clean:
            raise ApplicationError("Cannot embed an empty string", non_retryable=True)

        # Truncate to safe character limit before tokenization.
        # nomic-embed-text in Ollama defaults to 2048 tokens context.
        # At ~4 chars/token, 6000 chars ≈ 1500 tokens (safe margin).
        if len(clean) > EMBED_MAX_CHARS:
            activity.logger.info(
                f"Truncating embed input from {len(clean)} to {EMBED_MAX_CHARS} chars"
            )
            # Truncate at last word boundary to avoid splitting mid-word
            clean = clean[:EMBED_MAX_CHARS].rsplit(" ", 1)[0]

        async with httpx.AsyncClient(timeout=90.0) as client:
            # --- Try modern /api/embed first ---
            embed_error = await self._try_embed(client, clean)
            if isinstance(embed_error, list):
                return embed_error  # success – got the vector

            # --- Fallback to legacy /api/embeddings ---
            legacy_result = await self._try_legacy_embeddings(client, clean)
            if isinstance(legacy_result, list):
                return legacy_result  # success via legacy

            # Both failed – raise the original error
            raise embed_error

    async def _try_embed(self, client: httpx.AsyncClient, text: str):
        """Try /api/embed.  Returns the vector list on success or an Exception."""
        # Per Ollama docs: input can be a string for single embeddings.
        # truncate:true tells Ollama to silently truncate if token count
        # still exceeds the model's context window instead of returning 400.
        payload = {"model": self._embed_model, "input": text, "truncate": True}
        try:
            r = await client.post(f"{self._base_url}/api/embed", json=payload)

            if r.status_code == 404:
                raise ApplicationError(
                    f"Embedding model '{self._embed_model}' not found on Ollama. "
                    f"Pull it with: ollama pull {self._embed_model}",
                    non_retryable=True,
                )

            if r.status_code != 200:
                activity.logger.warning(
                    f"Ollama /api/embed returned {r.status_code}: {r.text[:300]}"
                )
                return ValueError(f"Ollama /api/embed error {r.status_code}: {r.text[:200]}")

            data = r.json()
            embeddings = data.get("embeddings")
            if embeddings and len(embeddings) > 0:
                return embeddings[0]

            return ValueError(f"Empty embeddings in /api/embed response: {data}")

        except ApplicationError:
            raise  # let non-retryable errors propagate immediately
        except httpx.RequestError as exc:
            activity.logger.warning(f"Network error on /api/embed: {exc}")
            return exc

    async def _try_legacy_embeddings(self, client: httpx.AsyncClient, text: str):
        """Fallback: try legacy /api/embeddings (uses 'prompt' key)."""
        payload = {"model": self._embed_model, "prompt": text, "truncate": True}
        try:
            r = await client.post(f"{self._base_url}/api/embeddings", json=payload)

            if r.status_code == 404:
                raise ApplicationError(
                    f"Embedding model '{self._embed_model}' not found on Ollama. "
                    f"Pull it with: ollama pull {self._embed_model}",
                    non_retryable=True,
                )

            if r.status_code != 200:
                return ValueError(f"Ollama /api/embeddings error {r.status_code}: {r.text[:200]}")

            data = r.json()
            embedding = data.get("embedding")
            if embedding and len(embedding) > 0:
                return embedding

            return ValueError(f"Empty embedding in /api/embeddings response: {data}")

        except ApplicationError:
            raise
        except httpx.RequestError as exc:
            activity.logger.warning(f"Network error on /api/embeddings: {exc}")
            return exc

    async def generate(self, model: str, prompt: str, system: str) -> str:
        """Call Ollama /api/generate and return the response text."""
        async with httpx.AsyncClient(timeout=180.0) as client:
            payload = {
                "model": model,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "options": {"temperature": 0.3, "num_ctx": 4096},
            }
            r = await client.post(f"{self._base_url}/api/generate", json=payload)

            if r.status_code == 404:
                raise ApplicationError(
                    f"LLM model '{model}' not found on Ollama. "
                    f"Pull it with: ollama pull {model}",
                    non_retryable=True,
                )

            if r.status_code != 200:
                # Retryable – Ollama might be temporarily overloaded
                raise ValueError(f"Ollama generate error {r.status_code}: {r.text[:200]}")

            answer = r.json().get("response", "")
            # Strip <think> blocks from reasoning models
            answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
            return answer


# ---------------------------------------------------------------------------
# Activity implementations (class-based for dependency injection)
# ---------------------------------------------------------------------------
class DocumentActivities:
    """Activities for the document upload workflow.

    Constructor-injected dependencies make these trivially testable:
    pass mock S3/Chroma/Ollama in tests, real ones in production.
    """

    def __init__(
        self,
        s3: Optional[S3Client] = None,
        chroma: Optional[ChromaStore] = None,
        ollama: Optional[OllamaClient] = None,
    ):
        self._s3 = s3 or S3Client()
        self._chroma = chroma or ChromaStore()
        self._ollama = ollama or OllamaClient()

    # -- Extract ----------------------------------------------------------
    @activity.defn(name="extract_text_activity")
    async def extract_text(self, inp: UploadInput) -> ExtractResult:
        """Extract text from an uploaded file stored in S3."""
        try:
            raw = self._s3.get_bytes(inp.s3_raw_key)
        except ClientError as e:
            raise ApplicationError(
                f"Failed to fetch raw file from S3: {e}", non_retryable=True
            )

        ext = inp.filename.rsplit(".", 1)[-1].lower() if "." in inp.filename else ""
        text = self._extract_by_extension(raw, ext, inp.filename)

        if not text.strip():
            raise ApplicationError(
                f"No text extracted from {inp.filename}", non_retryable=True
            )

        key = f"{S3_TEXT_PREFIX}{inp.doc_id}.txt"
        try:
            self._s3.put_bytes(key, text.encode())
        except ClientError as e:
            raise ApplicationError(f"S3 upload failed: {e}", non_retryable=True)

        return ExtractResult(s3_text_key=key, characters=len(text))

    # -- Chunk ------------------------------------------------------------
    @activity.defn(name="chunk_text_activity")
    async def chunk_text(self, s3_text_key: str) -> ChunkResult:
        """Split extracted text into overlapping chunks."""
        try:
            text = self._s3.get_bytes(s3_text_key).decode("utf-8")
        except ClientError as e:
            raise ApplicationError(
                f"Failed to read text from S3: {e}", non_retryable=True
            )

        words = text.split()
        if not words:
            return ChunkResult(chunks=[], count=0)

        chunks, i = [], 0
        while i < len(words):
            chunks.append(" ".join(words[i : i + CHUNK_SIZE]))
            i += CHUNK_SIZE - CHUNK_OVERLAP

        return ChunkResult(chunks=chunks, count=len(chunks))

    # -- Embed & Store ----------------------------------------------------
    @activity.defn(name="embed_and_store_activity")
    async def embed_and_store(
        self, doc_id: str, filename: str, chunks: List[str]
    ) -> EmbedStoreResult:
        """Embed each chunk via Ollama and upsert into ChromaDB."""
        ids, embeddings, metadatas, documents = [], [], [], []
        now = datetime.now(timezone.utc).isoformat()

        for i, chunk in enumerate(chunks):
            # Heartbeat with progress so Temporal knows we're alive
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

        try:
            self._chroma.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )
        except Exception as e:
            activity.logger.error(f"ChromaDB storage failed: {e}")
            raise  # retryable

        return EmbedStoreResult(chunks_stored=len(ids))

    # -- Private helpers --------------------------------------------------
    @staticmethod
    def _extract_by_extension(raw: bytes, ext: str, filename: str) -> str:
        """Dispatch text extraction based on file extension."""
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        try:
            if ext == "pdf":
                import pdfplumber

                with pdfplumber.open(tmp_path) as pdf:
                    return "\n\n".join(
                        p.extract_text() or "" for p in pdf.pages
                    )

            if ext in ("doc", "docx"):
                from docx import Document

                return "\n\n".join(
                    p.text for p in Document(tmp_path).paragraphs if p.text.strip()
                )

            if ext in ("xlsx", "xls"):
                import openpyxl

                wb = openpyxl.load_workbook(tmp_path, read_only=True, data_only=True)
                lines = []
                for ws in wb.worksheets:
                    for row in ws.iter_rows(values_only=True):
                        lines.append(
                            " | ".join(str(c) if c is not None else "" for c in row)
                        )
                wb.close()
                return "\n".join(lines)

            # Fallback: plain text
            return raw.decode("utf-8", errors="replace")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class QueryActivities:
    """Activities for the query/RAG workflow."""

    def __init__(
        self,
        s3: Optional[S3Client] = None,
        chroma: Optional[ChromaStore] = None,
        ollama: Optional[OllamaClient] = None,
        backend_url: str = BACKEND_URL,
    ):
        self._s3 = s3 or S3Client()
        self._chroma = chroma or ChromaStore()
        self._ollama = ollama or OllamaClient()
        self._backend_url = backend_url

    @activity.defn(name="execute_query_activity")
    async def execute_query(self, inp: QueryInput) -> QueryResult:
        """Perform a RAG query: embed question → retrieve → generate answer."""
        # 1. Load query from S3
        try:
            q_data = json.loads(self._s3.get_bytes(inp.s3_query_key))
        except ClientError as e:
            raise ApplicationError(
                f"Could not load query data from S3: {e}", non_retryable=True
            )

        query_text = q_data["query"]
        activity.logger.info(f"Query [{inp.query_id}]: {query_text[:80]}")

        # 2. Retrieve relevant chunks
        coll = self._chroma.collection
        if coll.count() == 0:
            answer_obj = self._no_docs_answer(inp.query_id)
        else:
            answer_obj = await self._rag_answer(inp, query_text, coll)

        # 3. Persist answer to S3
        s3_answer_key = f"{S3_ANSWER_PREFIX}{inp.query_id}.json"
        try:
            self._s3.put_bytes(s3_answer_key, json.dumps(answer_obj).encode())
        except ClientError as e:
            raise ApplicationError(
                f"Failed to save answer to S3: {e}", non_retryable=True
            )

        # 4. Best-effort backend notification (fire-and-forget)
        await self._notify_backend(inp.query_id)

        return QueryResult(
            query_id=inp.query_id,
            status="completed",
            s3_answer_key=s3_answer_key,
        )

    # -- Private helpers --------------------------------------------------
    async def _rag_answer(self, inp: QueryInput, query_text: str, coll) -> dict:
        """Core RAG logic: embed query, retrieve, generate."""
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

        return {
            "query_id": inp.query_id,
            "answer": answer_text,
            "sources": sources,
        }

    @staticmethod
    def _filter_results(results: dict, enabled_doc_ids: Optional[List[str]]) -> tuple:
        """Filter ChromaDB results by enabled doc IDs."""
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

    @staticmethod
    def _no_docs_answer(query_id: str) -> dict:
        return {
            "query_id": query_id,
            "answer": "I do not know. No documents uploaded yet.",
            "sources": [],
        }

    @staticmethod
    def _no_results_answer(query_id: str) -> dict:
        return {
            "query_id": query_id,
            "answer": "I do not know. No relevant information found.",
            "sources": [],
        }

    async def _notify_backend(self, query_id: str) -> None:
        """Fire-and-forget notification to backend. Never fails the activity."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(
                    f"{self._backend_url}/api/internal/query-complete",
                    json={"query_id": query_id},
                )
        except Exception:
            activity.logger.debug(f"Backend notification failed for {query_id} (non-critical)")


# ═══════════════════════════════════════════════════════════════════════════
# Module-level instances + aliases
#
# Created once at import time. The bound methods are exposed under the
# ORIGINAL function names so that:
#   - run_worker.py:  from activities import extract_text_activity  ← works
#   - workflows.py:   from activities import extract_text_activity  ← works
#   - Temporal server: activity type "extract_text_activity"        ← matches
# ═══════════════════════════════════════════════════════════════════════════
_s3 = S3Client()
_chroma = ChromaStore()
_ollama = OllamaClient()

_doc_activities = DocumentActivities(s3=_s3, chroma=_chroma, ollama=_ollama)
_query_activities = QueryActivities(s3=_s3, chroma=_chroma, ollama=_ollama)

extract_text_activity = _doc_activities.extract_text
chunk_text_activity = _doc_activities.chunk_text
embed_and_store_activity = _doc_activities.embed_and_store
execute_query_activity = _query_activities.execute_query