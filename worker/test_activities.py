"""Tests for activities — runnable without a Temporal server.

Run with: pytest test_activities.py -v

Demonstrates:
- Testing activities as plain async functions (no Temporal infrastructure)
- Mocking S3/Chroma/Ollama via the DI seams in the activity classes
- Testing the Ollama embed fallback logic specifically
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from activities import (
    DocumentActivities,
    QueryActivities,
    OllamaClient,
    S3Client,
    ChromaStore,
)
from shared.models import UploadInput, QueryInput


# ═══════════════════════════════════════════════════════════════════════════
# Mock infrastructure
# ═══════════════════════════════════════════════════════════════════════════
class MockS3Client:
    """In-memory S3 replacement."""

    def __init__(self, initial_data: dict = None):
        self._store = dict(initial_data or {})
        self.bucket = "test-bucket"

    def get_bytes(self, key: str) -> bytes:
        if key not in self._store:
            from botocore.exceptions import ClientError
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": f"Key not found: {key}"}},
                "GetObject",
            )
        return self._store[key]

    def put_bytes(self, key: str, data: bytes) -> None:
        self._store[key] = data


class MockChromaStore:
    """Minimal ChromaDB mock."""

    def __init__(self, count: int = 0):
        self.collection = MagicMock()
        self.collection.count.return_value = count


class MockOllamaClient:
    """Deterministic Ollama mock."""

    def __init__(self, embed_dim: int = 768):
        self._dim = embed_dim

    async def embed(self, text: str) -> list:
        return [0.01] * self._dim

    async def generate(self, model: str, prompt: str, system: str) -> str:
        return "Mock answer from context."


def _make_doc_activities(s3=None, chroma=None, ollama=None):
    return DocumentActivities(
        s3=s3 or MockS3Client(),
        chroma=chroma or MockChromaStore(),
        ollama=ollama or MockOllamaClient(),
    )


def _make_query_activities(s3=None, chroma=None, ollama=None):
    return QueryActivities(
        s3=s3 or MockS3Client(),
        chroma=chroma or MockChromaStore(),
        ollama=ollama or MockOllamaClient(),
        backend_url="http://fake:6976",
    )


# ═══════════════════════════════════════════════════════════════════════════
# extract_text tests
# ═══════════════════════════════════════════════════════════════════════════
class TestExtractText:

    @pytest.mark.asyncio
    async def test_plain_text_extraction(self):
        """A .txt file should be returned as-is."""
        s3 = MockS3Client({"raw/doc1.txt": b"Hello world, this is a test."})
        acts = _make_doc_activities(s3=s3)

        result = await acts.extract_text(
            UploadInput(doc_id="doc1", filename="readme.txt", s3_raw_key="raw/doc1.txt")
        )

        assert result.characters == 27
        assert "doc1" in result.s3_text_key
        assert s3._store[result.s3_text_key] == b"Hello world, this is a test."

    @pytest.mark.asyncio
    async def test_empty_file_raises_non_retryable(self):
        from temporalio.exceptions import ApplicationError

        s3 = MockS3Client({"raw/empty.txt": b"   "})
        acts = _make_doc_activities(s3=s3)

        with pytest.raises(ApplicationError, match="No text extracted"):
            await acts.extract_text(
                UploadInput(doc_id="x", filename="empty.txt", s3_raw_key="raw/empty.txt")
            )

    @pytest.mark.asyncio
    async def test_missing_s3_key_raises_non_retryable(self):
        from temporalio.exceptions import ApplicationError

        acts = _make_doc_activities(s3=MockS3Client())

        with pytest.raises(ApplicationError, match="Failed to fetch"):
            await acts.extract_text(
                UploadInput(doc_id="x", filename="gone.txt", s3_raw_key="raw/gone.txt")
            )


# ═══════════════════════════════════════════════════════════════════════════
# chunk_text tests
# ═══════════════════════════════════════════════════════════════════════════
class TestChunkText:

    @pytest.mark.asyncio
    async def test_creates_chunks(self):
        text = " ".join(f"word{i}" for i in range(1000))
        s3 = MockS3Client({"text/doc1.txt": text.encode()})
        acts = _make_doc_activities(s3=s3)

        result = await acts.chunk_text("text/doc1.txt")

        assert result.count > 1
        assert len(result.chunks) == result.count

    @pytest.mark.asyncio
    async def test_empty_returns_zero(self):
        s3 = MockS3Client({"text/empty.txt": b""})
        acts = _make_doc_activities(s3=s3)

        result = await acts.chunk_text("text/empty.txt")

        assert result.count == 0
        assert result.chunks == []


# ═══════════════════════════════════════════════════════════════════════════
# embed_and_store tests
# ═══════════════════════════════════════════════════════════════════════════
class TestEmbedAndStore:

    @pytest.mark.asyncio
    async def test_stores_chunks_in_chroma(self):
        chroma = MockChromaStore()
        acts = _make_doc_activities(chroma=chroma)

        result = await acts.embed_and_store("doc1", "test.txt", ["chunk1", "chunk2"])

        assert result.chunks_stored == 2
        chroma.collection.add.assert_called_once()
        call_kwargs = chroma.collection.add.call_args
        assert len(call_kwargs.kwargs["ids"]) == 2


# ═══════════════════════════════════════════════════════════════════════════
# OllamaClient.embed tests (the bug fix)
# ═══════════════════════════════════════════════════════════════════════════
class TestOllamaEmbed:

    @pytest.mark.asyncio
    async def test_modern_api_success(self):
        """Verify /api/embed with string input works."""
        client = OllamaClient(base_url="http://fake:6970")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.embed("test text")
            assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_fallback_to_legacy_on_400(self):
        """If /api/embed returns 400, should fall back to /api/embeddings."""
        client = OllamaClient(base_url="http://fake:6970")
        call_count = 0

        async def mock_post(url, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            if "/api/embed" in url and call_count == 1:
                resp.status_code = 400
                resp.text = "bad request"
            else:
                resp.status_code = 200
                resp.json.return_value = {"embedding": [0.4, 0.5, 0.6]}
            return resp

        with patch("httpx.AsyncClient.post", side_effect=mock_post):
            result = await client.embed("test text")
            assert result == [0.4, 0.5, 0.6]
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_404_is_non_retryable(self):
        """404 = model not found, should fail immediately."""
        from temporalio.exceptions import ApplicationError

        client = OllamaClient(base_url="http://fake:6970")
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            with pytest.raises(ApplicationError, match="not found"):
                await client.embed("test")

    @pytest.mark.asyncio
    async def test_empty_string_is_non_retryable(self):
        from temporalio.exceptions import ApplicationError

        client = OllamaClient()
        with pytest.raises(ApplicationError, match="empty string"):
            await client.embed("   ")


# ═══════════════════════════════════════════════════════════════════════════
# execute_query tests
# ═══════════════════════════════════════════════════════════════════════════
class TestExecuteQuery:

    @pytest.mark.asyncio
    async def test_no_documents_returns_idk(self):
        """When ChromaDB is empty, should return 'I do not know'."""
        query_data = json.dumps({"query": "What is the meaning of life?"}).encode()
        s3 = MockS3Client({"queries/q1.json": query_data})
        chroma = MockChromaStore(count=0)

        acts = _make_query_activities(s3=s3, chroma=chroma)

        result = await acts.execute_query(
            QueryInput(query_id="q1", s3_query_key="queries/q1.json", model="llama3.2")
        )

        assert result.status == "completed"
        stored_answer = json.loads(s3._store[result.s3_answer_key])
        assert "do not know" in stored_answer["answer"].lower()