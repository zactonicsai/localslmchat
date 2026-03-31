"""Temporal workflows for document upload and RAG query.

Best practices:
- Workflows contain ONLY orchestration logic (no I/O, no side effects)
- All I/O delegated to activities
- Start-to-close timeout set on every activity (Temporal strongly recommends this)
- Heartbeat timeout on long-running activities (embed_and_store, execute_query)
- Non-retryable errors (ApplicationError) stop retries immediately
- Workflow catches exceptions and returns a typed error result instead of failing
  the entire execution — callers get a clean status/error response
"""

from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from shared.models import (
        UploadInput, UploadResult, ExtractResult, ChunkResult,
        EmbedStoreResult, QueryInput, QueryResult,
    )
    from activities import (
        extract_text_activity,
        chunk_text_activity,
        embed_and_store_activity,
        execute_query_activity,
    )


@workflow.defn
class DocumentUploadWorkflow:
    """Orchestrates: extract text -> chunk -> embed & store in vector DB."""

    @workflow.run
    async def run(self, inp: UploadInput) -> UploadResult:
        try:
            # Step 1: Extract text from the raw file in S3
            ext = await workflow.execute_activity(
                extract_text_activity,
                inp,
                result_type=ExtractResult,
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=RetryPolicy(
                    maximum_attempts=3,
                    initial_interval=timedelta(seconds=2),
                    non_retryable_error_types=["ApplicationError"],
                ),
            )

            # Step 2: Split text into overlapping chunks
            ch = await workflow.execute_activity(
                chunk_text_activity,
                ext.s3_text_key,
                result_type=ChunkResult,
                start_to_close_timeout=timedelta(minutes=2),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

            if ch.count == 0:
                return UploadResult(
                    doc_id=inp.doc_id,
                    filename=inp.filename,
                    chunks=0,
                    characters=ext.characters,
                    status="failed",
                    error="No extractable content found in document",
                )

            # Step 3: Embed chunks and store in ChromaDB
            st = await workflow.execute_activity(
                embed_and_store_activity,
                args=[inp.doc_id, inp.filename, ch.chunks],
                result_type=EmbedStoreResult,
                start_to_close_timeout=timedelta(minutes=30),
                heartbeat_timeout=timedelta(minutes=2),
                retry_policy=RetryPolicy(
                    maximum_attempts=3,
                    initial_interval=timedelta(seconds=5),
                    non_retryable_error_types=["ApplicationError"],
                ),
            )

            return UploadResult(
                doc_id=inp.doc_id,
                filename=inp.filename,
                chunks=st.chunks_stored,
                characters=ext.characters,
                status="completed",
            )

        except Exception as e:
            workflow.logger.error(f"Upload workflow failed for {inp.doc_id}: {e}")
            return UploadResult(
                doc_id=inp.doc_id,
                filename=inp.filename,
                chunks=0,
                characters=0,
                status="failed",
                error=str(e),
            )


@workflow.defn
class QueryWorkflow:
    """Orchestrates a single RAG query: embed question -> retrieve -> generate."""

    @workflow.run
    async def run(self, inp: QueryInput) -> QueryResult:
        try:
            return await workflow.execute_activity(
                execute_query_activity,
                inp,
                result_type=QueryResult,
                start_to_close_timeout=timedelta(minutes=10),
                heartbeat_timeout=timedelta(minutes=3),
                retry_policy=RetryPolicy(
                    maximum_attempts=2,
                    initial_interval=timedelta(seconds=3),
                    non_retryable_error_types=["ApplicationError"],
                ),
            )
        except Exception as e:
            workflow.logger.error(f"Query workflow failed for {inp.query_id}: {e}")
            return QueryResult(
                query_id=inp.query_id,
                status="failed",
                error=str(e),
            )
