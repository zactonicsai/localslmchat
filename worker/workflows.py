"""Temporal workflow for document upload processing."""
from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from shared.models import UploadInput, UploadResult
    from activities import (
        extract_text_activity,
        chunk_text_activity,
        embed_and_store_activity,
    )


@workflow.defn
class DocumentUploadWorkflow:
    """
    Workflow: Save file → Extract text → Chunk → Embed → Store in ChromaDB.

    Each step is a separate activity so Temporal handles retries/timeouts.
    """

    @workflow.run
    async def run(self, inp: UploadInput) -> UploadResult:
        workflow.logger.info(f"Starting upload workflow for {inp.filename} (doc_id={inp.doc_id})")

        try:
            # Step 1: Extract text
            extract_result = await workflow.execute_activity(
                extract_text_activity,
                inp,
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=workflow.RetryPolicy(
                    maximum_attempts=3,
                    initial_interval=timedelta(seconds=2),
                ),
            )

            # Step 2: Chunk the text
            chunk_result = await workflow.execute_activity(
                chunk_text_activity,
                extract_result.text,
                start_to_close_timeout=timedelta(minutes=2),
                retry_policy=workflow.RetryPolicy(maximum_attempts=3),
            )

            if chunk_result.count == 0:
                return UploadResult(
                    doc_id=inp.doc_id,
                    filename=inp.filename,
                    chunks=0,
                    characters=extract_result.characters,
                    status="failed",
                    error="No content after chunking",
                )

            # Step 3: Embed and store in ChromaDB
            store_result = await workflow.execute_activity(
                embed_and_store_activity,
                args=[inp.doc_id, inp.filename, chunk_result.chunks],
                start_to_close_timeout=timedelta(minutes=30),
                heartbeat_timeout=timedelta(minutes=2),
                retry_policy=workflow.RetryPolicy(
                    maximum_attempts=3,
                    initial_interval=timedelta(seconds=5),
                ),
            )

            workflow.logger.info(
                f"Upload complete: {inp.filename} → {store_result.chunks_stored} chunks"
            )

            return UploadResult(
                doc_id=inp.doc_id,
                filename=inp.filename,
                chunks=store_result.chunks_stored,
                characters=extract_result.characters,
                status="completed",
            )

        except Exception as e:
            workflow.logger.error(f"Upload workflow failed for {inp.filename}: {e}")
            return UploadResult(
                doc_id=inp.doc_id,
                filename=inp.filename,
                chunks=0,
                characters=0,
                status="failed",
                error=str(e),
            )
