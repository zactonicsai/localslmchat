"""Temporal workflow for document upload processing."""
from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from shared.models import (
        UploadInput, UploadResult, ExtractResult, ChunkResult, EmbedStoreResult,
    )
    from activities import (
        extract_text_activity, chunk_text_activity, embed_and_store_activity,
    )


@workflow.defn
class DocumentUploadWorkflow:
    @workflow.run
    async def run(self, inp: UploadInput) -> UploadResult:
        workflow.logger.info(f"Starting workflow for {inp.filename} (doc_id={inp.doc_id})")

        try:
            # Step 1: Extract text from S3 file, save text back to S3
            extract_result: ExtractResult = await workflow.execute_activity(
                extract_text_activity,
                inp,
                result_type=ExtractResult,
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=RetryPolicy(maximum_attempts=3, initial_interval=timedelta(seconds=2)),
            )

            workflow.logger.info(
                f"Extracted {extract_result.characters} chars, text at {extract_result.s3_text_key}"
            )

            # Step 2: Chunk the extracted text from S3
            chunk_result: ChunkResult = await workflow.execute_activity(
                chunk_text_activity,
                extract_result.s3_text_key,
                result_type=ChunkResult,
                start_to_close_timeout=timedelta(minutes=2),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

            workflow.logger.info(f"Chunked into {chunk_result.count} chunks")

            if chunk_result.count == 0:
                return UploadResult(
                    doc_id=inp.doc_id, filename=inp.filename,
                    chunks=0, characters=extract_result.characters,
                    status="failed", error="No content after chunking",
                )

            # Step 3: Embed and store in ChromaDB
            store_result: EmbedStoreResult = await workflow.execute_activity(
                embed_and_store_activity,
                args=[inp.doc_id, inp.filename, chunk_result.chunks],
                result_type=EmbedStoreResult,
                start_to_close_timeout=timedelta(minutes=30),
                heartbeat_timeout=timedelta(minutes=2),
                retry_policy=RetryPolicy(maximum_attempts=3, initial_interval=timedelta(seconds=5)),
            )

            workflow.logger.info(f"Stored {store_result.chunks_stored} chunks in ChromaDB")

            return UploadResult(
                doc_id=inp.doc_id, filename=inp.filename,
                chunks=store_result.chunks_stored, characters=extract_result.characters,
                status="completed",
            )
        except Exception as e:
            workflow.logger.error(f"Workflow FAILED for {inp.filename}: {e}")
            return UploadResult(
                doc_id=inp.doc_id, filename=inp.filename,
                chunks=0, characters=0, status="failed", error=str(e),
            )
