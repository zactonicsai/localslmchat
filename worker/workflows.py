"""Temporal workflows."""
from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from shared.models import (UploadInput, UploadResult, ExtractResult, ChunkResult,
                               EmbedStoreResult, QueryInput, QueryResult)
    from activities import (extract_text_activity, chunk_text_activity,
                           embed_and_store_activity, execute_query_activity)

@workflow.defn
class DocumentUploadWorkflow:
    @workflow.run
    async def run(self, inp: UploadInput) -> UploadResult:
        try:
            ext = await workflow.execute_activity(extract_text_activity, inp, result_type=ExtractResult,
                start_to_close_timeout=timedelta(minutes=5), retry_policy=RetryPolicy(maximum_attempts=3, initial_interval=timedelta(seconds=2)))
            ch = await workflow.execute_activity(chunk_text_activity, ext.s3_text_key, result_type=ChunkResult,
                start_to_close_timeout=timedelta(minutes=2), retry_policy=RetryPolicy(maximum_attempts=3))
            if ch.count == 0:
                return UploadResult(doc_id=inp.doc_id, filename=inp.filename, chunks=0, characters=ext.characters, status="failed", error="No content")
            st = await workflow.execute_activity(embed_and_store_activity, args=[inp.doc_id, inp.filename, ch.chunks],
                result_type=EmbedStoreResult, start_to_close_timeout=timedelta(minutes=30), heartbeat_timeout=timedelta(minutes=2),
                retry_policy=RetryPolicy(maximum_attempts=3, initial_interval=timedelta(seconds=5)))
            return UploadResult(doc_id=inp.doc_id, filename=inp.filename, chunks=st.chunks_stored, characters=ext.characters, status="completed")
        except Exception as e:
            return UploadResult(doc_id=inp.doc_id, filename=inp.filename, chunks=0, characters=0, status="failed", error=str(e))

@workflow.defn
class QueryWorkflow:
    @workflow.run
    async def run(self, inp: QueryInput) -> QueryResult:
        try:
            return await workflow.execute_activity(execute_query_activity, inp, result_type=QueryResult,
                start_to_close_timeout=timedelta(minutes=10), heartbeat_timeout=timedelta(minutes=3),
                retry_policy=RetryPolicy(maximum_attempts=2, initial_interval=timedelta(seconds=3)))
        except Exception as e:
            return QueryResult(query_id=inp.query_id, status="failed", error=str(e))
