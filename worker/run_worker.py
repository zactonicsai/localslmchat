"""Temporal worker with upload + query task queues."""
import asyncio, os
from temporalio.client import Client
from temporalio.worker import Worker
from workflows import DocumentUploadWorkflow, QueryWorkflow
from activities import (extract_text_activity, chunk_text_activity,
                       embed_and_store_activity, execute_query_activity)
from shared.models import TASK_QUEUE, QUERY_TASK_QUEUE

async def main():
    client = await Client.connect(os.getenv("TEMPORAL_ADDRESS", "localhost:7233"))
    print(f"[WORKER] Connected. Upload queue='{TASK_QUEUE}', Query queue='{QUERY_TASK_QUEUE}' (max 1)")
    await asyncio.gather(
        Worker(client, task_queue=TASK_QUEUE, workflows=[DocumentUploadWorkflow],
               activities=[extract_text_activity, chunk_text_activity, embed_and_store_activity]).run(),
        Worker(client, task_queue=QUERY_TASK_QUEUE, workflows=[QueryWorkflow],
               activities=[execute_query_activity], max_concurrent_activities=1).run(),
    )

if __name__ == "__main__":
    asyncio.run(main())
